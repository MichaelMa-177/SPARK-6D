"""Experimental XFeat-backed CAD-anchor recovery for every-frame RefineNet.

Only the seed changes. The current RGB-D neural 6D residual still determines
every output, including failed/occluded frames. A continuous numerical working
state is separate from strictly admitted appearance memory. This combination is
a research candidate using existing XFeat/PnP components, not a novelty claim.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from learned_features import FeatureSet, XFeatExtractor, match_features
from robust_pose import RobustPoseConfig, RobustPoseController, _valid_pose


@dataclass(frozen=True)
class LearnedRecoveryConfig(RobustPoseConfig):
    continuation_policy: str = "continuous"
    enable_relocalization: bool = True
    feature_device: str = "cuda"
    compile_features: bool = True
    max_image_features: int = 2048
    max_memory_features: int = 256
    feature_similarity: float = 0.80
    feature_ratio: float = 0.90
    feature_spatial_quota: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.feature_device not in ("cpu", "cuda"):
            raise ValueError("feature_device must be cpu or cuda")
        for key in ("max_image_features", "max_memory_features"):
            if type(getattr(self, key)) is not int or getattr(self, key) < self.min_pnp_inliers:
                raise ValueError(f"{key} must be an integer >= min_pnp_inliers")
        if not 0 <= self.feature_similarity <= 1 or not 0 < self.feature_ratio <= 1:
            raise ValueError("invalid learned descriptor thresholds")
        if type(self.feature_spatial_quota) is not int or not 0 <= self.feature_spatial_quota <= 1024:
            raise ValueError("invalid feature spatial quota")


class LearnedRecoveryController(RobustPoseController):
    def __init__(self, config=None, extractor=None):
        config = config or LearnedRecoveryConfig()
        if not isinstance(config, LearnedRecoveryConfig):
            raise TypeError("expected LearnedRecoveryConfig")
        super().__init__(config)
        self.extractor = extractor or XFeatExtractor(
            device=config.feature_device, max_features=config.max_image_features,
            compile_model=config.compile_features, spatial_quota=config.feature_spatial_quota)
        self._feature_diagnostics = {}

    def reset(self, *args, **kwargs):
        self._feature_diagnostics = {}
        # Complete feature graph warm-up during registration even when the
        # first depth gate admits no memory. This never uses a future frame.
        rgb = args[1] if len(args) > 1 else kwargs["rgb"]
        self.extractor.extract(rgb)
        return super().reset(*args, **kwargs)

    def _remember(self, rgb, depth, K, pose):
        evidence = self._evidence(pose, depth, K)
        if (not self._supported(evidence)
                or evidence["occluded_fraction"] > self.config.max_memory_occluded_fraction):
            return
        uv, surface_z, _ = self._project_visible(pose, K, depth.shape)
        if len(uv) < 3:
            return
        mask = np.zeros(depth.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 255)
        features = self.extractor.extract(rgb, mask)
        if not len(features.keypoints):
            return
        pixels = features.keypoints.astype(np.float64)
        nearest = np.rint(pixels).astype(int)
        inside = ((nearest[:, 0] >= 0) & (nearest[:, 0] < depth.shape[1])
                  & (nearest[:, 1] >= 0) & (nearest[:, 1] < depth.shape[0]))
        ids = np.flatnonzero(inside)
        z = depth[nearest[ids, 1], nearest[ids, 0]]
        valid = np.isfinite(z) & (z > 1e-4)
        ids, z = ids[valid], z[valid]
        if not len(ids):
            return
        rays = np.c_[pixels[ids], np.ones(len(ids))] @ np.linalg.inv(K).T
        anchors = (rays * z[:, None] - pose[:3, 3]) @ pose[:3, :3]
        distance, _ = self._tree.query(anchors, k=1)
        projection_distance, surface_id = cKDTree(uv).query(pixels[ids], k=1)
        anchor_tolerance = min(self._depth_tolerance,
                               self._diameter * self.config.anchor_distance_diameter)
        valid = ((distance <= anchor_tolerance)
                 & (projection_distance <= self.config.max_anchor_projection_distance_px)
                 & (np.abs(z - surface_z[surface_id]) <= self._depth_tolerance))
        ids, anchors = ids[valid], anchors[valid]
        if len(ids) < self.config.min_pnp_inliers:
            return
        # Spatially balanced memory: keep strongest keypoint in each 8px cell,
        # then descending reliability, with bounded storage independent of age.
        order = np.argsort(-features.scores[ids], kind="stable")
        cells = np.floor(features.keypoints[ids] / 8).astype(np.int64)
        used, keep = set(), []
        for index in order:
            cell = tuple(cells[index])
            if cell not in used:
                used.add(cell)
                keep.append(index)
            if len(keep) >= self.config.max_memory_features:
                break
        if len(keep) < self.config.min_pnp_inliers:
            return
        self._keyframes.append((anchors[keep].astype(np.float32), features.subset(ids[keep])))
        if len(self._keyframes) > self.config.max_keyframes:
            self._keyframes.pop(1 if self.config.max_keyframes > 1 else 0)

    def _relocalize(self, rgb, K):
        self._last_pnp_inliers = 0
        self._last_pnp_inlier_fraction = 0.0
        self._feature_diagnostics = {"query_features": 0, "bank_matches": [],
                                     "pnp_solver_calls": 0, "candidate_count": 0}
        if not self._keyframes:
            return None
        current = self.extractor.extract(rgb)
        self._feature_diagnostics["query_features"] = len(current.keypoints)
        best_pose, best_count = None, 0
        for anchors, saved in self._keyframes:
            matches = match_features(saved, current, self.config.feature_similarity,
                                     self.config.feature_ratio)
            self._feature_diagnostics["bank_matches"].append(len(matches))
            if len(matches) < self.config.min_pnp_inliers:
                continue
            object_points = anchors[matches[:, 0]]
            image_points = current.keypoints[matches[:, 1]]
            # Collinear anchors cannot determine a full 6D candidate. Planar
            # targets remain allowed, subject to RANSAC and current depth gates.
            singular = np.linalg.svd(object_points - object_points.mean(0), compute_uv=False)
            image_singular = np.linalg.svd(image_points - image_points.mean(0), compute_uv=False)
            if (singular[1] < 1e-5 or image_singular[1] < 1e-3
                    or len(np.unique(image_points, axis=0)) < self.config.min_pnp_inliers):
                continue
            self._feature_diagnostics["pnp_solver_calls"] += 1
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points, image_points, K, None, iterationsCount=100,
                    reprojectionError=3.0, confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
            except cv2.error:
                continue
            count = 0 if inliers is None else len(inliers)
            if not success or count < self.config.min_pnp_inliers or count < len(matches) * 0.5:
                continue
            # Refinement uses only RANSAC inliers, no GT and no current depth
            # mutation. Global hypothesis depth verification remains in prepare.
            try:
                rvec, tvec = cv2.solvePnPRefineLM(object_points[inliers[:, 0]],
                                                 image_points[inliers[:, 0]], K, None, rvec, tvec)
            except cv2.error:
                continue
            candidate = np.eye(4)
            candidate[:3, :3] = cv2.Rodrigues(rvec)[0]
            candidate[:3, 3] = tvec.reshape(3)
            camera_z = (object_points @ candidate[:3, :3].T + candidate[:3, 3])[:, 2]
            if not _valid_pose(candidate) or np.any(camera_z <= 1e-4):
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
            errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            count = int(np.count_nonzero(np.isfinite(errors) & (errors <= 3.0)))
            if count < self.config.min_pnp_inliers or count < len(matches) * 0.5:
                continue
            self._feature_diagnostics["candidate_count"] += 1
            if count > best_count:
                best_pose, best_count = candidate, count
                self._last_pnp_inliers = count
                self._last_pnp_inlier_fraction = count / len(matches)
        return best_pose

    def prepare(self, *args, **kwargs):
        self._feature_diagnostics = {}
        return super().prepare(*args, **kwargs)

    def observe(self, *args, **kwargs):
        result = super().observe(*args, **kwargs)
        result["learned_correspondence"] = dict(self._feature_diagnostics)
        result["memory_anchor_counts"] = [len(x[0]) for x in self._keyframes]
        self.diagnostics = result
        return dict(result)
