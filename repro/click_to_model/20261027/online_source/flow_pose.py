"""Causal sparse optical-flow seeds before the current-frame neural 6D update.

This optional CPU LK/PnP layer keeps short-lived, individually depth-checked CAD
anchors separate from the stricter learned appearance memory. It neither emits
an output pose nor bypasses the caller's RefineNet. Existing XFeat recovery is
inherited unchanged. LK, PnP, checks, and feature refresh belong inside timing.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from learned_pose import LearnedRecoveryConfig, LearnedRecoveryController
from robust_pose import _valid_pose


@dataclass(frozen=True)
class FlowRecoveryConfig(LearnedRecoveryConfig):
    enable_prediction: bool = False
    flow_max_points: int = 96
    flow_refresh_interval: int = 10
    flow_refresh_min_points: int = 24
    flow_window_size: int = 21
    flow_pyramid_levels: int = 4
    flow_fb_threshold_px: float = 1.5
    flow_pnp_threshold_px: float = 3.0
    flow_min_inlier_fraction: float = 0.60
    flow_reprojection_improvement_px: float = 0.75
    flow_depth_score_slack: float = 0.03
    flow_corner_quality: float = 0.01
    flow_corner_distance_px: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        for name in ("flow_max_points", "flow_refresh_interval",
                     "flow_refresh_min_points", "flow_window_size",
                     "flow_pyramid_levels"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.flow_max_points < self.min_pnp_inliers:
            raise ValueError("flow_max_points must be >= min_pnp_inliers")
        if not self.min_pnp_inliers <= self.flow_refresh_min_points <= self.flow_max_points:
            raise ValueError("flow_refresh_min_points must lie within point limits")
        if self.flow_window_size < 3 or self.flow_window_size % 2 != 1:
            raise ValueError("flow_window_size must be odd and >= 3")
        for name in ("flow_fb_threshold_px", "flow_pnp_threshold_px",
                     "flow_reprojection_improvement_px", "flow_corner_distance_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("flow_min_inlier_fraction", "flow_corner_quality"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in (0,1]")
        if not 0 <= self.flow_depth_score_slack <= 1:
            raise ValueError("flow_depth_score_slack must lie in [0,1]")


class FlowRecoveryController(LearnedRecoveryController):
    """Same reset/prepare/observe API as LearnedRecoveryController.

    Flow seeds require strict current depth support plus an improvement in
    reprojection of the *same* current tracked points. This permits translation
    parallel to a nearly flat surface, for which scalar depth scores can tie.
    Neither depth nor optical flow establishes object identity or resolves an
    unobservable pose behind a complete occluder.
    """

    def __init__(self, config=None, extractor=None):
        config = config or FlowRecoveryConfig()
        if not isinstance(config, FlowRecoveryConfig):
            raise TypeError("expected FlowRecoveryConfig")
        super().__init__(config, extractor=extractor)
        self._flow_gray = None
        self._flow_pixels = np.empty((0, 2), np.float32)
        self._flow_anchors = np.empty((0, 3), np.float64)
        self._flow_pending_tracks = None
        self._flow_diagnostics = {}

    def reset(self, centered_pose, rgb, depth, K, model_points, model_normals,
              diameter, timestamp=None):
        result = super().reset(centered_pose, rgb, depth, K, model_points,
                               model_normals, diameter, timestamp=timestamp)
        # Flow remains usable for an explicit no-XFeat-recovery ablation.
        self._flow_tree = self._tree if self._tree is not None else cKDTree(model_points)
        rgb, depth, K = self._validate_frame(rgb, depth, K)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pixels, anchors = self._refresh(gray, depth, K, np.asarray(centered_pose))
        self._flow_gray, self._flow_pixels, self._flow_anchors = gray, pixels, anchors
        self._flow_pending_tracks = None
        self._flow_diagnostics = {"active_points": len(pixels), "selected": False,
                                  "reason": "registration_anchor_admission"}
        self.diagnostics["flow_correspondence"] = dict(self._flow_diagnostics)
        return result

    @staticmethod
    def _project(points, pose, K):
        camera = points @ pose[:3, :3].T + pose[:3, 3]
        projected = camera @ K.T
        safe_z = np.where(camera[:, 2] > 1e-4, projected[:, 2], np.nan)
        return projected[:, :2] / safe_z[:, None], camera[:, 2]

    def _admit_points(self, pixels, anchors, pose, depth, K):
        """Per-anchor observation test; this NEVER admits appearance memory."""
        if not len(pixels):
            return np.zeros(0, dtype=bool)
        finite = np.isfinite(pixels).all(1) & np.isfinite(anchors).all(1)
        nearest = np.rint(np.where(np.isfinite(pixels), pixels, -1)).astype(np.int64)
        inside = (finite & (nearest[:, 0] >= 0) & (nearest[:, 0] < depth.shape[1])
                  & (nearest[:, 1] >= 0) & (nearest[:, 1] < depth.shape[0]))
        ids = np.flatnonzero(inside)
        accepted = np.zeros(len(pixels), dtype=bool)
        if not len(ids):
            return accepted
        z = depth[nearest[ids, 1], nearest[ids, 0]]
        predicted, expected_z = self._project(anchors[ids], pose, K)
        valid = (np.isfinite(z) & (z > 1e-4) & (expected_z > 1e-4)
                 & (np.abs(z - expected_z) <= self._depth_tolerance)
                 & (np.linalg.norm(predicted - pixels[ids], axis=1)
                    <= self.config.flow_pnp_threshold_px))
        ids, z = ids[valid], z[valid]
        if not len(ids):
            return accepted
        rays = np.c_[pixels[ids], np.ones(len(ids))] @ np.linalg.inv(K).T
        observed_model = (rays * z[:, None] - pose[:3, 3]) @ pose[:3, :3]
        distance, _ = self._flow_tree.query(observed_model, k=1)
        uv, surface_z, _ = self._project_visible(pose, K, depth.shape)
        if not len(uv):
            return accepted
        projection_distance, surface_id = cKDTree(uv).query(pixels[ids], k=1)
        tolerance = min(self._depth_tolerance,
                        self._diameter * self.config.anchor_distance_diameter)
        valid = ((distance <= tolerance)
                 & (projection_distance <= self.config.max_anchor_projection_distance_px)
                 & (np.abs(z - surface_z[surface_id]) <= self._depth_tolerance))
        accepted[ids[valid]] = True
        return accepted

    def _refresh(self, gray, depth, K, pose):
        empty = (np.empty((0, 2), np.float32), np.empty((0, 3), np.float64))
        evidence = self._evidence(pose, depth, K)
        if (not self._supported(evidence)
                or evidence["occluded_fraction"] > self.config.max_memory_occluded_fraction):
            return empty
        uv, _, _ = self._project_visible(pose, K, depth.shape)
        if len(uv) < 3:
            return empty
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        try:
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.config.flow_max_points,
                qualityLevel=self.config.flow_corner_quality,
                minDistance=self.config.flow_corner_distance_px, mask=mask, blockSize=5)
        except cv2.error:
            return empty
        if corners is None:
            return empty
        pixels = corners.reshape(-1, 2).astype(np.float32)
        nearest = np.rint(pixels).astype(np.int64)
        z = depth[nearest[:, 1], nearest[:, 0]]
        valid = np.isfinite(z) & (z > 1e-4)
        pixels, z = pixels[valid], z[valid]
        rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
        anchors = (rays * z[:, None] - pose[:3, 3]) @ pose[:3, :3]
        admitted = self._admit_points(pixels, anchors, pose, depth, K)
        return pixels[admitted].copy(), anchors[admitted].copy()

    def _track_candidate(self, gray, depth, K):
        """Compute a proposal without mutating any committed controller state."""
        empty = (np.empty((0, 2), np.float32), np.empty((0, 3), np.float64))
        diagnostic = {"previous_points": len(self._flow_pixels), "fb_points": 0,
                      "pnp_inliers": 0, "depth_inliers": 0, "selected": False,
                      "pnp_solver_calls": 0, "reason": "insufficient_previous_points"}
        if (self._flow_gray is None or self._flow_gray.shape != gray.shape
                or len(self._flow_pixels) < self.config.min_pnp_inliers):
            return None, empty, diagnostic
        parameters = dict(winSize=(self.config.flow_window_size,) * 2,
                          maxLevel=self.config.flow_pyramid_levels,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        initial = self._flow_pixels.reshape(-1, 1, 2).astype(np.float32)
        try:
            forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
                self._flow_gray, gray, initial, None, **parameters)
            if forward is None or status_forward is None:
                diagnostic["reason"] = "forward_flow_failed"
                return None, empty, diagnostic
            safe_forward = np.where(np.isfinite(forward), forward, initial).astype(np.float32)
            backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._flow_gray, safe_forward, None, **parameters)
        except cv2.error:
            diagnostic["reason"] = "optical_flow_error"
            return None, empty, diagnostic
        if backward is None or status_backward is None:
            diagnostic["reason"] = "backward_flow_failed"
            return None, empty, diagnostic
        pixels = forward.reshape(-1, 2)
        error = np.linalg.norm(backward.reshape(-1, 2) - self._flow_pixels, axis=1)
        keep = ((status_forward.ravel() != 0) & (status_backward.ravel() != 0)
                & np.isfinite(pixels).all(1) & np.isfinite(error)
                & (error <= self.config.flow_fb_threshold_px)
                & (pixels[:, 0] >= 0) & (pixels[:, 0] < gray.shape[1] - 1)
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < gray.shape[0] - 1))
        pixels, anchors = pixels[keep].copy(), self._flow_anchors[keep].copy()
        tracks = (pixels, anchors)
        diagnostic["fb_points"] = len(pixels)
        diagnostic["reason"] = "insufficient_forward_backward_tracks"
        if len(pixels) < self.config.min_pnp_inliers:
            return None, tracks, diagnostic
        singular = np.linalg.svd(anchors - anchors.mean(0), compute_uv=False)
        image_singular = np.linalg.svd(pixels - pixels.mean(0), compute_uv=False)
        if (singular[1] < 1e-5 or image_singular[1] < 1e-3
                or len(np.unique(pixels, axis=0)) < self.config.min_pnp_inliers):
            diagnostic["reason"] = "degenerate_correspondences"
            return None, tracks, diagnostic
        diagnostic["pnp_solver_calls"] = 1
        diagnostic["reason"] = "pnp_rejected"
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                anchors, pixels, K, None, iterationsCount=100,
                reprojectionError=self.config.flow_pnp_threshold_px, confidence=.99,
                # ITERATIVE refits the consensus set using planar/nonplanar
                # initialization; EPNP's final fit can be ill-conditioned for a
                # nearly fronto-parallel object face despite good RANSAC inliers.
                flags=cv2.SOLVEPNP_ITERATIVE)
            count = 0 if inliers is None else len(inliers)
            if (not success or count < self.config.min_pnp_inliers
                    or count / len(pixels) < self.config.flow_min_inlier_fraction):
                return None, tracks, diagnostic
            ids = inliers.ravel()
            rvec, tvec = cv2.solvePnPRefineLM(anchors[ids], pixels[ids], K, None, rvec, tvec)
            candidate = np.eye(4)
            candidate[:3, :3] = cv2.Rodrigues(rvec)[0]
            candidate[:3, 3] = tvec.ravel()
        except cv2.error:
            return None, tracks, diagnostic
        if not _valid_pose(candidate):
            return None, tracks, diagnostic
        projection, camera_z = self._project(anchors, candidate, K)
        errors = np.linalg.norm(projection - pixels, axis=1)
        reprojection_inliers = (np.isfinite(errors) & (camera_z > 1e-4)
                                & (errors <= self.config.flow_pnp_threshold_px))
        diagnostic["pnp_inliers"] = int(reprojection_inliers.sum())
        admitted = self._admit_points(pixels, anchors, candidate, depth, K) & reprojection_inliers
        count = int(admitted.sum())
        diagnostic["depth_inliers"] = count
        diagnostic["reason"] = "insufficient_current_depth_inliers"
        if (count < self.config.min_pnp_inliers
                or count / len(pixels) < self.config.flow_min_inlier_fraction):
            return None, tracks, diagnostic
        diagnostic["candidate_median_reprojection_px"] = float(np.median(errors[admitted]))
        diagnostic["reason"] = "candidate_pending_comparison"
        return (candidate, pixels[admitted], anchors[admitted]), tracks, diagnostic

    def prepare(self, rgb, depth, K, timestamp=None):
        if self._trusted_pose is None:
            raise RuntimeError("reset must precede prepare")
        if self._pending:
            raise RuntimeError("observe must follow each prepare before the next frame")
        rgb, depth, K = self._validate_frame(rgb, depth, K)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        proposal, tracks, diagnostic = self._track_candidate(gray, depth, K)
        # super commits the frame clock only after its optional recovery succeeds.
        # Our proposal and cache remain local if it raises, allowing retry.
        seed = super().prepare(rgb, depth, K, timestamp=timestamp)
        if proposal is not None:
            candidate, pixels, anchors = proposal
            evidence = self._evidence(candidate, depth, K)
            seed_projection, _ = self._project(anchors, seed, K)
            seed_errors = np.linalg.norm(seed_projection - pixels, axis=1)
            seed_median = float(np.median(seed_errors)) if np.isfinite(seed_errors).all() else None
            diagnostic["base_median_reprojection_px"] = seed_median
            diagnostic["candidate_support_score"] = evidence["support_score"]
            diagnostic["candidate_coverage"] = evidence["coverage"]
            if not self._supported(evidence):
                diagnostic["reason"] = "candidate_depth_unsupported"
            elif (evidence["support_score"] < self.diagnostics["support_score"]
                  - self.config.flow_depth_score_slack):
                diagnostic["reason"] = "candidate_depth_score_regression"
            elif (seed_median is not None and seed_median
                  <= diagnostic["candidate_median_reprojection_px"]
                  + self.config.flow_reprojection_improvement_px):
                diagnostic["reason"] = "candidate_reprojection_not_better"
            else:
                seed = candidate.copy()
                self._pending_source = "flow_pnp"
                self._pending_seed = seed.copy()
                self._pending_seed_state_age = 0
                self._pending_relocalization_valid = False
                self.diagnostics.update(evidence)
                self.diagnostics.update(source="flow_pnp", seed_state_age_frames=0)
                diagnostic.update(selected=True, reason="current_depth_and_reprojection_improvement")
        self._flow_pending_tracks = (gray, *tracks)
        self._flow_diagnostics = diagnostic
        self.diagnostics["flow_correspondence"] = dict(diagnostic)
        return seed.copy()

    def observe(self, neural_pose_np, rgb, depth, K):
        if not self._pending:
            raise RuntimeError("prepare must precede observe")
        rgb, depth, K = self._validate_frame(rgb, depth, K)
        pixels, anchors = np.empty((0, 2), np.float32), np.empty((0, 3), np.float64)
        refreshed = False
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if _valid_pose(neural_pose_np):
            pose = np.asarray(neural_pose_np)
            _, tracked_pixels, tracked_anchors = self._flow_pending_tracks
            valid = self._admit_points(tracked_pixels, tracked_anchors, pose, depth, K)
            pixels, anchors = tracked_pixels[valid].copy(), tracked_anchors[valid].copy()
            if (len(pixels) < self.config.flow_refresh_min_points
                    or self._step % self.config.flow_refresh_interval == 0):
                fresh_pixels, fresh_anchors = self._refresh(gray, depth, K, pose)
                if len(fresh_pixels) >= len(pixels):
                    pixels, anchors, refreshed = fresh_pixels, fresh_anchors, True
        # Appearance extraction can fail in super.observe; do not commit the
        # cheap local cache until the neural observation transaction succeeds.
        result = super().observe(neural_pose_np, rgb, depth, K)
        if not result["working_state_updated"]:
            pixels, anchors = pixels[:0], anchors[:0]
        self._flow_gray, self._flow_pixels, self._flow_anchors = gray, pixels, anchors
        self._flow_pending_tracks = None
        self._flow_diagnostics.update(active_points=len(pixels), refreshed=refreshed)
        result["flow_correspondence"] = dict(self._flow_diagnostics)
        self.diagnostics = result
        return dict(result)
