"""Observation-gated seeds for a frozen, every-frame neural 6D refiner.

This module does not estimate a replacement output pose. ``prepare`` selects a
seed; the caller must run its neural refiner on the current frame and return that
refiner's output. ``observe`` only decides whether to admit the result to trusted
temporal state. ``verified`` means sparse depth support, NOT certified pose
correctness or a calibrated probability. Missing depth is not called occlusion.

Inputs/poses are CPU arrays, metres, and centered-CAD-to-camera transforms.
Without timestamps, velocities use frame units; with timestamps they use the
provided monotonically increasing time unit (normally seconds). No Euler angles,
target-specific training, frame skipping, or hidden re-registration is used.
An extreme single-step innovation guard only rejects catastrophic jumps; it does not resolve
same-depth distractors, object identity, symmetry, or calibrated confidence.
Rejected but numerically valid, bounded neural corrections can advance a separate
quarantined working pose. The default ``supported`` continuation policy requires
visible depth support before reusing that pose. The opt-in ``continuous`` policy
reuses the last bounded, valid working pose even with missing depth or complete
occlusion, separating local numerical continuation from observation trust. This
removes one stale-seed fallback mechanism, but can propagate bounded drift and
does not establish accuracy under occlusion. Neither policy writes appearance
memory until the strict observation and per-anchor memory gates pass.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RobustPoseConfig:
    max_points: int = 512
    depth_tolerance_m: float = 0.008
    depth_tolerance_diameter: float = 0.035
    min_inliers: int = 12
    min_coverage: float = 0.18
    max_contradiction_fraction: float = 0.30
    min_occluded_fraction: float = 0.50
    seed_improvement_margin: float = 0.03
    max_translation_diameter: float = 0.20
    max_rotation_radians: float = np.deg2rad(20.0)
    max_innovation_translation_diameter: float = 0.25
    max_innovation_rotation_radians: float = np.deg2rad(45.0)
    max_prediction_frames: int = 3
    enable_prediction: bool = True
    enable_quarantine: bool = True
    continuation_policy: str = "supported"
    enable_relocalization: bool = False
    relocalization_interval: int = 5
    keyframe_interval: int = 30
    max_keyframes: int = 4
    orb_features: int = 600
    min_pnp_inliers: int = 8
    anchor_distance_diameter: float = 0.06
    max_memory_occluded_fraction: float = 0.50
    max_anchor_projection_distance_px: float = 5.0

    def __post_init__(self):
        if (not isinstance(self.continuation_policy, str)
                or self.continuation_policy not in ("supported", "continuous")):
            raise ValueError("continuation_policy must be 'supported' or 'continuous'")
        for name in ("max_points", "min_inliers", "max_prediction_frames",
                     "relocalization_interval", "keyframe_interval",
                     "max_keyframes", "orb_features", "min_pnp_inliers"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("depth_tolerance_m", "depth_tolerance_diameter",
                     "max_translation_diameter", "max_rotation_radians",
                     "anchor_distance_diameter", "max_anchor_projection_distance_px",
                     "max_innovation_translation_diameter", "max_innovation_rotation_radians"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_coverage", "max_contradiction_fraction",
                     "min_occluded_fraction", "seed_improvement_margin",
                     "max_memory_occluded_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")


def _valid_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False
    rotation = pose[:3, :3]
    return bool(
        pose[2, 3] > 1e-4
        and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
        and abs(np.linalg.det(rotation) - 1.0) < 2e-3
    )


def _cap_vector(vector, maximum):
    length = np.linalg.norm(vector)
    return vector * min(1.0, maximum / max(length, 1e-15))


class RobustPoseController:
    """Optional seed/memory policy; the default excludes expensive ORB search.

    Call ``reset``, then exactly one ``prepare`` and ``observe`` per input frame.
    Rejected neural outputs MUST remain in the caller's complete pose/accuracy
    record, together with ``diagnostics['verified'] == False``. Never improve
    reported accuracy by excluding these frames or replacing their neural output
    with ``trusted_pose``. Full occlusion cannot determine an arbitrary 6D pose.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else RobustPoseConfig()
        if not isinstance(self.config, RobustPoseConfig):
            raise TypeError("config must be RobustPoseConfig")
        self.diagnostics = {"status": "uninitialized", "verified": False}
        self._trusted_pose = None
        self._pending = False
        self._keyframes = []
        self._last_pnp_inliers = 0
        self._last_pnp_inlier_fraction = 0.0

    @property
    def trusted_pose(self):
        return None if self._trusted_pose is None else self._trusted_pose.copy()

    @property
    def working_pose(self):
        """Quarantined convergence state, not a trusted or replacement output."""
        return None if self._trusted_pose is None else self._working_pose.copy()

    def configuration(self):
        return asdict(self.config)

    @staticmethod
    def _validate_frame(rgb, depth, K):
        rgb, depth, K = np.asarray(rgb), np.asarray(depth), np.asarray(K)
        if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
            raise ValueError("rgb must be HxWx3 and depth must be matching HxW")
        if min(depth.shape, default=0) <= 0:
            raise ValueError("empty frames are not supported")
        if rgb.dtype != np.uint8 or not np.issubdtype(depth.dtype, np.number):
            raise ValueError("rgb must be uint8; depth must be numeric metres")
        if (K.shape != (3, 3) or not np.isfinite(K).all()
                or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0, 0, 1], atol=1e-6)):
            raise ValueError("K must be finite calibrated 3x3 camera intrinsics")
        return rgb, depth, K.astype(np.float64, copy=False)

    def reset(self, centered_pose, rgb, depth, K, model_points, model_normals,
              diameter, timestamp=None):
        rgb, depth, K = self._validate_frame(rgb, depth, K)
        if not _valid_pose(centered_pose):
            raise ValueError("registration pose must be finite SE(3), in front of camera")
        points = np.asarray(model_points, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
                or not np.isfinite(points).all()):
            raise ValueError("model_points must be finite Nx3 with at least three points")
        normals = None
        if model_normals is not None:
            normals = np.asarray(model_normals, dtype=np.float64)
            if normals.shape != points.shape or not np.isfinite(normals).all():
                raise ValueError("model_normals must be finite and match model_points")
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(lengths, 1e-15)
        diameter = float(diameter)
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("diameter must be finite and positive")
        if timestamp is not None and not np.isfinite(float(timestamp)):
            raise ValueError("timestamp must be finite")
        indices = np.linspace(0, len(points) - 1,
                              min(len(points), self.config.max_points), dtype=int)
        self._points = points[indices].copy()
        self._normals = None if normals is None else normals[indices].copy()
        self._diameter = diameter
        self._depth_tolerance = max(self.config.depth_tolerance_m,
                                    diameter * self.config.depth_tolerance_diameter)
        self._trusted_pose = np.asarray(centered_pose, dtype=np.float64).copy()
        self._working_pose = self._trusted_pose.copy()
        self._timestamp_mode = timestamp is not None
        self._frame_time = float(timestamp) if timestamp is not None else 0.0
        self._trusted_time = self._frame_time
        self._translation_velocity = np.zeros(3)
        self._rotation_velocity = np.zeros(3)
        self._step = 0
        self._trusted_step = 0
        self._working_step = 0
        self._rejected_streak = 0
        self._pending = False
        self._keyframes = []
        self._tree = None
        if self.config.enable_relocalization:
            self._tree = cKDTree(points)
            self._orb = cv2.ORB_create(nfeatures=self.config.orb_features)
            self._remember(rgb, depth, K, self._trusted_pose)
        self.diagnostics = {
            **self._evidence(self._trusted_pose, depth, K),
            "status": "registered", "verified": False,
            "reason": "external_registration_not_tracking_verification",
            "source": "registration", "rejected_streak": 0,
            "relocalization_attempted": False, "frame_index": 0,
            "continuation_policy": self.config.continuation_policy,
            "trusted_age_frames": 0, "working_age_frames": 0,
            "seed_state_age_frames": 0, "working_state_updated": False,
        }
        return self.trusted_pose

    def _project_visible(self, pose, K, shape):
        camera_points = self._points @ pose[:3, :3].T + pose[:3, 3]
        valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-4)
        if self._normals is not None:
            camera_normals = self._normals @ pose[:3, :3].T
            normal_valid = np.linalg.norm(camera_normals, axis=1) > 0.5
            valid &= (~normal_valid) | (np.sum(camera_normals * camera_points, axis=1) < 0)
        ids = np.flatnonzero(valid)
        if not len(ids):
            return np.empty((0, 2), dtype=int), np.empty(0), ids
        projected = camera_points[ids] @ K.T
        coordinates = projected[:, :2] / projected[:, 2:3]
        height, width = shape
        inside = ((coordinates[:, 0] >= -0.5) & (coordinates[:, 0] < width - 0.5)
                  & (coordinates[:, 1] >= -0.5) & (coordinates[:, 1] < height - 0.5))
        ids = ids[inside]
        uv = np.rint(coordinates[inside]).astype(np.int64)
        if not len(ids):
            return uv, np.empty(0), ids
        # A sparse pixel z-buffer prevents coincident rear surfaces from being
        # counted as occluders. It is only an approximation to mesh visibility.
        z = camera_points[ids, 2]
        linear = uv[:, 1] * width + uv[:, 0]
        order = np.lexsort((z, linear))
        sorted_pixels = linear[order]
        unique = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
        keep = order[unique]
        return uv[keep], z[keep], ids[keep]

    def _evidence(self, pose, depth, K):
        uv, expected_z, _ = self._project_visible(pose, K, depth.shape)
        count = len(expected_z)
        measured = depth[uv[:, 1], uv[:, 0]] if count else np.empty(0)
        valid = np.isfinite(measured) & (measured > 1e-4)
        residual = measured - expected_z
        inlier = valid & (np.abs(residual) <= self._depth_tolerance)
        occluded = valid & (residual < -self._depth_tolerance)
        contradiction = valid & (residual > self._depth_tolerance)
        divisor = max(count, 1)
        coverage = float(inlier.sum() / divisor)
        contradictory = float(contradiction.sum() / divisor)
        return {
            "projected_count": count,
            "inlier_count": int(inlier.sum()),
            "coverage": coverage,
            "valid_depth_fraction": float(valid.sum() / divisor),
            "occluded_fraction": float(occluded.sum() / divisor),
            "contradiction_fraction": contradictory,
            "support_score": coverage - contradictory,
            "depth_tolerance_m": self._depth_tolerance,
        }

    def _supported(self, evidence):
        return bool(evidence["inlier_count"] >= self.config.min_inliers
                    and evidence["coverage"] >= self.config.min_coverage
                    and evidence["contradiction_fraction"] <= self.config.max_contradiction_fraction)

    def prepare(self, rgb, depth, K, timestamp=None):
        if self._trusted_pose is None:
            raise RuntimeError("reset must precede prepare")
        if self._pending:
            raise RuntimeError("observe must follow each prepare before the next frame")
        rgb, depth, K = self._validate_frame(rgb, depth, K)
        if (timestamp is not None) != self._timestamp_mode:
            raise ValueError("do not mix timestamp and frame-index time modes")
        next_time = float(timestamp) if timestamp is not None else self._frame_time + 1.0
        if not np.isfinite(next_time) or next_time <= self._frame_time:
            raise ValueError("timestamps must be finite and strictly increasing")
        next_step = self._step + 1
        seed = self.trusted_pose
        source = "trusted"
        best_evidence = self._evidence(seed, depth, K)
        continuation_depth_supported = False
        if self.config.enable_quarantine:
            evidence = (best_evidence
                        if np.array_equal(self._working_pose, self._trusted_pose)
                        else self._evidence(self._working_pose, depth, K))
            # Continuation eligibility and trusted-memory admission are separate
            # questions. ``supported`` retains the V3 coverage gate verbatim.
            # ``continuous`` treats working state as a numerical iterate, not a
            # visibility claim: sparse/missing depth must not repeatedly reset
            # a valid, bounded neural chain to stale trusted state. Appearance
            # memory and trusted velocity still require strict verification.
            continuation_depth_supported = bool(
                evidence["inlier_count"] >= self.config.min_inliers
                and evidence["coverage"] >= self.config.min_coverage)
            if (self.config.continuation_policy == "continuous"
                    or continuation_depth_supported):
                seed, source, best_evidence = self.working_pose, "continuation", evidence
        if (self.config.enable_prediction
                and self._rejected_streak < self.config.max_prediction_frames):
            dt = next_time - self._trusted_time
            prediction = self.trusted_pose
            prediction[:3, 3] += _cap_vector(
                self._translation_velocity * dt,
                self._diameter * self.config.max_translation_diameter)
            rotation = _cap_vector(self._rotation_velocity * dt,
                                   self.config.max_rotation_radians)
            prediction[:3, :3] = Rotation.from_rotvec(rotation).as_matrix() @ self._trusted_pose[:3, :3]
            if _valid_pose(prediction):
                evidence = self._evidence(prediction, depth, K)
                if (self._supported(evidence)
                        and evidence["support_score"] > best_evidence["support_score"]
                        + self.config.seed_improvement_margin):
                    seed, source, best_evidence = prediction, "prediction", evidence
        attempted = False
        self._last_pnp_inliers = 0
        self._last_pnp_inlier_fraction = 0.0
        if (self.config.enable_relocalization and self._rejected_streak >= 2
                and next_step % self.config.relocalization_interval == 0):
            attempted = True
            candidate = self._relocalize(rgb, K)
            pnp_verified = (self._last_pnp_inliers >= self.config.min_pnp_inliers
                            and self._last_pnp_inlier_fraction >= 0.5)
            if candidate is not None and pnp_verified and _valid_pose(candidate):
                evidence = self._evidence(candidate, depth, K)
                if (self._supported(evidence)
                        and evidence["support_score"] > best_evidence["support_score"]
                        + self.config.seed_improvement_margin):
                    seed, source, best_evidence = candidate, "relocalized", evidence
        # Commit the frame clock only after seed preparation succeeds. A failed
        # frame can be retried with the same timestamp without corrupting state.
        self._frame_time = next_time
        self._step = next_step
        self._pending = True
        self._pending_source = source
        self._pending_seed = seed.copy()
        # Age denotes the temporal state from which a seed was constructed,
        # not whether current-frame geometric support proves it correct.
        self._pending_seed_state_age = (
            next_step - self._working_step if source == "continuation" else
            0 if source == "relocalized" else next_step - self._trusted_step)
        self._pending_continuation_depth_supported = continuation_depth_supported
        self._relocalization_attempted = attempted
        self._pending_relocalization_valid = bool(
            source == "relocalized" and self._last_pnp_inliers >= self.config.min_pnp_inliers
            and self._last_pnp_inlier_fraction >= 0.5)
        self.diagnostics = {
            **best_evidence, "status": "pending_neural_observation", "verified": False,
            "source": source, "rejected_streak": self._rejected_streak,
            "relocalization_attempted": attempted, "frame_index": self._step,
            "continuation_policy": self.config.continuation_policy,
            "continuation_depth_supported": continuation_depth_supported,
            "trusted_age_frames": self._step - self._trusted_step,
            "working_age_frames": self._step - self._working_step,
            "seed_state_age_frames": self._pending_seed_state_age,
        }
        return seed.copy()

    def observe(self, neural_pose_np, rgb, depth, K):
        if not self._pending:
            raise RuntimeError("prepare must precede observe")
        rgb, depth, K = self._validate_frame(rgb, depth, K)
        pose_valid = _valid_pose(neural_pose_np)
        translation_innovation = rotation_innovation = None
        innovation_valid = False
        if pose_valid:
            pose = np.asarray(neural_pose_np, dtype=np.float64)
            evidence = self._evidence(pose, depth, K)
            # Limit the current neural correction, not the cumulative distance
            # from a stale trusted pose. A geometrically selected prediction,
            # quarantined iterate, or validated PnP seed can move the basin.
            translation_innovation = float(np.linalg.norm(pose[:3, 3] - self._pending_seed[:3, 3]))
            rotation_innovation = float(np.linalg.norm(Rotation.from_matrix(
                pose[:3, :3] @ self._pending_seed[:3, :3].T).as_rotvec()))
            innovation_valid = bool(
                translation_innovation <= self._diameter * self.config.max_innovation_translation_diameter
                and rotation_innovation <= self.config.max_innovation_rotation_radians)
        else:
            evidence = {"projected_count": 0, "inlier_count": 0, "coverage": 0.0,
                        "valid_depth_fraction": 0.0, "occluded_fraction": 0.0,
                        "contradiction_fraction": 0.0, "support_score": 0.0,
                        "depth_tolerance_m": self._depth_tolerance}
        verified = pose_valid and innovation_valid and self._supported(evidence)
        if verified:
            dt = self._frame_time - self._trusted_time
            translation_velocity = (pose[:3, 3] - self._trusted_pose[:3, 3]) / dt
            rotation_velocity = Rotation.from_matrix(
                pose[:3, :3] @ self._trusted_pose[:3, :3].T).as_rotvec() / dt
            if self._pending_source == "relocalized":
                translation_velocity.fill(0.0)
                rotation_velocity.fill(0.0)
            # Optional feature extraction may fail. Keep temporal state and the
            # pending observation unchanged until all admission work succeeds.
            if (self.config.enable_relocalization
                    and self._step % self.config.keyframe_interval == 0):
                self._remember(rgb, depth, K, pose)
            self._translation_velocity = translation_velocity
            self._rotation_velocity = rotation_velocity
            self._trusted_pose = pose.copy()
            self._trusted_time = self._frame_time
            self._trusted_step = self._step
            self._rejected_streak = 0
            status, reason = "verified", "sparse_depth_supported"
        else:
            self._rejected_streak += 1
            if not pose_valid:
                status, reason = "lost", "invalid_neural_pose"
            elif not innovation_valid:
                status, reason = "lost", "excessive_neural_innovation"
            elif evidence["projected_count"] < self.config.min_inliers:
                status, reason = "lost", "insufficient_projected_surface"
            elif evidence["valid_depth_fraction"] < self.config.min_coverage:
                status, reason = "lost", "insufficient_depth_not_occlusion"
            elif evidence["occluded_fraction"] >= self.config.min_occluded_fraction:
                status, reason = "occluded", "foreground_depth_occlusion_evidence"
            else:
                status, reason = "lost", "insufficient_depth_consistency"
        if pose_valid and innovation_valid:
            self._working_pose = pose.copy()
            self._working_step = self._step
        self._pending = False
        self.diagnostics = {
            **evidence, "status": status, "reason": reason, "verified": bool(verified),
            "neural_pose_valid": pose_valid, "source": self._pending_source,
            "translation_innovation_m": translation_innovation,
            "rotation_innovation_radians": rotation_innovation,
            "innovation_reference": "current_selected_seed",
            "relocalization_pnp_verified": self._pending_relocalization_valid,
            "relocalization_pnp_inliers": self._last_pnp_inliers,
            "rejected_streak": self._rejected_streak,
            "relocalization_attempted": self._relocalization_attempted,
            "frame_index": self._step, "keyframe_count": len(self._keyframes),
            "continuation_policy": self.config.continuation_policy,
            "continuation_depth_supported": self._pending_continuation_depth_supported,
            "trusted_age_frames": self._step - self._trusted_step,
            "working_age_frames": self._step - self._working_step,
            "seed_state_age_frames": self._pending_seed_state_age,
            "working_state_updated": bool(pose_valid and innovation_valid),
        }
        return dict(self.diagnostics)

    def _remember(self, rgb, depth, K, pose):
        # Whole-frame support never licenses all pixels inside the object hull:
        # that hull can contain a foreground occluder. This guard also applies
        # to registration memory, before any tracking verification has occurred.
        evidence = self._evidence(pose, depth, K)
        if (not self._supported(evidence)
                or evidence["occluded_fraction"] > self.config.max_memory_occluded_fraction):
            return
        uv, surface_z, _ = self._project_visible(pose, K, depth.shape)
        if len(uv) < 3:
            return
        mask = np.zeros(depth.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 255)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, mask)
        if descriptors is None:
            return
        pixels = np.asarray([point.pt for point in keypoints], dtype=np.float64)
        nearest = np.rint(pixels).astype(int)
        z = depth[nearest[:, 1], nearest[:, 0]]
        valid = np.isfinite(z) & (z > 1e-4)
        pixels, z, descriptors = pixels[valid], z[valid], descriptors[valid]
        if not len(z):
            return
        rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
        camera_points = rays * z[:, None]
        anchors = (camera_points - pose[:3, 3]) @ pose[:3, :3]
        distance, _ = self._tree.query(anchors, k=1)
        anchor_tolerance = min(self._depth_tolerance,
                               self._diameter * self.config.anchor_distance_diameter)
        projection_distance, surface_id = cKDTree(uv).query(pixels, k=1)
        valid = ((distance <= anchor_tolerance)
                 & (projection_distance <= self.config.max_anchor_projection_distance_px)
                 & (np.abs(z - surface_z[surface_id]) <= self._depth_tolerance))
        if valid.sum() < self.config.min_pnp_inliers:
            return
        self._keyframes.append((anchors[valid].astype(np.float32), descriptors[valid].copy()))
        if len(self._keyframes) > self.config.max_keyframes:
            # Keep registration appearance plus the most recent validated views.
            self._keyframes.pop(1 if self.config.max_keyframes > 1 else 0)

    def _relocalize(self, rgb, K):
        self._last_pnp_inliers = 0
        self._last_pnp_inlier_fraction = 0.0
        if not self._keyframes:
            return None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        current_points, current_descriptors = self._orb.detectAndCompute(gray, None)
        if current_descriptors is None or len(current_descriptors) < 2:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        best_pose, best_count = None, 0
        for anchors, descriptors in self._keyframes:
            pairs = matcher.knnMatch(descriptors, current_descriptors, k=2)
            matches = [pair[0] for pair in pairs if len(pair) == 2
                       and pair[0].distance < 0.75 * pair[1].distance]
            # Enforce one-to-one image assignments before RANSAC.
            unique = {}
            for match in sorted(matches, key=lambda item: item.distance):
                unique.setdefault(match.trainIdx, match)
            matches = list(unique.values())
            if len(matches) < self.config.min_pnp_inliers:
                continue
            object_points = np.asarray([anchors[m.queryIdx] for m in matches], dtype=np.float32)
            image_points = np.asarray([current_points[m.trainIdx].pt for m in matches], dtype=np.float32)
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points, image_points, K, None, iterationsCount=100,
                    reprojectionError=3.0, confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
            except cv2.error:
                continue
            count = 0 if inliers is None else len(inliers)
            if (not success or count < self.config.min_pnp_inliers
                    or count < len(matches) * 0.5 or count <= best_count):
                continue
            candidate = np.eye(4)
            candidate[:3, :3] = cv2.Rodrigues(rvec)[0]
            candidate[:3, 3] = tvec.reshape(3)
            if _valid_pose(candidate):
                best_pose, best_count = candidate, count
                self._last_pnp_inliers = count
                self._last_pnp_inlier_fraction = count / len(matches)
        return best_pose
