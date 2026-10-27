"""Transactional RGB-D point correspondences producing complete SE(3) seeds.

RGB is uint8 RGB, depth is registered camera-Z in metres, K is calibrated, and
poses map object coordinates to camera coordinates. A proposal is not a verified
pose or an object identity claim. The caller must run current-frame refinement
and independent validity/innovation checks before committing it.

No model weights are required. The default backend is forward/backward LK;
registration-only ORB anchors provide bounded relocalization proposals. Learned
point trackers can implement the stateless PointTracker protocol. This module
does not claim to integrate SAM3, CoTracker, or TAPIR.
"""

from dataclasses import dataclass
import hashlib
import hmac
import secrets
from typing import Protocol

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RGBDSeedConfig:
    max_points: int = 160
    min_points: int = 8
    corner_quality: float = 0.01
    corner_distance_px: float = 5.0
    lk_window: int = 21
    lk_levels: int = 3
    fb_threshold_px: float = 1.5
    residual_threshold_m: float = 0.008
    min_inlier_fraction: float = 0.60
    ransac_iterations: int = 128
    random_seed: int = 0
    min_geometry_spread_m: float = 0.003
    min_geometry_ratio: float = 0.02
    min_mask_support: float = 0.65
    max_mask_area_ratio: float = 3.0
    min_depth_m: float = 0.02
    max_depth_m: float = 10.0
    max_depth_jump_m: float = 0.03
    commit_reprojection_px: float = 3.0
    enable_relocalization: bool = True
    orb_max_features: int = 800
    orb_ratio: float = 0.75
    orb_max_hamming: float = 64.0
    replenish_min_points: int = 32

    def __post_init__(self):
        for name in ("max_points", "min_points", "lk_window", "ransac_iterations",
                     "orb_max_features", "replenish_min_points"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 3 <= self.min_points <= self.max_points:
            raise ValueError("require 3 <= min_points <= max_points")
        if not self.min_points <= self.replenish_min_points <= self.max_points:
            raise ValueError("replenish_min_points must lie within point limits")
        if self.lk_window < 3 or self.lk_window % 2 != 1:
            raise ValueError("lk_window must be odd and >= 3")
        for name in ("lk_levels", "random_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("corner_quality", "min_inlier_fraction", "min_geometry_ratio",
                     "min_mask_support", "orb_ratio"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in ("corner_distance_px", "fb_threshold_px", "residual_threshold_m",
                     "min_geometry_spread_m", "min_depth_m", "max_depth_m",
                     "max_depth_jump_m", "commit_reprojection_px", "orb_max_hamming"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.max_mask_area_ratio) or self.max_mask_area_ratio < 1:
            raise ValueError("max_mask_area_ratio must be finite and >= 1")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must exceed min_depth_m")
        if type(self.enable_relocalization) is not bool:
            raise ValueError("enable_relocalization must be bool")


@dataclass(frozen=True)
class PointTracks:
    pixels: np.ndarray
    valid: np.ndarray
    fb_error_px: np.ndarray


class PointTracker(Protocol):
    def track(self, previous_gray, current_gray, points) -> PointTracks:
        """No persistent mutation: outputs retain the order/number of points.

        Learned backends must handle confidence/occlusion internally and return
        false valid flags for unsupported tracks. Stateful online predictors
        require their own rollback adapter before satisfying this protocol.
        """


class LKPointTracker:
    def __init__(self, config=None):
        self.config = config or RGBDSeedConfig()

    def track(self, previous_gray, current_gray, points):
        points = np.asarray(points, np.float32).reshape(-1, 2)
        empty = PointTracks(points.copy(), np.zeros(len(points), bool),
                            np.full(len(points), np.inf))
        if not len(points) or previous_gray.shape != current_gray.shape:
            return empty
        params = dict(winSize=(self.config.lk_window,) * 2,
                      maxLevel=self.config.lk_levels,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        initial = points[:, None].copy()
        try:
            forward, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray, current_gray, initial, None, **params)
            if forward is None or forward_ok is None:
                return empty
            safe = np.where(np.isfinite(forward), forward, initial).astype(np.float32)
            backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
                current_gray, previous_gray, safe, None, **params)
        except cv2.error:
            return empty
        if backward is None or backward_ok is None:
            return empty
        pixels = forward.reshape(-1, 2)
        error = np.linalg.norm(backward.reshape(-1, 2) - points, axis=1)
        valid = (forward_ok.ravel().astype(bool) & backward_ok.ravel().astype(bool)
                 & np.isfinite(pixels).all(1) & np.isfinite(error)
                 & (error <= self.config.fb_threshold_px))
        return PointTracks(pixels.copy(), valid, error)


def _pose(value):
    pose = np.asarray(value, np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
        raise ValueError("pose must be a finite rigid object-to-camera 4x4 transform")
    return pose.copy()


def _geometry(points, config):
    if len(points) < 3:
        return False, 0., 0.
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False) / np.sqrt(len(points))
    ratio = float(singular[1] / max(singular[0], 1e-15))
    # Three noncollinear 3D correspondences determine a proper rigid transform;
    # planar support is valid, whereas rank-one/point support is not.
    return (bool(singular[1] >= config.min_geometry_spread_m
                 and ratio >= config.min_geometry_ratio), ratio, float(singular[1]))


def _kabsch(source, target):
    source_center, target_center = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, target_center - rotation @ source_center
    return pose


def robust_rigid_transform(source, target, config=None):
    """Return (object-to-camera pose or None, inlier mask, diagnostics).

    Uses deterministic 3-point RANSAC and proper-rotation Kabsch refitting. No
    scale is fitted: a distractor at incompatible depth cannot be explained by
    silently scaling the target. Diagnostics describe correspondence geometry,
    not symmetry-aware observability of the underlying physical object.
    """
    config = config or RGBDSeedConfig()
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape:
        raise ValueError("source and target must have matching Nx3 shapes")
    inliers = np.zeros(len(source), bool)
    diagnostic = dict(inlier_count=0, inlier_fraction=0., residual_m=None,
                      conditioning=0., geometry_spread_m=0., observable=False,
                      reason="insufficient_depth_correspondences")
    finite = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    ids = np.flatnonzero(finite)
    if len(ids) < config.min_points:
        return None, inliers, diagnostic
    # Repeated correspondences are not independent geometric evidence. Keep
    # at most one occurrence of either 3D endpoint, with deterministic order.
    _, source_unique = np.unique(source[ids], axis=0, return_index=True)
    ids = ids[np.sort(source_unique)]
    _, target_unique = np.unique(target[ids], axis=0, return_index=True)
    ids = ids[np.sort(target_unique)]
    if len(ids) < config.min_points:
        diagnostic["reason"] = "insufficient_unique_correspondences"
        return None, inliers, diagnostic
    a, b = source[ids], target[ids]
    for points in (a, b):
        valid, ratio, spread = _geometry(points, config)
        if not valid:
            diagnostic.update(reason="degenerate_correspondences", conditioning=ratio,
                              geometry_spread_m=spread)
            return None, inliers, diagnostic
    rng = np.random.default_rng(config.random_seed)
    best_count, best_error, best = 0, np.inf, None
    # Include the all-point fit for clean or mildly noisy correspondences.
    for iteration in range(config.ransac_iterations + 1):
        selected = np.arange(len(a)) if iteration == 0 else rng.choice(len(a), 3, replace=False)
        if not _geometry(a[selected], config)[0] or not _geometry(b[selected], config)[0]:
            continue
        candidate = _kabsch(a[selected], b[selected])
        errors = np.linalg.norm(a @ candidate[:3, :3].T + candidate[:3, 3] - b, axis=1)
        consensus = errors <= config.residual_threshold_m
        count = int(consensus.sum())
        median = float(np.median(errors[consensus])) if count else np.inf
        if count > best_count or (count == best_count and median < best_error):
            best_count, best_error, best = count, median, consensus
        if count == len(a):
            break
    if best is None or best_count < config.min_points:
        diagnostic["reason"] = "insufficient_rigid_inliers"
        return None, inliers, diagnostic
    for _ in range(3):
        candidate = _kabsch(a[best], b[best])
        errors = np.linalg.norm(a @ candidate[:3, :3].T + candidate[:3, 3] - b, axis=1)
        updated = errors <= config.residual_threshold_m
        if np.array_equal(updated, best) or updated.sum() < config.min_points:
            break
        best = updated
    # Refit the final consensus and report only support for that exact transform.
    candidate = _kabsch(a[best], b[best])
    errors = np.linalg.norm(a @ candidate[:3, :3].T + candidate[:3, 3] - b, axis=1)
    best = errors <= config.residual_threshold_m
    count = int(best.sum())
    fraction = count / len(source)
    geometry = [_geometry(points[best], config) for points in (a, b)]
    diagnostic.update(inlier_count=count, inlier_fraction=float(fraction),
                      residual_m=float(np.median(errors[best])) if count else None,
                      conditioning=min(item[1] for item in geometry),
                      geometry_spread_m=min(item[2] for item in geometry),
                      observable=all(item[0] for item in geometry))
    if count < config.min_points or fraction < config.min_inlier_fraction:
        diagnostic["reason"] = "insufficient_rigid_consensus"
        return None, inliers, diagnostic
    if not diagnostic["observable"]:
        diagnostic["reason"] = "degenerate_consensus"
        return None, inliers, diagnostic
    inliers[ids[best]] = True
    diagnostic["reason"] = "rigid_seed_pending_verification"
    return candidate, inliers, diagnostic


@dataclass(frozen=True)
class RGBDSeedProposal:
    pose: object
    accepted: bool
    diagnostics: dict
    token: int


@dataclass
class _Pending:
    proposal: RGBDSeedProposal
    gray: np.ndarray
    pixels: np.ndarray
    anchors: np.ndarray
    observed: np.ndarray
    K: np.ndarray
    mask_area: int


class RGBDSeedTracker:
    """reset -> propose -> externally verify -> commit/reject.

    All trusted frames, points, poses and descriptors are copied. Failed or
    rejected proposals cannot update them. Textureless initialization is valid
    but yields no point seed; geometric/neural tracking must handle that case.
    """

    def __init__(self, config=None, backend=None):
        self.config = config or RGBDSeedConfig()
        if not isinstance(self.config, RGBDSeedConfig):
            raise TypeError("config must be RGBDSeedConfig")
        self.backend = backend or LKPointTracker(self.config)
        self._gray = self._pose = self._pending = None
        self._pixels = np.empty((0, 2), np.float32)
        self._anchors = self._orb_anchors = np.empty((0, 3), np.float64)
        self._orb_descriptors = None
        self._mask_area, self._generation = 0, 0
        self._memory_identity = object()
        self._memory_revision = 0
        self._snapshot_key = secrets.token_bytes(32)

    @property
    def memory_identity(self):
        """Process-local identity; never confused with an instance identity claim."""
        return self._memory_identity

    @property
    def memory_revision(self):
        """Changes only when persistent local correspondence memory changes."""
        return self._memory_revision

    @property
    def initialized(self):
        return self._pose is not None

    @property
    def trusted_pose(self):
        return None if self._pose is None else self._pose.copy()

    @property
    def active_points(self):
        return self._pixels.copy()

    @property
    def needs_replenishment(self):
        return len(self._pixels) < self.config.replenish_min_points

    def _frame(self, rgb, depth, K, mask):
        rgb, depth, K = np.asarray(rgb), np.asarray(depth), np.asarray(K, np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be HxWx3 uint8 RGB")
        if depth.shape != rgb.shape[:2] or not np.issubdtype(depth.dtype, np.number):
            raise ValueError("depth must be registered HxW numeric camera-Z in metres")
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0
                or K[1, 1] <= 0 or abs(np.linalg.det(K)) < 1e-12
                or not np.allclose(K[2], [0, 0, 1])):
            raise ValueError("K must be a finite calibrated 3x3 intrinsic matrix")
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != depth.shape or not np.isfinite(mask).all():
                raise ValueError("mask must be finite HxW")
            mask = mask.astype(bool).copy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), depth, K.copy(), mask

    @staticmethod
    def _inside(pixels, shape):
        finite = np.isfinite(pixels).all(1)
        # Clip before integer conversion; malformed backends cannot overflow.
        safe = np.where(np.isfinite(pixels), pixels, -1)
        safe = np.clip(safe, -1, max(shape) + 1)
        indices = np.rint(safe).astype(np.int64)
        inside = (finite & (indices[:, 0] >= 1) & (indices[:, 0] < shape[1] - 1)
                  & (indices[:, 1] >= 1) & (indices[:, 1] < shape[0] - 1))
        return indices, inside

    def _lift(self, pixels, depth, K):
        pixels = np.asarray(pixels, np.float64)
        indices, inside = self._inside(pixels, depth.shape)
        result = np.full((len(pixels), 3), np.nan)
        valid = np.zeros(len(pixels), bool)
        ids = np.flatnonzero(inside)
        if not len(ids):
            return result, valid
        x, y = indices[ids].T
        z = np.asarray(depth[y, x], np.float64)
        good = np.isfinite(z) & (z >= self.config.min_depth_m) & (z <= self.config.max_depth_m)
        # Reject an actual discontinuity; do not median-fill missing depth or
        # let foreground/background interpolation create invented 3D tracks.
        neighbors = np.stack([depth[y + dy, x + dx] for dy in (-1, 0, 1)
                              for dx in (-1, 0, 1)], axis=1)
        good &= np.isfinite(neighbors).all(1) & (neighbors >= self.config.min_depth_m).all(1)
        good &= np.ptp(neighbors, axis=1) <= self.config.max_depth_jump_m
        ids, z = ids[good], z[good]
        rays = np.c_[pixels[ids], np.ones(len(ids))] @ np.linalg.inv(K).T
        result[ids] = rays * z[:, None]
        valid[ids] = True
        return result, valid

    def _orb(self, gray, mask=None):
        detector = cv2.ORB_create(nfeatures=self.config.orb_max_features, edgeThreshold=15)
        keys, descriptors = detector.detectAndCompute(
            gray, None if mask is None else mask.astype(np.uint8) * 255)
        pixels = np.asarray([key.pt for key in keys], np.float32).reshape(-1, 2)
        return pixels, descriptors

    def reset(self, rgb, depth, K, mask, pose):
        """Initialize from a user/externally verified mask and object pose.

        A failed reset preserves the old consistent state; no partial template
        replacement occurs. Every successful reset invalidates old proposals.
        """
        gray, depth, K, mask = self._frame(rgb, depth, K, mask)
        pose = _pose(pose)
        if mask is None or not mask.any():
            raise ValueError("reset requires a nonempty verified object mask")
        selection = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=self.config.max_points,
                    qualityLevel=self.config.corner_quality,
                    minDistance=self.config.corner_distance_px, mask=selection, blockSize=5)
        pixels = np.empty((0, 2), np.float32) if corners is None else corners.reshape(-1, 2)
        camera, valid = self._lift(pixels, depth, K)
        pixels = pixels[valid].copy()
        anchors = (camera[valid] - pose[:3, 3]) @ pose[:3, :3]
        orb_anchors, descriptors = np.empty((0, 3)), None
        if self.config.enable_relocalization:
            orb_pixels, descriptors = self._orb(gray, selection)
            camera, valid = self._lift(orb_pixels, depth, K)
            orb_anchors = (camera[valid] - pose[:3, 3]) @ pose[:3, :3]
            descriptors = None if descriptors is None else descriptors[valid].copy()
        # Atomic replacement after every fallible staging operation completed.
        self._gray, self._pose = gray.copy(), pose
        self._pixels, self._anchors = pixels, anchors.copy()
        self._orb_anchors, self._orb_descriptors = orb_anchors.copy(), descriptors
        self._mask_area, self._pending = int(mask.sum()), None
        self._generation += 1
        self._memory_revision += 1
        return dict(active_points=len(pixels), recovery_anchors=len(orb_anchors),
                    reason="registration", identity_verified=False)

    def replenish(self, rgb, depth, K, mask, verified_pose, model_points=None):
        """Refresh local tracks from an externally verified CAD-supported view.

        Invoke after commit/reject, and only after independent current-frame
        validity and continuity/identity admission. This method cannot certify
        those conditions. It requires explicit CAD geometry and a caller's
        current supported mask; it never updates the registration ORB library.
        Candidate anchors are snapped to CAD points and checked against current
        depth and image reprojection before a single atomic memory replacement.
        A failed admission returns updated=False and preserves the old state.
        """
        if not self.initialized:
            raise RuntimeError("reset must precede replenish")
        if self._pending is not None:
            raise RuntimeError("commit or reject the pending proposal before replenish")
        gray, depth, K, mask = self._frame(rgb, depth, K, mask)
        pose = _pose(verified_pose)
        if gray.shape != self._gray.shape:
            raise ValueError("frame dimensions changed; reset with a verified registration")
        if mask is None:
            raise ValueError("replenish requires a verified CAD-supported mask")
        model = np.asarray(model_points, np.float64)
        if (model.ndim != 2 or model.shape[1:] != (3,) or len(model) < 3
                or not np.isfinite(model).all()):
            raise ValueError("replenish requires finite Nx3 object-frame CAD surface points")
        diagnostic = dict(updated=False, active_points=len(self._pixels),
                          candidate_points=0, admitted_points=0,
                          reason="insufficient_cad_supported_corners",
                          identity_verified=False, orb_dictionary_updated=False)
        selection = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=self.config.max_points,
                    qualityLevel=self.config.corner_quality,
                    minDistance=self.config.corner_distance_px, mask=selection, blockSize=5)
        if corners is None:
            return diagnostic
        pixels = corners.reshape(-1, 2)
        camera, valid = self._lift(pixels, depth, K)
        pixels, camera = pixels[valid], camera[valid]
        diagnostic["candidate_points"] = len(pixels)
        if len(pixels) < self.config.min_points:
            return diagnostic
        local = (camera - pose[:3, 3]) @ pose[:3, :3]
        distance, surface_ids = cKDTree(model).query(local, k=1)
        anchors = model[surface_ids]
        expected = anchors @ pose[:3, :3].T + pose[:3, 3]
        projected = expected @ K.T
        reprojection = projected[:, :2] / np.maximum(projected[:, 2:], 1e-12)
        keep = ((distance <= self.config.residual_threshold_m)
                & (np.abs(expected[:, 2] - camera[:, 2]) <= self.config.residual_threshold_m)
                & (expected[:, 2] > self.config.min_depth_m)
                & (np.linalg.norm(reprojection - pixels, axis=1)
                   <= self.config.commit_reprojection_px))
        # Sparse CAD sampling can map many image corners onto one anchor. Do
        # not let repeated anchors inflate support; keep the closest match.
        ids = np.flatnonzero(keep)
        selected, used = [], set()
        for index in ids[np.argsort(distance[ids], kind="stable")]:
            key = tuple(anchors[index])
            if key not in used:
                used.add(key)
                selected.append(index)
        selected = np.asarray(selected, np.int64)
        diagnostic["admitted_points"] = len(selected)
        if len(selected) < self.config.min_points:
            return diagnostic
        if not _geometry(anchors[selected], self.config)[0]:
            diagnostic["reason"] = "degenerate_cad_supported_corners"
            return diagnostic
        staged_pixels, staged_anchors = pixels[selected].copy(), anchors[selected].copy()
        self._gray, self._pose = gray.copy(), pose
        self._pixels, self._anchors = staged_pixels, staged_anchors
        self._mask_area = int(mask.sum())
        self._memory_revision += 1
        diagnostic.update(updated=True, active_points=len(selected), reason="verified_cad_replenishment")
        return diagnostic

    def _candidate(self, pixels, anchors, valid, depth, K, mask, source):
        diagnostic = dict(seed_source=source, identity_verified=False,
                          tracked_points=int(valid.sum()), mask_support=None,
                          inlier_count=0, inlier_fraction=0., residual_m=None,
                          conditioning=0., observable=False)
        indices, inside = self._inside(pixels, depth.shape)
        valid = valid & inside
        if mask is not None and valid.any():
            ids = np.flatnonzero(valid)
            supported = mask[indices[ids, 1], indices[ids, 0]]
            diagnostic["mask_support"] = float(supported.mean())
            if supported.mean() < self.config.min_mask_support:
                diagnostic["reason"] = "mask_track_disagreement"
                return None, pixels[:0], anchors[:0], anchors[:0], diagnostic
            valid[ids[~supported]] = False
        observed, depth_valid = self._lift(pixels, depth, K)
        selected = valid & depth_valid
        diagnostic["depth_correspondences"] = int(selected.sum())
        pose, inliers, rigid_diagnostic = robust_rigid_transform(
            anchors[selected], observed[selected], self.config)
        diagnostic.update(rigid_diagnostic)
        return (pose, pixels[selected][inliers].copy(), anchors[selected][inliers].copy(),
                observed[selected][inliers].copy(), diagnostic)

    def _recover(self, gray, depth, K, mask):
        pixels, descriptors = self._orb(gray, mask=None)
        saved = self._orb_descriptors
        empty = (None, pixels[:0], self._anchors[:0], self._anchors[:0],
                 dict(seed_source="orb_rgbd", identity_verified=False,
                      reason="insufficient_recovery_matches", inlier_count=0,
                      inlier_fraction=0., residual_m=None, conditioning=0., observable=False))
        if descriptors is None or saved is None or len(saved) < 2 or len(descriptors) < 2:
            return empty
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        forward = matcher.knnMatch(saved, descriptors, k=2)
        backward = matcher.knnMatch(descriptors, saved, k=2)
        def admissible(pair):
            return (len(pair) == 2 and pair[0].distance <= self.config.orb_max_hamming
                    and pair[0].distance < self.config.orb_ratio * pair[1].distance)
        reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in backward if admissible(pair)}
        pairs = [(pair[0].queryIdx, pair[0].trainIdx) for pair in forward
                 if admissible(pair) and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
        if len(pairs) < self.config.min_points:
            return empty
        query, target = np.asarray(pairs).T
        return self._candidate(pixels[target], self._orb_anchors[query],
                    np.ones(len(query), bool), depth, K, mask, "orb_rgbd")

    def propose(self, rgb, depth, K, mask=None):
        if not self.initialized:
            raise RuntimeError("reset must precede propose")
        if self._pending is not None:
            raise RuntimeError("commit or reject the pending proposal first")
        gray, depth, K, mask = self._frame(rgb, depth, K, mask)
        if gray.shape != self._gray.shape:
            raise ValueError("frame dimensions changed; reset with a verified registration")
        mask_area = self._mask_area if mask is None else int(mask.sum())
        ratio = mask_area / max(self._mask_area, 1)
        area_ok = 1 / self.config.max_mask_area_ratio <= ratio <= self.config.max_mask_area_ratio
        if not area_ok:
            result = (None, self._pixels[:0], self._anchors[:0], self._anchors[:0],
                      dict(seed_source="lk_rgbd", identity_verified=False,
                           reason="mask_area_jump", mask_area_ratio=ratio, inlier_count=0,
                           inlier_fraction=0., residual_m=None, conditioning=0., observable=False))
        else:
            # Pass copies so an untrusted backend cannot overwrite active anchors.
            tracks = self.backend.track(self._gray.copy(), gray.copy(), self._pixels.copy())
            pixels = np.asarray(tracks.pixels, np.float64)
            valid, fb = np.asarray(tracks.valid), np.asarray(tracks.fb_error_px, np.float64)
            if (pixels.shape != self._pixels.shape or valid.shape != (len(pixels),)
                    or valid.dtype != bool or fb.shape != (len(pixels),)):
                raise ValueError("point backend must return Nx2 pixels and N bool/float arrays")
            valid = valid & np.isfinite(pixels).all(1) & np.isfinite(fb)
            valid &= (fb >= 0) & (fb <= self.config.fb_threshold_px)
            result = self._candidate(pixels, self._anchors, valid, depth, K, mask, "lk_rgbd")
            if (result[0] is None and self.config.enable_relocalization
                    and result[4]["reason"] != "mask_track_disagreement"):
                recovery = self._recover(gray, depth, K, mask)
                if recovery[0] is not None:
                    recovery[4]["local_failure"] = result[4]["reason"]
                    result = recovery
                else:
                    result[4]["recovery_failure"] = recovery[4]["reason"]
        pose, pixels, anchors, observed, diagnostic = result
        self._generation += 1
        public_pose = None if pose is None else pose.copy()
        if public_pose is not None:
            public_pose.setflags(write=False)
        proposal = RGBDSeedProposal(public_pose, pose is not None, dict(diagnostic), self._generation)
        self._pending = _Pending(proposal, gray.copy(), pixels.copy(), anchors.copy(),
                                 observed.copy(), K.copy(), mask_area)
        return proposal

    def commit(self, proposal, verified_pose=None):
        """Commit only after the caller's independent current-frame verification.

        The verified pose is mandatory. Agreement is checked on the original
        immutable observations; a rejected neural pose cannot refresh memory.
        A false return ends the transaction and preserves all trusted state.
        """
        pending = self._pending
        if pending is None or pending.proposal is not proposal:
            raise ValueError("proposal is stale or belongs to another tracker")
        if not proposal.accepted:
            self._pending = None
            return False
        if verified_pose is None:
            raise ValueError("commit requires an independently verified pose")
        pose = _pose(verified_pose)
        camera = pending.anchors @ pose[:3, :3].T + pose[:3, 3]
        error = np.linalg.norm(camera - pending.observed, axis=1)
        projected = camera @ pending.K.T
        pixel_error = np.linalg.norm(projected[:, :2] / np.maximum(projected[:, 2:], 1e-12)
                                    - pending.pixels, axis=1)
        keep = ((error <= self.config.residual_threshold_m)
                & (pixel_error <= self.config.commit_reprojection_px)
                & (camera[:, 2] > self.config.min_depth_m))
        if (keep.sum() < self.config.min_points or keep.mean() < self.config.min_inlier_fraction
                or not _geometry(pending.anchors[keep], self.config)[0]):
            self._pending = None
            return False
        staged_pixels, staged_anchors = pending.pixels[keep].copy(), pending.anchors[keep].copy()
        self._gray, self._pose = pending.gray, pose
        self._pixels, self._anchors = staged_pixels, staged_anchors
        self._mask_area, self._pending = pending.mask_area, None
        self._memory_revision += 1
        return True

    def _prepare_confirmed_promotion(self, snapshot, *, frame_id, rgb, depth, K):
        """Validate and allocate everything without modifying this tracker."""
        from quarantined_recovery import ConfirmedPointState
        try:
            if (type(snapshot) is not ConfirmedPointState or not self.initialized
                    or self._pending is not None or type(frame_id) is not int
                    or frame_id != snapshot.frame_id
                    or snapshot.source_identity is not self._memory_identity
                    or type(snapshot.source_memory_revision) is not int
                    or snapshot.source_memory_revision != self._memory_revision
                    or type(snapshot.mask_area) is not int or snapshot.mask_area < 1
                    or not isinstance(snapshot.seal, bytes)
                    or not hmac.compare_digest(snapshot.seal, self._seal_confirmed(snapshot))):
                return None
            gray, depth, K, _ = self._frame(rgb, depth, K, None)
            if (not np.array_equal(rgb, snapshot.rgb)
                    or not np.array_equal(gray, snapshot.gray)
                    or not np.array_equal(depth, snapshot.depth, equal_nan=True)
                    or not np.array_equal(K, snapshot.K)):
                return None
            pose = _pose(snapshot.pose)
            pixels, anchors = np.asarray(snapshot.pixels), np.asarray(snapshot.anchors)
            if (pose[2, 3] <= 0 or pixels.ndim != 2 or pixels.shape[1:] != (2,)
                    or anchors.shape != (len(pixels), 3)
                    or not self.config.min_points <= len(pixels) <= self.config.max_points
                    or not np.isfinite(pixels).all() or not np.isfinite(anchors).all()
                    or len(np.unique(pixels, axis=0)) != len(pixels)
                    or len(np.unique(anchors, axis=0)) != len(anchors)
                    or not _geometry(anchors, self.config)[0]):
                return None
            observed, valid = self._lift(pixels, depth, K)
            if not valid.all() or not _geometry(observed, self.config)[0]:
                return None
            # Use the existing commit contract on a disposable local tracker.
            # No caller/trusted state is changed if any staging check fails.
            candidate = RGBDSeedProposal(pose.copy(), True, {}, 1)
            staged = RGBDSeedTracker(self.config)
            staged._pending = _Pending(candidate, gray.copy(), pixels.copy(), anchors.copy(),
                                       observed.copy(), K.copy(), snapshot.mask_area)
            if not staged.commit(candidate, verified_pose=pose):
                return None
            # Commit may prune inconsistent pairs; require the confirmed chain
            # itself to remain intact instead of silently replacing its proof.
            if len(staged._pixels) != len(pixels):
                return None
            return (staged._gray, staged._pose, staged._pixels, staged._anchors,
                    snapshot.mask_area, self._generation + 1, self._memory_revision + 1)
        except Exception:
            return None

    def _seal_confirmed(self, snapshot):
        """Bind an in-process confirmation's exact contents against later edits.

        This is an internal consistency contract, not protection against code
        with arbitrary access to this process or the tracker's private state.
        """
        digest = hmac.new(self._snapshot_key, digestmod=hashlib.sha256)
        metadata = (snapshot.frame_id, snapshot.chain_id, snapshot.mask_area,
                    snapshot.source_memory_revision, id(snapshot.source_identity))
        digest.update(repr(metadata).encode('ascii'))
        for name in ('pose', 'rgb', 'gray', 'pixels', 'anchors', 'K', 'depth'):
            array = np.asarray(getattr(snapshot, name))
            if array.dtype.kind not in 'buif':
                raise ValueError('confirmation arrays must be numeric')
            digest.update(repr((name, array.shape, array.dtype.str)).encode('ascii'))
            digest.update(array.tobytes(order='C'))
        return digest.digest()

    def _install_confirmed_promotion(self, staged):
        (self._gray, self._pose, self._pixels, self._anchors, self._mask_area,
         self._generation, self._memory_revision) = staged
        self._pending = None

    def promote_confirmed(self, snapshot, *, frame_id, rgb, depth, K):
        """Atomically adopt a current confirmed chain, preserving registration ORB.

        Promotion requires the same source tracker/revision, the exact final
        RGB-D/K/frame, and no unresolved source proposal. Every failure leaves
        memory unchanged. Success advances the revision, preventing replay.
        This is explicit memory admission, never an instance identity claim.
        """
        staged = self._prepare_confirmed_promotion(snapshot, frame_id=frame_id, rgb=rgb, depth=depth, K=K)
        if staged is None:
            return False
        self._install_confirmed_promotion(staged)
        return True

    def reject(self, proposal):
        """Discard a proposal; stale rejects are harmless and cannot cancel a new one."""
        if self._pending is not None and self._pending.proposal is proposal:
            self._pending = None
            return True
        return False
