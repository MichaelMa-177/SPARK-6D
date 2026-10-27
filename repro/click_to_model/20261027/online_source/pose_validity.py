"""Observation support and ambiguity diagnostics for centered metric CAD poses.

This module never certifies instance identity or ground-truth pose correctness.
Depth consistency, local differential observability, texture, and declared global
symmetries are separate signals. A sparse z-buffer approximates visibility; it
does not implement rendered BOP VSD. See docs/CLICK_TO_MODEL_VALIDITY.md.
"""

from dataclasses import asdict, dataclass
from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation


def _pose(value, positive_depth=False):
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3),
                               atol=2e-3, rtol=0)
            or abs(np.linalg.det(value[:3, :3]) - 1) > 2e-3
            or (positive_depth and value[2, 3] <= 1e-4)):
        raise ValueError("pose must be finite SE(3), with positive camera depth when evaluated")
    return value


@dataclass(frozen=True)
class PoseValidityConfig:
    max_points: int = 1024
    min_inliers: int = 12
    depth_tolerance_m: float = 0.008
    depth_tolerance_diameter: float = 0.035
    min_depth_support_fraction: float = 0.18
    max_contradiction_fraction: float = 0.30
    min_mask_support_fraction: float = 0.50
    min_occluded_fraction: float = 0.50
    texture_gradient_threshold: float = 0.025
    min_texture_samples: int = 12
    rank_relative_threshold: float = 1e-3
    symmetry_max_step_diameter: float = 0.01
    max_symmetry_samples: int = 4096

    def __post_init__(self):
        for name in ("max_points", "min_inliers", "min_texture_samples", "max_symmetry_samples"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("depth_tolerance_m", "depth_tolerance_diameter",
                     "texture_gradient_threshold", "symmetry_max_step_diameter"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("min_depth_support_fraction", "max_contradiction_fraction",
                     "min_mask_support_fraction", "min_occluded_fraction", "rank_relative_threshold"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.rank_relative_threshold <= 0:
            raise ValueError("rank_relative_threshold must be positive")


class SymmetrySpec:
    """Explicit observational symmetries, expressed in centered model metres.

    ``discrete`` contains 4x4 rigid transforms (identity is implicit).
    ``continuous`` contains zero or one dict with ``axis`` and ``offset``.
    Multiple continuous axes need a full SO(3) group model, so are rejected.
    A textured cylinder is not necessarily observationally axis-symmetric.
    """

    def __init__(self, discrete=None, continuous=None):
        discrete_values = [] if discrete is None else list(discrete)
        transforms = [np.eye(4)]
        for item in discrete_values:
            transform = _pose(item).copy()
            if not any(np.allclose(transform, previous, atol=1e-8, rtol=0)
                       for previous in transforms):
                transforms.append(transform)
        continuous_values = [] if continuous is None else list(continuous)
        if len(continuous_values) > 1:
            raise ValueError("at most one continuous axis is supported; do not approximate SO(3) by an axis union")
        axes = []
        for item in continuous_values:
            if not isinstance(item, Mapping) or set(item) != {"axis", "offset"}:
                raise ValueError("continuous symmetry requires exactly axis and offset")
            axis, offset = np.asarray(item["axis"], float), np.asarray(item["offset"], float)
            if (axis.shape != (3,) or offset.shape != (3,)
                    or not np.isfinite(axis).all() or not np.isfinite(offset).all()
                    or np.linalg.norm(axis) < 1e-12):
                raise ValueError("continuous axis and offset must be finite 3-vectors; axis must be nonzero")
            axis = axis / np.linalg.norm(axis)
            axis.setflags(write=False)
            offset = offset.copy()
            offset.setflags(write=False)
            axes.append((axis, offset))
        for transform in transforms:
            transform.setflags(write=False)
        self.discrete = tuple(transforms)
        self.continuous = tuple(axes)

    @classmethod
    def from_bop(cls, model_info, model_center=(0., 0., 0.), unit_scale=0.001):
        """Convert BOP original CAD coordinates to centered model metres.

        ``model_center`` is in the SAME units as the BOP model (usually mm).
        If x_centered = scale * (x_original - center), a symmetry translation
        becomes scale * (R @ center + t - center).
        """
        if not isinstance(model_info, Mapping):
            raise ValueError("model_info must be a mapping")
        center = np.asarray(model_center, float)
        if (center.shape != (3,) or not np.isfinite(center).all()
                or isinstance(unit_scale, (bool, np.bool_))
                or not np.isfinite(unit_scale) or unit_scale <= 0):
            raise ValueError("model_center and positive unit_scale must be finite")
        discrete = []
        for item in model_info.get("symmetries_discrete", []):
            transform = _pose(np.asarray(item, float).reshape(4, 4)).copy()
            transform[:3, 3] = unit_scale * (transform[:3, :3] @ center + transform[:3, 3] - center)
            discrete.append(transform)
        continuous = [{"axis": item["axis"],
                       "offset": unit_scale * (np.asarray(item["offset"], float) - center)}
                      for item in model_info.get("symmetries_continuous", [])]
        return cls(discrete, continuous)

    @property
    def kind(self):
        if self.continuous:
            return "continuous" if len(self.discrete) == 1 else "discrete_and_continuous"
        return "discrete" if len(self.discrete) > 1 else "none"


class PoseValidityChecker:
    """Stateless CPU diagnostics; calling evaluate never updates any memory."""

    def __init__(self, model_points, model_normals, diameter, config=None, symmetries=None):
        self.config = PoseValidityConfig() if config is None else config
        if not isinstance(self.config, PoseValidityConfig):
            raise TypeError("config must be PoseValidityConfig")
        self.symmetries = SymmetrySpec() if symmetries is None else symmetries
        if not isinstance(self.symmetries, SymmetrySpec):
            raise TypeError("symmetries must be SymmetrySpec")
        points = np.asarray(model_points, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
                or not np.isfinite(points).all()):
            raise ValueError("model_points must be finite Nx3 with at least three points")
        if (isinstance(diameter, (bool, np.bool_)) or not np.isfinite(diameter) or diameter <= 0):
            raise ValueError("diameter must be positive metric length")
        normals = None
        if model_normals is not None:
            normals = np.asarray(model_normals, dtype=np.float64)
            if normals.shape != points.shape or not np.isfinite(normals).all():
                raise ValueError("model_normals must be finite and match model_points")
            length = np.linalg.norm(normals, axis=1)
            if np.any(length < 1e-12):
                raise ValueError("zero normals are not valid geometry constraints")
            normals = normals / length[:, None]
        ids = np.linspace(0, len(points) - 1, min(len(points), self.config.max_points), dtype=int)
        self.points = points[ids].copy()
        self.normals = None if normals is None else normals[ids].copy()
        self.diameter = float(diameter)
        self.tolerance = max(self.config.depth_tolerance_m,
                             self.config.depth_tolerance_diameter * self.diameter)
        self._symmetry_steps = 1
        self._symmetry_radius = 0.0
        if self.symmetries.continuous:
            axis, offset = self.symmetries.continuous[0]
            relative = points - offset
            self._symmetry_radius = float(np.linalg.norm(
                relative - (relative @ axis)[:, None] * axis, axis=1).max())
            self._symmetry_steps = max(4, int(np.ceil(
                2 * np.pi * self._symmetry_radius /
                (self.config.symmetry_max_step_diameter * self.diameter))))
        if self._symmetry_steps * len(self.symmetries.discrete) > self.config.max_symmetry_samples:
            raise ValueError("symmetry discretization exceeds max_symmetry_samples; adjust explicit bound")

    def configuration(self):
        return asdict(self.config)

    def _project(self, pose, K, shape):
        camera = self.points @ pose[:3, :3].T + pose[:3, 3]
        valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-4)
        if self.normals is not None:
            normals = self.normals @ pose[:3, :3].T
            valid &= np.sum(normals * camera, axis=1) < 0
        ids = np.flatnonzero(valid)
        projection = camera[ids] @ K.T
        uv = projection[:, :2] / projection[:, 2:3]
        height, width = shape
        inside = ((uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5)
                  & (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5))
        ids, uv = ids[inside], np.rint(uv[inside]).astype(np.int64)
        z = camera[ids, 2]
        linear = uv[:, 1] * width + uv[:, 0]
        order = np.lexsort((z, linear))
        unique = np.r_[True, linear[order][1:] != linear[order][:-1]] if len(order) else []
        keep = order[unique]
        return uv[keep], z[keep], ids[keep]

    def project_visible(self, centered_pose, K, shape):
        """Return (integer uv, metric z, sampled-model ids), using sparse z-buffer.

        This method projects the hypothesized surface only; it has not checked
        current depth, foreground membership, or instance identity. ``ids`` index
        ``checker.points``, not the caller's unsampled mesh. Invalid arguments
        raise ValueError, unlike evaluate's fail-closed report interface.
        """
        pose = _pose(centered_pose, positive_depth=True)
        K = np.asarray(K, float)
        if (len(shape) != 2 or any(type(x) not in (int, np.int32, np.int64) or x < 1 for x in shape)
                or K.shape != (3, 3) or not np.isfinite(K).all()
                or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0, 0, 1], atol=1e-7, rtol=0)
                or abs(np.linalg.det(K)) < 1e-12):
            raise ValueError("project_visible needs calibrated intrinsics and positive (height, width)")
        return self._project(pose, K, shape)

    def _empty_report(self):
        return {"status": "invalid_input", "reason": "invalid_input", "observation_valid": False,
                "identity_verified": False, "identity_status": "unknown",
                "pose_fully_observable": False, "rotation_ambiguous": True,
                "symmetry_type": self.symmetries.kind, "projected_count": 0,
                "inlier_count": 0, "depth_support_fraction": 0.0, "coverage": 0.0,
                "valid_depth_fraction": 0.0, "missing_depth_fraction": 0.0,
                "occluded_fraction": 0.0, "contradiction_fraction": 0.0,
                "mask_support_fraction": None, "texture_level": "unknown",
                "texture_score": 0.0, "texture_samples": 0, "texture_gradient_rms": 0.0,
                "geometry_rank": None, "rotation_observable_rank": None,
                "geometry_eigenvalues": [], "depth_residual_median_m": None,
                "visibility_method": "sparse_sample_zbuffer", "symmetry_distance": None,
                "observability_method": "local_point_to_plane_rank_heuristic",
                "local_differential_full_rank": False}

    def _texture(self, rgb, depth, mask, uv, expected_z):
        """Measure directional image information in depth-consistent 3x3 patches.

        Gradients alone do not establish repeatable matches or resolve symmetry.
        Patches that straddle missing depth, occlusion or mask boundaries are
        excluded, preventing background texture from determining object policy.
        """
        height, width = depth.shape
        inside = ((uv[:, 0] >= 1) & (uv[:, 0] < width - 1)
                  & (uv[:, 1] >= 1) & (uv[:, 1] < height - 1))
        uv, expected_z = uv[inside], expected_z[inside]
        if not len(uv):
            return {"texture_level": "unknown", "texture_score": 0.0,
                    "texture_samples": 0, "texture_gradient_rms": 0.0}
        valid = np.ones(len(uv), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                measured = depth[uv[:, 1] + dy, uv[:, 0] + dx]
                valid &= (np.isfinite(measured) & (measured > 1e-4)
                          & (np.abs(measured - expected_z) <= self.tolerance))
                if mask is not None:
                    valid &= mask[uv[:, 1] + dy, uv[:, 0] + dx]
        uv = uv[valid]
        if len(uv) < self.config.min_texture_samples:
            return {"texture_level": "unknown", "texture_score": 0.0,
                    "texture_samples": int(len(uv)), "texture_gradient_rms": 0.0}
        # Forward/backward differences avoid the Nyquist null of centered
        # differences (an alternating checkerboard must not look textureless).
        # RGB channels preserve isoluminant texture that grayscale would miss.
        x, y = uv.T
        c = rgb[y, x].astype(float) / 255.
        xp, xm = rgb[y, x + 1].astype(float) / 255., rgb[y, x - 1].astype(float) / 255.
        yp, ym = rgb[y + 1, x].astype(float) / 255., rgb[y - 1, x].astype(float) / 255.
        gradients = np.concatenate([np.stack((xp - c, yp - c), axis=-1),
                                    np.stack((c - xm, c - ym), axis=-1)], axis=0).reshape(-1, 2)
        eigenvalues = np.linalg.eigvalsh(gradients.T @ gradients / len(gradients))
        score = float(np.sqrt(max(eigenvalues[0], 0)))
        energy = float(np.sqrt(max(eigenvalues.sum(), 0)))
        level = ("textured" if score >= self.config.texture_gradient_threshold else
                 "directional" if energy >= self.config.texture_gradient_threshold else "textureless")
        return {"texture_level": level, "texture_score": score,
                "texture_samples": int(len(uv)), "texture_gradient_rms": energy}

    def _observability(self, pose, ids):
        if self.normals is None or len(ids) < self.config.min_inliers:
            return {"geometry_rank": None, "rotation_observable_rank": None, "geometry_eigenvalues": []}
        normals = self.normals[ids] @ pose[:3, :3].T
        relative = self.points[ids] @ pose[:3, :3].T / self.diameter
        jacobian = np.c_[np.cross(relative, normals), normals]
        hessian = jacobian.T @ jacobian / len(ids)
        eigenvalues = np.maximum(np.linalg.eigvalsh(hessian), 0)
        threshold = max(float(eigenvalues[-1]) * self.config.rank_relative_threshold, 1e-12)
        # Schur complement eliminates translation nuisance parameters. Merely
        # checking the upper-left block would overstate rotational constraints.
        rotational = (hessian[:3, :3] - hessian[:3, 3:] @
                      np.linalg.pinv(hessian[3:, 3:], rcond=self.config.rank_relative_threshold) @
                      hessian[3:, :3])
        rotation_rank = int(np.count_nonzero(np.linalg.eigvalsh(rotational) > threshold))
        if self.symmetries.continuous:
            rotation_rank = min(rotation_rank, 2)
        return {"geometry_rank": int(np.count_nonzero(eigenvalues > threshold)),
                "rotation_observable_rank": rotation_rank,
                "geometry_eigenvalues": eigenvalues.tolist()}

    def evaluate(self, centered_pose, rgb, depth, K, mask=None, reference_pose=None):
        """Return JSON-safe diagnostics, failing closed on malformed frame data.

        uint8 RGB, metric camera-Z depth, finite calibrated K, optional binary
        foreground mask. Missing depth (0/NaN/Inf) is unknown, not occlusion.
        Every fraction uses projected unique surface samples as denominator;
        mask_support_fraction uses depth-supported samples only.
        """
        report = self._empty_report()
        try:
            pose = _pose(centered_pose, positive_depth=True)
            rgb, depth, K = np.asarray(rgb), np.asarray(depth), np.asarray(K, float)
            if (depth.ndim != 2 or min(depth.shape, default=0) < 1
                    or rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8
                    or depth.dtype.kind not in "fiu" or K.shape != (3, 3)
                    or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                    or not np.allclose(K[2], [0, 0, 1], atol=1e-7, rtol=0)
                    or abs(np.linalg.det(K)) < 1e-12):
                raise ValueError("invalid RGB-D frame or camera intrinsics")
            if mask is not None:
                mask = np.asarray(mask)
                if (mask.shape != depth.shape or mask.dtype.kind not in "buif"
                        or not np.isfinite(mask).all()
                        or not np.isin(mask, [0, 1, 255]).all()):
                    raise ValueError("mask must be finite binary HxW")
                mask = mask != 0
            if reference_pose is not None:
                _pose(reference_pose, positive_depth=True)
        except (TypeError, ValueError, OverflowError) as error:
            report["reason"] = str(error)
            return report
        uv, expected_z, ids = self._project(pose, K, depth.shape)
        count = len(ids)
        if count == 0:
            report.update(status="out_of_view", reason="no_front_facing_projected_samples")
            return report
        measured = depth[uv[:, 1], uv[:, 0]].astype(float)
        valid = np.isfinite(measured) & (measured > 1e-4)
        residual = measured - expected_z
        inlier = valid & (np.abs(residual) <= self.tolerance)
        occluded = valid & (residual < -self.tolerance)
        contradicted = valid & (residual > self.tolerance)
        inside_mask = np.ones(count, bool) if mask is None else mask[uv[:, 1], uv[:, 0]]
        mask_fraction = None if mask is None else float(np.count_nonzero(inside_mask & inlier) / max(1, inlier.sum()))
        report.update(projected_count=count, inlier_count=int(inlier.sum()),
                      depth_support_fraction=float(inlier.mean()), coverage=float(inlier.mean()),
                      valid_depth_fraction=float(valid.mean()), missing_depth_fraction=float((~valid).mean()),
                      occluded_fraction=float(occluded.mean()), contradiction_fraction=float(contradicted.mean()),
                      mask_support_fraction=mask_fraction,
                      depth_residual_median_m=float(np.median(np.abs(residual[valid]))) if valid.any() else None)
        enough = (inlier.sum() >= self.config.min_inliers
                  and inlier.mean() >= self.config.min_depth_support_fraction
                  and contradicted.mean() <= self.config.max_contradiction_fraction
                  and inlier.sum() > contradicted.sum()
                  and (mask_fraction is None or mask_fraction >= self.config.min_mask_support_fraction))
        report["observation_valid"] = bool(enough)
        if enough:
            report.update(status="supported", reason="sparse_geometry_supported_identity_unverified")
        elif mask_fraction is not None and mask_fraction < self.config.min_mask_support_fraction and inlier.any():
            report.update(status="mask_mismatch", reason="depth_support_outside_tracked_foreground")
        elif occluded.mean() >= self.config.min_occluded_fraction:
            report.update(status="occluded", reason="measured_surface_in_front_of_model")
        elif contradicted.any():
            report.update(status="contradicted", reason="observed_free_space_behind_predicted_surface")
        else:
            report.update(status="insufficient_depth", reason="insufficient_depth_is_not_occlusion")
        supported = inlier & inside_mask
        report.update(self._texture(rgb, depth, mask, uv[supported], expected_z[supported]))
        report.update(self._observability(pose, ids[supported]))
        report["rotation_ambiguous"] = bool(not enough or self.symmetries.kind != "none"
                                            or report["rotation_observable_rank"] != 3)
        report["pose_fully_observable"] = bool(enough and self.symmetries.kind == "none"
                                               and report["geometry_rank"] == 6)
        # This is conditional LOCAL differential observability. It does not
        # rule out another globally indistinguishable pose or another instance.
        report["local_differential_full_rank"] = report["pose_fully_observable"]
        if reference_pose is not None:
            report["symmetry_distance"] = self.symmetry_distance(pose, reference_pose)
        return report

    def _axis_transform(self, angle):
        axis, offset = self.symmetries.continuous[0]
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        transform[:3, 3] = offset - transform[:3, :3] @ offset
        return transform

    def _closest_equivalent(self, movable, target):
        movable, target = _pose(movable), _pose(target)
        target_points = self.points @ target[:3, :3].T + target[:3, 3]

        def error(symmetry):
            candidate = movable @ symmetry
            points = self.points @ candidate[:3, :3].T + candidate[:3, 3]
            return float(np.linalg.norm(points - target_points, axis=1).max())

        best_error, best_transform = float("inf"), None
        for discrete in self.symmetries.discrete:
            if not self.symmetries.continuous:
                candidates = [discrete]
            else:
                angles = np.arange(self._symmetry_steps) * (2 * np.pi / self._symmetry_steps)
                errors = [error(self._axis_transform(angle) @ discrete) for angle in angles]
                index = int(np.argmin(errors))
                center, step = angles[index], 2 * np.pi / self._symmetry_steps
                # Refinement improves sub-grid equivalent rotations while the
                # sampled grid provides an explicit global discretization bound.
                result = minimize_scalar(lambda angle: error(self._axis_transform(angle) @ discrete),
                                         bounds=(center - step, center + step), method="bounded",
                                         options={"xatol": 1e-12})
                candidates = [self._axis_transform(center) @ discrete]
                if result.success and np.isfinite(result.fun):
                    candidates.append(self._axis_transform(float(result.x)) @ discrete)
            for transform in candidates:
                value = error(transform)
                if value < best_error:
                    best_error, best_transform = value, transform
        return movable @ best_transform, best_error

    def symmetry_distance(self, pose, reference_pose):
        """BOP-style min_S max_x ||pose*x - reference*S*x|| on sampled CAD.

        This is a sample-set MSSD diagnostic, not the official full-model BOP
        metric. The continuous grid bound covers angular discretization only;
        it does not cover surface subsampling or incorrect symmetry annotations.
        """
        pose, reference_pose = _pose(pose), _pose(reference_pose)
        equivalent, distance = self._closest_equivalent(reference_pose, pose)
        angle = Rotation.from_matrix(pose[:3, :3] @ equivalent[:3, :3].T).magnitude()
        bound = (2 * self._symmetry_radius * np.sin(np.pi / (2 * self._symmetry_steps))
                 if self.symmetries.continuous else 0.0)
        return {"sample_mssd_m": distance, "rotation_radians": float(angle),
                "translation_m": float(np.linalg.norm(pose[:3, 3] - equivalent[:3, 3])),
                "continuous_discretization_bound_m": float(bound),
                "model_sample_count": int(len(self.points)),
                "symmetry_type": self.symmetries.kind}

    def equivalent_pose(self, pose, reference_pose):
        """Choose a declared symmetry representative nearest a temporal reference.

        This changes the coordinate representative, not the observed object
        geometry. Keep raw neural poses separately in experimental records.
        """
        equivalent, _ = self._closest_equivalent(pose, reference_pose)
        return equivalent.copy()
