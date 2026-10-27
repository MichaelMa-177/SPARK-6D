"""Opt-in depth-ray seed candidates after FEAR xy alignment; no output replacement.

The NEW max_depth_increment_diameter bounds candidate translation minus the
xy-aligned seed, NOT total translation from the original continuation pose.
FEAR V1's xy correction remains unchanged and may exceed that increment bound.
No reference trajectory, GT, category, frame ID, or future data is read.

Current depth residual modes only propose input seeds. They do not identify
objects: nearby occluders and same-depth distractors can remain ambiguous. Every
frame still needs the caller's complete neural 6D refinement and raw output;
the inherited strict verification and fixed-template commit policy are unchanged.

The default bounded_modes_v1 retains its historical coupled search/cap behavior.
Experimental median_all and modes_all use the identical full valid residual pool;
only their final xyz increment is capped. They are seed rules, not extra refiners.
"""

from dataclasses import dataclass
import copy
from numbers import Real

import numpy as np

from fear_pose import FEARPoseConfig, FEARPoseController
from robust_pose import _cap_vector, _valid_pose


@dataclass(frozen=True)
class FEARDepthPoseConfig(FEARPoseConfig):
    max_depth_increment_diameter: float = 0.20
    max_depth_modes: int = 3
    depth_seed_policy: str = "bounded_modes_v1"

    def __post_init__(self):
        super().__post_init__()
        if (isinstance(self.max_depth_increment_diameter, (bool, np.bool_))
                or not isinstance(self.max_depth_increment_diameter, Real)
                or not np.isfinite(self.max_depth_increment_diameter)
                or self.max_depth_increment_diameter <= 0):
            raise ValueError("max_depth_increment_diameter must be finite and positive")
        if type(self.max_depth_modes) is not int or not 1 <= self.max_depth_modes <= 3:
            raise ValueError("max_depth_modes must be an integer between 1 and 3")
        if (type(self.depth_seed_policy) is not str
                or self.depth_seed_policy not in ("bounded_modes_v1", "median_all", "modes_all")):
            raise ValueError("depth_seed_policy must be bounded_modes_v1, median_all, or modes_all")


def depth_residual_modes(residuals, bound, tolerance, min_points, max_modes=3):
    """Deterministic dense windows; widths come only from the existing tolerance."""
    for name, value in (("bound", bound), ("tolerance", tolerance)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if type(min_points) is not int or min_points < 1:
        raise ValueError("min_points must be a positive integer")
    if type(max_modes) is not int or not 1 <= max_modes <= 3:
        raise ValueError("max_modes must be an integer between 1 and 3")
    remaining = np.asarray(residuals, np.float64)
    if remaining.ndim != 1:
        raise ValueError("residuals must be a one-dimensional numeric array")
    remaining = remaining[np.isfinite(remaining) & (np.abs(remaining) <= bound)].copy()
    centers = np.arange(-bound, bound + tolerance / 4, tolerance / 2)
    modes = []
    for _ in range(max_modes):
        if len(remaining) < min_points or not len(centers):
            break
        counts = np.count_nonzero(np.abs(remaining[:, None] - centers[None]) <= tolerance, axis=0)
        best = int(np.argmax(counts))
        if counts[best] < min_points:
            break
        support = remaining[np.abs(remaining - centers[best]) <= tolerance]
        dz = float(np.median(support))
        modes.append({"dz_m": dz, "point_count": int(len(support))})
        remaining = remaining[np.abs(remaining - dz) > 2 * tolerance]
    return modes


def _all_finite_residuals(residuals, min_points):
    if type(min_points) is not int or min_points < 1:
        raise ValueError("min_points must be a positive integer")
    values = np.asarray(residuals)
    if values.ndim != 1 or values.dtype.kind not in "fiu":
        raise ValueError("residuals must be a one-dimensional real numeric array")
    source_finite = np.isfinite(values)
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float64)
    if np.any(source_finite & ~np.isfinite(values)):
        raise ValueError("finite residual outside float64 range; never silently remove it from the shared pool")
    return values[np.isfinite(values)].copy()


def _finite_median(values):
    # Ordinary values follow np.median exactly. Splitting only an overflowing
    # middle-pair sum avoids inventing inf from finite extreme observations.
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.median(values))
    if not np.isfinite(result):
        ordered = np.sort(values)
        middle = len(ordered) // 2
        result = float(ordered[middle - 1] / 2 + ordered[middle] / 2)
    return result


def depth_residual_median_all(residuals, min_points):
    """Exactly one median of ALL finite residuals; no cap, windows or mode count."""
    values = _all_finite_residuals(residuals, min_points)
    if len(values) < min_points:
        return []
    return [{"dz_m": _finite_median(values), "point_count": int(len(values))}]


def _sparse_best_window(values, tolerance):
    """At most five grid-bin events per point, never a dense residual-range grid.

    Ordinary indices use vectorized integer events. Extreme finite ratios use
    exact integer ratios instead of overflowing an int64 or allocating empty
    space. That rare path is still O(N) events, with deterministic lowest-bin ties.
    """
    step = tolerance / 2
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = values / step
    if np.isfinite(scaled).all() and np.max(np.abs(scaled)) <= 2**50:
        nearest = np.rint(scaled).astype(np.int64)
        bins = nearest[:, None] + np.array([-2, -1, 0, 1, 2], dtype=np.int64)
        with np.errstate(over="ignore", invalid="ignore"):
            centers = bins * step
        if np.isfinite(centers).all():
            with np.errstate(over="ignore", invalid="ignore"):
                support = np.abs(values[:, None] - centers) <= tolerance
            unique, counts = np.unique(bins[support], return_counts=True)
            best = int(unique[np.argmax(counts)])  # unique is sorted: lowest-center tie
            center = best * step
            with np.errstate(over="ignore", invalid="ignore"):
                selected = np.flatnonzero(np.abs(values - center) <= tolerance)
            return best, float(center), selected
    # Every float is an exact rational. Since tolerance == 2*step, a point
    # supports five centers iff exactly on-grid, otherwise four. No float-to-int
    # overflow, giant np.arange, or loss of finite outliers from the common pool.
    step_num, step_den = float(step).as_integer_ratio()
    events = {}
    for point_id, value in enumerate(values):
        value_num, value_den = float(value).as_integer_ratio()
        q, remainder = divmod(value_num * step_den, value_den * step_num)
        first = q - (2 if remainder == 0 else 1)
        for grid_id in range(first, q + 3):
            events.setdefault(grid_id, []).append(point_id)
    best = min(events, key=lambda grid_id: (-len(events[grid_id]), grid_id))
    # A theoretical grid center can exceed float64 although its window contains
    # finite observations. Preserve its exact index and use None, not JSON inf.
    from fractions import Fraction
    try:
        center = float(Fraction(best * step_num, step_den))
    except OverflowError:
        center = None
    return best, center, np.asarray(events[best], dtype=np.intp)


def depth_residual_modes_all(residuals, tolerance, min_points, max_modes=3):
    """Cap-independent, zero-anchored tau/2 windows over ALL valid residuals.

    Highest count wins, then lowest grid center. Propose its median, remove
    points within 2*tau of that median, repeat at most three times. Setting
    max_modes=1 remains one densest mode; it is NOT the whole-pool median rule.
    """
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance <= 0
            or tolerance / 2 == 0 or (tolerance / 2) * 2 != tolerance):
        raise ValueError("tolerance must be finite positive with exactly representable nonzero half-step")
    if type(max_modes) is not int or not 1 <= max_modes <= 3:
        raise ValueError("max_modes must be an integer between 1 and 3")
    remaining = _all_finite_residuals(residuals, min_points)
    modes = []
    for _ in range(max_modes):
        if len(remaining) < min_points:
            break
        grid_id, center, selected = _sparse_best_window(remaining, float(tolerance))
        if len(selected) < min_points:
            break
        dz = _finite_median(remaining[selected])
        modes.append({"dz_m": dz, "point_count": int(len(selected)),
                      "window_center_m": center, "window_grid_index": str(grid_id)})
        with np.errstate(over="ignore", invalid="ignore"):
            separation = np.abs(remaining - dz)
            exclusion = 2 * tolerance
        if np.isfinite(exclusion):
            remaining = remaining[separation > exclusion]
        else:
            from fractions import Fraction
            exact_dz, exact_width = Fraction(dz), 2 * Fraction(float(tolerance))
            remaining = remaining[[abs(Fraction(float(v)) - exact_dz) > exact_width for v in remaining]]
    return modes


class FEARDepthPoseController(FEARPoseController):
    """Existing FEAR policy plus bounded depth increments, no extra 6D inference."""

    def __init__(self, config=None, tracker=None):
        config = config if config is not None else FEARDepthPoseConfig()
        if not isinstance(config, FEARDepthPoseConfig):
            raise TypeError("config must be FEARDepthPoseConfig")
        super().__init__(config, tracker=tracker)
        self._depth_diagnostics = {}

    @property
    def proposal_model_provenance(self):
        result = super().proposal_model_provenance
        result["candidate_role"] = "xy_or_depth_ray_translation_seed_only_rotation_unchanged_full_current_neural_output_required"
        result["depth_seed_policy"] = {
            "max_depth_increment_diameter": self.config.max_depth_increment_diameter,
            "max_modes": self.config.max_depth_modes,
            "mode_halfwidth": "existing_depth_tolerance",
            "mode_grid_step": "existing_depth_tolerance_divided_by_2",
            "cap_reference": "xy_aligned_seed",
            "total_continuation_displacement_is_not_capped": True,
            "rotation_changed_by_proposal": False,
        }
        if self.config.depth_seed_policy != "bounded_modes_v1":
            result["depth_seed_policy"].update({
                "estimator": self.config.depth_seed_policy,
                "residual_pool": "all finite currently paired CAD-depth residuals; independent of cap",
                "residual_admission_uses_cap": False,
                "max_modes": 1 if self.config.depth_seed_policy == "median_all" else self.config.max_depth_modes,
                "mode_halfwidth": None if self.config.depth_seed_policy == "median_all" else "existing_depth_tolerance",
                "mode_grid_step": None if self.config.depth_seed_policy == "median_all" else "existing_depth_tolerance_divided_by_2",
                "mode_grid_anchor_m": None if self.config.depth_seed_policy == "median_all" else 0.0,
                "mode_counting": "not_used" if self.config.depth_seed_policy == "median_all" else "sparse, at most five bin events per residual",
            })
        return result

    def reset(self, *args, **kwargs):
        self._depth_diagnostics = {}
        return super().reset(*args, **kwargs)

    def _make_depth_candidate(self, xy_seed, dz, box, K, shape, bound, diagnostics=None):
        new_z = xy_seed[2, 3] + dz
        if not np.isfinite(new_z) or new_z <= 1e-4:
            return None, "nonpositive_or_invalid_candidate_z"
        candidate = xy_seed.copy()
        candidate[:3, 3] *= new_z / xy_seed[2, 3]
        current_box = self._visible_bbox(candidate, K, shape)
        if current_box is None:
            return None, "no_nondegenerate_candidate_projection"
        current_box = np.asarray(current_box, np.float64)
        delta_pixels = box[:2] + box[2:] / 2 - current_box[:2] - current_box[2:] / 2
        candidate[:2, 3] += new_z * np.linalg.solve(np.asarray(K)[:2, :2], delta_pixels)
        increment = candidate[:3, 3] - xy_seed[:3, 3]
        if diagnostics is not None:
            diagnostics["uncapped_increment_xyz_m"] = increment.tolist()
        candidate[:3, 3] = xy_seed[:3, 3] + _cap_vector(increment, bound)
        if diagnostics is not None:
            diagnostics["capped_increment_xyz_m"] = (candidate[:3, 3] - xy_seed[:3, 3]).tolist()
        if not _valid_pose(candidate):
            return None, "invalid_rigid_candidate"
        return candidate, None

    def prepare(self, rgb, depth, K, timestamp=None):
        if not self._ready or self._pending:
            # Let the existing lifecycle guard raise WITHOUT rejecting an
            # earlier valid pending observation on an accidental double call.
            return super().prepare(rgb, depth, K, timestamp=timestamp)
        snapshot = self.__dict__.copy()
        try:
            chosen = super().prepare(rgb, depth, K, timestamp=timestamp)
            reference_evidence = dict(self.diagnostics)
            original_seed = self.working_pose  # FEAR requires continuous/no prediction
            initial_score = self.diagnostics["support_score"]
            initial_source = self._pending_source
            bound = self._diameter * self.config.max_depth_increment_diameter
            details = {
                "cap_reference": "xy_aligned_seed",
                "max_depth_increment_m": float(bound),
                "total_continuation_displacement_is_not_capped": True,
                "mode_halfwidth_m": float(self._depth_tolerance),
                "base_xy_selected_source": initial_source,
                "base_xy_selected_support_score": float(initial_score),
                "residual_valid_points": 0, "residual_in_bound_points": 0,
                "modes": [], "selected": False, "selected_mode": None,
                "selection_reason": "no_nondegenerate_xy_candidate",
                "depth_ray_increment_m": None,
                "total_from_continuation_m": float(np.linalg.norm(chosen[:3, 3] - original_seed[:3, 3])),
            }
            if self._seed_admission_enabled:
                details["seed_admission"] = {**self._seed_admission_policy(),
                    "reference_source": initial_source,
                    "reference_projected_count": reference_evidence["projected_count"],
                    "reference_support_score": initial_score, "selected_by_seed_policy": False}
            experimental = self.config.depth_seed_policy != "bounded_modes_v1"
            if experimental:
                details.update({"depth_seed_policy": self.config.depth_seed_policy,
                                "residual_admission_uses_cap": False, "residual_proposal_pool_points": 0,
                                "mode_grid_anchor_m": 0.0 if self.config.depth_seed_policy == "modes_all" else None,
                                "max_raw_proposals": 1 if self.config.depth_seed_policy == "median_all" else self.config.max_depth_modes})
            xy_translation = self._fear_diagnostics.get("candidate_translation_xy_m")
            if xy_translation is not None:
                xy_seed = original_seed.copy()
                xy_seed[:2, 3] = xy_translation
                box = np.asarray(self._fear_proposal.bbox, np.float64)
                uv, expected_z, _ = self._project_visible(xy_seed, K, depth.shape)
                measured = depth[uv[:, 1], uv[:, 0]]
                valid = np.isfinite(measured) & (measured > 1e-4)
                residuals = measured[valid] - expected_z[valid]
                details["residual_valid_points"] = int(valid.sum())
                details["residual_in_bound_points"] = int(np.count_nonzero(np.abs(residuals) <= bound))
                if self.config.depth_seed_policy == "bounded_modes_v1":
                    modes = depth_residual_modes(residuals, bound, self._depth_tolerance,
                                                  self.config.min_inliers, self.config.max_depth_modes)
                elif self.config.depth_seed_policy == "median_all":
                    modes = depth_residual_median_all(residuals, self.config.min_inliers)
                else:
                    modes = depth_residual_modes_all(residuals, self._depth_tolerance,
                                                      self.config.min_inliers, self.config.max_depth_modes)
                if experimental:
                    details["residual_proposal_pool_points"] = int(np.isfinite(residuals).sum())
                details["selection_reason"] = (("no_all_residual_proposal" if experimental else "no_bounded_residual_mode")
                                               if not modes else "no_supported_improving_mode")
                best_pose, best_evidence, best_index = None, None, None
                for index, mode in enumerate(modes):
                    geometry = {} if experimental else None
                    candidate, invalid_reason = self._make_depth_candidate(
                        xy_seed, mode["dz_m"], box, K, depth.shape, bound, diagnostics=geometry)
                    item = {**mode, "valid": candidate is not None, "reason": invalid_reason,
                            "supported": False, "improves_existing_seed": False}
                    if self._seed_admission_enabled:
                        item["seed_admission"] = {**self._seed_admission(None, reference_evidence),
                                                  "selected_by_seed_policy": False}
                    if geometry is not None:
                        item.update(geometry)
                    if candidate is not None:
                        evidence = self._evidence(candidate, depth, K)
                        supported = self._supported(evidence)
                        improves = supported and evidence["support_score"] > initial_score + self.config.seed_improvement_margin
                        admission = self._seed_admission(evidence, reference_evidence) if self._seed_admission_enabled else None
                        item.update({
                            "evidence": evidence, "supported": bool(supported),
                            "improves_existing_seed": bool(improves),
                            "candidate_translation_xyz_m": candidate[:3, 3].tolist(),
                            "depth_ray_increment_m": float(np.linalg.norm(candidate[:3, 3] - xy_seed[:3, 3])),
                            "total_from_continuation_m": float(np.linalg.norm(candidate[:3, 3] - original_seed[:3, 3])),
                        })
                        if self._seed_admission_enabled:
                            item["seed_admission"] = {**admission, "selected_by_seed_policy": False}
                        seed_admitted = admission["admitted_to_seed_comparison"] if admission is not None else bool(improves)
                        if seed_admitted and (best_evidence is None or evidence["support_score"] > best_evidence["support_score"]):
                            best_pose, best_evidence, best_index = candidate, evidence, index
                    details["modes"].append(item)
                if best_pose is not None:
                    chosen = best_pose
                    self._pending_seed = chosen.copy()
                    self._pending_source = "fear_depth_ray"
                    self.diagnostics.update(best_evidence)
                    self.diagnostics["source"] = self._pending_source
                    details.update({
                        "selected": True, "selected_mode": best_index,
                        "selection_reason": ("current_depth_support_improved" if details["modes"][best_index]["improves_existing_seed"]
                                             else "experimental_seed_admission_full_neural_required"),
                        "depth_ray_increment_m": details["modes"][best_index]["depth_ray_increment_m"],
                        "total_from_continuation_m": details["modes"][best_index]["total_from_continuation_m"],
                    })
                    if self._seed_admission_enabled:
                        details["seed_admission"]["selected_by_seed_policy"] = True
                        details["modes"][best_index]["seed_admission"]["selected_by_seed_policy"] = True
            self._depth_diagnostics = details
            self.diagnostics["fear_depth"] = copy.deepcopy(details)
            return chosen.copy()
        except Exception:
            inner_invalidated = not self._ready
            self.__dict__.update(snapshot)
            if getattr(self.tracker, "pending", None) is not None:
                try:
                    self.tracker.reject(self.tracker.pending)
                except Exception:
                    self._invalidate("fear_depth_prepare_rollback_failed_explicit_reset_required")
            if inner_invalidated:
                self._invalidate("fear_depth_nested_prepare_failure_explicit_reset_required")
            raise

    def observe(self, neural_pose_np, rgb, depth, K):
        result = super().observe(neural_pose_np, rgb, depth, K)
        result["fear_depth"] = copy.deepcopy(self._depth_diagnostics)
        self.diagnostics = result
        return copy.deepcopy(result)
