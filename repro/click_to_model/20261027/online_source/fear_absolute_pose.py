"""Opt-in absolute RGB-box/depth seed controls, not DynamicPose reproduction.

One existing FEAR proposal, one extra geometric candidate, unchanged strict
support/margin, and the caller's full current-frame neural output. No cap,
anchor, rotation search, output substitution, new model, or template update.
Depth/bbox agreement cannot certify identity, including during full occlusion.
See docs/CHEAP_ABSOLUTE_DEPTH_SEED_BASELINES.md for the pinned source and math.
"""

from dataclasses import dataclass
import copy
import hashlib

import numpy as np

from fear_pose import FEARPoseConfig, FEARPoseController
from robust_pose import _valid_pose


ABSOLUTE_SEED_POLICIES = (
    "center_pixel_abs", "bbox_median_abs", "bbox_median_surface_compensated_abs",
)
_SOURCE_COMMIT = "537992938bd9bee12cfd24b61b1d12d20834d632"
_SOURCE_HASHES = (
    ("object_pose_estimator/include/Utils/utils.py", "90de993767df4d31291da57c2616c37cb8cb1fb45c101a39bf686d13f9009659"),
    ("object_pose_estimator/include/Utils/foundationpose.py", "ad8c5134654d5d3b35d5dc82740650a55311c2558a92d1269e296dd9e0ba8e9c"),
    ("object_pose_estimator/include/FoundationPose/estimater.py", "2e1fa56ad2566ffdb0a0b8ccb09bb8af7567d3415ba7a31bfcbcd075917bad0e"),
    ("object_pose_estimator/scripts/pose_estimator.py", "27926dc821f6c4bd794922b59854e1ac06a16caffd16e1a2131440b27bd05570"),
)


@dataclass(frozen=True)
class FEARAbsolutePoseConfig(FEARPoseConfig):
    absolute_seed_policy: str = "bbox_median_surface_compensated_abs"

    def __post_init__(self):
        super().__post_init__()
        if (not isinstance(self.absolute_seed_policy, str)
                or self.absolute_seed_policy not in ABSOLUTE_SEED_POLICIES):
            raise ValueError("absolute_seed_policy must be one of " + repr(ABSOLUTE_SEED_POLICIES))


def _real_array(value, name, *, preserve_dtype=False):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(name + " must be a real numeric array, not bool/complex/string")
    return array if preserve_dtype else array.astype(np.float64, copy=False)


def _median(values):
    """Median on a private partition; half-sum avoids finite same-sign overflow."""
    n = len(values)
    middle = n // 2
    partitioned = np.partition(values, (middle - 1, middle) if n % 2 == 0 else middle)
    if n % 2:
        return float(partitioned[middle])
    return float(partitioned[middle - 1]) / 2 + float(partitioned[middle]) / 2


def absolute_depth_seed(centered_pose, depth_m, K, bbox_xywh, *,
                        policy="bbox_median_surface_compensated_abs",
                        centered_points=None, project_visible=None):
    """Return ``(new_centered_pose_or_None, diagnostics)`` without state writes.

    The three policies use the clipped integer rectangle's pixel centroid,
    not FEAR's continuous box center. Depth is camera-z metres. Compensation
    takes one median rotated-CAD offset at the provisional *surface* candidate.
    ``project_visible(pose, K, shape)`` is a read-only geometry callback returning
    pixel coordinates, camera z, and indices into ``centered_points``; the host
    passes its existing _project_visible and its already sampled _points. No
    previous-pose visibility, GT, trust, or tracker state is read here.

    Bad API inputs/callback contracts raise ValueError; unavailable depth or
    invalid geometric candidates return None with a reason. Callback exceptions
    propagate so a controller can roll back its external pending transaction.
    This function never invokes neural refinement or commits a proposal.
    """
    if not isinstance(policy, str) or policy not in ABSOLUTE_SEED_POLICIES:
        raise ValueError("unsupported absolute seed policy")
    pose = _real_array(centered_pose, "centered_pose")
    # Preserve the full input image's dtype; only sampled median values need
    # arithmetic. Avoid a redundant full-frame float64 allocation per prepare.
    depth = _real_array(depth_m, "depth_m", preserve_dtype=True)
    intrinsic = _real_array(K, "K")
    box = _real_array(bbox_xywh, "bbox_xywh")
    if not _valid_pose(pose):
        raise ValueError("centered_pose must be a finite valid rigid pose with positive z")
    if depth.ndim != 2 or min(depth.shape, default=0) <= 0:
        raise ValueError("depth_m must be a nonempty HxW array")
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-6)):
        raise ValueError("K must be calibrated finite 3x3 intrinsics")
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
        raise ValueError("bbox_xywh must be finite with positive size")
    points = None
    if policy == "bbox_median_surface_compensated_abs":
        points = _real_array(centered_points, "centered_points")
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
                or not np.isfinite(points).all() or not callable(project_visible)):
            raise ValueError("compensation requires finite centered Nx3 points and project_visible")

    with np.errstate(over="ignore", invalid="ignore"):
        continuous_center = box[:2] + box[2:] / 2
    details = {
        "policy": policy, "candidate_frame": "centered_CAD_to_camera",
        "mapping": "clipped_integer_rectangle_pixel_centroid_origin_ray",
        "depth_valid_min_m": 1e-4, "proposal_bbox_xywh": box.tolist(),
        "continuous_box_center_uv": (continuous_center.tolist() if np.isfinite(continuous_center).all() else None),
        "support_rectangle_xyxy_exclusive": None, "pixel_centroid_uv": None,
        "center_sample_uv": None, "valid_depth_count": 0,
        "measured_depth_statistic_m": None, "surface_offset_m": 0.0,
        "compensation_reference": ("provisional_absolute_surface_candidate" if points is not None else "none"),
        "compensation_reference_pose": None, "visible_cad_count": 0,
        "visible_cad_indices_sha256": None, "absolute_origin_z_m": None,
        "translation_capped": False, "cap_reference": None,
        "candidate_pose": None, "valid": False, "reason": None,
        "geometry_or_box_agreement_guarantees_identity": False,
    }

    def unavailable(reason):
        details["reason"] = reason
        return None, details

    height, width = depth.shape
    with np.errstate(over="ignore", invalid="ignore"):
        upper = box[:2] + box[2:]
    if not np.isfinite(upper).all():
        return unavailable("nonfinite_box_endpoint")
    x0, y0 = min(max(int(box[0]), 0), width), min(max(int(box[1]), 0), height)
    x1, y1 = min(max(int(upper[0]), 0), width), min(max(int(upper[1]), 0), height)
    details["support_rectangle_xyxy_exclusive"] = [x0, y0, x1, y1]
    if x1 <= x0 or y1 <= y0:
        return unavailable("empty_clipped_rectangle")
    center = np.array([(x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2, 1.0])
    details["pixel_centroid_uv"] = center[:2].tolist()
    try:
        ray = np.linalg.solve(intrinsic, center)
    except np.linalg.LinAlgError as error:
        raise ValueError("K must be invertible") from error
    if not np.isfinite(ray).all() or not np.isclose(ray[2], 1.0, atol=1e-6):
        return unavailable("nonfinite_or_non_camera_z_ray")
    if policy == "center_pixel_abs":
        u, v = int(center[0]), int(center[1])
        measured = float(depth[v, u])
        details["center_sample_uv"] = [u, v]
        details["valid_depth_count"] = int(np.isfinite(measured) and measured > 1e-4)
        if not details["valid_depth_count"]:
            return unavailable("invalid_center_depth_no_fallback")
    else:
        # Cast only the box, BEFORE masking, so lower-precision threshold
        # comparisons have exactly the former float64 validity semantics.
        patch = depth[y0:y1, x0:x1].astype(np.float64, copy=False)
        samples = patch[np.isfinite(patch) & (patch > 1e-4)]
        details["valid_depth_count"] = int(len(samples))
        if not len(samples):
            return unavailable("no_valid_rectangle_depth")
        measured = _median(samples)
    details["measured_depth_statistic_m"] = measured
    candidate = pose.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        candidate[:3, 3] = measured * ray
    if not _valid_pose(candidate):
        return unavailable("invalid_provisional_surface_pose")
    offset = 0.0
    if points is not None:
        details["compensation_reference_pose"] = candidate.tolist()
        callback_pose, callback_K = candidate.copy(), intrinsic.copy()
        callback_pose.setflags(write=False)
        callback_K.setflags(write=False)
        uv, expected_z, ids = project_visible(callback_pose, callback_K, depth.shape)
        uv, expected_z, ids = np.asarray(uv), np.asarray(expected_z), np.asarray(ids)
        n = len(ids) if ids.ndim == 1 else -1
        if (ids.dtype.kind not in "iu" or n < 0 or uv.shape != (n, 2)
                or uv.dtype.kind not in "iu" or expected_z.shape != (n,) or expected_z.dtype.kind not in "iuf"
                or not np.isfinite(expected_z).all() or np.any(expected_z <= 1e-4)
                or np.any(ids < 0) or np.any(ids >= len(points))
                or np.any(uv < 0) or np.any(uv[:, 0] >= width) or np.any(uv[:, 1] >= height)
                or len(np.unique(ids)) != n):
            raise ValueError("project_visible returned invalid geometry/point indices")
        details["visible_cad_count"] = n
        details["visible_cad_indices_sha256"] = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
        # Same nondegenerate pixel bbox requirements as the FEAR host.
        if n < 3 or np.any(uv.max(axis=0) - uv.min(axis=0) + 1 < 3):
            return unavailable("no_nondegenerate_compensation_projection")
        with np.errstate(over="ignore", invalid="ignore"):
            offsets = (points[ids] @ pose[:3, :3].T)[:, 2]
        if not np.isfinite(offsets).all():
            return unavailable("nonfinite_modeled_surface_offsets")
        offset = _median(offsets)
        details["surface_offset_m"] = offset
    with np.errstate(over="ignore", invalid="ignore"):
        origin_z = measured - offset
        candidate[:3, 3] = origin_z * ray
    if not np.isfinite(origin_z) or origin_z <= 1e-4 or not _valid_pose(candidate):
        return unavailable("invalid_compensated_or_absolute_pose")
    details.update(valid=True, reason="candidate_constructed_not_verified",
                   absolute_origin_z_m=float(origin_z), candidate_pose=candidate.tolist())
    return candidate, details


class FEARAbsolutePoseController(FEARPoseController):
    """One additional current-depth proposal; no additional learned forward."""

    def __init__(self, config=None, tracker=None):
        config = config if config is not None else FEARAbsolutePoseConfig()
        if not isinstance(config, FEARAbsolutePoseConfig):
            raise TypeError("config must be FEARAbsolutePoseConfig")
        super().__init__(config, tracker=tracker)
        self._absolute_diagnostics = {}

    @property
    def proposal_model_provenance(self):
        result = super().proposal_model_provenance
        result["candidate_role"] = "current_box_absolute_depth_seed_only_R_unchanged_full_current_neural_output_required"
        result["absolute_seed_policy"] = {
            "name": self.config.absolute_seed_policy, "max_extra_geometric_candidates": 1,
            "mapping": "clipped_integer_rectangle_pixel_centroid_origin_ray",
            "translation_capped": False, "additional_learned_2d_calls": 0,
            "current_support_rule_and_margin_unchanged": True, "identity_guaranteed": False,
            "surface_offset_visibility": "provisional_absolute_surface_candidate_existing_sparse_CAD_projector",
            "upstream_relation": "seed_rule_adaptation_not_complete_DynamicPose_reproduction",
            "upstream_commit": _SOURCE_COMMIT, "upstream_files_sha256": dict(_SOURCE_HASHES),
            "boxaligned_and_matched_cap_variants": "not_implemented",
        }
        return result

    def reset(self, *args, **kwargs):
        self._absolute_diagnostics = {}
        result = super().reset(*args, **kwargs)
        self._absolute_diagnostics = {"policy": self.config.absolute_seed_policy, "attempted": False,
                                      "selected": False, "selection_reason": "registration"}
        self.diagnostics = {**self.diagnostics, "fear_absolute": copy.deepcopy(self._absolute_diagnostics)}
        return result

    def prepare(self, rgb, depth, K, timestamp=None):
        if not self._ready or self._pending:
            return super().prepare(rgb, depth, K, timestamp=timestamp)
        snapshot = self.__dict__.copy()
        try:
            chosen = super().prepare(rgb, depth, K, timestamp=timestamp)
            base_score, base_source = float(self.diagnostics["support_score"]), self._pending_source
            candidate, details = absolute_depth_seed(
                self.working_pose, depth, K, self._fear_proposal.bbox,
                policy=self.config.absolute_seed_policy,
                centered_points=self._points, project_visible=self._project_visible)
            evidence = self._evidence(candidate, depth, K) if candidate is not None else None
            supported = evidence is not None and self._supported(evidence)
            improves = bool(supported and evidence["support_score"] > base_score + self.config.seed_improvement_margin)
            details.update(attempted=True, selected=improves, base_source=base_source,
                           base_support_score=base_score, candidate_evidence=evidence,
                           candidate_depth_supported=bool(supported),
                           total_from_continuation_m=(None if candidate is None else
                               float(np.linalg.norm(candidate[:3, 3] - self._working_pose[:3, 3]))),
                           selection_reason=("current_depth_support_improved" if improves else
                               "candidate_unavailable" if candidate is None else
                               "not_supported_by_current_depth" if not supported else "insufficient_support_improvement"))
            if improves:
                chosen = candidate.copy()
                self._pending_seed = chosen.copy()
                self._pending_source = "fear_absolute_depth"
                self.diagnostics = {**self.diagnostics, **evidence, "source": self._pending_source}
            self._absolute_diagnostics = details
            self.diagnostics = {**self.diagnostics, "fear_absolute": copy.deepcopy(details)}
            return chosen.copy()
        except Exception:
            inner_invalidated = not self._ready
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            if getattr(self.tracker, "pending", None) is not None:
                try:
                    self.tracker.reject(self.tracker.pending)
                except Exception:
                    self._invalidate("fear_absolute_prepare_rollback_failed_explicit_reset_required")
            if inner_invalidated:
                self._invalidate("fear_absolute_nested_prepare_failure_explicit_reset_required")
            raise

    def observe(self, neural_pose_np, rgb, depth, K):
        details = copy.deepcopy(self._absolute_diagnostics)
        result = super().observe(neural_pose_np, rgb, depth, K)
        details["completed_fear_transaction"] = result["fear_2d"]["transaction"]
        self._absolute_diagnostics = details
        result["fear_absolute"] = copy.deepcopy(details)
        self.diagnostics = result
        return copy.deepcopy(result)
