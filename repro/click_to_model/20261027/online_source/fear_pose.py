"""Experimental local FEAR bbox proposals for every-frame neural 6D refinement.

FEAR changes ONLY a candidate input seed's x/y translation. R and z are copied
unchanged. No depth median, category prior, GT, cached output, global recognition,
dynamic template update, or additional 6D inference is performed here. The caller
must still execute the complete current-frame neural refiner exactly once (or its
unchanged configured iteration count) and retain its raw output on ALL frames.

By default, current sparse depth selects a seed only when it improves existing
strict support by the unchanged RobustPoseConfig margin. Opt-in candidate-only
coverage/reference-count policies never relax final neural verification. Neither
seed eligibility nor sparse support proves identity. Independently,
2D state commits require the final NN output's original strict verification AND
agreement with the appearance proposal. Full occlusion freezes 2D state while
the bounded numerical NN continuation remains active. Neither gate certifies
6D correctness, and bbox overlap is not a global re-identification capability.
"""

from dataclasses import dataclass
import copy
from numbers import Real

import numpy as np

from fear_tracker import FEARTracker
from robust_pose import RobustPoseConfig, RobustPoseController, _valid_pose


@dataclass(frozen=True)
class FEARPoseConfig(RobustPoseConfig):
    continuation_policy: str = "continuous"
    enable_prediction: bool = False
    enable_relocalization: bool = False
    tracker_device: str = "cuda"
    compile_tracker: bool = True
    tracker_warmup_calls: int = 1
    commit_iou: float = 0.30
    resync_verified_box: bool = True
    seed_allow_partial_coverage: bool = False
    seed_bypass_low_projection_reference: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.continuation_policy != "continuous" or not self.enable_quarantine:
            raise ValueError("FEARPoseConfig requires continuous bounded NN continuation")
        if self.enable_prediction or self.enable_relocalization:
            raise ValueError("FEARPoseConfig excludes motion prediction and ORB relocalization")
        if self.tracker_device != "cpu" and not (
                self.tracker_device == "cuda" or self.tracker_device.startswith("cuda:")):
            raise ValueError("tracker_device must be cpu, cuda, or cuda:N")
        if type(self.tracker_warmup_calls) is not int or self.tracker_warmup_calls < 0:
            raise ValueError("tracker_warmup_calls must be a nonnegative integer")
        if not np.isfinite(self.commit_iou) or not 0.30 <= self.commit_iou <= 1.0:
            raise ValueError("commit_iou must be finite and >= 0.30, <= 1")
        if type(self.compile_tracker) is not bool or type(self.resync_verified_box) is not bool:
            raise ValueError("compile_tracker and resync_verified_box must be bool")
        for name in ("seed_allow_partial_coverage", "seed_bypass_low_projection_reference"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")


def bbox_iou_xywh(first, second):
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if (a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or np.any(a[2:] <= 0) or np.any(b[2:] <= 0)):
        return 0.0
    intersection_size = np.maximum(0.0, np.minimum(a[:2] + a[2:], b[:2] + b[2:])
                                   - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(intersection_size))
    return intersection / (float(np.prod(a[2:]) + np.prod(b[2:])) - intersection)


class FEARPoseController(RobustPoseController):
    """prepare -> caller's COMPLETE neural forward -> observe, never output substitution.

    Like RobustPoseController, observe returns diagnostics, not a replacement
    pose. Tracker failures are explicit exceptions. A failed reset invalidates
    the controller until a successful reset, preventing mixed old/new templates.
    Ordinary prepare/observe input failures leave the frame retryable. Unexpected
    tracker transaction failures fail closed until reset rather than silently
    trusting mixed states. Initialization warm-up uses only registration RGB.
    """

    def __init__(self, config=None, tracker=None):
        config = config if config is not None else FEARPoseConfig()
        if not isinstance(config, FEARPoseConfig):
            raise TypeError("config must be FEARPoseConfig")
        super().__init__(config)
        self.tracker = (tracker if tracker is not None else FEARTracker(
            device=config.tracker_device, compile_model=config.compile_tracker))
        self._ready = False
        self._fear_proposal = None
        self._fear_diagnostics = {}
        self._warmup_completed = 0

    @property
    def proposal_model_provenance(self):
        result = copy.deepcopy(self.tracker.provenance())
        result["controller_warmup_calls_requested"] = self.config.tracker_warmup_calls
        result["controller_warmup_calls_completed"] = self._warmup_completed
        result["warmup_scope"] = "registration_rgb_only_propose_reject_excluded_with_registration"
        result["candidate_role"] = "xy_translation_seed_only_rotation_and_z_unchanged"
        if self._seed_admission_enabled:
            result["seed_admission"] = self._seed_admission_policy()
        return result

    @property
    def _seed_admission_enabled(self):
        return self.config.seed_allow_partial_coverage or self.config.seed_bypass_low_projection_reference

    def _seed_admission_policy(self):
        return {
            "allow_partial_coverage": self.config.seed_allow_partial_coverage,
            "bypass_low_projection_reference": self.config.seed_bypass_low_projection_reference,
            "scope": "existing_candidates_only_full_current_neural_refinement_required",
            "reference": "fixed_actual_parent_layer_selected_winner_evidence",
            "scores": "raw_coverage_minus_contradiction_never_replaced",
            "low_projection_policy": "reference_projected_count_below_min_inliers_not_missing_score_estimation",
            "final_observe_trusted_and_commit_rules_unchanged": True,
            "memory_decisions_guaranteed_unchanged": False,
            "innovation_reference": "current_selected_seed_so_changed_seed_can_change_final_admission",
            "identity_guaranteed": False,
        }

    def _seed_admission(self, evidence, reference):
        """Candidate-only, stateless comparison. Never changes strict _supported.

        Coverage may be omitted only for a seed proposal; all raw scores remain
        intact. B is an explicit count-based policy, including the case of an
        11-projection reference with score 1; it is NOT an estimate of missing
        support. This layer's fixed reference is never replaced by a peer winner.
        """
        def well_formed(value):
            if not isinstance(value, dict):
                return False
            for name in ("projected_count", "inlier_count"):
                number = value.get(name)
                if (isinstance(number, (bool, np.bool_)) or not isinstance(number, (int, np.integer))
                        or number < 0):
                    return False
            if value["inlier_count"] > value["projected_count"]:
                return False
            for name in ("coverage", "contradiction_fraction", "support_score"):
                number = value.get(name)
                if (isinstance(number, (bool, np.bool_)) or not isinstance(number, Real)
                        or not np.isfinite(number)):
                    return False
            return (0 <= value["coverage"] <= 1 and 0 <= value["contradiction_fraction"] <= 1
                    and -1 <= value["support_score"] <= 1)

        candidate_valid, reference_valid = well_formed(evidence), well_formed(reference)
        strict = bool(candidate_valid and self._supported(evidence))
        eligible = bool(candidate_valid and evidence["inlier_count"] >= self.config.min_inliers
                        and evidence["contradiction_fraction"] <= self.config.max_contradiction_fraction
                        and (self.config.seed_allow_partial_coverage
                             or evidence["coverage"] >= self.config.min_coverage))
        margin = bool(candidate_valid and reference_valid and evidence["support_score"] >
                      reference["support_score"] + self.config.seed_improvement_margin)
        low_projection = bool(reference_valid and reference["projected_count"] < self.config.min_inliers)
        bypass = bool(self.config.seed_bypass_low_projection_reference and low_projection)
        return {
            "candidate_evidence_valid": candidate_valid, "reference_evidence_valid": reference_valid,
            "strict_depth_supported": strict, "seed_eligible": eligible,
            "numeric_margin_passed": margin,
            "reference_projected_count": reference.get("projected_count") if reference_valid else None,
            "reference_support_score": reference.get("support_score") if reference_valid else None,
            "candidate_support_score": evidence.get("support_score") if candidate_valid else None,
            "reference_below_min_projected_count": low_projection,
            "low_projection_reference_bypass_allowed": bypass,
            "margin_bypass_used": bool(eligible and bypass and not margin),
            "admitted_to_seed_comparison": bool(eligible and reference_valid and (margin or bypass)),
            "verified": False,
        }

    @staticmethod
    def _visible_bbox_for(controller, pose, K, shape):
        uv, _, _ = controller._project_visible(pose, K, shape)
        if len(uv) < 3:
            return None
        lower, upper = uv.min(axis=0), uv.max(axis=0)
        side = upper - lower + 1  # integer pixel-support bounding rectangle
        if np.any(side < 3):
            return None
        return tuple(map(int, np.r_[lower, side]))

    def _visible_bbox(self, pose, K, shape):
        return self._visible_bbox_for(self, pose, K, shape)

    def _invalidate(self, reason):
        # No cached pose is returned by this error path. Explicit reset required.
        self._ready = False
        self._pending = False
        self._fear_proposal = None
        self.diagnostics = {"status": "uninitialized", "verified": False, "reason": reason}

    def reset(self, centered_pose, rgb, depth, K, model_points, model_normals,
              diameter, timestamp=None):
        self._ready = False
        self._warmup_completed = 0
        # Stage all base validation/geometry before replacing the 2D template.
        staged = RobustPoseController(self.config)
        try:
            result = staged.reset(centered_pose, rgb, depth, K, model_points,
                                  model_normals, diameter, timestamp=timestamp)
            box = self._visible_bbox_for(staged, np.asarray(centered_pose), K, np.shape(depth))
            if box is None:
                raise ValueError("Registration CAD projection has no nondegenerate visible bbox")
            self.tracker.initialize(rgb, box)
            for _ in range(self.config.tracker_warmup_calls):
                proposal = self.tracker.propose(rgb)
                self.tracker.reject(proposal)
                self._warmup_completed += 1
        except Exception:
            if getattr(self.tracker, "pending", None) is not None:
                try:
                    self.tracker.reject(self.tracker.pending)
                except Exception:
                    pass  # remain invalid; never proceed with uncertain state
            self._invalidate("fear_reset_failed_explicit_reset_required")
            raise
        self.__dict__.update(staged.__dict__)
        self._ready = True
        self._fear_proposal = None
        self._fear_diagnostics = {
            "status": "registered_fixed_template", "initial_bbox_xywh": list(box),
            "warmup_calls": self._warmup_completed, "transaction": "none",
        }
        self.diagnostics["fear_2d"] = copy.deepcopy(self._fear_diagnostics)
        return result

    def prepare(self, rgb, depth, K, timestamp=None):
        if not self._ready:
            raise RuntimeError("successful reset must precede FEAR prepare")
        if self._pending:
            raise RuntimeError("observe must follow each prepare before the next frame")
        snapshot = self.__dict__.copy()
        proposal = None
        try:
            seed = super().prepare(rgb, depth, K, timestamp=timestamp)
            reference_evidence = dict(self.diagnostics)
            seed_box = self._visible_bbox(seed, K, depth.shape)
            proposal = self.tracker.propose(rgb)
            box = np.asarray(proposal.bbox, dtype=np.float64)
            if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0)
                    or not np.isfinite(proposal.raw_score) or not np.isfinite(proposal.raw_logit)):
                raise ValueError("Malformed FEAR proposal bbox/activation")
            base_score = self.diagnostics["support_score"]
            diagnostic = {
                "proposal_bbox_xywh": box.tolist(), "raw_score": float(proposal.raw_score),
                "raw_logit": float(proposal.raw_logit),
                "raw_score_is_success_probability": False,
                "seed_bbox_xywh": None if seed_box is None else list(seed_box),
                "base_source": self._pending_source, "base_support_score": float(base_score),
                "candidate_depth_supported": False, "candidate_support_score": None,
                "candidate_evidence": None, "selected": False,
                "selection_reason": "no_nondegenerate_seed_projection",
                "transaction": "pending", "transaction_reason": "await_current_neural_observation",
            }
            if self._seed_admission_enabled:
                diagnostic["seed_admission"] = {**self._seed_admission_policy(),
                    **self._seed_admission(None, reference_evidence), "candidate_generated": False,
                    "selected_by_seed_policy": False}
            if seed_box is not None:
                seed_box_array = np.asarray(seed_box, dtype=np.float64)
                delta_pixels = box[:2] + box[2:] / 2 - seed_box_array[:2] - seed_box_array[2:] / 2
                candidate = seed.copy()
                # At fixed z, use the full 2x2 intrinsic block (also handles skew).
                candidate[:2, 3] += seed[2, 3] * np.linalg.solve(np.asarray(K)[:2, :2], delta_pixels)
                evidence = self._evidence(candidate, depth, K) if _valid_pose(candidate) else None
                supported = evidence is not None and self._supported(evidence)
                improved = (supported and evidence["support_score"] >
                            base_score + self.config.seed_improvement_margin)
                admission = self._seed_admission(evidence, reference_evidence) if self._seed_admission_enabled else None
                seed_admitted = admission["admitted_to_seed_comparison"] if admission is not None else bool(improved)
                diagnostic.update({
                    "candidate_translation_xy_m": candidate[:2, 3].tolist(),
                    "candidate_depth_supported": bool(supported),
                    "candidate_support_score": None if evidence is None else evidence["support_score"],
                    "candidate_evidence": evidence,
                    "selected": bool(seed_admitted),
                    "selection_reason": ("experimental_seed_admission_full_neural_required" if seed_admitted and not improved else
                                         "current_depth_support_improved" if improved else
                                         "not_supported_by_current_depth" if not supported else
                                         "insufficient_support_improvement"),
                })
                if self._seed_admission_enabled:
                    diagnostic["seed_admission"] = {**self._seed_admission_policy(), **admission,
                        "candidate_generated": True, "selected_by_seed_policy": bool(seed_admitted)}
                if seed_admitted:
                    seed = candidate
                    self._pending_seed = candidate.copy()
                    self._pending_source = "fear_translation"
                    self.diagnostics.update(evidence)
                    self.diagnostics["source"] = self._pending_source
            self._fear_proposal = proposal
            self._fear_diagnostics = diagnostic
            self.diagnostics["fear_2d"] = copy.deepcopy(diagnostic)
            return seed.copy()
        except Exception:
            self.__dict__.update(snapshot)
            if getattr(self.tracker, "pending", None) is not None:
                try:
                    self.tracker.reject(self.tracker.pending)
                except Exception:
                    self._invalidate("fear_prepare_rollback_failed_explicit_reset_required")
            raise

    def observe(self, neural_pose_np, rgb, depth, K):
        if not self._ready or not self._pending or self._fear_proposal is None:
            raise RuntimeError("successful FEAR prepare must precede observe")
        snapshot = self.__dict__.copy()
        # Base observe owns ALL strict pose/innovation/depth gates. It does not
        # modify neural_pose_np and returns diagnostics only, not a cached pose.
        result = super().observe(neural_pose_np, rgb, depth, K)
        proposal = self._fear_proposal
        try:
            final_box = (self._visible_bbox(neural_pose_np, K, depth.shape)
                         if result["neural_pose_valid"] else None)
            iou = bbox_iou_xywh(final_box, proposal.bbox) if final_box is not None else 0.0
            appearance_agreement = iou >= self.config.commit_iou
            commit = bool(result["verified"] and appearance_agreement)
            if commit:
                self.tracker.commit(proposal)
                if self.config.resync_verified_box:
                    self.tracker.reset_box(final_box)
            else:
                self.tracker.reject(proposal)
        except Exception:
            # Public tracker commits are atomic. A foreign implementation that
            # throws during a transaction must be explicitly reset, not retried
            # as if its search box/template were known to be synchronized.
            self.__dict__.update(snapshot)
            if getattr(self.tracker, "pending", None) is not None:
                try:
                    self.tracker.reject(self.tracker.pending)
                except Exception:
                    pass
            self._invalidate("fear_observe_transaction_failed_explicit_reset_required")
            raise
        self._fear_diagnostics = {
            **self._fear_diagnostics, "final_bbox_xywh": None if final_box is None else list(final_box),
            "final_proposal_iou": float(iou), "appearance_bbox_agreement": bool(appearance_agreement),
            "transaction": "commit" if commit else "reject",
            "transaction_reason": ("verified_nn_and_proposal_bbox_agree" if commit else
                                   "neural_output_not_strictly_verified" if not result["verified"] else
                                   "verified_depth_but_no_proposal_bbox_agreement"),
            "search_box_resynchronized": bool(commit and self.config.resync_verified_box),
            "fixed_template_updated": False,
        }
        self._fear_proposal = None
        result["fear_2d"] = copy.deepcopy(self._fear_diagnostics)
        self.diagnostics = result
        return copy.deepcopy(result)
