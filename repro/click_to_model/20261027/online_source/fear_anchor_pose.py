"""Opt-in jointly admitted historical full-pose seeds; never output replacement.

One existing FEAR proposal and one caller-owned full current-frame 6D forward
remain mandatory. A past pose is eligible only after strict neural/depth gates
and a successful FEAR appearance transaction. Those gates can still falsely
admit a distractor; neither past nor current geometric support proves identity.

Nested preparation failures preserve the parent's invalidated state; explicit
reset is required after an uncertain external tracker transaction.
"""
from dataclasses import dataclass
import copy

import numpy as np

from fear_depth_pose import (FEARDepthPoseConfig, FEARDepthPoseController,
                             depth_residual_modes, depth_residual_median_all, depth_residual_modes_all)
from robust_pose import _valid_pose


@dataclass(frozen=True)
class FEARAnchorPoseConfig(FEARDepthPoseConfig):
    anchor_min_rejected_streak: int = 2
    enable_anchor_depth: bool = True

    def __post_init__(self):
        super().__post_init__()
        if type(self.anchor_min_rejected_streak) is not int or self.anchor_min_rejected_streak < 2:
            raise ValueError("anchor_min_rejected_streak must be an integer >= 2")
        if type(self.enable_anchor_depth) is not bool:
            raise ValueError("enable_anchor_depth must be boolean")


class FEARAnchorPoseController(FEARDepthPoseController):
    def __init__(self, config=None, tracker=None):
        config = config if config is not None else FEARAnchorPoseConfig()
        if not isinstance(config, FEARAnchorPoseConfig):
            raise TypeError("config must be FEARAnchorPoseConfig")
        super().__init__(config, tracker=tracker)
        self._clear_joint_anchor()

    def _clear_joint_anchor(self):
        self._joint_anchor_pose = None
        self._joint_anchor_step = None
        self._joint_anchor_time = None
        self._joint_anchor_admission = None
        self._anchor_diagnostics = {}

    @property
    def joint_anchor_pose(self):
        return None if self._joint_anchor_pose is None else self._joint_anchor_pose.copy()

    @property
    def joint_anchor_step(self):
        return self._joint_anchor_step

    @property
    def proposal_model_provenance(self):
        result = super().proposal_model_provenance
        result["candidate_role"] = "current_working_FEAR_depth_or_historical_joint_anchor_seed_only_full_current_neural_output_required"
        result["depth_seed_policy"]["rotation_invariance_scope"] = "each_depth_ray_operator_preserves_its_input_pose_rotation_only"
        result["joint_anchor_policy"] = {
            "memory_slots": 1, "admission": "current_neural_strict_verified_AND_successful_FEAR_commit",
            "registration_eligible": False, "anchor_min_rejected_streak": self.config.anchor_min_rejected_streak,
            "rejected_streak_semantics": "existing_neural_verification_failures_not_appearance_rejections",
            "enable_anchor_depth": self.config.enable_anchor_depth,
            "rotation_and_initial_z_source": "historical_jointly_admitted_full_neural_pose",
            "full_anchor_can_change_rotation_and_z_relative_to_working": True,
            "depth_cap_reference": "anchor_xy_aligned_seed_not_total_from_working_state",
            "current_support_rule_and_margin_unchanged": not self._seed_admission_enabled, "additional_learned_2d_calls": 0,
            "joint_admission_guarantees_identity": False, "parent_trusted_and_commit_rules_unchanged": True,
        }
        if self._seed_admission_enabled:
            result["joint_anchor_policy"]["memory_decisions_guaranteed_unchanged"] = False
            result["joint_anchor_policy"]["seed_relative_innovation_can_change_admission"] = True
        return result

    def reset(self, *args, **kwargs):
        self._clear_joint_anchor()
        result = super().reset(*args, **kwargs)
        self._anchor_diagnostics = {"attempted": False, "selected": False,
                                    "selection_reason": "registration_is_not_joint_verification",
                                    "anchor_admitted_on_observe": False, "stored_anchor_step_after_observe": None}
        self.diagnostics = {**self.diagnostics, "fear_anchor": copy.deepcopy(self._anchor_diagnostics)}
        return result

    def _anchor_depth_modes(self, residuals, bound):
        if self.config.depth_seed_policy == "bounded_modes_v1":
            return depth_residual_modes(residuals, bound, self._depth_tolerance,
                                        self.config.min_inliers, self.config.max_depth_modes)
        if self.config.depth_seed_policy == "median_all":
            return depth_residual_median_all(residuals, self.config.min_inliers)
        return depth_residual_modes_all(residuals, self._depth_tolerance,
                                        self.config.min_inliers, self.config.max_depth_modes)

    def prepare(self, rgb, depth, K, timestamp=None):
        if not self._ready or self._pending:
            # An accidental double prepare must retain the previous transaction.
            return super().prepare(rgb, depth, K, timestamp=timestamp)
        snapshot = self.__dict__.copy()
        try:
            chosen = super().prepare(rgb, depth, K, timestamp=timestamp)
            reference_evidence = dict(self.diagnostics)
            initial_score = float(self.diagnostics["support_score"])
            anchor = self.joint_anchor_pose
            anchor_step = self._joint_anchor_step
            age = None if anchor_step is None else self._step - anchor_step
            bound = self._diameter * self.config.max_depth_increment_diameter
            details = {"attempted": False, "selected": False, "selected_kind": None,
                       "anchor_source_step": anchor_step, "anchor_age_frames": age,
                       "anchor_source_time": self._joint_anchor_time,
                       "anchor_admission_evidence": copy.deepcopy(self._joint_anchor_admission),
                       "base_source": self._pending_source, "base_support_score": initial_score,
                       "anchor_min_rejected_streak": self.config.anchor_min_rejected_streak,
                       "rejected_streak_before_frame": self._rejected_streak,
                       "depth_seed_policy": self.config.depth_seed_policy,
                       "depth_cap_reference": "anchor_xy_aligned_seed",
                       "max_depth_increment_m": float(bound), "candidates": [],
                       "selection_reason": "no_jointly_admitted_anchor" if anchor is None else "rejection_streak_below_threshold",
                       "anchor_admitted_on_observe": False}
            if self._seed_admission_enabled:
                details["seed_admission"] = {**self._seed_admission_policy(),
                    "reference_source": self._pending_source,
                    "reference_projected_count": reference_evidence["projected_count"],
                    "reference_support_score": initial_score, "selected_by_seed_policy": False}
            if anchor is not None and self._rejected_streak >= self.config.anchor_min_rejected_streak:
                details["attempted"] = True
                box = np.asarray(self._fear_proposal.bbox, np.float64)
                anchor_box = self._visible_bbox(anchor, K, depth.shape)
                details["anchor_bbox_xywh"] = None if anchor_box is None else list(anchor_box)
                details["proposal_bbox_xywh"] = box.tolist()
                details["selection_reason"] = "no_nondegenerate_anchor_projection"
                if anchor_box is not None:
                    projected_box = np.asarray(anchor_box, np.float64)
                    pixels = box[:2] + box[2:] / 2 - projected_box[:2] - projected_box[2:] / 2
                    xy_seed = anchor.copy()
                    xy_seed[:2, 3] += anchor[2, 3] * np.linalg.solve(np.asarray(K)[:2, :2], pixels)
                    candidates = [("xy", xy_seed, {})]
                    if self.config.enable_anchor_depth and _valid_pose(xy_seed):
                        uv, expected_z, _ = self._project_visible(xy_seed, K, depth.shape)
                        measured = depth[uv[:, 1], uv[:, 0]]
                        valid = np.isfinite(measured) & (measured > 1e-4)
                        residuals = measured[valid] - expected_z[valid]
                        details["depth_residual_valid_points"] = int(valid.sum())
                        for mode_index, mode in enumerate(self._anchor_depth_modes(residuals, bound)):
                            geometry = {}
                            candidate, reason = self._make_depth_candidate(xy_seed, mode["dz_m"], box, K,
                                                                          depth.shape, bound, diagnostics=geometry)
                            candidates.append(("depth", candidate, {"mode_index": mode_index, **mode, **geometry,
                                                                    "construction_reason": reason}))
                    best_pose = best_evidence = best_kind = best_index = None
                    details["selection_reason"] = "no_strictly_supported_improving_anchor_candidate"
                    for index, (kind, candidate, metadata) in enumerate(candidates):
                        valid = candidate is not None and _valid_pose(candidate)
                        evidence = self._evidence(candidate, depth, K) if valid else None
                        supported = evidence is not None and self._supported(evidence)
                        improved = supported and evidence["support_score"] > initial_score + self.config.seed_improvement_margin
                        admission = self._seed_admission(evidence, reference_evidence) if self._seed_admission_enabled else None
                        item = {"kind": kind, "valid": bool(valid), "supported": bool(supported),
                                "improves_existing_seed": bool(improved), "evidence": evidence, **metadata}
                        if self._seed_admission_enabled:
                            item["seed_admission"] = {**admission, "selected_by_seed_policy": False}
                        if valid:
                            item.update(candidate_translation_xyz_m=candidate[:3, 3].tolist(),
                                        depth_increment_from_anchor_xy_m=float(np.linalg.norm(candidate[:3, 3] - xy_seed[:3, 3])),
                                        total_from_working_m=float(np.linalg.norm(candidate[:3, 3] - self._working_pose[:3, 3])))
                        details["candidates"].append(item)
                        seed_admitted = admission["admitted_to_seed_comparison"] if admission is not None else bool(improved)
                        if seed_admitted and (best_evidence is None or evidence["support_score"] > best_evidence["support_score"]):
                            best_pose, best_evidence, best_kind, best_index = candidate, evidence, kind, index
                    if best_pose is not None:
                        chosen = best_pose.copy()
                        self._pending_seed = chosen.copy()
                        self._pending_source = "fear_joint_anchor_" + best_kind
                        self._pending_seed_state_age = age
                        self._pending_relocalization_valid = False
                        self.diagnostics = {**self.diagnostics, **best_evidence, "source": self._pending_source,
                                            "seed_state_age_frames": age}
                        details.update(selected=True, selected_kind=best_kind, selected_candidate=best_index,
                                       selection_reason=("current_depth_support_improved" if details["candidates"][best_index]["improves_existing_seed"]
                                                         else "experimental_seed_admission_full_neural_required"))
                        if self._seed_admission_enabled:
                            details["seed_admission"]["selected_by_seed_policy"] = True
                            details["candidates"][best_index]["seed_admission"]["selected_by_seed_policy"] = True
            self._anchor_diagnostics = details
            self.diagnostics = {**self.diagnostics, "fear_anchor": copy.deepcopy(details)}
            return chosen.copy()
        except Exception:
            # Requires the parent's nested rollback fix: preserve an already
            # irreversible tracker failure, even if a later cleanup succeeds.
            inner_invalidated = not self._ready
            self.__dict__.clear()  # also remove attributes first created by the failed prepare
            self.__dict__.update(snapshot)
            if getattr(self.tracker, "pending", None) is not None:
                try:
                    self.tracker.reject(self.tracker.pending)
                except Exception:
                    self._invalidate("fear_anchor_prepare_rollback_failed_explicit_reset_required")
            if inner_invalidated:
                self._invalidate("fear_anchor_nested_prepare_failure_explicit_reset_required")
            raise

    def observe(self, neural_pose_np, rgb, depth, K):
        # Allocate a private copy before the external transaction can commit.
        staged_anchor = np.asarray(neural_pose_np, np.float64).copy() if _valid_pose(neural_pose_np) else None
        result = super().observe(neural_pose_np, rgb, depth, K)
        admitted = bool(result["verified"] and result["fear_2d"]["transaction"] == "commit")
        if admitted:
            self._joint_anchor_pose = staged_anchor
            self._joint_anchor_step = self._step
            self._joint_anchor_time = self._frame_time
            self._joint_anchor_admission = {
                "step": self._step, "time": self._frame_time,
                "gate": "strict_current_neural_verified_and_successful_fear_commit",
                "inlier_count": result["inlier_count"], "coverage": result["coverage"],
                "contradiction_fraction": result["contradiction_fraction"], "support_score": result["support_score"],
                "final_proposal_iou": result["fear_2d"]["final_proposal_iou"],
                "identity_guaranteed": False,
            }
        # Preserve the construction age of THIS frame's seed, even if its
        # current neural observation has just admitted a newer anchor.
        self._anchor_diagnostics = {**self._anchor_diagnostics, "anchor_admitted_on_observe": admitted,
                                    "stored_anchor_step_after_observe": self._joint_anchor_step,
                                    "completed_fear_transaction": result["fear_2d"]["transaction"]}
        result["fear_anchor"] = copy.deepcopy(self._anchor_diagnostics)
        self.diagnostics = result
        return copy.deepcopy(result)
