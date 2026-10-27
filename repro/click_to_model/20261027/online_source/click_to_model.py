"""RGB-D seed, observable validity and adaptive neural refinement orchestration.

The neural callback consumes and returns centered-CAD-to-camera poses (metres).
All raw iterations are retained, including invalid/rejected outputs. A rejected
frame is not silently replaced by an old pose or removed from evaluation.
"""
from dataclasses import asdict, dataclass, is_dataclass
import cv2
import numpy as np
from scipy.spatial import cKDTree
from adaptive_refinement import AdaptiveRefinementConfig, AdaptiveRefinementPolicy
from pose_validity import PoseValidityChecker
from quarantined_recovery import QuarantineConfig, QuarantinedRecovery
from rgbd_seed import RGBDSeedTracker, _pose as _seed_rigid_pose


@dataclass(frozen=True)
class ClickToModelConfig:
    max_translation_diameter: float = 0.35
    max_rotation_rad: float = 0.8
    max_unobserved_frames: int = 2
    enable_rgbd_seed: bool = True
    replenish_interval: int = 10
    urgent_replenishment: bool = True
    motion_prior: str = "none"
    enable_quarantined_recovery: bool = False

    def __post_init__(self):
        if (not np.isfinite(self.max_translation_diameter) or self.max_translation_diameter <= 0
                or not np.isfinite(self.max_rotation_rad) or self.max_rotation_rad <= 0):
            raise ValueError("innovation bounds must be finite positive")
        if type(self.max_unobserved_frames) is not int or self.max_unobserved_frames < 0:
            raise ValueError("max_unobserved_frames must be a nonnegative integer")
        if type(self.replenish_interval) is not int or self.replenish_interval < 1:
            raise ValueError("replenish_interval must be a positive integer")
        if type(self.urgent_replenishment) is not bool:
            raise ValueError("urgent_replenishment must be bool")
        if self.motion_prior not in ("none", "constant_velocity"):
            raise ValueError("motion_prior must be none or constant_velocity")
        if type(self.enable_quarantined_recovery) is not bool:
            raise ValueError("enable_quarantined_recovery must be bool")
        if self.enable_quarantined_recovery and not self.enable_rgbd_seed:
            raise ValueError("quarantined recovery requires the RGB-D seed provider")


class ClickToModelTracker:
    def __init__(self, checker: PoseValidityChecker, diameter, config=None,
                 seed_tracker=None, refinement_config=None, anchor_model_points=None,
                 recovery_config=None):
        self.checker = checker
        self.diameter = float(diameter)
        if not np.isfinite(self.diameter) or self.diameter <= 0:
            raise ValueError("diameter must be positive metres")
        self.config = config or ClickToModelConfig()
        self.seed_tracker = seed_tracker or RGBDSeedTracker()
        if self.config.enable_quarantined_recovery and not isinstance(self.seed_tracker, RGBDSeedTracker):
            raise TypeError("quarantined recovery requires a transactional RGBDSeedTracker")
        if self.config.enable_quarantined_recovery and recovery_config is None:
            recovery_config = QuarantineConfig(
                max_neural_translation_diameter=self.config.max_translation_diameter,
                max_neural_rotation_rad=self.config.max_rotation_rad,
                max_chain_step_translation_diameter=self.config.max_translation_diameter,
                max_chain_step_rotation_rad=self.config.max_rotation_rad)
        if self.config.enable_quarantined_recovery and (
                recovery_config.max_neural_translation_diameter > self.config.max_translation_diameter
                or recovery_config.max_neural_rotation_rad > self.config.max_rotation_rad
                or recovery_config.max_chain_step_translation_diameter > self.config.max_translation_diameter
                or recovery_config.max_chain_step_rotation_rad > self.config.max_rotation_rad):
            raise ValueError("recovery must preserve the controller's local motion/correction bounds")
        self._recovery = (QuarantinedRecovery(checker, diameter, config=recovery_config)
                         if self.config.enable_quarantined_recovery else None)
        self.policy = AdaptiveRefinementPolicy(refinement_config or AdaptiveRefinementConfig())
        self._pose = None
        self._trusted_pose = None
        self._lost = 0
        self._frame = 0
        self._last_replenish = 0
        self._motion_history = []
        self._anchor_model_points = np.asarray(
            self.checker.points if anchor_model_points is None else anchor_model_points, dtype=float)
        if (self._anchor_model_points.ndim != 2 or self._anchor_model_points.shape[1] != 3
                or len(self._anchor_model_points) < 3 or not np.isfinite(self._anchor_model_points).all()):
            raise ValueError("anchor_model_points must be finite Nx3 surface samples")
        self._surface_tree = cKDTree(self._anchor_model_points)

    def configuration(self):
        provider = {"class": type(self.seed_tracker).__name__}
        for name in ("config", "view_config"):
            value = getattr(self.seed_tracker, name, None)
            if is_dataclass(value):
                provider[name] = asdict(value)
        return {"tracking": asdict(self.config), "refinement": self.policy.configuration(),
                "seed_provider": provider,
                "recovery": None if self._recovery is None else asdict(self._recovery.config)}

    @property
    def trusted_pose(self):
        return None if self._trusted_pose is None else self._trusted_pose.copy()

    def _anchor_mask(self, pose, depth, K, mask=None):
        # A supplied segmentation can contain hands/background. Only depth
        # points near the registered CAD can initialize persistent identity
        # anchors. This is a geometric admission test, not identity proof.
        if mask is None:
            uv, _, _ = self.checker.project_visible(pose, K, depth.shape)
            raster = np.zeros(depth.shape, dtype=np.uint8)
            if len(uv) >= 3:
                cv2.fillConvexPoly(raster, cv2.convexHull(uv.astype(np.int32)), 1)
            mask = raster > 0
        else:
            mask = np.asarray(mask, dtype=bool).copy()
        yy, xx = np.nonzero(mask & np.isfinite(depth) & (depth > 1e-4))
        camera = np.column_stack((xx, yy, np.ones(len(xx)))) @ np.linalg.inv(K).T
        camera *= depth[yy, xx, None]
        local = (camera - pose[:3, 3]) @ pose[:3, :3]
        distance, _ = self._surface_tree.query(local)
        admitted = distance <= max(2 * self.checker.tolerance, self.diameter * .04)
        anchor_mask = np.zeros_like(mask)
        anchor_mask[yy[admitted], xx[admitted]] = True
        return anchor_mask

    def reset(self, pose, rgb, depth, K, mask):
        evidence = self.checker.evaluate(pose, rgb, depth, K, mask=mask)
        if not evidence.get("observation_valid", False):
            raise ValueError("registration lacks visible model/depth support; cannot initialize identity memory")
        anchor_mask = self._anchor_mask(pose, depth, K, mask)
        self.seed_tracker.reset(rgb, depth, K, anchor_mask, pose)
        if self._recovery is not None:
            self._recovery.cancel("new_registration")
        self.policy.calibrate(evidence)
        self._pose = np.asarray(pose, float).copy()
        self._trusted_pose = self._pose.copy()
        self._lost = 0
        self._frame = 0
        self._last_replenish = 0
        self._motion_history = []
        return {"status": "registered", "evidence": evidence,
                "initial_anchor_pixels": int(anchor_mask.sum()),
                "identity_status": "user_initialized_not_independently_verified"}

    def _innovation(self, pose, reference, *, symmetry_aligned=False):
        equivalent = pose if symmetry_aligned else self.checker.equivalent_pose(pose, reference)
        dt = float(np.linalg.norm(equivalent[:3, 3] - reference[:3, 3]))
        cosine = (np.trace(equivalent[:3, :3] @ reference[:3, :3].T) - 1) / 2
        dr = float(np.arccos(np.clip(cosine, -1, 1)))
        ok = (dt <= self.diameter * self.config.max_translation_diameter
              and dr <= self.config.max_rotation_rad)
        return ok, {"translation_m": dt, "rotation_rad_mod_symmetry": dr}

    def _predicted_reference(self):
        # Optional one-input prediction, based only on the two immediately
        # preceding successful correspondence commits. Any failure clears the
        # history; there is no extrapolation through a gap or from raw outputs.
        # Equal input spacing is an assumption, not an inferred timestamp.
        if self.config.motion_prior != "constant_velocity" or len(self._motion_history) != 2:
            return None
        previous, current = self._motion_history
        predicted = current.copy()
        predicted[:3, 3] = 2 * current[:3, 3] - previous[:3, 3]
        predicted[:3, :3] = current[:3, :3] @ previous[:3, :3].T @ current[:3, :3]
        return predicted if self._numerically_valid(predicted) else None

    @staticmethod
    def _numerically_valid(pose):
        # Persistent point anchors use this exact SE(3) predicate. Sharing it
        # avoids accepting a matrix here that commit would reject by exception.
        # This checks the raw output; it never projects/repairs its rotation.
        try:
            _seed_rigid_pose(pose)
        except (ValueError, TypeError):
            return False
        return bool(pose[2, 3] > 0)

    @staticmethod
    def _json_diagnostic(value):
        # JSON has no nonfinite numbers. Invalid raw recovery outputs are kept
        # exactly in raw_iterations; only the human-readable copy uses null.
        if isinstance(value, np.ndarray):
            return ClickToModelTracker._json_diagnostic(value.tolist())
        if isinstance(value, dict):
            return {key: ClickToModelTracker._json_diagnostic(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ClickToModelTracker._json_diagnostic(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    def step(self, rgb, depth, K, refine_one, mask=None):
        if self._pose is None:
            raise RuntimeError("reset must precede step")
        proposal = None
        history, iterations = [], []
        seed = self._pose.copy()
        seed_source = "last_neural_output"
        seed_diagnostics = {}
        seed_admitted = False
        seed_geometry_valid = False
        seed_continuous = False
        try:
            if self.config.enable_rgbd_seed:
                proposal = self.seed_tracker.propose(rgb, depth, K, mask=mask)
                seed_diagnostics = dict(proposal.diagnostics)
                seed_diagnostics["candidate_pose_centered"] = (None if proposal.pose is None
                    else np.asarray(proposal.pose, dtype=float).tolist())
                if proposal.accepted and proposal.pose is not None:
                    seed_evidence = self.checker.evaluate(proposal.pose, rgb, depth, K, mask=mask)
                    seed_geometry_valid = bool(seed_evidence.get("observation_valid", False))
                    working_evidence = self.checker.evaluate(self._pose, rgb, depth, K, mask=mask)
                    temporal_reference = self._pose if working_evidence.get("observation_valid", False) else self._trusted_pose
                    reference_name = "working" if temporal_reference is self._pose else "trusted"
                    predicted_reference = self._predicted_reference()
                    if predicted_reference is not None:
                        _, unpredicted_innovation = self._innovation(proposal.pose, temporal_reference)
                        seed_diagnostics["innovation_without_prediction"] = unpredicted_innovation
                        seed_diagnostics["predicted_reference_centered"] = predicted_reference.tolist()
                        temporal_reference = predicted_reference
                        reference_name = "constant_velocity_two_commits"
                    seed_continuous, seed_innovation = self._innovation(proposal.pose, temporal_reference)
                    seed_diagnostics["innovation_to_reference"] = seed_innovation
                    seed_diagnostics["innovation_reference"] = reference_name
                    seed_diagnostics["temporal_innovation_admitted"] = seed_continuous
                    # RGB-D rigid consensus is correspondence evidence. Current
                    # CAD geometry must also agree before moving the neural basin.
                    if seed_geometry_valid and seed_continuous:
                        seed = np.asarray(proposal.pose, float).copy()
                        seed_admitted = True
                        seed_source = "rgbd_correspondences"
            frame_seed = seed.copy()
            gauge_reference = proposal.pose if seed_admitted else frame_seed
            while True:
                more, stop_reason = self.policy.decide(history)
                if not more:
                    break
                pose = np.asarray(refine_one(seed.copy()), dtype=float)
                if pose.shape != (4, 4):
                    raise ValueError("refine_one must return 4x4 centered pose")
                evidence = self.checker.evaluate(pose, rgb, depth, K, mask=mask)
                rigid = self._numerically_valid(pose)
                if not rigid:
                    evidence = dict(evidence, status="invalid_pose", observation_valid=False,
                                    reason="neural_output_violates_seed_se3_contract")
                history.append(evidence)
                iterations.append(pose.copy())
                if not np.isfinite(pose).all():
                    stop_reason = "nonfinite_neural_output"
                    break
                if not rigid:
                    stop_reason = "invalid_neural_se3"
                    break
                seed = pose.copy()
            output = iterations[-1]
            evidence = history[-1]
            innovation = {"translation_m": None, "rotation_rad_mod_symmetry": None}
            numerically_valid = self._numerically_valid(output)
            internal_pose = output.copy()
            internal_pose_valid = False
            gauge = {"reference": "rgbd_proposal" if seed_admitted else "frame_seed",
                     "raw_output_preserved": True, "symmetry_alignment_applied": False,
                     "raw_to_internal_translation_m": None, "raw_to_internal_rotation_rad": None}
            innovation_ok = False
            if numerically_valid:
                # Anchor coordinates belong to the proposal's existing object
                # gauge. A symmetry-equivalent neural representative must be
                # brought back to that gauge before commit/replenish/state use.
                # The reported pose and every neural iteration remain raw.
                internal_pose = self.checker.equivalent_pose(output, gauge_reference)
                internal_pose_valid = self._numerically_valid(internal_pose)
                cosine = (np.trace(internal_pose[:3, :3] @ output[:3, :3].T) - 1) / 2
                gauge.update(
                    symmetry_alignment_applied=not np.allclose(internal_pose, output, atol=1e-8, rtol=0),
                    raw_to_internal_translation_m=float(np.linalg.norm(internal_pose[:3, 3] - output[:3, 3])),
                    raw_to_internal_rotation_rad=float(np.arccos(np.clip(cosine, -1, 1))))
                # A local correction is measured from this frame's actual seed,
                # never from a stale trusted keyframe. Otherwise a correct
                # moving trajectory eventually exceeds any fixed motion bound.
                if internal_pose_valid:
                    innovation_ok, innovation = self._innovation(internal_pose, frame_seed,
                                                                symmetry_aligned=True)
            continuous = seed_admitted or self._lost <= self.config.max_unobserved_frames
            accepted = bool(evidence.get("observation_valid", False) and innovation_ok and continuous)
            memory_updated = False
            recovery_result = None
            recovery_neural_call_attempts = 0
            recovery_returned_outputs = []
            iteration_sources = ["tracking"] * len(iterations)
            output_iteration_index = len(iterations) - 1
            if proposal is not None and accepted and seed_admitted:
                memory_updated = bool(self.seed_tracker.commit(proposal, verified_pose=internal_pose))
                accepted = memory_updated
                proposal = None
            if self._recovery is not None:
                recovery = self._recovery
                if memory_updated:
                    if recovery.active:
                        recovery_result = recovery.cancel("normal_commit_supersedes_recovery")
                else:
                    # This callback invokes the real refiner again from the
                    # isolated current candidate. The normal path's previously
                    # computed raw output cannot stand in for this observation.
                    def refine_recovery(candidate, *, rgb, depth, K, frame_id):
                        nonlocal recovery_neural_call_attempts
                        recovery_neural_call_attempts += 1
                        returned = refine_one(candidate)
                        recovery_returned_outputs.append(np.asarray(returned, dtype=float).copy())
                        return returned
                    if recovery.active:
                        alternatives = (() if proposal is None or not proposal.accepted
                                        else (proposal.pose,))
                        recovery_result = recovery.advance(self._frame + 1, rgb, depth, K,
                            refine_recovery, chain_id=recovery.chain_id, alternatives=alternatives)
                    elif (proposal is not None and proposal.accepted and seed_geometry_valid
                          and not seed_continuous):
                        recovery_result = recovery.begin(self.seed_tracker, proposal,
                            self._frame + 1, rgb, depth, K, refine_recovery)
                    if recovery_result is not None:
                        # While quarantined, even a locally plausible normal
                        # neural continuation cannot move the trusted state.
                        accepted = False
            if proposal is not None:
                self.seed_tracker.reject(proposal)
                proposal = None
            if recovery_result is not None:
                # Capture at the callback boundary. A later CAD/clock/source
                # exception in the recovery module must not erase an actual
                # network return that already happened this frame.
                recovery_raw = (recovery_returned_outputs[-1] if recovery_returned_outputs
                                else recovery_result.get("raw_pose"))
                if recovery_returned_outputs:
                    recovery_result["returned_neural_outputs"] = recovery_returned_outputs
                if recovery_raw is not None:
                    # Retain every extra neural call, including rejected ones.
                    # The explicit output index distinguishes these from the
                    # normal result retained during pending confirmation.
                    recovery_raw = np.asarray(recovery_raw, dtype=float)
                    if recovery_raw.shape == (4, 4):
                        iterations.append(recovery_raw.copy())
                        iteration_sources.append("quarantined_recovery")
                        current = (recovery_result.get("current_observation") or
                                   (recovery_result.get("trace") or [{}])[-1])
                        history.append(current.get("geometry", {
                            "status": "rejected_recovery_output", "observation_valid": False}))
                if recovery_result.get("status") == "confirmed":
                    snapshot = self._recovery.take_confirmed_state()
                    promoted = bool(snapshot is not None and self.seed_tracker.promote_confirmed(
                        snapshot, frame_id=self._frame + 1, rgb=rgb, depth=depth, K=K))
                    recovery_result["trusted_memory_updated"] = promoted
                    if promoted:
                        recovery_result["status"] = "promoted"
                        memory_updated = accepted = True
                        output = recovery_raw.copy()
                        output_iteration_index = len(iterations) - 1
                        internal_pose = np.asarray(recovery_result["internal_pose"], dtype=float).copy()
                        current = recovery_result["trace"][-1]
                        frame_seed = np.asarray(current["input_pose"], dtype=float)
                        evidence = current["geometry"]
                        numerically_valid = self._numerically_valid(output)
                        internal_pose_valid = self._numerically_valid(internal_pose)
                        innovation_ok, innovation = self._innovation(internal_pose, frame_seed,
                                                                    symmetry_aligned=True)
                        cosine = (np.trace(internal_pose[:3, :3] @ output[:3, :3].T) - 1) / 2
                        gauge = dict(reference="quarantined_rgbd_candidate", raw_output_preserved=True,
                            symmetry_alignment_applied=not np.allclose(internal_pose, output, atol=1e-8, rtol=0),
                            raw_to_internal_translation_m=float(np.linalg.norm(internal_pose[:3, 3] - output[:3, 3])),
                            raw_to_internal_rotation_rad=float(np.arccos(np.clip(cosine, -1, 1))))
                        seed_source = "quarantine_confirmed_rgbd"
                    else:
                        recovery_result.update(status="rejected", reason="confirmed_state_not_promoted")
                # Keep diagnostics JSON-safe; all numerical raw matrices also
                # remain in the separate complete neural iteration archive.
                recovery_result = self._json_diagnostic(recovery_result)
            self._frame += 1
            replenishment = None
            view_admission = None
            anchor_mask = None
            if memory_updated and hasattr(self.seed_tracker, "accept_view"):
                try:
                    anchor_mask = self._anchor_mask(internal_pose, depth, K, mask)
                    view_admission = self.seed_tracker.accept_view(
                        rgb, depth, K, anchor_mask, internal_pose,
                        model_points=self._anchor_model_points)
                except Exception as exc:
                    # A view is optional memory derived from a completed
                    # transaction, never a reason to roll that transaction back.
                    view_admission = {"updated": False, "reason": "view_admission_error",
                                      "error": f"{type(exc).__name__}: {exc}"}
            seed_config = getattr(self.seed_tracker, "config", None)
            active_count = (len(self.seed_tracker.active_points)
                            if memory_updated and hasattr(self.seed_tracker, "active_points") else None)
            reserve_low = (self.config.urgent_replenishment and active_count is not None
                           and active_count < 2 * getattr(seed_config, "min_points", 8))
            interval_due = self._frame - self._last_replenish >= self.config.replenish_interval
            if (memory_updated and hasattr(self.seed_tracker, "replenish")
                    and (interval_due or reserve_low)
                    and active_count is not None
                    and active_count < getattr(seed_config, "replenish_min_points", 32)):
                try:
                    if anchor_mask is None:
                        anchor_mask = self._anchor_mask(internal_pose, depth, K, mask)
                    replenishment = self.seed_tracker.replenish(
                        rgb, depth, K, anchor_mask, internal_pose,
                        model_points=self._anchor_model_points)
                except Exception as exc:
                    # Optional resampling must not undo a successful neural /
                    # correspondence transaction. Record the failure explicitly.
                    replenishment = {"updated": False, "reason": "replenishment_error",
                                     "error": f"{type(exc).__name__}: {exc}"}
                replenishment["trigger"] = "low_point_reserve" if reserve_low else "interval"
                replenishment["active_points_before"] = active_count
                self._last_replenish = self._frame
            if accepted:
                self._trusted_pose = internal_pose.copy()
                self._lost = 0
            else:
                self._lost += 1
            if memory_updated:
                self._motion_history = (self._motion_history + [internal_pose.copy()])[-2:]
            else:
                self._motion_history = []
            # Bounded neural continuation is quarantined. A catastrophic result
            # remains in the record but cannot corrupt the following frame seed.
            self._pose = internal_pose.copy() if internal_pose_valid and innovation_ok else self._trusted_pose.copy()
            return {
                "pose": output.copy(), "raw_iterations": np.asarray(iterations),
                "diagnostics": {
                    "status": "tracking" if accepted else "uncertain_or_lost",
                    "observation_valid": bool(evidence.get("observation_valid", False)),
                    "verified": accepted,
                    "verified_semantics": "local observation and continuity admission, not GT accuracy",
                    "identity_verified": False,
                    "identity_status": "correspondence_supported" if memory_updated else "unresolved",
                    "source": seed_source, "seed": seed_diagnostics,
                    "seed_admitted": seed_admitted,
                    "memory_updated": memory_updated, "lost_frames": self._lost,
                    "replenishment": replenishment,
                    "view_admission": view_admission,
                    "recovery": recovery_result,
                    "neural_iterations": len(iterations), "stop_reason": stop_reason,
                    "iteration_sources": iteration_sources,
                    "output_iteration_index": output_iteration_index,
                    "recovery_neural_call_attempts": recovery_neural_call_attempts,
                    "numerically_valid": numerically_valid,
                    "internal_pose_valid": internal_pose_valid, "internal_pose_gauge": gauge,
                    "innovation": innovation, "evidence": evidence,
                    "iteration_evidence": history,
                },
            }
        except BaseException:
            self._motion_history = []
            if self._recovery is not None:
                self._recovery.cancel("controller_exception")
            if proposal is not None:
                self.seed_tracker.reject(proposal)
            raise
