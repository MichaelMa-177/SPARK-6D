"""Three-observation recovery confirmation without mutating trusted memory.

A seed rejected by a normal controller may start an isolated RGB-D point chain.
Every observation calls a supplied current-frame neural refiner from its actual
candidate pose. Only a continuous, unambiguous, geometrically supported chain
can export a confirmed point snapshot. The caller decides whether to promote
that snapshot; this module never writes a trusted tracker or claims identity.
"""
from dataclasses import dataclass, replace
import copy
import time

import numpy as np

from rgbd_seed import RGBDSeedTracker, RGBDSeedProposal, _Pending, _pose


@dataclass(frozen=True)
class QuarantineConfig:
    required_observations: int = 3
    max_pending_frames: int = 5
    max_pending_seconds: float = 1.
    max_hypotheses: int = 8
    max_neural_translation_diameter: float = .35
    max_neural_rotation_rad: float = .8
    max_chain_step_translation_diameter: float = .35
    max_chain_step_rotation_rad: float = .8
    equivalent_translation_m: float = .015
    equivalent_rotation_rad: float = .15

    def __post_init__(self):
        for name in ("required_observations", "max_pending_frames", "max_hypotheses"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if self.required_observations < 3 or self.required_observations > self.max_pending_frames:
            raise ValueError("at least three observations must fit in the pending frame budget")
        for name in ("max_pending_seconds", "max_neural_translation_diameter", "max_neural_rotation_rad",
                     "max_chain_step_translation_diameter", "max_chain_step_rotation_rad",
                     "equivalent_translation_m", "equivalent_rotation_rad"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite positive")


@dataclass(frozen=True)
class ConfirmedPointState:
    """Transferred snapshot; not an implicit write to the original tracker."""
    frame_id: int
    chain_id: int
    pose: np.ndarray
    gray: np.ndarray
    pixels: np.ndarray
    anchors: np.ndarray
    K: np.ndarray
    depth: np.ndarray
    rgb: np.ndarray
    mask_area: int
    source_identity: object
    source_memory_revision: int
    seal: bytes = b""


class QuarantinedRecovery:
    def __init__(self, checker, diameter, config=None, backend=None, clock=None):
        self.checker, self.diameter = checker, float(diameter)
        if not np.isfinite(self.diameter) or self.diameter <= 0:
            raise ValueError("diameter must be positive metres")
        self.config = config or QuarantineConfig()
        if not isinstance(self.config, QuarantineConfig):
            raise TypeError("config must be QuarantineConfig")
        self._backend, self._clock = backend, clock or time.monotonic
        self._serial = 0
        self._branch = self._confirmed = None
        self._anchors = set()
        self._trace = []
        self._started = self._last_frame = None
        self._source = self._source_identity = self._source_revision = None
        self._current_attempt = None

    @property
    def active(self):
        return self._branch is not None

    @property
    def chain_id(self):
        return self._serial

    def _clear(self):
        self._branch = self._confirmed = None
        self._anchors = set()
        self._trace = []
        self._started = self._last_frame = None
        self._source = self._source_identity = self._source_revision = None
        self._current_attempt = None

    def cancel(self, reason="cancelled"):
        self._clear()
        return self._rejection(reason)

    @staticmethod
    def _rejection(reason):
        return dict(status="rejected", reason=reason, trusted_memory_updated=False,
                    identity_verified=False, raw_pose=None, internal_pose=None)

    def _abort(self, error):
        current, trace, chain = copy.deepcopy(self._current_attempt), copy.deepcopy(self._trace), self._serial
        result = self.cancel("source_or_refiner_error")
        if not isinstance(error, Exception):
            raise error
        result["error"] = f"{type(error).__name__}: {error}"
        if current is not None:
            result.update(current_observation=current, trace=trace,
                          frame_id=current["frame_id"], chain_id=chain)
            if current["raw_pose"] is not None:
                result["raw_pose"] = np.asarray(current["raw_pose"], float)
        return result

    def take_confirmed_state(self):
        if self._confirmed is not None and not self._source_current():
            self._clear()
            return None
        result = self._confirmed
        if result is not None:
            self._clear()
        return result

    def _source_current(self):
        return (self._source is not None and self._source.memory_identity is self._source_identity
                and self._source.memory_revision == self._source_revision)

    @staticmethod
    def _valid_frame(frame_id):
        return type(frame_id) is int and frame_id >= 0

    def _expired(self):
        elapsed = self._clock() - self._started
        return not np.isfinite(elapsed) or elapsed < 0 or elapsed > self.config.max_pending_seconds

    def _equivalent(self, pose, reference):
        equivalent = _pose(self.checker.equivalent_pose(pose, reference))
        translation = float(np.linalg.norm(equivalent[:3, 3] - reference[:3, 3]))
        rotation = float(np.arccos(np.clip((np.trace(equivalent[:3, :3] @ reference[:3, :3].T) - 1) / 2, -1, 1)))
        return equivalent, translation, rotation

    def begin(self, source_seed, source_proposal, frame_id, rgb, depth, K, refine_one, *, alternatives=()):
        """Copy an active proposal; source tracker/transaction remain untouched.

        refine_one(seed, *, rgb, depth, K, frame_id) must perform a new current-
        frame neural call. Saved neural outputs are not accepted as an input to
        this API. The source owner remains responsible for its own reject/commit.
        """
        if self.active or self._confirmed is not None:
            return self.cancel("pending_chain_replaced")
        if not self._valid_frame(frame_id):
            return self.cancel("invalid_frame_id")
        try:
            pending = source_seed._pending
            if pending is None or pending.proposal is not source_proposal or not source_proposal.accepted:
                return self.cancel("no_active_supported_source_proposal")
            self._source = source_seed
            self._source_identity = source_seed.memory_identity
            self._source_revision = source_seed.memory_revision
            gray, measured_depth, intrinsics, _ = source_seed._frame(rgb, depth, K, None)
            pose = _pose(source_proposal.pose)
            observed, valid = source_seed._lift(pending.pixels, measured_depth, intrinsics)
            if (pose[2, 3] <= 0 or not np.array_equal(gray, pending.gray)
                    or not np.array_equal(intrinsics, pending.K) or not valid.all()
                    or not np.allclose(observed, pending.observed, atol=1e-9, rtol=0)):
                return self.cancel("source_observation_mismatch")
            branch = RGBDSeedTracker(replace(source_seed.config, enable_relocalization=False), self._backend)
            copied = RGBDSeedProposal(pose.copy(), True, copy.deepcopy(source_proposal.diagnostics), 1)
            branch._gray, branch._pose = gray.copy(), pose.copy()
            branch._pixels, branch._anchors = pending.pixels.copy(), pending.anchors.copy()
            branch._mask_area, branch._generation = pending.mask_area, 1
            branch._pending = _Pending(copied, gray.copy(), pending.pixels.copy(), pending.anchors.copy(),
                pending.observed.copy(), intrinsics.copy(), pending.mask_area)
            self._serial += 1
            self._branch = branch
            self._anchors = {tuple(point) for point in pending.anchors}
            self._started = self._clock()
            self._trace = []
            return self._observe(copied, frame_id, rgb, measured_depth, intrinsics, refine_one, alternatives)
        except BaseException as error:
            return self._abort(error)

    def advance(self, frame_id, rgb, depth, K, refine_one, *, chain_id, alternatives=()):
        if not self.active:
            # An already confirmed snapshot remains available until explicit
            # consumption/cancellation, even if a caller attempts to advance.
            return self._rejection("no_pending_chain")
        try:
            if not self._source_current():
                return self.cancel("source_memory_changed")
            if type(chain_id) is not int or chain_id != self._serial:
                return self.cancel("different_anchor_chain")
            if not self._valid_frame(frame_id) or frame_id != self._last_frame + 1:
                return self.cancel("nonconsecutive_observation")
            if len(self._trace) >= self.config.max_pending_frames or self._expired():
                return self.cancel("pending_budget_exhausted")
            proposal = self._branch.propose(rgb, depth, K)
            if not proposal.accepted:
                return self.cancel("point_chain_lost")
            return self._observe(proposal, frame_id, rgb, depth, K, refine_one, alternatives)
        except BaseException as error:
            return self._abort(error)

    def _observe(self, proposal, frame_id, rgb, depth, K, refine_one, alternatives):
        branch, cfg = self._branch, self.config
        pending = branch._pending
        if not self._source_current():
            return self.cancel("source_memory_changed")
        rgb_before, depth_before, K_before = np.asarray(rgb).copy(), np.asarray(depth).copy(), np.asarray(K).copy()
        current = dict(frame_id=frame_id, input_pose=np.asarray(proposal.pose).tolist(),
                       raw_pose=None, internal_pose=None, current_frame_neural_called=False)
        self._current_attempt = current
        def reject(reason):
            # Retain failed current observations for audit while destroying all
            # pending tracking state. A failed call is never silently scored as
            # a successful observation in the prior trace.
            prior, chain = copy.deepcopy(self._trace), self._serial
            result = self.cancel(reason)
            result.update(frame_id=frame_id, chain_id=chain, trace=prior,
                          current_observation=copy.deepcopy(current))
            if current["raw_pose"] is not None:
                result["raw_pose"] = np.asarray(current["raw_pose"], float)
            return result
        if len(alternatives) + 1 > cfg.max_hypotheses or self._expired():
            return reject("pending_budget_exhausted")
        if (not {tuple(point) for point in pending.anchors} <= self._anchors
                or len({tuple(point) for point in pending.anchors}) < branch.config.min_points):
            return reject("different_or_insufficient_anchor_chain")
        pose = _pose(proposal.pose)
        if pose[2, 3] <= 0:
            return reject("invalid_candidate_pose")
        if self._trace:
            _, translation, rotation = self._equivalent(pose, branch.trusted_pose)
            current.update(candidate_step_translation_m=translation, candidate_step_rotation_rad=rotation)
            if (translation > self.diameter * cfg.max_chain_step_translation_diameter
                    or rotation > cfg.max_chain_step_rotation_rad):
                return reject("point_chain_motion_rejected")
        evidence = self.checker.evaluate(pose, rgb, depth, K)
        current["candidate_geometry"] = evidence
        if evidence.get("observation_valid") is not True:
            return reject("candidate_geometry_rejected")
        for alternative in alternatives:
            alternate = _pose(alternative)
            if alternate[2, 3] <= 0:
                return reject("invalid_alternative_pose")
            if self.checker.evaluate(alternate, rgb, depth, K).get("observation_valid") is True:
                _, translation, rotation = self._equivalent(alternate, pose)
                if translation > cfg.equivalent_translation_m or rotation > cfg.equivalent_rotation_rad:
                    return reject("ambiguous_recovery_hypotheses")
        # The callback receives this exact current observation and seed. There
        # is no API parameter for old neural predictions or precomputed scores.
        raw = np.asarray(refine_one(pose.copy(), rgb=rgb, depth=depth, K=K, frame_id=frame_id), dtype=float)
        current.update(raw_pose=raw.tolist(), current_frame_neural_called=True)
        if self._branch is not branch:
            return reject("chain_cancelled_during_refinement")
        if not self._source_current():
            return reject("source_memory_changed")
        if (not np.array_equal(rgb, rgb_before) or not np.array_equal(depth, depth_before, equal_nan=True)
                or not np.array_equal(K, K_before)):
            return reject("observation_changed_during_refinement")
        try:
            raw_rigid = _pose(raw)
            if raw_rigid[2, 3] <= 0:
                raise ValueError("neural pose is behind the camera")
            internal, translation, rotation = self._equivalent(raw_rigid, pose)
            if internal[2, 3] <= 0:
                raise ValueError("internal pose is behind the camera")
        except (ValueError, TypeError):
            return reject("invalid_neural_pose")
        current.update(internal_pose=internal.tolist(), neural_translation_m=translation,
                       neural_rotation_rad=rotation)
        evidence = self.checker.evaluate(raw, rgb, depth, K)
        current["geometry"] = evidence
        if (evidence.get("observation_valid") is not True
                or translation > self.diameter * cfg.max_neural_translation_diameter
                or rotation > cfg.max_neural_rotation_rad):
            return reject("neural_or_geometry_rejected")
        if self._trace:
            _, translation, rotation = self._equivalent(internal, branch.trusted_pose)
            current.update(neural_step_translation_m=translation, neural_step_rotation_rad=rotation)
            if (translation > self.diameter * cfg.max_chain_step_translation_diameter
                    or rotation > cfg.max_chain_step_rotation_rad):
                return reject("neural_chain_motion_rejected")
        if self._expired():
            return reject("pending_budget_exhausted")
        if not branch.commit(proposal, verified_pose=internal):
            return reject("neural_point_consistency_rejected")
        if not self._source_current():
            return reject("source_memory_changed")
        if self._expired():
            return reject("pending_budget_exhausted")
        self._last_frame = frame_id
        current["anchor_count"] = len(branch.active_points)
        self._trace.append(current)
        confirmed = len(self._trace) >= cfg.required_observations
        result = dict(status="confirmed" if confirmed else "pending", reason="consistent_current_frame_chain",
            frame_id=frame_id, chain_id=self._serial, observations=len(self._trace),
            raw_pose=raw.copy(), internal_pose=internal.copy(), trace=copy.deepcopy(self._trace),
            trusted_memory_updated=False, identity_verified=False,
            hypothesis_scope="candidate and supplied alternatives; no independent instance-identity truth")
        if confirmed:
            def frozen(array):
                array = np.asarray(array).copy()
                array.setflags(write=False)
                return array
            self._confirmed = ConfirmedPointState(frame_id, self._serial, frozen(branch.trusted_pose),
                frozen(branch._gray), frozen(branch.active_points), frozen(branch._anchors),
                frozen(K_before), frozen(depth_before), frozen(rgb_before), branch._mask_area,
                self._source_identity, self._source_revision)
            self._confirmed = replace(self._confirmed, seal=self._source._seal_confirmed(self._confirmed))
            if self._expired():
                return reject("pending_budget_exhausted")
            if not self._source_current():
                return reject("source_memory_changed")
            self._branch = None
            self._anchors = set()
        self._current_attempt = None
        return result
