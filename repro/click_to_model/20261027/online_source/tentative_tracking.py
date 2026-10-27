"""Neural continuation of an unresolved initial pose, without trusted memory.

This advances one working representative for output continuity. It does not
propagate, reject, or resolve the other initial hypotheses. Even repeated
geometry support can never turn this class into a trusted tracker.
"""
from copy import deepcopy
import json

import numpy as np

from adaptive_refinement import AdaptiveRefinementConfig, AdaptiveRefinementPolicy
from rgbd_seed import _pose


class TentativeStepError(RuntimeError):
    """An attempted frame failed; preserve all actual returns before the error."""
    def __init__(self, error, raw_returns, diagnostics):
        super().__init__(f'{type(error).__name__}: {error}')
        self.raw_returns = tuple(np.asarray(value).copy() for value in raw_returns)
        self.diagnostics = dict(diagnostics)


class TentativePoseTracker:
    def __init__(self, checker, diameter, *, admission_id, refinement_config=None,
                 max_translation_diameter=.35, max_rotation_rad=.8):
        if not isinstance(admission_id, str) or not admission_id.strip():
            raise ValueError('unresolved admission ID required')
        values = [diameter, max_translation_diameter, max_rotation_rad]
        if any(isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('finite positive diameter and working motion bounds required')
        self.checker, self.diameter, self._admission_id = checker, float(diameter), admission_id
        self.policy = AdaptiveRefinementPolicy(refinement_config or AdaptiveRefinementConfig())
        self.max_translation_diameter, self.max_rotation_rad = float(max_translation_diameter), float(max_rotation_rad)
        self._working_pose = None
        self._frame = 0
        self._last_raw = ()

    @property
    def trusted_pose(self):
        return None

    @property
    def admission_id(self):
        return self._admission_id

    @property
    def working_pose(self):
        return None if self._working_pose is None else self._working_pose.copy()

    @property
    def last_raw_returns(self):
        return tuple(value.copy() for value in self._last_raw)

    def configuration(self):
        return deepcopy(dict(class_name=type(self).__name__, admission_id=self.admission_id,
            refinement=self.policy.configuration(), max_translation_diameter=self.max_translation_diameter,
            max_rotation_rad=self.max_rotation_rad, trusted_memory_enabled=False,
            alternative_hypotheses='unresolved; not propagated, discarded or resolved by this continuation'))

    @staticmethod
    def _record(value):
        if isinstance(value, np.ndarray):
            return TentativePoseTracker._record(value.tolist())
        if isinstance(value, np.generic):
            return TentativePoseTracker._record(value.item())
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError('diagnostic keys must be strings')
            return {key: TentativePoseTracker._record(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [TentativePoseTracker._record(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _valid_pose(pose):
        try:
            value = _pose(pose)
            return bool(value[2, 3] > 0)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _inputs(rgb, depth, K, mask):
        rgb, depth, K = map(np.asarray, (rgb, depth, K))
        if (rgb.dtype != np.uint8 or depth.ndim != 2 or min(depth.shape, default=0) <= 0 or depth.dtype.kind not in 'fi'
                or rgb.shape != (*depth.shape, 3) or K.shape != (3, 3)
                or K.dtype.kind not in 'fiu' or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.array_equal(K[2], [0, 0, 1]) or np.linalg.matrix_rank(K) != 3):
            raise ValueError('registered RGB uint8, metric depth and calibrated K required')
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != depth.shape or mask.dtype.kind not in 'bui' or not np.isin(mask, [0, 1, 255]).all():
                raise ValueError('registered binary mask required')

    def _flags(self):
        return dict(status='tentative_initialization', admission_id=self.admission_id,
            verified=False, initial_orientation_resolved=False, identity_verified=False,
            trusted_memory_updated=False, memory_updated=False, seed_admitted=False,
            identity_status='unresolved_initial_hypotheses', source='tentative_last_neural_output',
            verified_semantics='unresolved initialization; local support cannot resolve untested alternatives',
            frame_id=self._frame)

    def _evaluate(self, pose, rgb, depth, K, mask):
        # Independent snapshots prevent a checker bug from altering either the
        # actual neural return or the caller's current observation.
        snapshots = [np.asarray(value).copy() for value in (pose, rgb, depth, K)]
        mask_snapshot = None if mask is None else np.asarray(mask).copy()
        arguments = [value.copy() for value in snapshots]
        mask_argument = None if mask_snapshot is None else mask_snapshot.copy()
        evidence = self.checker.evaluate(*arguments, mask=mask_argument)
        if any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(arguments, snapshots)) or (
                mask_snapshot is not None and not np.array_equal(mask_argument, mask_snapshot)):
            raise ValueError('observation checker modified its input')
        if not isinstance(evidence, dict):
            raise TypeError('checker must return diagnostics')
        evidence = deepcopy(evidence)
        value = evidence.get('observation_valid', False)
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError('observation_valid must be a boolean')
        evidence['observation_valid'] = bool(value)
        for key in ('depth_support_fraction', 'contradiction_fraction'):
            if key in evidence:
                value = evidence[key]
                if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                        or not np.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(key+' must be finite in [0,1]')
                evidence[key] = float(value)
        for key in ('status', 'texture_level'):
            if key in evidence and not isinstance(evidence[key], str):
                raise ValueError(key+' must be a string')
        if 'rotation_ambiguous' in evidence and not isinstance(evidence['rotation_ambiguous'], (bool, np.bool_)):
            raise ValueError('rotation_ambiguous must be a boolean')
        evidence = self._record(evidence)
        json.dumps(evidence, allow_nan=False)  # Validate before any working-state commit.
        return evidence

    def reset(self, pose, rgb, depth, K, mask):
        if self._working_pose is not None:
            raise RuntimeError('a tentative continuation is single-admission; construct a fresh instance')
        self._inputs(rgb, depth, K, mask)
        if not self._valid_pose(pose):
            raise ValueError('finite positive-depth original working SE3 pose required')
        original_pose = np.asarray(pose, dtype=float).copy()
        # Calibrating compute from local evidence does not establish orientation.
        try:
            evidence = self._evaluate(original_pose, rgb, depth, K, mask)
        except Exception as error:
            evidence = dict(observation_valid=False, status='initial_check_failure',
                            error=f'{type(error).__name__}: {error}')
        result = dict(self._flags(), evidence=evidence, initial_anchor_pixels=0,
                      observation_valid=evidence['observation_valid'])
        self.policy.calibrate(evidence)
        self._working_pose = original_pose
        return result

    def step(self, rgb, depth, K, refine_one, mask=None):
        if self._working_pose is None:
            raise RuntimeError('reset must precede step')
        self._inputs(rgb, depth, K, mask)
        self._frame += 1
        frame_seed = self._working_pose.copy()
        seed, raw, evidence_rows = frame_seed.copy(), [], []
        self._last_raw = ()
        check_error = None
        conversion_failure = None
        try:
            while True:
                more, stop_reason = self.policy.decide(evidence_rows)
                if not more:
                    break
                # Copy and retain the actual return before checking shape or evidence.
                network_return = refine_one(seed.copy())
                try:
                    returned = np.asarray(network_return).copy()
                except Exception:
                    conversion_failure = dict(return_type=type(network_return).__name__,
                        raw_numeric_return_preserved=False,
                        reason='return could not be represented as a numeric array; no fabricated pose')
                    raise
                raw.append(returned)
                self._last_raw = tuple(value.copy() for value in raw)
                if returned.shape != (4, 4) or returned.dtype.kind not in 'fi':
                    raise ValueError('refiner must return a numeric 4x4 centered pose')
                pose = returned.astype(float, copy=False)
                rigid = self._valid_pose(pose)
                if not rigid:
                    evidence = dict(status='invalid_pose', observation_valid=False,
                                    reason='raw_neural_output_violates_se3')
                    evidence_rows.append(evidence)
                    stop_reason = 'invalid_neural_output'
                    break
                try:
                    evidence = self._evaluate(pose, rgb, depth, K, mask)
                except Exception as error:
                    check_error = f'{type(error).__name__}: {error}'
                    evidence_rows.append(dict(status='check_failure', observation_valid=False, error=check_error))
                    stop_reason = 'observation_check_failure'
                    break
                evidence_rows.append(evidence)
                seed = pose.copy()
            output = raw[-1]
            finite = self._valid_pose(output)
            work_updated = False
            staged_working_pose = frame_seed.copy()
            innovation = dict(translation_m=None, rotation_rad_mod_symmetry=None)
            if finite and check_error is None:
                pose_argument, reference_argument = output.copy(), frame_seed.copy()
                internal = np.asarray(self.checker.equivalent_pose(pose_argument, reference_argument), dtype=float).copy()
                if (not np.array_equal(pose_argument, output) or not np.array_equal(reference_argument, frame_seed)):
                    raise ValueError('symmetry checker modified its input')
                if self._valid_pose(internal):
                    delta = float(np.linalg.norm(internal[:3, 3]-frame_seed[:3, 3]))
                    angle = float(np.arccos(np.clip((np.trace(internal[:3, :3] @ frame_seed[:3, :3].T)-1)/2, -1, 1)))
                    innovation = dict(translation_m=delta, rotation_rad_mod_symmetry=angle)
                    if delta <= self.diameter*self.max_translation_diameter and angle <= self.max_rotation_rad:
                        staged_working_pose = internal.copy()
                        work_updated = True
            diagnostics = dict(self._flags(), neural_iterations=len(raw), evidence=evidence_rows[-1],
                observation_valid=bool(evidence_rows[-1].get('observation_valid', False)),
                iteration_evidence=evidence_rows, iteration_sources=['tentative_continuation']*len(raw),
                output_iteration_index=len(raw)-1, stop_reason=stop_reason,
                numerically_valid=finite, working_seed_updated=work_updated, innovation=innovation,
                lost_frames=self._frame, recovery=None, replenishment=None, view_admission=None)
            result = dict(pose=output.copy(), raw_iterations=np.asarray(raw), diagnostics=diagnostics)
            # Commit working state only after all callbacks, checks and output
            # construction succeed. This still never creates trusted state.
            self._working_pose = staged_working_pose
            return result
        except Exception as error:
            diagnostics = dict(self._flags(), status='tentative_step_failed', neural_iterations=len(raw),
                               iteration_evidence=evidence_rows, working_seed_updated=False,
                               conversion_failure=conversion_failure)
            raise TentativeStepError(error, raw, diagnostics) from error
