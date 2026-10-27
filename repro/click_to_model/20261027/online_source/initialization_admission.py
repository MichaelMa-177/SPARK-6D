"""Pure, finite-set assessment of already returned initialization hypotheses.

No registration, tracking, model loading or memory writes occur here. Scores
choose a representative only after every alternative is accounted for. Missing
depth and skipped work cannot establish unique orientation or instance identity.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
from collections.abc import Mapping
from types import MappingProxyType
from numbers import Real
import hashlib
import json
import time

import numpy as np

from pose_validity import PoseValidityChecker
from rgbd_seed import _pose


SCOPE = "conditional uniqueness within the supplied finite candidate set and declared symmetry; not object-instance identity"


def _freeze(value):
    if isinstance(value, np.ndarray):
        out = value.copy()
        out.setflags(write=False)
        return out
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _numeric(value):
    out = np.asarray(value).copy()
    if out.dtype.kind not in 'buif':
        raise ValueError('raw returned arrays must be numeric; no object serialization')
    out.setflags(write=False)
    return out


def _optional_numeric(value):
    return None if value is None else _numeric(value)


def _record(value):
    if is_dataclass(value):
        return {f.name: _record(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return _record(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record(item) for item in value]
    if isinstance(value, np.generic):
        return _record(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {'unsupported_type': type(value).__module__+'.'+type(value).__qualname__}


def _error_text(error):
    try:
        return type(error).__name__+': '+str(error)
    except Exception:
        return type(error).__name__+': exception text unavailable'


def _checked_evidence(value, depth=0):
    """Reject unsealable diagnostics at the callback boundary, including cycles."""
    if depth > 32:
        raise ValueError('geometry diagnostic nesting/cycle exceeds 32 levels')
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'buif' or not np.isfinite(value).all():
            raise ValueError('geometry diagnostic arrays must be finite numeric values')
        return value.copy()
    if isinstance(value, np.generic):
        return _checked_evidence(value.item(), depth+1)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('geometry diagnostic keys must be strings')
        return {key: _checked_evidence(item, depth+1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_checked_evidence(item, depth+1) for item in value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('geometry diagnostic numbers must be finite')
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('unsupported geometry diagnostic type '+type(value).__module__+'.'+type(value).__qualname__)


def _digest(value):
    """Bind array dtype/shape/bytes, including NaN payloads; not JSON rounding."""
    digest = hashlib.sha256()
    def visit(item):
        if is_dataclass(item):
            visit({f.name: getattr(item, f.name) for f in fields(item) if f.name != 'content_digest'})
        elif isinstance(item, np.ndarray):
            digest.update(b'array:')
            visit((item.dtype.str, item.shape))
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, Mapping):
            digest.update(b'mapping:')
            for key in sorted(item):
                visit(str(key)); visit(item[key])
            digest.update(b'end')
        elif isinstance(item, (tuple, list)):
            digest.update(b'sequence:')
            for child in item:
                visit(child)
            digest.update(b'end')
        else:
            if isinstance(item, np.generic):
                item = item.item()
            if isinstance(item, float) and not np.isfinite(item):
                digest.update(b'nonfinite-float:'+np.asarray(item, dtype=np.float64).tobytes())
                return
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
            digest.update(b';')
    visit(value)
    return digest.hexdigest()


def _observation(rgb, depth, K, mask):
    rgb, depth, K, mask = map(np.asarray, (rgb, depth, K, mask))
    if (rgb.dtype != np.uint8 or depth.ndim != 2 or min(depth.shape, default=0) < 1
            or depth.dtype.kind not in 'fiu' or rgb.shape != (*depth.shape, 3)
            or mask.shape != depth.shape or mask.dtype.kind not in 'buif'
            or not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 255]).all()
            or not (mask != 0).any() or K.shape != (3, 3) or K.dtype.kind not in 'fiu' or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or abs(np.linalg.det(K)) < 1e-12
            or not np.allclose(K[2], [0, 0, 1], atol=1e-7, rtol=0)):
        raise ValueError('calibrated RGB, metric depth and nonempty binary mask required')
    return tuple(_freeze(item) for item in (rgb, depth, K, mask != 0))


def checker_fingerprint(checker, cad_identity):
    """Bind actual sampled geometry, declared symmetry and checker settings."""
    if not isinstance(checker, PoseValidityChecker) or not isinstance(cad_identity, str) or not cad_identity:
        raise ValueError('PoseValidityChecker and explicit CAD asset identity required')
    return _digest(dict(cad_identity=cad_identity,
        checker_type=type(checker).__module__+'.'+type(checker).__qualname__,
        points=np.asarray(checker.points), normals=None if checker.normals is None else np.asarray(checker.normals),
        configuration=checker.configuration(), diameter=checker.diameter, tolerance=checker.tolerance,
        discrete=checker.symmetries.discrete, continuous=checker.symmetries.continuous,
        symmetry_steps=checker._symmetry_steps, symmetry_radius=checker._symmetry_radius))


@dataclass(frozen=True)
class InitializationAdmissionConfig:
    max_candidates: int = 1024
    max_equivalence_checks: int = 65536
    max_seconds: float = 30.
    equivalent_translation_m: float = .015
    equivalent_rotation_rad: float = .15

    def __post_init__(self):
        for key in ('max_candidates', 'max_equivalence_checks'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key+' must be a positive integer')
        for key in ('max_seconds', 'equivalent_translation_m', 'equivalent_rotation_rad'):
            value = getattr(self, key)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(key+' must be finite positive')


@dataclass(frozen=True)
class RegistrationHypothesisBundle:
    session_id: str
    frame_id: int
    mask_id: int
    poses_centered: np.ndarray | None
    scores: np.ndarray | None
    returned_pose_original: np.ndarray | None
    centered_from_original: np.ndarray
    selected_backend_index: int | None
    order: str
    source_observation_digest: str
    checker_digest: str
    error: str | None
    content_digest: str

    @classmethod
    def capture(cls, *, session_id, frame_id, mask_id, poses_centered, scores,
                returned_pose_original, centered_from_original, selected_backend_index,
                rgb, depth, K, mask, checker, cad_identity,
                order='est.poses score order', error=None):
        if (not isinstance(session_id, str) or not session_id or type(frame_id) is not int
                or frame_id < 0 or type(mask_id) is not int or mask_id < 0
                or not isinstance(order, str) or not order
                or (error is not None and not isinstance(error, str))):
            raise ValueError('explicit session/frame/mask/order required')
        observation = _observation(rgb, depth, K, mask)
        bundle = cls(session_id, frame_id, mask_id, _optional_numeric(poses_centered), _optional_numeric(scores),
            _optional_numeric(returned_pose_original), _numeric(centered_from_original), selected_backend_index,
            order, _digest(observation), checker_fingerprint(checker, cad_identity), error, '')
        return replace(bundle, content_digest=_digest(bundle))


@dataclass(frozen=True)
class InitialCandidateAssessment:
    index: int
    mask_id: int
    backend_index: int
    pose_centered: np.ndarray
    score: float | None
    pose_original: np.ndarray | None = None
    status: str = 'unchecked'
    masked_evidence: Mapping | None = None
    unmasked_evidence: Mapping | None = None
    error: str | None = None


@dataclass(frozen=True)
class InitializationAdmission:
    session_id: str
    frame_id: int
    accepted: bool
    reason: str
    working_index: int | None
    trusted_index: int | None
    bundles: tuple
    candidates: tuple
    pair_checks: tuple
    diagnostics: Mapping
    checker_digest: str | None
    observation_digests: Mapping
    config: InitializationAdmissionConfig
    identity_verified: bool = False
    hypothesis_scope: str = SCOPE
    content_digest: str = ''

    @property
    def trusted_candidate(self):
        return None if self.trusted_index is None else self.candidates[self.trusted_index]

    def as_record(self):
        return _record(self)

    def matches_observation(self, *, session_id, frame_id, rgb, depth, K, masks, checker, cad_identity):
        """Integrity/context check only; consuming a controller transaction is external."""
        try:
            return bool(self.content_digest == _digest(self)
                and isinstance(session_id, str) and type(frame_id) is int
                and isinstance(masks, Mapping) and all(type(key) is int and key >= 0 for key in masks)
                and session_id == self.session_id and frame_id == self.frame_id
                and self.checker_digest == checker_fingerprint(checker, cad_identity)
                and dict(self.observation_digests) == {
                    str(key): _digest(_observation(rgb, depth, K, value)) for key, value in masks.items()})
        except Exception:
            return False


def _seal_result(result):
    """A final serialization failure must retain raw arrays and reject admission."""
    try:
        return replace(result, content_digest=_digest(result))
    except Exception as error:
        rejected = replace(result, accepted=False, trusted_index=None, reason='incomplete_assessment',
            content_digest='', diagnostics=_freeze({**dict(result.diagnostics),
                'errors': tuple(result.diagnostics['errors'])+('content seal: '+_error_text(error),)}))
        try:
            return replace(rejected, content_digest=_digest(rejected))
        except Exception:
            return rejected


class _Deadline:
    @staticmethod
    def _read(clock):
        value = clock()
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError('assessment clock must return a finite real number, not bool or a coercible value')
        return float(value)

    def __init__(self, clock, seconds):
        self.clock, self.seconds = clock, seconds
        self.start = self.last = self._read(clock)

    def check(self):
        now = self._read(self.clock)
        if now < self.last:
            raise ValueError('nonfinite or nonmonotonic assessment clock')
        self.last = now
        if now-self.start > self.seconds:
            raise TimeoutError('initialization assessment deadline exceeded')
        return now-self.start


def _strong_contradiction(evidence, checker):
    """A fixed free-space rejection; `observation_valid=False` alone is not it."""
    if evidence.get('status') != 'contradicted' or evidence.get('observation_valid') is not False:
        return False
    try:
        count = int(evidence['projected_count'])
        fraction = float(evidence['contradiction_fraction'])
        valid = float(evidence['valid_depth_fraction'])
        inliers = int(evidence['inlier_count'])
        return bool(count >= checker.config.min_inliers and 0 <= inliers <= count
            and np.isfinite([fraction, valid]).all() and 0 <= fraction <= 1 and 0 <= valid <= 1
            and fraction > checker.config.max_contradiction_fraction
            and valid >= checker.config.min_depth_support_fraction and fraction*count > inliers)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def assess_initial_hypotheses(bundles, *, session_id, frame_id, rgb, depth, K, masks,
                             checker, cad_identity, config=None, clock=None):
    """Assess the complete supplied mask × returned-pose set, without inference.

    Budgets include copying, binding, all checks and result construction. A late
    callback is retained but cannot produce an admission. Raw bundles are never
    truncated to satisfy a budget. No mutable tracker/seed object is accepted.
    """
    config = InitializationAdmissionConfig() if config is None else config
    if not isinstance(config, InitializationAdmissionConfig):
        raise TypeError('config must be InitializationAdmissionConfig')
    deadline = None
    start_error = None
    try:
        deadline = _Deadline(time.monotonic if clock is None else clock, config.max_seconds)
    except Exception as error:
        start_error = error
    bundles = tuple(bundles)
    if any(not isinstance(bundle, RegistrationHypothesisBundle) for bundle in bundles):
        raise TypeError('captured registration bundles required')
    candidates, pairs, observations, errors = [], [], {}, []
    checker_digest = None
    reason, working, trusted, elapsed = None, None, None, None
    complete, equivalence_complete = False, False
    # Copy the original numeric returns before trying any fallible checker.
    kept = tuple(replace(bundle, poses_centered=_optional_numeric(bundle.poses_centered), scores=_optional_numeric(bundle.scores),
        returned_pose_original=_optional_numeric(bundle.returned_pose_original),
        centered_from_original=_numeric(bundle.centered_from_original)) for bundle in bundles)
    for bundle in kept:
        if bundle.poses_centered is not None and bundle.poses_centered.ndim:
            for index, pose in enumerate(bundle.poses_centered):
                score = (float(bundle.scores[index]) if bundle.scores is not None and bundle.scores.ndim == 1 and index < len(bundle.scores)
                         and np.isfinite(bundle.scores[index]) else None)
                candidates.append(InitialCandidateAssessment(len(candidates), bundle.mask_id, index, _freeze(pose), score))
    try:
        if start_error is not None:
            raise start_error
        deadline.check()
        if (not isinstance(session_id, str) or not session_id or type(frame_id) is not int or frame_id < 0
                or not isinstance(masks, Mapping) or not masks
                or any(type(key) is not int or key < 0 for key in masks)):
            raise ValueError('explicit session, frame and complete integer-keyed mask mapping required')
        checker_digest = checker_fingerprint(checker, cad_identity)
        snapshots = {key: _observation(rgb, depth, K, value) for key, value in masks.items()}
        observations = {str(key): _digest(value) for key, value in snapshots.items()}
        if len(kept) != len(masks) or {b.mask_id for b in kept} != set(masks):
            raise ValueError('one returned bundle per supplied initialization mask is required')
        transforms, usable_by_mask = [], {}
        for bundle in kept:
            deadline.check()
            if (bundle.content_digest != _digest(bundle) or bundle.session_id != session_id or bundle.frame_id != frame_id
                    or bundle.source_observation_digest != observations[str(bundle.mask_id)]
                    or bundle.checker_digest != checker_digest):
                raise ValueError('registration bundle content/observation/session/CAD binding failed')
            if bundle.error:
                errors.append('backend error: '+bundle.error)
            poses, scores = bundle.poses_centered, bundle.scores
            if (poses is None or scores is None or poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses)
                    or scores.shape != (len(poses),) or not np.isfinite(scores).all()):
                errors.append('mask '+str(bundle.mask_id)+': incomplete or malformed complete poses/scores')
                continue
            transform = _pose(bundle.centered_from_original)
            transforms.append(transform)
            selected = bundle.selected_backend_index
            if (type(selected) is not int or not 0 <= selected < len(poses)
                    or scores[selected] != scores.max() or bundle.returned_pose_original is None
                    or bundle.returned_pose_original.shape != (4, 4)
                    or not np.allclose(bundle.returned_pose_original, poses[selected].astype(float)@transform,
                                       atol=2e-7, rtol=0, equal_nan=False)):
                errors.append('mask '+str(bundle.mask_id)+': returned pose/selected score/center transform binding failed')
                continue
            usable_by_mask[bundle.mask_id] = bundle
        if any(not np.array_equal(transforms[0], transform) for transform in transforms[1:]):
            raise ValueError('mask registrations disagree on centered CAD transform')
        if len(candidates) > config.max_candidates:
            reason = 'candidate_budget_exhausted'
        else:
            for index, candidate in enumerate(candidates):
                deadline.check()
                bundle = usable_by_mask.get(candidate.mask_id)
                if bundle is None:
                    continue
                try:
                    pose = _pose(candidate.pose_centered)
                    if pose[2, 3] <= 0:
                        raise ValueError('candidate center is not in front of camera')
                except (ValueError, TypeError):
                    candidates[index] = replace(candidate, status='invalid_pose', error='invalid positive-depth SE3')
                    continue
                original = _freeze(pose @ bundle.centered_from_original)
                values, failures = [], []
                for masked in (True, False):
                    deadline.check()
                    image, metric_depth, intrinsics, foreground = snapshots[candidate.mask_id]
                    work = [item.copy() for item in (pose, image, metric_depth, intrinsics, foreground)]
                    evidence = None
                    try:
                        evidence = checker.evaluate(work[0], work[1], work[2], work[3],
                            **({'mask': work[4]} if masked else {}))
                        if (not isinstance(evidence, dict) or type(evidence.get('observation_valid')) is not bool
                                or not isinstance(evidence.get('status'), str)
                                or (evidence['observation_valid'] and evidence['status'] != 'supported')):
                            raise ValueError('checker must return explicit observation_valid diagnostics')
                        evidence = _checked_evidence(evidence)
                        json.dumps(_record(evidence), allow_nan=False)
                        if any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(
                                work, (pose, image, metric_depth, intrinsics, foreground))):
                            raise ValueError('checker mutated candidate/observation input')
                    except Exception as error:
                        message = ('masked' if masked else 'unmasked')+': '+_error_text(error)
                        failures.append(message)
                        # An arbitrary auxiliary Python object cannot become a
                        # serialized supported observation. Preserve a failure
                        # record and every numeric pose/score instead.
                        if evidence is not None:
                            evidence = dict(status='check_error', observation_valid=False, error=message)
                    values.append(_freeze(evidence) if isinstance(evidence, dict) else None)
                    # Preserve the returned evidence before checking a late callback.
                    candidates[index] = replace(candidate, pose_original=original, status='partial',
                        masked_evidence=values[0], unmasked_evidence=values[1] if len(values)>1 else None,
                        error='; '.join(failures) or None)
                    deadline.check()
                if failures:
                    status = 'check_error'; errors.extend(failures)
                elif all(value.get('observation_valid') is True for value in values):
                    status = 'supported'
                elif all(_strong_contradiction(value, checker) for value in values):
                    status = 'contradicted'
                else:
                    status = 'inconclusive'
                candidates[index] = replace(candidates[index], status=status)
            complete = all(c.status not in ('unchecked', 'partial', 'check_error') for c in candidates)
            supported = [candidate for candidate in candidates if candidate.status == 'supported']
            ambiguous = False
            exhausted = False
            for i, left in enumerate(supported):
                for right in supported[i+1:]:
                    if len(pairs) >= config.max_equivalence_checks:
                        exhausted = True
                        break
                    deadline.check()
                    moving, reference = left.pose_centered.copy(), right.pose_centered.copy()
                    aligned = _pose(checker.equivalent_pose(moving, reference))
                    if not np.array_equal(moving, left.pose_centered) or not np.array_equal(reference, right.pose_centered):
                        raise ValueError('symmetry checker mutated candidate inputs')
                    dt = float(np.linalg.norm(aligned[:3, 3]-right.pose_centered[:3, 3]))
                    dr = float(np.arccos(np.clip((np.trace(aligned[:3, :3]@right.pose_centered[:3, :3].T)-1)/2,-1,1)))
                    equivalent = dt <= config.equivalent_translation_m and dr <= config.equivalent_rotation_rad
                    pairs.append(_freeze(dict(left=left.index, right=right.index, translation_m=dt,
                                              rotation_rad_mod_symmetry=dr, equivalent=equivalent)))
                    ambiguous |= not equivalent
                    deadline.check()
                if exhausted:
                    break
            equivalence_complete = not exhausted
            if errors or not complete:
                reason = 'incomplete_assessment'
            elif exhausted:
                reason = 'equivalence_budget_exhausted'
            elif ambiguous:
                reason = 'ambiguous_supported_set'
            elif any(candidate.status == 'inconclusive' for candidate in candidates):
                reason = 'inconclusive_observation'
            elif not supported:
                reason = 'no_supported_candidate'
            else:
                reason = 'unique_supported_finite_set'
            # A tentative display representative never authorizes trusted state.
            if supported:
                working = max(supported, key=lambda c: (c.score, -c.index)).index
            if reason == 'unique_supported_finite_set':
                trusted = working
        if checker_fingerprint(checker, cad_identity) != checker_digest:
            raise ValueError('checker geometry/symmetry/config changed during assessment')
        elapsed = deadline.check()
    except Exception as error:
        reason = 'assessment_timeout' if isinstance(error, TimeoutError) else 'incomplete_assessment'
        errors.append(_error_text(error))
        trusted = None
    diagnostics = _freeze(dict(candidate_count=len(candidates), outputs_truncated=False,
        planned_mask_ids=tuple(masks) if isinstance(masks, Mapping) else (),
        returned_bundle_mask_ids=tuple(bundle.mask_id for bundle in kept),
        candidate_checks_complete=complete, equivalence_checks_complete=equivalence_complete,
        equivalence_checks=len(pairs), expected_supported_pairs=sum(c.status=='supported' for c in candidates)*(sum(c.status=='supported' for c in candidates)-1)//2,
        elapsed_seconds=elapsed, elapsed_seconds_scope='before result construction and content seal; final budget check includes both',
        errors=errors, no_tracker_or_memory_writes=True,
        timing='assessment budget includes construction/seal; caller measures full wall time separately; prior registration excluded'))
    result = InitializationAdmission(session_id, frame_id, trusted is not None, reason or 'incomplete_assessment',
        working, trusted, kept, tuple(candidates), tuple(pairs), diagnostics,
        checker_digest, _freeze(observations), config)
    result = _seal_result(result)
    # Include construction/sealing in the budget. Clock errors after apparent
    # success must also remove admission; no final unguarded clock invocation.
    if deadline is not None:
        try:
            deadline.check()
            # Return the already sealed object unchanged. A second digest or
            # reconstruction after this check would reopen the deadline hole.
        except Exception as error:
            reason = 'assessment_timeout' if isinstance(error, TimeoutError) else 'incomplete_assessment'
            result = replace(result, accepted=False, trusted_index=None, reason=reason,
                diagnostics=_freeze({**dict(result.diagnostics), 'errors': tuple(result.diagnostics['errors'])+(_error_text(error),),
                                     'elapsed_seconds': deadline.last-deadline.start,
                                     'elapsed_seconds_scope': 'last valid clock reading on rejected assessment'}))
            result = _seal_result(result)
    return result
