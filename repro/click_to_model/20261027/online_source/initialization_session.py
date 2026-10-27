"""One-use installation of a bound initialization admission into a fresh tracker.

An accepted finite-set assessment may create trusted correspondence memory.
Unresolved, but correctly bound, evidence creates only tentative continuation.
The owner is consumed on its first attempt, including a failed attempt; a new
experiment requires a new owner. No tracker is published before staging passes.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import weakref

import numpy as np

from adaptive_refinement import AdaptiveRefinementConfig, AdaptiveRefinementPolicy
from click_to_model import ClickToModelConfig, ClickToModelTracker
from initialization_admission import (InitializationAdmission, InitialCandidateAssessment,
    RegistrationHypothesisBundle, _digest, _observation, checker_fingerprint, _strong_contradiction)
from rgbd_seed import RGBDSeedTracker, _pose
from tentative_tracking import TentativePoseTracker


_RESERVED_SEEDS = weakref.WeakSet()


@dataclass(frozen=True)
class InitializationWorkingFallback:
    mask_id: int
    pose_original: np.ndarray

    def __post_init__(self):
        if type(self.mask_id) is not int or self.mask_id < 0:
            raise ValueError('fallback mask ID must be a nonnegative integer')
        value = np.asarray(self.pose_original).copy()
        if value.dtype.kind not in 'fi' or value.shape != (4, 4):
            raise ValueError('raw numeric original-frame top-1 pose required')
        _pose(value)
        value.setflags(write=False)
        object.__setattr__(self, 'pose_original', value)


def _positive_pose(value):
    value = np.asarray(value)
    if value.dtype.kind not in 'fi' or value.shape != (4, 4):
        raise ValueError('numeric raw 4x4 pose required')
    checked = _pose(value)
    if checked[2, 3] <= 0:
        raise ValueError('positive-depth working pose required')
    return value.copy()


def _callback_id(value):
    return (id(getattr(value, '__self__', None)), id(getattr(value, '__func__', value)))


class InitializationTrackingSession:
    def __init__(self, session_id, checker, diameter, cad_identity, tracking_config=None,
                 refinement_config=None, seed_factory=None, anchor_model_points=None):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('explicit session ID required')
        if isinstance(diameter, (bool, np.bool_)) or not np.isfinite(diameter) or diameter <= 0:
            raise ValueError('finite positive metric diameter required')
        self._session_id, self._cad_identity, self._checker = session_id, cad_identity, checker
        self._diameter = float(diameter)
        self._tracking_config = ClickToModelConfig() if tracking_config is None else tracking_config
        self._refinement_config = AdaptiveRefinementConfig() if refinement_config is None else refinement_config
        if type(self._tracking_config) is not ClickToModelConfig or type(self._refinement_config) is not AdaptiveRefinementConfig:
            raise TypeError('typed tracking/refinement configurations required')
        if seed_factory is not None and not callable(seed_factory):
            raise TypeError('seed_factory must construct a fresh provider; an existing provider is not accepted')
        self._seed_factory = seed_factory
        self._anchors = np.asarray(checker.points if anchor_model_points is None else anchor_model_points, dtype=float).copy()
        if self._anchors.ndim != 2 or self._anchors.shape[1] != 3 or len(self._anchors) < 3 or not np.isfinite(self._anchors).all():
            raise ValueError('finite metric CAD anchor points required')
        self._anchors.setflags(write=False)
        self._checker_digest = checker_fingerprint(checker, cad_identity)
        if not np.isclose(self._diameter, checker.diameter, atol=1e-12, rtol=0):
            raise ValueError('session diameter must equal the bound checker CAD diameter')
        self._context = self._context_digest()
        self._attempted = False
        self._tracker = None
        self._initial_diagnostics = None
        self._selected = None

    @property
    def session_id(self): return self._session_id

    @property
    def attempted(self): return self._attempted

    @property
    def tracker(self):
        if self._tracker is None:
            raise RuntimeError('no completely installed tracker is available')
        return self._tracker

    @property
    def initial_diagnostics(self): return deepcopy(self._initial_diagnostics)

    @property
    def selected_index(self): return None if self._selected is None else self._selected['index']

    @property
    def selected_mask_id(self): return None if self._selected is None else self._selected['mask_id']

    @property
    def selected_backend_index(self): return None if self._selected is None else self._selected['backend_index']

    @property
    def selected_pose_centered(self): return None if self._selected is None else self._selected['pose_centered'].copy()

    @property
    def selected_pose_original(self): return None if self._selected is None else self._selected['pose_original'].copy()

    @property
    def selected_source(self): return None if self._selected is None else self._selected['source']

    @property
    def selected_mask(self): return None if self._selected is None else self._selected['mask'].copy()

    def _context_digest(self):
        return _digest(dict(session_id=self._session_id, cad_identity=self._cad_identity,
            checker=checker_fingerprint(self._checker, self._cad_identity), diameter=self._diameter,
            tracking=self._tracking_config, refinement=self._refinement_config, anchors=self._anchors,
            seed_factory=id(self._seed_factory),
            callbacks=[_callback_id(getattr(self._checker, key)) for key in ('evaluate', 'equivalent_pose', 'project_visible')]))

    def _guard_context(self):
        if self._context_digest() != self._context:
            raise ValueError('session/checker/CAD/configuration context changed')

    def _bind(self, admission, frame_id, rgb, depth, K, masks):
        if type(admission) is not InitializationAdmission:
            raise TypeError('a typed InitializationAdmission is required')
        if type(admission.accepted) is not bool or admission.identity_verified is not False:
            raise ValueError('admission flags must retain exact finite-set/identity semantics')
        if not admission.matches_observation(session_id=self._session_id, frame_id=frame_id,
                rgb=rgb, depth=depth, K=K, masks=masks, checker=self._checker, cad_identity=self._cad_identity):
            raise ValueError('admission does not match session/frame/observation/CAD/configuration')
        if (not admission.bundles or len(admission.bundles) != len(masks)
                or any(type(b) is not RegistrationHypothesisBundle for b in admission.bundles)
                or {b.mask_id for b in admission.bundles} != set(masks)):
            raise ValueError('complete uniquely bound mask/bundle set required even for tentative use')
        by_mask = {}
        for bundle in admission.bundles:
            if (bundle.content_digest != _digest(bundle) or bundle.session_id != self._session_id
                    or bundle.frame_id != frame_id or bundle.checker_digest != self._checker_digest
                    or bundle.source_observation_digest != admission.observation_digests[str(bundle.mask_id)]):
                raise ValueError('raw registration bundle binding failed; no tentative bypass')
            _pose(bundle.centered_from_original)
            by_mask[bundle.mask_id] = bundle
        transforms = [b.centered_from_original for b in admission.bundles]
        if any(not np.array_equal(transforms[0], value) for value in transforms[1:]):
            raise ValueError('mask bundles disagree on CAD centering')
        # The admission seal detects mutation, but is not an authority to drop
        # competitors and reseal a smaller candidate set. Verify the full raw
        # flattening before either trusted or tentative installation.
        raw = [(bundle, i, pose) for bundle in admission.bundles
               if bundle.poses_centered is not None and bundle.poses_centered.ndim
               for i, pose in enumerate(bundle.poses_centered)]
        if len(raw) != len(admission.candidates) or admission.diagnostics.get('candidate_count') != len(raw):
            raise ValueError('admission candidate set does not cover every raw backend return')
        for index, (bundle, backend, pose) in enumerate(raw):
            candidate = admission.candidates[index]
            score = (float(bundle.scores[backend]) if bundle.scores is not None and bundle.scores.ndim == 1
                     and backend < len(bundle.scores) and np.isfinite(bundle.scores[backend]) else None)
            if (type(candidate) is not InitialCandidateAssessment or type(candidate.index) is not int
                    or candidate.index != index or type(candidate.backend_index) is not int
                    or candidate.backend_index != backend or candidate.mask_id != bundle.mask_id
                    or _digest(candidate.pose_centered) != _digest(pose) or candidate.score != score):
                raise ValueError('admission raw candidate order/pose/score binding differs')
        return by_mask

    @staticmethod
    def _candidate(admission, index, bundles):
        if type(index) is not int or not 0 <= index < len(admission.candidates):
            raise ValueError('valid explicit candidate index required')
        candidate = admission.candidates[index]
        if type(candidate) is not InitialCandidateAssessment or candidate.index != index or candidate.mask_id not in bundles:
            raise ValueError('candidate identity differs from its complete bundle')
        bundle = bundles[candidate.mask_id]
        backend = candidate.backend_index
        if (type(backend) is not int or bundle.poses_centered is None or bundle.poses_centered.ndim != 3
                or bundle.poses_centered.shape[1:] != (4, 4) or not 0 <= backend < len(bundle.poses_centered)
                or not np.array_equal(candidate.pose_centered, bundle.poses_centered[backend])):
            raise ValueError('candidate is not the preserved raw centered backend pose')
        centered = _positive_pose(candidate.pose_centered)
        if candidate.pose_original is None or not np.allclose(candidate.pose_original,
                centered.astype(float) @ bundle.centered_from_original, atol=2e-7, rtol=0):
            raise ValueError('candidate original/centered pose binding differs')
        return dict(index=index, mask_id=candidate.mask_id, backend_index=backend,
                    pose_centered=centered, pose_original=np.asarray(candidate.pose_original).copy(),
                    source='admission_candidate'), candidate

    @staticmethod
    def _fallback(fallback, bundles):
        if type(fallback) is not InitializationWorkingFallback or fallback.mask_id not in bundles:
            raise TypeError('an explicitly typed original-top1 fallback bound to a mask is required')
        bundle = bundles[fallback.mask_id]
        index, poses = bundle.selected_backend_index, bundle.poses_centered
        if (bundle.returned_pose_original is None or not np.array_equal(fallback.pose_original, bundle.returned_pose_original)
                or type(index) is not int or poses is None or poses.ndim != 3 or poses.shape[1:] != (4, 4)
                or not 0 <= index < len(poses) or bundle.scores is None or bundle.scores.shape != (len(poses),)
                or not np.isfinite(bundle.scores).all() or bundle.scores[index] != bundle.scores.max()):
            raise ValueError('fallback must be the captured raw top-1 with its original complete centered pose')
        centered = _positive_pose(poses[index])
        if not np.allclose(fallback.pose_original, centered.astype(float) @ bundle.centered_from_original, atol=2e-7, rtol=0):
            raise ValueError('fallback top-1 original/centered binding failed')
        return dict(index=None, mask_id=fallback.mask_id, backend_index=index,
                    pose_centered=centered, pose_original=fallback.pose_original.copy(), source='bound_original_top1_fallback')

    def _trusted_contract(self, admission, candidate):
        if (admission.reason != 'unique_supported_finite_set' or admission.trusted_index != admission.working_index
                or candidate.status != 'supported'
                or admission.diagnostics.get('candidate_checks_complete') is not True
                or admission.diagnostics.get('equivalence_checks_complete') is not True
                or admission.diagnostics.get('errors')
                or any(c.status not in ('supported','contradicted','invalid_pose') for c in admission.candidates)):
            raise ValueError('accepted admission lacks complete supported finite-set evidence')
        for bundle in admission.bundles:
            if bundle.error is not None:
                raise ValueError('accepted admission contains a failed raw backend bundle')
            poses, scores, index = bundle.poses_centered, bundle.scores, bundle.selected_backend_index
            if (poses is None or poses.ndim != 3 or poses.shape[1:] != (4,4) or not len(poses)
                    or scores is None or scores.shape != (len(poses),) or not np.isfinite(scores).all()
                    or type(index) is not int or not 0 <= index < len(poses) or scores[index] != scores.max()
                    or bundle.returned_pose_original is None or bundle.returned_pose_original.shape != (4,4)
                    or not np.allclose(bundle.returned_pose_original, poses[index].astype(float) @
                        bundle.centered_from_original, atol=2e-7, rtol=0)):
                raise ValueError('accepted admission lacks complete original top-1/backend binding')
        for alternative in admission.candidates:
            if alternative.status == 'invalid_pose':
                try: _positive_pose(alternative.pose_centered)
                except (TypeError, ValueError): continue
                raise ValueError('valid raw alternative was labelled invalid_pose')
            self._candidate(admission, alternative.index, {b.mask_id:b for b in admission.bundles})
            for evidence in (alternative.masked_evidence, alternative.unmasked_evidence):
                if not isinstance(evidence, Mapping):
                    raise ValueError('complete supported or contradicted candidate evidence required')
                if alternative.status == 'supported':
                    if evidence.get('observation_valid') is not True or evidence.get('status') != 'supported':
                        raise ValueError('trusted candidate requires supported masked and unmasked evidence')
                elif not _strong_contradiction(evidence, self._checker):
                    raise ValueError('excluded raw alternative lacks independent geometric contradiction')
        supported = [c.index for c in admission.candidates if c.status == 'supported']
        expected = {(a,b) for i,a in enumerate(supported) for b in supported[i+1:]}
        actual = [(p.get('left'), p.get('right')) for p in admission.pair_checks]
        if len(actual) != len(expected) or set(actual) != expected or any(p.get('equivalent') is not True for p in admission.pair_checks):
            raise ValueError('every supported pair must have explicit equivalence evidence')
        if (admission.diagnostics.get('equivalence_checks') != len(expected)
                or admission.diagnostics.get('expected_supported_pairs') != len(expected)
                or admission.trusted_index != max(supported, key=lambda i:(admission.candidates[i].score,-i))):
            raise ValueError('accepted representative/pair accounting is inconsistent')
        for pair in admission.pair_checks:
            dt, dr = pair.get('translation_m'), pair.get('rotation_rad_mod_symmetry')
            if (type(dt) not in (float,int) or type(dr) not in (float,int)
                    or not np.isfinite([dt,dr]).all() or dt < 0 or dr < 0
                    or dt > admission.config.equivalent_translation_m or dr > admission.config.equivalent_rotation_rad):
                raise ValueError('supported pair lacks finite within-bound equivalence distances')

    def _fresh_seed(self):
        provider = RGBDSeedTracker() if self._seed_factory is None else self._seed_factory()
        if not isinstance(provider, RGBDSeedTracker):
            raise TypeError('fresh transactional RGBDSeedTracker provider required')
        if (provider in _RESERVED_SEEDS or provider.initialized or provider.trusted_pose is not None
                or provider.memory_revision != 0 or provider._generation != 0 or provider._gray is not None
                or provider._pending is not None or len(provider.active_points) or len(provider._anchors)
                or len(provider._orb_anchors) or provider._orb_descriptors is not None
                or getattr(provider, 'trusted_view_inventory', []) or getattr(provider, '_commit_index', 0)):
            raise ValueError('seed provider is already used or initialized; no external memory reuse')
        provider_checker = getattr(provider, 'checker', None)
        if provider_checker is not None and checker_fingerprint(provider_checker, self._cad_identity) != self._checker_digest:
            raise ValueError('seed provider checker differs from bound CAD context')
        _RESERVED_SEEDS.add(provider)
        return provider

    def _provider_context(self, provider):
        return _digest(dict(provider_type=type(provider).__module__+'.'+type(provider).__qualname__,
            config=provider.config, view_config=getattr(provider, 'view_config', None),
            backend_type=type(provider.backend).__module__+'.'+type(provider.backend).__qualname__,
            backend_config=getattr(provider.backend, 'config', None), memory_identity=id(provider.memory_identity),
            reset_callback=_callback_id(provider.reset),
            checker=None if getattr(provider, 'checker', None) is None else
                checker_fingerprint(provider.checker, self._cad_identity)))

    @staticmethod
    def _seed_state(provider):
        return _digest({name:getattr(provider,name) for name in ('_gray','_pose','_pixels','_anchors',
            '_orb_anchors','_orb_descriptors','_mask_area','_pending','_generation','_memory_revision')})

    @staticmethod
    def _empty_view_state(provider):
        return (not getattr(provider,'trusted_view_inventory', [])
            and getattr(provider,'_commit_index',0) == 0
            and not getattr(provider,'_view_available',False)
            and all(getattr(provider,key,None) is None for key in ('_latest_view','_committed_K',
                '_pending_depth','_pending_depth_proposal','_committed_depth','_recovery_rgb','_recovery_details')))

    @staticmethod
    def _recovery_state(recovery):
        if recovery is None: return None
        if recovery.active or recovery._confirmed is not None or recovery._anchors or recovery._trace:
            raise ValueError('initial recovery state must be empty')
        fields = ('_branch','_confirmed','_started','_last_frame','_source','_source_identity',
                  '_source_revision','_current_attempt','_serial')
        return _digest(dict(identity=id(recovery), checker=id(recovery.checker), diameter=recovery.diameter,
                            config=recovery.config, state={key:getattr(recovery,key) for key in fields}))

    def _verify_staged(self, staged, selected, admission, diagnostics, stage_keys,
                       seed=None, provider_context=None, expected_seed=None, recovery_state=None):
        if set(vars(staged)) != stage_keys or staged.checker is not self._checker or staged.diameter != self._diameter:
            raise ValueError('staged controller configuration/context changed')
        if staged.policy.config is not self._refinement_config:
            raise ValueError('staged refinement configuration/context changed')
        expected_policy = AdaptiveRefinementPolicy(self._refinement_config)
        expected_policy.calibrate(diagnostics.get('evidence',{}))
        if _digest(staged.policy.configuration()) != _digest(expected_policy.configuration()):
            raise ValueError('staged initial refinement reference differs from observation')
        if admission.accepted:
            if self._provider_context(seed) != provider_context:
                raise ValueError('staged seed provider configuration/context changed')
            if (staged.seed_tracker is not seed or staged.config is not self._tracking_config
                    or not np.array_equal(staged._anchor_model_points,self._anchors)
                    or not np.array_equal(staged._surface_tree.data,self._anchors)):
                raise ValueError('staged controller configuration/context changed')
            if (not np.array_equal(staged.trusted_pose,selected['pose_centered'])
                    or not np.array_equal(staged._pose,selected['pose_centered'])
                    or diagnostics.get('evidence',{}).get('observation_valid') is not True
                    or any(getattr(staged,key) != 0 for key in ('_lost','_frame','_last_replenish'))
                    or staged._motion_history):
                raise ValueError('staged trusted/working pose or initial counters differ from admission')
            if self._seed_state(seed) != expected_seed or not self._empty_view_state(seed):
                raise ValueError('staged RGB-D memory differs from the exact initial observation and pose')
            if self._recovery_state(staged._recovery) != recovery_state:
                raise ValueError('staged recovery state changed during installation')
        elif (staged.trusted_pose is not None or not np.array_equal(staged.working_pose,selected['pose_centered'])
                or staged.admission_id != admission.content_digest or staged._frame != 0 or staged._last_raw
                or staged.max_translation_diameter != self._tracking_config.max_translation_diameter
                or staged.max_rotation_rad != self._tracking_config.max_rotation_rad):
            raise ValueError('tentative installation changed pose/context or created trusted state')

    def start(self, admission, *, frame_id, rgb, depth, K, masks, working_fallback=None):
        if self._attempted:
            raise RuntimeError('this installation owner was already consumed; use a fresh session')
        self._attempted = True
        self._guard_context()
        bundles = self._bind(admission, frame_id, rgb, depth, K, masks)
        input_digest = _digest(dict(rgb=np.asarray(rgb), depth=np.asarray(depth), K=np.asarray(K),
                                   masks={k:np.asarray(v) for k,v in masks.items()}))
        if admission.accepted:
            selected, candidate = self._candidate(admission, admission.trusted_index, bundles)
            self._trusted_contract(admission, candidate)
        else:
            if admission.trusted_index is not None:
                raise ValueError('unresolved admission cannot contain a trusted candidate')
            if admission.working_index is not None:
                selected, _ = self._candidate(admission, admission.working_index, bundles)
            else:
                selected = self._fallback(working_fallback, bundles)
        image, metric_depth, intrinsics, foreground = _observation(rgb, depth, K, masks[selected['mask_id']])
        arguments = [selected['pose_centered'].copy(), image.copy(), metric_depth.copy(), intrinsics.copy(), foreground.copy()]
        argument_digest = _digest(arguments)
        if admission.accepted:
            seed = self._fresh_seed()
            provider_context = self._provider_context(seed)
            self._guard_context()
            staged = ClickToModelTracker(self._checker, self._diameter, config=self._tracking_config,
                seed_tracker=seed, refinement_config=self._refinement_config, anchor_model_points=self._anchors.copy())
            # Independently rebuild the standard deterministic first-frame
            # corner/ORB lift, before the fallible controller reset. This is
            # CPU-only and creates no externally visible tracker or history.
            reference = RGBDSeedTracker(config=seed.config)
            anchor_mask = ClickToModelTracker._anchor_mask(staged, *[arguments[i].copy() for i in (0,2,3,4)])
            reference.reset(arguments[1].copy(), arguments[2].copy(), arguments[3].copy(),
                            anchor_mask, arguments[0].copy())
            expected_seed = self._seed_state(reference)
            recovery_state = self._recovery_state(staged._recovery)
        else:
            staged = TentativePoseTracker(self._checker, self._diameter, admission_id=admission.content_digest,
                refinement_config=self._refinement_config,
                max_translation_diameter=self._tracking_config.max_translation_diameter,
                max_rotation_rad=self._tracking_config.max_rotation_rad)
            seed = provider_context = expected_seed = recovery_state = None
        stage_keys = set(vars(staged))
        diagnostics = staged.reset(*arguments)
        if _digest(arguments) != argument_digest:
            raise ValueError('staged reset mutated its pose/observation inputs')
        if input_digest != _digest(dict(rgb=np.asarray(rgb), depth=np.asarray(depth), K=np.asarray(K),
                                       masks={k:np.asarray(v) for k,v in masks.items()})):
            raise ValueError('caller observation changed during installation')
        self._guard_context()
        self._bind(admission, frame_id, rgb, depth, K, masks)
        self._verify_staged(staged, selected, admission, diagnostics, stage_keys,
                            seed, provider_context, expected_seed, recovery_state)
        result = dict(diagnostics, initialization_session_id=self._session_id, initialization_frame_id=frame_id,
            admission_id=admission.content_digest, admission_reason=admission.reason,
            admission_accepted=admission.accepted, trusted_memory_initialized=admission.accepted,
            identity_verified=False, initial_orientation_resolved=admission.accepted,
            hypothesis_scope=admission.hypothesis_scope, selected_index=selected['index'],
            selected_mask_id=selected['mask_id'], selected_backend_index=selected['backend_index'],
            selected_source=selected['source'], selected_pose_centered=selected['pose_centered'],
            selected_pose_original=selected['pose_original'])
        result = TentativePoseTracker._record(result)
        json.dumps(result, allow_nan=False)
        selected['mask'] = foreground.copy()
        internal_result, returned_result = deepcopy(result), deepcopy(result)
        # Copies/serialization may fail: complete them before the final guards
        # and publication. A thrown start() never exposes a staged tracker.
        self._guard_context()
        self._bind(admission, frame_id, rgb, depth, K, masks)
        if _digest(arguments) != argument_digest or input_digest != _digest(dict(
                rgb=np.asarray(rgb), depth=np.asarray(depth), K=np.asarray(K),
                masks={k:np.asarray(v) for k,v in masks.items()})):
            raise ValueError('observation changed before publication')
        self._verify_staged(staged, selected, admission, diagnostics, stage_keys,
                            seed, provider_context, expected_seed, recovery_state)
        # No further fallible callbacks after the one-time publication point.
        self._initial_diagnostics, self._selected, self._tracker = internal_result, selected, staged
        return returned_result

    def step(self, *args, **kwargs):
        self._guard_context()
        return self.tracker.step(*args, **kwargs)
