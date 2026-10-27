"""Bounded measured pre-loss anchors, always tentative and in the first gauge.

The original point stream is external and never mutated. An injected factory
creates at most one extra causal stream, with a separate FixedCameraPointMotion
owner. Support pixels propose object membership; measured depth supplies their
coordinates. No neural pose, CAD surface, GT or trusted memory is accepted here.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
import math
from types import MappingProxyType

import numpy as np

from tools.tentative_point_motion import (FixedCameraMotionConfig, FixedCameraPointMotion,
    PointMotionObservation, PointMotionResult, array_digest, geometry, lift, readonly,
    unique_sample_mask)


@dataclass(frozen=True)
class PrelossAnchorConfig:
    max_queries: int = 64
    max_birth_attempts: int = 4
    original_streak: int = 3
    confirmation_streak: int = 3
    pending_window: int = 15
    retirement_cooldown: int = 15
    birth_rotation_degrees: float = 10.

    def __post_init__(self):
        expected = dict(max_queries=64,max_birth_attempts=4,original_streak=3,
                        confirmation_streak=3,pending_window=15,retirement_cooldown=15,
                        birth_rotation_degrees=10.)
        if any(getattr(self, k) != v or isinstance(getattr(self, k), bool) for k,v in expected.items()):
            raise ValueError('this bounded protocol fixes 64/4/3/3/15/15/10; use a separate protocol for changes')
        if any(type(getattr(self,k)) is not int for k in expected if k != 'birth_rotation_degrees'):
            raise ValueError('resource and clock bounds require builtin integers')


def snapshot(value):
    """Copy actual numeric returns before a shared backend reuses scratch."""
    if isinstance(value, np.ndarray): return readonly(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return MappingProxyType({k:snapshot(v) for k,v in value.items()})
    if isinstance(value, (tuple,list)): return tuple(snapshot(x) for x in value)
    if is_dataclass(value): return type(value)(**{f.name:snapshot(getattr(value,f.name)) for f in fields(value)})
    if value is None or isinstance(value, (str,int,float,bool)): return value
    raise TypeError('unsupported mutable raw evidence type: '+type(value).__name__)


def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)


def transform(points, pose):
    return np.asarray(points)@np.asarray(pose)[:3,:3].T+np.asarray(pose)[:3,3]


def valid_transform(value, cfg):
    a=np.asarray(value)
    if (a.shape!=(4,4) or a.dtype.kind not in 'fiu' or a.dtype.itemsize>8
            or not np.isfinite(a).all() or np.max(np.abs(a))>cfg.max_hypothesis_magnitude): return False
    R=a[:3,:3].astype(float)
    return bool(np.max(np.abs(a[3]-[0,0,0,1]))<=1e-7
        and np.max(np.abs(R.T@R-np.eye(3)))<=cfg.max_raw_rotation_defect
        and abs(np.linalg.det(R)-1)<=cfg.max_raw_rotation_defect)


@dataclass(frozen=True)
class SupportPixelPacket:
    frame_id: int
    source_frame_id: int
    algorithm: str
    initial_hypotheses_sha256: str
    working_index: int
    working_pose: np.ndarray
    pixels_xy: np.ndarray
    rgb_sha256: str
    depth_sha256: str
    K_sha256: str
    identity_verified: bool = False

    def __post_init__(self):
        object.__setattr__(self,'working_pose',readonly(self.working_pose))
        object.__setattr__(self,'pixels_xy',readonly(self.pixels_xy))


@dataclass(frozen=True)
class FirstGaugeSeed:
    source: str
    frame_id: int
    source_frame_id: int
    anchor_source_frame_id: int
    measurement_birth_source_frame_id: int | None
    initial_hypotheses_sha256: str
    selected_backend_index: int
    current_from_first: np.ndarray
    transported_hypotheses: np.ndarray
    selected_pose: np.ndarray
    birth_from_first: np.ndarray | None = None
    current_from_birth: np.ndarray | None = None
    identity_verified: bool = False
    absolute_orientation_resolved: bool = False
    trusted_memory_updated: bool = False


@dataclass(frozen=True)
class AnchorStepResult:
    frame_id: int
    source_frame_id: int
    state: str
    decision: str
    original_available: bool
    original_streak: int
    confirmation_streak: int
    birth_decision: str
    secondary_call: str | None
    secondary_observation: object
    secondary_raw_return: object
    raw_capture_status: str
    secondary_motion: PointMotionResult | None
    verification: object
    birth_record: object
    retirement: object
    original_proposal: FirstGaugeSeed | None
    secondary_proposal: FirstGaugeSeed | None
    seed: FirstGaugeSeed | None
    identity_verified: bool = False
    absolute_orientation_resolved: bool = False
    trusted_memory_updated: bool = False


def diverse_pixels(pixels, indices, limit):
    """Farthest-pixel sampling; every tie chooses lexicographic (x,y,input ID)."""
    order=sorted(map(int,indices),key=lambda i:(float(pixels[i,0]),float(pixels[i,1]),i))
    if not order: return np.empty(0,np.int64)
    remaining=np.asarray(order,np.int64);chosen=[];distance=np.full(len(remaining),np.inf)
    for _ in range(min(limit,len(order))):
        position=int(np.argmax(distance));index=int(remaining[position]);chosen.append(index)
        delta=np.asarray(pixels[remaining],float)-pixels[index]
        distance=np.minimum(distance,np.sum(delta*delta,axis=1));distance[position]=-np.inf
    return np.asarray(chosen,np.int64)


class PrelossMeasuredAnchorManager:
    """One extra stream; all absolute hypotheses retain their original order.

    Call step once per input after advancing the original stream. The injected
    secondary_stream_factory() must create a fresh stream with start(rgb, xy,
    frame_id=0) and advance(rgb, frame_id=local_step). It can share model weights
    with the original stream, but must own distinct recurrence. Returned objects
    expose frame_id/query_ids/pixels_xy/visible and optionally raw_return.
    """
    def __init__(self, initial_hypotheses, selected_backend_index, *, anchor_source_frame_id,
                 secondary_stream_factory, config=None):
        self.config=PrelossAnchorConfig() if config is None else config
        if not isinstance(self.config,PrelossAnchorConfig): raise TypeError('typed pre-loss configuration required')
        self.motion_config=FixedCameraMotionConfig()
        q=np.asarray(initial_hypotheses)
        if (q.ndim!=3 or q.shape[1:]!=(4,4) or not 1<=len(q)<=self.motion_config.max_hypotheses
                or any(not valid_transform(p,self.motion_config) for p in q)):
            raise ValueError('complete finite raw initial hypothesis bank required; no repair')
        if type(selected_backend_index) is not int or not 0<=selected_backend_index<len(q):
            raise ValueError('actual original backend index required')
        if type(anchor_source_frame_id) is not int or anchor_source_frame_id<0:
            raise ValueError('nonnegative original source frame ID required')
        if not callable(secondary_stream_factory): raise TypeError('fresh secondary stream factory required')
        self._hypotheses=readonly(q);self._hypotheses_hash=array_digest(q)
        self._selected=selected_backend_index;self._anchor_source=anchor_source_frame_id
        self._factory=secondary_stream_factory;self._frame=None;self._source=None;self._K=None;self._shape=None
        self._original_signature=None;self._original_streak=0;self._attempts=[]
        self._last_attempt_D=np.eye(4);self._retired_at=None
        self._stream=self._motion=self._birth=None;self._state='empty';self._confirmation=0

    @property
    def frame_id(self): return self._frame
    @property
    def state(self): return self._state
    @property
    def birth_attempts(self): return tuple(snapshot(x) for x in self._attempts)
    @property
    def initial_hypotheses(self): return self._hypotheses.copy()

    def _inputs(self,rgb,depth,K,frame,source):
        if type(frame) is not int or frame!=(0 if self._frame is None else self._frame+1):
            raise ValueError('global input frames must be consecutive and consumed once')
        if (type(source) is not int or source<0 or (self._source is None and source!=self._anchor_source)
                or (self._source is not None and source<=self._source)):
            raise ValueError('original source clock must start at anchor and increase strictly')
        rgb,depth,K=np.asarray(rgb),np.asarray(depth),np.asarray(K)
        if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<3
                or depth.shape!=rgb.shape[:2] or depth.dtype not in (np.dtype('float32'),np.dtype('float64'))):
            raise ValueError('registered uint8 RGB and float32/64 camera-Z metres required')
        if (K.shape!=(3,3) or K.dtype.kind not in 'fiu' or not np.isfinite(K).all()
                or K[0,0]<=0 or K[1,1]<=0 or not np.array_equal(K[2],[0,0,1])
                or np.linalg.cond(K.astype(float))>1e12): raise ValueError('fixed nonsingular camera calibration required')
        K=K.astype(float,copy=True)
        if self._shape is not None and (self._shape!=depth.shape or not np.array_equal(self._K,K)):
            raise ValueError('registered camera shape/K changed')
        return rgb.copy(),depth.copy(),K

    def _original(self,result,frame,source,depth,K):
        if result is None: return None,None
        if not isinstance(result,PointMotionResult): raise TypeError('original source must be PointMotionResult or None')
        if (result.frame_id!=frame or result.source_frame_id!=source or result.anchor_source_frame_id!=self._anchor_source
                or result.raw_observation.frame_id!=frame or result.raw_observation.source_frame_id!=source):
            raise ValueError('original motion clock/gauge differs from current input')
        if any(getattr(result,k) is not False for k in ('identity_verified','absolute_orientation_resolved','trusted_memory_updated')):
            raise ValueError('relative source must not assert absolute or identity trust')
        if result.status=='invalid_input': return None,None
        if not same(result.raw_initial_hypotheses,self._hypotheses): raise ValueError('original raw candidate inventory/order changed')
        obs=result.raw_observation;n=len(obs.query_ids)
        if (obs.query_ids.ndim!=1 or obs.query_ids.dtype.kind not in 'iu' or not 1<=n<=self.motion_config.max_queries
                or len(set(obs.query_ids.tolist()))!=n or np.any(obs.query_ids<0)
                or obs.pixels_xy.shape!=(n,2) or obs.pixels_xy.dtype.kind not in 'fiu'
                or obs.visible.shape!=(n,) or obs.visible.dtype!=bool
                or result.source_valid_mask.shape!=(n,) or result.source_valid_mask.dtype!=bool):
            raise ValueError('original raw observation identity/shape differs')
        points,patches,indices,_,depth_valid=lift(obs.pixels_xy,depth,K,self.motion_config)
        expected_valid=unique_sample_mask(indices,result.source_valid_mask&obs.visible&depth_valid)
        if (not same(result.current_camera_points,points) or not same(result.sampled_depth_3x3_m,patches)
                or not same(result.valid_mask,expected_valid)):
            raise ValueError('original observation/depth lift/current valid mask chain differs')
        if self._original_signature is None:
            if frame!=0 or result.status!='reference_only':
                raise ValueError('original measured reference must first be bound on input zero')
            source_points,source_patches,source_indices,_,source_depth_valid=lift(result.source_query_pixels,depth,K,self.motion_config)
            source_possible=unique_sample_mask(source_indices,source_depth_valid)
            if (not same(result.source_camera_points,source_points) or not same(result.source_depth_3x3_m,source_patches)
                    or np.any(result.source_valid_mask&~source_possible)):
                raise ValueError('original source camera anchors are not measured input-zero depth')
        signature=tuple(array_digest(getattr(result,k)) for k in ('source_query_pixels','source_depth_3x3_m',
            'source_camera_points','source_valid_mask'))+(array_digest(result.raw_observation.query_ids),)
        if self._original_signature is not None and signature!=self._original_signature:
            raise ValueError('original fixed anchor/IDs changed; successor cannot impersonate original')
        if (result.diagnostics.get('depth_sha256')!=array_digest(depth)
                or result.diagnostics.get('K_sha256')!=array_digest(K)):
            raise ValueError('original motion bound to another depth/calibration')
        if result.diagnostics.get('config')!=asdict(self.motion_config):
            raise ValueError('original fit changed the fixed geometric policy')
        if result.status!='measured': return None,signature
        D=result.current_from_source
        if (not valid_transform(D,self.motion_config) or result.selected_candidate_index is None
                or not 0<=result.selected_candidate_index<len(result.candidate_transforms)
                or not same(D,result.candidate_transforms[result.selected_candidate_index])
                or not same(D,result.candidate_current_from_source)
                or not same(result.transported_hypotheses,D@self._hypotheses.astype(float))):
            raise ValueError('original measured motion is not its actual selected raw fit/transport')
        if (result.reason!='measured_relative_motion' or not self._comparison(
                transform(result.source_camera_points,D),result.current_camera_points,result.valid_mask)['passed']):
            raise ValueError('original selected transform lacks the declared measured consensus')
        return np.array(D,copy=True),signature

    def _seed(self,D,source,frame,source_frame,local=None):
        poses=np.asarray(D,float)@self._hypotheses.astype(float)
        return FirstGaugeSeed(source,frame,source_frame,self._anchor_source,
            None if local is None else self._birth['source_frame_id'],self._hypotheses_hash,self._selected,
            readonly(D),readonly(poses),readonly(poses[self._selected]),
            None if local is None else readonly(self._birth['birth_from_first']),
            None if local is None else readonly(local))

    def _retire(self,reason,frame,raw_error=None):
        record=snapshot(dict(reason=reason,frame_id=frame,birth_frame_id=self._birth['frame_id'] if self._birth else None,
                             prior_state=self._state,confirmation_streak=self._confirmation,error=raw_error))
        self._stream=self._motion=self._birth=None;self._state='empty';self._confirmation=0;self._retired_at=frame
        return record

    def _comparison(self,a,b,eligible):
        cfg=self.motion_config;valid=np.asarray(eligible,bool).copy()
        valid&=np.isfinite(a).all(1)&np.isfinite(b).all(1)
        for points in (a,b):
            ids=np.flatnonzero(valid)
            if len(ids):
                _,reverse,counts=np.unique(points[ids],axis=0,return_inverse=True,return_counts=True)
                valid[ids]=counts[reverse]==1
        residuals=np.full(len(a),np.nan);residuals[valid]=np.linalg.norm(a[valid]-b[valid],axis=1)
        inliers=valid&(residuals<=cfg.residual_threshold_m);n=int(valid.sum());count=int(inliers.sum())
        shapes=[geometry(x[inliers],cfg) for x in (a,b)]
        passed=bool(count>=cfg.min_points and n>0 and count/n>=cfg.min_inlier_fraction and all(x[0] for x in shapes))
        return snapshot(dict(passed=passed,eligible_mask=valid,inlier_mask=inliers,residuals_m=residuals,
            fixed_query_denominator=len(a),independent_count=n,inlier_count=count,inlier_fraction=count/n if n else 0.,
            geometry_spread_m=min(x[1] for x in shapes),geometry_ratio=min(x[2] for x in shapes)))

    def _observation(self,value,local,source):
        if value.frame_id!=local: raise ValueError('secondary output local clock differs')
        return PointMotionObservation(local,source,value.query_ids,value.pixels_xy,value.visible)

    @staticmethod
    def _capture_raw(stream, previous, *, actual=None, error=None):
        """An unchanged shared scratch pointer is not evidence of a new return."""
        if actual is not None:
            value=getattr(actual,'raw_return',None)
            if value is not None:return snapshot(value),'captured_current_return'
        if error is not None:
            value=getattr(error,'raw_return',None)
            if value is not None and value is not previous:return snapshot(value),'captured_exception_return'
        value=getattr(stream,'last_raw_return',None)
        if value is not None and value is not previous:return snapshot(value),'captured_fresh_stream_return'
        return None,'absent_or_stale'

    def _support(self,packet,rgb,depth,K,D,frame,source):
        if not isinstance(packet,SupportPixelPacket): raise TypeError('explicit typed support packet required')
        if (packet.frame_id!=frame or packet.source_frame_id!=source or packet.working_index!=self._selected
                or packet.initial_hypotheses_sha256!=self._hypotheses_hash
                or not isinstance(packet.algorithm,str) or not packet.algorithm.strip() or len(packet.algorithm)>512
                or packet.identity_verified is not False): raise ValueError('support algorithm, time, hypothesis or trust binding invalid')
        if (not same(packet.working_pose,D@self._hypotheses[self._selected].astype(float))
                or packet.rgb_sha256!=array_digest(rgb) or packet.depth_sha256!=array_digest(depth)
                or packet.K_sha256!=array_digest(K)): raise ValueError('support packet belongs to another working hypothesis/input')
        pixels=packet.pixels_xy
        if (pixels.ndim!=2 or pixels.shape[1:]!=(2,) or pixels.dtype.kind not in 'fiu' or pixels.dtype.itemsize>8
                or not 0<=len(pixels)<=4096 or not np.isfinite(pixels).all()):
            raise ValueError('bounded finite proposed support pixels required')
        points,patches,indices,inside,valid=lift(pixels,depth,K,self.motion_config)
        valid=unique_sample_mask(indices,valid)
        selected=diverse_pixels(pixels,np.flatnonzero(valid),self.config.max_queries)
        good=bool(len(selected)>=self.motion_config.min_points and geometry(points[selected],self.motion_config)[0])
        return dict(packet=snapshot(packet),proposed_query_count=len(pixels),selected_support_indices=readonly(selected),
            query_pixels=readonly(pixels[selected]),birth_camera_points=readonly(points[selected]),
            birth_depth_3x3_m=readonly(patches[selected]),valid_support_mask=readonly(valid),
            image_spread_pixels=float(np.linalg.svd(pixels[selected]-pixels[selected].mean(0),compute_uv=False)[-1]/np.sqrt(len(selected))) if len(selected)>1 else 0.,
            geometry_spread_m=geometry(points[selected],self.motion_config)[1],eligible=good)

    def step(self,rgb,depth_m,K,*,frame_id,source_frame_id,original_motion,support_packet=None):
        rgb,depth,K=self._inputs(rgb,depth_m,K,frame_id,source_frame_id)
        D,signature=self._original(original_motion,frame_id,source_frame_id,depth,K)
        # These preflights do not advance either owned state. Afterwards this
        # global input is consumed even if the extra backend fails.
        self._frame,self._source=frame_id,source_frame_id
        if self._shape is None:self._shape=depth.shape;self._K=readonly(K)
        if signature is not None and self._original_signature is None:self._original_signature=signature
        self._original_streak=self._original_streak+1 if D is not None else 0
        original=None if D is None else self._seed(D,'original',frame_id,source_frame_id)
        secondary=seed=None;verification=birth_record=retirement=None;observation=raw=motion=None;call=None
        raw_capture_status='not_called'
        if self._stream is not None:
            call='advance';local=frame_id-self._birth['frame_id']
            previous_raw=getattr(self._stream,'last_raw_return',None)
            try:
                actual=self._stream.advance(rgb.copy(),frame_id=local)
                raw,raw_capture_status=self._capture_raw(self._stream,previous_raw,actual=actual)
                observation=snapshot(actual)
                motion=self._motion.advance(self._observation(actual,local,source_frame_id),depth,K)
                if motion.status=='invalid_input' or self._motion.poisoned:raise ValueError('secondary motion protocol failure')
                if self._state=='pending':
                    if D is not None:
                        verification=self._comparison(transform(motion.current_camera_points,np.linalg.inv(D)),
                            self._birth['first_camera_points'],motion.valid_mask)
                        self._confirmation=self._confirmation+1 if verification['passed'] else 0
                    else:
                        self._confirmation=0
                        verification=snapshot(dict(passed=False,reason='original_motion_unavailable',
                            fixed_query_denominator=len(self._birth['query_pixels']),residuals_m=np.full(len(self._birth['query_pixels']),np.nan)))
                    if self._confirmation>=self.config.confirmation_streak:self._state='ready'
                    elif local>=self.config.pending_window:retirement=self._retire('pending_window_expired',frame_id)
                if self._state=='ready' and motion.current_from_source is not None and motion.status=='measured':
                    local_D=motion.current_from_source;composed=local_D@self._birth['birth_from_first']
                    secondary=self._seed(composed,'secondary',frame_id,source_frame_id,local_D)
            except Exception as exc:
                if raw is None:
                    try:raw,raw_capture_status=self._capture_raw(self._stream,previous_raw,error=exc)
                    except Exception:pass
                retirement=self._retire('secondary_protocol_or_backend_failure',frame_id,type(exc).__name__+': '+str(exc))
        decision='neural_fallback'
        if original is not None and secondary is not None:
            agreement=self._comparison(transform(self._birth['first_camera_points'],original.current_from_first),
                transform(self._birth['first_camera_points'],secondary.current_from_first),motion.valid_mask)
            verification=snapshot(dict(confirmation=verification,motion_agreement=agreement))
            if agreement['passed']:seed=original;decision='original_secondary_agree'
            else:decision='conflicting_relative_motions'
        elif original is not None:seed=original;decision='original_only'
        elif secondary is not None:seed=secondary;decision='ready_secondary_only'
        if self._stream is not None:birth_decision='active_secondary'
        elif D is None:birth_decision='original_motion_unavailable'
        elif self._original_streak<self.config.original_streak:birth_decision='insufficient_original_streak'
        elif len(self._attempts)>=self.config.max_birth_attempts:birth_decision='attempt_budget_exhausted'
        elif self._retired_at is not None and frame_id-self._retired_at<self.config.retirement_cooldown:birth_decision='retirement_cooldown'
        else:
            relative=D[:3,:3]@np.linalg.inv(self._last_attempt_D[:3,:3])
            angle=float(math.degrees(math.acos(float(np.clip((np.trace(relative)-1)/2,-1.,1.)))))
            birth_decision='view_change_below_threshold' if angle<self.config.birth_rotation_degrees else 'support_not_supplied'
            if angle>=self.config.birth_rotation_degrees and support_packet is not None:
                birth_decision='attempted'
                attempt=dict(attempt_index=len(self._attempts),frame_id=frame_id,source_frame_id=source_frame_id,
                    rotation_from_last_attempt_degrees=angle,birth_from_first=readonly(D),
                    original_streak=self._original_streak,backend_called=False,status='failed',identity_verified=False)
                self._last_attempt_D=readonly(D)
                try:
                    details=self._support(support_packet,rgb,depth,K,D,frame_id,source_frame_id);attempt.update(details)
                    if not details['eligible']:raise ValueError('insufficient independent measured support')
                    first=transform(details['birth_camera_points'],np.linalg.inv(D))
                    attempt['first_camera_points']=readonly(first)
                    original_pixels=original_motion.raw_observation.pixels_xy
                    original_valid=original_motion.valid_mask&np.isfinite(original_pixels).all(1)
                    original_samples={tuple(x) for x in np.rint(original_pixels[original_valid]).astype(np.int64)}
                    attempt['overlap_with_current_original_valid_samples']=sum(tuple(x) in original_samples
                        for x in np.rint(details['query_pixels']).astype(np.int64))
                    stream=self._factory()
                    if getattr(stream,'frame_id','missing') is not None or getattr(stream,'poisoned',True) is not False:
                        raise ValueError('factory must provide a fresh independent secondary state')
                    self._stream=stream;self._motion=FixedCameraPointMotion(self.motion_config)
                    self._birth=attempt;call='start';attempt['backend_called']=True
                    previous_raw=getattr(stream,'last_raw_return',None)
                    actual=stream.start(rgb.copy(),details['query_pixels'].copy(),frame_id=0)
                    raw,raw_capture_status=self._capture_raw(stream,previous_raw,actual=actual);observation=snapshot(actual)
                    local_obs=self._observation(actual,0,source_frame_id)
                    if not np.array_equal(local_obs.query_ids,np.arange(len(details['query_pixels']))):
                        raise ValueError('secondary original query IDs must retain 0..N-1 order')
                    mask=np.zeros(depth.shape,bool);xy=np.rint(details['query_pixels']).astype(np.int64);mask[xy[:,1],xy[:,0]]=True
                    motion=self._motion.start(local_obs,depth,K,np.eye(4)[None],mask,initial_query_pixels=details['query_pixels'])
                    if motion.status!='reference_only' or self._motion.poisoned:raise ValueError('secondary measured anchor start failed')
                    if not same(motion.source_camera_points,details['birth_camera_points']):raise ValueError('birth measurements changed')
                    self._state='pending';self._confirmation=0;attempt.update(status='pending',confirmation_streak=0)
                except Exception as exc:
                    if raw is None and self._stream is not None:
                        try:raw,raw_capture_status=self._capture_raw(self._stream,previous_raw,error=exc)
                        except Exception:pass
                    attempt['error']=type(exc).__name__+': '+str(exc)
                    retirement=self._retire('birth_failed',frame_id,attempt['error'])
                birth_record=snapshot(attempt);self._attempts.append(birth_record)
        return AnchorStepResult(frame_id,source_frame_id,self._state,decision,D is not None,self._original_streak,
            self._confirmation,birth_decision,
            call,observation,raw,raw_capture_status,snapshot(motion),verification,birth_record,retirement,original,secondary,seed)
