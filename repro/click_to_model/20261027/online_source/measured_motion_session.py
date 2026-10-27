"""Product-facing measured 2D/depth motion and tentative neural continuation.

The session owns one original point stream and at most one preloss recurrence.
It never installs trusted RGB-D memory or resolves physical identity. No probe
runner, dataset, GT loader, model constructor or historical query file is used.
"""
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
import hashlib
import json
import math
import re

import cv2
import numpy as np

from adaptive_refinement import AdaptiveRefinementConfig
from relative_motion_refinement import RelativeMotionRefinementTracker
from tentative_tracking import TentativePoseTracker, TentativeStepError
from tools.preloss_anchor_support import propose_support
from tools.preloss_measured_anchors import PrelossMeasuredAnchorManager, PrelossAnchorConfig, snapshot, valid_transform
from tools.tentative_point_motion import FixedCameraPointMotion, FixedCameraMotionConfig, PointMotionObservation, array_digest


def select_initial_queries(rgb, depth_m, mask):
    """Reproduce the measured first-frame query policy on current inputs only."""
    eroded=cv2.erode(np.asarray(mask,bool).astype(np.uint8),np.ones((5,5),np.uint8))
    corners=cv2.goodFeaturesToTrack(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),64,.01,5.,mask=eroded)
    raw=np.empty((0,2),np.float64) if corners is None else corners[:,0].astype(np.float64)
    ij=np.rint(raw).astype(np.int64)
    z=np.asarray(depth_m)[ij[:,1],ij[:,0]]
    valid=np.isfinite(z)&(z>.02)&(z<10.)
    selected=raw[valid].copy()
    return dict(algorithm='eroded5_GFTT64_quality0.01_distance5_depth0.02_to10/v1',
        raw_corner_pixels=raw,raw_corner_depth_m=z.copy(),depth_valid_mask=valid,
        pixels_xy=selected,query_ids=np.arange(len(selected),dtype=np.int64),
        eroded_mask=eroded.astype(bool),historical_queries_read=False,identity_verified=False)


class DirectoryEvidenceSink:
    """Synchronous append-only numeric evidence, preserving nonfinite arrays."""
    def __init__(self, output_dir):
        self.directory=Path(output_dir)
        self.directory.mkdir(parents=True,exist_ok=True)
        if any(self.directory.iterdir()):raise FileExistsError('fresh evidence directory required')
        self._index=0

    def __call__(self,kind,payload):
        if not isinstance(kind,str)or not re.fullmatch(r'[a-zA-Z0-9_]+',kind):raise ValueError('safe evidence kind required')
        arrays={}
        def pack(value):
            if isinstance(value,np.ndarray):
                if value.dtype.kind not in 'biufcSU':raise TypeError('numeric/string arrays only; no pickle evidence')
                key=f'array_{len(arrays):06d}';arrays[key]=np.array(value,copy=True)
                return dict(array=key,dtype=str(value.dtype),shape=list(value.shape),sha256=array_digest(value))
            if isinstance(value,np.generic):return pack(value.item())
            if is_dataclass(value):return dict(type=type(value).__name__,fields={f.name:pack(getattr(value,f.name))for f in fields(value)})
            if isinstance(value,Mapping):
                if any(not isinstance(k,str)for k in value):raise TypeError('string evidence keys required')
                return {k:pack(v)for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [pack(v)for v in value]
            if isinstance(value,float)and not math.isfinite(value):return dict(nonfinite_float=str(value))
            if value is None or isinstance(value,(str,int,float,bool)):return value
            raise TypeError('unsupported evidence value '+type(value).__name__)
        record=dict(schema='measured_motion_session_evidence/1',kind=kind,sequence=self._index,payload=pack(payload))
        prefix=self.directory/f'{self._index:06d}_{kind}'
        if prefix.with_suffix('.json').exists()or prefix.with_suffix('.npz').exists():raise FileExistsError('evidence already exists')
        self._index+=1  # Failed writes leave their attempted slot; never overwrite it.
        np.savez_compressed(prefix.with_suffix('.npz'),**arrays)
        record['npz_sha256']=hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest()
        prefix.with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        return dict(path=str(prefix.with_suffix('.json')),npz=str(prefix.with_suffix('.npz')),sha256=hashlib.sha256(prefix.with_suffix('.json').read_bytes()).hexdigest())


class MeasuredMotionSessionError(TentativeStepError):
    """Terminal session failure with actual neural returns retained."""


class MeasuredMotionSession:
    def __init__(self,checker,diameter,*,session_id,stream_factory,evidence_sink=None,
                 refinement_config=None,enable_preloss=True):
        if not isinstance(session_id,str)or not session_id.strip():raise ValueError('nonempty session ID required')
        if not callable(stream_factory):raise TypeError('fresh independent point-stream factory required')
        if evidence_sink is not None and not callable(evidence_sink):raise TypeError('callable evidence sink required')
        if type(enable_preloss)is not bool:raise TypeError('enable_preloss must be bool')
        self.checker,self.diameter,self.session_id=checker,float(diameter),session_id
        self._factory,self._sink,self.enable_preloss=stream_factory,evidence_sink,enable_preloss
        self._tracker=RelativeMotionRefinementTracker(checker,diameter,admission_id=session_id,
            refinement_config=refinement_config or AdaptiveRefinementConfig())
        self._started=False;self._closed=False;self._frame=None;self._source=None;self._timestamp=None
        self._shape=self._K=self._centering=self._bank=None;self._selected=None
        self._stream=self._motion=self._manager=None;self._queries=None
        self._point_unavailable_reason=None;self._last_record=None;self._failure=None;self._sink_failure=None
        self._original_raw=None;self._checker_binding=None;self._stage='constructed'
        self._last_neural_raw=();self._evidence_failure=None;self._anchor_source=None

    @property
    def frame_id(self):return self._frame
    @property
    def source_frame_id(self):return self._source
    @property
    def working_pose(self):return self._tracker.working_pose
    @property
    def last_raw_returns(self):return tuple(value.copy()for value in self._last_neural_raw)
    @property
    def last_record(self):return snapshot(self._last_record)
    @property
    def closed(self):return self._closed

    @staticmethod
    def _flags():
        return dict(verified=False,identity_verified=False,initial_orientation_resolved=False,
            absolute_orientation_resolved=False,trusted_memory_updated=False,memory_updated=False,seed_admitted=False)

    def configuration(self):
        return dict(class_name=type(self).__name__,session_id=self.session_id,policy='tentative-preloss'if self.enable_preloss else'tentative-original',
            trusted_memory_enabled=False,point_config=asdict(FixedCameraMotionConfig()),preloss_config=asdict(PrelossAnchorConfig()),
            refinement=self._tracker.configuration(),evidence_enabled=self._sink is not None,
            initial_query_policy='eroded5_GFTT64_quality0.01_distance5_depth0.02_to10/v1')

    def record_evidence(self,kind,payload):
        value=snapshot(dict(session_id=self.session_id,frame_id=self._frame,source_frame_id=self._source,
            stage=self._stage,payload=payload))
        self._last_record=value  # Keep in-memory evidence even if serialization fails.
        try:return None if self._sink is None else self._sink(kind,value)
        except BaseException as exc:
            if self._evidence_failure is None:self._evidence_failure=value
            try:exc.evidence_payload=value
            except Exception:pass
            raise

    def _binding(self):
        return dict(points=array_digest(self.checker.points),normals=None if self.checker.normals is None else array_digest(self.checker.normals),
            diameter=float(self.checker.diameter),tolerance=float(self.checker.tolerance),configuration=self.checker.configuration())

    def _guard_checker(self):
        if self._checker_binding is not None and self._binding()!=self._checker_binding:raise ValueError('registered checker geometry/configuration changed; new epoch required')

    def _input(self,rgb,depth_m,K,*,frame_id,source_frame_id,timestamp,mask=None):
        if type(frame_id)is not int or frame_id!=(0 if self._frame is None else self._frame+1):raise ValueError('input clock must be consecutive and consumed once')
        if type(source_frame_id)is not int or source_frame_id<0 or(self._source is not None and source_frame_id<=self._source):raise ValueError('source clock must increase; new epoch required')
        if timestamp is not None and(isinstance(timestamp,(bool,np.bool_))or not np.isscalar(timestamp)or not np.isfinite(timestamp)):
            raise ValueError('finite timestamp required')
        if timestamp is not None and self._timestamp is not None and timestamp<self._timestamp:raise ValueError('timestamp moved backwards; new epoch required')
        rgb,depth,K=map(np.asarray,(rgb,depth_m,K))
        TentativePoseTracker._inputs(rgb,depth,K,mask)
        if depth.dtype not in (np.dtype('float32'),np.dtype('float64'))or min(depth.shape)<3:raise ValueError('registered float32/64 camera-Z metres required')
        canonical=K.astype(float,copy=True)
        if self._shape is not None and(depth.shape!=self._shape or not np.array_equal(canonical,self._K)):
            raise ValueError('registered K/shape changed; new epoch required')
        return rgb.copy(),depth.copy(),canonical

    def _fresh_stream(self):
        stream=self._factory()
        if stream is self._stream or getattr(stream,'frame_id','missing')is not None or getattr(stream,'poisoned',True)is not False:
            raise ValueError('factory must create a fresh independent point recurrence')
        return stream

    def _original(self,rgb,depth,K,frame_id,source_frame_id,mask=None):
        previous=getattr(self._stream,'last_raw_return',None)
        self._stage='original_point_call'
        try:
            obs=(self._stream.start(rgb.copy(),self._queries['pixels_xy'].copy(),frame_id=0)if frame_id==0
                else self._stream.advance(rgb.copy(),frame_id=frame_id))
        except BaseException as exc:
            raw=getattr(exc,'raw_return',None)
            current=getattr(self._stream,'last_raw_return',None)
            if raw is None and current is not previous:raw=current
            self._original_raw=snapshot(raw)
            self.record_evidence('original_point_failure',dict(raw_return=raw,error=type(exc).__name__+': '+str(exc),
                raw_capture_status='captured'if raw is not None else'absent_or_stale'))
            raise
        self._original_raw=snapshot(getattr(obs,'raw_return',getattr(self._stream,'last_raw_return',None)))
        self.record_evidence('original_point_return',dict(observation=obs,raw_return=self._original_raw))
        self._stage='original_depth_fit'
        current=PointMotionObservation(frame_id,source_frame_id,obs.query_ids,obs.pixels_xy,obs.visible)
        if obs.frame_id!=frame_id:raise ValueError('original native/public clock differs from actual input')
        result=(self._motion.start(current,depth,K,self._bank,mask,initial_query_pixels=self._queries['pixels_xy'])if frame_id==0
            else self._motion.advance(current,depth,K))
        self.record_evidence('original_motion',dict(result=result))
        if result.status=='invalid_input':raise ValueError('original fixed-camera motion owner poisoned')
        return result

    def _selection(self,original,rgb,depth,K,frame_id,source_frame_id):
        proposal=event=None
        if self._manager is not None:
            if self._manager.state=='empty'and original.status=='measured':
                self._stage='support_projection'
                proposal=propose_support(self.checker,rgb,depth,K,frame_id=frame_id,source_frame_id=source_frame_id,
                    initial_hypotheses=self._bank,selected_backend_index=self._selected,current_from_first=original.current_from_source)
                self.record_evidence('support_proposal',dict(proposal=proposal))
                if proposal.record.get('projection_invoked')is True and any(proposal.record.get(k)is not True for k in('checker_unchanged','projection_arguments_unchanged')):
                    raise ValueError('actual support integrity failed or unproven')
                self._guard_checker()
            self._stage='preloss_manager'
            event=self._manager.step(rgb,depth,K,frame_id=frame_id,source_frame_id=source_frame_id,
                original_motion=original,support_packet=None if proposal is None else proposal.packet)
            self.record_evidence('preloss_event',dict(event=event))
            seed=None if event.seed is None else np.asarray(event.seed.selected_pose).copy()
            source=event.seed.source if event.seed is not None else'conflict'if event.decision=='conflicting_relative_motions'else'fallback'
        else:
            D=None if original is None else original.current_from_source
            seed=None if D is None else (D@self._bank.astype(float))[self._selected].copy()
            source='original'if D is not None else'fallback'
        return seed,source,event

    def _fail(self,exc):
        self._closed=True
        self._failure=dict(self._flags(),status='session_failed',error=type(exc).__name__+': '+str(exc),stage=self._stage,
            frame_id=self._frame,source_frame_id=self._source,working_seed_updated=False,session_closed=True)
        try:self.record_evidence('session_failure',dict(diagnostics=self._failure,raw_neural_returns=self.last_raw_returns,
            original_raw=self._original_raw,original_evidence_failure=self._evidence_failure))
        except BaseException as sink_error:self._sink_failure=type(sink_error).__name__+': '+str(sink_error)
        return MeasuredMotionSessionError(exc,self.last_raw_returns,dict(self._failure,evidence_error=self._sink_failure))

    def start(self,*,initial_hypotheses,selected_backend_index,centered_from_original,rgb,depth_m,K,mask,
              source_frame_id=0,timestamp=None,initial_pose_original=None,registration_metadata=None):
        if self._started or self._closed:raise RuntimeError('a session starts only once; construct a new epoch')
        self._started=True;self._stage='start_preflight'
        try:
            self.record_evidence('registration',dict(initial_hypotheses=np.asarray(initial_hypotheses),selected_backend_index=selected_backend_index,
                centered_from_original=np.asarray(centered_from_original),initial_pose_original=initial_pose_original,metadata=registration_metadata))
            rgb,depth,K=self._input(rgb,depth_m,K,frame_id=0,source_frame_id=source_frame_id,timestamp=timestamp,mask=mask)
            mask=np.asarray(mask,bool).copy();bank=np.asarray(initial_hypotheses)
            if bank.ndim!=3 or bank.shape[1:]!=(4,4)or bank.dtype.kind not in 'fiu' or type(selected_backend_index)is not int or not 0<=selected_backend_index<len(bank):
                raise ValueError('complete numeric bank and actual selected backend index required')
            C=np.asarray(centered_from_original)
            if not valid_transform(C,FixedCameraMotionConfig()):
                raise ValueError('finite raw rigid actual CAD centering required')
            actual_original=None if initial_pose_original is None else np.asarray(initial_pose_original)
            conversion=dict(actual_original_provided=actual_original is not None,checked=False,
                absolute_matrix_element_tolerance=2e-7,relative_tolerance=0.,
                tolerance_basis='frozen preloss actual FP original-return vs centered-return @ centering comparison',
                actual_dtype=None if actual_original is None else str(actual_original.dtype),
                derived_dtype=str((bank[selected_backend_index]@C).dtype))
            if actual_original is not None:
                if (actual_original.shape!=(4,4)or actual_original.dtype.kind not in 'fiu'
                    or actual_original.dtype.itemsize>8 or not np.isfinite(actual_original).all()
                    or not np.allclose(actual_original,bank[selected_backend_index]@C,atol=2e-7,rtol=0)):
                    raise ValueError('actual initial original pose differs from selected raw centered candidate @ actual C')
                conversion['checked']=True
                conversion['maximum_matrix_element_difference']=float(np.max(np.abs(actual_original-(bank[selected_backend_index]@C))))
            self.record_evidence('initial_coordinate_check',conversion)
            self._bank=np.array(bank,copy=True);self._selected=selected_backend_index;self._centering=C.copy()
            self._shape=depth.shape;self._K=K.copy();self._checker_binding=self._binding()
            self._frame=0;self._source=source_frame_id;self._timestamp=None if timestamp is None else float(timestamp)
            self._anchor_source=source_frame_id
            self.record_evidence('initial_input',dict(rgb=rgb,depth_m=depth,K=K,mask=mask,checker_binding=self._checker_binding,configuration=self.configuration()))
            self._stage='tentative_reset'
            initial=self._tracker.reset(bank[selected_backend_index],rgb,depth,K,mask,source_frame_id=source_frame_id)
            self._queries=select_initial_queries(rgb,depth,mask)
            self.record_evidence('initial_queries',self._queries)
            if len(self._queries['pixels_xy'])<FixedCameraMotionConfig().min_points:
                self._point_unavailable_reason='fewer_than_eight_current_depth_valid_corners'
            else:
                try:
                    candidate_manager=PrelossMeasuredAnchorManager(self._bank,self._selected,anchor_source_frame_id=source_frame_id,
                        secondary_stream_factory=self._fresh_stream)
                except (ValueError,TypeError)as exc:
                    candidate_manager=None;self._point_unavailable_reason='full_initial_bank_unsupported: '+str(exc)
                if self._point_unavailable_reason is None:
                    self._manager=candidate_manager if self.enable_preloss else None
                    self._stream=self._fresh_stream();self._motion=FixedCameraPointMotion()
                    original=self._original(rgb,depth,K,0,source_frame_id,mask)
                    self._selection(original,rgb,depth,K,0,source_frame_id)
            self._guard_checker();self._stage='started'
            diagnostics=dict(initial,**self._flags(),status='tentative_registration',frame_id=0,source_frame_id=source_frame_id,
                point_initialization_available=self._stream is not None,point_unavailable_reason=self._point_unavailable_reason,
                selected_backend_index=self._selected,session_id=self.session_id,source='initialization',
                initial_coordinate_check=conversion)
            result=dict(pose=bank[self._selected].copy(),pose_original=bank[self._selected]@self._centering,
                raw_iterations=np.empty((0,4,4)),diagnostics=diagnostics)
            self.record_evidence('started',result)
            return result
        except BaseException as exc:raise self._fail(exc)from exc

    def step(self,rgb,depth_m,K,refine_one,*,frame_id=None,source_frame_id,timestamp=None):
        if not self._started or self._closed:raise RuntimeError('active started session required')
        self._stage='frame_preflight'
        self._last_neural_raw=();self._original_raw=None
        try:
            expected=self._frame+1;frame_id=expected if frame_id is None else frame_id
            rgb,depth,K=self._input(rgb,depth_m,K,frame_id=frame_id,source_frame_id=source_frame_id,timestamp=timestamp)
            if not callable(refine_one):raise TypeError('real centered-pose callback required')
            self._guard_checker()
            self._frame=frame_id;self._source=source_frame_id;self._timestamp=self._timestamp if timestamp is None else float(timestamp)
            self.record_evidence('frame_input',dict(rgb=rgb,depth_m=depth,K=K,timestamp=timestamp))
            original=None if self._stream is None else self._original(rgb,depth,K,frame_id,source_frame_id)
            seed,source,event=self._selection(original,rgb,depth,K,frame_id,source_frame_id)
            self.record_evidence('selected_seed',dict(seed=seed,source=source,event=event,raw_initial_hypotheses=self._bank,
                point_unavailable_reason=self._point_unavailable_reason))
            self._stage='neural_refinement'
            def callback(actual_seed):
                self.record_evidence('neural_input',dict(centered_seed=np.asarray(actual_seed).copy()))
                returned=np.asarray(refine_one(actual_seed)).copy()
                self._last_neural_raw=self._last_neural_raw+(returned.copy(),)
                self.record_evidence('neural_return',dict(centered_return=returned))
                return returned
            staged=copy(self._tracker)
            staged.policy=deepcopy(self._tracker.policy)
            staged._working_pose=self._tracker.working_pose
            result=staged.step(rgb,depth,K,callback,frame_seed=seed,frame_id=frame_id,source_frame_id=source_frame_id,
                anchor_source_frame_id=self._anchor_source)
            self._guard_checker()
            result=dict(result,pose_original=np.asarray(result['pose'])@self._centering)
            result['diagnostics']=dict(result['diagnostics'],**self._flags(),session_id=self.session_id,
                frame_id=frame_id,source_frame_id=source_frame_id,timestamp=timestamp,selected_motion_source=source,
                manager_state='disabled'if self._manager is None else self._manager.state,
                point_initialization_available=self._stream is not None,point_unavailable_reason=self._point_unavailable_reason,
                derived_original_pose=True)
            self._stage='frame_complete';self.record_evidence('frame_result',result)
            self._tracker=staged  # Publish only after all fallible evidence writes.
            return result
        except BaseException as exc:raise self._fail(exc)from exc

    def close(self,reason='closed'):
        if not isinstance(reason,str)or not reason:reason='closed'
        already=self._closed;self._closed=True;self._stage='closed'
        result=dict(self._flags(),status='closed',reason=reason,already_closed=already,session_id=self.session_id,
            frame_id=self._frame,source_frame_id=self._source,failure=self._failure)
        try:self.record_evidence('closed',result)
        except BaseException as exc:
            self._sink_failure=type(exc).__name__+': '+str(exc);result['evidence_error']=self._sink_failure
        return result
