"""Fixed source-camera RGB-D motion while absolute CAD pose stays unresolved.

This module has no model, seed bank, GT, CAD depth, query replenishment, pose
selection, or adjacent-frame motion chain. It consumes observations produced
by a separately advancing causal point stream. Every transform is measured
directly from the original observed camera points to the current observation.
"""
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from numbers import Real
from types import MappingProxyType

import numpy as np


def readonly(value):
    result=np.array(value,copy=True);result.setflags(write=False);return result


def frozen(value):
    if isinstance(value,np.ndarray):return readonly(value)
    if isinstance(value,dict):return MappingProxyType({key:frozen(item) for key,item in value.items()})
    if isinstance(value,(list,tuple)):return tuple(frozen(item) for item in value)
    return value


def array_digest(value):
    value=np.asarray(value);h=hashlib.sha256()
    h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FixedCameraMotionConfig:
    max_queries:int=256
    max_hypotheses:int=512
    min_points:int=8
    ransac_iterations:int=128
    random_seed:int=0
    residual_threshold_m:float=.008
    min_inlier_fraction:float=.60
    min_geometry_spread_m:float=.003
    min_geometry_ratio:float=.02
    min_depth_m:float=.02
    max_depth_m:float=10.
    max_depth_jump_m:float=.03
    max_raw_rotation_defect:float=.02
    max_hypothesis_magnitude:float=1e6

    def __post_init__(self):
        for name in ('max_queries','max_hypotheses','min_points','ransac_iterations','random_seed'):
            value=getattr(self,name)
            if type(value)is not int or value<(0 if name=='random_seed' else 1):raise ValueError(name+' has invalid integer bound')
        if not 3<=self.min_points<=self.max_queries:raise ValueError('3 <= min_points <= max_queries required')
        if self.max_queries>4096 or self.max_hypotheses>4096 or self.ransac_iterations>4096:raise ValueError('explicit bounded work limits exceeded')
        for name in ('residual_threshold_m','min_inlier_fraction','min_geometry_spread_m','min_geometry_ratio',
                     'min_depth_m','max_depth_m','max_depth_jump_m','max_raw_rotation_defect','max_hypothesis_magnitude'):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<=0:
                raise ValueError(name+' must be finite positive real')
        if self.min_inlier_fraction>1 or self.min_geometry_ratio>1 or self.max_depth_m<=self.min_depth_m:
            raise ValueError('fraction or depth interval invalid')


@dataclass(frozen=True)
class PointMotionObservation:
    """IDs and pixels from one stream frame; original-source time is separate.

Use the same query_ids, pixels_xy and visible fields from either TAPIR or
TAPNext. Their native network outputs must remain in the caller's own trace.
These copies retain invalid numeric pixels so downstream failures are auditable.
"""
    frame_id:int
    source_frame_id:int
    query_ids:np.ndarray
    pixels_xy:np.ndarray
    visible:np.ndarray

    def __post_init__(self):
        for name in ('query_ids','pixels_xy','visible'):object.__setattr__(self,name,readonly(getattr(self,name)))


@dataclass(frozen=True)
class PointMotionResult:
    status:str
    reason:str
    frame_id:int
    source_frame_id:int
    anchor_source_frame_id:int
    raw_observation:PointMotionObservation
    source_query_pixels:np.ndarray
    source_depth_3x3_m:np.ndarray
    source_camera_points:np.ndarray
    source_valid_mask:np.ndarray
    raw_initial_hypotheses:np.ndarray
    sampled_depth_3x3_m:np.ndarray
    current_camera_points:np.ndarray
    valid_mask:np.ndarray
    inlier_mask:np.ndarray
    residuals_m:np.ndarray
    candidate_transforms:np.ndarray
    selected_candidate_index:int|None
    candidate_current_from_source:np.ndarray|None
    current_from_source:np.ndarray|None
    transported_hypotheses:np.ndarray|None
    diagnostics:object
    identity_verified:bool=False
    absolute_orientation_resolved:bool=False
    trusted_memory_updated:bool=False

    def numeric_arrays(self):
        names=('source_query_pixels','source_depth_3x3_m','source_camera_points','source_valid_mask',
            'raw_initial_hypotheses','sampled_depth_3x3_m','current_camera_points','valid_mask',
            'inlier_mask','residuals_m','candidate_transforms','candidate_current_from_source',
            'current_from_source','transported_hypotheses')
        result={name:np.array(getattr(self,name),copy=True) for name in names if getattr(self,name) is not None}
        result.update(query_ids=self.raw_observation.query_ids.copy(),pixels_xy=self.raw_observation.pixels_xy.copy(),
            visible=self.raw_observation.visible.copy())
        return result

    def as_record(self):
        def json_safe(value):
            if isinstance(value,Mapping):return {str(k):json_safe(v) for k,v in value.items()}
            if isinstance(value,np.ndarray):return json_safe(value.tolist())
            if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
            if isinstance(value,np.generic):return json_safe(value.item())
            if isinstance(value,float) and not np.isfinite(value):return None
            return value
        numeric=self.numeric_arrays()
        return dict(status=self.status,reason=self.reason,frame_id=self.frame_id,source_frame_id=self.source_frame_id,
            anchor_source_frame_id=self.anchor_source_frame_id,selected_candidate_index=self.selected_candidate_index,
            identity_verified=False,absolute_orientation_resolved=False,trusted_memory_updated=False,
            diagnostics=json_safe(self.diagnostics),arrays={name:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=array_digest(value))
                for name,value in numeric.items()})


def geometry(points,config):
    if len(points)<3:return False,0.,0.
    singular=np.linalg.svd(points-points.mean(axis=0),compute_uv=False)/np.sqrt(len(points))
    spread=float(singular[1]);ratio=float(singular[1]/max(singular[0],1e-15))
    return spread>=config.min_geometry_spread_m and ratio>=config.min_geometry_ratio,spread,ratio


def kabsch(source,target):
    a,b=source.mean(0),target.mean(0)
    U,_,Vt=np.linalg.svd((source-a).T@(target-b))
    correction=np.eye(3);correction[2,2]=np.linalg.det(Vt.T@U.T)
    D=np.eye(4);D[:3,:3]=Vt.T@correction@U.T;D[:3,3]=b-D[:3,:3]@a
    return D


def rigid_fit(source,target,valid,config):
    """Retain every actual fit, including the best rejected candidate.

    Consensus denominator is all currently eligible independent endpoints;
    source/visibility/depth losses are reported separately over all fixed IDs.
    No endpoint is duplicated to reach min_points or inflate consensus.
    """
    n=len(source);ids=np.flatnonzero(valid);candidate_list=[];best_index=None
    empty=np.zeros(n,bool);residuals=np.full(n,np.nan)
    detail=dict(eligible_count=len(ids),independent_count=0,inlier_count=0,inlier_fraction=0.,
        attempted_subsets=0,fit_calls=0,geometry_spread_m=None,geometry_ratio=None)
    if len(ids)<config.min_points:return None,None,empty,residuals,candidate_list,'insufficient_correspondences',detail
    # Remove all ambiguous duplicated endpoints, rather than choose one member
    # and hide contradictory pixels assigned to the same physical coordinate.
    _,inverse,counts=np.unique(source[ids],axis=0,return_inverse=True,return_counts=True)
    ids=ids[counts[inverse]==1]
    _,inverse,counts=np.unique(target[ids],axis=0,return_inverse=True,return_counts=True)
    ids=ids[counts[inverse]==1];detail['independent_count']=len(ids)
    if len(ids)<config.min_points:return None,None,empty,residuals,candidate_list,'insufficient_independent_correspondences',detail
    a,b=source[ids],target[ids]
    try:
        if not all(geometry(x,config)[0] for x in (a,b)):
            return None,None,empty,residuals,candidate_list,'degenerate_correspondences',detail
    except Exception as error:
        detail['error']=type(error).__name__+': '+str(error)
        return None,None,empty,residuals,candidate_list,'fit_exception',detail
    rng=np.random.default_rng(config.random_seed);best_count=-1;best_median=np.inf;best_mask=None
    try:
        for attempt in range(config.ransac_iterations+1):
            selected=np.arange(len(a)) if attempt==0 else rng.choice(len(a),3,replace=False)
            detail['attempted_subsets']+=1
            if not all(geometry(x[selected],config)[0] for x in (a,b)):continue
            D=kabsch(a[selected],b[selected]);candidate_list.append(D.copy());detail['fit_calls']+=1
            errors=np.linalg.norm(a@D[:3,:3].T+D[:3,3]-b,axis=1);mask=errors<=config.residual_threshold_m
            count=int(mask.sum());median=float(np.median(errors[mask])) if count else np.inf
            if count>best_count or (count==best_count and median<best_median):
                best_index=len(candidate_list)-1;best_mask=mask;best_count=count;best_median=median
            if count==len(ids):break
        if best_index is None:return None,None,empty,residuals,candidate_list,'no_nondegenerate_fit',detail
        if best_count>=3:
            for _ in range(3):
                D=kabsch(a[best_mask],b[best_mask]);candidate_list.append(D.copy());detail['fit_calls']+=1
                errors=np.linalg.norm(a@D[:3,:3].T+D[:3,3]-b,axis=1);new_mask=errors<=config.residual_threshold_m
                best_index=len(candidate_list)-1
                if np.array_equal(new_mask,best_mask) or new_mask.sum()<3:
                    best_mask=new_mask;break
                best_mask=new_mask
        candidate=candidate_list[best_index]
        residuals[valid]=np.linalg.norm(source[valid]@candidate[:3,:3].T+candidate[:3,3]-target[valid],axis=1)
        mask=residuals[ids]<=config.residual_threshold_m;inliers=empty.copy();inliers[ids[mask]]=True
        count=int(mask.sum());fraction=count/len(ids)
        shapes=[geometry(x[mask],config) for x in (a,b)]
        detail.update(inlier_count=count,inlier_fraction=float(fraction),
            geometry_spread_m=min(x[1] for x in shapes),geometry_ratio=min(x[2] for x in shapes))
        reason='measured_relative_motion'
        if count<config.min_points or fraction<config.min_inlier_fraction:reason='insufficient_rigid_consensus'
        elif not all(x[0] for x in shapes):reason='degenerate_consensus'
        return candidate,best_index,inliers,residuals,candidate_list,reason,detail
    except Exception as error:
        detail['error']=type(error).__name__+': '+str(error)
        candidate=None if best_index is None else candidate_list[best_index]
        return candidate,best_index,empty,residuals,candidate_list,'fit_exception',detail


def lift(pixels,depth,K,config):
    pixels=np.asarray(pixels);n=len(pixels);height,width=depth.shape
    indices=np.rint(np.clip(np.where(np.isfinite(pixels),pixels,-1),-1,max(height,width)+1)).astype(np.int64)
    inside=(np.isfinite(pixels).all(1)&(pixels[:,0]>=0)&(pixels[:,0]<=width-1)&(pixels[:,1]>=0)&(pixels[:,1]<=height-1)
        &(indices[:,0]>=1)&(indices[:,0]<width-1)&(indices[:,1]>=1)&(indices[:,1]<height-1))
    patches=np.full((n,9),np.nan);points=np.full((n,3),np.nan);ids=np.flatnonzero(inside)
    if len(ids):
        x,y=indices[ids].T
        patches[ids]=np.stack([depth[y+dy,x+dx] for dy in (-1,0,1) for dx in (-1,0,1)],axis=1)
    depth_valid=(inside&np.isfinite(patches).all(1)&(patches>=config.min_depth_m).all(1)
        &(patches<=config.max_depth_m).all(1)&(np.ptp(patches,axis=1)<=config.max_depth_jump_m))
    ids=np.flatnonzero(depth_valid)
    if len(ids):points[ids]=np.c_[pixels[ids],np.ones(len(ids))]@np.linalg.inv(K).T*patches[ids,4:5]
    return points,patches,indices,inside,depth_valid


def unique_sample_mask(indices,valid):
    result=valid.copy();ids=np.flatnonzero(valid)
    if len(ids):
        _,inverse,counts=np.unique(indices[ids],axis=0,return_inverse=True,return_counts=True)
        result[ids]=counts[inverse]==1
    return result


class FixedCameraPointMotion:
    """Single source observation, strict stream clock, immutable camera anchors."""
    def __init__(self,config=None):
        config=FixedCameraMotionConfig() if config is None else config
        if not isinstance(config,FixedCameraMotionConfig):raise TypeError('typed fixed-camera motion config required')
        self._config=deepcopy(config)
        self._attempted=False;self._frame=None;self._source_frame=None;self._anchor_source=None
        self._ids=self._anchors=self._anchor_valid=self._hypotheses=self._K=self._shape=None
        self._source_depth_patches=self._source_query_pixels=None
        self._poisoned=False;self._last_result=None

    @property
    def config(self):return deepcopy(self._config)
    @property
    def frame_id(self):return self._frame
    @property
    def source_frame_id(self):return self._source_frame
    @property
    def poisoned(self):return self._poisoned
    @property
    def initial_camera_points(self):return None if self._anchors is None else self._anchors.copy()
    @property
    def initial_hypotheses(self):return None if self._hypotheses is None else self._hypotheses.copy()
    @property
    def last_result(self):return self._last_result

    def _observation(self,observation):
        if not isinstance(observation,PointMotionObservation):raise TypeError('typed PointMotionObservation required')
        if type(observation.frame_id)is not int or type(observation.source_frame_id)is not int or observation.source_frame_id<0:
            raise ValueError('independent integer stream/source times required')
        ids,pixels,visible=observation.query_ids,observation.pixels_xy,observation.visible
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not 1<=len(ids)<=self.config.max_queries
                or len(set(ids.tolist()))!=len(ids) or np.any(ids<0)
                or pixels.shape!=(len(ids),2) or pixels.dtype.kind not in 'fiu' or pixels.dtype.itemsize>8
                or visible.shape!=(len(ids),) or visible.dtype!=bool):raise ValueError('fixed unique query IDs and numeric observations required')
        if self._ids is not None and not np.array_equal(ids,self._ids):raise ValueError('query identity inventory/order changed; no implicit replacement')

    def _camera(self,depth,K):
        depth,K=np.asarray(depth),np.asarray(K)
        if depth.ndim!=2 or min(depth.shape)<3 or depth.dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('explicit float32/64 registered camera-Z metres required')
        if (K.shape!=(3,3) or K.dtype.kind not in 'fiu' or K.dtype.itemsize>8 or not np.isfinite(K).all()
                or K[0,0]<=0 or K[1,1]<=0 or not np.array_equal(K[2],[0,0,1])
                or np.linalg.cond(K.astype(float))>1e12):raise ValueError('finite calibrated nonsingular K required')
        if self._shape is not None and (depth.shape!=self._shape or not np.array_equal(K,self._K)):
            raise ValueError('registered resolution/K cannot change in a fixed source-camera stream')
        return depth.copy(),K.astype(float,copy=True)

    def _result(self,obs,depth,K,*,reference=False):
        points,patches,indices,inside,depth_valid=lift(obs.pixels_xy,depth,K,self.config)
        before_unique=self._anchor_valid&obs.visible&depth_valid
        valid=unique_sample_mask(indices,before_unique)
        detail=dict(config=asdict(self.config),source_camera_anchor_count=int(self._anchor_valid.sum()),
            fixed_query_count=len(self._ids),visible_count=int(obs.visible.sum()),inside_count=int(inside.sum()),
            depth_valid_count=int(depth_valid.sum()),query_correspondence_count=int(valid.sum()),
            duplicated_current_sample_queries=int(before_unique.sum()-valid.sum()),
            full_fixed_query_denominator=len(self._ids),depth_sha256=array_digest(depth),K_sha256=array_digest(K),
            source_anchor_sha256=array_digest(self._anchors),initial_hypotheses_sha256=array_digest(self._hypotheses),
            sample_pixel_indices=indices,inside_mask=inside,depth_valid_mask=depth_valid,
            depth_sampling='nearest np.rint center, actual 3x3 neighborhood, subpixel ray; no interpolation/fill',
            motion_reference='fixed first observed camera points; no adjacent-frame chain',
            candidate_pose_is_identity_verification=False)
        D=None;candidate=None;selected=None;candidates=[];inliers=np.zeros(len(self._ids),bool);residuals=np.full(len(self._ids),np.nan)
        status='unobservable';reason='reference_observation'
        if reference:
            # The coordinate reference is defined, not estimated or verified.
            D=np.eye(4);status='reference_only'
        else:
            candidate,selected,inliers,residuals,candidates,reason,fit=rigid_fit(self._anchors,points,valid,self.config)
            detail.update(fit)
            if reason=='measured_relative_motion':D=candidate;status='measured'
            elif reason in ('insufficient_rigid_consensus','degenerate_consensus'):status='rejected'
        transported=None if D is None else D@self._hypotheses.astype(np.float64)
        if transported is not None and not np.isfinite(transported).all():
            status='rejected';reason='nonfinite_hypothesis_transport';D=None;transported=None
        return PointMotionResult(status,reason,obs.frame_id,obs.source_frame_id,self._anchor_source,
            PointMotionObservation(obs.frame_id,obs.source_frame_id,obs.query_ids,obs.pixels_xy,obs.visible),
            readonly(self._source_query_pixels),readonly(self._source_depth_patches),readonly(self._anchors),
            readonly(self._anchor_valid),readonly(self._hypotheses),
            readonly(patches),readonly(points),readonly(valid),readonly(inliers),readonly(residuals),
            readonly(np.asarray(candidates).reshape(-1,4,4)),selected,None if candidate is None else readonly(candidate),
            None if D is None else readonly(D),None if transported is None else readonly(transported),frozen(detail))

    def _failure(self,obs,error,depth,K):
        """Preserve this packet as evidence; never relabel the prior result."""
        n=len(obs.query_ids) if isinstance(obs,PointMotionObservation) and obs.query_ids.ndim==1 else 0
        def saved(value,shape):return readonly(np.full(shape,np.nan) if value is None else value)
        diagnostics=dict(error=type(error).__name__+': '+str(error),poisoned=True,
            depth_shape=list(np.shape(depth)),K_shape=list(np.shape(K)),no_candidate_published=True)
        for name,value in (('depth',depth),('K',K)):
            try:
                raw=np.asarray(value)
                if raw.dtype.kind in 'buifc':diagnostics[name+'_sha256']=array_digest(raw)
            except Exception:pass
        return PointMotionResult('invalid_input','packet_cannot_be_processed',obs.frame_id,obs.source_frame_id,
            self._anchor_source if self._anchor_source is not None else obs.source_frame_id,obs,
            saved(self._source_query_pixels,(n,2)),saved(self._source_depth_patches,(n,9)),
            saved(self._anchors,(n,3)),readonly(np.zeros(n,bool) if self._anchor_valid is None else self._anchor_valid),
            saved(self._hypotheses,(0,4,4)),readonly(np.full((n,9),np.nan)),readonly(np.full((n,3),np.nan)),
            readonly(np.zeros(n,bool)),readonly(np.zeros(n,bool)),readonly(np.full(n,np.nan)),
            readonly(np.empty((0,4,4))),None,None,None,None,frozen(diagnostics))

    def start(self,observation,depth_m,K,initial_hypotheses,initial_mask,*,initial_query_pixels):
        if self._attempted or self._poisoned:raise RuntimeError('initial observation cannot be replaced; create a new owner')
        self._attempted=True
        try:
            self._observation(observation)
            if observation.frame_id!=0:raise ValueError('initial stream frame must be zero')
            depth,K=self._camera(depth_m,K);queries=np.asarray(initial_query_pixels);mask=np.asarray(initial_mask)
            if (queries.shape!=observation.pixels_xy.shape or queries.dtype.kind not in 'fiu' or queries.dtype.itemsize>8
                    or not np.isfinite(queries).all() or mask.shape!=depth.shape or mask.dtype!=bool):
                raise ValueError('explicit original query pixels and Boolean first object mask required')
            hypotheses=np.asarray(initial_hypotheses)
            if hypotheses.dtype.kind in 'buifc':self._hypotheses=readonly(hypotheses)
            if (hypotheses.ndim!=3 or hypotheses.shape[1:]!=(4,4) or hypotheses.dtype.kind not in 'fiu'
                    or hypotheses.dtype.itemsize>8 or not 1<=len(hypotheses)<=self.config.max_hypotheses
                    or not np.isfinite(hypotheses).all()
                    or np.max(np.abs(hypotheses.astype(float)))>self.config.max_hypothesis_magnitude):raise ValueError('bounded complete raw initial hypotheses required')
            for pose in hypotheses:
                R=pose[:3,:3].astype(float)
                if (np.max(np.abs(R.T@R-np.eye(3)))>self.config.max_raw_rotation_defect
                        or abs(np.linalg.det(R)-1)>self.config.max_raw_rotation_defect
                        or np.max(np.abs(pose[3]-[0,0,0,1]))>1e-7):raise ValueError('raw hypothesis outside declared domain; no rotation repair')
            anchors,patches,indices,inside,depth_valid=lift(queries,depth,K,self.config)
            selected=np.flatnonzero(inside);on_mask=np.zeros(len(queries),bool)
            on_mask[selected]=mask[indices[selected,1],indices[selected,0]]
            # Initial visibility is not permission to fill depth or add IDs.
            self._ids=readonly(observation.query_ids);self._anchors=readonly(anchors)
            self._anchor_valid=readonly(unique_sample_mask(indices,depth_valid&on_mask))
            self._hypotheses=readonly(hypotheses);self._K=readonly(K);self._shape=depth.shape
            self._source_depth_patches=readonly(patches);self._source_query_pixels=readonly(queries)
            self._anchor_source=observation.source_frame_id
            result=self._result(observation,depth,K,reference=True)
            self._frame=0;self._source_frame=observation.source_frame_id;self._last_result=result
            return result
        except Exception as error:
            self._poisoned=True
            if not isinstance(observation,PointMotionObservation):raise
            self._last_result=self._failure(observation,error,depth_m,K)
            return self._last_result

    def advance(self,observation,depth_m,K):
        if self._poisoned or self._frame is None:raise RuntimeError('start a valid fresh owner before advancing')
        # Preflight mistakes consume no frame; a correctly timed stream packet
        # that cannot be processed is terminal, so it cannot be silently replayed.
        if (not isinstance(observation,PointMotionObservation) or type(observation.frame_id)is not int
                or observation.frame_id!=self._frame+1):
            raise ValueError('strictly consecutive stream frame IDs required')
        if type(observation.source_frame_id)is not int or observation.source_frame_id<=self._source_frame:
            raise ValueError('source frame IDs must be strictly increasing, independently of stream step')
        self._frame=observation.frame_id;self._source_frame=observation.source_frame_id
        try:
            self._observation(observation);depth,K=self._camera(depth_m,K)
            result=self._result(observation,depth,K);self._last_result=result;return result
        except Exception as error:
            self._poisoned=True;self._last_result=self._failure(observation,error,depth_m,K)
            return self._last_result
