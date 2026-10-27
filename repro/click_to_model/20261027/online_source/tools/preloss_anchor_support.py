"""Traceable sparse CAD support proposals; anchor coordinates remain measured RGB-D.

The CAD projection only proposes pixels. This module neither lifts anchor points
nor invokes a neural model, reads GT, selects a pose, or asserts object identity.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import time
import numpy as np

from pose_validity import PoseValidityChecker
from tools.preloss_measured_anchors import SupportPixelPacket, snapshot, valid_transform
from tools.tentative_point_motion import FixedCameraMotionConfig, array_digest

ALGORITHM='sparse_checker_projection_observed_depth/v1'
OWNED=('tools/preloss_anchor_support.py','tests/test_preloss_anchor_support.py','docs/PRELOSS_ANCHOR_SUPPORT.md')


@dataclass(frozen=True)
class SupportProposal:
    packet: SupportPixelPacket | None
    arrays: Mapping
    record: Mapping


def _numeric(value):
    a=np.asarray(value)
    if a.dtype.kind not in 'biufc':raise TypeError('actual projection evidence must be numeric')
    return np.array(a,copy=True)


def _same(left,right):
    left,right=np.asarray(left),np.asarray(right)
    return left.shape==right.shape and left.dtype==right.dtype and np.array_equal(left,right,equal_nan=True)


def _identity(value):
    a=np.ascontiguousarray(value)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def _projection_lineage(points,normals,pose,K,shape):
    """Retain every checker-point slot before the existing sparse z-buffer."""
    camera=points@pose[:3,:3].T+pose[:3,3]
    finite_positive=np.isfinite(camera).all(1)&(camera[:,2]>1e-4)
    facing=np.ones(len(points),bool);camera_normals=None
    if normals is not None:
        camera_normals=normals@pose[:3,:3].T
        facing=np.sum(camera_normals*camera,axis=1)<0
    valid=finite_positive&facing;ids=np.flatnonzero(valid)
    homogeneous=camera[ids]@K.T;uv=homogeneous[:,:2]/homogeneous[:,2,None]
    float_uv=np.full((len(points),2),np.nan);float_uv[ids]=uv
    height,width=shape
    inside=((uv[:,0]>=-.5)&(uv[:,0]<width-.5)&(uv[:,1]>=-.5)&(uv[:,1]<height-.5))
    in_image=np.zeros(len(points),bool);in_image[ids[inside]]=True
    ids=ids[inside];rounded=np.rint(uv[inside]).astype(np.int64)
    pixel=np.full((len(points),2),-1,np.int64);pixel[ids]=rounded
    z=camera[ids,2];linear=rounded[:,1]*width+rounded[:,0];order=np.lexsort((z,linear))
    unique=np.r_[True,linear[order][1:]!=linear[order][:-1]] if len(order) else np.empty(0,bool)
    chosen=order[unique];retained_ids=ids[chosen]
    keep=np.zeros(len(points),bool);keep[retained_ids]=True
    values=dict(checker_point_ids=np.arange(len(points),dtype=np.int64),camera_points=camera,
        finite_positive_depth_mask=finite_positive,front_facing_mask=facing,
        front_facing_positive_mask=valid,projected_uv_float_by_point=float_uv,
        in_image_mask=in_image,rounded_uv_by_point=pixel,zbuffer_keep_mask=keep)
    if camera_normals is not None:values['camera_normals']=camera_normals
    return values,(rounded[chosen],z[chosen],retained_ids)


def propose_support(checker,rgb,depth_m,K,*,frame_id,source_frame_id,initial_hypotheses,
                    selected_backend_index,current_from_first):
    """Return complete projection evidence and a typed pixel packet, or an error.

    A valid projection with zero compatible pixels returns an empty packet; the
    manager decides whether it is a bounded failed birth attempt. Malformed
    inputs/projection return packet=None with the available raw prefix retained.
    This module does not write caller-owned inputs or geometry. Unexpected
    mutation by the checker is reported, without attempting to repair it.
    """
    started=time.perf_counter();arrays={};packet=None
    record=dict(schema='preloss_support_proposal/1',algorithm=ALGORITHM,status='unavailable',
        reason=None,frame_id=frame_id,source_frame_id=source_frame_id,
        projection_invoked=False,projection_returned=False,projected_count=None,
        projection_arguments_unchanged=None,checker_unchanged=None,
        post_projection_checks_completed=False,post_projection_validation_errors=[],
        compatible_count=0,identity_verified=False,trusted_memory_updated=False,
        absolute_orientation_resolved=False,GT_read=False,NN_called=False,
        anchor_coordinates='not produced here; manager lifts actual 3x3 observed depth',
        model_sample_ids='indices into checker.points; not original CAD vertex IDs',
        coordinate_frame='checker points are FP centered metric CAD; D@Q0 maps them into current camera metres')
    try:
        if type(frame_id)is not int or frame_id<0 or type(source_frame_id)is not int or source_frame_id<0:
            raise ValueError('nonnegative builtin stream/source frame IDs required')
        if not isinstance(checker,PoseValidityChecker):raise TypeError('actual PoseValidityChecker required')
        rgb,depth,raw_K=np.asarray(rgb),np.asarray(depth_m),np.asarray(K)
        bank,D=np.asarray(initial_hypotheses),np.asarray(current_from_first)
        if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<3
                or depth.shape!=rgb.shape[:2] or depth.dtype not in (np.dtype('float32'),np.dtype('float64'))):
            raise ValueError('registered uint8 RGB and float32/64 camera-Z depth required')
        if (raw_K.shape!=(3,3) or raw_K.dtype.kind not in 'fiu' or raw_K.dtype.itemsize>8
                or not np.isfinite(raw_K).all() or raw_K[0,0]<=0 or raw_K[1,1]<=0
                or not np.array_equal(raw_K[2],[0,0,1]) or np.linalg.cond(raw_K.astype(float))>1e12):
            raise ValueError('finite fixed camera calibration required')
        bounds=FixedCameraMotionConfig()
        if (bank.ndim!=3 or bank.shape[1:]!=(4,4) or bank.dtype.kind not in 'fiu' or bank.dtype.itemsize>8
                or not 1<=len(bank)<=bounds.max_hypotheses or not np.isfinite(bank).all()
                or type(selected_backend_index)is not int or not 0<=selected_backend_index<len(bank)):
            raise ValueError('complete numeric initial bank and actual selected index required')
        if not valid_transform(D,bounds):
            raise ValueError('actual finite current-from-first transform required')
        # Same canonical K and raw-bank multiplication used by the manager.
        camera=raw_K.astype(float,copy=True);bank=bank.copy();D=D.copy()
        working=D@bank[selected_backend_index].astype(float)
        arrays.update(raw_K=raw_K.copy(),K_used=camera.copy(),raw_initial_hypotheses=bank,
            current_from_first=D,working_pose=working.copy(),checker_points=checker.points.copy())
        if checker.normals is not None:arrays['checker_normals']=checker.normals.copy()
        tolerance=float(checker.tolerance);diameter=float(checker.diameter)
        if not np.isfinite(tolerance) or tolerance<=0 or not np.isfinite(diameter) or diameter<=0:
            raise ValueError('positive finite actual checker tolerance/diameter required')
        config=snapshot(checker.configuration())
        record.update(input_rgb_sha256=array_digest(rgb),input_depth_sha256=array_digest(depth),
            input_raw_K_sha256=array_digest(raw_K),input_K_sha256=array_digest(camera),
            input_shape=list(depth.shape),depth_dtype=str(depth.dtype),
            initial_hypotheses_sha256=array_digest(bank),initial_candidate_count=len(bank),
            selected_backend_index=selected_backend_index,working_pose_sha256=array_digest(working),
            current_from_first_sha256=array_digest(D),checker_points_sha256=array_digest(checker.points),
            checker_normals_sha256=None if checker.normals is None else array_digest(checker.normals),
            checker_points_identity=_identity(checker.points),
            checker_normals_identity=None if checker.normals is None else _identity(checker.normals),
            checker_configuration=config,checker_tolerance_m=tolerance,diameter_m=diameter,
            comparison='finite positive observed depth and abs(observed_z - expected_z) <= actual checker.tolerance',
            proposal_pixel_limit=4096,checker_class=type(checker).__module__+'.'+type(checker).__qualname__)
        lineage,expected_projection=_projection_lineage(arrays['checker_points'],arrays.get('checker_normals'),
            working,camera,depth.shape)
        arrays.update(lineage)
        record.update(checker_point_denominator=len(checker.points),
            finite_positive_count=int(lineage['finite_positive_depth_mask'].sum()),
            front_facing_positive_count=int(lineage['front_facing_positive_mask'].sum()),
            in_image_count=int(lineage['in_image_mask'].sum()),zbuffer_count=int(lineage['zbuffer_keep_mask'].sum()))
        call_pose,call_K=working.copy(),camera.copy()
        record['projection_invoked']=True
        # Once the checker has run, unknown integrity is a failure. Each check
        # and capturable numeric field is independent: a malformed checker must
        # not hide an actual return or turn the integrity result back to None.
        record['projection_arguments_unchanged']=False;record['checker_unchanged']=False
        projection_error=None
        try:
            returned=checker.project_visible(call_pose,call_K,depth.shape)
            record['projection_returned']=True
        except Exception as error:
            projection_error=error
            record['projection_error']=type(error).__name__+': '+str(error)
        finally:
            errors=record['post_projection_validation_errors']
            for name,getter in (('projection_pose_after',lambda:call_pose),
                    ('projection_K_after',lambda:call_K),('checker_points_after',lambda:checker.points),
                    ('checker_normals_after',lambda:checker.normals)):
                try:
                    value=getter()
                    if value is not None:arrays[name]=_numeric(value)
                except Exception as error:errors.append(name+': '+type(error).__name__+': '+str(error))
            try:record['projection_arguments_unchanged']=bool(_same(call_pose,working) and _same(call_K,camera))
            except Exception as error:errors.append('projection_arguments: '+type(error).__name__+': '+str(error))
            try:
                # Derive identities only from numeric copies, never object-array
                # memory addresses or a second read of mutable checker state.
                record['checker_points_identity_after']=_identity(arrays['checker_points_after'])
                record['checker_normals_identity_after']=(None if checker.normals is None else
                    _identity(arrays['checker_normals_after']))
                record['checker_unchanged']=bool(_same(checker.points,arrays['checker_points'])
                    and (checker.normals is None)==('checker_normals' not in arrays)
                    and (checker.normals is None or _same(checker.normals,arrays['checker_normals']))
                    and float(checker.tolerance)==tolerance and float(checker.diameter)==diameter
                    and checker.configuration()==dict(config))
            except Exception as error:errors.append('checker_integrity: '+type(error).__name__+': '+str(error))
            if errors:
                record['checker_unchanged']=False
            record['post_projection_checks_completed']=not errors
        if projection_error is not None:raise projection_error
        if not isinstance(returned,(tuple,list)) or len(returned)!=3:
            raise ValueError('projection must return uv, expected_z, model_sample_ids')
        failures=[]
        for name,value in zip(('projected_uv','expected_z_m','model_sample_ids'),returned):
            try:arrays[name]=_numeric(value)
            except Exception as error:failures.append(name+': '+type(error).__name__+': '+str(error))
        if failures:
            record['capture_errors']=failures
            raise ValueError('some actual projection return fields were not numeric')
        uv,z,ids=arrays['projected_uv'],arrays['expected_z_m'],arrays['model_sample_ids']
        n=len(uv) if uv.ndim else None;record['projected_count']=n
        if record['projection_arguments_unchanged']is not True or record['checker_unchanged']is not True:
            raise ValueError('projection mutated its input pose/K or bound checker geometry/configuration')
        if (uv.ndim!=2 or uv.shape[1:]!=(2,) or uv.dtype.kind not in 'iu' or n>4096
                or z.shape!=(n,) or z.dtype.kind not in 'fiu' or not np.isfinite(z).all() or np.any(z<=0)
                or ids.shape!=(n,) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(checker.points))
                or np.any(uv<0) or np.any(uv[:,0]>=depth.shape[1]) or np.any(uv[:,1]>=depth.shape[0])
                or len(np.unique(uv,axis=0))!=n or len(np.unique(ids))!=n):
            raise ValueError('complete unique in-image sparse projection and model sample IDs required')
        if any(not _same(left,right) for left,right in zip((uv,z,ids),expected_projection)):
            raise ValueError('actual sparse projection differs from recorded centered-point/camera/zbuffer lineage')
        measured=depth[uv[:,1],uv[:,0]].copy()
        residual=measured.astype(float)-z.astype(float)
        valid=np.isfinite(measured)&(measured>0)
        compatible=valid&(np.abs(residual)<=tolerance)
        measured_by_point=np.full(len(checker.points),np.nan);measured_by_point[ids]=measured
        residual_by_point=np.full(len(checker.points),np.nan);residual_by_point[ids]=residual
        depth_valid_by_point=np.zeros(len(checker.points),bool);depth_valid_by_point[ids]=valid
        compatible_by_point=np.zeros(len(checker.points),bool);compatible_by_point[ids]=compatible
        arrays.update(observed_z_m=measured,depth_residual_m=residual,observed_depth_valid=valid,
            compatible_mask=compatible,compatible_projection_indices=np.flatnonzero(compatible),
            observed_z_by_point=measured_by_point,depth_residual_by_point=residual_by_point,
            observed_depth_valid_by_point=depth_valid_by_point,depth_compatible_by_point=compatible_by_point,
            packet_pixels_xy=uv[compatible].astype(float))
        packet=SupportPixelPacket(frame_id,source_frame_id,ALGORITHM,record['initial_hypotheses_sha256'],
            selected_backend_index,working,arrays['packet_pixels_xy'],record['input_rgb_sha256'],
            record['input_depth_sha256'],record['input_K_sha256'])
        record.update(status='proposed',reason='compatible_pixels' if compatible.any() else 'no_compatible_pixels',
            compatible_count=int(compatible.sum()),projected_denominator=n,
            compatible_fraction=float(compatible.mean()) if n else None)
    except Exception as error:
        record.update(status='unavailable',reason=type(error).__name__+': '+str(error))
    record['wall_seconds']=time.perf_counter()-started
    record['array_digests']={key:array_digest(value) for key,value in arrays.items()}
    return SupportProposal(packet,snapshot(arrays),snapshot(record))
