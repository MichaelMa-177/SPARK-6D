"""Causal candidate publication gate. NumPy only; no model, GT or file access."""
from dataclasses import dataclass
import hashlib
import numpy as np

FLAGS=('verified','identity_verified','initial_orientation_resolved',
       'absolute_orientation_resolved','trusted_memory_updated','memory_updated','seed_admitted')

@dataclass(frozen=True)
class Config:
    rgb_energy: float = .025
    depth_structure_m: float = .008
    depth_jump_m: float = .03
    depth_min_m: float = .02
    depth_max_m: float = 10.
    residual_m: float = .008
    min_points: int = 8
    min_fraction: float = .60
    spread_m: float = .003
    geometry_ratio: float = .02
    confirmation_streak: int = 3

def signature(a):
    a=np.asarray(a)
    return (a.dtype.str,a.shape,hashlib.sha256(a.tobytes()).hexdigest())

def array_digest(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def transform(points, pose):
    p=np.asarray(pose,float)
    return np.asarray(points,float)@p[:3,:3].T+p[:3,3]

def rigid(p):
    p=np.asarray(p)
    if p.shape!=(4,4)or p.dtype.kind not in 'fiu'or not np.isfinite(p).all():
        raise ValueError('finite 4x4 pose required')
    if not np.allclose(p[3],[0,0,0,1],rtol=0,atol=1e-6):raise ValueError('rigid bottom row')
    if np.linalg.norm(p[:3,:3].T@p[:3,:3]-np.eye(3))>.02 or np.linalg.det(p[:3,:3])<=0:
        raise ValueError('rigid rotation required')
    return p.astype(float)

def geometry(points,cfg):
    p=np.asarray(points,float)
    if len(p)<3 or not np.isfinite(p).all():return False
    s=np.linalg.svd(p-p.mean(0),compute_uv=False)/np.sqrt(len(p))
    return bool(s[1]>=cfg.spread_m and s[1]/max(s[0],1e-15)>=cfg.geometry_ratio)

def cues(rgb,depth,pixels,K,cfg):
    """Per-query source/current cue availability and camera-Z lifting."""
    rgb,depth,pixels,K=map(np.asarray,(rgb,depth,pixels,K))
    if rgb.dtype!=np.uint8 or rgb.shape!=depth.shape+(3,) or depth.ndim!=2:
        raise ValueError('uint8 RGB and camera-Z image required')
    if pixels.ndim!=2 or pixels.shape[1]!=2 or K.shape!=(3,3) or not np.isfinite(K).all():
        raise ValueError('pixels/K shape')
    n=len(pixels);h,w=depth.shape
    idx=np.rint(np.clip(np.where(np.isfinite(pixels),pixels,-1),-1,max(h,w)+1)).astype(np.int64)
    inside=np.isfinite(pixels).all(1)&(pixels[:,0]>=0)&(pixels[:,0]<=w-1)&(pixels[:,1]>=0)&(pixels[:,1]<=h-1)&(idx[:,0]>=1)&(idx[:,0]<w-1)&(idx[:,1]>=1)&(idx[:,1]<h-1)
    patches=np.full((n,9),np.nan);energy=np.zeros(n);points=np.full((n,3),np.nan)
    ids=np.flatnonzero(inside)
    if len(ids):
        x,y=idx[ids].T
        patches[ids]=np.stack([depth[y+dy,x+dx]for dy in (-1,0,1)for dx in (-1,0,1)],axis=1)
        c=rgb[y,x].astype(float)/255
        grads=np.stack((rgb[y,x+1]/255.-c,c-rgb[y,x-1]/255.,rgb[y+1,x]/255.-c,c-rgb[y-1,x]/255.),axis=1)
        energy[ids]=np.sqrt(np.sum(grads**2,axis=(1,2))/6.)
    valid=inside&np.isfinite(patches).all(1)&(patches>=cfg.depth_min_m).all(1)&(patches<=cfg.depth_max_m).all(1)&(np.ptp(patches,axis=1)<=cfg.depth_jump_m)
    ids=np.flatnonzero(valid)
    if len(ids):points[ids]=np.c_[pixels[ids],np.ones(len(ids))]@np.linalg.inv(K).T*patches[ids,4:5]
    return dict(rgb=valid&(energy>=cfg.rgb_energy),depth=valid&(np.ptp(patches,axis=1)>=cfg.depth_structure_m),valid=valid,points=points,energy=energy)

class Branch:
    def __init__(self,motion,rgb,depth,K,cfg,first_points=None):
        self.ids=np.asarray(motion['raw_observation']['query_ids']).copy()
        self.points=np.asarray(motion['source_camera_points']if first_points is None else first_points).copy()
        self.anchor_signature=tuple(signature(motion[k])for k in ('source_query_pixels','source_depth_3x3_m','source_camera_points','source_valid_mask'))
        self.source_cues=cues(rgb,depth,motion['source_query_pixels'],K,cfg)
        self.source_valid=np.asarray(motion['source_valid_mask'],bool).copy()
        self.previous=None;self.alive=True;self.current=None;self.current_points=None
        self.birth_frame=None;self.birth_source=None;self.birth_D=None;self.confirmations=0;self.bridged=False

    def update(self,motion,rgb,depth,K,cfg):
        if self.anchor_signature!=tuple(signature(motion[k])for k in ('source_query_pixels','source_depth_3x3_m','source_camera_points','source_valid_mask')):
            raise ValueError('fixed source anchors changed')
        obs=motion['raw_observation']
        if (obs['frame_id'],obs['source_frame_id'])!=(motion['frame_id'],motion['source_frame_id']):
            raise ValueError('nested raw observation clock differs from current motion')
        if signature(obs['query_ids'])!=signature(self.ids):raise ValueError('query IDs changed')
        observed=cues(rgb,depth,obs['pixels_xy'],K,cfg)
        # Same lifting convention as the recorded motion, but computed here from current input.
        if not np.array_equal(observed['points'],motion['current_camera_points'],equal_nan=True):
            raise ValueError('current points are not this RGB-D/K observation')
        current=self.source_valid&np.asarray(motion['valid_mask'],bool)&np.asarray(obs['visible'],bool)&((self.source_cues['rgb']&observed['rgb'])|(self.source_cues['depth']&observed['depth']))
        overlap=int(current.sum())if self.previous is None else int(np.count_nonzero(current&self.previous))
        self.alive=bool(self.alive and overlap>0)
        self.previous=current.copy();self.current=current;self.current_points=observed['points']
        return dict(cue_count=int(current.sum()),continuing_query_count=overlap,alive=self.alive,
                    rgb_cue_count=int(np.count_nonzero(current&self.source_cues['rgb']&observed['rgb'])),
                    depth_cue_count=int(np.count_nonzero(current&self.source_cues['depth']&observed['depth'])))

    def candidate_fit(self,D,cfg):
        valid=self.current.copy();ids=np.flatnonzero(valid)
        # Duplicated anchors/endpoints cannot inflate the denominator.
        if len(ids):
            for p in (self.points,self.current_points):
                _,inv,c=np.unique(p[ids],axis=0,return_inverse=True,return_counts=True)
                valid[ids]&=c[inv]==1
        ids=np.flatnonzero(valid);res=np.full(len(valid),np.nan)
        if len(ids):res[ids]=np.linalg.norm(transform(self.points[ids],D)-self.current_points[ids],axis=1)
        good=valid&(res<=cfg.residual_m);n=int(valid.sum());k=int(good.sum())
        ok=bool(n>=cfg.min_points and k>=cfg.min_points and k/n>=cfg.min_fraction and geometry(self.points[good],cfg)and geometry(self.current_points[good],cfg))
        return dict(passed=ok,eligible=n,inliers=k,inlier_fraction=k/n if n else None,
                    median_residual_m=float(np.median(res[ids]))if n else None,
                    max_residual_m=float(np.max(res[ids]))if n else None)

class ContinuityGate:
    def __init__(self,epoch_id,config=Config(),selected_backend_index=0,centered_from_original=None):
        if not isinstance(epoch_id,str)or not epoch_id:raise ValueError('explicit click/epoch ID required')
        self.epoch_id=epoch_id;self.cfg=config;self.frame=-1;self.source=None
        if type(selected_backend_index)is not int or selected_backend_index<0:raise ValueError('actual selected bank index required')
        self.selected=selected_backend_index;self.bank_hash=None
        self.C=np.eye(4)if centered_from_original is None else np.array(centered_from_original,copy=True)
        rigid(self.C)
        self.original=None;self.secondary=None;self.initial_pose=None;self.K=None;self.shape=None
        self.lost=False;self.initial_unestablished=False;self.bank_signature=None;self.last=None

    def update(self,*,frame,source,rgb,depth,K,motion,event,candidate_centered,candidate_original,observation_valid):
        if type(frame)is not int or frame!=self.frame+1 or type(source)is not int or source<0 or(self.source is not None and source<=self.source):
            raise ValueError('causal consecutive input/increasing source clock required')
        Q=rigid(candidate_centered);rigid(candidate_original)
        if not np.allclose(np.asarray(candidate_centered)@self.C,candidate_original,atol=2e-7,rtol=0):
            raise ValueError("actual original pose does not match centered pose and fixed C")
        if type(observation_valid)is not bool:raise ValueError('explicit observation support bool required')
        if self.K is not None and(not np.array_equal(K,self.K)or np.shape(depth)!=self.shape):raise ValueError('epoch K/shape changed')
        if motion is not None:
            if (motion['frame_id'],motion['source_frame_id'])!=(frame,source)or(motion['raw_observation']['frame_id'],motion['raw_observation']['source_frame_id'])!=(frame,source):raise ValueError('original clock mismatch')
            sig=signature(motion['raw_initial_hypotheses'])
            if self.bank_signature is not None and sig!=self.bank_signature:raise ValueError('original bank changed')
            self.bank_signature=sig
            self.bank_hash=array_digest(motion['raw_initial_hypotheses'])
            if any(motion.get(k)is not False for k in ('identity_verified','absolute_orientation_resolved','trusted_memory_updated')):raise ValueError('point motion must remain untrusted')
        if event is not None and(event['frame_id'],event['source_frame_id'])!=(frame,source):raise ValueError('preloss outer clock mismatch')
        if frame==0:
            self.initial_pose=Q.copy();self.K=np.array(K,copy=True);self.shape=np.shape(depth)
            if motion is not None:
                if motion['status']!='reference_only':raise ValueError('initial reference required')
                bank=np.asarray(motion['raw_initial_hypotheses'])
                if bank.ndim!=3 or bank.shape[1:]!=(4,4) or not 0<=self.selected<len(bank):
                    raise ValueError('complete bank shape and actual selected index required')
                selected=rigid(bank[self.selected])
                if not np.allclose(selected,candidate_centered,atol=2e-7,rtol=0):
                    raise ValueError('initial centered candidate differs from selected bank return')
                self.original=Branch(motion,rgb,depth,K,self.cfg)
        self.frame,self.source=frame,source
        if self.original is not None and motion is None:
            self.original.alive=False
            orig=dict(alive=False,cue_count=0,continuing_query_count=0,reason='original_measurement_absent')
        else:
            orig={}if self.original is None else self.original.update(motion,rgb,depth,K,self.cfg)
        if frame==0 and not orig.get('alive',False):self.initial_unestablished=True
        sec={};birth_note=None
        if event is None and self.secondary is not None:
            self.secondary.alive=False
            sec=dict(alive=False,reason='current_secondary_event_absent')
        if event is not None:
            if event['retirement']is not None:self.secondary=None
            b=event['birth_record'];m=event['secondary_motion']
            if b is not None and b.get('status')=='pending':
                can=(not self.lost and not self.initial_unestablished and orig.get('alive',False)and motion['status']=='measured'and event['original_available']is True)
                if can:
                    if (b['frame_id'],b['source_frame_id'])!=(frame,source)or b['backend_called']is not True or event['secondary_call']!='start' or m is None:raise ValueError('actual birth clock/call mismatch')
                    D=rigid(b['birth_from_first'])
                    if not np.array_equal(D,motion['current_from_source']):raise ValueError('birth not from current original measurement')
                    if not np.allclose(transform(b['birth_camera_points'],np.linalg.inv(D)),b['first_camera_points'],atol=1e-12,rtol=0):raise ValueError('birth first-camera gauge mismatch')
                    if signature(b['query_pixels'])!=signature(m['source_query_pixels'])or not np.array_equal(b['birth_camera_points'],m['source_camera_points'],equal_nan=True):raise ValueError('birth anchors/query mismatch')
                    self.secondary=Branch(m,rgb,depth,K,self.cfg,b['first_camera_points'])
                    self.secondary.birth_frame=frame;self.secondary.birth_source=source;self.secondary.birth_D=D
                    birth_note='pending_from_connected_original'
                else:birth_note='ignored_unconnected_or_lost_birth'
            branch=self.secondary
            if branch is not None:
                fresh=event['raw_capture_status']in ('captured_current_return','captured_fresh_stream_return')
                if m is None or not fresh or(m['frame_id'],m['source_frame_id'])!=(frame-branch.birth_frame,source):
                    branch.alive=False;sec=dict(alive=False,reason='missing_or_stale_secondary')
                else:
                    sec=branch.update(m,rgb,depth,K,self.cfg)
                    if frame>branch.birth_frame:
                        if event['secondary_call']!='advance':raise ValueError('secondary call not current advance')
                        verify=event['verification']or{}
                        confirm=verify.get('confirmation')if 'confirmation'in verify else verify
                        passed=isinstance(confirm,dict)and confirm.get('passed')is True
                        if not branch.bridged:
                            branch.confirmations=branch.confirmations+1 if passed and orig.get('alive',False)and event['original_available']is True and branch.alive else 0
                            if branch.confirmations>=self.cfg.confirmation_streak and event['state']=='ready':branch.bridged=True
                    sec.update(bridged=branch.bridged,confirmation_streak=branch.confirmations)
                    proposal=event['secondary_proposal']
                    if proposal is not None:
                        if (proposal['frame_id'],proposal['source_frame_id'],proposal['measurement_birth_source_frame_id'])!=(frame,source,branch.birth_source):raise ValueError('secondary proposal clock/birth mismatch')
                        if not np.array_equal(proposal['birth_from_first'],branch.birth_D):raise ValueError('secondary birth transform changed')
                        if m['current_from_source']is None or not np.array_equal(proposal['current_from_birth'],m['current_from_source']):raise ValueError('secondary motion proposal mismatch')
                        if not np.array_equal(proposal['current_from_first'],np.asarray(m['current_from_source'])@branch.birth_D):raise ValueError('secondary first gauge composition mismatch')
                        if proposal['initial_hypotheses_sha256']!=self.bank_hash or proposal['selected_backend_index']!=self.selected:raise ValueError('secondary candidate bank/index changed')
        live=[]
        if orig.get('alive',False):live.append(('original',self.original))
        if self.secondary is not None and self.secondary.alive and self.secondary.bridged:live.append(('secondary',self.secondary))
        if not live and not self.initial_unestablished:self.lost=True
        D=Q@np.linalg.inv(self.initial_pose)
        fits={name:b.candidate_fit(D,self.cfg)for name,b in live}
        conflict=event is not None and event['decision']=='conflicting_relative_motions'
        good=bool(live and any(x['passed']for x in fits.values())and observation_valid and not conflict)
        if self.initial_unestablished:status='initial_continuity_unestablished';reason='no_initial_query_cue_chain'
        elif self.lost:status='identity_continuity_lost';reason='all_connected_query_branches_broken'
        elif good:status='tentative_continuous';reason='connected_cues_and_candidate_anchor_agreement'
        else:status='temporarily_unobservable';reason='conflicting_measured_motions'if conflict else'observation_support_missing'if not observation_valid else'insufficient_candidate_anchor_agreement'
        published=status=='tentative_continuous'
        result=dict(epoch_id=self.epoch_id,frame_id=frame,source_frame_id=source,status=status,reason=reason,
                    candidate_pose=np.array(candidate_original,copy=True),published_pose=np.array(candidate_original,copy=True)if published else None,
                    pose_available=published,observation_valid=observation_valid,original=orig,secondary=sec,
                    birth_note=birth_note,live_branches=[name for name,_ in live],candidate_anchor_fits=fits,
                    identity_loss_latched=self.lost,**{k:False for k in FLAGS})
        self.last=result
        return result
