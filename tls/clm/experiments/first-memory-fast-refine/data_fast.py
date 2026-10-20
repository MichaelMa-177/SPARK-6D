"""RGB-D input preparation; validation annotations are loaded only for scoring."""
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch

DATA=Path(os.environ['CLM_DATA_ROOT'])
ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene(number,supervision=False):
    base=DATA/f'{number:06d}'
    camera=json.loads((base/'scene_camera.json').read_text())
    gt=json.loads((base/'scene_gt.json').read_text()) if supervision else None
    records=[]
    for path in sorted((base/'rgb').glob('*.png')):
        fid=int(path.stem);cam=camera[str(fid)]
        r=dict(scene=number,frame=fid,rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB),
               depth=cv2.imread(str(base/'depth'/path.name),-1),
               K=np.asarray(cam['cam_K'],np.float32).reshape(3,3),scale=cam['depth_scale']/1000,
               rgb_path=str(path))
        if supervision:
            index,item=next((i,x) for i,x in enumerate(gt[str(fid)]) if x['obj_id']==15)
            r['mask']=cv2.imread(str(base/f'mask_visib/{fid:06d}_{index:06d}.png'),0)>0
            t=np.eye(4,dtype=np.float32);t[:3,:3]=np.asarray(item['cam_R_m2c']).reshape(3,3);t[:3,3]=np.asarray(item['cam_t_m2c'])/1000
            r['gt']=t
        records.append(r)
    return records


def lift(r):
    z=r['depth'].astype(np.float32)*r['scale'];h,w=z.shape
    yy,xx=np.indices((h,w),dtype=np.float32);k=r['K']
    xyz=np.stack(((xx-k[0,2])*z/k[0,0],(yy-k[1,2])*z/k[1,1],z),-1)
    valid=(z>.05)&(z<3)&np.isfinite(z)
    xyz[~valid]=0
    return xyz,valid


def box_from_mask(mask):
    y,x=np.where(mask)
    return np.array([x.min(),y.min(),x.max()+1,y.max()+1],np.float32)


def crop(r,box,center,mask=None):
    h,w=r['depth'].shape
    cx,cy=(box[:2]+box[2:])/2;side=max(box[2:]-box[:2])*1.5+8
    x1=max(0,int(cx-side/2));y1=max(0,int(cy-side/2));x2=min(w,max(x1+8,int(cx+side/2)));y2=min(h,max(y1+8,int(cy+side/2)))
    x1=min(x1,w-8);y1=min(y1,h-8)
    xs=np.floor(np.arange(128)*(x2-x1)/128).astype(int)+x1
    ys=np.floor(np.arange(128)*(y2-y1)/128).astype(int)+y1
    z=r['depth'][ys[:,None],xs[None]].astype(np.float32)*r['scale'];k=r['K']
    xyz=np.stack(((xs[None]-k[0,2])*z/k[0,0],(ys[:,None]-k[1,2])*z/k[1,1],z),-1).astype(np.float32)
    valid=(z>.05)&(z<3)&np.isfinite(z);xyz[~valid]=0
    rgb=cv2.resize(r['rgb'][y1:y2,x1:x2],(128,128),interpolation=cv2.INTER_AREA)
    result=dict(rgb=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255,
                xyz=torch.from_numpy(xyz.transpose(2,0,1).copy()),valid=torch.from_numpy(valid.astype(np.float32)[None]),
                center=torch.as_tensor(center,dtype=torch.float32),box=[x1,y1,x2,y2])
    if mask is not None:result['mask']=torch.from_numpy(mask[ys[:,None],xs[None]].copy()).float()
    return result


def first_reference(records,mask):
    r=records[0];xyz,valid=lift(r);points=xyz[mask&valid]
    center=points.mean(0);item=crop(r,box_from_mask(mask),center,mask)
    eligible=np.flatnonzero((item['mask'][::4,::4].numpy()>.5)&(item['valid'][0,::4,::4].numpy()>.5))
    assert len(eligible)>=12,'Too few measured first-frame anchors'
    item['indices']=torch.from_numpy(eligible[np.linspace(0,len(eligible)-1,256).round().astype(int)])
    return item


def target_pose(record,first,center):
    origin=np.eye(4,dtype=np.float32);origin[:3,3]=center
    return record['gt']@np.linalg.inv(first['gt'])@origin


def projected_box(points,pose,k,shape):
    p=np.einsum('ni,ji->nj',points,pose[:3,:3])+pose[:3,3]
    good=p[:,2]>.05
    if good.sum()<4:return np.array([0,0,shape[1],shape[0]],np.float32)
    p=p[good];u=p[:,0]*k[0,0]/p[:,2]+k[0,2];v=p[:,1]*k[1,1]/p[:,2]+k[1,2]
    low=np.percentile(np.stack((u,v),1),1,axis=0);high=np.percentile(np.stack((u,v),1),99,axis=0)
    return np.r_[low,high]


def inputs(item,device):
    return [item[k][None].to(device) for k in ('rgb','xyz','valid','center')]
