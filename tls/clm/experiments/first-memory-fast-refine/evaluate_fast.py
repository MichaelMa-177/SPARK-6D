"""Fresh RGB-D inference first; validation GT is opened afterwards for scoring."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from fast_model import FastRefiner
from train_fast import prior
from data_fast import ROOT,scene,first_reference,inputs,crop,projected_box,target_pose,sha


def stats(a):
    a=np.asarray(a);return dict(p50=float(np.median(a)),p95=float(np.percentile(a,95)),
                               p99=float(np.percentile(a,99)),max=float(a.max()),mean=float(a.mean()))


def build(model,records,mode):
    mask=cv2.imread(str(ROOT/'inputs/mask.png'),0)>0
    ref=first_reference(records,mask);center=ref['center'].numpy()
    memory=model.memory(model.encode(*inputs(ref,'cuda')),ref['indices'][None].cuda(),
                        ref['center'][None].cuda(),*prior(59,'cuda'))
    if mode=='no_generated':memory={**memory,'p':memory['p'][:,:256],'f':memory['f'][:,:256]}
    if mode=='shuffled_observed':
        order=torch.randperm(256,device='cuda',generator=torch.Generator(device='cuda').manual_seed(0))
        memory={**memory,'f':torch.cat((memory['f'][:,order],memory['f'][:,256:]),1)}
    memory=model.cache(memory)
    pose=np.eye(4,dtype=np.float32);pose[:3,3]=center
    return memory,pose


def run(model,records,mode,png=False,warmup=False):
    torch.cuda.synchronize();begin=time.perf_counter();memory,pose=build(model,records,mode)
    points=memory['p'][0].cpu().numpy();torch.cuda.synchronize();init_ms=(time.perf_counter()-begin)*1000
    poses=[pose.copy()];timing=[];extras=[]
    for r0 in records[1:]:
        begin=time.perf_counter();r=r0
        if png:
            r=dict(r0,rgb=cv2.cvtColor(cv2.imread(r0['rgb_path']),cv2.COLOR_BGR2RGB),
                   depth=cv2.imread(str(Path(r0['rgb_path']).parent.parent/'depth'/Path(r0['rgb_path']).name),-1))
        read_end=time.perf_counter()
        box=projected_box(points,pose,r['K'],r['depth'].shape)
        item=crop(r,box,pose[:3,3]);prep_end=time.perf_counter()
        tensors=inputs(item,'cuda')
        events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        events[0].record();encoded=model.encode(*tensors);events[1].record()
        result=model.refine(encoded,memory,torch.from_numpy(pose)[None].cuda());events[2].record()
        pose=result['T'][0].cpu().numpy()
        end=time.perf_counter()
        poses.append(pose.copy())
        timing.append(dict(frame=r['frame'],gap=r['frame']-r0.get('previous_frame',records[len(poses)-2]['frame']),
                           ms=(end-begin)*1000,read_ms=(read_end-begin)*1000,roi_xyz_ms=(prep_end-read_end)*1000,
                           encode_gpu_ms=events[0].elapsed_time(events[1]),match2_gpu_ms=events[1].elapsed_time(events[2])))
        extras.append(dict(frame=r['frame'],residual_mm=float(result['residual'][0].sqrt().item()*1000),
                           weight_sum=float(result['weight_sum'][0]),membership_mean=float(result['membership'][0].mean()),
                           generated_attention=float(result['association'][0,:,256:].sum(-1).mean()) if mode!='no_generated' else 0,
                           query_tokens=1024,memory_tokens=len(points),roi=item['box'],fresh_rgbd_inference=True))
    return np.asarray(poses),timing,extras,init_ms,points


def error(pred,truth):
    trans=np.linalg.norm(pred[:,:3,3]-truth[:,:3,3],axis=-1)*1000
    dr=pred[:,:3,:3]@truth[:,:3,:3].transpose(0,2,1)
    rot=np.rad2deg(np.arccos(np.clip((np.trace(dr,axis1=1,axis2=2)-1)/2,-1,1)))
    return trans,rot


def video(records,poses,truth,points,errors,path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(640,480))
    for i,(r,pose) in enumerate(zip(records,poses)):
        im=cv2.cvtColor(r['rgb'],cv2.COLOR_RGB2BGR)
        def project(p,t):
            p=p@t[:3,:3].T+t[:3,3];uv=p[:,:2]/p[:,2:3]@np.diag([r['K'][0,0],r['K'][1,1]])+r['K'][:2,2]
            return uv.round().astype(int)
        for p in project(points[:256],pose):
            if 0<=p[0]<640 and 0<=p[1]<480:cv2.circle(im,tuple(p),1,(30,230,255),-1)
        axes=np.array([[0,0,0],[.06,0,0],[0,.06,0],[0,0,.06]],np.float32)
        for t,color in ((truth[i],(200,200,200)),(pose,None)):
            q=project(axes,t)
            for j,c in enumerate(((0,0,255),(0,220,0),(255,100,30)),1):cv2.line(im,tuple(q[0]),tuple(q[j]),c if color is None else color,2)
        cv2.rectangle(im,(0,0),(640,65),(0,0,0),-1)
        cv2.putText(im,f"frame {r['frame']}  {errors[0][i]:.1f}mm / {errors[1][i]:.1f}deg",(10,24),0,.60,(255,255,255),1)
        cv2.putText(im,'Pred: RGB axes / yellow memory. GT: gray (scoring only)',(10,46),0,.45,(255,255,255),1)
        cv2.putText(im,'Sparse BOP keyframes; playback 8fps != sensor fps',(10,62),0,.4,(255,255,255),1)
        writer.write(im)
    writer.release()


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.benchmark=True
    out=ROOT/'evaluation';out.mkdir(exist_ok=True)
    assert not (out/'DONE').exists(),'Do not overwrite completed evaluation'
    model=FastRefiner().cuda().eval();ckpt=ROOT/'training/last.pt'
    model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=False)['model'])
    records=scene(59,False)
    assert sha(records[0]['rgb_path'])==sha(ROOT/'inputs/first_frame.png')
    # Warm kernels then discard every recurrent state and construct memories again.
    run(model,records[:4],'full',warmup=True)
    results={}
    for mode in ('full','no_generated','shuffled_observed'):
        poses,timing,extras,init,points=run(model,records,mode)
        np.savez_compressed(out/f'{mode}_predictions.npz',poses=poses,points=points,frames=[r['frame'] for r in records])
        (out/f'{mode}_timing.json').write_text(json.dumps(timing,indent=2))
        (out/f'{mode}_diagnostics.json').write_text(json.dumps(extras,indent=2))
        results[mode]=(poses,timing,extras,init,points)
        print(json.dumps(dict(event='inference_complete_before_GT',mode=mode,frames=len(poses),latency=stats([x['ms'] for x in timing]))),flush=True)
    pngposes,pngtiming,_,pnginit,_=run(model,records,'full',png=True)
    np.savez_compressed(out/'full_png_predictions.npz',poses=pngposes)
    (out/'png_timing.json').write_text(json.dumps(pngtiming,indent=2))
    # First access to validation poses/masks is AFTER every above prediction file.
    labels=scene(59,True);origin=results['full'][0][0,:3,3]
    truth=np.asarray([target_pose(r,labels[0],origin) for r in labels]);np.save(out/'scoring_truth.npy',truth)
    previous=np.repeat(results['full'][0][:1],len(records),axis=0)
    results['hold_initial_pose']=(previous,[],[],0,results['full'][4])
    report=dict(protocol='scene50/54 supervised; scene59 development validation; 75 sparse keyframes, no CAD',
                initialization='saved human mask; first-camera axes; measured centroid origin',
                validation_GT='loaded only after all prediction files; one fixed initial gauge, no per-frame alignment',
                checkpoint_sha256=sha(ckpt),code_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},
                first_prior=json.loads((ROOT/'priors/scene59.json').read_text()),frames=len(records),metrics_exclude_initial_frame=True,
                variants={},png_to_pose_ms=stats([x['ms'] for x in pngtiming]),
                decoded_vs_png_pose_max_difference=float(np.max(np.abs(results['full'][0]-pngposes))))
    errors={}
    for mode,(poses,timing,extras,init,points) in results.items():
        trans,rot=error(poses,truth);errors[mode]=(trans,rot)
        row=dict(translation_mm=stats(trans[1:]),rotation_deg=stats(rot[1:]),
                 pass_10mm_5deg=float(((trans[1:]<=10)&(rot[1:]<=5)).mean()),reference_init_ms=init)
        if timing:
            row.update(decoded_rgbd_to_cpu_pose_ms=stats([x['ms'] for x in timing]),
                       slowest_frame=max(timing,key=lambda x:x['ms']),
                       stages={k:stats([x[k] for x in timing]) for k in ('roi_xyz_ms','encode_gpu_ms','match2_gpu_ms')})
        report['variants'][mode]=row
        (out/f'{mode}_errors.json').write_text(json.dumps([dict(frame=r['frame'],translation_mm=float(a),rotation_deg=float(b)) for r,a,b in zip(records,trans,rot)],indent=2))
    mainrow=report['variants']['full']
    report['frozen_memory_pass']=mainrow['translation_mm']['p50']<=10 and mainrow['rotation_deg']['p50']<=5
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    for mode in ('full','no_generated','shuffled_observed'):
        video(records,results[mode][0],truth,results[mode][4],errors[mode],out/f'{mode}.mp4')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True)
    ids=[r['frame'] for r in records]
    for mode,(poses,timing,_,_,_) in results.items():
        ax[0].plot(ids,errors[mode][0],label=mode);ax[1].plot(ids,errors[mode][1],label=mode)
        if timing:ax[2].plot(ids[1:],[x['ms'] for x in timing],label=mode)
    for a,label in zip(ax,['Translation error (mm)','Rotation error (deg)','Decoded RGB-D to CPU pose (ms)']):a.set_ylabel(label);a.grid(alpha=.2)
    ax[0].legend(ncol=2);ax[2].set_xlabel('Original BOP frame ID (sparse sampling)');fig.tight_layout();fig.savefig(out/'accuracy_latency.png',dpi=150);plt.close(fig)
    (out/'DONE').write_text('All fresh inference, scoring and video export complete')


if __name__=='__main__':main()
