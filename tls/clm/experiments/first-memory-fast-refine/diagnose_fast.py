"""GT-conditioned isolation tests. None is a deployable tracking result."""
import json
import cv2
import numpy as np
import torch
import geometry as g
from data_fast import ROOT,scene,crop,box_from_mask,target_pose,inputs,projected_box
from fast_model import FastRefiner
from evaluate_fast import build,error,stats,run


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.benchmark=True
    out=ROOT/'diagnosis';out.mkdir(exist_ok=True)
    model=FastRefiner().cuda().eval()
    model.load_state_dict(torch.load(ROOT/'training/last.pt',map_location='cuda',weights_only=False)['model'])
    records=scene(59,True)  # Explicitly GT-conditioned diagnostics, NOT tracking.
    memory,origin=build(model,records,'full');points=memory['p'][0].cpu().numpy()
    truth=np.asarray([target_pose(r,records[0],origin[:3,3]) for r in records])
    trajectories={k:[origin.copy()] for k in ('GT_previous_state','GT_current_ROI_and_state','appearance_only_GT_ROI','oracle_3D_pairs_GT_ROI')}
    rows=[]
    for i,r in enumerate(records[1:],1):
        gt=torch.from_numpy(truth[i:i+1]).cuda()
        prev=torch.from_numpy(truth[i-1:i]).cuda()
        # Reset previous state every frame. GT current pose is not used here.
        box=projected_box(points,truth[i-1],r['K'],r['depth'].shape)
        enc=model.encode(*inputs(crop(r,box,truth[i-1,:3,3]),'cuda'))
        trajectories['GT_previous_state'].append(model.refine(enc,memory,prev)['T'][0].cpu().numpy())
        item=crop(r,box_from_mask(r['mask']),truth[i,:3,3],r['mask'])
        enc=model.encode(*inputs(item,'cuda'))
        refined=model.refine(enc,memory,gt)
        appearance=model.associate(enc,memory,gt,geometric=False)
        trajectories['GT_current_ROI_and_state'].append(refined['T'][0].cpu().numpy())
        trajectories['appearance_only_GT_ROI'].append(appearance['T'][0].cpu().numpy())
        fg=(item['mask'][::4,::4].flatten()[None].cuda()>.5)&(enc['v']>.5)
        # Measured-memory-only oracle; never use generated geometry as truth.
        distances=torch.cdist(enc['p'],g.transform(gt,memory['p'][:,:256]))
        closest,ids=distances.min(-1);support=fg&(closest<.015)
        src=memory['p'][:,:256].gather(1,ids[...,None].expand(-1,-1,3))
        oracle,_,singular=g.rigid(src,enc['p'],support.float())
        trajectories['oracle_3D_pairs_GT_ROI'].append(oracle[0].cpu().numpy())
        # Appearance-only hit, including out-of-surface cases separately.
        pick=appearance['appearance'][...,:256].argmax(-1)
        chosen=distances.gather(-1,pick[...,None]).squeeze(-1)
        predmask=refined['membership']>.5
        rows.append(dict(frame=r['frame'],foreground_queries=int(fg.sum()),oracle_pairs=int(support.sum()),
                         measured_surface_coverage=float(support.sum()/fg.sum().clamp_min(1)),
                         descriptor_hit15mm_on_covered=float(((chosen<.015)&support).sum()/support.sum().clamp_min(1)),
                         mask_iou_GT_ROI=float((fg&predmask).sum()/(fg|predmask).sum().clamp_min(1)),
                         generated_attention_mean=float(refined['association'][0,:,256:].sum(-1).mean()),
                         oracle_singular=singular[0].cpu().tolist()))
    report=dict(role='GT-conditioned failure diagnosis, NOT valid deployment accuracy',variants={},
                observed_memory_slots=256,unique_observed_points=int(np.unique(points[:256],axis=0).shape[0]))
    for name,poses in trajectories.items():
        poses=np.asarray(poses);a,b=error(poses,truth)
        report['variants'][name]=dict(translation_mm=stats(a[1:]),rotation_deg=stats(b[1:]))
        np.savez_compressed(out/f'{name}.npz',poses=poses,translation_mm=a,rotation_deg=b)
    for key in ('measured_surface_coverage','descriptor_hit15mm_on_covered','mask_iou_GT_ROI','generated_attention_mean'):
        report[key]=stats([x[key] for x in rows])
    (out/'per_frame.json').write_text(json.dumps(rows,indent=2));(out/'report.json').write_text(json.dumps(report,indent=2))
    # Repeat true fresh inference with all GT fields removed. No oracle carry-over.
    clean=[{k:v for k,v in r.items() if k not in ('mask','gt')} for r in records]
    repeated=[]
    for n in range(5):
        _,timing,_,init,_=run(model,clean,'full')
        repeated.append(dict(repetition=n,reference_init_ms=init,frames=timing))
    allrows=[x for rep in repeated for x in rep['frames']]
    timing_report=dict(repetitions=5,post_init_frames=len(allrows),decoded_rgbd_to_cpu_pose_ms=stats([x['ms'] for x in allrows]),
                       slowest=max(allrows,key=lambda x:x['ms']),all_runs_reinitialized=True,steady_state_accuracy_unchanged=True)
    (out/'repeat_latency.json').write_text(json.dumps(repeated,indent=2));(out/'repeat_latency_summary.json').write_text(json.dumps(timing_report,indent=2))
    print(json.dumps(report),flush=True);print(json.dumps(timing_report),flush=True)
    (out/'DONE').write_text('GT diagnosis and independent fresh latency repeats complete')


if __name__=='__main__':main()
