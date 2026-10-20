"""Declared post-failure development ablation; same weights, two fixed radii."""
import json
import numpy as np
import torch
from fast_model import FastRefiner
from data_fast import ROOT,scene,crop,box_from_mask,inputs,target_pose
from evaluate_fast import run,build,stats,error,video


class RadiusRefiner(FastRefiner):
    def __init__(self,radii):
        super().__init__();self.radii=radii

    def refine(self,encoded,memory,predicted):
        first=self.associate(encoded,memory,predicted,self.radii[0])
        return self.associate(encoded,memory,first['T'],self.radii[1])


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.benchmark=True
    out=ROOT/'radius_ablation';out.mkdir(exist_ok=True)
    clean=scene(59,False);results={};models={}
    weights=torch.load(ROOT/'training/last.pt',map_location='cuda',weights_only=False)['model']
    for radii in ((.025,.01),(.015,.0075)):
        model=RadiusRefiner(radii).cuda().eval();model.load_state_dict(weights)
        run(model,clean[:4],'full')
        for mode in ('full','no_generated'):
            name=f'{radii[0]*1000:g}_{radii[1]*1000:g}mm_{mode}'
            poses,timing,extras,init,points=run(model,clean,mode)
            np.savez_compressed(out/f'{name}_predictions.npz',poses=poses,points=points)
            (out/f'{name}_timing.json').write_text(json.dumps(timing,indent=2))
            (out/f'{name}_diagnostics.json').write_text(json.dumps(extras,indent=2))
            results[name]=(poses,timing,points,radii,mode);models[name]=model
            print(json.dumps(dict(event='fresh_inference_complete_before_GT',name=name)),flush=True)
    # Only now read validation labels. These are development comparisons.
    records=scene(59,True);origin=next(iter(results.values()))[0][0,:3,3]
    truth=np.asarray([target_pose(r,records[0],origin) for r in records]);report={}
    for name,(poses,timing,points,radii,mode) in results.items():
        a,b=error(poses,truth)
        memory,_=build(models[name],clean,mode);teacher=[]
        for i,r in enumerate(records[1:],1):
            item=crop(r,box_from_mask(r['mask']),truth[i,:3,3])
            enc=models[name].encode(*inputs(item,'cuda'))
            teacher.append(models[name].refine(enc,memory,torch.from_numpy(truth[i:i+1]).cuda())['T'][0].cpu().numpy())
        ta,tb=error(np.asarray(teacher),truth[1:])
        report[name]=dict(role='post-failure development ablation, not independent validation',radii_m=radii,
                          translation_mm=stats(a[1:]),rotation_deg=stats(b[1:]),
                          pass_10mm_5deg=float(((a[1:]<=10)&(b[1:]<=5)).mean()),
                          latency_ms=stats([x['ms'] for x in timing]),
                          GT_current_ROI_and_state_diagnostic=dict(translation_mm=stats(ta),rotation_deg=stats(tb)))
        (out/f'{name}_errors.json').write_text(json.dumps([dict(frame=r['frame'],translation_mm=float(x),rotation_deg=float(y)) for r,x,y in zip(records,a,b)],indent=2))
        if mode=='full':video(clean,poses,truth,points,(a,b),out/f'{name}.mp4')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    (out/'DONE').write_text('All declared development radii reported; no online memory writes')


if __name__=='__main__':main()
