"""Fixed-budget, scene50/54-supervised correspondence training; no scene59 GT."""
import json
import time
import random
import numpy as np
import torch
from torch.nn import functional as F
import geometry as g
from fast_model import FastRefiner
from data_fast import ROOT,scene,first_reference,crop,box_from_mask,target_pose,inputs,sha


def prior(number,device):
    a=np.load(ROOT/f'priors/scene{number}.npz')
    return [torch.from_numpy(a[k]).float()[None].to(device) for k in ('points','features')]


def main():
    torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    out=ROOT/'training';out.mkdir(exist_ok=True)
    assert not (out/'DONE').exists(),'Completed training must not be overwritten'
    model=FastRefiner().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    references={};examples=[]
    for number in (50,54):
        records=scene(number,True);ref=first_reference(records,records[0]['mask'])
        center=ref['center'].numpy();references[number]=(inputs(ref,'cuda'),ref['indices'][None].cuda(),prior(number,'cuda'))
        for r in records:
            truth=target_pose(r,records[0],center)
            item=crop(r,box_from_mask(r['mask']),truth[:3,3],r['mask'])
            examples.append((number,inputs(item,'cuda'),item['mask'][::4,::4].flatten()[None].cuda(),torch.from_numpy(truth)[None].cuda()))
    print(json.dumps(dict(event='data_ready',scenes=[50,54],frames=len(examples),validation_annotations_loaded=False,
                          parameters=sum(p.numel() for p in model.parameters()),train_steps=1200,batch=4)),flush=True)
    start=time.perf_counter();history=[];best=float('inf')
    for step in range(1200):
        chosen=random.choices(examples,k=4);numbers=[e[0] for e in chosen]
        rgb,xyz,valid,center=[torch.cat([e[1][j] for e in chosen]) for j in range(4)]
        mask=torch.cat([e[2] for e in chosen]);truth=torch.cat([e[3] for e in chosen])
        ref_inputs=[torch.cat([references[n][0][j] for n in numbers]) for j in range(4)]
        indices=torch.cat([references[n][1] for n in numbers]);gp,gf=[torch.cat([references[n][2][j] for n in numbers]) for j in range(2)]
        reference=model.encode(*ref_inputs);memory=model.memory(reference,indices,ref_inputs[3],gp,gf)
        rotation=g.exp(torch.cat((torch.zeros(4,3,device='cuda'),torch.randn(4,3,device='cuda')*.10),-1))
        predicted=truth.clone();predicted[:,:3,:3]=rotation[:,:3,:3]@truth[:,:3,:3]
        predicted[:,:3,3]+=torch.randn(4,3,device='cuda')*.015
        encoded=model.encode(rgb,xyz,valid,predicted[:,:3,3])
        result=model.associate(encoded,memory,predicted)
        distances=torch.cdist(encoded['p'],g.transform(truth,memory['p']))
        nearest=distances.min(-1).values
        pos=(mask>.5)&(encoded['v']>.5)&(nearest<.025)
        neg=(mask<.5)&(encoded['v']>.5)
        target=torch.exp(-distances.square()/(2*.0075**2))*(distances<.025)
        target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
        logprob=result['appearance'].log_softmax(-1)
        positive=(-(target*logprob[...,:-1]).sum(-1)*pos).sum()/pos.sum().clamp_min(1)
        negative=(-logprob[...,-1]*neg).sum()/neg.sum().clamp_min(1)
        bce=F.binary_cross_entropy_with_logits(result['member_logits'],mask,reduction='none')
        foreground=(mask>.5)&(encoded['v']>.5)
        segmentation=(bce*foreground).sum()/foreground.sum().clamp_min(1)+(bce*neg).sum()/neg.sum().clamp_min(1)
        loss=positive+.25*negative+segmentation
        pose=loss.new_zeros(())
        if step>=800:
            result=model.refine(encoded,memory,predicted)
            pose=(result['T'][:,:3,3]-truth[:,:3,3]).abs().mean()/.02
            pose=pose+(result['T'][:,:3,:3]-truth[:,:3,:3]).square().mean()*5
            loss=loss+pose
        assert torch.isfinite(loss),'Non-finite training loss'
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
        if step%25==0 or step==1199:
            with torch.no_grad():
                pick=result['appearance'][...,:-1].argmax(-1)
                distance=distances.gather(-1,pick[...,None]).squeeze(-1)
                hit=((distance<.015)&pos).sum()/pos.sum().clamp_min(1)
            row=dict(step=step+1,loss=loss.item(),appearance_positive=positive.item(),appearance_negative=negative.item(),
                     member=segmentation.item(),pose=pose.item(),hit15mm=hit.item(),supported=pos.sum().item(),
                     grad_norm=grad.item(),elapsed_s=time.perf_counter()-start)
            history.append(row);print(json.dumps(row),flush=True)
        if (step+1)%100==0:
            ckpt=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,seed=0,train_scenes=[50,54],
                      architecture='FastRefiner64_RGBXYZ128_observed256_SS96',source_sha256=sha(ROOT/'fast_model.py'))
            torch.save(ckpt,out/'last.pt')
            if step==799:torch.save(ckpt,out/'descriptor.pt')
            if step>=800 and float(loss.detach())<best:
                best=float(loss.detach());torch.save(ckpt,out/'best_train_loss.pt')
            (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'DONE').write_text('1200 fixed updates complete; evaluation uses last.pt, not scene59 selection')


if __name__=='__main__':main()
