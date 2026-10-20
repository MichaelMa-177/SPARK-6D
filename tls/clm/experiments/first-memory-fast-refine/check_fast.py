"""Math/interface checks, explicitly not evidence of tracking accuracy."""
import json
import cv2
import torch
import geometry as g
from data_fast import ROOT,scene,first_reference,inputs
from train_fast import prior
from fast_model import FastRefiner


torch.manual_seed(0);torch.set_num_threads(4)
p=torch.randn(1,100,3,device='cuda')*.1
t=g.exp(torch.tensor([[.01,.02,.03,.1,-.2,.3]],device='cuda'))
estimate,_,_=g.rigid(p,g.transform(t,p),torch.ones(1,100,device='cuda'))
err=float((estimate-t).abs().max());assert err<1e-4
records=scene(59,False);assert not any('gt' in r or 'mask' in r for r in records)
ref=first_reference(records,cv2.imread(str(ROOT/'inputs/mask.png'),0)>0)
model=FastRefiner().cuda();enc=model.encode(*inputs(ref,'cuda'))
memory=model.memory(enc,ref['indices'][None].cuda(),ref['center'][None].cuda(),*prior(59,'cuda'))
pose=torch.eye(4,device='cuda')[None];pose[:,:3,3]=ref['center'].cuda()
result=model.refine(enc,memory,pose)
loss=result['appearance'].square().mean()+result['member_logits'].square().mean()+result['T'].square().mean()
loss.backward()
grads={key:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
       for key,module in [('rgb',model.rgb),('xyz',model.xyz),('prior',model.prior),('member',model.member)]}
assert all(v>0 for v in grads.values())
row=dict(weighted_svd_max_error=err,encoder_shape=list(enc['f'].shape),memory_shape=list(memory['f'].shape),
         unique_observed_anchors=int(ref['indices'].unique().numel()),gradient_l1=grads,
         validation_GT_loaded=False,tracking_accuracy_verified=False)
(ROOT/'math_checks.json').write_text(json.dumps(row,indent=2));print(json.dumps(row))
