"""Frozen first-frame memories + fresh local RGB-XYZ learned correspondence."""
import torch
from torch import nn
from torch.nn import functional as F
import geometry as g


def block(a,b,stride=1):
    return nn.Sequential(nn.Conv2d(a,b,3,stride,1),nn.GroupNorm(8,b),nn.GELU())


class FastRefiner(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.rgb=nn.Sequential(block(3,32,2),block(32,width,2))
        self.xyz=nn.Sequential(block(4,32,2),block(32,width,2))
        self.fuse=nn.Sequential(nn.Conv2d(width*2,width,1),nn.GELU(),block(width,width))
        self.query=nn.Linear(width,width,bias=False)
        self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.prior=nn.Sequential(nn.Linear(11,width),nn.GELU(),nn.Linear(width,width),nn.LayerNorm(width))
        self.member=nn.Sequential(nn.Linear(width*2+2,width),nn.GELU(),nn.Linear(width,1))
        self.null=nn.Parameter(torch.tensor(3.))
        nn.init.eye_(self.query.weight);nn.init.eye_(self.key.weight);nn.init.eye_(self.value.weight)

    def encode(self,rgb,xyz,valid,center):
        local=(xyz-center[:,:,None,None])/.15
        a=self.rgb(rgb-.5);b=self.xyz(torch.cat((local.clamp(-4,4),valid),1))
        f=self.fuse(torch.cat((a,b),1)).flatten(2).transpose(1,2)
        p=F.interpolate(xyz,(32,32),mode='nearest').flatten(2).transpose(1,2)
        v=F.interpolate(valid,(32,32),mode='nearest').flatten(1)
        return dict(f=f,p=p,v=v)

    def memory(self,encoded,indices,center,prior_points,prior_feats):
        ids=indices[:,:,None]
        f=encoded['f'].gather(1,ids.expand(-1,-1,encoded['f'].shape[-1]))
        p=encoded['p'].gather(1,ids.expand(-1,-1,3))-center[:,None]
        gp=prior_points-center[:,None]
        gf=self.prior(torch.cat((prior_feats,gp/.15),-1))
        return dict(p=torch.cat((p,gp),1),f=torch.cat((f,gf),1),observed=p.shape[1])

    def cache(self,memory):
        return dict(memory,k=F.normalize(self.key(memory['f']),dim=-1),v=self.value(memory['f']))

    def associate(self,encoded,memory,predicted,sigma=.04,geometric=True):
        q=F.normalize(self.query(encoded['f']),dim=-1)
        k=memory.get('k')
        if k is None:k=F.normalize(self.key(memory['f']),dim=-1)
        similarity=q@k.transpose(-1,-2)*12
        dst=g.transform(predicted,memory['p'])
        dist2=torch.cdist(encoded['p'],dst).square()
        logits=similarity-dist2/(2*sigma*sigma) if geometric else similarity
        credit=logits.new_zeros(logits.shape[-1]);credit[memory['observed']:]=-2.
        logits=logits+credit
        null=self.null.expand(*logits.shape[:-1],1)
        a=torch.cat((logits,null),-1).softmax(-1)[...,:-1]
        v=memory.get('v')
        if v is None:v=self.value(memory['f'])
        context=a@v
        auxiliary=torch.stack((similarity.max(-1).values/12,dist2.min(-1).values.clamp_min(1e-10).sqrt()/.15),-1)
        membership_logits=self.member(torch.cat((encoded['f'],context,auxiliary),-1)).squeeze(-1)
        membership=membership_logits.sigmoid()*encoded['v']
        weights,ids=a.topk(3,-1)
        weights=weights*membership[...,None]
        source=memory['p'][:,None].expand(-1,len(encoded['p'][0]),-1,-1).gather(2,ids[...,None].expand(-1,-1,-1,3))
        dest=encoded['p'][:,:,None].expand_as(source)
        estimate,residual,singular=g.rigid(source.flatten(1,2),dest.flatten(1,2),weights.flatten(1))
        return dict(T=estimate,residual=residual,singular=singular,membership=membership,
                    member_logits=membership_logits,appearance=torch.cat((similarity+credit,null),-1),
                    association=a,weight_sum=weights.sum((1,2)))

    def refine(self,encoded,memory,predicted):
        result=self.associate(encoded,memory,predicted,.05)
        result=self.associate(encoded,memory,result['T'],.025)
        return result
