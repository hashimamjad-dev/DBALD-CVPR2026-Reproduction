"""Rollout-aware objective and schedule for the road-restoration student (v3).

Benign restoration only: no detector gradients, poison labels or attack objective.
Motivation (docs/distillation_v2_diagnosis.md): v1/v2 trained on single transitions, so the error of the
student's own four-step rollout accumulated; v3 also trains on that rollout.
"""
import math
import torch
from torch.nn import functional as F
from .distillation import ddim_transition

def latent_mask(conditioning): return F.max_pool2d(conditioning[:,:1],5,1,2)

def region_mean(error,mask): return (error*mask).sum()/(mask.sum()*error.shape[1]).clamp_min(1)

def balanced(error,mask): return .5*error.mean()+.5*region_mean(error,mask)

def cosine_lr(step,total,peak,floor):
    if not 1<=step<=total: raise ValueError('Step outside schedule')
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*(step-1)/max(total-1,1)))

def rollout_loss(model,states,conditioning,timesteps,alphas,weights=(.25,.25,.25,1.)):
    """Student rolls out from the teacher's noise; every coarse state is matched, the final one most strongly."""
    z=states[:,0]; mask=latent_mask(conditioning); total=0.
    for i,w in enumerate(weights):
        z=ddim_transition(z,model(z,conditioning,timesteps[i]),alphas[i],alphas[i+1])
        total=total+w*balanced((z-states[:,i+1]).square(),mask)
    return total/sum(weights)

def training_loss(model,states,conditioning,targets,timesteps,alphas,stage,rollout_weight):
    """states (B,5,4,H,W); targets (B,4,4,H,W) equivalent epsilons; stage (B,) teacher-forced transition index."""
    rows=torch.arange(len(states),device=states.device)
    forced=balanced((model(states[rows,stage],conditioning,timesteps[stage])-targets[rows,stage]).square(),latent_mask(conditioning))
    rolled=rollout_loss(model,states,conditioning,timesteps,alphas) if rollout_weight else forced.new_zeros(())
    return forced+rollout_weight*rolled,dict(forced=float(forced.detach()),rollout=float(rolled.detach()))
