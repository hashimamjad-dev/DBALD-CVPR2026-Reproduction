"""Benign road-restoration diffusion distillation primitives.

No lane-detector gradients, target lanes, poisoned labels or attack objectives.
"""
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROMPT='Photorealistic ordinary gray asphalt road surface, consistent with the surrounding road, natural daylight, preserve existing lane markings.'

def conditions(record,root):
    image=cv2.imread(str(Path(root)/record['root']/record['raw_file']))
    if image is None: raise FileNotFoundError(record['raw_file'])
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    h,w=image.shape[:2]; x=(w-512)//2; y=h-512
    crop=image[y:y+512,x:x+512].copy()
    lanes=np.zeros((h,w),np.uint8)
    for lane in record['lanes']:
        if record['dataset']=='TuSimple': pts=[[a,b] for a,b in zip(lane,record['h_samples']) if a>=0]
        else: pts=lane
        if len(pts)>1: cv2.polylines(lanes,[np.asarray(pts,np.int32)],False,1,5)
    protected=cv2.dilate(lanes,np.ones((25,25),np.uint8))[y:y+512,x:x+512]
    mask=np.zeros((512,512),np.uint8)
    # CULane's camera includes the vehicle hood. Fix a narrower strip above it;
    # this dataset-specific rule is established in input QA, before inference.
    if record['dataset']=='CULane': mask[312:328,256:352]=255
    else: mask[352:448,192:320]=255
    mask[protected>0]=0
    if (mask>0).sum()<1024: raise ValueError('Fixed mask has insufficient unprotected road area')
    return dict(original=image,crop=crop,mask=mask,lane_mask=lanes,crop_xy=(x,y))

def composite(condition,generated):
    result=condition['original'].copy(); x,y=condition['crop_xy']
    view=result[y:y+512,x:x+512]; valid=condition['mask']>0
    view[valid]=generated[valid]
    return result

def ddim_transition(z,epsilon,alpha_t,alpha_s):
    """Deterministic eta=0 transition, alpha arguments are cumulative products."""
    a=torch.as_tensor(alpha_t,device=z.device,dtype=z.dtype).sqrt()
    b=(1-torch.as_tensor(alpha_t,device=z.device,dtype=z.dtype)).sqrt()
    c=torch.as_tensor(alpha_s,device=z.device,dtype=z.dtype).sqrt()
    d=(1-torch.as_tensor(alpha_s,device=z.device,dtype=z.dtype)).sqrt()
    return c/a*z+(d-c*b/a)*epsilon

def equivalent_epsilon(z,z_next,alpha_t,alpha_s):
    a=torch.as_tensor(alpha_t,dtype=z.dtype,device=z.device).sqrt()
    b=(1-torch.as_tensor(alpha_t,dtype=z.dtype,device=z.device)).sqrt()
    c=torch.as_tensor(alpha_s,dtype=z.dtype,device=z.device).sqrt()
    d=(1-torch.as_tensor(alpha_s,dtype=z.dtype,device=z.device)).sqrt()
    denominator=d-c*b/a
    if torch.any(denominator.abs()<1e-8): raise ValueError('Degenerate diffusion transition')
    return (z_next-c/a*z)/denominator

class Residual(nn.Module):
    def __init__(self,cin,cout,time_dim=192):
        super().__init__()
        self.norm1=nn.GroupNorm(8,cin); self.conv1=nn.Conv2d(cin,cout,3,padding=1)
        self.time=nn.Linear(time_dim,cout); self.norm2=nn.GroupNorm(8,cout)
        self.conv2=nn.Conv2d(cout,cout,3,padding=1)
        self.skip=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1)
    def forward(self,x,t):
        h=self.conv1(F.silu(self.norm1(x)))+self.time(F.silu(t))[:,:,None,None]
        return self.skip(x)+self.conv2(F.silu(self.norm2(h)))

class RoadStudent(nn.Module):
    """Compact conditional epsilon model; 9 input channels, four diffusion steps."""
    def __init__(self,width=48):
        super().__init__(); self.width=width
        self.time=nn.Sequential(nn.Linear(64,192),nn.SiLU(),nn.Linear(192,192))
        self.input=nn.Conv2d(9,width,3,padding=1)
        self.a=Residual(width,width); self.down1=nn.Conv2d(width,width*2,3,stride=2,padding=1)
        self.b=Residual(width*2,width*2); self.down2=nn.Conv2d(width*2,width*4,3,stride=2,padding=1)
        self.mid1=Residual(width*4,width*4); self.mid2=Residual(width*4,width*4)
        self.up1=Residual(width*6,width*2); self.up2=Residual(width*3,width)
        self.out=nn.Conv2d(width,4,3,padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self,z,conditioning,t):
        t=torch.as_tensor(t,device=z.device).reshape(-1).float()
        if len(t)==1: t=t.expand(z.shape[0])
        frequencies=torch.exp(-math.log(10000)*torch.arange(32,device=z.device)/31)
        angles=t[:,None]*frequencies[None,:]
        embedding=self.time(torch.cat([angles.sin(),angles.cos()],1).to(z.dtype))
        a=self.a(self.input(torch.cat([z,conditioning],1)),embedding)
        b=self.b(self.down1(a),embedding)
        h=self.mid2(self.mid1(self.down2(b),embedding),embedding)
        h=self.up1(torch.cat([F.interpolate(h,size=b.shape[-2:],mode='nearest'),b],1),embedding)
        h=self.up2(torch.cat([F.interpolate(h,size=a.shape[-2:],mode='nearest'),a],1),embedding)
        return z+self.out(h)

def student_rollout(model,initial,conditioning,timesteps,alphas):
    z=initial
    for i,t in enumerate(timesteps):
        z=ddim_transition(z,model(z,conditioning,t),alphas[i],alphas[i+1])
    return z
