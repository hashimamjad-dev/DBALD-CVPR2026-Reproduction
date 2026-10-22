import pytest
import torch
from lane_repro.distillation import RoadStudent,equivalent_epsilon
from lane_repro.distill_v3 import cosine_lr,rollout_loss,training_loss,balanced,latent_mask

T=torch.tensor([951.,701.,451.,201.]); A=torch.tensor([.01,.1,.4,.8,.99])

class Oracle(torch.nn.Module):
    """Returns the exact equivalent epsilon of each coarse transition, so a perfect rollout has zero loss."""
    def __init__(self,states): super().__init__(); self.states=states; self.call=0
    def forward(self,z,conditioning,t):
        i=self.call%4; self.call+=1
        return equivalent_epsilon(z,self.states[:,i+1],A[i],A[i+1])

def batch(n=2,size=16):
    torch.manual_seed(0)
    states=torch.randn(n,5,4,size,size); conditioning=torch.rand(n,5,size,size)
    targets=torch.stack([equivalent_epsilon(states[:,i],states[:,i+1],A[i],A[i+1]) for i in range(4)],1)
    return states,conditioning,targets

def test_cosine_lr_endpoints_and_monotone():
    values=[cosine_lr(s,100,2e-4,1e-5) for s in range(1,101)]
    assert values[0]==pytest.approx(2e-4) and values[-1]==pytest.approx(1e-5) and all(a>=b for a,b in zip(values,values[1:]))
    with pytest.raises(ValueError): cosine_lr(0,100,1,0)

def test_rollout_loss_zero_for_perfect_student():
    states,conditioning,_=batch()
    assert float(rollout_loss(Oracle(states),states,conditioning,T,A))==pytest.approx(0,abs=1e-8)

def test_balanced_weights_region_more_than_background():
    conditioning=torch.zeros(1,5,16,16); conditioning[:,0,6:9,6:9]=1; mask=latent_mask(conditioning)
    inside=torch.zeros(1,4,16,16); inside[:,:,7,7]=1; outside=torch.zeros(1,4,16,16); outside[:,:,0,0]=1
    assert balanced(inside,mask)>balanced(outside,mask)

def test_training_loss_backward_with_real_student():
    states,conditioning,targets=batch(); model=RoadStudent(8)
    loss,parts=training_loss(model,states,conditioning,targets,T,A,torch.tensor([0,3]),1.)
    loss.backward()
    assert torch.isfinite(loss) and parts['rollout']>0 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_rollout_weight_zero_skips_rollout():
    states,conditioning,targets=batch(); model=RoadStudent(8)
    _,parts=training_loss(model,states,conditioning,targets,T,A,torch.tensor([1,2]),0.)
    assert parts['rollout']==0
