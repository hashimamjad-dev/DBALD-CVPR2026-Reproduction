import torch
import pytest
from lane_repro.distillation import ddim_transition,equivalent_epsilon,RoadStudent

def test_coarse_diffusion_target_recovers_known_transition():
    generator=torch.Generator().manual_seed(7)
    z=torch.randn(2,4,8,8,generator=generator)
    epsilon=torch.randn(2,4,8,8,generator=generator)
    next_z=ddim_transition(z,epsilon,.03,.7)
    inferred=equivalent_epsilon(z,next_z,.03,.7)
    torch.testing.assert_close(inferred,epsilon,atol=2e-6,rtol=2e-6)
    with pytest.raises(ValueError): equivalent_epsilon(z,z,.5,.5)

def test_student_learns_from_transition_target():
    torch.manual_seed(4); torch.set_num_threads(2)
    model=RoadStudent(width=16)
    z=torch.randn(2,4,16,16); condition=torch.randn(2,5,16,16)
    target=z*.6; optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    first=float((model(z,condition,torch.tensor([201,451]))-target).square().mean().detach())
    for _ in range(8):
        loss=(model(z,condition,torch.tensor([201,451]))-target).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert torch.isfinite(loss) and float(loss)<first


def test_transition_matches_diffusers_ddim_including_last_step():
    from diffusers import DDIMScheduler
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,
        beta_schedule='scaled_linear',clip_sample=False,set_alpha_to_one=False,steps_offset=1)
    scheduler.set_timesteps(20)
    z=torch.randn(1,4,8,8); epsilon=torch.randn_like(z)
    for index in (0,9,19):
        t=scheduler.timesteps[index]
        alpha_s=scheduler.alphas_cumprod[scheduler.timesteps[index+1]] if index<19 else scheduler.final_alpha_cumprod
        expected=scheduler.step(epsilon,t,z,eta=0).prev_sample
        actual=ddim_transition(z,epsilon,scheduler.alphas_cumprod[t],alpha_s)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
