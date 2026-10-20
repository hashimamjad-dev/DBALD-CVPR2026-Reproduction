"""Frozen SD inpainting teacher and shared VAE/safety-checker components."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline,DDIMScheduler
from .distillation import PROMPT,ddim_transition

class RoadTeacher:
    def __init__(self,path):
        self.pipe=StableDiffusionInpaintPipeline.from_pretrained(str(path),torch_dtype=torch.float16,
            variant='fp16',use_safetensors=True,local_files_only=True).to('cuda')
        self.pipe.enable_attention_slicing(); self.pipe.enable_vae_slicing()
        self.scheduler=DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(20,device='cuda')
        self.timesteps=self.scheduler.timesteps
        self.coarse_indices=[0,5,10,15,20]
        self.coarse_t=self.timesteps[[0,5,10,15]].cpu()
        self.coarse_a=torch.stack([self.scheduler.alphas_cumprod[int(self.timesteps[i])] for i in (0,5,10,15)]+[self.scheduler.final_alpha_cumprod]).cpu()
        with torch.inference_mode():
            positive,negative=self.pipe.encode_prompt(PROMPT,'cuda',1,True)
            self.embeddings=torch.cat([negative,positive])
        self.unet_parameters=sum(p.numel() for p in self.pipe.unet.parameters())
        self.shared_parameters=sum(p.numel() for p in self.pipe.vae.parameters())+sum(p.numel() for p in self.pipe.safety_checker.parameters())

    @torch.inference_mode()
    def encode(self,condition):
        image=torch.from_numpy(condition['crop'].transpose(2,0,1).copy()).unsqueeze(0).to('cuda',torch.float16)/127.5-1
        mask=torch.from_numpy(condition['mask'].copy()).unsqueeze(0).unsqueeze(0).to('cuda',torch.float16)/255
        masked=self.pipe.vae.encode(image*(1-mask)).latent_dist.mode()*self.pipe.vae.config.scaling_factor
        return torch.cat([F.interpolate(mask,size=(64,64),mode='nearest'),masked],1)

    @torch.inference_mode()
    def noise(self,seed):
        return torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(seed)).to('cuda',torch.float16)

    @torch.inference_mode()
    def epsilon(self,z,condition,t):
        input=torch.cat([z,condition],1).repeat(2,1,1,1)
        uncond,cond=self.pipe.unet(input,t,encoder_hidden_states=self.embeddings,return_dict=False)[0].chunk(2)
        return uncond+7.5*(cond-uncond)

    @torch.inference_mode()
    def sample(self,conditioning,seed,steps=20,capture=True):
        z=self.noise(seed); states=[z.float().cpu()] if capture else []
        if steps==20:
            for i,t in enumerate(self.timesteps):
                z=self.scheduler.step(self.epsilon(z,conditioning,t),t,z,eta=0,return_dict=False)[0]
                if capture and i+1 in self.coarse_indices[1:]: states.append(z.float().cpu())
        elif steps==4:
            for i,t in enumerate(self.coarse_t):
                epsilon=self.epsilon(z,conditioning,t.to('cuda'))
                z=ddim_transition(z.float(),epsilon.float(),self.coarse_a[i],self.coarse_a[i+1]).half()
                if capture: states.append(z.float().cpu())
        else: raise ValueError('Protocol permits 20 or 4 teacher steps')
        return z,torch.cat(states) if capture else None

    @torch.inference_mode()
    def decode(self,z):
        image=self.pipe.vae.decode(z.to(torch.float16)/self.pipe.vae.config.scaling_factor,return_dict=False)[0]
        image=(image/2+.5).clamp(0,1).permute(0,2,3,1).float().cpu().numpy()
        checked,flags=self.pipe.run_safety_checker(image,'cuda',torch.float16)
        if flags is not None and any(flags): raise RuntimeError('Safety checker rejected generated restoration')
        return (checked[0]*255).round().astype(np.uint8)

    def release_denoiser(self):
        # Keep exactly the same VAE and safety checker for student comparison.
        self.pipe.unet=None; self.pipe.text_encoder=None; self.embeddings=None
        torch.cuda.empty_cache()
