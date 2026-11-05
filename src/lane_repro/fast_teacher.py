"""Training-free variants of the frozen SD1.5 inpainting teacher: window crop, optional guidance, N-step DDIM.

Same weights, prompt, VAE and safety checker as RoadTeacher; no student, poisoning or attack objective.
"""
import torch
from torch.nn import functional as F
from .distillation import ddim_transition
from .road_teacher import RoadTeacher
from .crop_geometry import schedule_indices

class FastTeacher(RoadTeacher):
    def __init__(self,path):
        super().__init__(path)
        self.alphas=self.scheduler.alphas_cumprod.to('cuda')
        self.final_alpha=self.scheduler.final_alpha_cumprod

    @torch.inference_mode()
    def encode_window(self,condition,box,size):
        y0,x0=box
        image=torch.from_numpy(condition['crop'][y0:y0+size,x0:x0+size].transpose(2,0,1).copy()).unsqueeze(0).to('cuda',torch.float16)/127.5-1
        mask=torch.from_numpy(condition['mask'][y0:y0+size,x0:x0+size].copy()).unsqueeze(0).unsqueeze(0).to('cuda',torch.float16)/255
        masked=self.pipe.vae.encode(image*(1-mask)).latent_dist.mode()*self.pipe.vae.config.scaling_factor
        return torch.cat([F.interpolate(mask,size=(size//8,size//8),mode='nearest'),masked],1)

    @torch.inference_mode()
    def window_noise(self,seed,box,size):
        """Slice of the full 64x64 noise, so every window shares the noise values of the 512 baseline."""
        y0,x0=box
        return self.noise(seed)[:,:,y0//8:(y0+size)//8,x0//8:(x0+size)//8].contiguous()

    @torch.inference_mode()
    def guided_epsilon(self,z,condition,t,guidance):
        if guidance==1.0:
            return self.pipe.unet(torch.cat([z,condition],1),t,encoder_hidden_states=self.embeddings[1:],return_dict=False)[0]
        return self.epsilon(z,condition,t) if guidance==7.5 else self._cfg(z,condition,t,guidance)

    @torch.inference_mode()
    def _cfg(self,z,condition,t,guidance):
        uncond,cond=self.pipe.unet(torch.cat([z,condition],1).repeat(2,1,1,1),t,encoder_hidden_states=self.embeddings,return_dict=False)[0].chunk(2)
        return uncond+guidance*(cond-uncond)

    @torch.inference_mode()
    def sample_fast(self,conditioning,noise,steps,guidance):
        indices=schedule_indices(steps); z=noise
        alpha=lambda i: self.alphas[int(self.timesteps[i])] if i<20 else torch.as_tensor(self.final_alpha,device='cuda')
        for a,b in zip(indices[:-1],indices[1:]):
            epsilon=self.guided_epsilon(z,conditioning,self.timesteps[a],guidance)
            z=ddim_transition(z.float(),epsilon.float(),alpha(a),alpha(b)).half()
        return z
