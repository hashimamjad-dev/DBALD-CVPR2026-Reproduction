"""DBALD's actual trigger generator (UltraEdit, StableDiffusion3InstructPix2PixPipeline), to replace
poison_pipeline.paste_placeholder_trigger's flat-colour patch. Matches poison_pipeline.poison_sample's
render_trigger(image, y0, x0, size, prompt) -> image signature, so it is a drop-in swap at its call
site, per that module's docstring -- not a pipeline rewrite.

VRAM risk, read before running on an 8 GB GPU: a standard SD3 pipeline loads three text encoders
(CLIP-L, CLIP-G, T5-XXL). T5-XXL alone is roughly 9-10 GB in fp16 -- likely more than the whole
budget by itself. drop_t5=True (the default here) loads without it (text_encoder_3=None,
tokenizer_3=None), a standard, documented SD3 VRAM-reduction option that trades some prompt-following
quality for fitting in far less memory; this is NOT verified against the full-T5 pipeline's output on
this project's prompts, and should be noted as a deviation if used for a reported result. Run
feasibility_check() first -- it loads the model and times one tiny edit -- before any real run, so an
OOM is caught in seconds, not after poisoning has already started on a batch.

UltraEdit's native resolution is 512x512 (per its own usage examples). This module edits a context
window around the trigger (not the whole, much larger, TuSimple/CULane frame) and reuses
crop_geometry.paste (the same compositing helper poison_pipeline.py and the SD1.5 distillation code
already share) to put the edited window back into the full image.

Mask-composited paste-back (fixed): the SD3 InstructPix2Pix pipeline conditions on the whole crop
through the UNet even with mask_img supplied, and is not guaranteed to leave pixels outside the mask
byte-for-byte unchanged (this is exactly the artifact the paper's own L_lane/L_env losses exist to
suppress -- see Figure 4 of the paper, "vanilla masked diffusion"). __call__ explicitly composites:
only pixels inside the trigger mask are taken from the model's output; everything else in the window
is copied verbatim from the original crop before pasting back. Without this, any outside-mask drift
from the pipeline silently leaks into the "poisoned" image, contaminating the clean-vs-poisoned ACC
comparison with noise unrelated to the trigger itself.

Stealth losses L_lane / L_env (paper §4.3) -- STATUS: the loss math is implemented and tested in
diffusion_losses.py, and `compute_stealth_losses` below scores any decoded edit with it (usable now,
CPU, to quantify how much an edit disturbs lanes/environment). The paper APPLIES these losses *during*
denoising (supplementary §1: "back-propagate through the denoising steps") to guide the latent; that
latent-guided loop needs the SD3 pipeline's internals and a GPU and is NOT yet active -- it is the
clearly-marked stub `latent_guided_edit`. So __call__ today produces a trigger WITHOUT loss guidance
(the "vanilla masked diffusion" branch of the paper's Figure 4); the composite above is the only
outside-mask protection until latent_guided_edit is built. Any number produced now must be labelled
"without L_lane/L_env" (SOURCES.md §7).
"""
import numpy as np
from PIL import Image
from .crop_geometry import paste

CHECKPOINT='BleachNick/SD3_UltraEdit_w_mask'
MUD_PROMPT='Add a small and sparse brown mud'
CONE_PROMPT='Add a small traffic cone'

class UltraEditTrigger:
    def __init__(self,checkpoint=CHECKPOINT,device='cuda',drop_t5=True,cpu_offload=False,
            edit_window=512,steps=50,image_guidance_scale=1.5,guidance_scale=7.5):
        import torch
        from diffusers import StableDiffusion3InstructPix2PixPipeline
        kwargs=dict(torch_dtype=torch.float16)
        if drop_t5: kwargs.update(text_encoder_3=None,tokenizer_3=None)
        self.pipe=StableDiffusion3InstructPix2PixPipeline.from_pretrained(checkpoint,**kwargs)
        if cpu_offload: self.pipe.enable_model_cpu_offload()
        else: self.pipe=self.pipe.to(device)
        self.pipe.enable_attention_slicing()
        self.edit_window=edit_window; self.steps=steps
        self.image_guidance_scale=image_guidance_scale; self.guidance_scale=guidance_scale
        self.drop_t5=drop_t5

    def __call__(self,image,y0,x0,size,prompt,seed=None):
        """image: full-resolution RGB uint8 array. y0,x0,size: the trigger footprint (as chosen by
        trigger_placement/poison_pipeline), NOT the edit window -- a larger context window around it
        is what actually gets sent to UltraEdit, matching its 512x512-native resolution, and only the
        trigger footprint within that window is masked (white=edit, matching the README's mask_img
        convention). Returns the full image with that window's edited result pasted back in, with
        everything outside the trigger footprint forced identical to the original crop (see module
        docstring: the pipeline itself does not guarantee this)."""
        height,width=image.shape[:2]
        window=min(self.edit_window,height,width)
        wy0=max(0,min(y0+size//2-window//2,height-window))
        wx0=max(0,min(x0+size//2-window//2,width-window))
        if not (wy0<=y0 and wx0<=x0 and wy0+window>=y0+size and wx0+window>=x0+size):
            raise ValueError('Trigger footprint does not fit inside the edit window; reduce trigger size or enlarge edit_window')
        crop=image[wy0:wy0+window,wx0:wx0+window]
        mask=np.zeros((window,window),np.uint8)
        mask[y0-wy0:y0-wy0+size,x0-wx0:x0-wx0+size]=255
        import torch
        generator=torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        edited=self.pipe(prompt,image=Image.fromarray(crop),mask_img=Image.fromarray(mask).convert('RGB'),
            negative_prompt='',num_inference_steps=self.steps,
            image_guidance_scale=self.image_guidance_scale,guidance_scale=self.guidance_scale,
            generator=generator).images[0]
        edited=np.asarray(edited.resize((window,window)))
        mask_bool=mask[...,None]>0
        composed=np.where(mask_bool,edited,crop).astype(np.uint8)
        return paste(image,composed,wy0,wx0)

    @staticmethod
    def compute_stealth_losses(clean_image,edited_image,lanes_xy,trigger_window,
            w_lane=1.0,w_env=1.0,lane_thickness=12):
        """Score a decoded edit with DBALD's L_lane/L_env (diffusion_losses.stealth_loss). Pure,
        model-free, CPU -- usable right now to measure how much a produced trigger disturbs lane
        markings (L_lane) and the surrounding scene (L_env), e.g. to compare the placeholder, a
        vanilla UltraEdit edit, and (later) a loss-guided edit on the SAME image.

        clean_image, edited_image: full-resolution RGB uint8 arrays, identical shape. lanes_xy:
        ground-truth lanes as lists of (x,y) pixel points (x<0 absent), for the lane mask.
        trigger_window: dict(y0,x0,size) or (y0,x0,size); defines the env-mask hole. Returns a dict
        of python floats {l_lane, l_env, total}. env_mask is the documented proxy (everything but the
        trigger and lanes) pending a real object segmenter -- see diffusion_losses.env_mask_proxy."""
        from . import diffusion_losses as dl
        clean_image=np.asarray(clean_image); edited_image=np.asarray(edited_image)
        if clean_image.shape!=edited_image.shape: raise ValueError('clean/edited shape mismatch')
        h,w=clean_image.shape[:2]
        lane_mask=dl.lane_mask_from_lanes(lanes_xy,h,w,lane_thickness)
        env_mask=dl.env_mask_proxy(h,w,trigger_window,lane_mask)
        clean_t,edited_t=dl.to_tensor(clean_image),dl.to_tensor(edited_image)
        out=dl.stealth_loss(edited_t,clean_t,lane_mask,env_mask,w_lane=w_lane,w_env=w_env)
        return {k:float(v) for k,v in out.items()}

    def latent_guided_edit(self,*args,**kwargs):
        """STUB (not yet active): the paper's loss-GUIDED generation, where L_lane/L_env are applied
        during denoising and back-propagated through the diffusion steps (supplementary §1) to steer
        the latent, rather than scored after the fact. Building it means intercepting the SD3
        InstructPix2Pix pipeline's per-step latent (decoding to image space, computing
        diffusion_losses.stealth_loss against the clean crop and its lane/env masks, and adding the
        gradient to the noise prediction), which requires the pipeline internals and a GPU. Until then
        use __call__ (vanilla masked diffusion) for generation and compute_stealth_losses to score it.
        Raising rather than silently falling back, so no run mistakes this for the real guided path."""
        raise NotImplementedError(
            'latent_guided_edit is not implemented: L_lane/L_env are available as a post-hoc score '
            '(compute_stealth_losses); guiding the latent through denoising needs the SD3 pipeline '
            'internals and a GPU. Use __call__ for now and label results "without L_lane/L_env".')

def feasibility_check(checkpoint=CHECKPOINT,device='cuda',drop_t5=True,cpu_offload=False):
    """Loads the model and runs one tiny edit; reports peak VRAM and elapsed time. Run this before
    any real poisoning batch -- it turns a possible multi-minute-then-OOM failure into a fast,
    isolated check. Returns a dict; raises whatever the pipeline itself raises (e.g. OutOfMemoryError)
    rather than swallowing it, since the failure mode itself is the useful information here."""
    import time
    import torch
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    trigger=UltraEditTrigger(checkpoint,device,drop_t5,cpu_offload,edit_window=512,steps=10)
    load_seconds=time.perf_counter()-start
    image=np.full((512,512,3),120,np.uint8)
    start=time.perf_counter()
    result=trigger(image,236,236,40,MUD_PROMPT,seed=0)
    edit_seconds=time.perf_counter()-start
    return dict(status='ok',drop_t5=drop_t5,cpu_offload=cpu_offload,load_seconds=load_seconds,
        edit_seconds=edit_seconds,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        output_shape=result.shape)

if __name__=='__main__':
    import json,sys
    print(json.dumps(feasibility_check(drop_t5='--full-t5' not in sys.argv,cpu_offload='--cpu-offload' in sys.argv),indent=2))
