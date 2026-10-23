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
        convention). Returns the full image with that window's edited result pasted back in."""
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
        edited=self.pipe(prompt,image=Image.fromarray(crop),mask_img=Image.fromarray(mask),
            negative_prompt='',num_inference_steps=self.steps,
            image_guidance_scale=self.image_guidance_scale,guidance_scale=self.guidance_scale,
            generator=generator).images[0]
        edited=np.asarray(edited.resize((window,window)))
        return paste(image,edited,wy0,wx0)

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
