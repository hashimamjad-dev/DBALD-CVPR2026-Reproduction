"""Fixed-mask ordinary diffusion inpainting; no gradient-guided edits or training."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from lane_repro.prototype import consistency
from lane_repro.models import LaneATTAdapter,sample_predictions

def main():
    torch.set_num_threads(4); torch.manual_seed(0)
    out=Path('results/overnight/diffusion'); out.mkdir(parents=True,exist_ok=True)
    record=json.loads(Path('datasets/raw/tusimple_subset/labels.jsonl').read_text().splitlines()[0])
    original=cv2.cvtColor(cv2.imread(str(Path('datasets/raw/tusimple_subset')/record['raw_file'])),cv2.COLOR_BGR2RGB)
    # Fixed 512px crop and mask chosen independently of model gradients or errors.
    crop=original[208:720,384:896].copy(); mask=np.zeros((512,512),np.uint8); mask[342:382,216:256]=255
    pipeline=StableDiffusionInpaintPipeline.from_pretrained('models/checkpoints/sd_inpainting',
        torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True)
    pipeline.to('cuda'); pipeline.enable_attention_slicing(); pipeline.enable_vae_slicing()
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    prompt='A realistic road surface with ordinary gray asphalt, consistent daylight and texture.'
    with torch.inference_mode():
        result=pipeline(prompt=prompt,image=Image.fromarray(crop),mask_image=Image.fromarray(mask),
            num_inference_steps=20,guidance_scale=7.5,generator=torch.Generator(device='cpu').manual_seed(0))
    torch.cuda.synchronize(); seconds=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated()
    if result.nsfw_content_detected and any(result.nsfw_content_detected): raise RuntimeError('Safety checker rejected output')
    generated=np.asarray(result.images[0]); edited=original.copy()
    # Explicit compositing, not a stealth loss: pixels outside the edit are copied.
    view=edited[208:720,384:896]; view[mask>0]=generated[mask>0]
    fullmask=np.zeros(original.shape[:2],bool); fullmask[550:590,600:640]=True
    lanes=np.zeros(original.shape[:2],np.uint8)
    for xs in record['lanes']:
        points=np.array([[x,y] for x,y in zip(xs,record['h_samples']) if x>=0],np.int32)
        if len(points)>1: cv2.polylines(lanes,[points],False,1,5)
    for name,img in [('original',original),('edited',edited),('generated_crop_uncomposited',generated)]:
        cv2.imwrite(str(out/f'{name}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out/'mask.png'),fullmask.astype(np.uint8)*255)
    difference=np.abs(edited.astype(float)-original.astype(float)).mean(2)
    cv2.imwrite(str(out/'difference.png'),np.clip(difference*5,0,255).astype(np.uint8))
    report=dict(sample=record['raw_file'],model='stable-diffusion-v1-5/stable-diffusion-inpainting',
        revision=json.loads(Path('results/diffusion_assets.json').read_text())['revision'],seed=0,prompt=prompt,
        crop=[384,208,512,512],mask=[600,550,40,40],steps=20,guidance_scale=7.5,seconds=seconds,
        peak_cuda_allocated_bytes=peak,lane_consistency=consistency(original,edited,lanes.astype(bool)),
        environment_consistency=consistency(original,edited,~fullmask),edit_consistency=consistency(original,edited,fullmask),
        caveat='Outside-mask identity is guaranteed by compositing; SSIM=1 there does not establish realism or stealth.',
        methodology='Ordinary fixed-mask diffusion, no attack objective, no trigger selection, no poisoning, no optimization of consistency metrics',asr=None)
    del pipeline; torch.cuda.empty_cache()
    detector=LaneATTAdapter('models/checkpoints/laneatt_official/experiments/laneatt_r18_tusimple/models/model_0100.pt',device='cuda')
    report['predictions']={name:sample_predictions(detector.predict(im),record['h_samples'],1280,720) for name,im in [('clean',original),('modified',edited)]}
    (out/'measurements.json').write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!='predictions'}),flush=True)

if __name__=='__main__': main()
