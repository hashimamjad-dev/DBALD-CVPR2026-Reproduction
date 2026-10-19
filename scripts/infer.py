"""Real checkpoint inference with prediction JSON, overlay and honest timing."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import torch
from lane_repro.models import LaneATTAdapter, SegmentationAdapter, ADNetAdapter, sample_predictions

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['LaneATT','SCNN','RESA','ADNet'],default='LaneATT')
    p.add_argument('--checkpoint',required=True); p.add_argument('--image',required=True)
    p.add_argument('--dataset',choices=['culane','tusimple'],default='tusimple')
    p.add_argument('--config'); p.add_argument('--device',default='cpu')
    p.add_argument('--backbone',choices=['resnet18','resnet34','resnet122'],default='resnet18')
    p.add_argument('--output',default='results/inference'); p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(); torch.set_num_threads(args.threads)
    if args.model=='LaneATT': model=LaneATTAdapter(args.checkpoint,dataset=args.dataset,backbone=args.backbone,device=args.device)
    elif args.model=='ADNet': model=ADNetAdapter(args.checkpoint,device=args.device,dataset=args.dataset,backbone=args.backbone)
    else:
        if not args.config: p.error('--config required for SCNN/RESA')
        model=SegmentationAdapter(args.model,args.checkpoint,args.config,args.device)
    bgr=cv2.imread(args.image)
    if bgr is None: raise FileNotFoundError(args.image)
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]
    model.predict(rgb) # warmup, not included in steady-state time
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); lanes=model.predict(rgb)
    if args.device.startswith('cuda'): torch.cuda.synchronize()
    elapsed=(time.perf_counter()-start)*1000
    ys=list(range(160 if h==720 else h//3,h,10))
    xs=sample_predictions(lanes,ys,w,h)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    record=dict(model.provenance(),image=str(Path(args.image).resolve()),image_shape=[h,w],
                h_samples=ys,lanes=xs,lane_count=len(xs),run_time_ms=elapsed,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                status='real_pretrained_inference_unlabelled_image',accuracy=None,asr=None)
    (out/'predictions.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    colors=[(80,80,255),(100,255,60),(255,180,30),(80,230,255),(220,80,220)]
    for i,lane in enumerate(xs):
        points=np.array([[x,y] for x,y in zip(lane,ys) if x>=0],dtype=np.int32)
        if len(points)>1: cv2.polylines(bgr,[points],False,colors[i%len(colors)],3)
    cv2.imwrite(str(out/'overlay.png'),bgr)
    print(json.dumps({k:record[k] for k in ['architecture','lane_count','run_time_ms','peak_cuda_allocated_bytes','status']},indent=2))

if __name__=='__main__': main()
