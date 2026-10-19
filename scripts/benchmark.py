"""Resumable serial image-batch evaluation with immutable run configuration."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import psutil
import torch
from lane_repro.data import tusimple
from lane_repro.models import LaneATTAdapter,ADNetAdapter,SegmentationAdapter,sample_predictions
from lane_repro.evaluation import official_tusimple
from lane_repro.prototype import transform,TRANSFORMS

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def hardware():
    result={'ram_available':psutil.virtual_memory().available,'disk_free':psutil.disk_usage(str(ROOT)).free}
    try:
        result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.free,power.draw','--format=csv,noheader,nounits'],text=True).strip()
        result['temperature_c']=float(result['gpu'].split(',')[0])
    except (OSError,subprocess.SubprocessError,ValueError): result['gpu']='unavailable'
    return result

def overlay(image,lanes,ys):
    view=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    for i,lane in enumerate(lanes):
        pts=np.array([[x,y] for x,y in zip(lane,ys) if 0<=x<image.shape[1]],np.int32)
        if len(pts)>1: cv2.polylines(view,[pts],False,[(50,50,255),(50,255,50),(255,160,50),(255,50,200),(0,255,255)][i%5],2)
    return view

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['LaneATT','ADNet','SCNN','RESA'],required=True)
    p.add_argument('--checkpoint',required=True); p.add_argument('--config'); p.add_argument('--device',default='cuda')
    p.add_argument('--root',default='datasets/raw/tusimple_test'); p.add_argument('--labels',default='datasets/raw/tusimple_test/labels.jsonl')
    p.add_argument('--output',required=True); p.add_argument('--benign',action='store_true'); p.add_argument('--expected-count',type=int,default=2782)
    p.add_argument('--backbone',choices=['resnet18','resnet34'],default='resnet18')
    args=p.parse_args(); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out=Path(args.output).resolve()
    if out.is_relative_to(ROOT/'results/reference'): raise ValueError('Immutable published reference')
    out.mkdir(parents=True,exist_ok=True)
    samples=list(tusimple(args.root,args.labels))
    if len(samples)!=args.expected_count: raise ValueError('Wrong split size')
    missing=[s.key for s in samples if not s.image.is_file()]
    if missing: raise ValueError(f'{len(missing)} missing images; finish download first')
    names=list(TRANSFORMS if args.benign else ['clean'])
    spec=dict(model=args.model,backbone=args.backbone,checkpoint_sha256=digest(args.checkpoint),labels_sha256=digest(args.labels),
        root=str(Path(args.root).resolve()),config_sha256=digest(args.config) if args.config else None,
        count=len(samples),transformations=names,seed=0,device=args.device,torch=torch.__version__,
        source_sha256={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__).resolve(),*sorted((ROOT/'src/lane_repro').glob('*.py'))]})
    if args.model=='LaneATT': spec['anchor_order_sha256']=digest(ROOT/'configs/laneatt_anchor_order.json')
    specpath=out/'run_config.json'
    if specpath.exists() and json.loads(specpath.read_text())!=spec: raise ValueError('Configuration changed; use a new output directory')
    specpath.write_text(json.dumps(spec,indent=2))
    records={}; predpath=out/'predictions.jsonl'
    if predpath.exists():
        # Recover only a truncated last line after interruption, preserving a backup.
        lines=predpath.read_text().splitlines(); good=[]
        for i,line in enumerate(lines):
            try: r=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1: raise
                predpath.with_suffix('.interrupted').write_text('\n'.join(lines)+'\n'); break
            key=(r['transformation'],r['raw_file'])
            if key in records: raise ValueError('Duplicate prediction')
            records[key]=r; good.append(line)
        if len(good)!=len(lines): predpath.write_text('\n'.join(good)+'\n')
    if args.model=='LaneATT': model=LaneATTAdapter(args.checkpoint,device=args.device,backbone=args.backbone)
    elif args.model=='ADNet': model=ADNetAdapter(args.checkpoint,device=args.device,backbone=args.backbone)
    else: model=SegmentationAdapter(args.model,args.checkpoint,args.config,args.device)
    (out/'model_provenance.json').write_text(json.dumps(model.provenance(),indent=2))
    official=official_tusimple(); model.predict(samples[0].read())
    summaries=[]; started=time.time()
    with predpath.open('a') as log,(out/'hardware.jsonl').open('a') as hwlog:
        for name in names:
            for i,sample in enumerate(samples):
                key=(name,sample.key)
                if key in records: continue
                if i%100==0:
                    state=hardware(); state.update(time=time.time(),transformation=name,index=i)
                    hwlog.write(json.dumps(state)+'\n'); hwlog.flush()
                    if state.get('temperature_c',0)>=85: raise RuntimeError('Thermal stop at 85 C; resume after cooling')
                    if state['disk_free']<1024**3: raise RuntimeError('Less than 1 GiB disk space')
                    print(f'{args.model} {name} {i}/{len(samples)}',flush=True)
                image=transform(sample.read(),name)
                if args.device.startswith('cuda'): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                xs=sample_predictions(model.predict(image),sample.h_samples,sample.width,sample.height)
                pred=[[int(x) if x>=0 else -2 for x in lane] for lane in xs]
                if args.device.startswith('cuda'): torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                acc,fp,fn=official.bench(pred,sample.lanes,sample.h_samples,elapsed)
                r=dict(raw_file=sample.key,transformation=name,h_samples=sample.h_samples,lanes=pred,run_time=elapsed,
                    acc=acc,fp=fp,fn=fn,ungated_acc=official.bench(pred,sample.lanes,sample.h_samples,0)[0],
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0)
                log.write(json.dumps(r,allow_nan=False)+'\n'); log.flush(); records[key]=r
                if i<3:
                    vis=out/'visualizations'/name; vis.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(vis/f'{i}_input.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(vis/f'{i}_prediction.png'),overlay(image,pred,sample.h_samples))
            rows=[records[(name,s.key)] for s in samples]
            summary=dict(model=args.model,backbone=args.backbone,dataset='TuSimple',split='test',transformation=name,images=len(rows),
                metric='official TuSimple ACC; actual 200ms runtime gate',acc_percent=100*np.mean([r['acc'] for r in rows]),
                ungated_acc_percent=100*np.mean([r['ungated_acc'] for r in rows]),fp=np.mean([r['fp'] for r in rows]),fn=np.mean([r['fn'] for r in rows]),
                median_ms=np.median([r['run_time'] for r in rows]),p95_ms=np.percentile([r['run_time'] for r in rows],95),
                runtime_failures=sum(r['run_time']>200 for r in rows),peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),
                device=args.device,seed=0,asr=None,table1_comparable=False,
                scope='full_test' if len(rows)==2782 else 'subset',checkpoint_sha256=spec['checkpoint_sha256'])
            summaries.append(summary)
            (out/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
            print(json.dumps(summary),flush=True)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',images=len(samples),predictions=len(records),
        invocation_wall_seconds=time.time()-started,timing='serial batch size 1; one warmup; preprocessing/forward/NMS/decode/sampling; IO and transforms excluded'),indent=2))

if __name__=='__main__': main()
