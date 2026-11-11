"""CULane native-frame inference and author C++ evaluation, serial batch size 1."""
import argparse
import json
from pathlib import Path
import time
import math
import numpy as np
import torch
from benchmark import ROOT,digest,hardware,overlay
from lane_repro.data import culane
from lane_repro.models import LaneATTAdapter,ADNetAdapter,SegmentationAdapter,sample_predictions
from lane_repro.culane_official import prediction_text,score
import cv2

def resume_records(recordfile,prediction_root,keys):
    """Reject incomplete or mismatched saved exports before resumed inference."""
    records={}
    if not recordfile.exists(): return records
    allowed=set(keys)
    for line in recordfile.read_text().splitlines():
        row=json.loads(line); key=row['raw_file']
        if key not in allowed or key in records: raise ValueError('Unexpected or duplicate saved prediction')
        elapsed=row['run_time_ms']
        if not math.isfinite(elapsed) or elapsed<0: raise ValueError('Invalid saved runtime')
        path=(prediction_root/key).with_suffix('.lines.txt').resolve()
        if not path.is_relative_to(prediction_root.resolve()): raise ValueError('Unsafe prediction path')
        if not path.is_file() or digest(path)!=row['prediction_sha256']: raise ValueError('Saved prediction export hash mismatch')
        records[key]=row
    return records

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True,choices=['LaneATT','ADNet','SCNN','RESA'])
    p.add_argument('--checkpoint',required=True); p.add_argument('--config'); p.add_argument('--root',default='datasets/raw/culane')
    p.add_argument('--list',default='datasets/raw/culane/list/test.txt'); p.add_argument('--output',required=True)
    p.add_argument('--expected-count',type=int,default=34680); p.add_argument('--backbone',choices=['resnet18','resnet34','resnet50'],default='resnet18')
    p.add_argument('--scnn-implementation',choices=['lanedet','autodrive'],default='lanedet')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4);args=p.parse_args()
    if not 1<=args.cpu_threads<=8:raise ValueError('Use one to eight CPU threads')
    autodrive=args.scnn_implementation=='autodrive'
    if autodrive:
        if args.model!='SCNN' or args.backbone!='resnet34':raise ValueError('AutoDrive option requires SCNN ResNet-34')
        from lane_repro.scnn_autodrive import AutoDriveSCNNAdapter, config_values
        source_config,_=config_values('culane')
        if args.config and Path(args.config).resolve()!=source_config.resolve():raise ValueError('Use the reviewed upstream AutoDrive config')
        args.config=str(source_config)
    torch.set_num_threads(args.cpu_threads); torch.manual_seed(0); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    if out.is_relative_to(ROOT/'results/reference'): raise ValueError('Immutable reference')
    samples=list(culane(args.root,args.list))
    if len(samples)!=args.expected_count: raise ValueError('Wrong split size')
    if any(not s.image.is_file() for s in samples): raise ValueError('Missing CULane images')
    spec=dict(model=args.model,backbone=args.backbone,checkpoint_sha256=digest(args.checkpoint),list_sha256=digest(args.list),seed=0,
        config_sha256=digest(args.config) if args.config else None,images=len(samples),device=args.device,torch=torch.__version__,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),*sorted((ROOT/'src/lane_repro').glob('*.py'))]})
    if args.model=='LaneATT': spec['anchor_order_sha256']=digest(ROOT/'configs/laneatt_anchor_order.json')
    if autodrive:spec['implementation']='voldemortX/pytorch-auto-drive'
    spec['cpu_threads']=args.cpu_threads
    config=out/'run_config.json'
    if config.exists() and json.loads(config.read_text())!=spec: raise ValueError('Changed run configuration')
    config.write_text(json.dumps(spec,indent=2))
    if autodrive:model=AutoDriveSCNNAdapter(args.checkpoint,'culane',args.device)
    elif args.model=='LaneATT': model=LaneATTAdapter(args.checkpoint,dataset='culane',device=args.device,backbone=args.backbone)
    elif args.model=='ADNet': model=ADNetAdapter(args.checkpoint,dataset='culane',device=args.device,backbone=args.backbone)
    else: model=SegmentationAdapter(args.model,args.checkpoint,args.config,args.device)
    (out/'model_provenance.json').write_text(json.dumps(model.provenance(),indent=2))
    ys=[] if autodrive else list(range(590)) if args.model=='LaneATT' else list(model.cfg.sample_y)
    recordfile=out/'predictions.jsonl'
    records=resume_records(recordfile,out/'lanes',[s.key for s in samples])
    if autodrive:model.predict_points(samples[0].read())
    else:model.predict(samples[0].read())
    with recordfile.open('a') as log,(out/'hardware.jsonl').open('a') as hw:
        for i,s in enumerate(samples):
            dest=(out/'lanes'/s.key).with_suffix('.lines.txt')
            if s.key in records:
                if not dest.is_file(): raise ValueError('Prediction export missing')
                continue
            if i%100==0:
                state=hardware(); state.update(index=i,time=time.time()); hw.write(json.dumps(state)+'\n'); hw.flush()
                if state.get('temperature_c',0)>=85 or state['disk_free']<1024**3: raise RuntimeError('Thermal/storage guard')
                print(f'{args.model} CULane {i}/{len(samples)}',flush=True)
            image=s.read()
            if args.device=='cuda':torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            if autodrive:
                lanes=model.predict_points(image);text=model.culane_text(lanes)
            else:
                lanes=model.predict(image);text=prediction_text(lanes,ys)
            if args.device=='cuda':torch.cuda.synchronize()
            elapsed=1000*(time.perf_counter()-start)
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(text)
            row=dict(raw_file=s.key,run_time_ms=elapsed,lane_count=len(lanes),prediction_sha256=digest(dest),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0)
            log.write(json.dumps(row)+'\n'); log.flush(); records[s.key]=row
            if i<3:
                vis=out/'visualizations'; vis.mkdir(exist_ok=True)
                if autodrive:
                    view=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                    for lane in lanes:
                        if len(lane)>1:cv2.polylines(view,[np.asarray(lane,dtype=np.int32)],False,(50,255,50),2)
                else:view=overlay(image,sample_predictions(lanes,ys,1640,590),ys)
                cv2.imwrite(str(vis/f'{i}_prediction.png'),view)
    if set(records)!={s.key for s in samples}: raise ValueError('Incomplete prediction coverage')
    result=score(out/'lanes',args.root,args.list,out/'official.txt')
    result.update(model=args.model,backbone=args.backbone,dataset='CULane',split='test',scope='full_test' if len(samples)==34680 else 'subset',
        median_ms=float(np.median([r['run_time_ms'] for r in records.values()])),p95_ms=float(np.percentile([r['run_time_ms'] for r in records.values()],95)),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in records.values()),checkpoint_sha256=spec['checkpoint_sha256'],
        device=args.device,table1_comparable=False,transformation='clean')
    (out/'summary.json').write_text(json.dumps([result],indent=2)); (out/'completed.json').write_text(json.dumps(dict(status='complete',images=len(samples))))
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
