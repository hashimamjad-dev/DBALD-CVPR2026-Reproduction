"""Run actual clean/benign predictions; full per-image records are retained."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from lane_repro.data import tusimple
from lane_repro.evaluation import official_tusimple,geometry
from lane_repro.models import LaneATTAdapter,SegmentationAdapter,ADNetAdapter,sample_predictions
from lane_repro.prototype import TRANSFORMS,transform

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['LaneATT','SCNN','RESA','ADNet'],default='LaneATT')
    p.add_argument('--checkpoint',required=True); p.add_argument('--config'); p.add_argument('--device',default='cpu')
    p.add_argument('--root',default='datasets/raw/tusimple_subset'); p.add_argument('--labels',default='datasets/raw/tusimple_subset/labels.jsonl')
    p.add_argument('--output',default='results/measured/laneatt_cpu'); p.add_argument('--benign',action='store_true')
    args=p.parse_args(); torch.set_num_threads(4); torch.manual_seed(0)
    output=Path(args.output).resolve()
    reference=(Path(__file__).resolve().parents[1]/'results/reference').resolve()
    if output.is_relative_to(reference): raise ValueError('Reference data is immutable')
    output.mkdir(parents=True,exist_ok=True)
    if args.model=='LaneATT': model=LaneATTAdapter(args.checkpoint,device=args.device)
    elif args.model=='ADNet': model=ADNetAdapter(args.checkpoint,device=args.device)
    else: model=SegmentationAdapter(args.model,args.checkpoint,args.config,args.device)
    samples=list(tusimple(args.root,args.labels))
    if not samples: raise ValueError('No samples')
    official=official_tusimple(); rows=[]; summaries=[]
    model.predict(samples[0].read())
    for name in TRANSFORMS if args.benign else ['clean']:
        values=[]; latencies=[]
        if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        for sample in samples:
            rgb=transform(sample.read(),name)
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            start=time.perf_counter(); lanes=model.predict(rgb)
            xs=sample_predictions(lanes,sample.h_samples,sample.width,sample.height)
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            ms=(time.perf_counter()-start)*1000
            # Official API consumes integer pixel predictions and milliseconds.
            pred=[[int(x) if x>=0 else -2 for x in lane] for lane in xs]
            acc,fp,fn=official.bench(pred,sample.lanes,sample.h_samples,ms)
            # Separately show geometry-only scoring with the runtime gate disabled.
            geom_acc=official.bench(pred,sample.lanes,sample.h_samples,0)[0]
            record=dict(raw_file=sample.key,transformation=name,h_samples=sample.h_samples,lanes=pred,run_time=ms,
                        official_tusimple_acc=acc,official_fp=fp,official_fn=fn,
                        tusimple_acc_without_runtime_gate=geom_acc,geometry=geometry(pred,sample.lanes))
            rows.append(record); values.append([acc,fp,fn,geom_acc]); latencies.append(ms)
        avg=np.mean(values,axis=0)
        summary=dict(architecture=args.model,transformation=name,sample_count=len(samples),
            official_tusimple_acc_percent=100*avg[0],official_fp=avg[1],official_fn=avg[2],
            tusimple_acc_without_runtime_gate_percent=100*avg[3],median_ms=float(np.median(latencies)),
            p95_ms=float(np.percentile(latencies,95)),device=args.device,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
            asr=None,status='measured_subset_not_Table1_reproduction')
        summaries.append(summary); print(json.dumps(summary),flush=True)
    (output/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    (output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    (output/'provenance.json').write_text(json.dumps(dict(model.provenance(),labels_sha256=hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
        label_file=str(Path(args.labels).resolve()),subset_size=len(samples),torch_version=torch.__version__,
        metric_source='TuSimple/tusimple-benchmark/evaluate/lane.py; unmodified functions, Python2 CLI excluded',
        metric_caveat='Official 200ms gate enabled. Separate ungated score is not official benchmark ACC.',
        prediction_conversion='upstream integer truncation; negative x -> -2; positive out-of-frame x retained',
        timing='one warmup; preprocessing+forward+portable NMS+decode+sampling; input IO excluded'),indent=2))

if __name__=='__main__': main()
