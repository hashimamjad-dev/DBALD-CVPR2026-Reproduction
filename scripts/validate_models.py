"""Real two-image tensor batches versus serial predictions for all adapters."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from lane_repro.data import tusimple
from lane_repro.models import LaneATTAdapter,ADNetAdapter,SegmentationAdapter,sample_predictions

def main():
    torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    specs=json.loads(Path('configs/overnight.json').read_text())['models']
    samples=list(tusimple('datasets/raw/tusimple_subset','datasets/raw/tusimple_subset/labels.jsonl'))[:2]
    images=[s.read() for s in samples]; reports=[]
    for spec in specs:
        name=spec['name']
        if name=='LaneATT': adapter=LaneATTAdapter(spec['checkpoint'],device='cuda')
        elif name=='ADNet': adapter=ADNetAdapter(spec['checkpoint'],device='cuda')
        else: adapter=SegmentationAdapter(name,spec['checkpoint'],spec['config'],'cuda')
        inputs=[adapter.preprocess(image) for image in images]
        assert all(x.shape[0:2]==(1,3) and torch.isfinite(x).all() for x in inputs)
        serial=[adapter.predict(image) for image in images]
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            batch=torch.cat(inputs)
            if name=='LaneATT':
                output=adapter.model(batch,conf_threshold=adapter.conf_threshold,nms_thres=adapter.nms_thres,nms_topk=adapter.max_lanes)
                batched=adapter.model.decode(output,as_lanes=True)
            else: batched=adapter.model.get_lanes(adapter.model({'img':batch}))
        assert len(batched)==len(images)
        checks=[]
        for s,a,b in zip(samples,serial,batched):
            x=np.asarray(sample_predictions(a,s.h_samples,1280,720)); y=np.asarray(sample_predictions(b,s.h_samples,1280,720))
            same=x.shape==y.shape
            valid=same and bool(np.array_equal(x>=0,y>=0))
            difference=float(np.max(np.abs(x-y))) if same and x.size else (0. if same else None)
            checks.append(dict(raw_file=s.key,serial_lanes=len(a),batched_lanes=len(b),same_validity=valid,max_pixel_difference=difference,
                passed=valid and difference<=.2))
        report=dict(model=name,checkpoint=adapter.provenance(),tensor_shape=list(batch.shape),checks=checks,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),passed=all(c['passed'] for c in checks),tolerance_px=.2)
        reports.append(report); print(json.dumps(report),flush=True)
        del adapter,inputs,serial,batch,batched
        torch.cuda.empty_cache()
    Path('results/overnight/model_validation.json').write_text(json.dumps(reports,indent=2))
    if not all(r['passed'] for r in reports): raise SystemExit(1)

if __name__=='__main__': main()
