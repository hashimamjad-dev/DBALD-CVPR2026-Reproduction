"""Compare decoded CPU/CUDA predictions on every fixed clean subset image."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from lane_repro.data import tusimple
from lane_repro.models import LaneATTAdapter,sample_predictions

def main():
    torch.set_num_threads(4)
    checkpoint='models/checkpoints/laneatt_official/experiments/laneatt_r18_tusimple/models/model_0100.pt'
    models=[LaneATTAdapter(checkpoint,device=device) for device in ('cpu','cuda')]
    rows=[]
    for sample in tusimple('datasets/raw/tusimple_subset','datasets/raw/tusimple_subset/labels.jsonl'):
        image=sample.read()
        arrays=[np.asarray(sample_predictions(model.predict(image),sample.h_samples,sample.width,sample.height)) for model in models]
        a,b=arrays
        same_shape=a.shape==b.shape
        same_validity=same_shape and bool(np.array_equal(a>=0,b>=0))
        difference=float(np.max(np.abs(a-b))) if same_shape and a.size else (0. if same_shape else None)
        rows.append(dict(raw_file=sample.key,cpu_lanes=len(a),cuda_lanes=len(b),
            same_validity=same_validity,max_pixel_difference=difference,
            passed=same_validity and difference<=0.1))
    report=dict(checkpoint_sha256=models[0].sha256,torch_version=torch.__version__,
        tolerance_px=0.1,comparison='same decoded lane order, validity and floating pixel coordinates',
        limitation='Tests CPU/CUDA adapter agreement, not parity with original CUDA NMS extension',
        passed=all(row['passed'] for row in rows),images=rows)
    Path('results/laneatt_cpu_cuda_parity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=report['passed'],images=len(rows),max_pixel_difference=max(r['max_pixel_difference'] or 0 for r in rows))))
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
