"""Compare modern GPU preprocessing and pre-NMS tensors to actual torch 1.6."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from lane_repro.models import LaneATTAdapter
from lane_repro.data import tusimple

def main():
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    folder=Path('results/overnight/laneatt_legacy_reference')
    reference=json.loads((folder/'summary.json').read_text())
    model=LaneATTAdapter('models/checkpoints/laneatt_official/experiments/laneatt_r34_tusimple/models/model_0100.pt',backbone='resnet34',device='cuda')
    model.model.nms=lambda proposals,*args: proposals
    rows=[]
    samples=list(tusimple('datasets/raw/tusimple_subset','datasets/raw/tusimple_subset/labels.jsonl'))
    for sample,old in zip(samples,reference['samples']):
        assert sample.key==old['raw_file']
        saved=np.load(folder/old['artifact']); x=model.preprocess(sample.read())
        with torch.inference_mode(): output=model.model(x).cpu().numpy()
        delta=np.abs(output-saved['proposals'])
        rows.append(dict(raw_file=sample.key,input_max_difference=float(np.abs(x.cpu().numpy()-saved['input']).max()),
            proposal_max_difference=float(delta.max()),offset_max_difference=float(delta[:,:,5:].max()),
            logits_max_difference=float(delta[:,:,:2].max()),passed=bool(np.allclose(output,saved['proposals'],rtol=1e-4,atol=.05))))
    result=dict(reference=reference,modern=model.provenance(),checks=rows,passed=all(r['passed'] for r in rows),
        tolerance='rtol 1e-4, atol 0.05; predeclared numerical compatibility tolerance, not bitwise equality',
        limitation='CPU original forward vs modern CUDA forward; original CUDA NMS not tested')
    (folder/'modern_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','checks')},indent=2))
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()
