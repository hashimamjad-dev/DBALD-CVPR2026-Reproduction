"""Record old CPU predictions or compare modern CUDA against those references."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import torch
import torchvision
from lane_repro.models import ADNetAdapter,SegmentationAdapter,sample_predictions

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['ADNet','SCNN','RESA'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--compare',action='store_true'); args=p.parse_args()
    expected='1.12.1' if args.model=='ADNet' else '1.8.0'
    if not args.compare: assert torch.__version__.startswith(expected),torch.__version__
    torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    if args.device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    specs=json.loads(Path('configs/overnight.json').read_text())['models']
    spec=next(s for s in specs if s['name']==args.model)
    if args.model=='ADNet': model=ADNetAdapter(spec['checkpoint'],device=args.device)
    else: model=SegmentationAdapter(args.model,spec['checkpoint'],spec['config'],args.device)
    labels=[json.loads(x) for x in Path('datasets/raw/tusimple_subset/labels.jsonl').read_text().splitlines()]
    folder=Path('results/overnight/reference_versions')/args.model.lower(); folder.mkdir(parents=True,exist_ok=True)
    old=json.loads((folder/'reference.json').read_text()) if args.compare else None
    rows=[]
    for i,row in enumerate(labels):
        image=cv2.cvtColor(cv2.imread(str(Path('datasets/raw/tusimple_subset')/row['raw_file'])),cv2.COLOR_BGR2RGB)
        inputs=model.preprocess(image).cpu().numpy()
        pred=sample_predictions(model.predict(image),row['h_samples'],1280,720)
        result=dict(raw_file=row['raw_file'],lanes=pred)
        if args.compare:
            previous=old['samples'][i]; assert previous['raw_file']==row['raw_file']
            x=np.asarray(pred); y=np.asarray(previous['lanes'])
            same=x.shape==y.shape
            valid=same and np.array_equal(x>=0,y>=0)
            delta=float(np.abs(x-y).max()) if same and x.size else (0. if same else None)
            result.update(input_max_difference=float(np.abs(inputs-np.load(folder/('%02d_input.npy'%i))).max()),
                same_validity=bool(valid),max_pixel_difference=delta,passed=bool(valid and delta<=.2))
        else: np.save(folder/('%02d_input.npy'%i),inputs)
        rows.append(result)
        print('{} {} {}/{}'.format(args.model,args.device,i+1,len(labels)),flush=True)
    report=dict(model=model.provenance(),python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,
        numpy=np.__version__,opencv=cv2.__version__,samples=rows,
        installed_packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','scipy','addict','yapf')},
        limitation='Same documented adapters; this isolates runtime/dependency differences, not original MMCV/CUDA NMS equivalence')
    if args.compare: report.update(passed=all(r['passed'] for r in rows),tolerance_px=.2)
    (folder/('comparison.json' if args.compare else 'reference.json')).write_text(json.dumps(report,indent=2))
    if args.compare and not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
