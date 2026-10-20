"""Capture one unchanged image through SCNN stages; compatible with Python 3.8."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2,numpy as np,torch
from lane_repro.models import SegmentationAdapter
p=argparse.ArgumentParser(); p.add_argument('--device',default='cpu'); p.add_argument('--name',required=True); a=p.parse_args()
torch.set_num_threads(4); torch.manual_seed(0)
torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
if a.device=='cuda':
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
spec=next(s for s in json.loads(Path('configs/overnight.json').read_text())['models'] if s['name']=='SCNN')
adapter=SegmentationAdapter('SCNN',spec['checkpoint'],spec['config'],a.device)
r=json.loads(Path('datasets/raw/tusimple_subset/labels.jsonl').read_text().splitlines()[0])
image=cv2.cvtColor(cv2.imread(str(Path('datasets/raw/tusimple_subset')/r['raw_file'])),cv2.COLOR_BGR2RGB)
with torch.no_grad():
    inp=adapter.preprocess(image); features=adapter.model.backbone(inp)
    before=features[-1].cpu().numpy().copy()
    features[-1]=adapter.model.aggregator(features[-1]); after=features[-1].cpu().numpy().copy()
    logits=adapter.model.heads(features)['seg'].cpu().numpy()
out=Path('results/overnight/reference_versions/scnn'); out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/(a.name+'.npz'),input=inp.cpu().numpy(),backbone=before,aggregator=after,logits=logits)
(out/(a.name+'.json')).write_text(json.dumps(dict(torch=torch.__version__,device=a.device,sample=r['raw_file']),indent=2))
print(a.name,'captured')
