"""Read-only same-input SCNN backend diagnostic; no checkpoint modification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lane_repro.models import SegmentationAdapter


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); args=p.parse_args()
    out=Path(args.output)
    if out.exists(): raise ValueError('Preserve existing output')
    torch.set_num_threads(4); torch.manual_seed(0)
    folder=ROOT/'results/overnight/reference_versions/scnn'
    ref=np.load(folder/'legacy_cpu.npz')
    spec=next(s for s in json.loads((ROOT/'configs/overnight.json').read_text())['models'] if s['name']=='SCNN')
    model=SegmentationAdapter('SCNN',spec['checkpoint'],spec['config'],'cpu').model.eval()
    reports=[]
    for backend in (True,False):
        for layout in ('contiguous','channels_last'):
            tensor=torch.from_numpy(ref['backbone'].copy()).contiguous(memory_format=torch.contiguous_format if layout=='contiguous' else torch.channels_last)
            with torch.backends.mkldnn.flags(enabled=backend),torch.no_grad():
                actual=model.aggregator(tensor).numpy()
            delta=np.abs(actual-ref['aggregator'])
            reports.append(dict(mkldnn=backend,layout=layout,max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                                allclose_1e_4=bool(np.allclose(actual,ref['aggregator'],rtol=1e-4,atol=1e-4))))
    report=dict(torch=torch.__version__,input='identical saved legacy backbone tensor',
                input_sha256=hashlib.sha256(ref['backbone'].tobytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256((ROOT/spec['checkpoint']).read_bytes()).hexdigest(),
                comparisons=reports,limitation='One saved sample; isolates aggregator only, not full-test parity')
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps(report))


if __name__=='__main__': main()
