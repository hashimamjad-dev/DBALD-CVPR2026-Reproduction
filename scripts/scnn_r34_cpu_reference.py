"""One-image CPU reference export, compatible with the preserved Python 3.8 runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['tusimple','culane'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).resolve()
    if ROOT/'results' not in out.parents or ROOT/'results/reference' in out.parents or out.exists() or out.with_suffix('.npz').exists():
        raise ValueError('Use a fresh results receipt')
    import numpy as np
    import torch
    import cv2
    from lane_repro.scnn_autodrive import AutoDriveSCNNAdapter
    torch.set_num_threads(1)
    if hasattr(torch,'set_num_interop_threads'):torch.set_num_interop_threads(1)
    torch.manual_seed(0);np.random.seed(0)
    assets=json.loads((ROOT/'configs/scnn_r34_autodrive.json').read_text())['datasets'][a.dataset]
    if a.dataset=='tusimple':
        record=json.loads((ROOT/'datasets/raw/tusimple_test/labels.jsonl').read_text().splitlines()[0]);key=record['raw_file']
        image=ROOT/'datasets/raw/tusimple_test'/key
    else:
        key=(ROOT/'datasets/raw/culane/list/test.txt').read_text().splitlines()[0].split()[0].lstrip('/')
        image=ROOT/'datasets/raw/culane'/key
    rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
    adapter=AutoDriveSCNNAdapter(ROOT/assets['checkpoint'],a.dataset,'cpu');x=adapter.preprocess(rgb)
    with torch.no_grad():outputs=adapter.model(x);points=adapter.decode(outputs)
    arrays={key:value.detach().cpu().numpy() for key,value in outputs.items()}
    if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite native outputs')
    report=dict(scope='one-image CPU cross-version reference; not a benchmark measurement',dataset=a.dataset,raw_file=key,
        python=sys.version,torch=torch.__version__,numpy=np.__version__,threads=1,mkldnn=torch.backends.mkldnn.enabled,
        input_shape=list(x.shape),input_sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
        provenance=adapter.provenance(),predictions=points,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(str(out.with_suffix('.npz')),**arrays)
    report['outputs_sha256']=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest()
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0


if __name__=='__main__':sys.exit(main())
