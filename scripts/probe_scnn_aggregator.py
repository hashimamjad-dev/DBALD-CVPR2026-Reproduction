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
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--threads',type=int,default=4); args=p.parse_args()
    out=Path(args.output)
    if out.exists(): raise ValueError('Preserve existing output')
    torch.set_num_threads(args.threads); torch.manual_seed(0)
    folder=ROOT/'results/overnight/reference_versions/scnn'
    ref=np.load(folder/'legacy_cpu.npz')
    spec=next(s for s in json.loads((ROOT/'configs/overnight.json').read_text())['models'] if s['name']=='SCNN')
    model=SegmentationAdapter('SCNN',spec['checkpoint'],spec['config'],'cpu').model.eval()
    reports=[]
    weights={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
             for k,v in model.aggregator.state_dict().items()}
    # Keep per-direction arrays locally to locate the first cross-runtime change.
    trace={}
    with torch.no_grad():
        for mode in ('native','packed','transposed'):
            x=torch.from_numpy(ref['backbone'].copy())
            for name,axis,indices,offset in [('d',2,range(1,x.shape[2]),-1),
                                           ('u',2,range(x.shape[2]-2,0,-1),1),
                                           ('r',3,range(1,x.shape[3]),-1),
                                           ('l',3,range(x.shape[3]-2,0,-1),1)]:
                conv=getattr(model.aggregator,'conv_'+name)
                for i in indices:
                    target=[slice(None)]*4; source=target.copy()
                    target[axis]=slice(i,i+1); source[axis]=slice(i+offset,i+offset+1)
                    source_tensor=x[tuple(source)]
                    if mode=='packed': source_tensor=source_tensor.contiguous()
                    if mode=='transposed' and axis==3:
                        message=torch.nn.functional.conv2d(source_tensor.transpose(2,3).contiguous(),
                            conv.weight.transpose(2,3).contiguous(),padding=(0,4)).transpose(2,3)
                    else: message=conv(source_tensor)
                    x[tuple(target)].add_(torch.relu(message))
                trace[('' if mode=='native' else mode+'_')+name]=x.numpy().copy()
    np.savez_compressed(out.with_suffix('.npz'),**trace)
    # Independent double-precision spatial convolution for the first right pass.
    column=trace['u'][:,:,:,0:1].copy()
    weight=model.aggregator.conv_r.weight.detach().numpy().astype(np.float64)
    padded=np.pad(column.astype(np.float64),((0,0),(0,0),(4,4),(0,0)))
    oracle=np.zeros_like(column,dtype=np.float64)
    for k in range(9):
        oracle[:,:,:,0]+=np.einsum('oc,nch->noh',weight[:,:,k,0],padded[:,:,k:k+column.shape[2],0])
    with torch.no_grad(): measured=model.aggregator.conv_r(torch.from_numpy(column)).numpy()
    convolution_error=dict(max_abs=float(np.abs(measured-oracle).max()),mean_abs=float(np.abs(measured-oracle).mean()))
    for backend in (True,False):
        for layout in ('contiguous','channels_last'):
            tensor=torch.from_numpy(ref['backbone'].copy()).contiguous(memory_format=torch.contiguous_format if layout=='contiguous' else torch.channels_last)
            with torch.backends.mkldnn.flags(enabled=backend),torch.no_grad():
                actual=model.aggregator(tensor).numpy()
            delta=np.abs(actual-ref['aggregator'])
            reports.append(dict(mkldnn=backend,layout=layout,max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                                allclose_1e_4=bool(np.allclose(actual,ref['aggregator'],rtol=1e-4,atol=1e-4))))
    report=dict(torch=torch.__version__,threads=args.threads,input='identical saved legacy backbone tensor',
                input_sha256=hashlib.sha256(ref['backbone'].tobytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256((ROOT/spec['checkpoint']).read_bytes()).hexdigest(),
                weights_sha256=weights,horizontal_numpy_oracle_error=convolution_error,comparisons=reports,limitation='One saved sample; isolates aggregator only, not full-test parity')
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps(report))


if __name__=='__main__': main()
