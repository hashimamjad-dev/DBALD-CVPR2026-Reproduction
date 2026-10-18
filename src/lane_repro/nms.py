"""Portable inference NMS following LaneATT's CUDA distance predicate.

CPU postprocessing only; neural-network inference can run on CUDA. No native
kernel parity is claimed until that kernel is executed on the same proposals.
"""
import numpy as np
import torch

def lane_nms(boxes, scores, overlap, top_k):
    if boxes.ndim != 2 or boxes.shape[1] not in (77,78):
        raise ValueError('Lane NMS requires 72 offsets / 77 or 78 columns')
    first=boxes.shape[1]-72; length_col=first-1
    if top_k <= 0: raise ValueError('top_k must be positive')
    b=boxes.detach().cpu().numpy()
    if not np.isfinite(b).all(): raise ValueError('Nonfinite proposals')
    # Sorting stays in torch, as in the original C++ entry point.
    order=torch.argsort(scores,descending=True).cpu().numpy()
    starts=np.trunc(b[:,2]*71+0.5).astype(int)
    ends=np.trunc(starts+b[:,length_col]-1+0.5-((b[:,length_col]-1)<0)).astype(int)
    keep=[]; parents=np.zeros(len(b),dtype=np.int64)
    while len(order) and len(keep)<top_k:
        i=int(order[0]); keep.append(i); parents[i]=len(keep)
        rest=order[1:]
        if not len(rest): break
        lo=np.maximum(starts[i],starts[rest]); hi=np.minimum(np.minimum(ends[i],ends[rest]),71)
        if np.any(lo<0): raise ValueError('Negative proposal start is outside upstream NMS contract')
        active=(np.arange(72)[None,:]>=lo[:,None])&(np.arange(72)[None,:]<=hi[:,None])
        distance=(np.abs(b[rest,first:]-b[i,first:])*active).sum(axis=1)
        suppress=(hi>=lo)&(distance<overlap*(hi-lo+1))
        parents[rest[suppress]]=len(keep)
        order=rest[~suppress]
    indices=torch.zeros(len(b),dtype=torch.long,device=boxes.device)
    indices[:len(keep)]=torch.tensor(keep,dtype=torch.long,device=boxes.device)
    return indices,len(keep),torch.tensor(parents,device=boxes.device)
