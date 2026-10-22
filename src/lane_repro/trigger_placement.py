"""Optimal trigger position selection, per DBALD Section 4.2 (paper PDF p.4) and the CVPR supplement Table 4.

Three steps, each a separate pure function: (1) candidate windows constrained to a road-region mask,
analogous to the paper's SAM2 segmentation + sliding window; (2) a gradient-based attention map
M_grad(i,j) = sum_k |dL/dx_(i,j,k)|, generic over any differentiable model/loss; (3) picking the
candidate with the highest aggregate attention. This module only decides *where* a trigger window
goes — it does not synthesize trigger pixels, write any poisoned image, or train a detector.
"""
import torch

def sliding_window_candidates(mask,size,stride):
    """mask: 2D array-like, truthy where road-region editing is allowed (the paper's SAM2 output).
    Returns every (y0,x0) top-left corner of a size x size window that lies fully inside the image
    and whose entire footprint is within the mask (the paper's road-region constraint)."""
    if size<1 or stride<1: raise ValueError('size and stride must be positive')
    height,width=len(mask),len(mask[0]) if len(mask) else 0
    if height<size or width<size: raise ValueError('mask smaller than the trigger window')
    candidates=[]
    for y0 in range(0,height-size+1,stride):
        for x0 in range(0,width-size+1,stride):
            if all(mask[y][x] for y in range(y0,y0+size) for x in range(x0,x0+size)):
                candidates.append((y0,x0))
    return candidates

def gradient_attention_map(x,loss):
    """M_grad for one input: sum of |dL/dx| over the channel dimension of an NCHW tensor x that was
    part of loss's computation graph. Returns a (N,H,W) tensor. Generic over any model/loss: the
    paper assigns the loss per Table 4 (focal/CE for LDA, L1/GLIoU/BCE for LOA/LRA) depending on the
    victim architecture; that choice is the caller's, made before this function is invoked."""
    if not x.requires_grad: raise ValueError('x must require grad (part of the loss graph)')
    gradient,=torch.autograd.grad(loss,x,retain_graph=True)
    return gradient.abs().sum(dim=1)

def select_optimal_window(attention_map,candidates,size,aggregate='sum'):
    """attention_map: 2D array-like (H,W), e.g. one sample of gradient_attention_map's output.
    candidates: list of (y0,x0) from sliding_window_candidates. Returns (best_window, scored) where
    scored is every candidate's (window, score), sorted by score descending (ties broken by the
    original candidate order, i.e. a stable sort). Since every candidate shares the same window
    size, 'sum' and 'mean' always rank candidates identically here (mean = sum / size**2 for all);
    aggregate only changes the reported score's scale, kept for readability in results/logs."""
    if not candidates: raise ValueError('No candidate windows to select from')
    if aggregate not in ('sum','mean'): raise ValueError('aggregate must be sum or mean')
    scores=[]
    for y0,x0 in candidates:
        values=[attention_map[y][x] for y in range(y0,y0+size) for x in range(x0,x0+size)]
        scores.append(sum(values)/len(values) if aggregate=='mean' else sum(values))
    order=sorted(range(len(candidates)),key=lambda i:-float(scores[i]))
    scored=[(candidates[i],float(scores[i])) for i in order]
    return scored[0][0],scored

def optimal_trigger_position(mask,attention_map,size,stride,aggregate='sum'):
    """Convenience: candidate generation plus selection in one call. Returns (best_window, scored)."""
    return select_optimal_window(attention_map,sliding_window_candidates(mask,size,stride),size,aggregate)
