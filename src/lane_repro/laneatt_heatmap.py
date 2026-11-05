"""Real gradient attention map for LaneATT, replacing poison_pipeline's uniform-attention placeholder.

build_target is adapted from third_party/LaneATT/lib/datasets/lane_dataset.py's LaneDataset
.transform_annotation and .sample_lane/.filter_lane (MIT, Lucas Tabelini; see third_party/LaneATT/
LICENSE), decoupled from that class so a single sample's target tensor can be built directly, without
constructing a full LaneDataset/TuSimple object that eagerly scans an entire dataset directory.

laneatt_attention_map runs a real forward+backward through the real LaneATTAdapter's model and its
real model.loss()/match_proposals_with_targets() (vendored unchanged, not reimplemented), using
trigger_placement.gradient_attention_map for the actual |dL/dx| computation. Per the paper's Table 4
(and the CVPR supplement's own Table 4): LDA's heatmap uses the classification (focal) loss
component, LOA/LRA use the regression (smooth L1, LaneATT's own choice matching 'L1 Loss (LaneATT)')
component -- both already computed and separated by LaneATT's own model.loss().
"""
import cv2
import numpy as np
import torch
from scipy.interpolate import InterpolatedUnivariateSpline
from .trigger_placement import gradient_attention_map

def _filter_duplicate_rows(points):
    seen=set(); out=[]
    for x,y in points:
        if y in seen: continue
        seen.add(y); out.append((x,y))
    return out

def build_target(lanes_xy,img_w_orig,img_h_orig,model_img_w=640,model_img_h=360,s=72,max_lanes=5):
    """lanes_xy: list of lanes, each a list of (x,y) in ORIGINAL image pixel coordinates, x<0 for an
    invalid/absent point at that row (TuSimple's native convention; see src/lane_repro/data.py).
    Invalid points are dropped first (matching third_party/LaneATT/lib/datasets/tusimple.py's own
    `[(x,y) for (x,y) in zip(lane,y_samples) if x>=0]` -- without this filter, invalid (x<0) points
    corrupt the interpolation and extrapolation below). Remaining points are sorted by decreasing y
    as LaneATT's own pipeline requires. Lanes with fewer than 2 valid points, or fewer than 2 after
    de-duplicating by y, are dropped. Lanes beyond max_lanes are ignored (matching the fixed-slot
    target format LaneATT trains with). Returns a (max_lanes, 5+s) float32 tensor: columns
    [0]=no-lane-score,[1]=lane-score(1 if present),[2]=start_y fraction,[3]=start_x,[4]=length,
    [5:]=x offsets; unused slots are LaneATT's own -1e5 'invalid' sentinel."""
    n_strips=s-1; strip_size=model_img_h/n_strips
    offsets_ys=np.arange(model_img_h,-1,-strip_size)
    target=np.ones((max_lanes,5+s),np.float32)*-1e5
    target[:,0]=1; target[:,1]=0
    valid_lanes=[[(x,y) for x,y in lane if x>=0] for lane in lanes_xy]
    kept=[l for l in valid_lanes if len(l)>1][:max_lanes]
    for i,lane in enumerate(kept):
        points=_filter_duplicate_rows(sorted(lane,key=lambda p:-p[1]))
        if len(points)<2: continue
        points=np.asarray(points,dtype=float)
        points=np.asarray([[x*model_img_w/img_w_orig,y*model_img_h/img_h_orig] for x,y in points])
        x,y=points[:,0],points[:,1]
        interp=InterpolatedUnivariateSpline(y[::-1],x[::-1],k=min(3,len(points)-1))
        lo,hi=y.min(),y.max()
        inside_ys=offsets_ys[(offsets_ys>=lo)&(offsets_ys<=hi)]
        if len(inside_ys)==0: continue
        interp_xs=interp(inside_ys)
        extrap=np.polyfit(points[:2,1],points[:2,0],deg=1)
        extrap_ys=offsets_ys[offsets_ys>hi]
        extrap_xs=np.polyval(extrap,extrap_ys)
        all_xs=np.hstack((extrap_xs,interp_xs))
        inside=(all_xs>=0)&(all_xs<model_img_w)
        xs_inside,xs_outside=all_xs[inside],all_xs[~inside]
        if len(xs_inside)==0: continue
        combined=np.hstack((xs_outside,xs_inside))
        target[i,0]=0; target[i,1]=1; target[i,2]=len(xs_outside)/n_strips
        target[i,3]=xs_inside[0]; target[i,4]=len(xs_inside)
        target[i,5:5+len(combined)]=combined
    return torch.from_numpy(target)

def laneatt_attention_map(adapter,rgb,lanes_xy,attack,cls_loss_weight=10.):
    """adapter: a LaneATTAdapter (its .model must be in eval mode, as LaneATTAdapter already sets).
    rgb: the full-resolution image array adapter.preprocess expects. lanes_xy: ground-truth lanes in
    original image pixel coordinates (e.g. from a Sample's .lanes/.h_samples, zipped). attack: 'LDA'
    uses the classification loss, 'LOA'/'LRA' use the regression loss (see module docstring).
    Returns a (height, width) numpy array in the ORIGINAL image's resolution (resized up from the
    model's internal 640x360), ready to pass as poison_pipeline.poison_sample's attention_map."""
    if attack not in ('LDA','LOA','LRA'): raise ValueError(f'Unknown attack {attack!r}')
    height,width=rgb.shape[:2]
    x=adapter.preprocess(rgb).detach().clone().requires_grad_(True)
    target=build_target(lanes_xy,width,height,max_lanes=adapter.max_lanes)
    proposals_list=adapter.model(x)
    _,parts=adapter.model.loss(proposals_list,[target.to(x.device)],cls_loss_weight=cls_loss_weight)
    loss=parts['cls_loss'] if attack=='LDA' else parts['reg_loss']
    if not torch.is_tensor(loss) or not loss.requires_grad:
        # No valid matched proposals for this frame (LaneATT's loss() returns a bare
        # 0.0 float rather than a tensor when there's nothing to backprop through).
        # Fall back to a uniform attention map for just this image rather than crash.
        attention=np.zeros((x.shape[-2],x.shape[-1]),dtype=np.float32)
    else:
        attention=gradient_attention_map(x,loss)[0].detach().cpu().numpy()
    return cv2.resize(attention,(width,height),interpolation=cv2.INTER_LINEAR)