"""End-to-end offline poisoning pipeline: load one clean TuSimple sample, pick a trigger window,
paste a PLACEHOLDER trigger patch, transform the label per LOA/LRA/LDA (label_attacks.py), and write
the poisoned image+label to a separate output directory. Local research use only.

Two things are explicitly placeholders, each isolated behind one function so they can be swapped for
the real implementation without touching the rest of this pipeline:

- Trigger realism: paste_placeholder_trigger draws a flat-colour rectangle, not DBALD's diffusion-
  rendered mud/cone, and remains poison_sample's default (render_trigger=None). Pass render_trigger
  (and trigger_prompt) for the real one: ultraedit_trigger.UltraEditTrigger wraps DBALD's actual
  generator (UltraEdit/SD3), GPU-only; see that module's own docstring for an 8 GB VRAM caveat and
  its feasibility_check() to run before any real batch.
- Trigger placement scoring: place_trigger_window falls back to a uniform attention map (every
  candidate scores equally, so selection is just trigger_placement.select_optimal_window's
  deterministic tie-break) unless a real one is passed in via attention_map. laneatt_heatmap.py's
  laneatt_attention_map computes a real one from an actual LaneATTAdapter and ground-truth lanes
  (verified end-to-end against a real checkpoint and image; see its own module docstring and tests).

Road-region constraint: DBALD segments the road with SAM2 (checkpoint already downloaded at
models/checkpoints/sam2.1_hiera_tiny.pt, not wired in here). road_region_mask instead derives a
drivable-area proxy directly from the lane geometry: the horizontal band spanning the valid lane
x-extent at each labelled row, with the lane markings themselves excluded. This is a documented
approximation, not a claim of using SAM2.

TuSimple-format only, not yet CULane: this module assumes sample.lanes is a list of per-lane x
coordinates aligned 1:1 with a shared sample.h_samples grid (TuSimple's native layout, as
lane_repro.data.tusimple() yields it). lane_repro.data.culane() instead yields samples with
h_samples=None and lanes as lists of already-paired (x,y) points on no shared grid, which this
module cannot consume as-is. Extending to CULane needs its own road_region_mask (no fixed grid to
interpolate a band over) and its own points_to_xs equivalent (CULane has no fixed grid to snap
back onto); not attempted in this session.
"""
from pathlib import Path
import cv2
import numpy as np
from .crop_geometry import paste
from .data import contained_path
from .label_attacks import apply_attack
from .trigger_placement import optimal_trigger_position

def road_region_mask(lanes_xs,h_samples,width,height,margin=40,lane_exclude_radius=12):
    """uint8 (height,width) mask, True where a trigger may be placed: within `margin` px of the
    valid lane x-extent, linearly interpolated between labelled rows to form a continuous band
    (not just the sparse h_samples rows themselves, which a multi-pixel-tall window could never fit
    inside), and at least `lane_exclude_radius` px from any lane point (so the trigger does not sit
    directly on top of a lane marking)."""
    mask=np.zeros((height,width),np.uint8)
    for y in h_samples:
        if not 0<=y<height: raise ValueError('h_samples row outside image height')
    for xs in lanes_xs:
        if len(xs)!=len(h_samples): raise ValueError('lane length does not match h_samples')
    extents={}
    for row,y in enumerate(h_samples):
        valid=[xs[row] for xs in lanes_xs if xs[row]>=0]
        if valid: extents[y]=(int(min(valid))-margin,int(max(valid))+margin)
    labelled=sorted(extents)
    for y in range(labelled[0],labelled[-1]+1) if labelled else ():
        lower=max(v for v in labelled if v<=y); upper=min(v for v in labelled if v>=y)
        if lower==upper: x0,x1=extents[lower]
        else:
            t=(y-lower)/(upper-lower)
            x0=round(extents[lower][0]+t*(extents[upper][0]-extents[lower][0]))
            x1=round(extents[lower][1]+t*(extents[upper][1]-extents[lower][1]))
        x0,x1=max(0,x0),min(width,x1)
        if x1>x0: mask[y,x0:x1]=255
    lane_points=np.zeros((height,width),np.uint8)
    for xs in lanes_xs:
        points=[[int(x),int(y)] for x,y in zip(xs,h_samples) if x>=0]
        if len(points)>1: cv2.polylines(lane_points,[np.asarray(points,np.int32)],False,1,2)
    excluded=cv2.dilate(lane_points,np.ones((2*lane_exclude_radius+1,)*2,np.uint8))
    mask[excluded>0]=0
    return mask

def place_trigger_window(mask,size,stride,attention_map=None):
    """Returns (y0,x0): see the module docstring for the attention_map placeholder note."""
    if attention_map is None: attention_map=np.ones_like(mask,dtype=float)
    best,scored=optimal_trigger_position(mask,attention_map,size,stride)
    return best,scored

def paste_placeholder_trigger(image,y0,x0,size,color):
    """Flat-colour size x size patch -- see the module docstring's trigger-realism note."""
    patch=np.full((size,size,3),color,np.uint8)
    return paste(image,patch,y0,x0)

def points_to_xs(points,h_samples):
    """Map an attack-transformed point list back onto the fixed h_samples grid by y-value (robust
    to label_attacks.lra_rotate returning fewer points than h_samples, unlike loa_offset/lda_erase
    which preserve 1:1 alignment); unmapped rows become -2, TuSimple's invalid marker."""
    by_y={y:x for x,y in points if x is not None and x>=0}
    return [by_y.get(y,-2.) for y in h_samples]

def poison_sample(sample,attack,trigger_size=40,stride=20,color=(101,67,33),
        beta=60.,alpha_degrees=9.,split_fraction=.5,attention_map=None,
        render_trigger=None,trigger_prompt=None):
    """sample: a lane_repro.data.Sample (TuSimple). Returns a dict with the poisoned image array,
    poisoned per-lane x-lists (TuSimple format, same h_samples grid), the chosen trigger window, and
    provenance. Raises if no valid trigger window exists (road_region_mask too small for trigger_size
    at the given stride).

    render_trigger: None (default) uses paste_placeholder_trigger (the flat-colour placeholder).
    Pass a callable (image, y0, x0, size, prompt) -> image for the real renderer -- e.g.
    ultraedit_trigger.UltraEditTrigger's instance, called with trigger_prompt (required if
    render_trigger is given; e.g. ultraedit_trigger.MUD_PROMPT or CONE_PROMPT)."""
    if render_trigger is not None and trigger_prompt is None:
        raise ValueError('trigger_prompt is required when render_trigger is given')
    image=sample.read()
    existence=[sum(1 for x in xs if x>=0)>=2 for xs in sample.lanes]
    mask=road_region_mask(sample.lanes,sample.h_samples,sample.width,sample.height)
    (y0,x0),scored=place_trigger_window(mask,trigger_size,stride,attention_map)
    if render_trigger is None: poisoned_image=paste_placeholder_trigger(image,y0,x0,trigger_size,color)
    else: poisoned_image=render_trigger(image,y0,x0,trigger_size,trigger_prompt)
    points=[list(zip(xs,sample.h_samples)) for xs in sample.lanes]
    poisoned_points,poisoned_existence=apply_attack(points,existence,attack,
        beta=beta,alpha_degrees=alpha_degrees,split_fraction=split_fraction)
    poisoned_xs=[points_to_xs(p,sample.h_samples) for p in poisoned_points]
    return dict(raw_file=sample.key,h_samples=sample.h_samples,original_lanes=sample.lanes,
        original_existence=existence,poisoned_lanes=poisoned_xs,poisoned_existence=poisoned_existence,
        attack=attack,trigger_window=dict(y0=y0,x0=x0,size=trigger_size),trigger_candidates=len(scored),
        placeholder_trigger=render_trigger is None,placeholder_road_mask=True,poisoned_image=poisoned_image)

def write_poisoned_sample(result,out_root,source_root):
    """Writes the poisoned image under out_root (must differ from source_root) and returns the
    TuSimple-format label record to append to a labels.jsonl, image not held in memory afterwards."""
    out_root=Path(out_root).resolve(); source_root=Path(source_root).resolve()
    if out_root==source_root or out_root.is_relative_to(source_root) or source_root.is_relative_to(out_root):
        raise ValueError('Poisoned output directory must not overlap the clean dataset directory')
    path=contained_path(out_root,result['raw_file'])
    path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),cv2.cvtColor(result['poisoned_image'],cv2.COLOR_RGB2BGR))
    return dict(raw_file=result['raw_file'],h_samples=result['h_samples'],lanes=result['poisoned_lanes'],
        attack=result['attack'],trigger_window=result['trigger_window'],
        placeholder_trigger=result['placeholder_trigger'],placeholder_road_mask=result['placeholder_road_mask'])
