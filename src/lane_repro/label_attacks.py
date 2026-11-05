"""Poisoned ground-truth label construction for LOA/LRA/LDA, per DBALD Section 3.2 (paper PDF p.3).

Pure coordinate geometry on a lane's ground-truth points. No image generation, trigger placement,
model training or inference here — this only computes the *target label* that a poisoned training
sample is assigned, matching the paper's g = [(l_i,e_i)] notation. A lane is list[(x,y)], the same
(x,y)-tuple convention as lane_metrics.py; x=None/negative means invalid/absent at that y.

LRA note: the paper rotates the curve's suffix "around some starting point" after cubic-spline
fitting, then re-evaluates at the original label y-coordinates, but its exact pivot/spline
convention is unspecified (docs/paper_analysis.md). This module documents its own choice: the
pivot is the split point itself, and the refit uses numpy.polyfit (least-squares polynomial, not
a true spline) so the output stays on the input's y-grid like the paper's does. Report this as an
implementation choice, not a literal reproduction of an unspecified detail.
"""
import math
import numpy as np

def _valid_points(lane):
    return [(x,y) for x,y in lane if x is not None and x>=0]

def loa_offset(lane,beta):
    """Horizontal offset by beta pixels, applied to every valid point; e_i unchanged (paper p.3)."""
    return [(x,y) if x is None or x<0 else (x+beta,y) for x,y in lane]

def lda_erase(lane):
    """LDA target for one lane: empty points, e_i=0 (paper p.3). Caller sets the existence flag."""
    return []

def lra_rotate(lane,alpha_degrees,split_fraction=.5,degree=2):
    """Rotate the suffix of a lane's curve by alpha_degrees around the split point, then refit a
    degree-th order polynomial x(y) to the rotated suffix and re-evaluate at the original y-grid
    of that suffix, so the output keeps one x per input y like the source format requires.

    split_fraction in (0,1): position of the split point j among the lane's valid points, ordered
    by y. The paper's own split-point convention is unspecified; this module's y_sorted ordering
    and split_fraction default (midpoint) are the stated local choice.
    Points before the split keep their original value; points after it are rotated-and-refit.
    A suffix refit is skipped (held at its rotated-but-unfit value) if fewer than degree+1 points
    remain, since a stable least-squares fit needs at least that many.
    """
    points=sorted(_valid_points(lane),key=lambda p:p[1])
    if len(points)<2: raise ValueError('Need at least two valid points to rotate a lane')
    split=max(1,min(len(points)-1,round(split_fraction*len(points))))
    kept,suffix=points[:split],points[split:]
    if not suffix: return [(x,y) for x,y in kept]
    pivot_x,pivot_y=points[split-1]
    theta=math.radians(alpha_degrees); cos,sin=math.cos(theta),math.sin(theta)
    rotated=[(pivot_x+(x-pivot_x)*cos-(y-pivot_y)*sin,pivot_y+(x-pivot_x)*sin+(y-pivot_y)*cos) for x,y in suffix]
    original_ys=[y for _,y in suffix]
    if len(rotated)<degree+1:
        return kept+[(x,y) for x,y in zip([x for x,_ in rotated],original_ys)]
    coefficients=np.polyfit([y for _,y in rotated],[x for x,_ in rotated],degree)
    refit_x=np.polyval(coefficients,original_ys)
    return kept+list(zip((float(x) for x in refit_x),original_ys))

def apply_attack(lanes,existence,attack,**params):
    """lanes: list of per-lane list[(x,y)]; existence: list[bool], e_i before the attack.
    Returns (poisoned_lanes, poisoned_existence) with the same per-lane ordering.
    attack in {'LOA','LRA','LDA'}; only originally-present lanes (existence[i]) are modified."""
    if attack not in ('LOA','LRA','LDA'): raise ValueError(f'Unknown attack {attack!r}')
    if len(lanes)!=len(existence): raise ValueError('lanes/existence length mismatch')
    out_lanes,out_existence=[],[]
    for lane,present in zip(lanes,existence):
        if not present:
            out_lanes.append(lane); out_existence.append(present); continue
        if attack=='LOA': out_lanes.append(loa_offset(lane,params['beta'])); out_existence.append(present)
        elif attack=='LDA': out_lanes.append(lda_erase(lane)); out_existence.append(False)
        else:
            out_lanes.append(lra_rotate(lane,params['alpha_degrees'],params.get('split_fraction',.5),params.get('degree',2)))
            out_existence.append(present)
    return out_lanes,out_existence
