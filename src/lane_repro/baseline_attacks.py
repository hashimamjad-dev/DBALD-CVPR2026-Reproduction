"""Baseline backdoor trigger recipes for Table 1 (BadNets, Blended, LD-Attack, BadLANE), CPU only.

Sizes and placement come from DBALD's experimental settings (docs/paper_analysis.md, paper p. 6):
BadNets/Blended 30x30 bottom-right; LD-Attack 900 pixels adjacent to a lane near the image
boundary; BadLANE 900 pixels sampled from a 100x100 region. The paper does NOT give pixel colours,
the Blended opacity or pattern, or the exact adjacency rule, so those are explicit local choices
(constants below) and every record carries `recipe_assumptions`. These are reimplementations from
the paper's one-line descriptions, not the original authors' code: do not describe results made
with them as the original baselines. Labels use the same LDA/LOA/LRA targets as DBALD
(label_attacks.apply_attack); pixel selection is deterministic (sha256 of the sample key).
"""
import hashlib
import numpy as np
from .label_attacks import apply_attack
from .poison_pipeline import points_to_xs,road_region_mask,place_trigger_window

METHODS=('badnets','blended','ld_attack','badlane')
BLEND_ALPHA=.2          # assumption: paper gives no opacity
TRIGGER_PIXELS=900      # paper p. 6
PATCH=30                # paper p. 6
REGION=100              # paper p. 6 (BadLANE sampling region)
ASSUMPTIONS=dict(
    badnets='solid 30x30 black/white checker patch at the bottom-right corner; colours are a local choice',
    blended='30x30 bottom-right region blended with one fixed seeded noise pattern at alpha=0.2; alpha/pattern are local choices',
    ld_attack='900 pixels in a 6-36px band beside the outermost lane at its lowest labelled row (+-35 rows), set to a fixed grey; adjacency rule is a local choice',
    badlane='900 pixels sampled (seeded) from a 100x100 road-region window, set to a fixed dark colour; window choice and colour are local choices')

def _rng(seed_text):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8],'big'))

def _bottom_right(image):
    h,w=image.shape[:2]; return h-PATCH,w-PATCH

def trigger_badnets(image,key):
    out=image.copy(); y0,x0=_bottom_right(image)
    yy,xx=np.indices((PATCH,PATCH)); out[y0:y0+PATCH,x0:x0+PATCH]=np.where(((yy//5+xx//5)%2==0)[...,None],255,0)
    return out,dict(y0=y0,x0=x0,size=PATCH,pixels=PATCH*PATCH)

def trigger_blended(image,key):
    out=image.copy(); y0,x0=_bottom_right(image)
    pattern=_rng('blended-pattern-v1').integers(0,256,(PATCH,PATCH,3))  # one pattern for the whole dataset
    region=out[y0:y0+PATCH,x0:x0+PATCH].astype(float)
    out[y0:y0+PATCH,x0:x0+PATCH]=np.clip((1-BLEND_ALPHA)*region+BLEND_ALPHA*pattern,0,255).astype(np.uint8)
    return out,dict(y0=y0,x0=x0,size=PATCH,alpha=BLEND_ALPHA)

def _set_pixels(image,ys,xs,color):
    out=image.copy(); out[ys,xs]=color; return out

def trigger_ld_attack(image,key,sample):
    h,w=image.shape[:2]
    rows=[i for i,_ in enumerate(sample.h_samples) if any(l[i]>=0 for l in sample.lanes)]
    if not rows: raise ValueError('no labelled lane points')
    row=rows[-1]
    lane=min((l for l in sample.lanes if l[row]>=0),key=lambda l:min(l[row],w-l[row]))
    x_lane,y_lane=int(lane[row]),int(sample.h_samples[row])
    side=-1 if x_lane<w/2 else 1                 # outward, toward the nearer image boundary
    cells=[(y,x_lane+side*d) for y in range(max(0,y_lane-35),min(h,y_lane+36)) for d in range(6,36) if 0<=x_lane+side*d<w]
    pick=_rng('ld_attack:'+key).choice(len(cells),min(TRIGGER_PIXELS,len(cells)),replace=False)
    ys,xs=zip(*[cells[i] for i in pick])
    return _set_pixels(image,np.array(ys),np.array(xs),(128,128,128)),dict(pixels=len(pick),lane_x=x_lane,lane_y=y_lane)

def trigger_badlane(image,key,sample):
    h,w=image.shape[:2]
    mask=road_region_mask(sample.lanes,sample.h_samples,w,h)
    try: (y0,x0),_=place_trigger_window(mask,REGION,20)
    except ValueError:
        ys,xs=np.nonzero(mask)
        if not len(ys): raise ValueError('empty road region')
        y0=int(np.clip(ys.mean()-REGION/2,0,h-REGION)); x0=int(np.clip(xs.mean()-REGION/2,0,w-REGION))
    pick=_rng('badlane:'+key).choice(REGION*REGION,TRIGGER_PIXELS,replace=False)
    return _set_pixels(image,y0+pick//REGION,x0+pick%REGION,(30,30,30)),dict(y0=y0,x0=x0,size=REGION,pixels=TRIGGER_PIXELS)

def baseline_poison_sample(sample,method,attack,beta=60.,alpha_degrees=9.,split_fraction=.5):
    """Same return shape as poison_pipeline.poison_sample so write_poisoned_sample works unchanged."""
    if method not in METHODS: raise ValueError(f'Unknown baseline {method!r}; choose from {METHODS}')
    image=sample.read()
    if method=='badnets': poisoned,info=trigger_badnets(image,sample.key)
    elif method=='blended': poisoned,info=trigger_blended(image,sample.key)
    elif method=='ld_attack': poisoned,info=trigger_ld_attack(image,sample.key,sample)
    else: poisoned,info=trigger_badlane(image,sample.key,sample)
    existence=[sum(1 for x in xs if x>=0)>=2 for xs in sample.lanes]
    points=[list(zip(xs,sample.h_samples)) for xs in sample.lanes]
    new_points,new_existence=apply_attack(points,existence,attack,beta=beta,alpha_degrees=alpha_degrees,split_fraction=split_fraction)
    return dict(raw_file=sample.key,h_samples=sample.h_samples,original_lanes=sample.lanes,original_existence=existence,
        poisoned_lanes=[points_to_xs(p,sample.h_samples) for p in new_points],poisoned_existence=new_existence,
        attack=attack,method=method,trigger_window=info,trigger_candidates=1,placeholder_trigger=False,
        placeholder_road_mask=method=='badlane',recipe_assumptions=ASSUMPTIONS[method],poisoned_image=poisoned)
