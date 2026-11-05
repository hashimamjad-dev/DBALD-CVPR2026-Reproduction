import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from lane_repro.laneatt_heatmap import build_target,laneatt_attention_map

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'models/checkpoints/laneatt_official/experiments/laneatt_r18_tusimple/models/model_0100.pt'
THIRD_PARTY=ROOT/'third_party/LaneATT/lib/models/laneatt.py'
HAVE_LANEATT=CHECKPOINT.exists() and THIRD_PARTY.exists() and importlib.util.find_spec('torchvision')
needs_laneatt=pytest.mark.skipif(not HAVE_LANEATT,reason='LaneATT checkpoint/third_party/torchvision not set up on this machine')

@pytest.mark.parametrize('kind', ['no_match', 'differentiable', 'detached', 'invalid_scalar'])
def test_attention_fallback_only_accepts_no_match_scalar_zero(kind):
    from types import SimpleNamespace

    class Model:
        def __call__(self, x):
            return x

        def loss(self, proposals, targets, **kwargs):
            loss = {'no_match': 0., 'differentiable': proposals.sum(),
                    'detached': torch.tensor(1.), 'invalid_scalar': 1.}[kind]
            return loss, {'cls_loss': loss, 'reg_loss': loss}

    adapter = SimpleNamespace(max_lanes=2, model=Model(),
                              preprocess=lambda rgb: torch.zeros(1, 3, 8, 8))
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    if kind in ('detached', 'invalid_scalar'):
        with pytest.raises(ValueError, match='differentiable'):
            laneatt_attention_map(adapter, image, [], 'LDA')
    else:
        attention = laneatt_attention_map(adapter, image, [], 'LDA')
        assert np.all(attention == (0. if kind == 'no_match' else 3.))

def test_build_target_shape_and_default_invalid_sentinel():
    t=build_target([],100,100,max_lanes=5,s=10)
    assert t.shape==(5,15) and (t[:,:2]==torch.tensor([1.,0.])).all() and (t[:,2:]==-1e5).all()

def test_build_target_flags_present_lane():
    lane=[(x,y) for x,y in zip(range(10,110,10),range(10,110,10))]
    t=build_target([lane],100,100,max_lanes=3,s=10)
    assert t[0,1]==1 and t[1,1]==0 and t[2,1]==0

def test_build_target_drops_invalid_points_before_fitting():
    """Regression test: without filtering x<0, invalid points corrupted interpolation/extrapolation
    and produced zero positive anchor matches against the real model (found running against real
    LaneATT output). A lane padded with invalid (-2) rows must score identically to the same lane
    with those rows simply absent."""
    clean=[(x,y) for x,y in zip(range(10,110,10),range(10,110,10))]
    padded=clean+[(-2,200),(-2,210),(-2,220)]
    a=build_target([clean],300,300,max_lanes=2,s=20)
    b=build_target([padded],300,300,max_lanes=2,s=20)
    assert torch.equal(a,b)

def test_build_target_drops_short_lanes():
    """Short lanes are dropped before slot assignment, so the surviving lane occupies slot 0
    (compacted), not the slot it originally had in the input list."""
    t=build_target([[(10,10)],[(10,10),(20,20)]],100,100,max_lanes=2,s=10)
    assert t[0,1]==1 and t[1,1]==0

def test_build_target_truncates_to_max_lanes():
    lane=[(x,y) for x,y in zip(range(10,110,10),range(10,110,10))]
    t=build_target([lane]*4,100,100,max_lanes=2,s=10)
    assert t.shape[0]==2

@needs_laneatt
def test_laneatt_attention_map_real_model_end_to_end():
    from lane_repro.models import LaneATTAdapter
    from lane_repro.data import tusimple
    adapter=LaneATTAdapter(CHECKPOINT,device='cpu')
    sample=next(iter(tusimple(ROOT/'datasets/raw/tusimple_subset',ROOT/'datasets/raw/tusimple_subset/labels.jsonl')))
    rgb=sample.read()
    lanes_xy=[list(zip(xs,sample.h_samples)) for xs in sample.lanes]
    lda=laneatt_attention_map(adapter,rgb,lanes_xy,'LDA')
    loa=laneatt_attention_map(adapter,rgb,lanes_xy,'LOA')
    assert lda.shape==(sample.height,sample.width)==loa.shape
    assert np.isfinite(lda).all() and (lda>=0).all()
    assert np.isfinite(loa).all() and (loa>=0).all()
    assert lda.max()>0 and loa.max()>0  # not a degenerate all-zero map
    assert not np.allclose(lda,loa)  # LDA (cls_loss) and LOA (reg_loss) use different loss components

@needs_laneatt
def test_laneatt_attention_map_rejects_unknown_attack():
    from lane_repro.models import LaneATTAdapter
    from lane_repro.data import tusimple
    adapter=LaneATTAdapter(CHECKPOINT,device='cpu')
    sample=next(iter(tusimple(ROOT/'datasets/raw/tusimple_subset',ROOT/'datasets/raw/tusimple_subset/labels.jsonl')))
    lanes_xy=[list(zip(xs,sample.h_samples)) for xs in sample.lanes]
    with pytest.raises(ValueError): laneatt_attention_map(adapter,sample.read(),lanes_xy,'XYZ')
