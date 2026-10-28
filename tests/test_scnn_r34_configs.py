"""Architecture tests only; random weights must never enter a results table."""
import importlib
from pathlib import Path
import runpy
import torch
import pytest
from lane_repro.models import ROOT, package


@pytest.mark.parametrize('dataset',['tusimple','culane'])
def test_r34_native_resolution_forward(dataset):
    torch.set_num_threads(1)
    repo=ROOT/'third_party/lanedet'
    for suffix in ('','models','models.backbones','models.aggregators','models.heads','models.nets'):
        name='lanedet'+('.'+suffix if suffix else '')
        package(name,repo/name.replace('.','/'))
    for name in ['backbones.resnet','aggregators.scnn','heads.plain_decoder','heads.exist_head','heads.lane_seg','nets.detector']:
        importlib.import_module('lanedet.models.'+name)
    from lanedet.utils.config import Config
    from lanedet.models.registry import build_net
    path=ROOT/'configs'/('scnn_r34_'+dataset+'.py')
    cfg=Config({k:v for k,v in runpy.run_path(str(path)).items() if not k.startswith('__')},filename=str(path))
    assert cfg.backbone.resnet=='resnet34' and cfg.backbone.pretrained is False
    model=build_net(cfg).eval()
    assert len(model.backbone.model.layer1)==3
    assert len(model.backbone.model.layer2)==4
    assert len(model.backbone.model.layer3)==6
    with torch.no_grad(): result=model({'img':torch.zeros(1,3,cfg.img_height,cfg.img_width)})
    assert result['seg'].shape==(1,cfg.num_classes,cfg.img_height,cfg.img_width)
    assert torch.isfinite(result['seg']).all()
