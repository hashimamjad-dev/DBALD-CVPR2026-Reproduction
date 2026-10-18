import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from lane_repro.nms import lane_nms
from lane_repro.models import LaneATTAdapter,sample_predictions
from lane_repro.data import contained_path,tusimple,culane
from lane_repro.evaluation import official_tusimple,geometry
from lane_repro.prototype import consistency,inpaint,windows,sensitivity

ROOT=Path(__file__).resolve().parents[1]

def test_reference_integrity():
    folder=ROOT/'results/reference'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert hashlib.sha256((folder/'table1.csv').read_bytes()).hexdigest()==manifest['sha256']
    rows=list(csv.DictReader((folder/'table1.csv').open()))
    assert len(rows)==20
    assert rows[2]['culane_lda_asr']=='71.24'
    assert rows[19]['tusimple_avg_asr']=='95.43'
    assert all('tusimple_lra_asr' not in row for row in rows)

def test_bgr_cubic_preprocessing():
    adapter=LaneATTAdapter.__new__(LaneATTAdapter); adapter.device=torch.device('cpu')
    rgb=np.zeros((720,1280,3),np.uint8); rgb[...,0]=255; rgb[...,2]=64
    x=adapter.preprocess(rgb)
    assert x.shape==(1,3,360,640)
    assert x[0,0,0,0]==pytest.approx(64/255)
    assert x[0,2,0,0]==1
    import cv2
    rng=np.random.default_rng(1); rgb=rng.integers(0,256,(720,1280,3),dtype=np.uint8)
    expected=cv2.resize(rgb[...,::-1],(640,360),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
    np.testing.assert_allclose(adapter.preprocess(rgb)[0].permute(1,2,0).numpy(),expected,atol=1e-7)

def boxes(xs):
    result=torch.zeros(len(xs),77); result[:,4]=72
    for i,x in enumerate(xs): result[i,5:]=x
    return result

def test_nms_duplicate_threshold_and_topk():
    b=boxes([100,105,150,250]); s=torch.tensor([.9,.8,.7,.6])
    keep,n,_=lane_nms(b,s,overlap=10,top_k=2)
    assert keep[:n].tolist()==[0,2]
    # Strict '<', not '<=' at exact threshold.
    keep,n,_=lane_nms(boxes([100,110]),s[:2],overlap=10,top_k=5)
    assert keep[:n].tolist()==[0,1]

def test_nms_no_shared_support():
    b=boxes([100,100]); b[0,4]=10; b[1,2]=.5
    keep,n,_=lane_nms(b,torch.tensor([.9,.8]),overlap=10,top_k=5)
    assert n==2

def test_official_tusimple_perfect_missing_and_runtime():
    evaluate=official_tusimple()
    gt=[[-2,100,100,100],[-2,300,300,300]]; ys=[0,10,20,30]
    assert evaluate.bench(gt,gt,ys,10)==(1.,0.,0.)
    assert evaluate.bench([],gt,ys,10)==(0.,0.,1.)
    assert evaluate.bench(gt,gt,ys,201)==(0.,0.,1.)

def test_geometry_missing_not_success():
    result=geometry([],[[100,100,-2]])
    assert result['point_accuracy_20px']==0 and result['coverage']==0 and result['mae_px'] is None
    assert geometry([[101,101,-2]],[[100,100,-2]])['mae_px']==1

def test_paths_and_label_validation(tmp_path):
    with pytest.raises(ValueError): contained_path(tmp_path,'../outside')
    assert contained_path(tmp_path,'/clips/a.jpg')==tmp_path/'clips/a.jpg'
    label=tmp_path/'labels.jsonl'
    label.write_text(json.dumps(dict(raw_file='a.jpg',h_samples=[200,210],lanes=[[100]])))
    with pytest.raises(ValueError): list(tusimple(tmp_path,label))

def test_culane_pairs(tmp_path):
    (tmp_path/'a.lines.txt').write_text('20 50 10 30\n')
    listing=tmp_path/'test.txt'; listing.write_text('/a.jpg\n')
    s=list(culane(tmp_path,listing))[0]
    assert s.lanes==[[[10.,30.],[20.,50.]]]

def test_inpaint_preserves_outside_and_ssim_identity():
    image=np.full((32,32,3),128,np.uint8); mask=np.zeros((32,32),bool); mask[10:15,10:15]=True
    result=inpaint(image,mask)
    np.testing.assert_array_equal(image[~mask],result[~mask])
    metrics=consistency(image,image,mask)
    assert metrics['masked_array_ssim']==pytest.approx(1) and metrics['selected_pixel_mse']==0
    with pytest.raises(ValueError): consistency(image,image,np.zeros_like(mask))

def test_windows_and_gradient():
    assert windows(np.ones((10,10)),(4,4),4)[-1]==(6,6,4,4)
    x=torch.ones(1,3,4,4)
    heat=sensitivity(torch.nn.Identity(),x,lambda y:y.sum())
    assert torch.equal(heat,torch.full((1,4,4),3.))

def test_spline_conversion_invalid_coordinates():
    def lane(ys): return np.array([-.1,.5,1.1])
    assert sample_predictions([lane],[0,100,200],1280,720)==[[-2.,640.,1408.]]

def test_deform_conv_zero_offsets_matches_conv():
    from lane_repro.adnet_compat import DeformConv2d
    torch.manual_seed(1)
    conv=DeformConv2d(4,3,3,padding=1,deform_groups=2)
    x=torch.randn(1,4,8,8); offsets=torch.zeros(1,36,8,8)
    expected=torch.nn.functional.conv2d(x,conv.weight,padding=1)
    torch.testing.assert_close(conv(x,offsets),expected,rtol=1e-5,atol=1e-5)
