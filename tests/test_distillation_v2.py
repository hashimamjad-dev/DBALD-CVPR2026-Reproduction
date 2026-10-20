import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from v2_common import Quality,balanced_loss,save_checkpoint

def test_balanced_loss_exposes_small_region_error():
    target=torch.zeros(1,4,32,32); pred=target.clone(); pred[:,:,12:20,12:20]=1
    condition=torch.zeros(1,5,32,32); condition[:,0,12:20,12:20]=1
    assert balanced_loss(pred,target,condition)>2*(pred-target).square().mean()
    assert balanced_loss(target,target,condition)==0

def test_local_metrics_detect_edit_and_boundary_error():
    torch.set_num_threads(2); metric=Quality()
    rng=np.random.default_rng(7); original=rng.integers(20,180,size=(128,128,3),dtype=np.uint8)
    mask=np.zeros((128,128),np.uint8); mask[40:88,40:88]=255
    c=dict(crop=original,mask=mask,lane_mask=np.ones((128,128),np.uint8),crop_xy=(0,0))
    same=metric.measure(original,original,c)
    edited=original.copy(); edited[mask>0]+=30
    changed=metric.measure(edited,original,c)
    assert same['masked_mae']==0 and same['boundary_error']==0 and abs(same['local_lpips'])<1e-8
    assert abs(changed['masked_mae']-30)<1e-8 and abs(changed['boundary_error']-30)<1e-8
    assert changed['masked_ssim']<1 and changed['local_lpips']>0

def test_interrupted_checkpoint_keeps_previous_state(tmp_path,monkeypatch):
    import pytest
    path=tmp_path/'last.pt'; save_checkpoint(path,{'step':100})
    def interrupted(value,target):
        target.write_bytes(b'incomplete checkpoint')
        raise OSError('simulated interrupted write')
    monkeypatch.setattr(torch,'save',interrupted)
    with pytest.raises(OSError): save_checkpoint(path,{'step':200})
    assert torch.load(path,weights_only=True)['step']==100
