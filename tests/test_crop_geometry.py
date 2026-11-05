import numpy as np
import pytest
from lane_repro.crop_geometry import tight_box,paste,schedule_indices

def mask(rows,cols):
    m=np.zeros((512,512),np.uint8); m[rows[0]:rows[1],cols[0]:cols[1]]=255; return m

def test_tight_box_contains_tusimple_and_culane_masks():
    for m in (mask((352,448),(192,320)),mask((312,328),(256,352))):
        y0,x0=tight_box(m,256); yy,xx=np.where(m>0)
        assert 0<=y0<=yy.min() and y0+256>=yy.max()+1 and 0<=x0<=xx.min() and x0+256>=xx.max()+1

def test_tight_box_clamps_to_crop_edge():
    y0,x0=tight_box(mask((480,512),(0,40)),128)
    assert (y0,x0)==(384,0)

def test_tight_box_rejects_oversized_mask_and_bad_size():
    with pytest.raises(ValueError): tight_box(mask((0,300),(0,300)),256)
    with pytest.raises(ValueError): tight_box(mask((10,20),(10,20)),250)
    with pytest.raises(ValueError): tight_box(np.zeros((512,512),np.uint8),256)

def test_paste_changes_only_window():
    base=np.zeros((512,512,3),np.uint8); out=paste(base,np.full((256,256,3),9,np.uint8),100,50)
    assert out[100:356,50:306].min()==9 and out.sum()==9*256*256*3 and base.sum()==0

def test_schedule_matches_existing_coarse_indices():
    assert schedule_indices(4)==[0,5,10,15,20] and schedule_indices(20)==list(range(21)) and schedule_indices(2)==[0,10,20]
