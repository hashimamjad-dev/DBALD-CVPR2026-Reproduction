import importlib.util
import numpy as np
import pytest
from lane_repro.stealthiness import frequency_energy_ratio,LPIPSScorer

HAVE_LPIPS=importlib.util.find_spec('lpips') is not None
needs_lpips=pytest.mark.skipif(not HAVE_LPIPS,reason='lpips package not installed on this machine')

def smooth_region(size=64,value=120):
    return np.full((size,size,3),value,np.uint8)

def noisy_region(size=64,seed=0):
    rng=np.random.default_rng(seed)
    return rng.integers(0,256,(size,size,3),dtype=np.uint8)

def test_frequency_energy_ratio_identical_regions_is_one():
    region=noisy_region()
    assert frequency_energy_ratio(region,region)==pytest.approx(1.0)

def test_frequency_energy_ratio_detects_added_high_frequency_content():
    rng=np.random.default_rng(1)
    clean=np.full((64,64,3),120,np.int16)+rng.integers(-3,4,(64,64,3))  # a little texture, like a real photo
    clean=clean.clip(0,255).astype(np.uint8)
    poisoned=clean.copy()
    poisoned[::2,::2]=255  # checkerboard: strong high-frequency content injected on top
    assert frequency_energy_ratio(clean,poisoned)>2.0

def test_frequency_energy_ratio_rejects_shape_mismatch():
    with pytest.raises(ValueError): frequency_energy_ratio(smooth_region(64),smooth_region(32))

def test_frequency_energy_ratio_rejects_degenerate_clean_region():
    flat=smooth_region()
    with pytest.raises(ValueError): frequency_energy_ratio(flat,noisy_region())

@needs_lpips
def test_lpips_scorer_identical_images_near_zero():
    scorer=LPIPSScorer()
    image=noisy_region(128)
    assert scorer.distance(image,image)==pytest.approx(0.0,abs=1e-4)

@needs_lpips
def test_lpips_scorer_different_images_positive():
    scorer=LPIPSScorer()
    a=np.zeros((128,128,3),np.uint8); b=np.full((128,128,3),255,np.uint8)
    assert scorer.distance(a,b)>0.1

@needs_lpips
def test_lpips_scorer_respects_box_crop():
    scorer=LPIPSScorer()
    a=np.zeros((128,128,3),np.uint8); b=a.copy(); b[40:60,40:60]=255
    inside=scorer.distance(a,b,box=(40,40,60,60))
    outside=scorer.distance(a,b,box=(0,0,20,20))
    assert inside>outside

@needs_lpips
def test_lpips_scorer_rejects_shape_mismatch_and_empty_box():
    scorer=LPIPSScorer()
    with pytest.raises(ValueError): scorer.distance(smooth_region(64),smooth_region(32))
    with pytest.raises(ValueError): scorer.distance(smooth_region(64),smooth_region(64),box=(10,10,10,10))
