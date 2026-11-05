from pathlib import Path
import cv2
import numpy as np
import pytest
from lane_repro.data import Sample
from lane_repro.poison_pipeline import (road_region_mask,place_trigger_window,
    paste_placeholder_trigger,points_to_xs,poison_sample,write_poisoned_sample)

WIDTH,HEIGHT=160,120
H_SAMPLES=[40,60,80,100]

def make_sample(tmp_path,xs,subdir='clean'):
    image=np.zeros((HEIGHT,WIDTH,3),np.uint8); image[:]=(200,200,200)
    path=tmp_path/subdir/'img.jpg'; path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),image)
    return Sample('img.jpg',path,[xs],WIDTH,HEIGHT,H_SAMPLES)

def straight_lane(x0=30,step=10): return [x0+step*i for i in range(len(H_SAMPLES))]

def test_road_region_mask_true_near_lane_false_far_away():
    mask=road_region_mask([straight_lane()],H_SAMPLES,WIDTH,HEIGHT,margin=20,lane_exclude_radius=5)
    assert mask[H_SAMPLES[0],45]>0 and mask[H_SAMPLES[0],WIDTH-1]==0  # near-but-not-on the lane at x=30

def test_road_region_mask_is_a_continuous_band_between_labelled_rows():
    mask=road_region_mask([straight_lane()],H_SAMPLES,WIDTH,HEIGHT,margin=20,lane_exclude_radius=5)
    for y in range(H_SAMPLES[0],H_SAMPLES[-1]+1): assert mask[y,:].sum()>0

def test_road_region_mask_excludes_the_lane_line_itself():
    mask=road_region_mask([straight_lane()],H_SAMPLES,WIDTH,HEIGHT,margin=20,lane_exclude_radius=5)
    for x,y in zip(straight_lane(),H_SAMPLES): assert mask[y,x]==0

def test_road_region_mask_ignores_unlabelled_rows():
    mask=road_region_mask([straight_lane()],H_SAMPLES,WIDTH,HEIGHT)
    assert mask[5,:].sum()==0  # above the first labelled row

def test_road_region_mask_rejects_length_mismatch():
    with pytest.raises(ValueError): road_region_mask([[1,2,3]],H_SAMPLES,WIDTH,HEIGHT)

def test_road_region_mask_rejects_out_of_range_h_samples():
    with pytest.raises(ValueError): road_region_mask([straight_lane()],[40,60,80,HEIGHT+10],WIDTH,HEIGHT)

def test_place_trigger_window_uniform_placeholder_stays_in_mask():
    mask=road_region_mask([straight_lane()],H_SAMPLES,WIDTH,HEIGHT,margin=25)
    (y0,x0),scored=place_trigger_window(mask,size=8,stride=4)
    assert mask[y0:y0+8,x0:x0+8].min()>0 and len(scored)>0

def test_place_trigger_window_follows_a_real_attention_map():
    mask=np.full((HEIGHT,WIDTH),255,np.uint8)
    attention=np.zeros((HEIGHT,WIDTH)); attention[50:60,70:80]=100.
    (y0,x0),_=place_trigger_window(mask,size=8,stride=4,attention_map=attention)
    assert 42<=y0<=60 and 62<=x0<=80  # the high-attention region, allowing for window/stride slack

def test_paste_placeholder_trigger_fills_exact_window():
    image=np.zeros((HEIGHT,WIDTH,3),np.uint8)
    out=paste_placeholder_trigger(image,10,20,8,(1,2,3))
    assert tuple(out[10,20])==(1,2,3) and tuple(out[0,0])==(0,0,0)

def test_points_to_xs_roundtrips_loa_shift():
    points=list(zip(straight_lane(),H_SAMPLES))
    shifted=[(x+5,y) for x,y in points]
    assert points_to_xs(shifted,H_SAMPLES)==[x+5 for x in straight_lane()]

def test_points_to_xs_lda_erase_becomes_all_invalid():
    assert points_to_xs([],H_SAMPLES)==[-2.]*len(H_SAMPLES)

def test_points_to_xs_handles_fewer_points_than_h_samples():
    """Covers label_attacks.lra_rotate's shorter-than-input return: missing rows become -2."""
    partial=[(99,H_SAMPLES[1]),(100,H_SAMPLES[2])]
    result=points_to_xs(partial,H_SAMPLES)
    assert result[0]==-2. and result[1]==99 and result[2]==100 and result[3]==-2.
    assert len(result)==len(H_SAMPLES)

@pytest.mark.parametrize('attack',['LOA','LRA','LDA'])
def test_poison_sample_end_to_end(tmp_path,attack):
    sample=make_sample(tmp_path,straight_lane())
    result=poison_sample(sample,attack,trigger_size=8,stride=4)
    assert result['poisoned_image'].shape==(HEIGHT,WIDTH,3)
    assert len(result['poisoned_lanes'])==1 and len(result['poisoned_lanes'][0])==len(H_SAMPLES)
    assert result['original_existence']==[True]
    assert result['poisoned_existence']==([False] if attack=='LDA' else [True])
    if attack=='LDA': assert result['poisoned_lanes'][0]==[-2.]*len(H_SAMPLES)
    y0,x0,size=result['trigger_window'].values()
    assert 0<=y0<=HEIGHT-size and 0<=x0<=WIDTH-size

def test_poison_sample_loa_shifts_by_beta(tmp_path):
    sample=make_sample(tmp_path,straight_lane())
    result=poison_sample(sample,'LOA',trigger_size=8,stride=4,beta=7.)
    assert result['poisoned_lanes'][0]==[x+7. for x in straight_lane()]

def test_write_poisoned_sample_writes_file_and_omits_image_array(tmp_path):
    sample=make_sample(tmp_path,straight_lane())
    result=poison_sample(sample,'LOA',trigger_size=8,stride=4)
    record=write_poisoned_sample(result,tmp_path/'poisoned',tmp_path/'clean')
    assert (tmp_path/'poisoned'/'img.jpg').exists() and 'poisoned_image' not in record
    assert record['raw_file']=='img.jpg' and record['lanes']==result['poisoned_lanes']

def test_write_poisoned_sample_rejects_overlapping_output_root(tmp_path):
    sample=make_sample(tmp_path,straight_lane())
    result=poison_sample(sample,'LOA',trigger_size=8,stride=4)
    with pytest.raises(ValueError): write_poisoned_sample(result,tmp_path/'clean',tmp_path/'clean')
    with pytest.raises(ValueError): write_poisoned_sample(result,tmp_path/'clean'/'nested',tmp_path/'clean')

def test_poison_sample_render_trigger_swap_is_used_instead_of_placeholder(tmp_path):
    """Covers the ultraedit_trigger.py integration point without needing a GPU: any callable
    matching (image, y0, x0, size, prompt) -> image can replace paste_placeholder_trigger."""
    sample=make_sample(tmp_path,straight_lane())
    calls=[]
    def fake_renderer(image,y0,x0,size,prompt):
        calls.append((y0,x0,size,prompt))
        out=image.copy(); out[y0:y0+size,x0:x0+size]=(1,2,3); return out
    result=poison_sample(sample,'LOA',trigger_size=8,stride=4,
        render_trigger=fake_renderer,trigger_prompt='Add a small and sparse brown mud')
    y0,x0,size=result['trigger_window'].values()
    assert calls==[(y0,x0,size,'Add a small and sparse brown mud')]
    assert tuple(result['poisoned_image'][y0,x0])==(1,2,3)
    assert result['placeholder_trigger'] is False

def test_poison_sample_render_trigger_requires_prompt(tmp_path):
    sample=make_sample(tmp_path,straight_lane())
    with pytest.raises(ValueError):
        poison_sample(sample,'LOA',trigger_size=8,stride=4,render_trigger=lambda *a:None)
