import pytest
import torch
from lane_repro.trigger_placement import (sliding_window_candidates,gradient_attention_map,
    select_optimal_window,optimal_trigger_position)

def full_mask(h,w): return [[True]*w for _ in range(h)]

def test_sliding_window_candidates_on_full_mask_covers_every_stride_position():
    candidates=sliding_window_candidates(full_mask(6,6),size=2,stride=2)
    assert candidates==[(0,0),(0,2),(0,4),(2,0),(2,2),(2,4),(4,0),(4,2),(4,4)]

def test_sliding_window_candidates_excludes_windows_touching_a_hole():
    mask=full_mask(6,6); mask[1][1]=False
    candidates=sliding_window_candidates(mask,size=2,stride=1)
    assert (0,0) not in candidates and (1,1) not in candidates and (2,2) in candidates

def test_sliding_window_candidates_rejects_window_larger_than_mask():
    with pytest.raises(ValueError): sliding_window_candidates(full_mask(3,3),size=5,stride=1)

def test_sliding_window_candidates_rejects_nonpositive_size_or_stride():
    with pytest.raises(ValueError): sliding_window_candidates(full_mask(4,4),size=0,stride=1)
    with pytest.raises(ValueError): sliding_window_candidates(full_mask(4,4),size=2,stride=0)

def test_gradient_attention_map_follows_a_known_linear_weight():
    x=torch.zeros(1,2,4,4,requires_grad=True)
    w=torch.zeros(1,2,4,4); w[0,:,1,1]=5.; w[0,:,0,0]=1.
    loss=(x*w).sum()
    attention=gradient_attention_map(x,loss)
    assert attention.shape==(1,4,4)
    assert attention[0,1,1].item()==pytest.approx(10.) and attention[0,0,0].item()==pytest.approx(2.)
    assert attention[0,1,1]>attention[0,0,0]>attention[0,2,2]

def test_gradient_attention_map_requires_grad_enabled_input():
    x=torch.zeros(1,1,2,2)
    with pytest.raises(ValueError): gradient_attention_map(x,x.sum())

def test_select_optimal_window_picks_highest_scoring_candidate():
    attention=[[0,0,0,0],[0,9,9,0],[0,9,9,0],[0,0,0,0]]
    best,scored=select_optimal_window(attention,[(0,0),(1,1),(2,2)],size=2)
    assert best==(1,1) and scored[0]==((1,1),36.) and [s for _,s in scored]==sorted((s for _,s in scored),reverse=True)

def test_select_optimal_window_sum_and_mean_agree_on_fixed_window_size():
    attention=[[1,5],[2,8]]
    best_sum,_=select_optimal_window(attention,[(0,0)],size=2,aggregate='sum')
    best_mean,_=select_optimal_window(attention,[(0,0)],size=2,aggregate='mean')
    assert best_sum==best_mean==(0,0)

def test_select_optimal_window_rejects_empty_candidates_or_bad_aggregate():
    with pytest.raises(ValueError): select_optimal_window([[1]],[],size=1)
    with pytest.raises(ValueError): select_optimal_window([[1]],[(0,0)],size=1,aggregate='max')

def test_optimal_trigger_position_combines_mask_and_attention():
    mask=full_mask(4,4); mask[0][0]=False
    attention=[[100,0,0,0],[0,0,0,0],[0,0,9,9],[0,0,9,9]]
    best,scored=optimal_trigger_position(mask,attention,size=2,stride=1)
    assert best==(2,2) and (0,0) not in [w for w,_ in scored]
