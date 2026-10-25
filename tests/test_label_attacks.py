import math
import pytest
from lane_repro.label_attacks import loa_offset,lda_erase,lra_rotate,apply_attack

def lane(xs,ys): return list(zip(xs,ys))

def test_loa_offset_shifts_every_valid_point():
    assert loa_offset(lane([10,20,30],[0,1,2]),5)==lane([15,25,35],[0,1,2])

def test_loa_offset_leaves_invalid_points_untouched():
    assert loa_offset(lane([10,-2,30],[0,1,2]),5)==lane([15,-2,35],[0,1,2])

def test_loa_offset_matches_paper_beta_sign():
    assert loa_offset([(100,50)],60)==[(160,50)]

def test_lda_erase_returns_empty():
    assert lda_erase(lane([10,20],[0,1]))==[]

def test_lra_rotate_requires_two_points():
    with pytest.raises(ValueError): lra_rotate([(10,0)],9)

def test_lra_rotate_zero_angle_is_identity_up_to_refit():
    points=lane([10,20,30,40],[0,10,20,30])
    result=lra_rotate(points,0.,split_fraction=.5)
    for (x1,y1),(x2,y2) in zip(points,result):
        assert y1==y2 and x1==pytest.approx(x2,abs=1e-6)

def test_lra_rotate_keeps_prefix_before_split_unchanged():
    points=lane([0,10,20,30,40],[0,10,20,30,40])
    result=lra_rotate(points,30.,split_fraction=.4)
    assert result[0]==points[0]

def test_lra_rotate_output_is_on_original_y_grid():
    points=lane([0,10,20,30,40],[0,10,20,30,40])
    result=lra_rotate(points,45.,split_fraction=.4)
    assert [y for _,y in result]==[y for _,y in points] and len(result)==len(points)

def test_lra_rotate_nonzero_angle_moves_suffix_x():
    points=lane([0,0,0,0],[0,10,20,30])
    result=lra_rotate(points,90.,split_fraction=.25,degree=1)
    assert any(abs(rx-ox)>1 for (rx,_),(ox,_) in zip(result[1:],points[1:]))

def test_apply_attack_only_modifies_present_lanes():
    lanes=[lane([10,20],[0,1]),lane([50,60],[0,1])]
    out_lanes,out_existence=apply_attack(lanes,[True,False],'LOA',beta=5)
    assert out_lanes[0]==lane([15,25],[0,1]) and out_lanes[1]==lanes[1] and out_existence==[True,False]

def test_apply_attack_lda_clears_existence_only_for_present():
    lanes=[lane([10,20],[0,1]),lane([50,60],[0,1])]
    out_lanes,out_existence=apply_attack(lanes,[True,False],'LDA')
    assert out_lanes[0]==[] and out_existence==[False,False] and out_lanes[1]==lanes[1]

def test_apply_attack_lra_forwards_params():
    lanes=[lane([0,10,20,30],[0,10,20,30])]
    out_lanes,out_existence=apply_attack(lanes,[True],'LRA',alpha_degrees=9.,split_fraction=.5)
    assert out_existence==[True] and [y for _,y in out_lanes[0]]==[0,10,20,30]

def test_apply_attack_rejects_unknown_strategy():
    with pytest.raises(ValueError): apply_attack([lane([1],[0])],[True],'XYZ')

def test_apply_attack_rejects_length_mismatch():
    with pytest.raises(ValueError): apply_attack([lane([1],[0])],[True,False],'LOA',beta=1)
