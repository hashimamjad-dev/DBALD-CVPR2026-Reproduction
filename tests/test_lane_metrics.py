import pytest
from lane_repro.lane_metrics import (point_accuracy,clip_accuracy,dataset_accuracy,
    lda_disappearance_fraction,dataset_lda_asr,THRESHOLD_PX)

def lane(xs,ys): return list(zip(xs,ys))

def test_point_accuracy_exact_match_is_perfect():
    gt=lane([10,20,30],[0,1,2])
    assert point_accuracy(gt,gt)==(3,3)

def test_point_accuracy_threshold_boundary():
    gt=lane([100],[0])
    assert point_accuracy(lane([100+THRESHOLD_PX],[0]),gt)==(1,1)
    assert point_accuracy(lane([100+THRESHOLD_PX+0.001],[0]),gt)==(0,1)

def test_point_accuracy_ignores_invalid_ground_truth_points():
    gt=lane([10,-2,30],[0,1,2])
    assert point_accuracy(lane([10,999,30],[0,1,2]),gt)==(2,2)

def test_point_accuracy_missing_predicted_sample_counts_as_wrong():
    gt=lane([10,20],[0,1])
    assert point_accuracy(lane([10],[0]),gt)==(1,2)

def test_point_accuracy_rejects_empty_ground_truth():
    with pytest.raises(ValueError): point_accuracy(lane([1],[0]),lane([],[]))

def test_clip_accuracy_vanished_lane_contributes_zero():
    gt=[lane([10,20],[0,1]),lane([50,60],[0,1])]
    predicted=[lane([10,20],[0,1])]
    assert clip_accuracy(predicted,gt)==pytest.approx(0.5)

def test_clip_accuracy_perfect_prediction_is_one():
    gt=[lane([10,20],[0,1]),lane([50,60],[0,1])]
    assert clip_accuracy(gt,gt)==pytest.approx(1.0)

def test_dataset_accuracy_aggregates_points_not_image_ratios():
    gt_a=[lane([0],[0])]; gt_b=[lane([0,0,0],[0,1,2])]
    pred_a=[lane([0],[0])]
    pred_b=[lane([999,0,0],[0,1,2])]
    expected=(1+2)/(1+3)
    assert dataset_accuracy([(pred_a,gt_a),(pred_b,gt_b)])==pytest.approx(expected)

def test_dataset_accuracy_rejects_all_invalid_data():
    with pytest.raises(ValueError): dataset_accuracy([([],[lane([-1],[0])])])

def test_lda_fraction_full_and_zero():
    assert lda_disappearance_fraction([[],[]],[True,True])==1.0
    gt=[lane([10,20],[0,1]),lane([50,60],[0,1])]
    assert lda_disappearance_fraction(gt,[True,True])==0.0

def test_lda_fraction_ignores_originally_absent_lanes():
    assert lda_disappearance_fraction([[]],[True,False])==1.0

def test_lda_single_point_counts_as_disappeared():
    assert lda_disappearance_fraction([lane([10],[0])],[True])==1.0

def test_lda_fraction_rejects_no_present_lanes():
    with pytest.raises(ValueError): lda_disappearance_fraction([[]],[False])

def test_dataset_lda_asr_aggregates_across_images():
    pairs=[([[]],[True]),([lane([10,20],[0,1])],[True]),([[]],[False])]
    assert dataset_lda_asr(pairs)==pytest.approx(0.5)
