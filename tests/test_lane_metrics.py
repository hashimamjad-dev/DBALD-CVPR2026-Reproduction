import importlib.util
import json
from pathlib import Path
import pytest
from lane_repro.lane_metrics import (point_accuracy,clip_accuracy,dataset_accuracy,
    lda_disappearance_fraction,dataset_lda_asr,asr_loa_lra,dataset_asr_loa_lra,THRESHOLD_PX)

ROOT=Path(__file__).resolve().parents[1]
EVALUATOR=ROOT/'third_party/tusimple-benchmark/evaluate/lane.py'
# The official evaluator (vendored, needed by asr_loa_lra/dataset_asr_loa_lra) requires its own repo
# clone plus scikit-learn and ujson; skip only the tests that need it on a machine without that setup.
HAVE_EVALUATOR=EVALUATOR.exists() and all(importlib.util.find_spec(m) for m in ('sklearn','ujson'))
needs_evaluator=pytest.mark.skipif(not HAVE_EVALUATOR,reason='third_party/tusimple-benchmark + sklearn/ujson not set up on this machine')

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

@needs_evaluator
def test_asr_loa_lra_perfect_match_is_one():
    xs=[10,20,30,40]; h=[0,1,2,3]
    assert asr_loa_lra([xs],[xs],h)==pytest.approx(1.0)

@needs_evaluator
def test_asr_loa_lra_far_off_target_is_zero():
    h=[0,1,2,3]
    assert asr_loa_lra([[10,20,30,40]],[[10000,10000,10000,10000]],h)==pytest.approx(0.0)

@needs_evaluator
def test_asr_loa_lra_matches_best_predicted_lane_not_list_position():
    """The whole point of this wrapper: a target lane at index 0 can be matched by a predicted lane
    at a different index, unlike the deprecated point_accuracy/clip_accuracy functions above."""
    h=[0,1,2,3]; xs=[10,20,30,40]
    predicted=[[-2,-2,-2,-2],xs]  # correct lane is second, wrong lane is first
    assert asr_loa_lra(predicted,[xs],h)==pytest.approx(1.0)

@needs_evaluator
def test_asr_loa_lra_zero_above_200ms_gate():
    xs=[10,20,30,40]; h=[0,1,2,3]
    assert asr_loa_lra([xs],[xs],h,running_time=201.)==pytest.approx(0.0)

@needs_evaluator
def test_dataset_asr_loa_lra_averages_per_image():
    xs=[10,20,30,40]; h=[0,1,2,3]; off=[1000,1000,1000,1000]
    triples=[([xs],[xs],h),([off],[xs],h)]
    assert dataset_asr_loa_lra(triples)==pytest.approx(0.5)

@needs_evaluator
def test_dataset_asr_loa_lra_rejects_empty():
    with pytest.raises(ValueError): dataset_asr_loa_lra([])

@needs_evaluator
def test_naive_positional_metric_disagrees_with_official_evaluator_on_real_predictions():
    """Regression test for the 2026-10-23 finding documented in lane_metrics.py's module docstring:
    on real LaneATT output, positional pairing (point_accuracy/dataset_accuracy) and the real official
    evaluator (asr_loa_lra, used against clean ground truth here) disagree by a wide margin, because
    predicted lane order has no guaranteed correspondence to TuSimple's ground-truth lane order.
    Skipped if the real downloaded predictions/labels this test depends on are not present."""
    pred_path=ROOT/'results/measured/laneatt_cpu_independent/predictions.jsonl'
    label_path=ROOT/'datasets/raw/tusimple_subset/labels.jsonl'
    if not (pred_path.exists() and label_path.exists()):
        pytest.skip('Real evaluation outputs not present on this machine')
    preds={json.loads(l)['raw_file']:json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()}
    labels={json.loads(l)['raw_file']:json.loads(l) for l in label_path.read_text().splitlines() if l.strip()}
    assert set(preds)==set(labels)
    naive_pairs=[]; official_triples=[]
    for key in preds:
        p,g=preds[key],labels[key]
        assert p['h_samples']==g['h_samples']
        naive_pairs.append(([lane(xs,p['h_samples']) for xs in p['lanes']],[lane(xs,g['h_samples']) for xs in g['lanes']]))
        official_triples.append((p['lanes'],g['lanes'],p['h_samples']))
    naive=dataset_accuracy(naive_pairs)
    official=dataset_asr_loa_lra(official_triples)
    assert official>0.9  # matches the independently-run evaluate.py summary (~96.7%)
    assert naive<official-0.3  # the naive positional metric understates accuracy by a wide margin
