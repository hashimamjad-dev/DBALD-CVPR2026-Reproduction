from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_culane_protocol import witnesses


def test_native_f1_and_candidate_acc_are_distinct(tmp_path):
    result=witnesses(tmp_path)
    rows={r['offset_px']:r for r in result['offset_witnesses']}
    assert rows[0]['official_lane_f1_percent']==rows[0]['candidate_point_acc_percent']==100
    assert rows[19]['candidate_point_acc_percent']==100 and rows[19]['official_lane_f1_percent']==0
    assert rows[20]['candidate_point_acc_percent']==0
    assert result['matching_witness']==dict(candidate_one_to_one_percent=50.,official_tusimple_best_per_gt_percent=100.)
    assert result['real_benchmark_images']==0 and result['paper_equivalence_verified'] is False
