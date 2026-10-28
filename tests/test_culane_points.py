import pytest
from lane_repro.culane_point_acc import parse_lanes,point_counts,PROTOCOL


def test_exact_and_strict_boundary():
    gt=parse_lanes('0 0 0 10')
    assert point_counts(gt,gt)==dict(correct=2,total=2)
    assert point_counts(parse_lanes('20 0 20 10'),gt)['correct']==0
    assert point_counts(parse_lanes('19 0 19 10'),gt)['correct']==2


def test_no_extrapolation_or_reusing_prediction():
    gt=parse_lanes('0 0 0 10\n0 0 0 10')
    assert point_counts(gt[:1],gt)==dict(correct=2,total=4)
    assert point_counts(parse_lanes('0 1 0 9'),gt)['correct']==0
    assert point_counts([],gt)==dict(correct=0,total=4)


@pytest.mark.parametrize('text',['nan 0 1 2','1 2 3','0 2 1 2','1 2'])
def test_reject_ambiguous_geometry(text):
    with pytest.raises(ValueError): parse_lanes(text)


def test_candidate_never_claims_author_equivalence():
    assert PROTOCOL['author_verified'] is False
    assert PROTOCOL['table1_comparable'] is False
