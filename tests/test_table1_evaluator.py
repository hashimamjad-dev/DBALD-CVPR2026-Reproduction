import csv
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_table1 import ROOT, cells, covered, verified_average


def test_reference_complete_and_no_fabricated_values():
    rows = cells(list(csv.DictReader((ROOT/'results/reference/table1.csv').open())), [])
    assert len(rows) == 220
    assert all(r['status'] == 'NOT MEASURED' and r['local_equivalent_percent'] == '' for r in rows)
    assert sum(r['metric'] == 'lra_asr' for r in rows) == 20


@pytest.mark.parametrize('rows,expected,count', [
    ([{'raw_file':'a'}, {'raw_file':'a'}], ['a','b'], 2),
    ([{'raw_file':'b'}], ['a'], 1),
    ([{'raw_file':'a'}], ['a'], 2782),
])
def test_reject_bad_coverage(rows, expected, count):
    with pytest.raises(ValueError):
        covered(rows, expected, count)


def test_derived_average_requires_every_verified_scenario():
    assert verified_average([], 2) is None
    assert verified_average([{'status':'NOT MEASURED'}], 2) is None
    assert verified_average([{'status':'VERIFIED','value_percent':20}], 2) is None
    assert verified_average([{'status':'VERIFIED','value_percent':20}, {'status':'VERIFIED','value_percent':40}], 2) == 30
    with pytest.raises(ValueError):
        verified_average([{'status':'VERIFIED','value_percent':float('nan')}, {'status':'VERIFIED','value_percent':20}], 2)


def test_clean_evidence_never_fills_poisoned_cells():
    records = [dict(model='LaneATT', dataset='TuSimple', metric='benign_acc', id='clean', value_percent=95.6, reason='Unresolved lineage')]
    rows = cells(list(csv.DictReader((ROOT/'results/reference/table1.csv').open())), records)
    assert sum(r['status'] == 'NOT COMPARABLE' for r in rows) == 5
    assert all(r['status'] == 'NOT MEASURED' for r in rows if r['metric'] != 'benign_acc')


def test_poisoned_accuracy_is_method_specific():
    record=dict(model='LaneATT',dataset='TuSimple',metric='poisoned_acc',method='DBALD',id='fixture',value_percent=90,reason='Unresolved lineage')
    rows=cells(list(csv.DictReader((ROOT/'results/reference/table1.csv').open())),[record])
    supporting=[r for r in rows if r['status']=='NOT COMPARABLE']
    assert len(supporting)==1 and supporting[0]['method']=='DBALD'


def test_independent_r34_baseline_does_not_require_private_author_weights():
    record=dict(model='LaneATT',dataset='TuSimple',metric='benign_acc',backbone='resnet34',
                comparison_basis='official_tusimple',status='VERIFIED',images=2782,id='independent',
                value_percent=95.6,reason='Independent checkpoint')
    rows=cells(list(csv.DictReader((ROOT/'results/reference/table1.csv').open())),[record])
    measured=[r for r in rows if r['status']=='MEASURED INDEPENDENT BASELINE']
    assert len(measured)==5 and all(r['local_equivalent_percent']==95.6 for r in measured)
    assert all(r['local_equivalent_percent']=='' for r in rows if r['metric']!='benign_acc')
