import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_clean_table1 import clean_manifest, completion_rows
from evaluate_table1 import ROOT, cells
from render_table1 import export_page7


def test_clean_runner_rejects_non_clean_and_cached_inputs(tmp_path):
    path=tmp_path/'manifest.json'
    for payload in [dict(schema_version=1,records=[dict(metric='poisoned_acc')]),
                    dict(schema_version=1,records=[dict(metric='benign_acc')],verified_caches=[{}])]:
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError): clean_manifest(path)


def test_completion_requires_full_split_and_does_not_call_f1_table1_acc():
    record=dict(model='LaneATT',dataset='CULane',metric='benign_acc',status='VERIFIED',
                official_metrics={'f1_percent':75},images=34680,backbone='resnet34',
                value_percent=75,evidence='fixture',checkpoint_sha256='fixture')
    matrix=completion_rows([record])
    row=next(r for r in matrix if r['model']=='LaneATT' and r['dataset']=='CULane')
    assert row['clean_evidence_status']=='VERIFIED' and row['table1_status']=='NOT COMPARABLE'
    assert 'not Table 1 ACC' in row['metric']
    assert sum(r['clean_evidence_status']=='MISSING' for r in matrix)==7
    with pytest.raises(ValueError): completion_rows([dict(record,images=16)])


def test_comparison_never_uses_reference_or_f1_as_local_accuracy(tmp_path):
    reference=list(csv.DictReader((ROOT/'results/reference/table1.csv').open()))
    clean=dict(model='LaneATT',dataset='TuSimple',metric='benign_acc',status='VERIFIED',
               comparison_basis='official_tusimple',images=2782,backbone='resnet34',
               id='zero_fixture',value_percent=0.,reason='synthetic unit test',evidence='fixture/zero')
    f1=dict(clean,dataset='CULane',comparison_basis='unconfirmed',id='f1_fixture',value_percent=75.)
    export_page7(reference,cells(reference,[clean,f1]),[clean,f1],tmp_path)
    rows=list(csv.DictReader((tmp_path/'table1_comparison.csv').open()))
    assert len(rows)==220
    for row in rows:
        if row['model']=='LaneATT' and row['dataset']=='TuSimple' and row['metric']=='benign_acc':
            assert float(row['local_equivalent_percent'])==0
            assert float(row['difference_percentage_points'])==-float(row['published_percent'])
            assert row['evidence_artifacts']=='fixture/zero'
        else:
            assert row['local_equivalent_percent']==row['difference_percentage_points']==''
