import csv
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import assemble_table1 as A
import plan_table1_gpu_runs as P

COMPARISON=ROOT/'results/table1_protocol_scnn_20261106/clean_recheck/table1_comparison.csv'

def receipt(metric,value,**kw):
    r=dict(dataset='TuSimple',model='LaneATT',method='DBALD',metric=metric,value_percent=value,images=2782,
           evidence='synthetic-test-only',sha256='0'*64,independently_rescored=True); r.update(kw); return r

def test_validate_rejects_partial_unscored_or_out_of_range():
    assert A.validate_receipt(receipt('lda_asr',50)) is None
    assert 'full-test' in A.validate_receipt(receipt('lda_asr',50,images=100))
    assert 're-scored' in A.validate_receipt(receipt('lda_asr',50,independently_rescored=False))
    assert 'outside' in A.validate_receipt(receipt('lda_asr',150))
    assert 'LRA' in A.validate_receipt(receipt('lra_asr',5))
    assert 'missing' in A.validate_receipt({k:v for k,v in receipt('lda_asr',5).items() if k!='sha256'})

def test_average_derived_only_when_every_scenario_validated():
    cells=list(csv.DictReader(COMPARISON.open(encoding='utf-8')))
    def row(cs): return {c['metric']:c for c in cs if (c['dataset'],c['model'],c['method'])==('TuSimple','LaneATT','DBALD')}
    partial=row(A.overlay(cells,[receipt('lda_asr',80.)]))
    assert partial['avg_asr']['local_equivalent_percent']=='' and partial['lda_asr']['evidence_status']=='VERIFIED'
    full=row(A.overlay(cells,[receipt('lda_asr',80.),receipt('loa_asr',90.)]))
    assert float(full['avg_asr']['local_equivalent_percent'])==85.0
    assert full['avg_asr']['status'].startswith('DERIVED')

def test_published_values_never_become_local_measurements():
    cells=list(csv.DictReader(COMPARISON.open(encoding='utf-8')))
    after=A.overlay(cells,[])
    for a,b in zip(cells,after): assert a['local_equivalent_percent']==b['local_equivalent_percent']
    assert sum(c['evidence_status']=='VERIFIED' for c in after)==15

def test_published_average_audit_reports_without_altering():
    cells=list(csv.DictReader(COMPARISON.open(encoding='utf-8')))
    findings=A.audit_published_averages(cells)
    assert len(findings)==40
    bad=[f for f in findings if not f['consistent']]
    assert [(f['model'],f['method'],f['dataset']) for f in bad]==[('LaneATT','DBALD','CULane')]

def test_cli_refuses_non_empty_output_and_writes_files(tmp_path):
    A.main(['--output',str(tmp_path/'new')])
    assert (tmp_path/'new/table1_improved.md').exists()
    with pytest.raises(SystemExit): A.main(['--output',str(tmp_path/'new')])

def test_plan_has_100_scenarios_and_no_gpu_commands():
    plan=P.build()
    assert plan['total_scenario_trainings']==100
    assert all(r['approval_required_before_gpu'] for r in plan['runs'])
    assert all('gpu' not in (r['cpu_step_poisoned_set'] or '').lower() for r in plan['runs'])
