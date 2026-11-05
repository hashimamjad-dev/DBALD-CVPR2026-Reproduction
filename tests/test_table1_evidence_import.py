import csv
import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import table1_evidence as evidence
from evaluate_table1 import cells, ROOT
from render_table1 import export_page7


def bundle(tmp_path, monkeypatch, metric='loa_asr', dataset='TuSimple'):
    monkeypatch.setattr(evidence,'ROOT',tmp_path)
    count=2782 if dataset=='TuSimple' else 34680
    keys=[f'clips/{i}.jpg' for i in range(count)]
    annotations=[dict(raw_file=k,h_samples=[1,2,3,4],lanes=[[10,10,10,10],[30,30,30,30]]) for k in keys]
    official=tmp_path/('datasets/raw/tusimple_test/labels.jsonl' if dataset=='TuSimple' else 'datasets/raw/culane/list/test.txt')
    official.parent.mkdir(parents=True)
    official.write_text(''.join(json.dumps(r)+'\n' for r in annotations) if dataset=='TuSimple' else '\n'.join('/'+k for k in keys))
    monkeypatch.setattr(evidence,'official_culane_points',lambda key: [[(10,y) for y in [1,2,3,4]],[(30,y) for y in [1,2,3,4]]])
    predictions=[dict(r,run_time=10) for r in annotations]
    protocol='fixed_slot_disappearance_v1' if metric=='lda_asr' else 'tusimple_official_runtime_v1'
    if metric=='lda_asr':
        predictions=[dict(r,lanes=[[-2]*4,[30]*4]) for r in predictions]
    artifacts={}
    def save(name,content):
        p=tmp_path/name; p.write_text(content)
        artifacts[name.split('.')[0]]=dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    save('checkpoint.bin','test-only fixture, never a research model')
    save('config.json','{}')
    save('split.json',json.dumps(keys))
    save('predictions.jsonl',''.join(json.dumps(r)+'\n' for r in predictions))
    save('annotations.jsonl',''.join(json.dumps(r)+'\n' for r in annotations))
    record=dict(id='fixture',model='LaneATT',dataset=dataset,metric=metric,method='DBALD',
        experiment_id='fixture_series',protocol_family='declared_candidate_v1',protocol=protocol,artifacts=artifacts)
    provenance={k:record[k] for k in ('model','dataset','metric','method','experiment_id','protocol')}
    provenance.update(checkpoint_sha256=artifacts['checkpoint']['sha256'],config_sha256=artifacts['config']['sha256'],
        evaluation_input='clean' if metric.endswith('acc') else 'supplied_scenario',lane_slot_semantics='stable_correspondence')
    save('provenance.json',json.dumps(provenance))
    return record


@pytest.mark.parametrize('metric',['benign_acc','poisoned_acc','loa_asr'])
def test_official_static_predictions(metric,tmp_path,monkeypatch):
    record=bundle(tmp_path,monkeypatch,metric)
    result=evidence.import_external(record)
    assert result['value_percent']==100 and result['images']==2782
    assert result['comparison_basis']=='unconfirmed'


@pytest.mark.parametrize('dataset',['TuSimple','CULane'])
def test_candidate_disappearance(dataset,tmp_path,monkeypatch):
    result=evidence.import_external(bundle(tmp_path,monkeypatch,'lda_asr',dataset))
    assert result['value_percent']==50


@pytest.mark.parametrize('metric',['benign_acc','poisoned_acc','loa_asr','lra_asr'])
def test_culane_candidate_target_scoring(metric,tmp_path,monkeypatch):
    record=bundle(tmp_path,monkeypatch,metric,'CULane')
    record['protocol']='culane_annotation_vertices_linear_assignment_v1'
    path=tmp_path/record['artifacts']['provenance']['path']
    provenance=json.loads(path.read_text()); provenance['protocol']=record['protocol']
    path.write_text(json.dumps(provenance)); record['artifacts']['provenance']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    result=evidence.import_external(record)
    assert result['value_percent']==100 and result['comparison_basis']=='unconfirmed'


def test_cache_roundtrip_and_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(evidence,'ROOT',tmp_path)
    folder=tmp_path/'run'; folder.mkdir()
    (tmp_path/'checkpoint').write_bytes(b'fixture')
    digest=hashlib.sha256(b'fixture').hexdigest()
    (folder/'summary.json').write_text('[]')
    (folder/'model_provenance.json').write_text(json.dumps(dict(checkpoint='checkpoint')))
    (folder/'run_config.json').write_text(json.dumps(dict(checkpoint_sha256=digest,backbone='resnet34')))
    row=dict(model='LaneATT',dataset='TuSimple',images=2782,evidence='run',metric='official TuSimple ACC',
        value_percent=90,checkpoint_sha256=digest,summary_sha256=hashlib.sha256(b'[]').hexdigest())
    path=tmp_path/'cache.json'; path.write_text(json.dumps(dict(verified=[row])))
    spec=dict(path='cache.json',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert evidence.import_cache(spec)[0]['comparison_basis']=='official_tusimple'
    (folder/'summary.json').write_text('[1]')
    with pytest.raises(ValueError,match='summary drift'): evidence.import_cache(spec)


def test_candidate_receipt_rejects_wrong_protocol(tmp_path,monkeypatch):
    monkeypatch.setattr(evidence,'ROOT',tmp_path)
    path=tmp_path/'candidate.json'; path.write_text(json.dumps(dict(protocol={})))
    with pytest.raises(ValueError,match='protocol'):
        evidence.import_point_candidate(dict(path='candidate.json',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


@pytest.mark.parametrize('failure',['hash','coverage','runtime','provenance','claim','protocol','official_split'])
def test_invalid_external_evidence_rejected(failure,tmp_path,monkeypatch):
    record=bundle(tmp_path,monkeypatch)
    if failure=='claim': record['claimed_percent']=99
    elif failure=='protocol': record['protocol']='unsupported'
    elif failure=='official_split': (tmp_path/'datasets/raw/tusimple_test/labels.jsonl').write_text('')
    else:
        name={'hash':'checkpoint','coverage':'split','runtime':'predictions','provenance':'provenance'}[failure]
        p=tmp_path/record['artifacts'][name]['path']
        if failure=='hash': p.write_text('changed')
        elif failure=='coverage': p.write_text(json.dumps(['duplicate']*2782))
        elif failure=='runtime':
            rows=[json.loads(line) for line in p.read_text().splitlines()]; rows[0]['run_time']=-1
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        else:
            obj=json.loads(p.read_text()); obj['checkpoint_sha256']='incorrect'; p.write_text(json.dumps(obj))
        if failure!='hash': record['artifacts'][name]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError): evidence.import_external(record)


def test_averages_require_same_series_and_all_scenarios():
    def row(metric,value,experiment='a'):
        return dict(id=metric,model='LaneATT',dataset='TuSimple',method='DBALD',metric=metric,
            status='VERIFIED',value_percent=value,experiment_id=experiment,protocol_family='candidate')
    assert evidence.derive_averages([row('lda_asr',20)])==[]
    assert evidence.derive_averages([row('lda_asr',20),row('loa_asr',40,'b')])==[]
    assert evidence.derive_averages([row('lda_asr',20),row('loa_asr',40)])[0]['value_percent']==30
    with pytest.raises(ValueError): evidence.derive_averages([row('lda_asr',20),row('lda_asr',20)])


def test_cache_hash_is_required(tmp_path,monkeypatch):
    monkeypatch.setattr(evidence,'ROOT',tmp_path)
    (tmp_path/'receipt.json').write_text('{"verified":[]}')
    with pytest.raises(ValueError,match='hash'): evidence.import_cache(dict(path='receipt.json',sha256='incorrect'))


def test_page7_exports_original_columns_and_separate_sources(tmp_path):
    reference=list(csv.DictReader((ROOT/'results/reference/table1.csv').open()))
    export_page7(reference,cells(reference,[]),[],tmp_path)
    rows=list(csv.DictReader((tmp_path/'table1.csv').open()))
    assert len(rows)==40 and list(rows[0])[1:]==list(reference[0])
    assert rows[0]['source']=='published_reference' and rows[20]['source']=='independent_results'
    assert all(v=='NOT MEASURED' for k,v in rows[20].items() if k not in ('source','model','attack'))


def test_single_entry_point_exports_and_rejects_unsupported_record(tmp_path,monkeypatch):
    import evaluate_table1
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1,records=[],external_records=[
        dict(id='invalid',model='LaneATT',dataset='TuSimple',metric='unknown') ])))
    output=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['evaluate_table1','--manifest',str(manifest),'--output',str(output)])
    assert evaluate_table1.main()==1
    assert len(json.loads((output/'verification.json').read_text())['rejected'])==1
    assert len(json.loads((output/'inventory.json').read_text())['unresolved_cells'])==220
    assert len(list(csv.DictReader((output/'table1.csv').open())))==40


def test_culane_average_requires_three_scenarios():
    rows=[dict(id=m,model='RESA',dataset='CULane',method='DBALD',metric=m,status='VERIFIED',
        experiment_id='series',protocol_family='candidate',value_percent=v)
        for m,v in [('lda_asr',20),('lra_asr',30),('loa_asr',40)]]
    assert evidence.derive_averages(rows[:2])==[]
    assert evidence.derive_averages(rows)[0]['value_percent']==30
