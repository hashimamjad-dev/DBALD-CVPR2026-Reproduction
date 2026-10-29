import csv
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import render_table1


def test_cached_render_exports_all_cells_without_scorer(tmp_path,monkeypatch):
    source=tmp_path/'cache.json'; source.write_text(json.dumps({'verified':[]}))
    output=tmp_path/'rendered'
    monkeypatch.setattr(sys,'argv',['render_table1','--output',str(output),'--evidence',str(source)])
    monkeypatch.setattr(render_table1,'official_tusimple',lambda: pytest.fail('Must not re-score cached evidence'))
    render_table1.main()
    rows=list(csv.DictReader((output/'table1_status.csv').open()))
    assert len(rows)==220 and all(r['status']=='NOT MEASURED' for r in rows)
    assert (output/'Table1_Comparison.html').is_file()


def test_subset_cannot_enter_full_culane_evidence(tmp_path,monkeypatch):
    source=tmp_path/'cache.json'; source.write_text(json.dumps({'verified':[]}))
    extra=tmp_path/'extra.json'; extra.write_text(json.dumps({'verified':[dict(status='VERIFIED',dataset='CULane',images=16)]}))
    monkeypatch.setattr(sys,'argv',['render_table1','--output',str(tmp_path/'rendered'),'--evidence',str(source),'--additional-evidence',str(extra)])
    with pytest.raises(ValueError,match='full CULane'): render_table1.main()


def test_candidate_cannot_claim_paper_equivalence(tmp_path,monkeypatch):
    source=tmp_path/'cache.json'; source.write_text(json.dumps({'verified':[]}))
    folder=tmp_path/'run'; folder.mkdir()
    (folder/'run_config.json').write_text(json.dumps({'backbone':'resnet34'}))
    (folder/'point_acc_candidate.json').write_text(json.dumps(dict(images=34680,full_test=True,total=100,correct=50,
        candidate_acc_percent=50,protocol={'table1_comparable':True})))
    extra=tmp_path/'extra.json'
    extra.write_text(json.dumps({'verified':[dict(status='VERIFIED',dataset='CULane',images=34680,
        evidence=str(folder),metric_definition='Official F1')]}))
    monkeypatch.setattr(sys,'argv',['render_table1','--output',str(tmp_path/'rendered'),'--evidence',str(source),'--additional-evidence',str(extra)])
    with pytest.raises(ValueError,match='equivalence claim'): render_table1.main()
