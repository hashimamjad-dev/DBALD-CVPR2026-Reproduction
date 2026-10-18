import json
from pathlib import Path
import numpy as np
import pytest
from lane_repro.prototype import TRANSFORMS,transform,window_statistics
from lane_repro.culane_official import prediction_text,score
from lane_repro.evaluation import official_tusimple

@pytest.mark.parametrize('name',TRANSFORMS)
def test_benign_transforms_repeatable_and_preserve_input(name):
    image=np.random.default_rng(4).integers(0,256,(72,128,3),dtype=np.uint8)
    original=image.copy(); a=transform(image,name); b=transform(image,name)
    assert a.shape==image.shape and a.dtype==np.uint8
    np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(image,original)

def test_window_statistics_known_values():
    rows=window_statistics(np.arange(16).reshape(4,4),[(0,0,2,2),(2,2,2,2)])
    assert rows[0]['mean']==2.5 and rows[1]['total']==50
    with pytest.raises(ValueError): window_statistics(np.zeros((4,4)),[(3,3,2,2)])

def test_culane_export_native_order_and_clipping():
    result=prediction_text([lambda ys:np.array([-.1,.5,1.1,.75])],[100,200,300,400])
    assert result=='1230.00000 400.00000 820.00000 200.00000'

def test_official_culane_known_counts(tmp_path):
    root=Path(__file__).resolve().parents[1]
    if not (root/'tools/culane_evaluator/evaluate.exe').exists(): pytest.skip('Build official evaluator first')
    anno=tmp_path/'anno'; pred=tmp_path/'pred'; anno.mkdir(); pred.mkdir()
    # Perfect curved lane, missed lane, spurious lane, all-negative image.
    curve='100 500 110 400 140 300 200 200\n'
    for name,gt,det in [('a',curve,curve),('b',curve,''),('c','',curve),('d','','')]:
        (anno/f'{name}.lines.txt').write_text(gt); (pred/f'{name}.lines.txt').write_text(det)
    listing=tmp_path/'list.txt'; listing.write_text('/a.jpg\n/b.jpg\n/c.jpg\n/d.jpg\n')
    result=score(pred,anno,listing,tmp_path/'official.txt')
    assert (result['tp'],result['fp'],result['fn'])==(1,1,1)
    assert result['f1_percent']==50.
    (pred/'a.lines.txt').unlink()
    with pytest.raises(FileNotFoundError): score(pred,anno,listing,tmp_path/'bad.txt')

def test_official_submission_aggregation(tmp_path):
    evaluator=official_tusimple(); ys=[100,200,300,400]
    gt=[dict(raw_file=f'{i}.jpg',h_samples=ys,lanes=[[100]*4]) for i in range(2)]
    pred=[dict(raw_file=f'{i}.jpg',lanes=[[100]*4],run_time=10 if i==0 else 201) for i in range(2)]
    labels=tmp_path/'labels.json'; submit=tmp_path/'pred.json'
    labels.write_text(''.join(json.dumps(r)+'\n' for r in gt)); submit.write_text(''.join(json.dumps(r)+'\n' for r in pred))
    report=json.loads(evaluator.bench_one_submit(str(submit),str(labels)))
    assert report[0]['value']==.5 and report[2]['value']==.5
