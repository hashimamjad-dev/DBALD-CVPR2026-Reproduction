import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from lane_repro.baseline_attacks import METHODS,baseline_poison_sample,TRIGGER_PIXELS,PATCH
from lane_repro.data import Sample
from lane_repro.tusimple_split import filter_samples,load_split

ROOT=Path(__file__).resolve().parents[1]
W,H=1280,720
HS=[300,400,500,600,700]

def make_sample(tmp_path,key='clips/a/1/20.jpg'):
    path=tmp_path/'clean'/key; path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),np.full((H,W,3),150,np.uint8))
    lanes=[[400+20*i for i in range(5)],[700+10*i for i in range(5)]]
    return Sample(key,path,lanes,W,H,HS)

@pytest.mark.parametrize('method',METHODS)
def test_every_baseline_changes_pixels_and_transforms_labels(tmp_path,method):
    s=make_sample(tmp_path); r=baseline_poison_sample(s,method,'LOA')
    diff=np.any(r['poisoned_image']!=s.read(),axis=2)
    assert diff.sum()>0
    assert diff.sum()<=max(TRIGGER_PIXELS,PATCH*PATCH)
    assert r['poisoned_lanes'][0][0]==s.lanes[0][0]+60
    assert r['recipe_assumptions'] and r['method']==method

def test_patch_methods_touch_bottom_right_only(tmp_path):
    s=make_sample(tmp_path)
    for m in ('badnets','blended'):
        diff=np.any(baseline_poison_sample(s,m,'LDA')['poisoned_image']!=s.read(),axis=2)
        ys,xs=np.nonzero(diff); assert ys.min()>=H-PATCH and xs.min()>=W-PATCH

def test_pixel_methods_use_exactly_900_pixels_and_are_deterministic(tmp_path):
    s=make_sample(tmp_path)
    for m in ('ld_attack','badlane'):
        a=baseline_poison_sample(s,m,'LDA'); b=baseline_poison_sample(s,m,'LDA')
        assert np.array_equal(a['poisoned_image'],b['poisoned_image'])
        assert np.any(a['poisoned_image']!=s.read(),axis=2).sum()==TRIGGER_PIXELS

def test_lda_erases_lanes(tmp_path):
    r=baseline_poison_sample(make_sample(tmp_path),'badnets','LDA')
    assert r['poisoned_existence']==[False,False]

def test_unknown_method_rejected(tmp_path):
    with pytest.raises(ValueError): baseline_poison_sample(make_sample(tmp_path),'nope','LOA')

def split_file(tmp_path,train,val):
    p=tmp_path/'split.json'
    p.write_text(json.dumps(dict(train=train,val=val,counts=dict(train=len(train),val=len(val)))))
    return p

def test_filter_samples_selects_part_and_rejects_unknown(tmp_path):
    a,b,c=(make_sample(tmp_path,f'clips/a/{i}/20.jpg') for i in range(3))
    sp=split_file(tmp_path,[a.key,b.key],[c.key])
    assert [s.key for s in filter_samples([a,b,c],'train',sp)]==[a.key,b.key]
    assert [s.key for s in filter_samples([a,b,c],'val',sp)]==[c.key]
    with pytest.raises(ValueError): filter_samples([make_sample(tmp_path,'clips/z/9/20.jpg')],'train',sp)

def test_load_split_rejects_overlap(tmp_path):
    with pytest.raises(ValueError): load_split(split_file(tmp_path,['x'],['x']))

def test_real_pinned_split_is_consistent():
    s=load_split(ROOT/'results/table1_split_20261111/tusimple_split.json')
    assert s['counts']=={'train':3263,'val':363}

def test_mix_cli_with_baseline_and_split(tmp_path):
    keys=[f'clips/a/{i}/20.jpg' for i in range(10)]
    samples=[make_sample(tmp_path,k) for k in keys]
    labels=tmp_path/'clean'/'labels.jsonl'
    labels.write_text(''.join(json.dumps(dict(raw_file=s.key,h_samples=HS,lanes=s.lanes))+'\n' for s in samples))
    sp=split_file(tmp_path,keys[:8],keys[8:])
    out=tmp_path/'mixed'
    r=subprocess.run([sys.executable,str(ROOT/'scripts/mix_poisoned_dataset.py'),'--root',str(tmp_path/'clean'),'--labels',str(labels),
        '--output',str(out),'--attack','LOA','--method','badnets','--rate','0.25','--split-file',str(sp),'--split-part','train'],
        cwd=ROOT,capture_output=True,text=True,timeout=120)
    assert r.returncode==0,r.stderr
    rows=[json.loads(l) for l in (out/'labels.jsonl').read_text().splitlines()]
    assert {x['raw_file'] for x in rows}==set(keys[:8])
    assert sum(x['poisoned'] for x in rows)==2
    assert json.loads((out/'manifest.json').read_text())['method']=='badnets'
