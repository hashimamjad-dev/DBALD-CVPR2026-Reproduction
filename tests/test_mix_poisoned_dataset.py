"""CLI-wiring test for scripts/mix_poisoned_dataset.py (placeholder path; the poisoning itself is
already unit-tested in test_poison_pipeline.py)."""
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/mix_poisoned_dataset.py'
WIDTH,HEIGHT=1280,720
H_SAMPLES=[200,300,400,500]

def make_split(tmp_path,n):
    root=tmp_path/'clean'; (root/'clips').mkdir(parents=True)
    rows=[]
    for i in range(n):
        image=np.full((HEIGHT,WIDTH,3),200,np.uint8)
        cv2.imwrite(str(root/f'clips/img{i}.jpg'),image)
        lane=[300+30*j for j in range(len(H_SAMPLES))]
        rows.append(dict(raw_file=f'clips/img{i}.jpg',h_samples=H_SAMPLES,lanes=[lane]))
    labels=root/'labels.jsonl'
    labels.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return root,labels

def run(args):
    return subprocess.run([sys.executable,str(SCRIPT)]+args,cwd=ROOT,capture_output=True,text=True,timeout=120)

def test_mixes_at_the_requested_rate_deterministically(tmp_path):
    root,labels=make_split(tmp_path,20)
    output=tmp_path/'mixed'
    result=run(['--root',str(root),'--labels',str(labels),'--output',str(output),
        '--attack','LOA','--rate','0.25','--trigger-size','8','--stride','4'])
    assert result.returncode==0,result.stderr
    rows=[json.loads(l) for l in (output/'labels.jsonl').read_text().splitlines()]
    assert len(rows)==20
    poisoned=[r for r in rows if r['poisoned']]
    assert len(poisoned)==5 and all(r['attack']=='LOA' for r in poisoned)
    assert all(r['attack'] is None for r in rows if not r['poisoned'])
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['poisoned_count']==5 and manifest['sample_count']==20

def test_same_salt_selects_the_same_subset_across_runs(tmp_path):
    root,labels=make_split(tmp_path,20)
    first=tmp_path/'a'; second=tmp_path/'b'
    for output in (first,second):
        r=run(['--root',str(root),'--labels',str(labels),'--output',str(output),
            '--attack','LOA','--rate','0.3','--salt','fixed-salt','--trigger-size','8','--stride','4'])
        assert r.returncode==0,r.stderr
    poisoned_a={json.loads(l)['raw_file'] for l in (first/'labels.jsonl').read_text().splitlines() if json.loads(l)['poisoned']}
    poisoned_b={json.loads(l)['raw_file'] for l in (second/'labels.jsonl').read_text().splitlines() if json.loads(l)['poisoned']}
    assert poisoned_a==poisoned_b and len(poisoned_a)==6

def test_every_image_is_present_in_output(tmp_path):
    root,labels=make_split(tmp_path,10)
    output=tmp_path/'mixed'
    result=run(['--root',str(root),'--labels',str(labels),'--output',str(output),
        '--attack','LDA','--rate','0.2','--trigger-size','8','--stride','4'])
    assert result.returncode==0,result.stderr
    for i in range(10): assert (output/f'clips/img{i}.jpg').exists()

def test_rejects_output_overlapping_source(tmp_path):
    root,labels=make_split(tmp_path,4)
    result=run(['--root',str(root),'--labels',str(labels),'--output',str(root),'--attack','LOA'])
    assert result.returncode!=0

def test_rejects_invalid_rate(tmp_path):
    root,labels=make_split(tmp_path,4)
    result=run(['--root',str(root),'--labels',str(labels),'--output',str(tmp_path/'mixed'),
        '--attack','LOA','--rate','0'])
    assert result.returncode!=0
