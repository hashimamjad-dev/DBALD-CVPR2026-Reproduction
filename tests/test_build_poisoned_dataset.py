"""CLI-wiring test for scripts/build_poisoned_dataset.py (placeholder/no-checkpoint path only; the
library functions it calls -- poison_sample, write_poisoned_sample, laneatt_attention_map -- have
their own direct unit tests in test_poison_pipeline.py and test_laneatt_heatmap.py)."""
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/build_poisoned_dataset.py'
WIDTH,HEIGHT=1280,720  # lane_repro.data.tusimple() hardcodes this native TuSimple resolution
H_SAMPLES=[200,300,400,500]

def make_split(tmp_path):
    root=tmp_path/'clean'; (root/'clips').mkdir(parents=True)
    image=np.full((HEIGHT,WIDTH,3),200,np.uint8)
    cv2.imwrite(str(root/'clips/img.jpg'),image)
    lane=[300+30*i for i in range(len(H_SAMPLES))]
    labels=root/'labels.jsonl'
    labels.write_text(json.dumps(dict(raw_file='clips/img.jpg',h_samples=H_SAMPLES,lanes=[lane]))+'\n')
    return root,labels

def run(args):
    return subprocess.run([sys.executable,str(SCRIPT)]+args,cwd=ROOT,capture_output=True,text=True,timeout=60)

def test_builds_poisoned_split_with_placeholder_attention(tmp_path):
    root,labels=make_split(tmp_path)
    output=tmp_path/'poisoned'
    result=run(['--root',str(root),'--labels',str(labels),'--output',str(output),
        '--attacks','LDA','LOA','--trigger-size','8','--stride','4'])
    assert result.returncode==0,result.stderr
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['attention_source']=='uniform placeholder (poison_pipeline default)'
    assert manifest['sample_count']==1 and {r['attack'] for r in manifest['records']}=={'LDA','LOA'}
    lda_rows=[json.loads(l) for l in (output/'lda/labels.jsonl').read_text().splitlines()]
    assert lda_rows[0]['lanes'][0]==[-2.0]*len(H_SAMPLES)
    assert (output/'lda/clips/img.jpg').exists() and (output/'loa/clips/img.jpg').exists()

def test_rejects_output_overlapping_source(tmp_path):
    root,labels=make_split(tmp_path)
    result=run(['--root',str(root),'--labels',str(labels),'--output',str(root),'--attacks','LDA'])
    assert result.returncode!=0
