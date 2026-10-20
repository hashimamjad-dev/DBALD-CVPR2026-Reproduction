"""Freeze disjoint road-restoration training/validation/evaluation manifests."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fetch_tusimple_subset import fetch,BASE
from lane_repro.data import culane

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/distillation'

def select(rows,n,salt):
    return sorted(rows,key=lambda r:hashlib.sha256((salt+r['raw_file']).encode()).hexdigest())[:n]

def main():
    OUT.mkdir(exist_ok=True)
    destination=ROOT/'datasets/raw/tusimple_distillation_train'; destination.mkdir(exist_ok=True)
    selected=[]
    for filename,split,count in [('label_data_0313.json','train',64),('label_data_0531.json','validation',16)]:
        label=destination/filename
        if not label.exists(): label.write_bytes(fetch('TUSimple/train_set/'+filename))
        rows=[json.loads(s) for s in label.read_text().splitlines() if s.strip()]
        for row in select(rows,count,'road-distillation-v1'):
            selected.append(dict(row,dataset='TuSimple',partition=split,root=str(destination.relative_to(ROOT))))
    def download(row):
        path=destination/row['raw_file']; path.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(3):
            try:
                if not path.exists():
                    data=fetch('TUSimple/train_set/'+row['raw_file'])
                    temporary=path.with_suffix('.part'); temporary.write_bytes(data); temporary.replace(path)
                image=cv2.imread(str(path))
                if image is None or image.shape!=(720,1280,3): raise ValueError('Invalid training image')
                row['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                return row
            except Exception:
                if attempt==2: raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=4) as pool:
        selected=list(pool.map(download,selected))
    tests=[json.loads(s) for s in (ROOT/'datasets/raw/tusimple_test/labels.jsonl').read_text().splitlines()]
    old={json.loads(s)['raw_file'] for s in (ROOT/'datasets/raw/tusimple_subset/labels.jsonl').read_text().splitlines()}
    for row in select([r for r in tests if r['raw_file'] not in old],8,'road-distillation-heldout-v1'):
        selected.append(dict(row,dataset='TuSimple',partition='evaluation',root='datasets/raw/tusimple_test'))
    samples=list(culane(ROOT/'datasets/raw/culane',ROOT/'datasets/culane_smoke16.txt'))
    for s in sorted(samples,key=lambda s:hashlib.sha256(('road-distillation-heldout-v1'+s.key).encode()).hexdigest())[:8]:
        selected.append(dict(raw_file=s.key,lanes=s.lanes,dataset='CULane',partition='evaluation',root='datasets/raw/culane',
            limitation='Available driver_37 subset; not representative of full CULane'))
    train_clips={r['raw_file'].rsplit('/',1)[0] for r in selected if r['partition']=='train'}
    val_clips={r['raw_file'].rsplit('/',1)[0] for r in selected if r['partition']=='validation'}
    all_test_clips={r['raw_file'].rsplit('/',1)[0] for r in tests}
    assert not (train_clips & val_clips or train_clips & all_test_clips or val_clips & all_test_clips)
    for i,row in enumerate(selected):
        row['id']='scene_%03d'%i
        path=ROOT/row['root']/row['raw_file']
        row['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(protocol='road-restoration-distillation-v1',selection='SHA256 ordering fixed before model execution',
        teacher_training_source='TuSimple official train (0313); validation (0531)',
        test_leakage_check='No training/validation clip overlaps any official TuSimple test clip',
        test_use='Held-out prototype editing evaluation only; all clean benchmark labels unchanged',
        training_seeds=[0],evaluation_seeds=[100,101],samples=selected,source=BASE)
    path=OUT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest: raise ValueError('Existing manifest differs; preserve it and use a new protocol')
    path.write_text(json.dumps(manifest,indent=2)); print({p:sum(r['partition']==p for r in selected) for p in ('train','validation','evaluation')})

if __name__=='__main__': main()
