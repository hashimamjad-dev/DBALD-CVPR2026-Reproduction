"""Resume the full TuSimple train split (label_data_0313/0531/0601.json, ~3,626 labelled frames) --
needed for a real poisoning/training run (scripts/build_poisoned_dataset.py +
scripts/sanity_poisoned_training.py currently only use the 16-image smoke subset and the test
split). Uses the same verified Hugging Face fallback as fetch_tusimple_subset.py (the Kaggle mirror
this project was built against returns HTTP 404 as of 2026-10-23; see docs/sources.md)."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from fetch_tusimple_subset import fetch, BASE

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'datasets/raw/tusimple_train'
LABEL_FILES=['label_data_0313.json','label_data_0531.json','label_data_0601.json']

def valid(data):
    im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    return im is not None and im.shape==(720,1280,3)

def download(record):
    name=record['raw_file']; path=(DEST/name).resolve()
    if not path.is_relative_to(DEST.resolve()): raise ValueError(name)
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        try:
            if shutil.disk_usage(DEST).free<2*1024**3: raise RuntimeError('Less than 2 GiB disk free')
            data=path.read_bytes() if path.exists() else fetch('TUSimple/train_set/'+name)
            if not valid(data): raise ValueError('Invalid image dimensions or decoding')
            if not path.exists():
                temporary=path.with_suffix('.part'); temporary.write_bytes(data); temporary.replace(path)
            return dict(raw_file=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),status='verified')
        except Exception as exc:
            if attempt==3: return dict(raw_file=name,status='failed',error=str(exc))
            time.sleep(2**attempt)

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    records=[]; seen=set()
    for filename in LABEL_FILES:
        label=DEST/filename
        if not label.exists(): label.write_bytes(fetch('TUSimple/train_set/'+filename))
        for line in label.read_text().splitlines():
            if not line.strip(): continue
            r=json.loads(line)
            if r['raw_file'] in seen: continue
            seen.add(r['raw_file']); records.append(r)
    (DEST/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    report=ROOT/'results/tusimple_train_integrity.jsonl'
    failures=0
    with report.open('w') as log,ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(download,r) for r in records]
        for n,job in enumerate(as_completed(jobs),1):
            row=job.result(); failures+=row['status']!='verified'
            log.write(json.dumps(row)+'\n'); log.flush()
            if n%50==0 or row['status']!='verified': print(f'{n}/{len(records)}, failures={failures}',flush=True)
    (ROOT/'results/tusimple_train_summary.json').write_text(json.dumps(dict(
        source='https://github.com/TuSimple/tusimple-benchmark/issues/3',
        mirror_status='Kaggle mirror returned HTTP 404 on 2026-10-23; used the Hugging Face fallback',
        fallback_mirror='https://huggingface.co/datasets/dhbloo/TuSimple',
        label_files=LABEL_FILES,expected=len(records),verified=len(records)-failures,failures=failures,
        labels_sha256=hashlib.sha256((DEST/'labels.jsonl').read_bytes()).hexdigest()),indent=2))
    print('Done:',len(records)-failures,'/',len(records),'verified')

if __name__=='__main__': main()
