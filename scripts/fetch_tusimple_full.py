"""Resume the 2,782 labelled test frames; omit unscored video context frames."""
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
DEST=ROOT/'datasets/raw/tusimple_test'

def valid(data):
    im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    return im is not None and im.shape==(720,1280,3)

def download(record):
    name=record['raw_file']; path=(DEST/name).resolve()
    if not path.is_relative_to(DEST.resolve()): raise ValueError(name)
    path.parent.mkdir(parents=True,exist_ok=True)
    old=ROOT/'datasets/raw/tusimple_subset'/name
    if not path.exists() and old.exists(): shutil.copy2(old,path)
    for attempt in range(4):
        try:
            if shutil.disk_usage(DEST).free<2*1024**3: raise RuntimeError('Less than 2 GiB disk free')
            data=path.read_bytes() if path.exists() else fetch('TUSimple/test_set/'+name)
            if not valid(data): raise ValueError('Invalid image dimensions or decoding')
            if not path.exists():
                temporary=path.with_suffix('.part'); temporary.write_bytes(data); temporary.replace(path)
            return dict(raw_file=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),status='verified')
        except Exception as exc:
            if attempt==3: return dict(raw_file=name,status='failed',error=str(exc))
            time.sleep(2**attempt)

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    label=ROOT/'datasets/raw/tusimple_subset/all_test_label.json'
    records=[json.loads(line) for line in label.read_text().splitlines() if line.strip()]
    assert len(records)==2782 and len({r['raw_file'] for r in records})==2782
    shutil.copy2(label,DEST/'labels.jsonl')
    report=ROOT/'results/tusimple_full_integrity.jsonl'
    failures=0
    with report.open('w') as log,ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(download,r) for r in records]
        for n,job in enumerate(as_completed(jobs),1):
            row=job.result(); failures+=row['status']!='verified'
            log.write(json.dumps(row)+'\n'); log.flush()
            if n%50==0 or row['status']!='verified': print(f'{n}/2782, failures={failures}',flush=True)
    manifest=dict(source='https://github.com/TuSimple/tusimple-benchmark/issues/3',mirror=BASE,
        split='test',expected=2782,verified=2782-failures,failures=failures,
        labels_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
        note='Only labelled final frames downloaded; context frames are not inputs to these single-frame models')
    (ROOT/'results/tusimple_full_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
