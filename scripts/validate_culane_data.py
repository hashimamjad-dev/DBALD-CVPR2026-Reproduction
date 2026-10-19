"""Decode and hash every official test image and verify raw lane annotations."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lane_repro.data import culane

def check(sample):
    try:
        image=sample.read()
        with sample.image.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        anno=sample.image.with_suffix('.lines.txt')
        return dict(raw_file=sample.key,image_sha256=sha,annotation_sha256=hashlib.sha256(anno.read_bytes()).hexdigest(),
            shape=list(image.shape),lanes=len(sample.lanes),status='verified')
    except Exception as exc: return dict(raw_file=sample.key,status='failed',error=str(exc))

def main():
    root=Path('datasets/raw/culane'); samples=list(culane(root,root/'list/test.txt'))
    if len(samples)!=34680: raise ValueError('Unexpected CULane test count')
    failures=0
    with ThreadPoolExecutor(max_workers=4) as pool,Path('results/culane_integrity.jsonl').open('w') as log:
        for i,r in enumerate(pool.map(check,samples),1):
            failures+=r['status']!='verified'; log.write(json.dumps(r)+'\n')
            if i%1000==0: log.flush(); print(f'{i}/34680, failures={failures}',flush=True)
    report=dict(dataset='CULane',split='test',expected=34680,verified=34680-failures,failures=failures,
        list_sha256=hashlib.sha256((root/'list/test.txt').read_bytes()).hexdigest())
    Path('results/culane_integrity_manifest.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
