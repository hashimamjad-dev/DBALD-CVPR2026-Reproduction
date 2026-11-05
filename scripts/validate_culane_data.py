"""Decode and hash every official test image and verify raw lane annotations."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from itertools import islice
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',help='New results directory; preserve existing integrity reports')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--retry-failures',help='Retry only failed rows from a preserved integrity JSONL receipt')
    args=parser.parse_args()
    if not 1<=args.workers<=8: raise ValueError('Use one to eight validation workers')
    out=Path(args.output).resolve() if args.output else Path('results').resolve()
    if args.output:
        if not out.is_relative_to(Path('results').resolve()) or out.exists():
            raise ValueError('Use a new results directory')
        out.mkdir(parents=True)
    root=Path('datasets/raw/culane'); listing=root/'list/test.txt'
    keys=[line.split()[0].lstrip('/') for line in listing.read_text().splitlines() if line.strip()]
    if len(keys)!=len(set(keys)) or len(keys)!=34680: raise ValueError('Unexpected CULane test split')
    retry_from=None
    if args.retry_failures:
        if not args.output: raise ValueError('Retry requires a new output directory')
        source=Path(args.retry_failures)
        rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(rows)!=34680 or {r['raw_file'] for r in rows}!=set(keys):
            raise ValueError('Retry source must cover the full official split exactly once')
        retry_keys=[r['raw_file'] for r in rows if r['status']!='verified']
        if not retry_keys: raise ValueError('No failed rows to retry')
        retry_from=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        listing=out/'retry_list.txt';listing.write_text('\n'.join(retry_keys)+'\n')
    expected=len(retry_keys) if retry_from else len(keys)
    samples=iter(culane(root,listing))
    failures=0
    processed=0
    retried=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool,(out/'culane_integrity.jsonl').open('w') as log:
        # Executor.map eagerly queues its input. Bound annotation objects and
        # pending futures instead of retaining the entire 34,680-image split.
        while batch:=list(islice(samples,64)):
            for r in pool.map(check,batch):
                processed+=1
                failures+=r['status']!='verified'; log.write(json.dumps(r)+'\n')
                if retry_from: retried.append(r)
                if processed%1000==0: log.flush(); print(f'{processed}/{expected}, failures={failures}',flush=True)
    if processed!=expected: raise ValueError('Incomplete dataset reader coverage')
    report=dict(dataset='CULane',split='test',expected=expected,verified=processed-failures,failures=failures,
        full_test=not bool(retry_from),retry_from=retry_from,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        list_sha256=hashlib.sha256(listing.read_bytes()).hexdigest())
    (out/'culane_integrity_manifest.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)
    if retry_from:
        combined={r['raw_file']:r for r in rows}
        combined.update({r['raw_file']:r for r in retried})
        digest=hashlib.sha256()
        for key in keys:
            digest.update(json.dumps(combined[key],sort_keys=True).encode()+b'\n')
        verified=sum(r['status']=='verified' for r in combined.values())
        reconciled=dict(dataset='CULane',split='test',expected=len(keys),verified=verified,
            failures=len(keys)-verified,full_test=True,retried=len(retried),
            original_receipt=retry_from,retry_receipt_sha256=hashlib.sha256((out/'culane_integrity.jsonl').read_bytes()).hexdigest(),
            combined_records_sha256=digest.hexdigest(),validator_sha256=report['validator_sha256'],
            note='Original full receipt plus explicit retry replacements; original errors are retained, not overwritten')
        (out/'culane_reconciled_integrity_manifest.json').write_text(json.dumps(reconciled,indent=2)+'\n')
        if reconciled['failures']: raise SystemExit(1)
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
