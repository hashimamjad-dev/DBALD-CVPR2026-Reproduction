"""Read-only CULane test-split readiness check; write only audit reports."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'datasets/raw/culane'
OUT = ROOT/'results/phase1_audit'


def check(key):
    image = (DATA/key.lstrip('/')).resolve()
    if not image.is_relative_to(DATA.resolve()):
        raise ValueError('Dataset path escapes root')
    annotation = image.with_suffix('.lines.txt')
    issues = []
    if not image.is_file():
        issues.append('missing_image')
    else:
        decoded = cv2.imread(str(image))
        if decoded is None or decoded.shape != (590,1640,3):
            issues.append('invalid_image')
        # OpenCV may return a plausible image while warning about truncated JPEGs.
        with image.open('rb') as stream:
            stream.seek(-2,2)
            if stream.read(2) != b'\xff\xd9': issues.append('missing_jpeg_end_marker')
    if not annotation.is_file():
        issues.append('missing_annotation')
    else:
        try:
            for line in annotation.read_text().splitlines():
                coords = np.asarray([float(v) for v in line.split()])
                if coords.size < 2 or coords.size % 2 or not np.isfinite(coords).all():
                    issues.append('invalid_annotation'); break
        except (ValueError,UnicodeError):
            issues.append('invalid_annotation')
    return dict(raw_file=key,group=key.lstrip('/').split('/')[0],issues=issues,ready=not issues)


def main():
    listing = DATA/'list/test.txt'
    keys = [line.split()[0] for line in listing.read_text().splitlines() if line.strip()]
    assert len(keys) == len(set(keys)) == 34680
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(check,keys),1):
            records.append(result)
            if i%5000 == 0: print(f'Checked {i}/34680',flush=True)
    counts=Counter(issue for r in records for issue in r['issues'])
    report=dict(expected=34680,ready=sum(r['ready'] for r in records),issues=dict(counts),
                groups={g:dict(listed=sum(r['group']==g for r in records),ready=sum(r['group']==g and r['ready'] for r in records)) for g in sorted({r['group'] for r in records})},
                test_list_sha256=hashlib.sha256(listing.read_bytes()).hexdigest(),
                validation='Decode present images at 1640x590; validate finite paired annotation coordinates. Does not establish Table 1 metric equivalence.')
    (OUT/'culane_readiness.json').write_text(json.dumps(report,indent=2))
    (OUT/'culane_missing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records if not r['ready']))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
