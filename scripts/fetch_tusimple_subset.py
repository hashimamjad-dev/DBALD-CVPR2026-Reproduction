"""Small deterministic labelled test subset. Primary mirror (Kaggle) returns HTTP 404 as of
2026-10-23 (dataset removed or renamed, confirmed by direct request); falls back to a Hugging
Face mirror of the same official TuSimple files, verified against the expected 2,782-record test
label count and (720,1280,3) image shape before use (see docs/sources.md)."""
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'datasets/raw/tusimple_subset'
BASE='https://www.kaggle.com/api/v1/datasets/download/manideep1108/tusimple/'
FALLBACK_BASE='https://huggingface.co/datasets/dhbloo/TuSimple/resolve/main/'

def _fallback_path(name):
    """Map the Kaggle-mirror-style name ('TUSimple/test_label.json', 'TUSimple/test_set/clips/...')
    onto the Hugging Face mirror's layout (labels at root, images under 'clips/')."""
    for prefix in ('TUSimple/test_set/','TUSimple/'):
        if name.startswith(prefix): return name[len(prefix):]
    return name

def fetch(name):
    try:
        response=requests.get(BASE+quote(name,safe=''),timeout=30)
        response.raise_for_status()
        data=response.content
    except requests.RequestException:
        data=None
    if data is None or (data[:1]==b'<' and b'<!DOCTYPE html>' in data[:200]):
        response=requests.get(FALLBACK_BASE+quote(_fallback_path(name),safe='/'),timeout=120)
        response.raise_for_status(); data=response.content
    if data.startswith(b'PK'):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members=[m for m in archive.namelist() if not m.endswith('/')]
            if len(members)!=1: raise ValueError('Expected a single-file response')
            data=archive.read(members[0])
    return data

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    label_path=DEST/'all_test_label.json'
    if not label_path.exists(): label_path.write_bytes(fetch('TUSimple/test_label.json'))
    records=[json.loads(line) for line in label_path.read_text().splitlines() if line.strip()]
    # Fixed, evenly spaced indices chosen before running any model.
    indices=[i*(len(records)-1)//15 for i in range(16)]
    selected=[records[i] for i in indices]
    provenance=[]
    for record in selected:
        name='TUSimple/test_set/'+record['raw_file']
        dest=(DEST/record['raw_file']).resolve()
        if not dest.is_relative_to(DEST.resolve()): raise ValueError('Invalid sample path')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.write_bytes(fetch(name))
        provenance.append(dict(raw_file=record['raw_file'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        print('Ready',record['raw_file'],flush=True)
    (DEST/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (ROOT/'results/dataset_subset.json').write_text(json.dumps(dict(source='https://github.com/TuSimple/tusimple-benchmark/issues/3',
        mirror='https://www.kaggle.com/datasets/manideep1108/tusimple',
        mirror_status='Kaggle mirror returned HTTP 404 on 2026-10-23; downloads used the Hugging Face fallback below',
        fallback_mirror='https://huggingface.co/datasets/dhbloo/TuSimple',
        status='16-image smoke subset, not full benchmark',
        full_test_label_count=len(records),indices=indices,label_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),samples=provenance),indent=2))

if __name__=='__main__': main()
