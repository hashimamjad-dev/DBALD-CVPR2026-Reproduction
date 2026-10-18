"""Small deterministic labelled test subset from the maintainer-linked mirror."""
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

def fetch(name):
    response=requests.get(BASE+quote(name,safe=''),timeout=120)
    response.raise_for_status()
    data=response.content
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
        provenance.append(dict(raw_file=record['raw_file'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),url=BASE+quote(name,safe='')))
        print('Ready',record['raw_file'],flush=True)
    (DEST/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (ROOT/'results/dataset_subset.json').write_text(json.dumps(dict(source='https://github.com/TuSimple/tusimple-benchmark/issues/3',
        mirror='https://www.kaggle.com/datasets/manideep1108/tusimple',status='16-image smoke subset, not full benchmark',
        full_test_label_count=len(records),indices=indices,label_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),samples=provenance),indent=2))

if __name__=='__main__': main()
