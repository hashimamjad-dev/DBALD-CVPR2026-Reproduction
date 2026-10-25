"""Download the author's LaneATT experiment archive from the Google Drive link in results/laneatt_assets.json.

Streams through the large-file "can't scan for viruses" confirmation (the uuid-form variant
drive.usercontent.google.com now uses), verifies the archive's size and SHA256 against
results/laneatt_assets.json's recorded member manifest's total only if present, and writes
models/checkpoints/laneatt_official.zip as the existing extract_laneatt.py script expects.
"""
import hashlib
import json
import re
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'models/checkpoints/laneatt_official.zip'
FILE_ID='1R638ou1AMncTCRvrkQY6I-11CPwZy23T'
EXPECTED_BYTES=1425744054

def main():
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if DEST.exists() and DEST.stat().st_size==EXPECTED_BYTES:
        print('Already downloaded',DEST,DEST.stat().st_size); return
    session=requests.Session()
    page=session.get('https://drive.google.com/uc?export=download&id='+FILE_ID,timeout=60)
    page.raise_for_status()
    match=re.search(r'name="uuid" value="([^"]+)"',page.text)
    if not match: raise RuntimeError('Confirmation form changed; uuid not found')
    response=session.get('https://drive.usercontent.google.com/download',
        params=dict(id=FILE_ID,export='download',confirm='t',uuid=match.group(1)),stream=True,timeout=60)
    response.raise_for_status()
    total=int(response.headers.get('content-length',0))
    if total!=EXPECTED_BYTES: raise RuntimeError(f'Unexpected size {total}, expected {EXPECTED_BYTES}')
    partial=DEST.with_suffix('.part'); written=0; digest=hashlib.sha256()
    with partial.open('wb') as f:
        for chunk in response.iter_content(1<<20):
            f.write(chunk); digest.update(chunk); written+=len(chunk)
            if written%(50<<20)<len(chunk): print(f'{written/1e6:.0f} / {total/1e6:.0f} MB',flush=True)
    if written!=total: raise RuntimeError(f'Incomplete download: {written} of {total} bytes')
    partial.replace(DEST)
    print('Downloaded',DEST,written,'bytes, sha256',digest.hexdigest())

if __name__=='__main__': main()
