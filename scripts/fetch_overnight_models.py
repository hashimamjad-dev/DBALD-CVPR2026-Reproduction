"""Public CULane model checkpoints; preserve download hashes and source URLs."""
import hashlib
import json
from pathlib import Path
import requests
import subprocess
import zipfile

ASSETS={
 'adnet_culane_res18.pth':'https://github.com/Sephirex-X/ADNet/releases/download/ckpt/culane_res18.pth',
 'scnn_r50_culane.zip':'https://github.com/Turoad/lanedet/releases/download/1.0/scnn_r50_culane.zip',
 'resa_r34_culane.zip':'https://github.com/Turoad/lanedet/releases/download/1.0/resa_r34_culane.zip'}

def main(manifest='results/culane_checkpoints.json'):
    folder=Path('models/checkpoints'); records=[]
    for name,url in ASSETS.items():
        path=folder/name
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.partial')
            subprocess.run(['C:/Windows/System32/curl.exe','--location','--fail','--silent','--show-error',
                '--retry','2','--retry-delay','3','--connect-timeout','30','--speed-time','60','--speed-limit','4096',
                '--max-time','1800','--continue-at','-','--output',str(partial),url],check=True)
            partial.replace(path)
        with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        record=dict(file=str(path),url=url,sha256=sha,bytes=path.stat().st_size)
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    if not (folder/entry.filename).resolve().is_relative_to(folder.resolve()): raise ValueError('Unsafe archive')
                archive.extractall(folder)
                record['members']=archive.namelist()
        records.append(record); Path(manifest).write_text(json.dumps(records,indent=2)); print(record,flush=True)

if __name__=='__main__': main()
