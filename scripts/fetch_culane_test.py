"""Official test-only archives; resumable downloads and safe extraction."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import gdown
import subprocess
import sys
import time
import gzip
import requests

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'datasets/raw/culane'
ARCHIVES=[('list.tar.gz','18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd'),
 ('driver_37_30frame.tar.gz','1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7'),
 ('driver_100_30frame.tar.gz','1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8'),
 ('driver_193_90frame.tar.gz','1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV')]

def download(file_id,path):
    url=f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
    with requests.get(url+'&metadata_probe=1',headers={'Range':'bytes=0-0'},stream=True,timeout=60) as response:
        response.raise_for_status()
        expected=int(response.headers['Content-Range'].split('/')[-1])
    for attempt in range(3):
        if path.exists() and path.stat().st_size==expected: return
        command=['C:/Windows/System32/curl.exe','--location','--fail','--silent','--show-error',
            '--connect-timeout','30','--speed-time','60','--speed-limit','4096','--max-time','1800',
            '--continue-at','-','--output',str(path),url+f'&download_session={time.monotonic_ns()}']
        child=subprocess.Popen(command)
        previous=-1; changed=time.monotonic()
        while child.poll() is None:
            size=sum(p.stat().st_size for p in path.parent.glob(path.name+'*') if p.is_file())
            if size!=previous: previous=size; changed=time.monotonic()
            if time.monotonic()-changed>120 or shutil.disk_usage(path.parent).free<2*1024**3:
                child.terminate(); child.wait(); print(f'Resuming stalled download: {path.name}, attempt {attempt+1}',flush=True); break
            time.sleep(5)
        if child.returncode==0 and path.stat().st_size==expected: return
    raise RuntimeError(f'Download remained unavailable after three bounded attempts: {path.name}')

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'results/culane_download.json'
    records=json.loads(manifest.read_text()) if manifest.exists() else []
    for name,file_id in ARCHIVES:
        if any(r['archive']==name and r.get('status')=='extracted' for r in records): continue
        if shutil.disk_usage(DEST).free<8*1024**3: raise RuntimeError('Need 8 GiB free before another archive')
        path=DEST/name
        download(file_id,path)
        # Validate the complete gzip stream and CRC before extracting anything.
        with gzip.open(path,'rb') as stream:
            while stream.read(8*1024*1024): pass
        with path.open('rb') as handle: checksum=hashlib.file_digest(handle,'sha256').hexdigest()
        count=0; size=0
        with tarfile.open(path,'r:gz') as archive:
            for member in archive:
                target=(DEST/member.name).resolve()
                if not target.is_relative_to(DEST.resolve()) or member.issym() or member.islnk(): raise ValueError('Unsafe member')
                if member.isfile():
                    if shutil.disk_usage(DEST).free<member.size+2*1024**3: raise RuntimeError('Insufficient extraction space')
                    archive.extract(member,DEST,filter='data'); count+=1; size+=member.size
        records.append(dict(archive=name,id=file_id,sha256=checksum,files=count,extracted_bytes=size,status='extracted',
            source='https://xingangpan.github.io/projects/CULane.html',license='non-commercial; no redistribution'))
        manifest.write_text(json.dumps(records,indent=2))
        # Only this verified archive is redundant; retain every extracted dataset file.
        path.unlink()
        print(json.dumps(records[-1]),flush=True)

if __name__=='__main__': main()
