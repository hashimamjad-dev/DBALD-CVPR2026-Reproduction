"""Strict bounded HTTP ranges for servers/proxies that reject open-ended resume."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import shutil
import time
import requests
import fetch_culane_test as official

def download(file_id,path):
    base=f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
    with requests.get(base+'&size_probe=bounded',headers={'Range':'bytes=0-0'},stream=True,timeout=60) as r:
        r.raise_for_status(); expected=int(r.headers['Content-Range'].split('/')[-1])
    start=path.stat().st_size if path.exists() else 0
    if start>expected: raise ValueError('Existing archive larger than server object')
    if start==expected: return
    chunkdir=path.parent/(path.name+'.chunks'); chunkdir.mkdir(exist_ok=True)
    chunk_size=8*1024*1024
    ranges=[(lo,min(lo+chunk_size,expected)-1) for lo in range(start,expected,chunk_size)]
    def get(interval):
        lo,hi=interval; dest=chunkdir/f'{lo}-{hi}.chunk'
        if dest.exists() and dest.stat().st_size==hi-lo+1: return lo,hi,dest
        for attempt in range(3):
            try:
                if shutil.disk_usage(path.parent).free<2*1024**3: raise RuntimeError('Low disk space')
                response=requests.get(base+f'&bounded_range={lo}-{hi}&attempt={attempt}&request_nonce={time.time_ns()}',headers={'Range':f'bytes={lo}-{hi}'},timeout=(30,90))
                response.raise_for_status()
                if 'text/html' in response.headers.get('Content-Type','') and 'Quota exceeded' in response.text:
                    raise PermissionError('Google Drive download quota exceeded; preserve ranges and resume after server reset')
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{expected}' or len(response.content)!=hi-lo+1:
                    raise ValueError(f"Expected bytes {lo}-{hi}/{expected}; received status {response.status_code}, range {response.headers.get('Content-Range')}, length {len(response.content)}")
                part=dest.with_suffix('.part'); part.write_bytes(response.content); part.replace(dest)
                return lo,hi,dest
            except Exception as error:
                print(f'Range {lo}-{hi}, attempt {attempt+1}: {type(error).__name__}: {error}',flush=True)
                if isinstance(error,PermissionError): raise
                if attempt==2: raise
                time.sleep(2**attempt)
    ready={}; next_byte=start
    with path.open('ab') as output,ThreadPoolExecutor(max_workers=4) as pool:
        pending=iter(ranges); futures={pool.submit(get,x) for x in [next(pending,None) for _ in range(min(4,len(ranges)))] if x is not None}; n=0
        while futures:
            done,futures=wait(futures,return_when=FIRST_COMPLETED)
            for future in done:
                lo,hi,dest=future.result(); ready[lo]=(hi,dest); n+=1
                while next_byte in ready:
                    end,part=ready.pop(next_byte)
                    output.write(part.read_bytes()); output.flush(); part.unlink(); next_byte=end+1
                interval=next(pending,None)
                if interval is not None: futures.add(pool.submit(get,interval))
                if n%16==0 or n==len(ranges): print(f'{path.name}: {n}/{len(ranges)} ranges; contiguous {next_byte}/{expected}',flush=True)
    if path.stat().st_size!=expected: raise ValueError('Incomplete archive')

if __name__=='__main__':
    official.download=download
    official.main()
