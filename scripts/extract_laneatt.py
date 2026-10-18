"""Extract completed ZIP members with CRC checks, including during a download.

Only the author archive's experiment .pt/.yaml files are selected. Once the full
archive exists, verify its central directory and record its SHA256 as well.
"""
import hashlib
import json
from pathlib import Path
import struct
import zlib

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'models/checkpoints'
paths=list(folder.glob('laneatt_official.zip')) or list(folder.glob('laneatt_official.zip*.part'))
if not paths: raise FileNotFoundError('Download the official archive first')
archive=paths[0]
records=[]
with archive.open('rb') as f:
    while True:
        header=f.read(30)
        if len(header)<30 or header[:4]!=b'PK\x03\x04': break
        _,version,flags,method,mtime,mdate,crc,compressed,uncompressed,nl,el=struct.unpack('<4s5H3I2H',header)
        name=f.read(nl).decode('utf-8'); f.read(el)
        if flags&8 or compressed==0xffffffff: raise ValueError('Unsupported ZIP member layout')
        if f.tell()+compressed>archive.stat().st_size: break
        wanted=name.endswith(('.pt','.yaml')) and name.startswith('experiments/')
        if not wanted: f.seek(compressed,1); continue
        data=f.read(compressed)
        if method==8: data=zlib.decompress(data,-15)
        elif method!=0: raise ValueError('Unsupported compression')
        if len(data)!=uncompressed or zlib.crc32(data)!=crc: raise ValueError('CRC/length mismatch')
        target=(folder/'laneatt_official'/name).resolve()
        if not target.is_relative_to((folder/'laneatt_official').resolve()): raise ValueError('Unsafe path')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): target.write_bytes(data)
        records.append(dict(member=name,bytes=len(data),crc32=f'{crc:08x}',sha256=hashlib.sha256(data).hexdigest(),path=str(target)))
report=dict(url='https://drive.google.com/uc?id=1R638ou1AMncTCRvrkQY6I-11CPwZy23T',
            provenance='Linked by lucastabelini/LaneATT README',archive_complete=archive.suffix=='.zip',members=records)
if archive.suffix=='.zip':
    import zipfile
    with zipfile.ZipFile(archive) as z:
        for row in records:
            assert z.getinfo(row['member']).CRC==int(row['crc32'],16)
    with archive.open('rb') as f: report['archive_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
(ROOT/'results/laneatt_assets.json').write_text(json.dumps(report,indent=2))
print('\n'.join(r['member'] for r in records))
