"""Validate an existing CULane gzip archive and extract only missing files.

Unlike the existing download helper's extractor, never overwrite existing data.
Existing members must match their archived bytes. No download or model execution.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'datasets/raw/culane'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',choices=['driver_100_30frame.tar.gz','driver_193_90frame.tar.gz'])
    args=parser.parse_args()
    archive=DATA/args.archive
    # Complete CRC check before modifying the dataset.
    with gzip.open(archive,'rb') as stream:
        while stream.read(8*1024*1024): pass
    with archive.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    added=matched=0
    with tarfile.open(archive,'r|gz') as bundle:
        for member in bundle:
            target=(DATA/member.name).resolve()
            if not target.is_relative_to(DATA.resolve()) or member.issym() or member.islnk():
                raise ValueError(f'Unsafe member: {member.name}')
            if member.isdir(): continue
            if not member.isfile(): raise ValueError(f'Unsupported member: {member.name}')
            source=bundle.extractfile(member)
            if target.exists():
                with target.open('rb') as current:
                    actual=hashlib.file_digest(current,'sha256').hexdigest()
                expected=hashlib.file_digest(source,'sha256').hexdigest()
                if actual != expected: raise ValueError(f'Existing file differs; preserved: {target}')
                matched+=1
            else:
                if shutil.disk_usage(DATA).free < member.size+2*1024**3:
                    raise RuntimeError('Less than 2 GiB reserve; extraction stopped')
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name(target.name+'.phase1-extract-part')
                with temporary.open('xb') as output:
                    shutil.copyfileobj(source,output)
                if temporary.stat().st_size != member.size: raise ValueError('Truncated member')
                temporary.rename(target)
                added+=1
            if (added+matched)%10000 == 0:
                print(f'{added} added, {matched} existing identical',flush=True)
    report=dict(archive=args.archive,sha256=digest,added=added,existing_identical=matched,
                archive_retained=True,no_existing_file_overwritten=True)
    (ROOT/'results/phase1_audit'/f'{args.archive}.extraction.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
