"""Fetch public trained SCNN R34 research checkpoints without replacing any weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

ASSETS={
    'tusimple':('1M0ROpEHV8DGJT4xWq2eURcbqMzpea1q7','resnet34_scnn_tusimple_20210424.pt'),
    'culane':('1JyPJQv8gpFZbr1sh7zRRiekUuDR4Aea8','resnet34_scnn_culane_20210220.pt')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['tusimple','culane','all'],default='all')
    p.add_argument('--receipt',required=True)
    a=p.parse_args();receipt=Path(a.receipt).resolve()
    if not receipt.is_relative_to(ROOT/'results') or receipt.is_relative_to(ROOT/'results/reference') or receipt.exists():raise ValueError('Use a new results receipt outside the immutable reference')
    folder=ROOT/'models/checkpoints/scnn_r34_autodrive';folder.mkdir(parents=True,exist_ok=True)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    report=dict(source='https://github.com/voldemortX/pytorch-auto-drive/blob/master/docs/MODEL_ZOO.md',
                implementation_commit=subprocess.check_output(['git','-C','third_party/pytorch_auto_drive','rev-parse','HEAD'],text=True).strip(),assets=[])
    for dataset,(file_id,name) in ASSETS.items():
        if a.dataset not in ('all',dataset):continue
        dest=folder/name;row=dict(dataset=dataset,path=str(dest),source_url=f'https://drive.google.com/file/d/{file_id}/view',status='INCOMPLETE')
        try:
            if not dest.exists():
                import gdown
                partial=dest.with_suffix(dest.suffix+f'.{time.time_ns()}.partial')
                if gdown.download(id=file_id,output=str(partial),quiet=False) is None:raise RuntimeError('Download unavailable')
                if partial.stat().st_size<1024**2:raise ValueError('Not a model-sized response; partial response preserved')
                partial.replace(dest)
            with dest.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            row.update(status='DOWNLOADED_NOT_YET_VALIDATED',bytes=dest.stat().st_size,sha256=digest)
        except Exception as error:row['error']=str(error)
        report['assets'].append(row);receipt.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    return 0 if all(r['status']=='DOWNLOADED_NOT_YET_VALIDATED' for r in report['assets']) else 1


if __name__=='__main__':raise SystemExit(main())
