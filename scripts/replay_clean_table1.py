"""Replay saved full clean predictions with existing scorers; never train or edit images.

The existing evaluator performs all checkpoint, split, prediction and metric checks.
This wrapper restricts its input to clean records and emits a completion matrix.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from audit_table1 import ROOT, read, sha
from table1_evidence import artifact


def clean_manifest(path):
    manifest=read(path)
    if manifest.get('schema_version')!=1 or not manifest.get('records'):
        raise ValueError('A version-1 manifest of saved clean runs is required')
    if manifest.get('external_records'):
        raise ValueError('Clean replay does not accept external scenario records')
    for cached in manifest.get('verified_caches',[]):
        if cached.get('dataset')!='CULane':
            raise ValueError('Only reviewed full clean CULane receipts can avoid redundant native rescoring')
        rows=read(artifact(cached))['verified']
        if not rows or any(r.get('dataset')!='CULane' or r.get('metric')!='benign_acc'
                           or r.get('status')!='VERIFIED' or not r.get('official_metrics') for r in rows):
            raise ValueError('Cached receipt must contain independently scored clean CULane records')
    for row in manifest['records']:
        if row.get('metric')!='benign_acc' or row.get('method'):
            raise ValueError('This runner accepts clean baselines only')
    return manifest


def completion_rows(records):
    rows=[]
    for model in ('LaneATT','ADNet','SCNN','RESA'):
        for dataset in ('TuSimple','CULane'):
            supported=[r for r in records if r['model']==model and r['dataset']==dataset
                       and r['metric']=='benign_acc' and r.get('status')=='VERIFIED'
                       and r.get('official_metrics')]
            expected=2782 if dataset=='TuSimple' else 34680
            if any(r['images']!=expected for r in supported):
                raise ValueError('Completion matrix requires full official splits')
            matched=[r for r in supported if r['backbone']=='resnet34']
            selected=matched or supported
            value=selected[0]['value_percent'] if len(selected)==1 else ''
            rows.append(dict(model=model,dataset=dataset,images=expected,
                clean_evidence_status='VERIFIED' if supported else 'MISSING',
                table1_status='NOT COMPARABLE' if supported else 'MISSING',
                paper_backbone_available=bool(matched),
                backbones='; '.join(sorted({r['backbone'] for r in supported})),
                metric='official TuSimple ACC (200 ms gate)' if dataset=='TuSimple' else 'official CULane lane IoU/F1; not Table 1 ACC',
                value_percent=value,
                evidence='; '.join(r['evidence'] for r in supported),
                checkpoint_sha256='; '.join(sorted({r['checkpoint_sha256'] for r in supported})),
                source_drift='; '.join(sorted({p for r in supported for p in r.get('source_drift',[])})),
                reason='Independent pretrained checkpoints/training; exact author equivalence unverified'+
                       ('; paper-compatible CULane point scoring unresolved' if dataset=='CULane' else '')+
                       ('; full-test SCNN ResNet-34 measurement unavailable in this evidence set' if model=='SCNN' and not matched else '')))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='configs/table1_clean_replay.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    clean_manifest(args.manifest)
    output=Path(args.output).resolve()
    if not output.is_relative_to(ROOT/'results') or output.exists():
        raise ValueError('Use a fresh results directory')
    command=[sys.executable,str(ROOT/'scripts/evaluate_table1.py'),'--manifest',args.manifest,'--output',str(output)]
    result=subprocess.run(command,cwd=ROOT)
    if result.returncode:
        return result.returncode
    receipt=read(output/'verification.json')
    rows=completion_rows(receipt['verified'])
    with (output/'completion_matrix.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(command=command,manifest_sha256=sha(args.manifest),verification_sha256=sha(output/'verification.json'),
                python=sys.version,experiment_kind='independent rescoring of existing clean predictions; no new inference or training',
                seed='Original inference seeds retained in each run_config.json',rows=rows)
    (output/'completion_matrix.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'clean_model_dataset_pairs':sum(r['clean_evidence_status']=='VERIFIED' for r in rows),'records':len(receipt['verified'])}))
    return 0


if __name__=='__main__':
    sys.exit(main())
