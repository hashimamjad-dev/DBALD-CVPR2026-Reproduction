"""Assemble the thesis results table from measured result files only; a missing measurement stays 'not measured'."""
import json
from pathlib import Path

MISSING='not measured'

def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None

def seconds(record):
    return None if record is None else record['median_seconds']

def mib(record):
    return None if record is None else record['peak_bytes']/2**20

def fmt(value,digits=3):
    return MISSING if value is None else f'{value:.{digits}f}'

def speedup(reference,value):
    return None if reference is None or value is None or value<=0 else reference/value

def asr_cell(record,key):
    if record is None or record.get(key) is None: return MISSING
    return f'{100*record[key]:.2f}'

def rows(root):
    root=Path(root); out=[]
    reference=seconds(read(root/'results/distillation/v2/timing/teacher20.json'))
    def add(label,timing,steps,note,asr=None):
        out.append(dict(label=label,steps=steps,seconds=seconds(timing),mib=mib(timing),speedup=speedup(reference,seconds(timing)),note=note,
            lda=asr_cell(asr,'lda'),lra=asr_cell(asr,'lra'),loa=asr_cell(asr,'loa')))
    for name,steps,note in [('teacher20',20,'SD1.5 inpainting proxy, reference'),('teacher4',4,'untrained, 4 steps'),('student_v1',4,'v1 student'),('mask_diverse',4,'v2 student')]:
        add(name,read(root/f'results/distillation/v2/timing/{name}.json'),steps,note,read(root/f'results/asr/{name}.json'))
    for path in sorted((root/'results/speedup/timing').glob('*.json')):
        if path.stem not in ('teacher20','teacher4'): add(path.stem,read(path),None,'training-free speedup',read(root/f'results/asr/{path.stem}.json'))
    for folder in sorted((root/'results/distillation/v3').glob('*/')):
        if (folder/'complete.json').exists(): add('v3_'+folder.name,read(folder/'timing.json'),4,'v3 student',read(root/f'results/asr/v3_{folder.name}.json'))
    return out

def render(table):
    head='| Method | Steps | Median s/image | Peak MiB | Speedup vs teacher20 | LDA ASR % | LRA ASR % | LOA ASR % | Note |\n|---|---|---|---|---|---|---|---|---|\n'
    body=''.join(f"| {r['label']} | {r['steps'] if r['steps'] else '-'} | {fmt(r['seconds'])} | {fmt(r['mib'],0)} | {fmt(r['speedup'],2)} | {r['lda']} | {r['lra']} | {r['loa']} | {r['note']} |\n" for r in table)
    return head+body
