"""Assemble the improved Table 1 side-by-side view (CPU only, reads existing evidence; trains nothing).

Improvements over scripts/render_table1.py's two separate tables:
  * one grid per dataset: every cell shows  published | local | delta (pp) | evidence status;
  * a coverage summary (verified / non-comparable / missing) per model and per dataset;
  * an audit of the *published* averages against the published ASR cells (reported, never altered);
  * a validated receipt import for poisoned-model measurements: a receipt only fills a cell if it is
    a full-test-split measurement with provenance. Average ASR is derived only when every applicable
    scenario for that row is validated; otherwise it stays MISSING. Published numbers are never copied
    into a measured column.

Receipts (JSON list or {"receipts":[...]}); each needs dataset, model, method, metric
(poisoned_acc|lda_asr|lra_asr|loa_asr), value_percent in [0,100], images (full test count),
evidence (path to prediction/score artifact), sha256 of that artifact's score file, and
independently_rescored=true.
"""
import argparse, csv, html, json, sys
from pathlib import Path

FULL_TEST={'TuSimple':2782,'CULane':34680}
ASR={'CULane':['lda_asr','lra_asr','loa_asr'],'TuSimple':['lda_asr','loa_asr']}
REQUIRED=('dataset','model','method','metric','value_percent','images','evidence','sha256','independently_rescored')
METRICS=('poisoned_acc','lda_asr','lra_asr','loa_asr')

def norm_method(m): return m.lower()

def validate_receipt(r):
    missing=[k for k in REQUIRED if k not in r or r[k] in (None,'')]
    if missing: return f'missing fields {missing}'
    if r['metric'] not in METRICS: return f"unsupported metric {r['metric']}"
    if r['dataset'] not in FULL_TEST: return 'unknown dataset'
    if r['metric']=='lra_asr' and r['dataset']=='TuSimple': return 'LRA is not in the TuSimple table'
    if r['images']!=FULL_TEST[r['dataset']]: return f"not a full-test measurement ({r['images']} != {FULL_TEST[r['dataset']]})"
    if not isinstance(r['value_percent'],(int,float)) or not 0<=r['value_percent']<=100: return 'value outside [0,100]'
    if r['independently_rescored'] is not True: return 'not independently re-scored'
    return None

def load_receipts(paths):
    ok,rejected=[],[]
    for p in paths:
        data=json.loads(Path(p).read_text())
        for r in (data['receipts'] if isinstance(data,dict) else data):
            err=validate_receipt(r)
            (rejected.append(dict(receipt=r,reason=err)) if err else ok.append(r))
    return ok,rejected

def audit_published_averages(cells):
    rows={}
    for c in cells:
        rows.setdefault((c['dataset'],c['model'],c['method']),{})[c['metric']]=float(c['published_percent'])
    findings=[]
    for (d,m,meth),v in rows.items():
        parts=[v[k] for k in ASR[d]]
        mean=sum(parts)/len(parts)
        findings.append(dict(dataset=d,model=m,method=meth,published_avg=v['avg_asr'],mean_of_published_asr=round(mean,2),
            diff_pp=round(v['avg_asr']-mean,2),consistent=abs(v['avg_asr']-mean)<=0.011))
    return findings

def overlay(cells,receipts):
    index={}
    for r in receipts: index[(r['dataset'],r['model'],norm_method(r['method']),r['metric'])]=r
    out=[]
    for c in cells:
        c=dict(c); r=index.get((c['dataset'],c['model'],norm_method(c['method']),c['metric']))
        if r and c['metric'] in METRICS:
            c.update(local_equivalent_percent=str(r['value_percent']),status='MEASURED (validated receipt)',evidence_status='VERIFIED',
                     evidence_artifacts=r['evidence'],reason='Full-test receipt, independently re-scored')
        out.append(c)
    by_row={}
    for c in out: by_row.setdefault((c['dataset'],c['model'],c['method']),{})[c['metric']]=c
    for (d,m,meth),row in by_row.items():
        parts=[row[k] for k in ASR[d]]
        if all(p['evidence_status']=='VERIFIED' and p['status'].startswith('MEASURED (validated') for p in parts):
            avg=sum(float(p['local_equivalent_percent']) for p in parts)/len(parts)
            row['avg_asr'].update(local_equivalent_percent=f'{avg:.6f}',status='DERIVED (mean of validated ASR)',evidence_status='VERIFIED',
                                  reason='Mean of all applicable validated scenario ASRs; same series')
    for c in out:
        v=c['local_equivalent_percent']
        c['difference_percentage_points']=f"{float(v)-float(c['published_percent']):.4f}" if v!='' else ''
    return out

def coverage(cells):
    table={}
    for c in cells:
        key=(c['dataset'],c['model']); t=table.setdefault(key,dict(VERIFIED=0,NOT_COMPARABLE=0,MISSING=0))
        t[c['evidence_status'].replace(' ','_')]+=1
    return table

def short(c):
    pub=c['published_percent']
    if c['evidence_status']=='VERIFIED': return f"{pub} → **{float(c['local_equivalent_percent']):.2f}** ({float(c['difference_percentage_points']):+.2f})"
    return f"{pub} → {'n/c' if c['evidence_status']=='NOT COMPARABLE' else '—'}"

def render(cells,findings,rejected):
    md=['# DBALD Table 1: published vs local, per cell','',
        'Each cell reads `published → local (delta pp)`. `—` = not measured; `n/c` = a local number exists but is NOT comparable (different metric or backbone, see table1_comparison.csv). Deltas are diagnostic and do not establish identical training conditions. Published numbers are never used as local measurements.','']
    cols={'CULane':[('benign_acc','B ACC'),('poisoned_acc','P ACC'),('lda_asr','LDA'),('lra_asr','LRA'),('loa_asr','LOA'),('avg_asr','Avg')],
          'TuSimple':[('benign_acc','B ACC'),('poisoned_acc','P ACC'),('lda_asr','LDA'),('loa_asr','LOA'),('avg_asr','Avg')]}
    for d,cl in cols.items():
        md+=[f'## {d}','','| Model | Method | '+' | '.join(h for _,h in cl)+' |','|---|---|'+'|'.join(['---']*len(cl))+'|']
        seen=[]
        for c in cells:
            k=(c['model'],c['method'])
            if c['dataset']==d and k not in seen: seen.append(k)
        for m,meth in seen:
            row={c['metric']:c for c in cells if c['dataset']==d and c['model']==m and c['method']==meth}
            md.append(f'| {m} | {meth} | '+' | '.join(short(row[k]) for k,_ in cl)+' |')
        md.append('')
    cov=coverage(cells); total=len(cells)
    md+=['## Coverage','','| Dataset | Model | Verified | Non-comparable | Missing |','|---|---|---:|---:|---:|']
    for (d,m),t in sorted(cov.items()): md.append(f"| {d} | {m} | {t['VERIFIED']} | {t['NOT_COMPARABLE']} | {t['MISSING']} |")
    s=lambda k:sum(t[k] for t in cov.values())
    md+=[f"| **All** | | **{s('VERIFIED')}** | **{s('NOT_COMPARABLE')}** | **{s('MISSING')}** |",'',f'{total} cells in total.','']
    bad=[f for f in findings if not f['consistent']]
    md+=['## Published-average audit','',f'{len(findings)-len(bad)} of {len(findings)} published average-ASR cells equal the mean of the published scenario ASRs (tolerance 0.011 pp). Inconsistent rows are listed as published; none is corrected.','']
    if bad:
        md+=['| Dataset | Model | Method | Published avg | Mean of published ASR | Diff (pp) |','|---|---|---|---:|---:|---:|']
        md+=[f"| {f['dataset']} | {f['model']} | {f['method']} | {f['published_avg']} | {f['mean_of_published_asr']} | {f['diff_pp']} |" for f in bad]
        md.append('')
    if rejected: md+=['## Rejected receipts','']+[f"- {r['reason']}: {r['receipt'].get('dataset')}/{r['receipt'].get('model')}/{r['receipt'].get('method')}/{r['receipt'].get('metric')}" for r in rejected]+['']
    return '\n'.join(md)+'\n'

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--comparison',default='results/table1_protocol_scnn_20261106/clean_recheck/table1_comparison.csv')
    ap.add_argument('--receipts',action='append',default=[])
    ap.add_argument('--output',required=True)
    a=ap.parse_args(argv)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise SystemExit('Output directory must be new or empty (evidence is never overwritten)')
    cells=list(csv.DictReader(Path(a.comparison).open(encoding='utf-8')))
    assert len(cells)==220,len(cells)
    receipts,rejected=load_receipts(a.receipts)
    final=overlay(cells,receipts); findings=audit_published_averages(cells)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'table1_improved.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(final[0])); w.writeheader(); w.writerows(final)
    text=render(final,findings,rejected)
    (out/'table1_improved.md').write_text(text,encoding='utf-8')
    (out/'table1_improved.html').write_text('<!doctype html><meta charset="utf-8"><title>DBALD Table 1</title><pre style="font:13px ui-monospace,monospace;white-space:pre-wrap">'+html.escape(text)+'</pre>',encoding='utf-8')
    (out/'published_average_audit.json').write_text(json.dumps(findings,indent=1))
    (out/'summary.json').write_text(json.dumps(dict(cells=len(final),coverage={f'{d}/{m}':t for (d,m),t in coverage(final).items()},
        receipts_accepted=len(receipts),receipts_rejected=len(rejected),inconsistent_published_averages=sum(not f['consistent'] for f in findings)),indent=1))
    print((out/'summary.json').read_text())

if __name__=='__main__': main()
