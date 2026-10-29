"""Render the published page-7 table and independently verified clean evidence.

Only reads reference/evaluation records; does not train or generate images.
Run after scripts/audit_table1.py and the optional fresh clean benchmark.
"""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lane_repro.evaluation import official_tusimple


def export_page7(reference, statuses, records, output):
    """Two explicitly separate 20-row tables with the original column ordering."""
    columns=list(reference[0])
    indexed={(r['model'],r['method'],r['dataset'],r['metric']):r for r in statuses}
    local=[]
    for reference_row in reference:
        row={'model':reference_row['model'],'attack':reference_row['attack']}
        for column in columns[2:]:
            dataset,metric=column.split('_',1)
            status=indexed[(row['model'],row['attack'],'TuSimple' if dataset=='tusimple' else 'CULane',metric)]
            row[column]=status['local_equivalent_percent'] if status['local_equivalent_percent']!='' else status['status']
        local.append(row)
    with (output/'table1.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['source']+columns); writer.writeheader()
        for label,rows in [('published_reference',reference),('independent_results',local)]:
            writer.writerows(dict(source=label,**row) for row in rows)
    headers=['Model','Method','CULane B ACC','CULane P ACC','LDA','LRA','LOA','Average',
             'TuSimple B ACC','TuSimple P ACC','LDA','LOA','Average']
    markdown=['# Original DBALD Table 1','',
        'Phase 1 scientifically complete: NO. Published references and independent results are separate. NOT COMPARABLE is not zero. See table1_status.csv and verification.json for provenance and reasons.','']
    page=['<!doctype html><meta charset="utf-8"><title>DBALD Table 1</title><style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;font-size:11px;margin-bottom:24px}td,th{border:1px solid #aaa;padding:5px}th{background:#eee}</style><h1>Original DBALD Table 1</h1><p>Phase 1 scientifically complete: NO. Published references and independently measured results are separate.</p>']
    for title, rows in [('Published reference — not local measurements',reference),('Independent results — unresolved protocols remain NOT COMPARABLE',local)]:
        markdown += ['## '+title,'','| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
        page += ['<h2>'+html.escape(title)+'</h2><table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>']
        for row in rows:
            values=[str(row[k]) for k in columns]
            markdown.append('| '+' | '.join(values)+' |')
            page.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
        markdown.append(''); page.append('</table>')
    markdown += ['## Supporting measurements','', '| Model | Backbone | Dataset | Images | Metric | Value % | Protocol / qualification |','|---|---|---|---:|---|---:|---|']
    page.append('<h2>Supporting measurements (not automatically paper-equivalent)</h2><table><tr><th>Model</th><th>Backbone</th><th>Dataset</th><th>Images</th><th>Metric</th><th>Value %</th><th>Protocol / qualification</th></tr>')
    for record in records:
        values=[record['model'],record.get('backbone','unconfirmed'),record['dataset'],str(record['images']),record['metric'],f"{record['value_percent']:.9f}",record.get('metric_definition',record['reason'])]
        markdown.append('| '+' | '.join(v.replace('|','\\|') for v in values)+' |')
        page.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    page.append('</table>')
    (output/'table1.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    (output/'table1.html').write_text('\n'.join(page),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/phase1_audit')
    parser.add_argument('--evidence',default='results/phase1_audit/clean_rescore.json')
    parser.add_argument('--additional-evidence',action='append',default=[])
    args=parser.parse_args()
    out = (ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'results') or out.is_relative_to(ROOT/'results/reference'):
        raise ValueError('Use a results output directory outside immutable reference')
    out.mkdir(parents=True,exist_ok=True)
    reference = ROOT/'results/reference/table1.csv'
    manifest = json.loads((reference.parent/'manifest.json').read_text())
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == manifest['sha256']
    rows = list(csv.DictReader(reference.open()))
    assert len(rows) == 20
    evidence = json.loads((ROOT/args.evidence).read_text())['verified']
    if args.additional_evidence:
        evidence=[r for r in evidence if r['dataset']=='TuSimple']
        for source in args.additional_evidence:
            for r in json.loads((ROOT/source).read_text())['verified']:
                if r['status']!='VERIFIED' or r['dataset']!='CULane' or r['images']!=34680:
                    raise ValueError('Additional evidence must be verified full CULane output')
                spec=json.loads((ROOT/r['evidence']/'run_config.json').read_text())
                evidence.append(dict(r,backbone=spec['backbone'],metric=r['metric_definition']))
                candidate_path=ROOT/r['evidence']/'point_acc_candidate.json'
                if candidate_path.exists():
                    candidate=json.loads(candidate_path.read_text())
                    if candidate['images']!=34680 or not candidate['full_test'] or candidate['total']<=0:
                        raise ValueError('Candidate point ACC lacks full coverage')
                    value=100*candidate['correct']/candidate['total']
                    if not 0<=value<=100 or abs(value-candidate['candidate_acc_percent'])>1e-9 or candidate['protocol']['table1_comparable']:
                        raise ValueError('Invalid candidate point ACC or equivalence claim')
                    evidence.append(dict(r,backbone=spec['backbone'],value_percent=value,
                                         metric='UNCONFIRMED pooled 20px point ACC; see protocol file'))
    fresh = out/'laneatt_full_cuda_verified'
    verification = None
    if (fresh/'completed.json').exists():
        labels = ROOT/'datasets/raw/tusimple_test/labels.jsonl'
        predictions = [json.loads(line) for line in (fresh/'predictions.jsonl').read_text().splitlines()]
        truth = [json.loads(line) for line in labels.read_text().splitlines()]
        keys = [p['raw_file'] for p in predictions]
        assert len(keys) == len(set(keys)) == len(truth) == 2782
        assert set(keys) == {g['raw_file'] for g in truth}
        assert all(p['transformation'] == 'clean' for p in predictions)
        truth_by_key = {g['raw_file']: g for g in truth}
        for prediction in predictions:
            assert prediction['h_samples'] == truth_by_key[prediction['raw_file']]['h_samples']
            assert math.isfinite(prediction['run_time']) and prediction['run_time'] >= 0
            assert all(len(lane) == len(prediction['h_samples']) and all(math.isfinite(x) for x in lane)
                       for lane in prediction['lanes'])
        metrics = json.loads(official_tusimple().bench_one_submit(str(fresh/'predictions.jsonl'),str(labels)))
        summary = json.loads((fresh/'summary.json').read_text())[0]
        assert abs(metrics[0]['value']*100-summary['acc_percent']) < 1e-9
        assert abs(metrics[1]['value']-summary['fp']) < 1e-9
        assert abs(metrics[2]['value']-summary['fn']) < 1e-9
        verification = dict(summary,independent_official_metrics=metrics,
            predictions_sha256=hashlib.sha256((fresh/'predictions.jsonl').read_bytes()).hexdigest())
        (out/'fresh_clean_verification.json').write_text(json.dumps(verification,indent=2))
    head = ['Model','Method','CULane B ACC','CULane P ACC','LDA','LRA','LOA','Average',
            'TuSimple B ACC','TuSimple P ACC','LDA','LOA','Average']
    md = ['# DBALD Table 1: published reference and independent verification','',
          'Published reference only: the following 20 rows reproduce the page-7 values, including printed averages. These are not local attack measurements. All values are percentages. B = benign model; P = poisoned model.','',
          '| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']
    data = [[r[k] for k in rows[0]] for r in rows]
    for row in data:
        md.append('| '+' | '.join(row)+' |')
    md += ['', '## Verified local clean baselines','',
           'These checkpoints are not established as identical to the paper checkpoints. CULane subset F1 must not replace Table 1 ACC.', '',
           '| Dataset | Model | Backbone | Images | Clean result % | Metric |','|---|---|---|---:|---:|---|']
    clean = []
    for r in evidence:
        line=[r['dataset'],r['model'],r['backbone'],str(r['images']),f"{r['value_percent']:.6f}",r['metric']]
        clean.append(line); md.append('| '+' | '.join(line)+' |')
    if verification:
        md += ['', '## Fresh GPU execution','',
               f"LaneATT ResNet-34, full TuSimple test split: {verification['images']} images, official ACC **{verification['acc_percent']:.6f}%**, runtime-gate failures {verification['runtime_failures']}. Independently re-scored with the official submission entry point.",
               '', 'Evidence: `laneatt_full_cuda_verified/` and `fresh_clean_verification.json`.']
    md += ['', '## Reproduction status','',
           'All local poisoned-model clean ACC and attack ASR cells remain unmeasured in this audit. Full CULane F1 evidence, where shown, does not establish paper point-ACC equivalence. Exact training/configuration equivalence remains unresolved. Phase 1 is incomplete.',
           '', 'Code: `scripts/render_table1.py` renders this comparison; `scripts/audit_table1.py` validates prior clean evidence and generates the 220-cell status CSV. No poisoning or training is performed by either script.']
    (out/'Table1_Comparison.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    def table(headers, records):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in records)+'</tbody></table>'
    page = '<!doctype html><html><meta charset="utf-8"><title>DBALD Table 1 comparison</title><style>body{font:15px system-ui;margin:30px;color:#17212b}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:12px}th,td{padding:7px;border:1px solid #bbc5ce;text-align:right}th{background:#e8eef4}td:first-child,td:nth-child(2){text-align:left}h1{font-size:24px}p{max-width:1000px}@media print{body{margin:10px}table{font-size:9px}thead{display:table-header-group}}</style><h1>DBALD Table 1 — published reference</h1><p>Page 7 reference values only. B = benign-model clean ACC; P = poisoned-model clean ACC. These are published numbers, not local attack measurements.</p>'
    page += table(head,data)+'<h2>Independently verified local clean evidence</h2><p>CULane F1 is a different metric from Table 1 ACC. Exact paper checkpoint equivalence is unestablished.</p>'
    page += table(['Dataset','Model','Backbone','Images','Clean result %','Metric'],clean)
    if verification:
        page += f'<h2>Fresh GPU evaluation</h2><p>LaneATT ResNet-34: {verification["images"]} TuSimple test images; official ACC {verification["acc_percent"]:.6f}%. Independently re-scored.</p>'
    page += '<h2>Phase 1 remains incomplete</h2><p>Poisoned-model ACC and ASR remain unmeasured. Full CULane F1 is not established as paper point ACC; exact training/configuration equivalence remains unresolved.</p></html>'
    (out/'Table1_Comparison.html').write_text(page,encoding='utf-8')
    from evaluate_table1 import cells
    supporting=[dict(r,id=r['model']+'_'+r['backbone']+' ['+r['metric']+']',metric='benign_acc',status='VERIFIED',
                     comparison_basis='official_tusimple' if r['dataset']=='TuSimple' and r['metric']=='official TuSimple ACC' else 'unconfirmed',
                     reason='Verified clean baseline; paper training/metric equivalence unconfirmed') for r in evidence]
    status=cells(rows,supporting)
    with (out/'table1_status.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(status[0])); writer.writeheader(); writer.writerows(status)
    print(json.dumps(dict(published_rows=len(rows),verified_prior_runs=len(evidence),fresh_clean=verification)))


if __name__ == '__main__':
    main()
