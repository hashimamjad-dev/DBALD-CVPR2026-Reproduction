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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lane_repro.evaluation import official_tusimple


def main():
    out = ROOT/'results/phase1_audit'
    reference = ROOT/'results/reference/table1.csv'
    manifest = json.loads((reference.parent/'manifest.json').read_text())
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == manifest['sha256']
    rows = list(csv.DictReader(reference.open()))
    assert len(rows) == 20
    evidence = json.loads((out/'clean_rescore.json').read_text())['verified']
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
           'All local poisoned-model clean ACC and attack ASR cells remain unmeasured in this audit. Full CULane evaluation, exact paper configurations/checkpoints, and SCNN cross-version parity remain unresolved. Phase 1 is incomplete.',
           '', 'Code: `scripts/render_table1.py` renders this comparison; `scripts/audit_table1.py` validates prior clean evidence and generates the 220-cell status CSV. No poisoning or training is performed by either script.']
    (out/'Table1_Comparison.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    def table(headers, records):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in records)+'</tbody></table>'
    page = '<!doctype html><html><meta charset="utf-8"><title>DBALD Table 1 comparison</title><style>body{font:15px system-ui;margin:30px;color:#17212b}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:12px}th,td{padding:7px;border:1px solid #bbc5ce;text-align:right}th{background:#e8eef4}td:first-child,td:nth-child(2){text-align:left}h1{font-size:24px}p{max-width:1000px}@media print{body{margin:10px}table{font-size:9px}thead{display:table-header-group}}</style><h1>DBALD Table 1 — published reference</h1><p>Page 7 reference values only. B = benign-model clean ACC; P = poisoned-model clean ACC. These are published numbers, not local attack measurements.</p>'
    page += table(head,data)+'<h2>Independently verified local clean evidence</h2><p>CULane F1 is a different metric from Table 1 ACC. Exact paper checkpoint equivalence is unestablished.</p>'
    page += table(['Dataset','Model','Backbone','Images','Clean result %','Metric'],clean)
    if verification:
        page += f'<h2>Fresh GPU evaluation</h2><p>LaneATT ResNet-34: {verification["images"]} TuSimple test images; official ACC {verification["acc_percent"]:.6f}%. Independently re-scored.</p>'
    page += '<h2>Phase 1 remains incomplete</h2><p>Poisoned-model ACC and ASR remain unmeasured in this audit. Full CULane evaluation and implementation equivalence remain unresolved.</p></html>'
    (out/'Table1_Comparison.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(published_rows=len(rows),verified_prior_runs=len(evidence),fresh_clean=verification)))


if __name__ == '__main__':
    main()
