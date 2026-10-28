"""Independently verify clean evidence and export all 220 Table 1 cells.

Input is a versioned JSON manifest of evidence directories, not numeric claims.
Unsupported metrics are recorded explicitly; no training or attack execution.
"""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import sys

from audit_table1 import ROOT, sha, read, jsonlines
from lane_repro.evaluation import official_tusimple
from lane_repro.culane_official import score

MODELS = {'LaneATT', 'ADNet', 'SCNN', 'RESA'}
METHODS = {'BadNets', 'Blended', 'LD-Attack', 'BadLane', 'DBALD'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def covered(rows, expected, count):
    keys = [r['raw_file'] for r in rows]
    require(len(keys) == len(set(keys)) == len(expected) == count, 'Wrong or duplicate prediction count')
    require(set(keys) == set(expected), 'Annotation coverage mismatch')


def verified_average(values, expected_count):
    """Never average a partial scenario set or unverified claims."""
    if expected_count not in (2, 3) or len(values) != expected_count or any(v.get('status') != 'VERIFIED' for v in values):
        return None
    numbers = [v['value_percent'] for v in values]
    require(all(math.isfinite(v) and 0 <= v <= 100 for v in numbers), 'Invalid percentage')
    return sum(numbers) / len(numbers)


def verify(record, output):
    require(record['model'] in MODELS, 'Unknown model')
    require(record['dataset'] in ('TuSimple', 'CULane'), 'Unknown dataset')
    require(record['metric'] == 'benign_acc', 'No independently validated importer for this metric; leave NOT MEASURED')
    folder = ROOT / record['evidence']
    spec = read(folder/'run_config.json')
    provenance = read(folder/'model_provenance.json')
    require(spec['model'] == record['model'], 'Model/config mismatch')
    require(provenance['architecture'] == record['model'], 'Model/provenance mismatch')
    require(provenance.get('dataset', record['dataset']).lower() == record['dataset'].lower(), 'Dataset mismatch')
    checkpoint = ROOT / provenance['checkpoint']
    digest = sha(checkpoint)
    require(digest == spec['checkpoint_sha256'] == provenance['checkpoint_sha256'], 'Checkpoint hash mismatch')
    require(provenance.get('strict_state_dict') is True, 'Strict checkpoint loading not established')
    config = provenance.get('config')
    if spec.get('config_sha256'):
        require(config is not None and sha(ROOT/config) == spec['config_sha256'], 'Config hash mismatch')
    require(bool(spec.get('source_sha256')), 'Source provenance missing')
    if spec.get('anchor_order_sha256'):
        require(sha(ROOT/'configs/laneatt_anchor_order.json') == spec['anchor_order_sha256'], 'Anchor configuration mismatch')
    summaries = read(folder/'summary.json')
    summary = next(s for s in summaries if s['transformation'] == 'clean')
    source_drift = [path for path, expected in spec['source_sha256'].items()
                    if not (ROOT/path).is_file() or sha(ROOT/path) != expected]
    result = dict(record, checkpoint_sha256=digest, config_sha256=sha(folder/'run_config.json'),
                  source_drift=source_drift, source_provenance=spec['source_sha256'],
                  provenance_sha256=sha(folder/'model_provenance.json'), summary_sha256=sha(folder/'summary.json'),
                  status='VERIFIED', table1_status='NOT COMPARABLE',
                  reason='Exact paper checkpoint/configuration equivalence is not established')
    if record['dataset'] == 'TuSimple':
        labels = ROOT/record['annotations']
        require(sha(labels) == spec['labels_sha256'], 'Label hash mismatch')
        truth_rows = jsonlines(labels)
        truth = {r['raw_file']: r for r in truth_rows}
        require(len(truth) == len(truth_rows), 'Duplicate annotations')
        rows = [r for r in jsonlines(folder/'predictions.jsonl') if r.get('transformation') == 'clean']
        covered(rows, truth, 2782)
        for row in rows:
            require(row['h_samples'] == truth[row['raw_file']]['h_samples'], 'Sampling grid mismatch')
            require(math.isfinite(row['run_time']) and row['run_time'] >= 0, 'Invalid runtime')
            require(all(len(lane) == len(row['h_samples']) and all(math.isfinite(x) for x in lane)
                        for lane in row['lanes']), 'Invalid lane coordinates')
        submission = output/(record['id']+'.submission.jsonl')
        submission.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        metrics = {m['name']: m['value'] for m in json.loads(official_tusimple().bench_one_submit(str(submission), str(labels)))}
        for name, key, scale in [('Accuracy','acc_percent',100), ('FP','fp',1), ('FN','fn',1)]:
            require(math.isfinite(summary[key]) and abs(metrics[name]*scale-summary[key]) < 1e-9, 'Saved result mismatch: '+key)
        result.update(images=2782, value_percent=metrics['Accuracy']*100, metric_definition='Official TuSimple ACC, 200 ms gate enabled',
                      predictions_sha256=sha(folder/'predictions.jsonl'), annotations_sha256=sha(labels),
                      scorer_sha256=sha(ROOT/'third_party/tusimple-benchmark/evaluate/lane.py'), official_metrics=metrics)
    else:
        listing = ROOT/record['annotations']
        require(sha(listing) == spec['list_sha256'], 'Split hash mismatch')
        keys = [line.split()[0].lstrip('/') for line in listing.read_text().splitlines() if line.strip()]
        rows = jsonlines(folder/'predictions.jsonl')
        require(len(keys) == len(set(keys)), 'Duplicate annotation list')
        covered(rows, keys, 34680)
        for row in rows:
            dest = (folder/'lanes'/row['raw_file']).with_suffix('.lines.txt')
            require(dest.resolve().is_relative_to((folder/'lanes').resolve()), 'Prediction path escapes evidence')
            require(sha(dest) == row['prediction_sha256'], 'Exported lane hash mismatch')
        metrics = score(folder/'lanes', ROOT/record['data_root'], listing, output/(record['id']+'.official.txt'))
        require(all(metrics[k] == summary[k] for k in ('tp','fp','fn')), 'CULane count mismatch')
        require(abs(metrics['f1_percent']-summary['f1_percent']) < 1e-9, 'CULane F1 mismatch')
        result.update(images=34680, value_percent=metrics['f1_percent'], metric_definition='Official CULane lane IoU/F1; not Table 1 ACC',
                      reason='CULane F1 differs from Table 1 ACC; paper-compatible point scoring unresolved', official_metrics=metrics)
    return result


def cells(reference, records):
    require(len(reference) == 20 and {(r['model'],r['attack'].lower()) for r in reference} == {(m,a.lower()) for m in MODELS for a in METHODS}, 'Reference structure mismatch')
    result = []
    for row in reference:
        for dataset in ('CULane', 'TuSimple'):
            metrics = ['benign_acc','poisoned_acc','lda_asr'] + (['lra_asr'] if dataset == 'CULane' else []) + ['loa_asr','avg_asr']
            for metric in metrics:
                supporting = [r for r in records if r['model'] == row['model'] and r['dataset'] == dataset and r['metric'] == metric]
                value = float(row[dataset.lower()+'_'+metric])
                require(math.isfinite(value) and 0 <= value <= 100, 'Invalid reference value')
                result.append(dict(dataset=dataset, model=row['model'], method=row['attack'], metric=metric,
                    published_percent=value, local_equivalent_percent='', status='NOT COMPARABLE' if supporting else 'NOT MEASURED',
                    supporting_measurements='; '.join(f"{r['id']}: {r['value_percent']:.9f}%" for r in supporting),
                    reason='; '.join(sorted({r['reason'] for r in supporting})) if supporting else 'No independently verified equivalent measurement'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(output.is_relative_to(ROOT/'results') and not output.is_relative_to(ROOT/'results/reference'), 'Use a dedicated results directory')
    require(not output.exists(), 'Output exists; choose a new directory to preserve evidence')
    manifest = read(args.manifest)
    require(manifest['schema_version'] == 1, 'Unsupported schema')
    ids = [r['id'] for r in manifest['records']]
    require(len(ids) == len(set(ids)) and all(i.replace('_','').isalnum() for i in ids), 'Invalid/duplicate record ID')
    reference = ROOT/'results/reference/table1.csv'
    require(sha(reference) == read(reference.parent/'manifest.json')['sha256'], 'Published reference hash mismatch')
    output.mkdir(parents=True)
    records = []
    errors = []
    for record in manifest['records']:
        try:
            records.append(verify(record, output))
            print('Verified '+record['id'], flush=True)
        except (ValueError, KeyError, OSError, StopIteration) as error:
            errors.append(dict(id=record['id'], error=str(error)))
    rows = cells(list(csv.DictReader(reference.open())), records)
    report = dict(schema_version=1, reference_sha256=sha(reference), manifest_sha256=sha(args.manifest),
                  verified=records, rejected=errors, phase1_complete=False)
    (output/'verification.json').write_text(json.dumps(report, indent=2))
    with (output/'table1.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    headings = list(rows[0])
    lines = ['# Original DBALD Table 1 comparison', '', 'Published values are references. Blank local values are not zero. Phase 1 scientifically complete: NO.', '', '| '+' | '.join(headings)+' |', '|'+'|'.join(['---']*len(headings))+'|']
    lines += ['| '+' | '.join(str(r[k]).replace('|','\\|') for k in headings)+' |' for r in rows]
    (output/'table1.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    page = '<!doctype html><meta charset="utf-8"><title>DBALD Table 1</title><style>body{font:14px system-ui}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}</style><h1>Original DBALD Table 1 comparison</h1><p>Published references and verified evidence are separate. Phase 1 complete: NO.</p><table><tr>'
    page += ''.join('<th>'+html.escape(k)+'</th>' for k in headings)+'</tr>'
    page += ''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in headings)+'</tr>' for r in rows)+'</table>'
    (output/'table1.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(verified=len(records), rejected=errors, cells=len(rows))))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
