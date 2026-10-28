"""Re-score existing clean evidence and render all published Table 1 cells.

No training, image editing, poisoning, or attack execution. Existing evidence is
read-only; generated submissions and reports stay under --output.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lane_repro.evaluation import official_tusimple
from lane_repro.culane_official import score


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/phase1_audit')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT / 'results') or out == ROOT / 'results/reference':
        raise ValueError('Use a dedicated results output directory')
    out.mkdir(parents=True, exist_ok=True)
    reference = ROOT / 'results/reference/table1.csv'
    assert sha(reference) == read(ROOT / 'results/reference/manifest.json')['sha256']
    published = list(csv.DictReader(reference.open()))
    measured = list(csv.DictReader((ROOT / 'results/measured_results.csv').open()))
    evaluator = official_tusimple()
    verified = []
    for row in measured:
        if row['validation_status'] != 'current' or row['transformation'] != 'clean':
            continue
        if row['dataset'] == 'TuSimple' and row['scope'] != 'full_test':
            continue
        folder = (ROOT / row['evidence']).parent
        identifier = '_'.join(folder.parts[-2:])
        summary = next(s for s in read(folder / 'summary.json') if s['transformation'] == 'clean')
        result = dict(model=row['model'], dataset=row['dataset'], backbone=row['backbone'],
                      evidence=str(folder.relative_to(ROOT)), summary_sha256=sha(folder/'summary.json'),
                      checkpoint_sha256=row['checkpoint_sha256'])
        if row['dataset'] == 'TuSimple':
            labels = ROOT / 'datasets/raw/tusimple_test/labels.jsonl'
            truth_rows = jsonlines(labels)
            truth = {r['raw_file']: r for r in truth_rows}
            predictions = [r for r in jsonlines(folder/'predictions.jsonl') if r['transformation'] == 'clean']
            keys = [r['raw_file'] for r in predictions]
            assert len(truth_rows) == len(truth) == len(keys) == len(set(keys)) == 2782
            assert set(keys) == set(truth)
            for r in predictions:
                assert r['h_samples'] == truth[r['raw_file']]['h_samples']
                assert all(len(lane) == len(r['h_samples']) for lane in r['lanes'])
            spec = read(folder/'run_config.json')
            assert sha(labels) == spec['labels_sha256']
            provenance = read(folder/'model_provenance.json')
            checkpoint = ROOT / provenance['checkpoint']
            assert sha(checkpoint) == row['checkpoint_sha256']
            submission = out / (identifier + '_submission.jsonl')
            submission.write_text(''.join(json.dumps({k:r[k] for k in ('raw_file','lanes','run_time')})+'\n' for r in predictions))
            metrics = {m['name']:m['value'] for m in json.loads(evaluator.bench_one_submit(str(submission),str(labels)))}
            assert abs(metrics['Accuracy']*100-summary['acc_percent']) < 1e-9
            assert abs(metrics['FP']-summary['fp']) < 1e-9
            assert abs(metrics['FN']-summary['fn']) < 1e-9
            result.update(images=2782, metric='official TuSimple ACC', value_percent=metrics['Accuracy']*100,
                          fp=metrics['FP'], fn=metrics['FN'], predictions_sha256=sha(folder/'predictions.jsonl'),
                          labels_sha256=sha(labels), checkpoint_hash_verified=True,
                          runtime_failures=sum(r['run_time']>200 for r in predictions),
                          image_files_present=sum((ROOT/'datasets/raw/tusimple_test'/k).is_file() for k in keys))
        else:
            values = score(folder/'lanes', ROOT/'datasets/raw/culane', ROOT/'datasets/culane_smoke16.txt', out/(identifier+'_official.txt'))
            assert all(values[k] == summary[k] for k in ('tp','fp','fn'))
            assert abs(values['f1_percent']-summary['f1_percent']) < 1e-9
            result.update(images=values['images'], metric='official CULane IoU/F1; NOT Table 1 ACC',
                          value_percent=values['f1_percent'], tp=values['tp'], fp=values['fp'], fn=values['fn'],
                          command=values['command'])
        verified.append(result)
        print(json.dumps(result), flush=True)
    listing = ROOT/'datasets/raw/culane/list/test.txt'
    keys = [line.split()[0].lstrip('/') for line in listing.read_text().splitlines() if line.strip()]
    coverage = dict(expected_test_images=34680, listed=len(keys),
                    image_files_present=sum((ROOT/'datasets/raw/culane'/k).is_file() for k in keys),
                    annotation_files_present=sum((ROOT/'datasets/raw/culane'/k).with_suffix('.lines.txt').is_file() for k in keys),
                    note='File existence only; not full decoding/integrity verification')
    (out/'clean_rescore.json').write_text(json.dumps(dict(verified=verified,culane_coverage=coverage,
        scorer_sha256=sha(ROOT/'third_party/tusimple-benchmark/evaluate/lane.py'),reference_sha256=sha(reference)),indent=2))
    # Preserve all 20 method rows in both datasets. A baseline is supporting
    # evidence, not a poisoned-model measurement or a claim of paper equivalence.
    cells=[]
    for ref in published:
        for dataset,prefix,metrics in [('TuSimple','tusimple',['benign_acc','poisoned_acc','lda_asr','loa_asr','avg_asr']),
                                       ('CULane','culane',['benign_acc','poisoned_acc','lda_asr','lra_asr','loa_asr','avg_asr'])]:
            candidates=[r for r in verified if r['dataset']==dataset and r['model']==ref['model']]
            supporting='; '.join(f"{r['backbone']}: {r['value_percent']:.6f}% ({r['images']} images; {r['metric']})" for r in candidates)
            for metric in metrics:
                cells.append(dict(dataset=dataset,model=ref['model'],method=ref['attack'],metric=metric,
                    published_percent=ref[prefix+'_'+metric],local_table1_equivalent_percent='',
                    status='clean baseline verified; exact paper equivalence unestablished' if metric=='benign_acc' and dataset=='TuSimple'
                        else 'unmeasured Table 1 ACC; subset F1 only' if metric=='benign_acc'
                        else 'unmeasured; no independently verified equivalent attack experiment',
                    supporting_clean_evidence=supporting if metric=='benign_acc' else ''))
    with (out/'table1_status.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)
    lines=['# Table 1 reproduction status','',
           'Generated by `scripts/audit_table1.py`. Published values remain references. Blank local Table 1 cells mean unmeasured, never zero.',
           '', '## Independently re-scored clean evidence','',
           '| Dataset | Model | Backbone | Images | Re-scored % | Metric |', '|---|---|---|---:|---:|---|']
    for r in verified:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['backbone']} | {r['images']} | {r['value_percent']:.6f} | {r['metric']} |")
    lines += ['', '## Published target and local status','',
              '| Dataset | Model | Method | Published B ACC | Published P ACC | Published LDA | Published LRA | Published LOA | Published average | Local P ACC / ASR |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for ref in published:
        for ds,p in [('CULane','culane'),('TuSimple','tusimple')]:
            values=[ref.get(p+'_'+m,'not reported') for m in ['benign_acc','poisoned_acc','lda_asr','lra_asr','loa_asr','avg_asr']]
            lines.append('| '+' | '.join([ds,ref['model'],ref['attack'],*values,'unmeasured'])+' |')
    lines += ['', '## Limits','',f'CULane file coverage: `{json.dumps(coverage)}`.',
              '', 'Phase 1 is incomplete. No poisoned-model ACC or ASR is independently verified here. Full CULane Table 1 ACC is unavailable; official IoU/F1 is a different metric. Public upstream checkpoints are not established as the paper checkpoints. SCNN cross-version numerical parity remains unresolved. Re-scoring saved predictions validates scoring and evidence consistency; it does not rerun original model inference or prove training provenance.',
              '', 'All seven current TuSimple checkpoint hashes and all 2,782 prediction keys per run are checked. Existing source records are not modified.']
    (out/'table1_status.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Verified {len(verified)} clean runs; generated {len(cells)} published/status cells.',flush=True)


if __name__ == '__main__':
    main()
