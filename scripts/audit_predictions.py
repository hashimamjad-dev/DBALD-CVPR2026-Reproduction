"""Independently rescore saved TuSimple submissions using the upstream entry point."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from lane_repro.evaluation import official_tusimple

def main():
    reports=[]; evaluator=official_tusimple()
    for folder in sorted(Path('results/overnight').glob('*/*')):
        if not (folder/'completed.json').exists() or not (folder/'summary.json').exists(): continue
        summaries=json.loads((folder/'summary.json').read_text())
        if not summaries or summaries[0].get('dataset')!='TuSimple': continue
        labels=Path('datasets/raw/tusimple_test/labels.jsonl' if summaries[0]['scope']=='full_test' else 'datasets/raw/tusimple_subset/labels.jsonl')
        truth={r['raw_file']:r for r in map(json.loads,labels.read_text().splitlines())}
        records=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
        for summary in summaries:
            selected=[r for r in records if r['transformation']==summary['transformation']]
            keys=[r['raw_file'] for r in selected]
            assert len(keys)==len(set(keys))==len(truth) and set(keys)==set(truth)
            for r in selected:
                assert np.isfinite(r['run_time']) and r['run_time']>=0
                assert all(len(lane)==len(truth[r['raw_file']]['h_samples']) and np.isfinite(lane).all() for lane in r['lanes'])
            path=folder/f'official_submission_{summary["transformation"]}.jsonl'
            path.write_text(''.join(json.dumps({k:r[k] for k in ('raw_file','lanes','run_time')})+'\n' for r in selected))
            metrics=json.loads(evaluator.bench_one_submit(str(path),str(labels)))
            assert abs(metrics[0]['value']*100-summary['acc_percent'])<1e-9
            assert abs(metrics[1]['value']-summary['fp'])<1e-9 and abs(metrics[2]['value']-summary['fn'])<1e-9
            reports.append(dict(directory=str(folder),transformation=summary['transformation'],images=len(selected),
                aggregation_matches=True,official_metrics=metrics))
    Path('results/overnight/prediction_audit.json').write_text(json.dumps(reports,indent=2))
    print(f'Validated {len(reports)} completed model/condition submissions; all aggregate scores match upstream.')

if __name__=='__main__': main()
