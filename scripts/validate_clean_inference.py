"""Check strict loading and one clean image against saved predictions for each pair.

This is a compatibility smoke check, not a replacement for full-test accuracy.
Existing adapters, preprocessing and saved checkpoints are used without training.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

from audit_table1 import ROOT, read, sha
from replay_clean_table1 import clean_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='configs/table1_clean_replay.json')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--dataset',choices=['TuSimple','CULane'],help='Optionally check only one dataset')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if not output.is_relative_to(ROOT/'results') or output.exists():
        raise ValueError('Use a new results receipt')
    import numpy as np
    import torch
    from lane_repro.data import tusimple, culane
    from lane_repro.models import LaneATTAdapter, ADNetAdapter, SegmentationAdapter, sample_predictions
    from lane_repro.culane_official import prediction_text

    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    specs=clean_manifest(args.manifest)['records']
    report=dict(scope='one clean image per model/dataset; compatibility only, not accuracy',seed=0,
                device=args.device,dataset_filter=args.dataset,python=sys.version,torch=torch.__version__,numpy=np.__version__,
                manifest_sha256=sha(args.manifest),source_sha256=sha(__file__),checks=[])
    if args.device=='cuda': report['gpu']=torch.cuda.get_device_name(0)
    output.parent.mkdir(parents=True,exist_ok=True)
    for model in ('LaneATT','ADNet','SCNN','RESA'):
        for dataset in ('TuSimple','CULane'):
            if args.dataset and dataset!=args.dataset: continue
            candidates=[r for r in specs if r['model']==model and r['dataset']==dataset]
            candidates.sort(key=lambda r:read(ROOT/r['evidence']/'run_config.json')['backbone']!='resnet34')
            record=candidates[0]
            folder=ROOT/record['evidence'];run=read(folder/'run_config.json');prov=read(folder/'model_provenance.json')
            checkpoint=ROOT/prov['checkpoint']
            check=dict(model=model,dataset=dataset,backbone=run['backbone'],evidence=record['evidence'],
                       checkpoint_sha256=sha(checkpoint),passed=False)
            try:
                if check['checkpoint_sha256']!=run['checkpoint_sha256']:
                    raise ValueError('Saved checkpoint drift')
                if model=='LaneATT':adapter=LaneATTAdapter(checkpoint,dataset.lower(),run['backbone'],args.device)
                elif model=='ADNet':adapter=ADNetAdapter(checkpoint,args.device,dataset.lower(),run['backbone'])
                else:adapter=SegmentationAdapter(model,checkpoint,prov['config'],args.device)
                if dataset=='TuSimple':
                    sample=next(tusimple(ROOT/'datasets/raw/tusimple_test',ROOT/record['annotations']))
                else:sample=next(culane(ROOT/record['data_root'],ROOT/record['annotations']))
                lanes=adapter.predict(sample.read())
                if dataset=='TuSimple':
                    xs=sample_predictions(lanes,sample.h_samples,sample.width,sample.height)
                    # Match benchmark.py's saved official submission encoding.
                    actual=[[int(x) if x>=0 else -2 for x in lane] for lane in xs]
                    with (folder/'predictions.jsonl').open() as stream:
                        saved=next(r['lanes'] for line in stream if (r:=json.loads(line))['raw_file']==sample.key)
                else:
                    ys=list(range(590)) if model=='LaneATT' else list(adapter.cfg.sample_y)
                    text=prediction_text(lanes,ys)
                    actual=[[float(x) for x in line.split()] for line in text.splitlines()]
                    saved=[[float(x) for x in line.split()] for line in (folder/'lanes'/sample.key).with_suffix('.lines.txt').read_text().splitlines()]
                same=len(actual)==len(saved) and all(len(a)==len(b) for a,b in zip(actual,saved))
                difference=max((float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(actual,saved) if a),default=0.) if same else None
                check.update(raw_file=sample.key,predictions=actual,saved_lane_count=len(saved),
                             strict_state_dict=adapter.provenance()['strict_state_dict'],
                             max_pixel_difference=difference,tolerance_px=.2,
                             passed=same and difference is not None and np.isfinite(difference) and difference<=.2)
                del adapter
            except Exception as error:
                check['error']=str(error)
            report['checks'].append(check)
            output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in check.items() if k!='predictions'}),flush=True)
            gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
    return 0 if all(r['passed'] for r in report['checks']) else 1


if __name__=='__main__':
    sys.exit(main())
