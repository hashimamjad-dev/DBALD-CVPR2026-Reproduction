"""Poison a TuSimple split: for each sample and each requested attack, place a trigger window (real
LaneATT gradient attention if --checkpoint is given, else poison_pipeline's uniform placeholder),
paste the placeholder trigger, transform the label, and write image+label under --output/<attack>/.

Still a placeholder pipeline: see poison_pipeline.py's and laneatt_heatmap.py's module docstrings for
exactly which two pieces (trigger realism, and -- without --checkpoint -- placement scoring) are
placeholders here. Local research use only; the output directory must not overlap the source dataset
(poison_pipeline.write_poisoned_sample enforces this) and is never committed to git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lane_repro.data import tusimple
from lane_repro.poison_pipeline import poison_sample,write_poisoned_sample

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='datasets/raw/tusimple_subset')
    p.add_argument('--labels',default='datasets/raw/tusimple_subset/labels.jsonl')
    p.add_argument('--output',required=True,help='Must not overlap --root')
    p.add_argument('--attacks',nargs='+',default=['LDA','LOA','LRA'],choices=['LDA','LOA','LRA'])
    p.add_argument('--checkpoint',help='LaneATTAdapter checkpoint; omit to use the uniform placeholder attention map')
    p.add_argument('--backbone',default='resnet18')
    p.add_argument('--device',default='cpu')
    p.add_argument('--limit',type=int,help='Poison only the first N samples (omit for the whole split)')
    p.add_argument('--trigger-size',type=int,default=40)
    p.add_argument('--stride',type=int,default=20)
    p.add_argument('--beta',type=float,default=60.,help='LOA horizontal offset, pixels (paper default)')
    p.add_argument('--alpha-degrees',type=float,default=9.,help='LRA rotation angle (paper default)')
    args=p.parse_args()

    samples=list(tusimple(args.root,args.labels))
    if args.limit: samples=samples[:args.limit]
    if not samples: raise ValueError('No samples selected')

    adapter=None
    if args.checkpoint:
        from lane_repro.models import LaneATTAdapter
        from lane_repro.laneatt_heatmap import laneatt_attention_map
        adapter=LaneATTAdapter(args.checkpoint,backbone=args.backbone,device=args.device)

    output=Path(args.output)
    manifest=dict(root=str(args.root),labels=str(args.labels),attacks=args.attacks,
        checkpoint=args.checkpoint,backbone=args.backbone if args.checkpoint else None,
        attention_source='real LaneATT gradient attention map' if adapter else 'uniform placeholder (poison_pipeline default)',
        trigger_size=args.trigger_size,stride=args.stride,beta=args.beta,alpha_degrees=args.alpha_degrees,
        placeholder_trigger=True,sample_count=len(samples),records=[])
    for attack in args.attacks:
        rows=[]
        for sample in samples:
            lanes_xy=[list(zip(xs,sample.h_samples)) for xs in sample.lanes] if adapter else None
            attention=laneatt_attention_map(adapter,sample.read(),lanes_xy,attack) if adapter else None
            result=poison_sample(sample,attack,trigger_size=args.trigger_size,stride=args.stride,
                beta=args.beta,alpha_degrees=args.alpha_degrees,attention_map=attention)
            record=write_poisoned_sample(result,Path(args.output)/attack.lower(),args.root)
            record['trigger_candidates']=result['trigger_candidates']
            record['image_sha256']=hashlib.sha256((Path(args.output)/attack.lower()/record['raw_file']).read_bytes()).hexdigest()
            rows.append(record)
            print(attack,record['raw_file'],record['trigger_window'],flush=True)
        (output/attack.lower()/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest['records'].append(dict(attack=attack,count=len(rows)))
    output.mkdir(parents=True,exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['records']))

if __name__=='__main__': main()
