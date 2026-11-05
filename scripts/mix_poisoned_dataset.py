"""Build the actual mixed training set DBALD trains on: a fixed, deterministic fraction of a split
poisoned (default 10%, the paper's rate), the rest copied through unchanged -- one combined output
directory + labels.jsonl, ready to train a detector on directly. build_poisoned_dataset.py instead
poisons every sample it's given (useful for inspecting trigger placement/ASR on an all-poisoned set,
not for training): this script is the one that actually produces a training-ready mix.

Selection is sha256(salt + raw_file)-ranked, the same deterministic-selection convention already
used elsewhere in this repo (e.g. scripts/prepare_distillation_data.py), so a given --salt always
selects the same subset, independent of iteration order. Local research use only; --output must not
overlap --root (enforced the same way as poison_pipeline.write_poisoned_sample) and is never
committed to git.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
from lane_repro.data import tusimple,contained_path
from lane_repro.poison_pipeline import poison_sample,write_poisoned_sample

def select_poison_set(samples,rate,salt):
    if not 0<rate<=1: raise ValueError('rate must be in (0, 1]')
    ordered=sorted(samples,key=lambda s:hashlib.sha256((salt+s.key).encode()).hexdigest())
    return set(s.key for s in ordered[:round(rate*len(samples))])

def copy_clean_sample(sample,out_root,source_root):
    out_root=Path(out_root).resolve(); source_root=Path(source_root).resolve()
    if out_root==source_root or out_root.is_relative_to(source_root) or source_root.is_relative_to(out_root):
        raise ValueError('Output directory must not overlap the clean dataset directory')
    path=contained_path(out_root,sample.key)
    path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(sample.image,path)
    return dict(raw_file=sample.key,h_samples=sample.h_samples,lanes=sample.lanes,attack=None,poisoned=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='datasets/raw/tusimple_train')
    p.add_argument('--labels',default='datasets/raw/tusimple_train/labels.jsonl')
    p.add_argument('--output',required=True,help='Must not overlap --root')
    p.add_argument('--attack',default='LOA',choices=['LDA','LOA','LRA'])
    p.add_argument('--rate',type=float,default=.10,help='Fraction poisoned; DBALD default is 10%%')
    p.add_argument('--salt',default='dbald-poison-v1',help='Changes the selection; keep fixed for a reproducible split across reruns')
    p.add_argument('--limit',type=int,help='Use only the first N samples (by the same deterministic order), for a quick trial run')
    p.add_argument('--checkpoint',help='LaneATTAdapter checkpoint for the real gradient attention map; omit for the uniform placeholder')
    p.add_argument('--backbone',default='resnet18')
    p.add_argument('--device',default='cpu')
    p.add_argument('--trigger-size',type=int,default=40)
    p.add_argument('--stride',type=int,default=20)
    p.add_argument('--beta',type=float,default=60.)
    p.add_argument('--alpha-degrees',type=float,default=9.)
    p.add_argument('--render-trigger',choices=['placeholder','ultraedit'],default='placeholder')
    p.add_argument('--trigger-kind',choices=['mud','cone'],default='mud')
    p.add_argument('--ultraedit-checkpoint',default='BleachNick/SD3_UltraEdit_w_mask')
    p.add_argument('--ultraedit-device',default='cuda')
    p.add_argument('--full-t5',action='store_true')
    p.add_argument('--cpu-offload',action='store_true')
    args=p.parse_args()

    samples=list(tusimple(args.root,args.labels))
    if args.limit: samples=sorted(samples,key=lambda s:hashlib.sha256((args.salt+s.key).encode()).hexdigest())[:args.limit]
    if not samples: raise ValueError('No samples selected')
    poison_keys=select_poison_set(samples,args.rate,args.salt)

    adapter=None
    if args.checkpoint:
        from lane_repro.models import LaneATTAdapter
        adapter=LaneATTAdapter(args.checkpoint,backbone=args.backbone,device=args.device)
    render_trigger=trigger_prompt=None
    if args.render_trigger=='ultraedit':
        from lane_repro.ultraedit_trigger import UltraEditTrigger,MUD_PROMPT,CONE_PROMPT
        render_trigger=UltraEditTrigger(args.ultraedit_checkpoint,args.ultraedit_device,
            drop_t5=not args.full_t5,cpu_offload=args.cpu_offload)
        trigger_prompt=MUD_PROMPT if args.trigger_kind=='mud' else CONE_PROMPT

    rows=[]
    for n,sample in enumerate(samples,1):
        if sample.key in poison_keys:
            from lane_repro.laneatt_heatmap import laneatt_attention_map
            lanes_xy=[list(zip(xs,sample.h_samples)) for xs in sample.lanes] if adapter else None
            attention=laneatt_attention_map(adapter,sample.read(),lanes_xy,args.attack) if adapter else None
            result=poison_sample(sample,args.attack,trigger_size=args.trigger_size,stride=args.stride,
                beta=args.beta,alpha_degrees=args.alpha_degrees,attention_map=attention,
                render_trigger=render_trigger,trigger_prompt=trigger_prompt)
            record=write_poisoned_sample(result,args.output,args.root)
            record['poisoned']=True
        else:
            record=copy_clean_sample(sample,args.output,args.root)
        rows.append(record)
        if n%50==0 or n==len(samples): print(f'{n}/{len(samples)} ({sum(r["poisoned"] for r in rows)} poisoned)',flush=True)

    output=Path(args.output)
    (output/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'manifest.json').write_text(json.dumps(dict(
        root=str(args.root),labels=str(args.labels),attack=args.attack,rate=args.rate,salt=args.salt,
        sample_count=len(samples),poisoned_count=sum(r['poisoned'] for r in rows),
        checkpoint=args.checkpoint,attention_source='real' if adapter else 'uniform placeholder',
        render_trigger=args.render_trigger,trigger_kind=args.trigger_kind if render_trigger else None,
        placeholder_trigger=render_trigger is None),indent=2))
    print('Done:',sum(r['poisoned'] for r in rows),'/',len(samples),'poisoned, written to',output)

if __name__=='__main__': main()
