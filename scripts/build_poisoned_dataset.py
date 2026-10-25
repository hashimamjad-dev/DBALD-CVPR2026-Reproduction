"""Poison a TuSimple split: for each sample and each requested attack, place a trigger window (real
LaneATT gradient attention if --checkpoint is given, else poison_pipeline's uniform placeholder),
render the trigger (DBALD's real UltraEdit generator if --render-trigger ultraedit, else the flat-
colour placeholder), transform the label, and write image+label under --output/<attack>/.

Local research use only; the output directory must not overlap the source dataset
(poison_pipeline.write_poisoned_sample enforces this) and is never committed to git. Before any real
--render-trigger ultraedit run, run `python src/lane_repro/ultraedit_trigger.py` first (its own
feasibility_check) to catch an OOM in seconds rather than partway through a batch.
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
    p.add_argument('--render-trigger',choices=['placeholder','ultraedit'],default='placeholder')
    p.add_argument('--trigger-kind',choices=['mud','cone'],default='mud',help='Only used with --render-trigger ultraedit')
    p.add_argument('--ultraedit-checkpoint',default='BleachNick/SD3_UltraEdit_w_mask')
    p.add_argument('--ultraedit-device',default='cuda')
    p.add_argument('--full-t5',action='store_true',help='Load all three SD3 text encoders; see ultraedit_trigger.py for the VRAM risk this avoids by default')
    p.add_argument('--cpu-offload',action='store_true',help='diffusers enable_model_cpu_offload(); slower, fits in less VRAM')
    p.add_argument('--edit-window',type=int,default=512)
    p.add_argument('--ultraedit-steps',type=int,default=50)
    args=p.parse_args()

    samples=list(tusimple(args.root,args.labels))
    if args.limit: samples=samples[:args.limit]
    if not samples: raise ValueError('No samples selected')

    adapter=None
    if args.checkpoint:
        from lane_repro.models import LaneATTAdapter
        from lane_repro.laneatt_heatmap import laneatt_attention_map
        adapter=LaneATTAdapter(args.checkpoint,backbone=args.backbone,device=args.device)

    render_trigger=trigger_prompt=None
    if args.render_trigger=='ultraedit':
        from lane_repro.ultraedit_trigger import UltraEditTrigger,MUD_PROMPT,CONE_PROMPT
        render_trigger=UltraEditTrigger(args.ultraedit_checkpoint,args.ultraedit_device,
            drop_t5=not args.full_t5,cpu_offload=args.cpu_offload,edit_window=args.edit_window,steps=args.ultraedit_steps)
        trigger_prompt=MUD_PROMPT if args.trigger_kind=='mud' else CONE_PROMPT

    output=Path(args.output)
    manifest=dict(root=str(args.root),labels=str(args.labels),attacks=args.attacks,
        checkpoint=args.checkpoint,backbone=args.backbone if args.checkpoint else None,
        attention_source='real LaneATT gradient attention map' if adapter else 'uniform placeholder (poison_pipeline default)',
        trigger_size=args.trigger_size,stride=args.stride,beta=args.beta,alpha_degrees=args.alpha_degrees,
        render_trigger=args.render_trigger,trigger_kind=args.trigger_kind if render_trigger else None,
        ultraedit_drop_t5=(not args.full_t5) if render_trigger else None,
        placeholder_trigger=render_trigger is None,sample_count=len(samples),records=[])
    for attack in args.attacks:
        rows=[]
        for sample in samples:
            lanes_xy=[list(zip(xs,sample.h_samples)) for xs in sample.lanes] if adapter else None
            attention=laneatt_attention_map(adapter,sample.read(),lanes_xy,attack) if adapter else None
            result=poison_sample(sample,attack,trigger_size=args.trigger_size,stride=args.stride,
                beta=args.beta,alpha_degrees=args.alpha_degrees,attention_map=attention,
                render_trigger=render_trigger,trigger_prompt=trigger_prompt)
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
