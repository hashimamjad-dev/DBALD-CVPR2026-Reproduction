"""Run available CULane checkpoints sequentially, preserving completed outputs."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

MODELS=[
 dict(name='LaneATT',backbone='resnet34',checkpoint='models/checkpoints/laneatt_official/experiments/laneatt_r34_culane/models/model_0015.pt'),
 dict(name='ADNet',backbone='resnet34',checkpoint='models/checkpoints/adnet_culane_res34.pth'),
 dict(name='SCNN',backbone='resnet50',pattern='scnn_r50_culane.pth',config='third_party/lanedet/configs/scnn/resnet50_culane.py'),
 dict(name='RESA',backbone='resnet34',pattern='resa_r34_culane.pth',config='third_party/lanedet/configs/resa/resa34_culane.py')]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--models',nargs='*'); args=p.parse_args()
    category='culane_smoke_r34' if args.smoke else 'culane_full'; root=Path('results/overnight')
    with (root/'execution.jsonl').open('a') as events:
        for spec in MODELS:
            if args.models and spec['name'] not in args.models: continue
            out=root/category/spec['name'].lower()
            if (out/'completed.json').exists(): continue
            if 'pattern' in spec:
                paths=list(Path('models/checkpoints').rglob(spec['pattern']))
                if len(paths)!=1: print(f'Unavailable checkpoint: {spec["name"]}',flush=True); continue
                checkpoint=str(paths[0])
            else: checkpoint=spec['checkpoint']
            if not Path(checkpoint).exists(): print(f'Unavailable checkpoint: {checkpoint}',flush=True); continue
            command=[sys.executable,'scripts/benchmark_culane.py','--model',spec['name'],'--checkpoint',checkpoint,
                '--backbone',spec['backbone'],'--output',str(out)]
            if 'config' in spec: command+=['--config',spec['config']]
            if args.smoke: command+=['--list','datasets/culane_smoke16.txt','--expected-count','16']
            event=dict(start=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command)
            events.write(json.dumps(dict(event,status='started'))+'\n'); events.flush()
            with (root/f'{category}_{spec["name"].lower()}.log').open('a') as log:
                r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            event.update(end=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode)
            events.write(json.dumps(event)+'\n'); events.flush(); print(json.dumps(event),flush=True)

if __name__=='__main__': main()
