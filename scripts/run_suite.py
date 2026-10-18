"""Run each architecture in a separate process, logging failures independently."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['robustness','full_test'],required=True)
    p.add_argument('--models',nargs='*'); p.add_argument('--config',default='configs/overnight.json'); p.add_argument('--output-category'); args=p.parse_args()
    spec=json.loads(Path(args.config).read_text()); dataset=spec[args.mode]
    category=args.output_category or args.mode
    folder=Path('results/overnight'); folder.mkdir(exist_ok=True)
    with (folder/'execution.jsonl').open('a') as events:
        for model in spec['models']:
            if args.models and model['name'] not in args.models: continue
            out=folder/category/model['name'].lower()
            if (out/'completed.json').exists(): continue
            command=[sys.executable,'scripts/benchmark.py','--model',model['name'],'--checkpoint',model['checkpoint'],
                '--device',spec['device'],'--root',dataset['root'],'--labels',dataset['labels'],'--expected-count',str(dataset['images']),'--output',str(out)]
            if 'config' in model: command+=['--config',model['config']]
            if 'backbone' in model: command+=['--backbone',model['backbone']]
            if args.mode=='robustness': command+=['--benign']
            event=dict(start=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command)
            events.write(json.dumps(dict(event,status='started'))+'\n'); events.flush()
            with (folder/f'{category}_{model["name"].lower()}.log').open('a') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            event.update(end=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=result.returncode)
            events.write(json.dumps(event)+'\n'); events.flush(); print(json.dumps(event),flush=True)

if __name__=='__main__': main()
