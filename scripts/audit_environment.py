import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]
def command(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=20)
    return (r.stdout or r.stderr).strip()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suffix',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    gpu=[]
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        gpu.append(dict(name=p.name,total_memory=p.total_memory,capability=list(torch.cuda.get_device_capability(i))))
    report=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),ram_total=psutil.virtual_memory().total,
        ram_available=psutil.virtual_memory().available,cpu_logical=psutil.cpu_count(),torch=torch.__version__,
        cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),cuda_arch_list=torch.cuda.get_arch_list(),gpu=gpu,
        nvidia_smi=command(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv']),
        cwd=str(ROOT),dataset_roots=[str(p) for p in (ROOT/'datasets').iterdir() if p.is_dir()])
    (ROOT/f'results/environment_{args.suffix}.json').write_text(json.dumps(report,indent=2))
    repos={}
    for folder in (ROOT/'third_party').iterdir():
        if (folder/'.git').is_dir():
            repos[folder.name]=dict(url=command(['git','-C',str(folder),'remote','get-url','origin']),
                commit=command(['git','-C',str(folder),'rev-parse','HEAD']),
                license_files=[p.name for p in folder.glob('*LICENSE*')])
    (ROOT/'results/repositories.json').write_text(json.dumps(repos,indent=2))
    (ROOT/f'requirements-lock-{args.suffix}.txt').write_text(command([sys.executable,'-m','pip','freeze'])+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
