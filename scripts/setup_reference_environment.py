"""Install isolated, pinned CPU reference environments; never alter GPU env."""
import argparse
from pathlib import Path
import subprocess
import sys
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]

def fetch(url,path):
    if path.is_file(): return
    part=path.with_suffix('.part')
    with requests.get(url,stream=True,timeout=(30,90)) as response:
        response.raise_for_status()
        with part.open('wb') as output:
            for chunk in response.iter_content(1024*1024): output.write(chunk)
    part.replace(path)

def main():
    p=argparse.ArgumentParser(); p.add_argument('architecture',choices=['LaneATT','LaneDet','ADNet']); args=p.parse_args()
    name=args.architecture.lower()
    dest=ROOT/'tools/python38' if name=='laneatt' else ROOT/'tools/reference_envs'/name
    archive=ROOT/'tmp/python-3.8.10.nupkg'
    archive.parent.mkdir(exist_ok=True)
    fetch('https://www.nuget.org/api/v2/package/python/3.8.10',archive)
    if not (dest/'python.exe').is_file():
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if not member.startswith('tools/') or member.endswith('/'): continue
                target=(dest/member[len('tools/'):]).resolve()
                if not target.is_relative_to(dest.resolve()): raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(member))
    interpreter=str(dest/'python.exe')
    pip_probe=subprocess.run([interpreter,'-m','pip','--version'],capture_output=True)
    if pip_probe.returncode:
        bootstrap=ROOT/'tmp/get-pip38.py'
        fetch('https://bootstrap.pypa.io/pip/3.8/get-pip.py',bootstrap)
        subprocess.run([interpreter,str(bootstrap),'--disable-pip-version-check','--no-warn-script-location'],check=True)
    requirement=ROOT/('requirements-laneatt-legacy.txt' if name=='laneatt' else f'requirements-{name}-reference.txt')
    subprocess.run([interpreter,'-m','pip','install','-r',str(requirement),'-f','https://download.pytorch.org/whl/torch_stable.html',
        '--disable-pip-version-check','--no-warn-script-location','--timeout','45','--retries','2'],check=True)
    frozen=subprocess.check_output([interpreter,'-m','pip','freeze'],text=True)
    (ROOT/f'requirements-lock-{name}-reference.txt').write_text(frozen)
    subprocess.run([interpreter,'-m','pip','check'],check=True)
    print(f'{args.architecture} reference ready: {interpreter}',flush=True)

if __name__=='__main__': main()
