"""Build author C++ scoring code with optional GUI removed, preserving metrics."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'third_party/SCNN/tools/lane_evaluation'; target=ROOT/'tools/culane_evaluator'
    target.mkdir(parents=True,exist_ok=True)
    original=(source/'src/evaluate.cpp').read_text(encoding='utf-8')
    text=original.replace('#include <opencv2/highgui/highgui.hpp>','')
    text=text[:text.index('\nvoid visualize(string &full_im_name',text.index('int main'))]
    text,n=re.subn(r'\s*if \(show\)\s*\{.*?waitKey\(0\);\s*\}', '',text,flags=re.S)
    if n!=1: raise ValueError('GUI patch did not match')
    (target/'evaluate_headless.cpp').write_text(text,encoding='utf-8')
    runtime=ROOT/'tools/culane_runtime/ucrt64'
    cmd=['C:/msys64/ucrt64/bin/g++.exe','-O2','-std=c++17','-fopenmp',
        '-I'+str(source/'include'),'-I'+str(runtime/'include/opencv4'),
        str(target/'evaluate_headless.cpp'),*[str(source/'src'/f) for f in ('counter.cpp','lane_compare.cpp','spline.cpp')],
        '-L'+str(runtime/'lib'),'-lopencv_core','-lopencv_imgproc','-o',str(target/'evaluate.exe')]
    subprocess.run(cmd,check=True)
    report=dict(source='XingangPan/SCNN/tools/lane_evaluation',changes=['Removed optional GUI visualization only'],
        opencv='4.13.0 (upstream used 2.4.x; rasterization version is a documented compatibility difference)',
        command=cmd,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'src').glob('*.cpp')})
    (target/'build_manifest.json').write_text(json.dumps(report,indent=2))
    env=os.environ.copy(); env['PATH']=str(runtime/'bin')+';C:/msys64/ucrt64/bin;'+env['PATH']
    subprocess.run([str(target/'evaluate.exe'),'-h'],env=env,check=True)

if __name__=='__main__': main()
