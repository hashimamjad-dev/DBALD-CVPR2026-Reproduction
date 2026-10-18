"""Validated boundary around the author's C++ CULane evaluator."""
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from .data import contained_path

ROOT=Path(__file__).resolve().parents[2]

def prediction_text(lanes,ys,width=1640,height=590):
    ys=np.asarray(ys,dtype=float)
    lines=[]
    for lane in lanes:
        xs=np.asarray(lane(ys/height))
        if not np.isfinite(xs).all(): raise ValueError('Nonfinite prediction')
        keep=(xs>=0)&(xs<1)
        points=list(zip(xs[keep]*width,ys[keep]))
        # Author exporter traverses from bottom to top.
        points.sort(key=lambda p:p[1],reverse=True)
        if points: lines.append(' '.join(f'{x:.5f} {y:.5f}' for x,y in points))
    return '\n'.join(lines)

def score(pred_dir,anno_dir,list_file,output):
    keys=[line.strip().split()[0] for line in Path(list_file).read_text().splitlines() if line.strip()]
    if not keys or len(keys)!=len(set(keys)): raise ValueError('Empty or duplicate evaluation list')
    # Upstream silently interprets missing files as empty; reject those here.
    for root in (pred_dir,anno_dir):
        for key in keys:
            path=contained_path(root,key).with_suffix('.lines.txt')
            if not path.is_file(): raise FileNotFoundError(path)
            for line in path.read_text().splitlines():
                coords=np.asarray([float(v) for v in line.split()])
                if len(coords)%2 or len(coords)<2 or not np.isfinite(coords).all(): raise ValueError(f'Invalid lane: {path}')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env['OMP_NUM_THREADS']='4'; env['OPENBLAS_NUM_THREADS']='1'
    env['PATH']=str(ROOT/'tools/culane_runtime/ucrt64/bin')+';C:/msys64/ucrt64/bin;'+env['PATH']
    command=[str(ROOT/'tools/culane_evaluator/evaluate.exe'),'-a',str(Path(anno_dir).resolve())+'/',
        '-d',str(Path(pred_dir).resolve())+'/', '-i',str(Path(anno_dir).resolve())+'/',
        '-l',str(Path(list_file).resolve()),'-w','30','-t','0.5','-c','1640','-r','590','-f','1','-o',str(output.resolve())]
    proc=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
    match=re.search(r'tp:\s*(\d+)\s+fp:\s*(\d+)\s+fn:\s*(\d+)',output.read_text())
    if not match: raise ValueError('Missing official counts')
    tp,fp,fn=map(int,match.groups())
    return dict(metric='CULane official C++ lane IoU/F1',images=len(keys),tp=tp,fp=fp,fn=fn,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
        f1_percent=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,lane_width=30,iou_threshold=.5,
        compatibility='Original metric/spline/matching C++; OpenCV 4.13 instead of 2.4; optional GUI removed',
        acc=None,asr=None,command=command,stdout=proc.stdout,stderr=proc.stderr)
