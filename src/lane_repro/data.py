"""Native-coordinate dataset readers; no silent rescaling or missing labels."""
from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np

@dataclass
class Sample:
    key: str
    image: Path
    lanes: list
    width: int
    height: int
    h_samples: list | None = None
    def read(self):
        image=cv2.imread(str(self.image))
        if image is None: raise FileNotFoundError(self.image)
        if image.shape[:2]!=(self.height,self.width): raise ValueError(f'{self.key}: incorrect native dimensions')
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def contained_path(root, relative):
    root=Path(root).resolve()
    path=(root/relative.replace('\\','/').lstrip('/')).resolve()
    if not path.is_relative_to(root): raise ValueError('Dataset path escapes root')
    return path

def tusimple(root, labels):
    seen=set()
    for line in Path(labels).read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); key=r['raw_file']
        if key in seen: raise ValueError('Duplicate sample')
        seen.add(key); ys=np.asarray(r['h_samples'],dtype=float)
        if len(ys)<2 or np.any(np.diff(ys)<=0) or np.any((ys<0)|(ys>=720)): raise ValueError('Invalid h_samples')
        for xs in r['lanes']:
            if len(xs)!=len(ys) or not np.isfinite(xs).all() or np.any(np.asarray(xs)>=1280): raise ValueError('Invalid lane')
        yield Sample(key,contained_path(root,key),r['lanes'],1280,720,r['h_samples'])

def culane(root,list_file):
    seen=set()
    for line in Path(list_file).read_text().splitlines():
        if not line.strip(): continue
        key=line.split()[0].lstrip('/')
        if key in seen: raise ValueError('Duplicate sample')
        seen.add(key); path=contained_path(root,key); lanes=[]
        for row in path.with_suffix('.lines.txt').read_text().splitlines():
            values=np.asarray([float(x) for x in row.split()])
            if len(values)%2: raise ValueError('Odd coordinate count')
            if len(values)<4: continue
            lane=values.reshape(-1,2)
            if not np.isfinite(lane).all(): raise ValueError('Nonfinite coordinates')
            lanes.append(lane[np.argsort(lane[:,1])].tolist())
        yield Sample(key,path,lanes,1640,590)
