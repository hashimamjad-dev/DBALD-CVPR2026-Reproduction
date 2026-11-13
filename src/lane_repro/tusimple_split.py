"""Load the pinned TuSimple train/validation partition (results/table1_split_20261111/tusimple_split.json).

Keeps the 363 validation frames out of poisoning and training. Pure CPU/file logic.
"""
import json
from pathlib import Path

DEFAULT_SPLIT='results/table1_split_20261111/tusimple_split.json'

def load_split(path=DEFAULT_SPLIT):
    split=json.loads(Path(path).read_text())
    train,val=set(split['train']),set(split['val'])
    if train&val: raise ValueError('Split file has overlapping train/val frames')
    if len(train)!=split['counts']['train'] or len(val)!=split['counts']['val']:
        raise ValueError('Split file counts do not match its frame lists')
    return split

def filter_samples(samples,part,path=DEFAULT_SPLIT):
    """Keep only samples whose raw_file belongs to `part` ('train' or 'val'). Any split frame
    absent from `samples` is tolerated (e.g. --limit); any sample absent from the split is an error."""
    if part not in ('train','val'): raise ValueError("part must be 'train' or 'val'")
    split=load_split(path); keep=set(split[part]); known=keep|set(split['val' if part=='train' else 'train'])
    out=[]
    for s in samples:
        if s.key not in known: raise ValueError(f'{s.key} is not in the pinned split')
        if s.key in keep: out.append(s)
    return out
