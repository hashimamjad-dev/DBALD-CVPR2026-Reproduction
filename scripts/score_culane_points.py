"""Score saved clean CULane exports under an explicit, unconfirmed point protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lane_repro.culane_point_acc import PROTOCOL,parse_lanes,point_counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True); p.add_argument('--annotations',default='datasets/raw/culane')
    p.add_argument('--list',default='datasets/raw/culane/list/test.txt')
    p.add_argument('--expected-count',type=int,default=34680); p.add_argument('--output',required=True)
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise ValueError('Output exists')
    listing=Path(a.list); keys=[line.split()[0].lstrip('/') for line in listing.read_text().splitlines() if line.strip()]
    if len(keys)!=a.expected_count or len(keys)!=len(set(keys)): raise ValueError('Wrong or duplicate split')
    correct=total=0; digests=[hashlib.sha256(),hashlib.sha256()]
    for key in keys:
        lanes=[]
        for index,root in enumerate((Path(a.predictions).resolve(),Path(a.annotations).resolve())):
            path=(root/key).with_suffix('.lines.txt').resolve()
            if not path.is_relative_to(root): raise ValueError('Unsafe path')
            data=path.read_bytes(); digests[index].update(key.encode()+b'\0'+hashlib.sha256(data).digest())
            lanes.append(parse_lanes(data.decode()))
        counts=point_counts(*lanes); correct+=counts['correct']; total+=counts['total']
    report=dict(protocol=PROTOCOL,images=len(keys),full_test=len(keys)==34680,correct=correct,total=total,
                candidate_acc_percent=100*correct/total if total else None,
                predictions_tree_sha256=digests[0].hexdigest(),annotations_tree_sha256=digests[1].hexdigest(),
                list_sha256=hashlib.sha256(listing.read_bytes()).hexdigest(),
                status='NOT COMPARABLE',reason='Author sampling/matching/aggregation conventions not independently established')
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps(report))


if __name__=='__main__': main()
