"""Synthetic metric-conformance witnesses; never produces benchmark measurements."""
import argparse
import json
from pathlib import Path
import sys

from audit_table1 import ROOT, sha
from lane_repro.culane_point_acc import point_counts, PROTOCOL
from lane_repro.culane_official import score
from lane_repro.evaluation import official_tusimple
import numpy as np


def witnesses(out):
    anno=out/'annotations';pred=out/'predictions';anno.mkdir();pred.mkdir()
    listing=out/'list.txt';listing.write_text('/fixture.jpg\n')
    truth=np.array([[100.,200.],[100.,500.]])
    (anno/'fixture.lines.txt').write_text('100 200 100 500\n')
    result=[]
    for offset in (0,19,20,21):
        estimated=truth.copy();estimated[:,0]+=offset
        (pred/'fixture.lines.txt').write_text(' '.join(str(v) for v in estimated.flatten())+'\n')
        native=score(pred,anno,listing,out/f'offset_{offset}.official.txt')
        counts=point_counts([estimated],[truth])
        result.append(dict(offset_px=offset,candidate_correct=counts['correct'],candidate_total=counts['total'],
                           candidate_point_acc_percent=100*counts['correct']/counts['total'],
                           official_lane_f1_percent=native['f1_percent'],native_counts={k:native[k] for k in ('tp','fp','fn')}))
    ys=[200,500];truth_two=[truth,np.array([[105.,200.],[105.,500.]])]
    candidate=point_counts([truth],truth_two)
    official=official_tusimple().bench([[100,100]],[[100,100],[105,105]],ys,0)[0]*100
    matching=dict(candidate_one_to_one_percent=100*candidate['correct']/candidate['total'],
                  official_tusimple_best_per_gt_percent=official)
    return dict(scope='SYNTHETIC PROTOCOL WITNESSES; NEVER TABLE 1 MEASUREMENTS',real_benchmark_images=0,
        candidate_protocol=PROTOCOL,offset_witnesses=result,matching_witness=matching,
        metric_identity='CULane lane F1 is not the stated point ACC',
        paper_literal_definition='Ratio of correct lane points to total lane points with a 20-pixel condition',
        established=['point-based definition differs from lane-level F1','candidate uses pooled point counts'],
        unresolved=['CULane sampling grid / annotation conversion','lane matching and duplicate handling',
                    'horizontal versus distance/angle-adjusted error','exact boundary rule at 20 pixels',
                    'runtime and excess-prediction rules'],
        paper_equivalence_verified=False,
        sources=['https://arxiv.org/html/2508.15778v2#S5.SS1',
                 'https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py',
                 'https://github.com/XingangPan/SCNN/tree/master/tools/lane_evaluation'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'src/lane_repro/culane_point_acc.py',ROOT/'src/lane_repro/culane_official.py',ROOT/'third_party/tusimple-benchmark/evaluate/lane.py']})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);a=parser.parse_args()
    out=Path(a.output).resolve()
    if not out.is_relative_to(ROOT/'results') or out.is_relative_to(ROOT/'results/reference') or out.exists():raise ValueError('Use a new results directory outside the immutable reference')
    out.mkdir(parents=True)
    report=witnesses(out);(out/'protocol_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return 0


if __name__=='__main__':sys.exit(main())
