import types
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]

def official_tusimple():
    """Load official functions unchanged; omit obsolete Python 2 CLI only."""
    path=ROOT/'third_party/tusimple-benchmark/evaluate/lane.py'
    source=path.read_text().split("if __name__ == '__main__':")[0]
    module=types.ModuleType('official_tusimple_evaluator')
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module.LaneEval

def geometry(pred, gt):
    """Additional one-to-one horizontal point metric, NOT TuSimple ACC or ASR.

    All valid GT points enter the accuracy denominator. Conditional MAE must be
    read together with coverage and extra/missing-lane counts.
    """
    g=np.asarray(gt,dtype=float)
    if len(gt)==0 or not (g>=0).any():
        return dict(point_accuracy_20px=None,mae_px=None,coverage=None,gt_points=0,correct_points=0,covered_points=0,extra_predictions=len(pred),unmatched_gt=len(gt))
    p=np.asarray(pred,dtype=float)
    total=int((g>=0).sum()); pairs=[]; correct=covered=0; errors=[]
    if len(pred):
        if p.shape[1]!=g.shape[1]: raise ValueError('Sampling mismatch')
        cost=np.zeros((len(gt),len(pred)))
        for i,a in enumerate(g):
            for j,b in enumerate(p):
                valid=a>=0; both=valid&(b>=0)
                cost[i,j]=np.where(both,np.abs(a-b),1000)[valid].mean() if valid.any() else 1e9
        rr,cc=linear_sum_assignment(cost)
        pairs=[(i,j) for i,j in zip(rr,cc) if cost[i,j]<1000]
        for i,j in pairs:
            valid=(g[i]>=0)&(p[j]>=0); error=np.abs(g[i,valid]-p[j,valid])
            correct+=int((error<20).sum()); covered+=len(error); errors.extend(error.tolist())
    return dict(point_accuracy_20px=correct/total,mae_px=float(np.mean(errors)) if errors else None,
                coverage=covered/total,gt_points=total,correct_points=correct,covered_points=covered,
                extra_predictions=len(pred)-len(pairs),unmatched_gt=len(gt)-len(pairs))
