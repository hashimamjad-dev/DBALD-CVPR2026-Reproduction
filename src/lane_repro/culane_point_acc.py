"""Explicit candidate interpretation of DBALD point ACC, NOT author-verified.

Uses existing annotation vertices, linear interpolation without extrapolation,
one-to-one matching maximizing correct points, strict <20 px horizontal error,
pooled point counts, no runtime gate or extra-prediction penalty. These choices
are explicit because the paper's formula alone does not specify them.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

PROTOCOL = dict(id='culane_annotation_vertices_linear_assignment_v1',
                threshold_px=20,comparison='strict_less_than',sampling='annotation_vertices',
                interpolation='linear_no_extrapolation',matching='one_to_one_max_correct_points',
                aggregation='pooled_points',runtime_gate=False,extra_prediction_penalty=False,
                author_verified=False,table1_comparable=False)


def parse_lanes(text):
    lanes=[]
    for line in text.splitlines():
        if not line.strip(): continue
        points=np.asarray([float(x) for x in line.split()],dtype=float)
        if points.size < 4 or points.size%2 or not np.isfinite(points).all():
            raise ValueError('Lane needs at least two finite xy pairs')
        points=points.reshape(-1,2)
        if len(set(points[:,1])) != len(points):
            raise ValueError('Duplicate y makes interpolation ambiguous')
        lanes.append(points[np.argsort(points[:,1])])
    return lanes


def point_counts(predictions,truth):
    total=sum(len(lane) for lane in truth)
    if not predictions or not truth: return dict(correct=0,total=total)
    counts=np.zeros((len(truth),len(predictions)),dtype=np.int64)
    for i,gt in enumerate(truth):
        for j,pred in enumerate(predictions):
            xs=np.interp(gt[:,1],pred[:,1],pred[:,0],left=np.nan,right=np.nan)
            counts[i,j]=np.count_nonzero(np.abs(xs-gt[:,0])<20)
    rows,cols=linear_sum_assignment(-counts)
    return dict(correct=int(counts[rows,cols].sum()),total=total)
