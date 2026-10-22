"""Point-matching lane metrics, per DBALD/TuSimple protocol (paper PDF p.6; docs/paper_analysis.md).

Metric = sum(C_i)/sum(S_i): C_i = predicted points within 20px of the matched ground-truth point
(evaluated only at the ground-truth y-samples), S_i = total ground-truth points. Used for clean ACC and
LOA/LRA ASR. LDA success is the fraction of ground-truth-present lanes predicted absent.

Pure coordinate comparison: no trigger placement, poisoning or label construction. A lane is
list[(x,y)] with x=None/negative for an absent point at that y-sample, matching TuSimple's h_samples
convention (also used for record['lanes'] in this repo's distillation manifests).
"""
THRESHOLD_PX=20.

def _valid_count(points):
    return sum(1 for x,_ in points if x is not None and x>=0)

def _valid_by_y(points):
    """(x,y) pairs with a valid x, keyed by y (the TuSimple h_samples row)."""
    return {y:x for x,y in points if x is not None and x>=0}

def point_accuracy(predicted,ground_truth,threshold=THRESHOLD_PX):
    """One lane: (correct, total) at the ground-truth's valid y-samples. Predicted value at a y-sample
    not present in predicted, or an invalid predicted value, counts as incorrect but still contributes
    to the denominator (a missed point is a failure, not an excluded one)."""
    truth=_valid_by_y(ground_truth)
    predicted_at=_valid_by_y(predicted)
    if not truth: raise ValueError('Ground-truth lane has no valid points')
    correct=sum(1 for y,gx in truth.items() if y in predicted_at and abs(predicted_at[y]-gx)<=threshold)
    return correct,len(truth)

def clip_accuracy(predicted_lanes,ground_truth_lanes,threshold=THRESHOLD_PX):
    """Metric over one image: predicted and ground-truth lanes paired by list position, as in a fixed
    max_lanes slot layout. Missing predicted lanes (fewer entries) count their ground truth as wholly
    incorrect, matching 'a vanished lane contributes zero correct points'."""
    if not ground_truth_lanes: raise ValueError('No ground-truth lanes')
    correct=total=0
    for i,truth in enumerate(ground_truth_lanes):
        prediction=predicted_lanes[i] if i<len(predicted_lanes) else []
        c,t=point_accuracy(prediction,truth,threshold)
        correct+=c; total+=t
    if total==0: raise ValueError('No valid ground-truth points across lanes')
    return correct/total

def dataset_accuracy(pairs,threshold=THRESHOLD_PX):
    """pairs: iterable of (predicted_lanes, ground_truth_lanes). Aggregate correct/total across the
    whole set, matching Metric = sum_i(C_i)/sum_i(S_i) rather than averaging per-image ratios."""
    correct=total=0
    for predicted_lanes,ground_truth_lanes in pairs:
        for i,truth in enumerate(ground_truth_lanes):
            prediction=predicted_lanes[i] if i<len(predicted_lanes) else []
            c,t=point_accuracy(prediction,truth,threshold)
            correct+=c; total+=t
    if total==0: raise ValueError('Empty or all-invalid dataset')
    return correct/total

def lda_disappearance_fraction(predicted_lanes,ground_truth_present):
    """ground_truth_present: list[bool], whether the lane existed before the attack (the attacker's
    LDA target is e_i=0 for every originally present lane). A predicted lane with fewer than 2 valid
    points, or no corresponding predicted entry, counts as disappeared."""
    present=[i for i,present in enumerate(ground_truth_present) if present]
    if not present: raise ValueError('No originally present lanes to attack')
    disappeared=sum(1 for i in present if i>=len(predicted_lanes) or _valid_count(predicted_lanes[i])<2)
    return disappeared/len(present)

def dataset_lda_asr(pairs):
    """pairs: iterable of (predicted_lanes, ground_truth_present)."""
    disappeared=present=0
    for predicted_lanes,ground_truth_present in pairs:
        for i,was_present in enumerate(ground_truth_present):
            if not was_present: continue
            present+=1
            if i>=len(predicted_lanes) or _valid_count(predicted_lanes[i])<2: disappeared+=1
    if present==0: raise ValueError('No originally present lanes across dataset')
    return disappeared/present
