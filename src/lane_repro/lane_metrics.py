"""Lane scoring, per DBALD/TuSimple protocol (paper PDF p.6).

CORRECTED after cross-checking against real LaneATT predictions (2026-10-23): the paper's displayed
formula, Metric = sum(C_i)/sum(S_i) with predicted lane i paired to ground-truth lane i by list
position, does NOT match "standard TuSimple protocol" in practice. The real official evaluator
(third_party/tusimple-benchmark/evaluate/lane.py, vendored unchanged) instead (a) matches each
ground-truth lane to whichever predicted lane scores highest against it, not by list position, since
a detector's output order has no guaranteed correspondence to TuSimple's label order; (b) uses an
angle-adjusted threshold (20px / cos(lane_angle)), not a flat 20px; (c) requires a per-lane fractional
accuracy >=0.85 to count as matched; (d) averages per-lane scores (capped to the 4 "best" lanes), not
a global point count. On the 16-image smoke subset this repo ships, the naive positional-pairing
functions below score 43.5%, the real official evaluator scores 96.7%, on the IDENTICAL predictions
(see test_lane_metrics.py's cross-check). asr_loa_lra below wraps the real, vendored evaluator and is
what this project's ASR numbers must use; the positional-pairing functions are kept only as a
documented, deliberately-labelled implementation of the paper's literal displayed formula, not as a
scorer anyone should trust for a real number.

LDA success is the fraction of ground-truth-present lanes predicted absent; the official evaluator has
no equivalent (TuSimple's native protocol does not score disappearance), so dataset_lda_asr below is
this project's own, and is not affected by the finding above.

A lane is list[(x,y)] with x=None/negative for an absent point at that y-sample, matching TuSimple's
h_samples convention (also used for record['lanes'] in this repo's distillation manifests). No trigger
placement, poisoning or label construction in this module.
"""
THRESHOLD_PX=20.

def _valid_count(points):
    return sum(1 for x,_ in points if x is not None and x>=0)

def _valid_by_y(points):
    """(x,y) pairs with a valid x, keyed by y (the TuSimple h_samples row)."""
    return {y:x for x,y in points if x is not None and x>=0}

def point_accuracy(predicted,ground_truth,threshold=THRESHOLD_PX):
    """NOT the official TuSimple/DBALD scorer — see the module docstring's 2026-10-23 correction.
    One lane: (correct, total) at the ground-truth's valid y-samples. Predicted value at a y-sample
    not present in predicted, or an invalid predicted value, counts as incorrect but still contributes
    to the denominator (a missed point is a failure, not an excluded one)."""
    truth=_valid_by_y(ground_truth)
    predicted_at=_valid_by_y(predicted)
    if not truth: raise ValueError('Ground-truth lane has no valid points')
    correct=sum(1 for y,gx in truth.items() if y in predicted_at and abs(predicted_at[y]-gx)<=threshold)
    return correct,len(truth)

def clip_accuracy(predicted_lanes,ground_truth_lanes,threshold=THRESHOLD_PX):
    """NOT the official TuSimple/DBALD scorer — see the module docstring's 2026-10-23 correction.
    Metric over one image: predicted and ground-truth lanes paired by list position, as in a fixed
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
    """NOT the official TuSimple/DBALD scorer — see the module docstring's 2026-10-23 correction.
    pairs: iterable of (predicted_lanes, ground_truth_lanes). Aggregate correct/total across the
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

def asr_loa_lra(predicted_lanes_xs,target_lanes_xs,h_samples,running_time=0.):
    """LOA/LRA ASR for one image, using the real official TuSimple evaluator (vendored unchanged at
    third_party/tusimple-benchmark/evaluate/lane.py), scored against the attack's TARGET label
    instead of the clean ground truth. This is the paper's own definition (p.6): 'ACC and ASR for
    LOA/LRA are defined as Metric = ...', i.e. the identical scoring function, with the reference
    label swapped from clean ground truth to the poisoned attack target (label_attacks.apply_attack's
    output). This is a thin wrapper, not a reimplementation, specifically to avoid the mismatch found
    in point_accuracy/clip_accuracy/dataset_accuracy above.

    predicted_lanes_xs, target_lanes_xs: list of per-lane x-coordinate lists, one x per h_samples
    entry, -2 for invalid/absent (TuSimple's native format; see src/lane_repro/data.py). running_time
    in ms; the official evaluator returns 0 accuracy outright above 200ms (its real-time gate) or if
    len(target)+2 < len(predicted), so pass 0 to score geometry only, matching this project's
    'tusimple_acc_without_runtime_gate' convention (scripts/evaluate.py).
    Returns the accuracy component only (float in [0,1]); the evaluator's fp/fn are not ASR."""
    from .evaluation import official_tusimple
    accuracy,_,_=official_tusimple().bench(predicted_lanes_xs,target_lanes_xs,h_samples,running_time)
    return float(accuracy)

def dataset_asr_loa_lra(triples,running_time=0.):
    """triples: iterable of (predicted_lanes_xs, target_lanes_xs, h_samples). Mean of asr_loa_lra
    over the dataset, matching the official evaluator's own per-image averaging (bench_one_submit)."""
    triples=list(triples)
    if not triples: raise ValueError('Empty dataset')
    return sum(asr_loa_lra(p,t,h,running_time) for p,t,h in triples)/len(triples)
