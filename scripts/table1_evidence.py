"""Read-only evidence imports. Never constructs targets, trains, or runs attacks."""
import hashlib
import json
import math
from pathlib import Path

from audit_table1 import ROOT, sha, read, jsonlines
from lane_repro.evaluation import official_tusimple


def check(ok, message):
    if not ok:
        raise ValueError(message)


def artifact(spec):
    path = (ROOT / spec['path']).resolve()
    check(path.is_relative_to(ROOT), 'Evidence must remain inside repository')
    check(sha(path) == spec['sha256'], 'Artifact hash mismatch: '+spec['path'])
    return path


def official_culane_points(key):
    from lane_repro.culane_point_acc import parse_lanes
    path=(ROOT/'datasets/raw/culane'/key).with_suffix('.lines.txt')
    return sorted(sorted(map(tuple,lane.tolist())) for lane in parse_lanes(path.read_text()))


def import_cache(spec):
    """Reuse a pinned local verification receipt, not an arbitrary numeric record.

    A hash binds a receipt to the reviewed manifest; it is not a signature or
    independent proof of experimental execution. Raw evidence replay uses verify.
    """
    path = artifact(spec)
    records = []
    for index, row in enumerate(read(path)['verified']):
        if spec.get('dataset') and row['dataset'] != spec['dataset']:
            continue
        expected = 2782 if row['dataset'] == 'TuSimple' else 34680
        check(row['images'] == expected, 'Cached subset is not full evaluation')
        folder = ROOT / row['evidence']
        check(sha(folder/'summary.json') == row['summary_sha256'], 'Cached summary drift')
        if row.get('predictions_sha256'):
            check(sha(folder/'predictions.jsonl') == row['predictions_sha256'], 'Cached prediction drift')
        run = read(folder/'run_config.json')
        provenance = read(folder/'model_provenance.json')
        check(run.get('model',row['model'])==row['model'], 'Cached model drift')
        check(not row.get('backbone') or row['backbone']==run['backbone'], 'Cached backbone drift')
        check(sha(ROOT/provenance['checkpoint']) == row['checkpoint_sha256'] == run['checkpoint_sha256'], 'Cached checkpoint drift')
        metric = row.get('metric_definition', row['metric'])
        check(math.isfinite(row['value_percent']) and 0 <= row['value_percent'] <= 100, 'Invalid cached value')
        records.append(dict(row, id=row.get('id',f"cached_{index}_{row['model']}"),
            metric='benign_acc', metric_definition=metric, backbone=run['backbone'],
            status='VERIFIED', verification_mode='pinned_prior_verification',
            receipt_sha256=spec['sha256'],
            comparison_basis='official_tusimple' if metric == 'official TuSimple ACC' else 'unconfirmed',
            reason='Prior verified clean measurement; independent training/configuration; CULane F1 is not paper ACC'))
    return records


def import_external(record):
    """Recalculate supplied static predictions under an explicit scoring protocol.

    Provenance hashes establish file identity, not authenticity of training claims.
    All scenario protocols remain NOT COMPARABLE to the original paper.
    """
    from evaluate_table1 import MODELS, METHODS
    check(record['model'] in MODELS and record['dataset'] in ('TuSimple','CULane'), 'Unknown model/dataset')
    metric = record['metric']
    check(metric in ('benign_acc','poisoned_acc','lda_asr','loa_asr','lra_asr'), 'Unsupported evidence metric')
    check(not (record['dataset']=='TuSimple' and metric=='lra_asr'), 'TuSimple has no Table 1 LRA cell')
    if metric != 'benign_acc':
        check(record['method'] in METHODS, 'Unknown method')
    files = {key:artifact(value) for key,value in record['artifacts'].items()}
    check({'predictions','annotations','split','checkpoint','config','provenance'} <= files.keys(), 'Missing provenance artifacts')
    provenance = read(files['provenance'])
    for key in ('model','dataset','metric','experiment_id','protocol'):
        check(provenance[key] == record[key], 'Provenance mismatch: '+key)
    check(provenance['checkpoint_sha256'] == sha(files['checkpoint']), 'Provenance checkpoint mismatch')
    check(provenance['config_sha256'] == sha(files['config']), 'Provenance config mismatch')
    check(provenance['evaluation_input'] == ('clean' if metric.endswith('acc') else 'supplied_scenario'), 'Wrong evaluation input')
    if metric != 'benign_acc':
        check(provenance['method'] == record['method'], 'Provenance method mismatch')
    keys = read(files['split'])
    expected = 2782 if record['dataset']=='TuSimple' else 34680
    check(isinstance(keys,list) and len(keys)==len(set(keys))==expected, 'Full unique split required')
    official_truth=None
    if record['dataset']=='TuSimple':
        official_rows=jsonlines(ROOT/'datasets/raw/tusimple_test/labels.jsonl')
        official_truth={r['raw_file']:r for r in official_rows}
        check(len(official_rows)==len(official_truth)==expected and set(keys)==set(official_truth), 'Official split mismatch')
    else:
        official_keys=[line.split()[0].lstrip('/') for line in (ROOT/'datasets/raw/culane/list/test.txt').read_text().splitlines() if line.strip()]
        check(len(official_keys)==len(set(official_keys))==expected and set(keys)==set(official_keys), 'Official split mismatch')
    predictions, annotations = jsonlines(files['predictions']), jsonlines(files['annotations'])
    for rows in (predictions, annotations):
        ids = [row['raw_file'] for row in rows]
        check(len(ids)==len(set(ids))==expected and set(ids)==set(keys), 'Prediction/annotation coverage mismatch')
    truth = {row['raw_file']:row for row in annotations}
    protocol = record['protocol']
    values, disappeared, present, correct, total = [], 0, 0, 0, 0
    for prediction in predictions:
        target = truth[prediction['raw_file']]
        ys = target['h_samples']
        if official_truth is not None:
            official=official_truth[prediction['raw_file']]
            check(ys==official['h_samples'], 'Official sampling grid mismatch')
            if metric.endswith('acc'):
                check(target['lanes']==official['lanes'], 'Clean annotations differ from official labels')
        elif metric.endswith('acc'):
            supplied=sorted(sorted((x,y) for x,y in zip(lane,ys) if x>=0) for lane in target['lanes'])
            check(supplied==official_culane_points(prediction['raw_file']), 'Clean annotations differ from official CULane vertices')
        check(len(ys)>1 and all(math.isfinite(y) for y in ys) and all(a<b for a,b in zip(ys,ys[1:])), 'Invalid sampling grid')
        check(prediction['h_samples']==ys, 'Sampling grid mismatch')
        for lanes in (prediction['lanes'],target['lanes']):
            check(all(len(lane)==len(ys) and all(math.isfinite(x) for x in lane) for lane in lanes), 'Invalid lane coordinates')
        if protocol == 'tusimple_official_runtime_v1':
            check(record['dataset']=='TuSimple' and metric in ('benign_acc','poisoned_acc','loa_asr'), 'Protocol/metric mismatch')
            runtime = prediction['run_time']
            check(math.isfinite(runtime) and runtime>=0, 'Invalid runtime')
            check(bool(target['lanes']) and all(sum(x>=0 for x in lane)>=2 for lane in target['lanes']), 'Invalid target lanes')
            values.append(float(official_tusimple().bench(prediction['lanes'],target['lanes'],ys,runtime)[0]))
        elif protocol == 'culane_annotation_vertices_linear_assignment_v1':
            check(record['dataset']=='CULane' and metric in ('benign_acc','poisoned_acc','loa_asr','lra_asr'), 'Protocol/metric mismatch')
            import numpy as np
            from lane_repro.culane_point_acc import point_counts
            def points(lanes):
                result=[]
                for lane in lanes:
                    pairs=[(x,y) for x,y in zip(lane,ys) if x>=0]
                    check(len(pairs)>=2, 'Candidate lane requires at least two points')
                    result.append(np.asarray(pairs,dtype=float))
                return result
            counts=point_counts(points(prediction['lanes']),points(target['lanes']))
            correct+=counts['correct']; total+=counts['total']
        elif protocol == 'fixed_slot_disappearance_v1':
            check(metric=='lda_asr', 'Protocol/metric mismatch')
            check(provenance.get('lane_slot_semantics')=='stable_correspondence', 'Stable lane slots required')
            check(len(prediction['lanes'])==len(target['lanes']), 'Lane slot coverage mismatch')
            for predicted, original in zip(prediction['lanes'],target['lanes']):
                if sum(x>=0 for x in original)>=2:
                    present += 1
                    disappeared += sum(x>=0 for x in predicted)<2
        else:
            raise ValueError('Unsupported scoring protocol; leave NOT MEASURED')
    check(bool(values) or present>0 or total>0, 'Empty scoring denominator')
    value = 100*(sum(values)/len(values) if values else correct/total if total else disappeared/present)
    if 'claimed_percent' in record:
        check(math.isfinite(record['claimed_percent']) and abs(value-record['claimed_percent'])<1e-8, 'Claimed result mismatch')
    return dict(record, status='VERIFIED', images=expected, value_percent=value,
        comparison_basis='unconfirmed', metric_definition=protocol,
        verification_mode='recomputed_static_predictions',
        reason='Supplied predictions rescored; experiment authenticity and paper protocol equivalence not established')


def import_point_candidate(spec):
    """Reuse pinned full-split candidate counts and bind them to prediction exports."""
    from lane_repro.culane_point_acc import PROTOCOL
    path=artifact(spec)
    row=read(path)
    check(row['protocol']==PROTOCOL and row['images']==34680 and row['full_test'], 'Candidate protocol/coverage mismatch')
    check(0<=row['correct']<=row['total'] and row['total']>0, 'Invalid candidate counts')
    check(abs(100*row['correct']/row['total']-row['candidate_acc_percent'])<1e-9, 'Candidate arithmetic mismatch')
    folder=path.parent
    check(sha(folder/'run_config.json')==spec['run_config_sha256'], 'Candidate configuration drift')
    check(sha(folder/'predictions.jsonl')==spec['predictions_sha256'], 'Candidate prediction index drift')
    config=read(folder/'run_config.json')
    check(config['list_sha256']==row['list_sha256'], 'Candidate split mismatch')
    predictions=jsonlines(folder/'predictions.jsonl')
    keys=[r['raw_file'] for r in predictions]
    check(len(keys)==len(set(keys))==34680, 'Candidate prediction coverage mismatch')
    digest=hashlib.sha256()
    for prediction in predictions:
        digest.update(prediction['raw_file'].encode()+b'\0'+bytes.fromhex(prediction['prediction_sha256']))
    check(digest.hexdigest()==row['predictions_tree_sha256'], 'Candidate export hash mismatch')
    return dict(id='candidate_'+config['model'],model=config['model'],dataset='CULane',
        backbone=config['backbone'],metric='benign_acc',images=34680,status='VERIFIED',
        value_percent=row['candidate_acc_percent'],receipt_sha256=spec['sha256'],
        comparison_basis='unconfirmed',metric_definition=PROTOCOL['id'],
        reason='Pinned candidate point scoring; sampling/matching/aggregation are not established as paper-equivalent')


def derive_averages(records):
    """Never combine scenarios from different experiment series/protocol families."""
    groups = {}
    for row in records:
        if row['metric'] not in ('lda_asr','lra_asr','loa_asr') or row.get('status')!='VERIFIED':
            continue
        key = tuple(row.get(k) for k in ('model','dataset','method','experiment_id','protocol_family'))
        if None in key:
            continue
        group = groups.setdefault(key,{})
        check(row['metric'] not in group, 'Duplicate scenario in experiment series')
        group[row['metric']] = row
    derived = []
    for key, group in groups.items():
        needed = {'lda_asr','loa_asr'} | ({'lra_asr'} if key[1]=='CULane' else set())
        if set(group)!=needed:
            continue
        check(all(math.isfinite(r['value_percent']) and 0<=r['value_percent']<=100 for r in group.values()), 'Invalid scenario percentage')
        row = dict(next(iter(group.values())))
        row.update(id='average_'+row['id'],metric='avg_asr',
            value_percent=sum(r['value_percent'] for r in group.values())/len(group),
            metric_definition='Arithmetic mean of '+', '.join(sorted(needed)),
            derived_from=[group[k]['id'] for k in sorted(needed)],
            reason='Arithmetic mean of all verified scenarios in one declared experiment series; paper equivalence unconfirmed')
        derived.append(row)
    return derived
