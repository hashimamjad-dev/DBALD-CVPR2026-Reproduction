"""Build a precise readiness inventory from verified receipts, without generating scores."""
import argparse
import csv
import json
import math
from pathlib import Path

from audit_table1 import ROOT, read, sha


def classify_cell(row):
    result=dict(row)
    if row['local_equivalent_percent']!='':
        result.update(measurement_status='VERIFIED',comparison_status='INDEPENDENT BENCHMARK BASELINE ONLY',
                      exact_paper_equivalence='UNVERIFIED')
    elif row['status']=='NOT COMPARABLE':
        result.update(measurement_status='VERIFIED SUPPORTING MEASUREMENT',comparison_status='NOT COMPARABLE',
                      exact_paper_equivalence='UNVERIFIED')
    else:
        result.update(measurement_status='MISSING',comparison_status='MISSING',exact_paper_equivalence='MISSING')
    return result


def preserved_records(accepted,replayed,expected):
    before={r['id']:r for r in accepted['verified']};after={r['id']:r for r in replayed['verified']}
    if len(before)!=expected or set(before)!=set(after) or replayed['rejected']:
        raise ValueError('Accepted evidence record set changed')
    for key in before:
        for field in ('model','dataset','metric','images','checkpoint_sha256','summary_sha256',
                      'receipt_sha256','predictions_sha256','config_sha256','provenance_sha256'):
            if before[key].get(field)!=after[key].get(field):raise ValueError('Evidence identity changed: '+key+'/'+field)
        if not all(math.isfinite(r[key]['value_percent']) for r in (before,after)) or abs(before[key]['value_percent']-after[key]['value_percent'])>1e-9:
            raise ValueError('Recomputed measurement changed: '+key)
    return sorted(before)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/table1_protocol_scnn_20261106')
    p.add_argument('--output',help='Optional fresh readiness output directory')
    a=p.parse_args();base=(ROOT/a.root).resolve()
    if not base.is_relative_to(ROOT/'results') or base.is_relative_to(ROOT/'results/reference'):raise ValueError('Use a results directory outside the immutable reference')
    out=(ROOT/a.output).resolve() if a.output else base/'readiness'
    if not out.is_relative_to(ROOT/'results') or out.is_relative_to(ROOT/'results/reference') or out.exists():raise ValueError('Use a fresh results report outside the immutable reference')
    clean=read(base/'clean_recheck/verification.json');unified=read(base/'unified_recheck/verification.json')
    accepted=ROOT/'results/table1_clean_20261106'
    clean_ids=preserved_records(read(accepted/'verified_clean_replay/verification.json'),clean,11)
    supporting_ids=preserved_records(read(accepted/'unified/verification.json'),unified,15)
    protocol=read(base/'protocol_witnesses/protocol_audit.json')
    cpu=read(base/'cpu_validation_final/receipt.json');drivers=read(base/'driver_validation/receipt.json')
    if len(cpu['checks'])!=8 or len(drivers['checks'])!=2 or not all(r['status']=='PASSED' for r in cpu['checks']+drivers['checks']):raise ValueError('CPU validation incomplete')
    cells=[classify_cell(r) for r in csv.DictReader((base/'unified_recheck/table1_comparison.csv').open())]
    if len(cells)!=220:raise ValueError('Incomplete Table 1 inventory')
    out.mkdir()
    with (out/'table1_readiness.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    report=dict(phase1_complete=False,accepted_clean_records_preserved=clean_ids,
                accepted_supporting_records_preserved=supporting_ids,
                baseline_only_cells=sum(r['measurement_status']=='VERIFIED' for r in cells),
                noncomparable_cells=sum(r['comparison_status']=='NOT COMPARABLE' for r in cells),
                missing_cells=sum(r['measurement_status']=='MISSING' for r in cells),
                culane_official_f1_matches_paper_acc=False,candidate_point_protocol_author_verified=False,
                scnn_r34_checkpoint_status='VERIFIED: both public checkpoints strictly loaded on CPU',
                scnn_r34_native_decode_status='VERIFIED: adapter/reference equality for one real image per dataset, MKLDNN on and off',
                scnn_r34_full_test_results='MISSING: no new full-test R34 inference has been run',
                cpu_checks_passed=len(cpu['checks']),cpu_driver_checks_passed=len(drivers['checks']),
                cpu_peak_private_mib=max(r['peak_private_mib'] for r in cpu['checks']+drivers['checks']),
                cpu_crash_root_cause='Not proven; isolated cases passed for both backends. Prior commit/RAM exhaustion is evidenced; do not claim all native crashes are explained.',
                gpu_work_started=False,
                partial_items=['SCNN R34 full-clean evaluations pending','Documented tiny poisoned-training run has no numerical receipt in this checkout'],
                prerequisites=['Author-confirmed CULane point sampling/matching/error/boundary/runtime rules',
                               'Full clean SCNN R34 evaluation, with GPU approval before extensive GPU use',
                               'Exact author checkpoint and training equivalence',
                               'Verified complete method/scenario experimental receipts before any poisoned ACC or ASR cells can be filled'],
                receipt_sha256={str(path.relative_to(ROOT)):sha(path) for path in
                    [base/'clean_recheck/verification.json',base/'unified_recheck/verification.json',
                     base/'protocol_witnesses/protocol_audit.json',base/'cpu_validation_final/receipt.json',base/'driver_validation/receipt.json']})
    comparison=base/'cross_version/comparison.json'
    report['r34_legacy_runtime_parity']=read(comparison) if comparison.exists() else {'status':'NOT EXECUTED'}
    (out/'readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['baseline_only_cells','noncomparable_cells','missing_cells','phase1_complete']}))


if __name__=='__main__':main()
