"""Emit the remaining Table 1 experiment matrix as a reviewable plan. Runs NOTHING (CPU or GPU).

Scenario model trainings = one per (dataset, model, method, attack): CULane 3 attacks x 20 rows and
TuSimple 2 attacks x 20 rows = 100, matching the 100 scenario-ASR cells. Which scenario's model
supplies the P ACC cell is NOT specified by the paper (assumption recorded in the plan: report each
scenario's clean ACC and decide the P ACC aggregation, e.g. mean, before launch).
CPU-feasible steps (poisoned-set construction on the pinned split) carry a concrete command;
GPU steps carry no command because no verified R34 poisoned-training driver exists yet, and the
project rule requires explicit user approval before any extensive GPU work.
"""
import argparse, json
from pathlib import Path

MODELS=['LaneATT','ADNet','SCNN','RESA']
METHODS=['badnets','blended','ld_attack','badlane','dbald']
ATTACKS={'TuSimple':['LDA','LOA'],'CULane':['LDA','LRA','LOA']}

def build(split='results/table1_split_20261111/tusimple_split.json'):
    runs=[]
    for dataset,attacks in ATTACKS.items():
        for model in MODELS:
            for method in METHODS:
                for attack in attacks:
                    rid=f'{dataset.lower()}_{model.lower()}_{method}_{attack.lower()}'
                    cpu=None
                    if dataset=='TuSimple':
                        cpu=(f'python scripts/mix_poisoned_dataset.py --root datasets/raw/tusimple_train --labels datasets/raw/tusimple_train/labels.jsonl '
                             f'--split-file {split} --split-part train --attack {attack} --method {method} --rate 0.10 --output <NEW_DIR>/{rid}')
                    runs.append(dict(id=rid,dataset=dataset,model=model,method=method,attack=attack,
                        cpu_step_poisoned_set=cpu,
                        cpu_step_blocker=None if cpu else 'CULane train/val images, lists and masks are absent locally (Drive quota); poisoning pipeline is TuSimple-format only',
                        dbald_trigger_blocker='UltraEdit trigger render needs a GPU (placeholder trigger is not paper-faithful)' if method=='dbald' else None,
                        gpu_steps=['train poisoned R34 model (driver not yet verified)','full-test clean ACC + ASR inference','independent re-score -> receipt for scripts/assemble_table1.py'],
                        approval_required_before_gpu=True))
    return dict(total_scenario_trainings=len(runs),assumptions=['P ACC aggregation across scenarios unspecified by the paper',
        'baseline trigger recipes are local reimplementations (src/lane_repro/baseline_attacks.py)'],runs=runs)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',required=True); a=ap.parse_args()
    plan=build(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise SystemExit('refusing to overwrite an existing plan')
    out.write_text(json.dumps(plan,indent=1)); print(plan['total_scenario_trainings'],'scenario trainings ->',out)
