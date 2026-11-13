"""Emit the LaneATT TuSimple poisoned-training plan (5 methods x 2 attacks) as reviewable JSON.

Runs NOTHING (no training, no diffusion, no GPU). Thin CLI over lane_repro.laneatt_harness.plan():
each cell carries its poisoned-mix build command, whether that mix step needs a GPU (dbald only),
and the config/label paths the harness would produce. Use it to review the matrix before any build.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lane_repro import laneatt_harness as H


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, help='Where to write the plan JSON')
    ap.add_argument('--output-root', default='<NEW_DIR>', help='Placeholder base dir for per-cell mixes')
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists(): raise SystemExit('refusing to overwrite an existing plan')
    out.parent.mkdir(parents=True, exist_ok=True)
    plan = H.plan(output_root=args.output_root, epochs=args.epochs)
    out.write_text(json.dumps(plan, indent=2))
    print(plan['cell_count'], 'LaneATT TuSimple cells ->', out)
    print('dbald cells need a GPU for the UltraEdit mix; baseline cells build on CPU.')


if __name__ == '__main__':
    main()
