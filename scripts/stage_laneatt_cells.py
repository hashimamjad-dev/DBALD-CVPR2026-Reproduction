"""Materialise the 10 LaneATT x TuSimple cells as ready-to-fire config + runbook -- but RUN NOTHING.

This is the pure setup/config step PLAN.md allows now: for each of the 5 methods x 2 attacks (LDA,
LOA) it writes the bound LaneATT ResNet-34 training config to disk and records, in one runbook, the
exact poisoned-mix build command and the training command for that cell, with the GPU/Kaggle gates
flagged. The moment the user confirms GPU/Kaggle, the sequence is: (1) run each cell's mix command
(baselines on CPU; the dbald cell needs a GPU for the UltraEdit render), (2) write our pinned split
into the mix via lane_repro.laneatt_harness.write_tusimple_labels, (3) launch each training command.

It does NOT build any poisoned mix and does NOT train: no diffusion, no GPU, no Kaggle here. The
configs it writes point their train/val roots at `<mix-root-base>/<cell_id>`, which those mixes will
populate later -- the config text is valid now; the image/label files arrive at mix time.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lane_repro import laneatt_harness as H

LANEATT_MAIN = 'third_party/LaneATT/main.py'


def train_command(cell_id, config_path):
    """LaneATT's own trainer (GPU-only; NMS is CUDA-only -- main.py refuses --cpu). Launched from the
    repo root so the config's repo-relative dataset roots resolve. Staged, not run."""
    return ['python', LANEATT_MAIN, 'train', '--exp_name', cell_id, '--cfg', str(config_path)]


def stage(stage_dir, mix_root_base, split_file, epochs, test_root, train_size):
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, attack in H.cells():
        cid = H.cell_id(method, attack)
        mix_root = f'{mix_root_base}/{cid}'
        cell_dir = stage_dir / cid
        config_path = cell_dir / f'laneatt_r34_{cid}.yml'
        H.write_config(method, attack, mix_root, config_path, test_root=test_root,
                       epochs=epochs, train_size=train_size)
        config_str = str(config_path).replace('\\', '/')         # one normalized path for row + command
        mix_cmd, needs_gpu = H.mix_command(method, attack, mix_root, split_file=split_file)
        rows.append(dict(
            id=cid, method=method, attack=attack, mix_root=mix_root,
            config_path=config_str,
            train_label_file=f'{mix_root}/{H.TRAIN_ANNO}',
            val_label_file=f'{mix_root}/{H.VAL_ANNO}',
            mix_command=' '.join(mix_cmd),
            mix_needs_gpu=needs_gpu,
            train_command=' '.join(train_command(cid, config_str)),
            train_needs_gpu=True))
    runbook = dict(dataset='TuSimple', model='LaneATT', backbone='resnet34', epochs=epochs,
                   mix_root_base=mix_root_base, split_file=split_file, test_root=test_root,
                   train_size=train_size, cell_count=len(rows),
                   note=('Staged only: configs written, nothing built or trained. Baseline mixes are '
                         'CPU; the dbald mix needs a GPU (UltraEdit). All trainings need a GPU. '
                         'Hold on Kaggle/local-GPU per PLAN.md until the user gives the go-ahead.'),
                   cells=rows)
    (stage_dir / 'runbook.json').write_text(json.dumps(runbook, indent=2))
    (stage_dir / 'RUNBOOK.md').write_text(_markdown(runbook))
    return runbook


def _markdown(rb):
    lines = [f"# LaneATT x TuSimple runbook ({rb['cell_count']} cells) -- STAGED, nothing run",
             '',
             f"Backbone {rb['backbone']}, {rb['epochs']} epochs, train_size {rb['train_size']}. "
             f"Mixes land under `{rb['mix_root_base']}/<cell_id>`; split `{rb['split_file']}`.",
             '',
             '> Hold on Kaggle and the local GPU per PLAN.md until the user confirms. Baseline mixes '
             'build on CPU; the dbald mix and every training need a GPU.',
             '']
    for c in rb['cells']:
        lines += [f"## {c['id']}  ({c['method']} / {c['attack']})",
                  f"- config: `{c['config_path']}`",
                  f"- mix ({'GPU' if c['mix_needs_gpu'] else 'CPU'}): `{c['mix_command']}`",
                  f"- labels: write our split into `{c['train_label_file']}` + `{c['val_label_file']}` "
                  f"(laneatt_harness.write_tusimple_labels)",
                  f"- train (GPU): `{c['train_command']}`",
                  '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage-dir', required=True, help='Output dir for the 10 configs + runbook.')
    ap.add_argument('--mix-root-base', default='<MIX_ROOT_BASE>',
                    help='Base dir the poisoned mixes WILL live under (per-cell subdir). Placeholder by default.')
    ap.add_argument('--split-file', default=H.DEFAULT_SPLIT)
    ap.add_argument('--test-root', default='datasets/raw/tusimple_test')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--train-size', type=int, default=3263)
    args = ap.parse_args(argv)
    rb = stage(args.stage_dir, args.mix_root_base, args.split_file, args.epochs, args.test_root, args.train_size)
    print(f"Staged {rb['cell_count']} LaneATT TuSimple cells -> {args.stage_dir}")
    print('Wrote: runbook.json, RUNBOOK.md, and one laneatt_r34_<cell>.yml per cell. Nothing was built or trained.')
    gpu_mix = [c['id'] for c in rb['cells'] if c['mix_needs_gpu']]
    print(f"Baseline mixes (CPU-buildable): {len([c for c in rb['cells'] if not c['mix_needs_gpu']])} cells; "
          f"GPU-mix cells: {gpu_mix}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
