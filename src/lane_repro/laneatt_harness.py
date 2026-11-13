"""LaneATT poisoned-training harness for the TuSimple Table 1 row: the 5 methods x 2 attacks
matrix, the pinned 3263/363 split, and per-cell LaneATT ResNet-34 configs.

Scope locked by PLAN.md: TuSimple only, LaneATT only, methods {badnets, blended, ld_attack,
badlane, dbald}, attacks {LDA, LOA}. This module builds and validates the WIRING only -- the cell
matrix, the per-cell poisoned-set build command (reusing scripts/mix_poisoned_dataset.py), the
LaneATT R34 training config bound to each poisoned mix, and the conversion of this repo's
labels.jsonl into the exact TuSimple label-file layout LaneATT's loader expects. It runs no training
and no diffusion; those are GPU/Kaggle steps gated on the user's go-ahead.

How the pinned split is imposed on LaneATT without patching LaneATT: LaneATT's TuSimple loader
(third_party/LaneATT/lib/datasets/tusimple.py) maps each split name to fixed annotation files --
'train' -> [label_data_0313.json, label_data_0601.json], 'val' -> [label_data_0531.json], each read
as JSON-lines of {raw_file, h_samples, lanes}, with images at root/<raw_file>. We therefore write
OUR 3263 training frames (poisoned subset already relabelled) into label_data_0313.json (0601 left
empty) and OUR 363 validation frames into label_data_0531.json, inside the poisoned mix directory,
and point the config's train/val root at that directory with split: train. LaneATT then trains on
exactly our split with no code change on its side.

A note on the DBALD cell: its poisoned images need the UltraEdit render (GPU). The mix command for
the dbald cell is emitted with the ultraedit flags, but it is a GPU step; baseline cells are fully
CPU-buildable now.
"""
import json
import math
from pathlib import Path

from .baseline_attacks import METHODS as BASELINE_METHODS

ATTACKS = ('LDA', 'LOA')                       # TuSimple Table 1 columns (no LRA; main paper p.7)
METHODS = (*BASELINE_METHODS, 'dbald')         # badnets, blended, ld_attack, badlane, dbald
DEFAULT_BASE_CFG = 'third_party/LaneATT/cfgs/laneatt_tusimple_resnet34.yml'
DEFAULT_SPLIT = 'results/table1_split_20261111/tusimple_split.json'


def cell_id(method, attack):
    if method not in METHODS: raise ValueError(f'Unknown method {method!r}; choose from {METHODS}')
    if attack not in ATTACKS: raise ValueError(f'Unknown attack {attack!r}; choose from {ATTACKS}')
    return f'laneatt_tusimple_{method}_{attack.lower()}'


def cells():
    """The full 5x2 matrix, in a stable order (method-major)."""
    return [(m, a) for m in METHODS for a in ATTACKS]


# ----- per-cell poisoned-set build command (reuses mix_poisoned_dataset.py) -----

def mix_command(method, attack, output_dir, split_file=DEFAULT_SPLIT,
                root='datasets/raw/tusimple_train', labels='datasets/raw/tusimple_train/labels.jsonl',
                rate=0.10, checkpoint=None, ultraedit_device='cuda'):
    """The exact `python scripts/mix_poisoned_dataset.py ...` invocation that materialises this cell's
    training-ready poisoned mix on the pinned split's train part. Returns (command_list, needs_gpu).
    For dbald, adds --render-trigger ultraedit (GPU); baselines render on CPU. --checkpoint (a LaneATT
    adapter) enables the real heatmap for the dbald cell; omitted -> uniform-placeholder placement."""
    cmd = ['python', 'scripts/mix_poisoned_dataset.py',
           '--root', root, '--labels', labels,
           '--split-file', split_file, '--split-part', 'train',
           '--attack', attack, '--method', method, '--rate', str(rate),
           '--output', str(output_dir)]
    needs_gpu = method == 'dbald'
    if method == 'dbald':
        cmd += ['--render-trigger', 'ultraedit', '--trigger-kind', 'mud', '--ultraedit-device', ultraedit_device]
        if checkpoint: cmd += ['--checkpoint', checkpoint, '--backbone', 'resnet34', '--device', ultraedit_device]
    return cmd, needs_gpu


# ----- LaneATT R34 config bound to a poisoned mix -----

def _replace_once(text, old, new):
    if text.count(old) < 1: raise ValueError(f'base config missing expected fragment: {old!r}')
    return text.replace(old, new)

def build_config(method, attack, mix_root, test_root='datasets/raw/tusimple_test',
                 base_cfg=DEFAULT_BASE_CFG, epochs=100, batch_size=8, train_size=3263):
    """Return the text of a LaneATT R34 training config for this cell, bound to the poisoned mix.

    Transforms the vendored base R34 config (preserving its augmentation `!!python/tuple` tags by
    editing text, not round-tripping YAML): train+val roots -> mix_root, test root -> test_root,
    train split -> 'train' (our label_data_0313.json only, so val is excluded from training), epochs
    and the cosine T_max -> match our train_size/batch_size. backbone stays resnet34 (supp. §3)."""
    text = Path(base_cfg).read_text()
    mix_root = str(mix_root).replace('\\', '/'); test_root = str(test_root).replace('\\', '/')
    # longer/more-specific string first so 'datasets/tusimple-test' is not hit by the tusimple rule
    text = _replace_once(text, 'root: "datasets/tusimple-test"', f'root: "{test_root}"')
    text = text.replace('root: "datasets/tusimple"', f'root: "{mix_root}"')   # train and val blocks
    text = _replace_once(text, 'split: train+val', 'split: train')
    text = _replace_once(text, 'epochs: 100', f'epochs: {epochs}')
    iters = math.ceil(train_size / batch_size)
    text = _replace_once(text, 'T_max: 45400 # 100 * 454 iterations',
                         f'T_max: {epochs * iters} # {epochs} * {iters} iterations ({train_size} imgs / bs {batch_size})')
    header = (f'# GENERATED by lane_repro.laneatt_harness for cell {cell_id(method, attack)}\n'
              f'# method={method} attack={attack} mix_root={mix_root} (poisoned 10% of pinned train split)\n'
              f'# Backbone ResNet-34 per supplementary 3; hyperparameters from the vendored official\n'
              f'# TuSimple R34 recipe (SOURCES.md 5). Not a verified training driver -- wiring only.\n')
    return header + text

def write_config(method, attack, mix_root, out_path, **kwargs):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_config(method, attack, mix_root, **kwargs))
    return out_path


# ----- TuSimple label-file layout LaneATT reads -----

TRAIN_ANNO = 'label_data_0313.json'            # LaneATT 'train' split, file 1 -> all our train frames
TRAIN_ANNO_EMPTY = 'label_data_0601.json'      # LaneATT 'train' split, file 2 -> left empty
VAL_ANNO = 'label_data_0531.json'              # LaneATT 'val' split -> our 363 val frames

def _label_line(record):
    """One JSON-lines annotation in the exact shape LaneATT's load_annotations reads: raw_file,
    h_samples, lanes. Extra keys in our records (attack/method/poisoned/...) are dropped. lanes are
    taken as-is (poisoned records already carry poisoned x-lists; -2 == invalid, LaneATT drops x<0)."""
    for key in ('raw_file', 'h_samples', 'lanes'):
        if key not in record: raise ValueError(f'record missing {key!r}')
    return json.dumps({'raw_file': record['raw_file'], 'h_samples': record['h_samples'], 'lanes': record['lanes']})

def write_tusimple_labels(train_records, val_records, mix_root):
    """Write our split into the files LaneATT's TuSimple loader expects, under mix_root:
    label_data_0313.json = train_records (poisoned where applicable), label_data_0601.json = empty,
    label_data_0531.json = val_records (clean). Returns the three paths. Images must already live at
    mix_root/<raw_file> (mix_poisoned_dataset.py writes them there)."""
    mix_root = Path(mix_root); mix_root.mkdir(parents=True, exist_ok=True)
    (mix_root / TRAIN_ANNO).write_text(''.join(_label_line(r) + '\n' for r in train_records))
    (mix_root / TRAIN_ANNO_EMPTY).write_text('')
    (mix_root / VAL_ANNO).write_text(''.join(_label_line(r) + '\n' for r in val_records))
    return mix_root / TRAIN_ANNO, mix_root / TRAIN_ANNO_EMPTY, mix_root / VAL_ANNO


# ----- full plan -----

def plan(output_root='<NEW_DIR>', split_file=DEFAULT_SPLIT, epochs=100):
    """A reviewable, run-nothing description of all 10 LaneATT TuSimple cells: each cell's id, its
    poisoned-mix build command, whether that step needs a GPU (dbald only), and the config path that
    build_config/write_config would produce. Mirrors scripts/plan_table1_gpu_runs.py's intent but is
    LaneATT-specific and carries the actual config binding."""
    rows = []
    for method, attack in cells():
        cid = cell_id(method, attack)
        mix_root = f'{output_root}/{cid}'
        cmd, needs_gpu = mix_command(method, attack, mix_root, split_file=split_file)
        rows.append(dict(id=cid, method=method, attack=attack, mix_root=mix_root,
                         mix_command=' '.join(cmd), poisoned_mix_needs_gpu=needs_gpu,
                         config_path=f'{mix_root}/laneatt_r34_{cid}.yml',
                         train_label_file=f'{mix_root}/{TRAIN_ANNO}',
                         val_label_file=f'{mix_root}/{VAL_ANNO}',
                         gpu_training_required=True, approval_required_before_gpu=True))
    return dict(dataset='TuSimple', model='LaneATT', backbone='resnet34', epochs=epochs,
                methods=list(METHODS), attacks=list(ATTACKS), cell_count=len(rows),
                note='Wiring only: no training/diffusion run. dbald mix needs GPU; baselines CPU-buildable.',
                cells=rows)
