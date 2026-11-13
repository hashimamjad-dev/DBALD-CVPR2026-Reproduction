"""Tests for scripts/stage_laneatt_cells.py -- the run-nothing config+runbook staging for the 10
LaneATT TuSimple cells. Verifies 10 valid configs are written, the runbook carries the mix/train
commands with correct GPU gating, and nothing is built or trained.
"""
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location('stage_laneatt_cells',
                                               ROOT / 'scripts' / 'stage_laneatt_cells.py')
stage_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_mod)


def _yaml_load(text):
    loader = yaml.SafeLoader
    loader.add_constructor('tag:yaml.org,2002:python/tuple',
                           lambda l, n: tuple(l.construct_sequence(n)))
    return yaml.load(text, Loader=loader)


def test_stage_writes_ten_valid_configs_and_runbook(tmp_path):
    rb = stage_mod.stage(tmp_path, mix_root_base='datasets/poisoned/t1', split_file='split.json',
                         epochs=100, test_root='datasets/raw/tusimple_test', train_size=3263)
    assert rb['cell_count'] == 10
    assert (tmp_path / 'runbook.json').exists() and (tmp_path / 'RUNBOOK.md').exists()
    for c in rb['cells']:
        cfg_path = Path(c['config_path'])
        assert cfg_path.exists(), c['config_path']
        cfg = _yaml_load(cfg_path.read_text())
        assert cfg['model']['parameters']['backbone'] == 'resnet34'
        assert cfg['epochs'] == 100
        assert cfg['datasets']['train']['parameters']['root'] == c['mix_root']
        assert cfg['datasets']['train']['parameters']['split'] == 'train'


def test_runbook_gpu_gating_matches_methods(tmp_path):
    rb = stage_mod.stage(tmp_path, 'datasets/poisoned/t1', 'split.json', 100,
                         'datasets/raw/tusimple_test', 3263)
    by_id = {c['id']: c for c in rb['cells']}
    assert all(c['train_needs_gpu'] for c in rb['cells'])                 # every training needs a GPU
    dbald = [c for c in rb['cells'] if c['method'] == 'dbald']
    assert len(dbald) == 2 and all(c['mix_needs_gpu'] for c in dbald)     # dbald mix = GPU
    assert all(not c['mix_needs_gpu'] for c in rb['cells'] if c['method'] != 'dbald')
    # train command targets LaneATT's own trainer with the cell's config
    loa = by_id['laneatt_tusimple_dbald_loa']
    assert 'third_party/LaneATT/main.py train' in loa['train_command']
    assert loa['config_path'] in loa['train_command']


def test_cli_main_runs_nothing(tmp_path):
    rc = stage_mod.main(['--stage-dir', str(tmp_path / 's'), '--mix-root-base', 'datasets/poisoned/t1'])
    assert rc == 0
    rb = json.loads((tmp_path / 's' / 'runbook.json').read_text())
    assert rb['cell_count'] == 10 and 'Staged only' in rb['note']
