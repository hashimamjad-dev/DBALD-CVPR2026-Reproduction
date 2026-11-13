"""Wiring tests for the LaneATT poisoned-training harness (src/lane_repro/laneatt_harness.py).

No training and no diffusion run here: these check the 5x2 cell matrix, the per-cell poisoned-set
build command, the LaneATT R34 config binding, and the TuSimple label-file layout LaneATT reads.
One test feeds a REAL subset sample through the baseline poisoner to confirm the record shape that
flows into the label writer is the real thing, not a hand-built dict.
"""
import json
from pathlib import Path

import pytest
import yaml

from lane_repro import laneatt_harness as H

ROOT = Path(__file__).resolve().parents[1]


def _yaml_load(text):
    """Tolerant loader: LaneATT's config uses !!python/tuple augmentation tags that SafeLoader
    rejects. Register a constructor so the rest of the config still parses for assertions."""
    loader = yaml.SafeLoader
    loader.add_constructor('tag:yaml.org,2002:python/tuple',
                           lambda l, n: tuple(l.construct_sequence(n)))
    return yaml.load(text, Loader=loader)


# ----- matrix ---------------------------------------------------------------

def test_cell_matrix_is_five_methods_by_two_attacks():
    cs = H.cells()
    assert len(cs) == 10
    assert {m for m, _ in cs} == {'badnets', 'blended', 'ld_attack', 'badlane', 'dbald'}
    assert {a for _, a in cs} == {'LDA', 'LOA'}
    ids = [H.cell_id(m, a) for m, a in cs]
    assert len(set(ids)) == 10
    assert 'laneatt_tusimple_dbald_loa' in ids


def test_cell_id_rejects_unknown():
    with pytest.raises(ValueError): H.cell_id('nope', 'LOA')
    with pytest.raises(ValueError): H.cell_id('dbald', 'LRA')      # LRA not a TuSimple column


# ----- mix command ----------------------------------------------------------

def test_mix_command_baseline_is_cpu_and_split_aware():
    cmd, needs_gpu = H.mix_command('badnets', 'LDA', '/out/x')
    assert needs_gpu is False
    assert '--render-trigger' not in cmd                           # baselines don't invoke UltraEdit
    assert cmd[cmd.index('--method') + 1] == 'badnets'
    assert cmd[cmd.index('--attack') + 1] == 'LDA'
    assert cmd[cmd.index('--split-part') + 1] == 'train'
    assert cmd[cmd.index('--rate') + 1] == '0.1'


def test_mix_command_dbald_requests_ultraedit_and_gpu():
    cmd, needs_gpu = H.mix_command('dbald', 'LOA', '/out/y', checkpoint='ckpt.pt')
    assert needs_gpu is True
    assert cmd[cmd.index('--render-trigger') + 1] == 'ultraedit'
    assert '--checkpoint' in cmd and cmd[cmd.index('--backbone') + 1] == 'resnet34'


# ----- config binding -------------------------------------------------------

def test_build_config_binds_roots_split_and_epochs():
    text = H.build_config('badnets', 'LOA', mix_root='datasets/poisoned/cellA',
                          test_root='datasets/raw/tusimple_test', epochs=100, train_size=3263)
    cfg = _yaml_load(text)
    assert cfg['model']['parameters']['backbone'] == 'resnet34'
    assert cfg['epochs'] == 100
    assert cfg['datasets']['train']['parameters']['root'] == 'datasets/poisoned/cellA'
    assert cfg['datasets']['train']['parameters']['split'] == 'train'      # val excluded from training
    assert cfg['datasets']['val']['parameters']['root'] == 'datasets/poisoned/cellA'
    assert cfg['datasets']['test']['parameters']['root'] == 'datasets/raw/tusimple_test'
    # cosine T_max recomputed for our train size / batch (3263 / 8 -> 408 iters)
    assert cfg['lr_scheduler']['parameters']['T_max'] == 100 * 408


def test_build_config_rejects_missing_base_fragment(tmp_path):
    base = tmp_path / 'bad.yml'; base.write_text('model:\n  parameters:\n    backbone: resnet34\n')
    with pytest.raises(ValueError):
        H.build_config('badnets', 'LOA', 'm', base_cfg=str(base))


def test_write_config_writes_file(tmp_path):
    out = H.write_config('dbald', 'LDA', 'datasets/poisoned/z', tmp_path / 'c.yml')
    assert out.exists() and 'resnet34' in out.read_text()


# ----- TuSimple label layout LaneATT reads ----------------------------------

def test_write_tusimple_labels_produces_laneatt_split_files(tmp_path):
    train = [{'raw_file': 'clips/a/20.jpg', 'h_samples': [300, 310], 'lanes': [[100, 110]], 'method': 'x'},
             {'raw_file': 'clips/b/20.jpg', 'h_samples': [300, 310], 'lanes': [[-2, 120]], 'poisoned': True}]
    val = [{'raw_file': 'clips/c/20.jpg', 'h_samples': [300, 310], 'lanes': [[90, 95]]}]
    t, empty, v = H.write_tusimple_labels(train, val, tmp_path)
    assert t.name == 'label_data_0313.json' and v.name == 'label_data_0531.json'
    assert empty.read_text() == ''                                 # 0601 intentionally empty
    train_lines = t.read_text().splitlines()
    assert len(train_lines) == 2
    rec = json.loads(train_lines[0])
    assert set(rec) == {'raw_file', 'h_samples', 'lanes'}          # extra keys stripped
    assert len(v.read_text().splitlines()) == 1


def test_write_tusimple_labels_rejects_malformed_record(tmp_path):
    with pytest.raises(ValueError):
        H.write_tusimple_labels([{'raw_file': 'x', 'h_samples': [1, 2]}], [], tmp_path)   # no 'lanes'


# ----- plan -----------------------------------------------------------------

def test_plan_lists_ten_cells_and_flags_dbald_gpu():
    p = H.plan()
    assert p['cell_count'] == 10 and p['backbone'] == 'resnet34'
    dbald = [c for c in p['cells'] if c['method'] == 'dbald']
    assert len(dbald) == 2 and all(c['poisoned_mix_needs_gpu'] for c in dbald)
    assert all(c['poisoned_mix_needs_gpu'] is False for c in p['cells'] if c['method'] != 'dbald')


# ----- real-data smoke: baseline record -> label writer ---------------------

def test_real_subset_record_flows_into_label_writer(tmp_path):
    """Load a real subset sample, poison it with a baseline recipe, and confirm the resulting record
    is accepted by the label writer and round-trips to a LaneATT-readable annotation line. No images
    are rendered to disk here beyond the label file; this checks the record SHAPE is real."""
    from lane_repro.data import tusimple
    from lane_repro.baseline_attacks import baseline_poison_sample
    sub = ROOT / 'datasets/raw/tusimple_subset'
    samples = list(tusimple(sub, sub / 'labels.jsonl'))
    assert samples, 'subset fixture missing'
    result = baseline_poison_sample(samples[0], 'badnets', 'LOA')
    record = {'raw_file': result['raw_file'], 'h_samples': result['h_samples'], 'lanes': result['poisoned_lanes']}
    t, _, _ = H.write_tusimple_labels([record], [], tmp_path)
    line = json.loads(t.read_text().splitlines()[0])
    assert line['raw_file'] == samples[0].key
    assert len(line['lanes']) == len(result['poisoned_lanes'])
