"""Staged-only tests for scripts/probe_diffusion_timing.py.

These verify the probe's PLAN (the run-nothing path): the config matrix, the §9 knobs, and that the
default/no-`--run` CLI path writes a plan without importing torch or touching a GPU. The timing run
itself needs a GPU + a local UltraEdit snapshot and is not exercised here.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location('probe_diffusion_timing',
                                               ROOT / 'scripts' / 'probe_diffusion_timing.py')
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_plan_lists_the_sweep_and_section9_knobs():
    p = probe.plan()
    assert p['config_count'] == len(probe.CONFIGS) >= 4
    assert p['paper_target_seconds'] == 6.0
    names = {c['name'] for c in p['configs']}
    assert {'paper_target', 'step0_baseline'} <= names
    for c in p['configs']:                                      # every §9 knob present on every config
        assert set(c) >= {'name', 'drop_t5', 'cpu_offload', 'slicing', 'steps', 'note'}


def test_step0_baseline_matches_the_measured_config():
    step0 = next(c for c in probe.CONFIGS if c['name'] == 'step0_baseline')
    assert step0['drop_t5'] is False and step0['cpu_offload'] is True
    assert step0['slicing'] is False and step0['steps'] == 50   # SOURCES.md §9 / step0_worker.py


def test_paper_target_is_resident_no_t5():
    pt = next(c for c in probe.CONFIGS if c['name'] == 'paper_target')
    assert pt['drop_t5'] is True and pt['cpu_offload'] is False


def test_default_cli_writes_plan_and_runs_nothing(tmp_path, monkeypatch):
    """No --run: must produce a plan file and never import torch."""
    out = tmp_path / 'plan.json'
    monkeypatch.setitem(sys.modules, 'torch', None)             # any torch import would now raise
    rc = probe.main(['--output', str(out)])
    assert rc == 0 and out.exists()
    data = json.loads(out.read_text())
    assert data['config_count'] == len(probe.CONFIGS)
    assert 'Runs nothing' in data['note']


def test_run_without_snapshot_refuses(monkeypatch):
    import pytest
    monkeypatch.setitem(sys.modules, 'torch', None)
    with pytest.raises(SystemExit):
        probe.main(['--run'])                                   # no --snapshot -> refuse, no load


def test_single_without_snapshot_refuses(monkeypatch):
    import pytest
    monkeypatch.setitem(sys.modules, 'torch', None)
    with pytest.raises(SystemExit):
        probe.main(['--single', 'paper_target'])               # isolated child also needs a snapshot


def test_every_config_name_is_unique():
    names = [c['name'] for c in probe.CONFIGS]
    assert len(names) == len(set(names))                        # --single / --only resolve by name
