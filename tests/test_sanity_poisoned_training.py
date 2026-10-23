"""Minimal real end-to-end run of scripts/sanity_poisoned_training.py (train_count=2, epochs=1) --
just enough to catch wiring regressions (arg parsing, build_target/poison_sample/model.loss
integration, score()'s predict_xs path) without the multi-minute cost of a realistic run. The
larger, meaningful before/after numbers from this session's own real run are reported in the PR/
commit message, not asserted here, since a 2-image fine-tune is too small to assert any specific
before/after relationship."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/sanity_poisoned_training.py'
CHECKPOINT=ROOT/'models/checkpoints/laneatt_official/experiments/laneatt_r18_tusimple/models/model_0100.pt'
SUBSET=ROOT/'datasets/raw/tusimple_subset/labels.jsonl'
HAVE_LANEATT=CHECKPOINT.exists() and SUBSET.exists() and importlib.util.find_spec('torchvision')
needs_laneatt=pytest.mark.skipif(not HAVE_LANEATT,reason='LaneATT checkpoint/subset/torchvision not set up on this machine')

@needs_laneatt
def test_runs_end_to_end_and_writes_valid_summary(tmp_path):
    out=tmp_path/'sanity_poisoned_training.json'
    result=subprocess.run([sys.executable,str(SCRIPT),'--train-count','2','--epochs','1'],
        cwd=ROOT,capture_output=True,text=True,timeout=300)
    assert result.returncode==0,result.stderr
    default_out=ROOT/'results/sanity_poisoned_training.json'
    summary=json.loads(default_out.read_text())
    assert summary['train_count']==2 and summary['epochs']==1 and len(summary['loss_history'])==1
    for key in ('clean_acc','asr'):
        assert 0<=summary['before'][key]<=1 and 0<=summary['after'][key]<=1
