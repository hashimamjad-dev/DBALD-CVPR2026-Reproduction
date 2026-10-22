import pytest
from lane_repro.training_budget import epoch_seconds,budget

def test_epoch_rounds_up_partial_batch():
    assert epoch_seconds(0.5,8,3626)==0.5*454

def test_budget_scales_with_epochs_and_runs():
    one=budget(1.0,8,800,10); many=budget(1.0,8,800,10,runs=6)
    assert one['epoch_seconds']==100 and one['run_hours']==pytest.approx(1000/3600) and many['total_hours']==pytest.approx(6*one['run_hours'])

def test_rejects_invalid_inputs():
    for args in ((0,8,100),(1,0,100),(1,8,0)):
        with pytest.raises(ValueError): epoch_seconds(*args)
