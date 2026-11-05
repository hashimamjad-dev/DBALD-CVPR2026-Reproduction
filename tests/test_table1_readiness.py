import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from table1_readiness import classify_cell, preserved_records


def test_zero_is_verified_but_does_not_claim_paper_equivalence():
    row=classify_cell(dict(local_equivalent_percent='0.0',status='MEASURED INDEPENDENT BASELINE'))
    assert row['measurement_status']=='VERIFIED' and row['exact_paper_equivalence']=='UNVERIFIED'
    assert classify_cell(dict(local_equivalent_percent='',status='NOT COMPARABLE'))['comparison_status']=='NOT COMPARABLE'
    assert classify_cell(dict(local_equivalent_percent='',status='NOT MEASURED'))['measurement_status']=='MISSING'


def test_preservation_rejects_changed_values_or_dropped_records():
    before=dict(verified=[dict(id='fixture',model='SCNN',value_percent=50.)],rejected=[])
    assert preserved_records(before,before,1)==['fixture']
    with pytest.raises(ValueError):preserved_records(before,dict(verified=[],rejected=[]),1)
    with pytest.raises(ValueError):preserved_records(before,dict(verified=[dict(id='fixture',model='SCNN',value_percent=51.)],rejected=[]),1)
