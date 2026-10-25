import json
from lane_repro.results_table import rows,render,speedup,fmt,MISSING

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))

def test_missing_files_render_as_not_measured(tmp_path):
    text=render(rows(tmp_path))
    assert MISSING in text and 'nan' not in text.lower()

def test_speedup_and_values_from_result_files(tmp_path):
    write(tmp_path/'results/distillation/v2/timing/teacher20.json',dict(median_seconds=3.6,peak_bytes=2**30))
    write(tmp_path/'results/distillation/v2/timing/mask_diverse.json',dict(median_seconds=.36,peak_bytes=2**29))
    table={r['label']:r for r in rows(tmp_path)}
    assert table['mask_diverse']['speedup']==10 and table['mask_diverse']['mib']==512 and table['teacher20']['speedup']==1
    assert table['mask_diverse']['lda']==MISSING

def test_asr_only_from_asr_file(tmp_path):
    write(tmp_path/'results/distillation/v2/timing/teacher20.json',dict(median_seconds=3.6,peak_bytes=1))
    write(tmp_path/'results/asr/teacher20.json',dict(lda=.8165,lra=None,loa=.7419))
    row={r['label']:r for r in rows(tmp_path)}['teacher20']
    assert row['lda']=='81.65' and row['lra']==MISSING and row['loa']=='74.19'

def test_helpers_reject_bad_values():
    assert speedup(1,0) is None and speedup(None,1) is None and fmt(None)==MISSING
