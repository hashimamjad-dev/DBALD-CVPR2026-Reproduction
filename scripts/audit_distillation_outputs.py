"""Independently check saved outputs, source integrity and measured mask errors."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from lane_repro.distillation import conditions
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/distillation/v1'
manifest=json.loads((ROOT/'datasets/distillation/manifest.json').read_text())['samples']
records={r['id']:r for r in manifest if r['partition']=='evaluation'}
rows=json.loads((OUT/'evaluation.json').read_text())
expected={(key,seed,method) for key in records for seed in (100,101) for method in ('teacher20','teacher4','student4')}
assert len(rows)==len(expected)==96
assert {(r['id'],r['seed'],r['method']) for r in rows}==expected
for r in manifest:
    assert hashlib.sha256((ROOT/r['root']/r['raw_file']).read_bytes()).hexdigest()==r['image_sha256']
maximum=0.
for row in rows:
    c=conditions(records[row['id']],ROOT); x,y=c['crop_xy']; mask=c['mask']>0
    folder=OUT/'evaluation'/row['id']; stem=row['method']+'_'+str(row['seed'])
    raw=np.asarray(Image.open(folder/(stem+'_raw.png')))
    final=np.asarray(Image.open(folder/(stem+'_composite.png')))
    ref=np.asarray(Image.open(folder/('teacher20_'+str(row['seed'])+'_raw.png')))
    native_mask=np.zeros(c['original'].shape[:2],bool); native_mask[y:y+512,x:x+512]=mask
    assert np.array_equal(final[~native_mask],c['original'][~native_mask])
    assert np.array_equal(final[y:y+512,x:x+512][mask],raw[mask])
    assert np.array_equal(final[c['lane_mask']>0],c['original'][c['lane_mask']>0])
    mae=float(np.abs(raw.astype(float)-ref).mean(2)[mask].mean())
    maximum=max(maximum,abs(mae-row['masked_teacher_mae']))
    assert len(row['seconds'])==2 and all(t>0 for t in row['seconds'])
assert maximum<1e-10
history=json.loads((OUT/'training_history.json').read_text())
protocol=json.loads((OUT/'evaluation_protocol.json').read_text())
assert protocol['checkpoint_step']==min(history,key=lambda r:r['validation_rollout_mse'])['step']
result=dict(passed=True,source_images_verified=len(manifest),outputs_verified=len(rows),
    maximum_masked_mae_recalculation_error=maximum,
    outside_mask_and_annotated_lane_pixels='Exactly copied as designed; not evidence of learned preservation',
    checkpoint_selection='Matches minimum validation rollout error')
(OUT/'output_audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
