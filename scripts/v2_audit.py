"""Read-only integrity and numerical checks for the completed v2 experiment."""
from v2_common import *
m=json.loads((OUT/'manifest.json').read_text()); lock=json.loads((OUT/'selection.json').read_text())
old=json.loads((OUT/'v1_preservation_hashes.json').read_text())
for relative,digest in old.items():
    assert hashlib.sha256((V1/relative).read_bytes()).hexdigest()==digest, relative
assert hashlib.sha256((ROOT/'docs/distillation_v2_protocol.md').read_bytes()).hexdigest()==m['protocol_sha256']
assert hashlib.sha256((ROOT/lock['selected']['path']).read_bytes()).hexdigest()==lock['selected']['sha256']
train={r['raw_file'] for r in m['train']}; val={r['raw_file'] for r in m['validation']}; test={r['raw_file'] for r in m['test']}
assert not (train&val or train&test or val&test)
for r in m['train']+m['validation']+m['test']:
    assert hashlib.sha256((ROOT/r['root']/r['raw_file']).read_bytes()).hexdigest()==r['image_sha256'], r['id']
rows=json.loads((OUT/'final_quality.json').read_text()); assert len(rows)==128
assert len({(r['id'],r['seed'],r['method']) for r in rows})==128
for r in m['test']:
    c=conditions(r,ROOT)
    for seed in (100,101):
        raw=np.asarray(Image.open(OUT/'evaluation'/r['id']/f'student_v2_{seed}_raw.png'))
        actual=np.asarray(Image.open(OUT/'evaluation'/r['id']/f'student_v2_{seed}_composite.png'))
        assert np.array_equal(actual,composite(c,raw))
        assert np.array_equal(actual[c['lane_mask']>0],c['original'][c['lane_mask']>0])
        ref=np.asarray(Image.open(V1/'evaluation'/r['id']/f'teacher20_{seed}_raw.png'))
        mae=float(np.abs(raw.astype(float)-ref.astype(float)).mean(2)[c['mask']>0].mean())
        row=next(x for x in rows if x['id']==r['id'] and x['seed']==seed and x['method']=='student_v2')
        assert abs(mae-row['masked_mae'])<1e-9
for name in m['candidates']:
    training=json.loads((OUT/'candidates'/name/'complete.json').read_text())
    assert training['updates']==1000 and training['initialized_from']==m['v1_checkpoint_sha256']
    assert training['examples']==(128 if name=='mask_diverse' else 64)
    for step in m['checkpoints']:
        r=json.loads((OUT/'candidates'/name/f'validation_{step}.json').read_text()); assert len(r)==24
        assert all(np.isfinite(x[k]) for x in r for k in ('masked_mae','masked_ssim','local_lpips','boundary_error'))
for method in ['teacher20','teacher4','student_v1']+m['candidates']:
    timing=json.loads((OUT/'timing'/f'{method}.json').read_text()); assert len(timing['rows'])==8
    assert all(r['seconds']>0 and r['peak_bytes']>0 for r in timing['rows'])
write(OUT/'audit.json',dict(passed=True,v1_files_unchanged=len(old),source_images_verified=len(m['train']+m['validation']+m['test']),final_quality_rows=128,composites_exact=32,lane_pixel_preservation='imposed, verified exact',candidate_checkpoints=6,timing_rows=48,asr_measured=False))
artifacts=list((OUT/'evaluation').rglob('*.png'))+list((OUT/'validation').rglob('*.png'))+list((OUT/'candidates').rglob('step_*.pt'))
write(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
print('V2 integrity audit passed')
