"""Lock validation selection, then measure inference and final reused benchmark."""
import argparse
from v2_common import *

def aggregate(rows):
    return [{"dataset":ds,"method":method,"n":len(group),
             **{k:float(np.mean([r[k] for r in group])) for k in ('masked_mae','masked_ssim','local_lpips','boundary_error')}}
            for ds in ('TuSimple','CULane') for method in sorted({r['method'] for r in rows})
            if (group:=[r for r in rows if r['dataset']==ds and r['method']==method])]

def select(manifest):
    if (OUT/'selection.json').exists():
        print('Existing selection lock preserved'); return
    baseline=json.loads((OUT/'baseline_validation.json').read_text())
    old=[r for r in baseline if r['method']=='student_v1']; choices=[]; allrows=baseline.copy(); candidates={}
    for name in manifest['candidates']:
        assert (OUT/'candidates'/name/'complete.json').exists(), name
        options=[]
        for step in manifest['checkpoints']:
            path=OUT/'candidates'/name/f'step_{step}.pt'
            rows=json.loads((path.parent/f'validation_{step}.json').read_text())
            assert len(rows)==24
            boundary_ok=all(np.mean([r['boundary_error'] for r in rows if r['dataset']==ds]) <= np.mean([r['boundary_error'] for r in old if r['dataset']==ds]) for ds in ('TuSimple','CULane'))
            score=float(np.mean([r['local_lpips'] for r in rows])); mae=float(np.mean([r['masked_mae'] for r in rows]))
            item=dict(method=name,step=step,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),local_lpips=score,masked_mae=mae,eligible=bool(boundary_ok and np.isfinite(score+mae)))
            options.append(item)
            allrows.extend([{**r,'method':f'{name}_{step}'} for r in rows])
        eligible=[r for r in options if r['eligible']]
        candidates[name]=min(eligible or options,key=lambda r:(r['local_lpips'],r['masked_mae']))
        choices.extend(eligible)
    choices.append(dict(method='student_v1',step=1700,path=str((V1/'student_best.pt').relative_to(ROOT)),sha256=manifest['v1_checkpoint_sha256'],local_lpips=float(np.mean([r['local_lpips'] for r in old])),masked_mae=float(np.mean([r['masked_mae'] for r in old])),eligible=True))
    selected=min(choices,key=lambda r:(r['local_lpips'],r['masked_mae']))
    write(OUT/'validation_summary.json',aggregate(allrows))
    write(OUT/'selection.json',dict(selected=selected,candidates=candidates,policy='Decoded validation LPIPS, boundary no worse than v1 on each dataset, MAE tie-break; test not read',locked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),protocol_sha256=manifest['protocol_sha256']))
    print(json.dumps(selected,indent=2))

def benchmark(manifest,method):
    from lane_repro.road_teacher import RoadTeacher
    lock=json.loads((OUT/'selection.json').read_text()); path=OUT/'timing'/f'{method}.json'
    if path.exists(): print('Completed timing preserved'); return
    teacher=RoadTeacher(ROOT/'models/checkpoints/sd_inpainting'); model=None
    if method not in ('teacher20','teacher4'):
        teacher.release_denoiser()
        checkpoint=V1/'student_best.pt' if method=='student_v1' else ROOT/lock['candidates'][method]['path']
        model,ck=student(checkpoint)
    records=[r for ds in ('TuSimple','CULane') for r in [x for x in manifest['validation'] if x['dataset']==ds][:2]]
    def generate(c):
        cond=teacher.encode(c)
        if model is None: z,_=teacher.sample(cond,17,20 if method=='teacher20' else 4,capture=False)
        else: z=student_rollout(model,teacher.noise(17).float(),cond.float(),ck['timesteps'].cuda(),ck['alphas'].cuda())
        return composite(c,teacher.decode(z))
    with torch.inference_mode():
        guard(); generate(conditions(records[0],ROOT)); sync()
        rows=[]
        for r in records:
            c=conditions(r,ROOT)
            for rep in range(2):
                guard(); sync(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                result=generate(c); sync(); elapsed=time.perf_counter()-start
                rows.append(dict(id=r['id'],dataset=r['dataset'],seed=17,repeat=rep,method=method,seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated()))
                print(method,r['id'],rep,elapsed,flush=True)
    write(path,dict(rows=rows,median_seconds=float(np.median([r['seconds'] for r in rows])),peak_bytes=max(r['peak_bytes'] for r in rows),scope='Four fixed validation scenes (first two per dataset), seed17, two repetitions, one warmup. Includes VAE encode, noise, denoising, VAE decode, safety check, composite; excludes loading, disk I/O and cooling. Student denoiser timing uses shared VAE/checker without teacher UNet.'))

def final(manifest):
    from lane_repro.road_teacher import RoadTeacher
    lock=json.loads((OUT/'selection.json').read_text()); chosen=lock['selected']; checkpoint=ROOT/chosen['path']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==chosen['sha256']
    teacher=RoadTeacher(ROOT/'models/checkpoints/sd_inpainting'); teacher.release_denoiser(); model,ck=student(checkpoint); quality=Quality()
    rows=[]; target=OUT/'final_quality.json'
    if target.exists(): rows=json.loads(target.read_text())
    for r in manifest['test']:
        c=conditions(r,ROOT)
        for seed in (100,101):
            if any(x['id']==r['id'] and x['seed']==seed and x['method']=='student_v2' for x in rows): continue
            guard()
            with torch.inference_mode():
                cond=teacher.encode(c); z=student_rollout(model,teacher.noise(seed).float(),cond.float(),ck['timesteps'].cuda(),ck['alphas'].cuda()); raw=teacher.decode(z)
            save_images(r,seed,'student_v2',raw,c,'evaluation')
            ref=np.asarray(Image.open(V1/'evaluation'/r['id']/f'teacher20_{seed}_raw.png'))
            for method,im in [('teacher20',ref),('teacher4',np.asarray(Image.open(V1/'evaluation'/r['id']/f'teacher4_{seed}_raw.png'))),('student_v1',np.asarray(Image.open(V1/'evaluation'/r['id']/f'student4_{seed}_raw.png'))),('student_v2',raw)]:
                rows.append(dict(id=r['id'],dataset=r['dataset'],seed=seed,method=method,**quality.measure(im,ref,c)))
            write(target,rows); print('final',r['id'],seed,flush=True)
    write(OUT/'final_summary.json',aggregate(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['select','benchmark','final']); p.add_argument('--method',choices=['teacher20','teacher4','student_v1','control_global','mask_single','mask_diverse']); args=p.parse_args()
    manifest=setup()
    if args.action=='select': select(manifest)
    elif args.action=='benchmark':
        if not args.method: p.error('--method required')
        benchmark(manifest,args.method)
    else: final(manifest)
