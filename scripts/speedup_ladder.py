"""Training-free speedup ladder for the benign road-restoration teacher (validation scenes only).

Actions: quality (fidelity against teacher20 on 12 validation scenes x seeds 0/17), timing (common protocol of v2_evaluate).
Writes only results/speedup/. The V1/V2 artifacts and the test set are never read.
"""
import argparse,json,time
from v2_common import ROOT,setup,write,sync,Quality,conditions,np,torch
from lane_repro.fast_teacher import FastTeacher
from lane_repro.crop_geometry import tight_box,paste
from lane_repro.gpu_guard import cool
OUT=ROOT/'results/speedup'
# name: (steps, guidance, window); window 512 = the existing teacher crop
METHODS={'teacher20':(20,7.5,512),'teacher4':(4,7.5,512),'teacher4_nocfg':(4,1.0,512),
    'teacher4_win256':(4,7.5,256),'teacher4_win256_nocfg':(4,1.0,256),'teacher2_win256_nocfg':(2,1.0,256)}


def generate(teacher,c,seed,method):
    """Raw 512x512 uint8 crop: window result pasted into the original crop (VAE-decoded in full for 512)."""
    steps,guidance,size=METHODS[method]
    box=(0,0) if size==512 else tight_box(c['mask'],size)
    z=teacher.sample_fast(teacher.encode_window(c,box,size),teacher.window_noise(seed,box,size),steps,guidance)
    image=teacher.decode(z)
    return image if size==512 else paste(c['crop'],image,*box)

def quality(manifest):
    teacher=FastTeacher(ROOT/'models/checkpoints/sd_inpainting'); measure=Quality(); target=OUT/'quality.json'
    rows=json.loads(target.read_text()) if target.exists() else []
    for r in manifest['validation']:
        c=conditions(r,ROOT)
        for seed in manifest['validation_seeds']:
            if any(x['id']==r['id'] and x['seed']==seed for x in rows): continue
            cool()
            with torch.inference_mode():
                reference=generate(teacher,c,seed,'teacher20')
                existing,_=teacher.sample(teacher.encode(c),seed,4,capture=False); existing=teacher.decode(existing)
                row=dict(id=r['id'],dataset=r['dataset'],seed=seed,methods={})
                for method in METHODS:
                    raw=reference if method=='teacher20' else generate(teacher,c,seed,method)
                    row['methods'][method]=measure.measure(raw,reference,c)
                    if method=='teacher4': row['teacher4_vs_existing_max_abs']=int(np.abs(raw.astype(int)-existing.astype(int)).max())
            rows.append(row); write(target,rows); print('quality',r['id'],seed,flush=True)
    summary={ds:{m:{k:float(np.mean([x['methods'][m][k] for x in rows if x['dataset']==ds])) for k in ('masked_mae','masked_ssim','local_lpips','boundary_error')} for m in METHODS} for ds in ('TuSimple','CULane')}
    write(OUT/'quality_summary.json',dict(summary=summary,n={ds:sum(x['dataset']==ds for x in rows) for ds in ('TuSimple','CULane')},
        consistency_max_abs=max(x['teacher4_vs_existing_max_abs'] for x in rows),
        note='Fidelity to teacher20 with the same noise values; proxy only, not realism. teacher4 reimplemented in the generic path must match RoadTeacher.sample (consistency_max_abs).'))

def timing(manifest):
    teacher=FastTeacher(ROOT/'models/checkpoints/sd_inpainting')
    records=[r for ds in ('TuSimple','CULane') for r in [x for x in manifest['validation'] if x['dataset']==ds][:2]]
    for method in METHODS:
        path=OUT/'timing'/f'{method}.json'
        if path.exists(): continue
        with torch.inference_mode():
            cool(); generate(teacher,conditions(records[0],ROOT),17,method); sync(); rows=[]
            for r in records:
                c=conditions(r,ROOT)
                for rep in range(2):
                    cool(); sync(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                    generate(teacher,c,17,method); sync(); seconds=time.perf_counter()-start
                    rows.append(dict(id=r['id'],dataset=r['dataset'],repeat=rep,seconds=seconds,peak_bytes=torch.cuda.max_memory_allocated())); print(method,r['id'],rep,seconds,flush=True)
        write(path,dict(rows=rows,median_seconds=float(np.median([x['seconds'] for x in rows])),peak_bytes=max(x['peak_bytes'] for x in rows),
            scope='Four fixed validation scenes, seed 17, two repetitions, one warmup; includes VAE encode, noise, denoise, decode, safety check; excludes loading, disk I/O and cooling.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['quality','timing']); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); manifest=setup()
    quality(manifest) if args.action=='quality' else timing(manifest)
