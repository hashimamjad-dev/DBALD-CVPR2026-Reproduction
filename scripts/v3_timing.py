"""Common-protocol timing of the selected v3 student (same scenes, seed, repeats and scope as v2_evaluate benchmark)."""
import argparse
from v2_common import *
from lane_repro.road_teacher import RoadTeacher
from lane_repro.gpu_guard import cool
OUT3=ROOT/'results/distillation/v3'
p=argparse.ArgumentParser(); p.add_argument('name'); args=p.parse_args()
manifest=setup(); folder=OUT3/args.name; done=json.loads((folder/'complete.json').read_text())
if (folder/'timing.json').exists(): print('Completed timing preserved'); raise SystemExit(0)
ck=load(folder/done['selected_checkpoint']); model=RoadStudent(ck['width']).cuda().eval(); model.load_state_dict(ck['state_dict'])
teacher=RoadTeacher(ROOT/'models/checkpoints/sd_inpainting'); teacher.release_denoiser(); t=ck['timesteps'].cuda(); a=ck['alphas'].cuda()
records=[r for ds in ('TuSimple','CULane') for r in [x for x in manifest['validation'] if x['dataset']==ds][:2]]
def generate(c):
    z=student_rollout(model,teacher.noise(17).float(),teacher.encode(c).float(),t,a)
    return composite(c,teacher.decode(z))
rows=[]
with torch.inference_mode():
    cool(); generate(conditions(records[0],ROOT)); sync()
    for r in records:
        c=conditions(r,ROOT)
        for rep in range(2):
            cool(); sync(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter(); generate(c); sync(); seconds=time.perf_counter()-start
            rows.append(dict(id=r['id'],dataset=r['dataset'],repeat=rep,seconds=seconds,peak_bytes=torch.cuda.max_memory_allocated())); print(args.name,r['id'],rep,seconds,flush=True)
write(folder/'timing.json',dict(rows=rows,median_seconds=float(np.median([x['seconds'] for x in rows])),peak_bytes=max(x['peak_bytes'] for x in rows),
    scope='Four fixed validation scenes, seed 17, two repetitions, one warmup; includes VAE encode, noise, student rollout, decode, safety check, composite; excludes loading, disk I/O and cooling.'))
