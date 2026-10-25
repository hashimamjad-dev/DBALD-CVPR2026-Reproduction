"""Cache extra teacher trajectories (new noise seeds) for v3 training scenes. Benign road restoration only."""
import argparse
from v2_common import ROOT,setup,sync,conditions,save_checkpoint,time,torch
from lane_repro.road_teacher import RoadTeacher
from lane_repro.gpu_guard import cool
OUT3=ROOT/'results/distillation/v3'
p=argparse.ArgumentParser(); p.add_argument('--extra-seeds',type=int,default=6); args=p.parse_args()
manifest=setup(); teacher=RoadTeacher(ROOT/'models/checkpoints/sd_inpainting')
for j in range(args.extra_seeds):
    for r in manifest['train']:
        seed=3000+100*j+int(r['id'].split('_')[-1]); path=OUT3/'cache'/f"{r['id']}_{seed}.pt"
        if path.exists(): continue
        cool(); c=conditions(r,ROOT); sync(); start=time.perf_counter()
        with torch.inference_mode():
            cond=teacher.encode(c); _,states=teacher.sample(cond,seed)
        sync(); save_checkpoint(path,dict(condition=cond.float().cpu(),states=states,timesteps=teacher.coarse_t,alphas=teacher.coarse_a,id=r['id'],seed=seed,seconds=time.perf_counter()-start))
        print('cache',r['id'],seed,flush=True)
