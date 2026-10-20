"""Cache validation references and diagnose v1 before choosing improvements."""
import argparse
from v2_common import *
from lane_repro.road_teacher import RoadTeacher
p=argparse.ArgumentParser(); p.add_argument('--training-diversity',action='store_true'); args=p.parse_args()
manifest=setup(); teacher=RoadTeacher(ROOT/'models/checkpoints/sd_inpainting')
if args.training_diversity:
    for r in manifest['train']:
        seed=2000+int(r['id'].split('_')[-1]); path=cache_path(r,seed)
        if path.exists(): continue
        guard(); c=conditions(r,ROOT); sync(); start=time.perf_counter()
        with torch.inference_mode():
            cond=teacher.encode(c); _,states=teacher.sample(cond,seed)
        sync(); item=dict(condition=cond.float().cpu(),states=states,timesteps=teacher.coarse_t,alphas=teacher.coarse_a,id=r['id'],seed=seed,seconds=time.perf_counter()-start)
        save_checkpoint(path,item)
        print('diversity',r['id'],seed,flush=True)
else:
    model,ck=student(); quality=Quality(); rows=[]; diagnostics=[]
    if (OUT/'baseline_validation.json').exists(): rows=json.loads((OUT/'baseline_validation.json').read_text())
    if (OUT/'diagnostics.json').exists(): diagnostics=json.loads((OUT/'diagnostics.json').read_text())
    for r in manifest['validation']:
        c=conditions(r,ROOT)
        for seed in manifest['validation_seeds']:
            if any(x['id']==r['id'] and x['seed']==seed for x in diagnostics): continue
            guard(); path=cache_path(r,seed)
            with torch.inference_mode():
                if path.exists(): item=load(path)
                else:
                    cond=teacher.encode(c); _,states=teacher.sample(cond,seed)
                    item=dict(condition=cond.float().cpu(),states=states,timesteps=teacher.coarse_t,alphas=teacher.coarse_a,id=r['id'],seed=seed)
                    save_checkpoint(path,item)
                cond=item['condition'].cuda(); states=item['states'].cuda(); t=item['timesteps']; a=item['alphas']; mask=latent_mask(cond)
                ref=teacher.decode(states[-1:]); guard()
                z4,_=teacher.sample(cond.half(),seed,4,capture=False); four=teacher.decode(z4)
                z=student_rollout(model,states[:1],cond,t,a); raw=teacher.decode(z)
                stages=[]
                for i in range(4):
                    predicted=ddim_transition(states[i:i+1],model(states[i:i+1],cond,t[i]),a[i],a[i+1])
                    err=(predicted-states[i+1:i+2]).square()
                    stages.append(dict(step=i,edit_mse=region_mean(err,mask).item(),background_mse=region_mean(err,1-mask).item()))
                rollout_error=(z-states[-1:]).square()
                diagnostics.append(dict(id=r['id'],dataset=r['dataset'],seed=seed,mask_fraction=mask.mean().item(),teacher_forced=stages,
                    rollout_edit_mse=region_mean(rollout_error,mask).item(),rollout_background_mse=region_mean(rollout_error,1-mask).item()))
            for method,image in [('teacher20',ref),('teacher4',four),('student_v1',raw)]:
                save_images(r,seed,method,image,c)
                rows.append(dict(id=r['id'],dataset=r['dataset'],seed=seed,method=method,**quality.measure(image,ref,c)))
            write(OUT/'baseline_validation.json',rows); write(OUT/'diagnostics.json',diagnostics)
            print('validation',r['id'],seed,rows[-1]['local_lpips'],flush=True)
