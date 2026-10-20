"""Offline, fixed-task road restoration experiment. No poisoning or attack code."""
import argparse, hashlib, json, time, sys, subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from PIL import Image
from lane_repro.distillation import conditions,composite,RoadStudent,student_rollout,equivalent_epsilon

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/distillation/v1'
def save_json(path,value):
    path.write_text(json.dumps(value,indent=2))
def sync(): torch.cuda.synchronize()
def thermal_guard():
    def read(): return subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    reading=read(); temperature=float(reading.split(',')[0]); start=time.perf_counter()
    if temperature>=78:
        print('Cooling GPU from',temperature,'C',flush=True)
        while float(reading.split(',')[0])>70:
            if time.perf_counter()-start>180: raise RuntimeError('GPU did not cool within 180 seconds; progress preserved')
            time.sleep(5); reading=read()
        with (OUT/'cooling.jsonl').open('a') as f: f.write(json.dumps(dict(start_temperature=temperature,seconds=time.perf_counter()-start,final=reading))+'\n')
    return reading
def load(path): return torch.load(path,map_location='cpu',weights_only=True)
def setup():
    torch.set_num_threads(4); torch.manual_seed(1729); np.random.seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'datasets/distillation/manifest.json'
    config=dict(protocol='road-restoration-distillation-v1',manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        teacher='SD1.5-inpainting proxy',teacher_revision='8a4288a76071f7280aedbdb3253bdb9e9d5d84bb',
        torch=torch.__version__,seed=1729,width=48,training_updates=2000,batch_size=8,learning_rate=.0002,
        note='Not DBALD attack reproduction; fixed ordinary asphalt restoration')
    p=OUT/'config.json'
    if p.exists(): assert json.loads(p.read_text())==config,'Experiment config changed: use new output directory'
    else: save_json(p,config)
    return json.loads(manifest.read_text())['samples']

def cache(records,limit):
    from lane_repro.road_teacher import RoadTeacher
    folder=OUT/'cache'; folder.mkdir(exist_ok=True)
    selected=[r for r in records if r['partition'] in ('train','validation')]
    if limit: selected=selected[:limit]
    pending=[r for r in selected if not (folder/(r['id']+'.pt')).exists()]
    if not pending: return
    teacher=RoadTeacher(ROOT/'models/checkpoints/sd_inpainting')
    for n,r in enumerate(pending):
        thermal_guard()
        c=conditions(r,ROOT); sync(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        with torch.inference_mode():
            conditioning=teacher.encode(c); z,states=teacher.sample(conditioning,0)
        sync()
        item=dict(condition=conditioning.float().cpu(),states=states,timesteps=teacher.coarse_t,alphas=teacher.coarse_a,
            id=r['id'],partition=r['partition'],seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated())
        torch.save(item,folder/(r['id']+'.pt'))
        print('cache',n+1,len(pending),r['id'],round(item['seconds'],3),flush=True)

def train(records,updates):
    if (OUT/'training.json').exists() and json.loads((OUT/'training.json').read_text())['updates']>=updates:
        print('Completed training preserved; use a new protocol directory for a new run',flush=True)
        return
    data={p:[load(OUT/'cache'/(r['id']+'.pt')) for r in records if r['partition']==p] for p in ('train','validation')}
    t=data['train'][0]['timesteps'].cuda(); a=data['train'][0]['alphas'].cuda()
    def tensors(items):
        states=torch.stack([x['states'] for x in items]).cuda()
        cond=torch.cat([x['condition'] for x in items]).cuda()
        targets=torch.stack([equivalent_epsilon(states[:,i],states[:,i+1],a[i],a[i+1]) for i in range(4)],1)
        return states,cond,targets
    states,cond,targets=tensors(data['train']); vs,vc,vt=tensors(data['validation'])
    model=RoadStudent().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=.0002)
    first_step=1
    last=OUT/'student_last.pt'
    if last.exists():
        resumed=load(last); model.load_state_dict(resumed['state_dict']); optimizer.load_state_dict(resumed['optimizer'])
        first_step=resumed['step']+1
        torch.set_rng_state(resumed['rng']); torch.cuda.set_rng_state(resumed['cuda_rng'])
    @torch.no_grad()
    def validate():
        model.eval(); losses=[]; roll=[]
        for j in range(len(vs)):
            losses.append(torch.stack([(model(vs[j:j+1,i],vc[j:j+1],t[i])-vt[j:j+1,i]).square().mean() for i in range(4)]).mean().item())
            z=student_rollout(model,vs[j:j+1,0],vc[j:j+1],t,a)
            roll.append((z-vs[j:j+1,-1]).square().mean().item())
        model.train(); return dict(validation_epsilon_mse=float(np.mean(losses)),validation_rollout_mse=float(np.mean(roll)))
    baseline=validate(); best=baseline['validation_rollout_mse']; history=[dict(step=0,**baseline)]
    if first_step>1:
        history=json.loads((OUT/'training_history.json').read_text()); baseline=history[0]
        best=min(x['validation_rollout_mse'] for x in history)
    sync(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    for step in range(first_step,updates+1):
        if step%20==1: thermal_guard()
        ix=torch.randint(len(states),(8,),device='cuda'); k=torch.randint(4,(8,),device='cuda')
        pred=model(states[ix,k],cond[ix],t[k]); loss=(pred-targets[ix,k]).square().mean()
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if step%100==0 or step==updates:
            metrics=validate(); row=dict(step=step,training_loss=loss.item(),session_elapsed_seconds=time.perf_counter()-start,**metrics); history.append(row)
            if metrics['validation_rollout_mse']<best:
                best=metrics['validation_rollout_mse']
                torch.save(dict(state_dict=model.state_dict(),width=48,step=step,timesteps=t.cpu(),alphas=a.cpu(),metrics=metrics),OUT/'student_best.pt')
            save_json(OUT/'training_history.json',history); print(row,flush=True)
            torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),OUT/'student_last.pt')
            thermal_guard()
    sync()
    save_json(OUT/'training.json',dict(updates=updates,seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),
        peak_bytes=torch.cuda.max_memory_allocated(),initial_validation=baseline,best_validation_rollout_mse=best,session_first_step=first_step,
        checkpoint_selected_on='validation rollout latent MSE; no evaluation scenes used'))

def evaluate(records,limit):
    from lane_repro.road_teacher import RoadTeacher
    from skimage.metrics import structural_similarity
    selected=[r for r in records if r['partition']=='evaluation']
    if limit: selected=selected[:limit]
    if (OUT/'evaluation_protocol.json').exists():
        print('Completed evaluation preserved; use a new protocol directory for a new run',flush=True)
        return
    checkpoint=load(OUT/'student_best.pt')
    teacher=RoadTeacher(ROOT/'models/checkpoints/sd_inpainting')
    folder=OUT/'evaluation'; folder.mkdir(exist_ok=True)
    rows=json.loads((OUT/'evaluation.json').read_text()) if (OUT/'evaluation.json').exists() else []
    completed={(r['id'],r['seed'],r['method']) for r in rows}; cached={}
    for r in rows:
        if r['method']=='teacher20': cached[(r['id'],r['seed'])]=np.asarray(Image.open(folder/r['id']/f"teacher20_{r['seed']}_raw.png"))
    # Warm each method, exclude model loading/prompt encoding. CUDA synchronized timings include VAE and safety checker.
    for method in ('teacher20','teacher4','student4'):
        if method=='student4':
            teacher.release_denoiser(); model=RoadStudent(checkpoint['width']).cuda().eval(); model.load_state_dict(checkpoint['state_dict'])
        def generate(c,seed):
            conditioning=teacher.encode(c)
            if method=='student4':
                z=student_rollout(model,teacher.noise(seed).float(),conditioning.float(),checkpoint['timesteps'],checkpoint['alphas'])
            else: z,_=teacher.sample(conditioning,seed,20 if method=='teacher20' else 4,capture=False)
            return teacher.decode(z)
        with torch.inference_mode(): generate(conditions(selected[0],ROOT),999)
        for r in selected:
            c=conditions(r,ROOT); scene=folder/r['id']; scene.mkdir(exist_ok=True)
            Image.fromarray(c['crop']).save(scene/'original_crop.png'); Image.fromarray(c['mask']).save(scene/'mask.png')
            x,y=c['crop_xy']; lane=c['lane_mask'][y:y+512,x:x+512]>0; mask=c['mask']>0
            for seed in (100,101):
                if (r['id'],seed,method) in completed: continue
                hardware=thermal_guard()
                times=[]
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    for repeat in range(2):
                        thermal_guard()
                        sync(); start=time.perf_counter(); image=generate(c,seed); final=composite(c,image); sync(); times.append(time.perf_counter()-start)
                Image.fromarray(image).save(scene/f'{method}_{seed}_raw.png')
                Image.fromarray(final).save(scene/f'{method}_{seed}_composite.png')
                key=(r['id'],seed)
                if method=='teacher20': cached[key]=image
                reference=cached[key]
                _,ssim=structural_similarity(reference,image,channel_axis=2,data_range=255,full=True)
                row=dict(id=r['id'],dataset=r['dataset'],seed=seed,method=method,steps=20 if method=='teacher20' else 4,
                    hardware_before=hardware,
                    seconds=times,median_seconds=float(np.median(times)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),mask_pixels=int(mask.sum()),
                    masked_teacher_mae=float(np.abs(image.astype(float)-reference).mean(2)[mask].mean()),
                    masked_teacher_ssim=float(ssim.mean(2)[mask].mean()),
                    raw_lane_mae=float(np.abs(image.astype(float)-c['crop']).mean(2)[lane].mean()) if lane.any() else None)
                rows.append(row); save_json(OUT/'evaluation.json',rows)
                print(method,r['id'],seed,round(row['median_seconds'],3),flush=True)
    save_json(OUT/'evaluation_protocol.json',dict(scenes=len(selected),seeds=[100,101],timing_repeats=2,warmup_per_method=1,
        timing='encode + denoise + VAE decode + safety checker + native composite; excludes loading, cached prompt encoding, disk IO',
        quality='SSIM and MAE measure teacher fidelity, not validated perceptual realism; raw lane MAE before compositing',
        teacher_denoiser_parameters=teacher.unet_parameters,shared_vae_safety_parameters=teacher.shared_parameters,
        student_parameters=sum(p.numel() for p in model.parameters()),checkpoint_step=checkpoint['step']))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['cache','train','evaluate']); p.add_argument('--limit',type=int); p.add_argument('--updates',type=int,default=2000)
    args=p.parse_args(); records=setup()
    if args.stage=='cache': cache(records,args.limit)
    elif args.stage=='train': train(records,args.updates)
    else: evaluate(records,args.limit)
