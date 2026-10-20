"""Matched-budget distillation ablations selected on decoded validation quality."""
import argparse
from v2_common import *
from lane_repro.road_teacher import RoadTeacher
p=argparse.ArgumentParser(); p.add_argument('candidate',choices=['control_global','mask_single','mask_diverse']); args=p.parse_args()
manifest=setup(); folder=OUT/'candidates'/args.candidate; folder.mkdir(parents=True,exist_ok=True)
if (folder/'complete.json').exists(): print('Completed candidate preserved'); raise SystemExit(0)
items=[]
for r in manifest['train']:
    items.append(load(cache_path(r,0)))
    if args.candidate=='mask_diverse': items.append(load(cache_path(r,2000+int(r['id'].split('_')[-1]))))
states=torch.stack([r['states'] for r in items]).cuda(); conditioning=torch.cat([r['condition'] for r in items]).cuda()
t=items[0]['timesteps'].cuda(); a=items[0]['alphas'].cuda()
targets=torch.stack([equivalent_epsilon(states[:,i],states[:,i+1],a[i],a[i+1]) for i in range(4)],1)
teacher=RoadTeacher(ROOT/'models/checkpoints/sd_inpainting'); teacher.release_denoiser()
model,ck=student(); model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=.0002)
quality=Quality(); validation=[]
for r in manifest['validation']:
    for seed in manifest['validation_seeds']:
        item=load(cache_path(r,seed)); c=conditions(r,ROOT)
        ref=np.asarray(Image.open(OUT/'validation'/r['id']/f'teacher20_{seed}_raw.png'))
        validation.append((r,seed,item,c,ref))
first=1; history=[]; previous_seconds=0.
if (folder/'last.pt').exists():
    last=load(folder/'last.pt'); model.load_state_dict(last['state_dict']); optimizer.load_state_dict(last['optimizer']); first=last['step']+1
    torch.set_rng_state(last['rng']); torch.cuda.set_rng_state(last['cuda_rng']); previous_seconds=last['seconds']
    history=json.loads((folder/'history.json').read_text())
sync(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
def validate_checkpoint(step):
    model.eval(); rows=[]
    for r,seed,item,c,ref in validation:
        guard()
        with torch.inference_mode():
            z=student_rollout(model,item['states'][:1].cuda(),item['condition'].cuda(),t,a); raw=teacher.decode(z)
        rows.append(dict(id=r['id'],dataset=r['dataset'],seed=seed,method=args.candidate,update=step,**quality.measure(raw,ref,c)))
        save_images(r,seed,args.candidate+'_'+str(step),raw,c)
    write(folder/f'validation_{step}.json',rows)
    print('decoded validation',args.candidate,step,float(np.mean([x['local_lpips'] for x in rows])),flush=True)
    model.train()

# Repair an interruption between saving a training checkpoint and completing
# its decoded validation. Restore the latest training state before continuing.
for pending in (500,1000):
    if pending < first and not (folder/f'validation_{pending}.json').exists():
        checkpoint=folder/f'step_{pending}.pt'
        if not checkpoint.exists():
            if pending != first-1: raise RuntimeError('Missing earlier selection checkpoint')
            save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),width=48,step=pending,timesteps=t.cpu(),alphas=a.cpu(),candidate=args.candidate))
        model.load_state_dict(load(checkpoint)['state_dict']); validate_checkpoint(pending)
        model.load_state_dict(last['state_dict'])
for step in range(first,1001):
    if step%20==1: guard()
    ix=torch.randint(len(states),(8,),device='cuda'); stage=torch.randint(4,(8,),device='cuda')
    pred=model(states[ix,stage],conditioning[ix],t[stage]); target=targets[ix,stage]
    loss=(pred-target).square().mean() if args.candidate=='control_global' else balanced_loss(pred,target,conditioning[ix])
    if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
    if step%100==0:
        history.append(dict(step=step,loss=loss.item(),seconds=previous_seconds+time.perf_counter()-start))
        write(folder/'history.json',history)
        save_checkpoint(folder/'last.pt',dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),seconds=previous_seconds+time.perf_counter()-start))
        print(args.candidate,'update',step,loss.item(),flush=True)
    if step in (500,1000):
        save_checkpoint(folder/f'step_{step}.pt',dict(state_dict=model.state_dict(),width=48,step=step,timesteps=t.cpu(),alphas=a.cpu(),candidate=args.candidate))
        validate_checkpoint(step)
        save_checkpoint(folder/'last.pt',dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),seconds=previous_seconds+time.perf_counter()-start))
sync()
write(folder/'complete.json',dict(candidate=args.candidate,updates=1000,examples=len(items),seconds=previous_seconds+time.perf_counter()-start,
    time_scope='training + checkpoint validation + cooling; excludes teacher-cache generation and model setup',
    peak_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),initialized_from=manifest['v1_checkpoint_sha256']))
