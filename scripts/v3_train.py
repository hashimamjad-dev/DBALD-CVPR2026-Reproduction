"""v3 student training: rollout-aware loss, more teacher seeds, selectable width; selection on decoded validation LPIPS only.

Reads v1/v2 caches and validation references read-only; writes results/distillation/v3/<name>/. Benign road restoration.
"""
import argparse
from v2_common import *
from lane_repro.road_teacher import RoadTeacher
from lane_repro.distill_v3 import training_loss,cosine_lr
from lane_repro.gpu_guard import cool
OUT3=ROOT/'results/distillation/v3'

def collect(manifest):
    items=[]
    for r in manifest['train']:
        for path in [cache_path(r,0),cache_path(r,2000+int(r['id'].split('_')[-1]))]+sorted((OUT3/'cache').glob(f"{r['id']}_*.pt")):
            if path.exists(): items.append(load(path))
    return items

def save_validation_images(r,seed,name,step,raw,c):
    folder=OUT3/name/'validation'/r['id']; folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(raw).save(folder/f'step{step}_{seed}_raw.png'); Image.fromarray(composite(c,raw)).save(folder/f'step{step}_{seed}_composite.png')

def main():
    p=argparse.ArgumentParser(); p.add_argument('name'); p.add_argument('--width',type=int,default=48); p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--rollout-weight',type=float,default=1.); p.add_argument('--eval-every',type=int,default=500); p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--init-v1',action='store_true',help='start from the v1 student (width 48 only)'); args=p.parse_args()
    if args.init_v1 and args.width!=48: p.error('--init-v1 requires --width 48')
    manifest=setup(); folder=OUT3/args.name; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): print('Completed run preserved; choose a new name'); return
    items=collect(manifest); assert len(items)>=2*len(manifest['train']),'Run v2_prepare --training-diversity first'
    states=torch.stack([x['states'] for x in items]).cuda(); conditioning=torch.cat([x['condition'] for x in items]).cuda()
    t=items[0]['timesteps'].cuda(); a=items[0]['alphas'].cuda()
    targets=torch.stack([equivalent_epsilon(states[:,i],states[:,i+1],a[i],a[i+1]) for i in range(4)],1)
    teacher=RoadTeacher(ROOT/'models/checkpoints/sd_inpainting'); teacher.release_denoiser(); quality=Quality()
    model=RoadStudent(args.width).cuda().train()
    if args.init_v1: model.load_state_dict(student()[0].state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    validation=[(r,seed,load(cache_path(r,seed)),conditions(r,ROOT),np.asarray(Image.open(OUT/'validation'/r['id']/f'teacher20_{seed}_raw.png')))
        for r in manifest['validation'] for seed in manifest['validation_seeds']]
    first,history,spent=1,[],0.
    if (folder/'last.pt').exists():
        last=load(folder/'last.pt'); model.load_state_dict(last['state_dict']); optimizer.load_state_dict(last['optimizer']); first=last['step']+1; spent=last['seconds']
        history=json.loads((folder/'history.json').read_text())
    def checkpoint(step,path): save_checkpoint(path,dict(state_dict=model.state_dict(),width=args.width,step=step,timesteps=t.cpu(),alphas=a.cpu(),name=args.name))
    def validate(step):
        model.eval(); rows=[]
        for r,seed,item,c,ref in validation:
            cool()
            with torch.inference_mode(): raw=teacher.decode(student_rollout(model,item['states'][:1].cuda(),item['condition'].cuda(),t,a))
            rows.append(dict(id=r['id'],dataset=r['dataset'],seed=seed,method=args.name,update=step,**quality.measure(raw,ref,c))); save_validation_images(r,seed,args.name,step,raw,c)
        write(folder/f'validation_{step}.json',rows); model.train()
        return float(np.mean([x['local_lpips'] for x in rows]))
    sync(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    for step in range(first,args.updates+1):
        if step%20==1: cool()
        for group in optimizer.param_groups: group['lr']=cosine_lr(step,args.updates,args.lr,args.lr/20)
        ix=torch.randint(len(states),(8,),device='cuda'); stage=torch.randint(4,(8,),device='cuda')
        loss,parts=training_loss(model,states[ix],conditioning[ix],targets[ix],t,a,stage,args.rollout_weight)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if step%100==0 or step==args.updates:
            history.append(dict(step=step,loss=loss.item(),**parts,seconds=spent+time.perf_counter()-start)); write(folder/'history.json',history)
            save_checkpoint(folder/'last.pt',dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,seconds=spent+time.perf_counter()-start))
            print(args.name,step,loss.item(),parts,flush=True)
        if step%args.eval_every==0 or step==args.updates:
            checkpoint(step,folder/f'step_{step}.pt'); score=validate(step); history[-1]['validation_lpips']=score; write(folder/'history.json',history)
            print('decoded validation',args.name,step,score,flush=True)
    sync(); scores={x['step']:x['validation_lpips'] for x in history if 'validation_lpips' in x}; best=min(scores,key=scores.get)
    write(folder/'complete.json',dict(name=args.name,updates=args.updates,examples=len(items),width=args.width,rollout_weight=args.rollout_weight,
        parameters=sum(q.numel() for q in model.parameters()),peak_bytes=torch.cuda.max_memory_allocated(),seconds=spent+time.perf_counter()-start,
        validation_lpips=scores,selected_step=best,selected_checkpoint=f'step_{best}.pt',
        selection='lowest mean decoded validation LPIPS; test set never read',v2_reference_lpips_note='compare with v2 validation_summary.json (same 24 validation cases)'))
    print('selected',best,scores[best])

main()
