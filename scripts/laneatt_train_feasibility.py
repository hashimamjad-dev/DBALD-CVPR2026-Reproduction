"""Can LaneATT train on this GPU? Measures peak VRAM and seconds per iteration on random 640x360 batches.

Uses a surrogate loss (mean of every differentiable proposal tensor), so activation, gradient and optimizer
memory match real training but the label-matching loss, data loading and augmentation are NOT included.
Treat results as a lower bound and confirm with a short real run. Writes results/feasibility/ only.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from lane_repro.models import LaneATTAdapter
from lane_repro.training_budget import budget

def hot():
    try: return float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    except Exception: return 0.0
def cool():
    start=time.perf_counter()
    while hot()>=78:
        if time.perf_counter()-start>180: raise RuntimeError('Cooling timeout')
        time.sleep(5)

def tensors(value):
    if torch.is_tensor(value):
        if value.requires_grad and value.is_floating_point(): yield value
    elif isinstance(value,(list,tuple)):
        for item in value: yield from tensors(item)

def trial(model,batch_size,amp,iterations,warmup):
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4); scaler=torch.amp.GradScaler('cuda',enabled=amp)
    images=torch.rand(batch_size,3,360,640,device='cuda')
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); times=[]
    for i in range(warmup+iterations):
        cool(); torch.cuda.synchronize(); start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            loss=sum(t.float().mean() for t in tensors(model(images)))
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        if i>=warmup: times.append(time.perf_counter()-start)
    return dict(batch_size=batch_size,amp=amp,status='ok',median_iteration_seconds=float(np.median(times)),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='models/checkpoints/laneatt_official/experiments/laneatt_r34_tusimple/models/model_0100.pt')
    p.add_argument('--backbone',default='resnet34'); p.add_argument('--batches',type=int,nargs='+',default=[2,4,8])
    p.add_argument('--iterations',type=int,default=15); p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--train-images',type=int,default=3626,help='TuSimple train size; set to your poisoned training set size')
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--runs',type=int,default=6,help='e.g. 3 attacks x 2 methods')
    args=p.parse_args()
    if not torch.cuda.is_available(): sys.exit('CUDA GPU required')
    adapter=LaneATTAdapter(ROOT/args.checkpoint,backbone=args.backbone,device='cuda'); model=adapter.model.train()
    rows=[]
    for amp in (False,True):
        for batch in args.batches:
            try: row=trial(model,batch,amp,args.iterations,args.warmup)
            except torch.cuda.OutOfMemoryError: row=dict(batch_size=batch,amp=amp,status='oom')
            torch.cuda.empty_cache()
            if row['status']=='ok': row['budget']=budget(row['median_iteration_seconds'],batch,args.train_images,args.epochs,args.runs)
            rows.append(row); print(json.dumps(row),flush=True)
    out=ROOT/'results/feasibility'; out.mkdir(parents=True,exist_ok=True)
    (out/'laneatt_train.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0),total_mib=torch.cuda.get_device_properties(0).total_memory/2**20,
        torch_version=torch.__version__,backbone=args.backbone,train_images=args.train_images,epochs=args.epochs,runs=args.runs,rows=rows,
        scope='Surrogate loss on random inputs: lower bound. Excludes label matching, data loading, augmentation, validation and cooling.'),indent=2))

if __name__=='__main__': main()
