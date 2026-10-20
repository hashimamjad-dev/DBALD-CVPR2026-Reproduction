"""Versioned benign diffusion research utilities; never write v1 artifacts."""
import hashlib,json,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2,numpy as np,torch
from PIL import Image
from skimage.metrics import structural_similarity
from lane_repro.distillation import conditions,composite,RoadStudent,student_rollout,equivalent_epsilon,ddim_transition
V1=ROOT/'results/distillation/v1'; OUT=ROOT/'results/distillation/v2'
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,indent=2)); temp.replace(path)
def load(path): return torch.load(path,map_location='cpu',weights_only=True)

def save_checkpoint(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary); temporary.replace(path)
def sync(): torch.cuda.synchronize()
def guard():
    def read(): return subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    reading=read(); start=time.perf_counter()
    if float(reading.split(',')[0])>=78:
        print('Cooling',reading,flush=True)
        while float(reading.split(',')[0])>70:
            if time.perf_counter()-start>180: raise RuntimeError('Cooling timeout; saved progress retained')
            time.sleep(5); reading=read()
        with (OUT/'cooling.jsonl').open('a') as f: f.write(json.dumps(dict(seconds=time.perf_counter()-start,final=reading))+'\n')
    return reading
def setup():
    OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); torch.manual_seed(1729); np.random.seed(1729)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    manifest_path=OUT/'manifest.json'
    if manifest_path.exists(): return json.loads(manifest_path.read_text())
    old=json.loads((ROOT/'datasets/distillation/manifest.json').read_text())['samples']
    train=[r for r in old if r['partition']=='train']
    validation=sorted([r for r in old if r['partition']=='validation'],key=lambda r:r['id'])[:8]
    from lane_repro.data import culane
    used={r['raw_file'] for r in old if r['dataset']=='CULane'}
    available=[s for s in culane(ROOT/'datasets/raw/culane',ROOT/'datasets/culane_smoke16.txt') if s.key not in used]
    selected=sorted(available,key=lambda s:hashlib.sha256(('v2-validation'+s.key).encode()).hexdigest())[:4]
    for i,s in enumerate(selected):
        validation.append(dict(id='culane_val_%02d'%i,raw_file=s.key,lanes=s.lanes,dataset='CULane',partition='validation',root='datasets/raw/culane',image_sha256=hashlib.sha256(s.image.read_bytes()).hexdigest()))
    data=dict(train=train,validation=validation,validation_seeds=[0,17],test=[r for r in old if r['partition']=='evaluation'],
        test_policy='Previously used v1 test set; no v2 tuning, final comparison only',
        candidates=['control_global','mask_single','mask_diverse'],updates=1000,checkpoints=[500,1000],
        v1_checkpoint_sha256=hashlib.sha256((V1/'student_best.pt').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/'docs/distillation_v2_protocol.md').read_bytes()).hexdigest())
    assert not ({r['raw_file'] for r in validation}&{r['raw_file'] for r in data['test']})
    write(manifest_path,data); return data
def cache_path(r,seed):
    if seed==0 and r['dataset']=='TuSimple': return V1/'cache'/(r['id']+'.pt')
    return OUT/'cache'/('%s_%d.pt'%(r['id'],seed))
def student(path=V1/'student_best.pt'):
    ck=load(path); model=RoadStudent(ck.get('width',48)).cuda().eval(); model.load_state_dict(ck['state_dict']); return model,ck
def latent_mask(condition): return torch.nn.functional.max_pool2d(condition[:,:1],5,1,2)
def region_mean(error,mask):
    return (error*mask).sum()/(mask.sum()*error.shape[1]).clamp_min(1)
def balanced_loss(pred,target,conditioning):
    error=(pred-target).square()
    return .5*error.mean()+.5*region_mean(error,latent_mask(conditioning))
class Quality:
    def __init__(self):
        import lpips
        torch.hub.set_dir(str(ROOT/'models/metrics/torch/hub'))
        self.perceptual=lpips.LPIPS(net='squeeze').eval()
    @torch.no_grad()
    def measure(self,raw,ref,c):
        mask=c['mask']>0; difference=np.abs(raw.astype(float)-ref.astype(float)).mean(2)
        _,ssim=structural_similarity(ref,raw,channel_axis=2,data_range=255,full=True)
        yy,xx=np.where(mask); box=(max(0,xx.min()-8),max(0,yy.min()-8),min(512,xx.max()+9),min(512,yy.max()+9))
        def tensor(im):
            array=np.asarray(Image.fromarray(im).crop(box).resize((128,128))).copy()
            return torch.from_numpy(array.transpose(2,0,1)).unsqueeze(0).float()/127.5-1
        perceptual=float(self.perceptual(tensor(raw),tensor(ref)).item())
        a=c['crop'].copy(); b=c['crop'].copy(); a[mask]=raw[mask]; b[mask]=ref[mask]
        edges=[]
        for axis in (0,1):
            border=np.diff(mask.astype(int),axis=axis)!=0
            residual=np.abs(np.diff(a.astype(float),axis=axis)-np.diff(b.astype(float),axis=axis)).mean(2)
            edges.extend(residual[border].tolist())
        x,y=c['crop_xy']; lane=c['lane_mask'][y:y+512,x:x+512]>0
        return dict(masked_mae=float(difference[mask].mean()),masked_ssim=float(ssim.mean(2)[mask].mean()),
            local_lpips=perceptual,boundary_error=float(np.mean(edges)),mask_pixels=int(mask.sum()),
            raw_lane_mae=float(np.abs(raw.astype(float)-c['crop']).mean(2)[lane].mean()) if lane.any() else None)
def save_images(r,seed,method,raw,c,partition='validation'):
    folder=OUT/partition/r['id']; folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(raw).save(folder/f'{method}_{seed}_raw.png')
    Image.fromarray(composite(c,raw)).save(folder/f'{method}_{seed}_composite.png')
    Image.fromarray(c['crop']).save(folder/'original.png'); Image.fromarray(c['mask']).save(folder/'mask.png')
