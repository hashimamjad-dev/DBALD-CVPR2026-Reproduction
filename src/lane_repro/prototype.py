"""Independent benign analysis tools; no poisoned labels or optimized triggers."""
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

TRANSFORMS=('clean','brightness_0.7','blur_sigma_1.5','contrast_0.75','rain_seed0','occlusion_fixed','jpeg_quality30')
def transform(image,name):
    if name=='clean': return image.copy()
    if name=='brightness_0.7': return np.clip(image.astype(float)*.7,0,255).astype(np.uint8)
    if name=='blur_sigma_1.5': return cv2.GaussianBlur(image,(9,9),1.5)
    if name=='contrast_0.75': return np.clip((image.astype(float)-127.5)*.75+127.5,0,255).astype(np.uint8)
    if name=='rain_seed0':
        rng=np.random.default_rng(0); overlay=image.copy(); h,w=image.shape[:2]
        for x,y in zip(rng.integers(0,w,400),rng.integers(0,h,400)):
            cv2.line(overlay,(int(x),int(y)),(int(x+5),int(y+18)),(190,190,190),1)
        return cv2.addWeighted(image,.65,overlay,.35,0)
    if name=='occlusion_fixed':
        result=image.copy(); h,w=image.shape[:2]
        result[int(h*.65):int(h*.80),int(w*.40):int(w*.60)]=127
        return result
    if name=='jpeg_quality30':
        ok,data=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,30])
        if not ok: raise RuntimeError('JPEG encoder failed')
        return cv2.cvtColor(cv2.imdecode(data,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    raise ValueError(name)

def sensitivity(model,image,clean_loss):
    x=image.detach().clone().requires_grad_(True)
    loss=clean_loss(model(x))
    if loss.ndim: raise ValueError('Expected scalar loss')
    grad=torch.autograd.grad(loss,x)[0]
    if not torch.isfinite(grad).all(): raise ValueError('Nonfinite gradient')
    return grad.abs().sum(1).detach()

def windows(mask,size=(40,40),stride=20,min_fraction=1.):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or stride<=0 or min(size)<=0 or not 0<=min_fraction<=1: raise ValueError('Invalid window arguments')
    h,w=mask.shape; wh,ww=size
    if wh>h or ww>w: return []
    ys=sorted(set(list(range(0,h-wh+1,stride))+[h-wh])); xs=sorted(set(list(range(0,w-ww+1,stride))+[w-ww]))
    return [(x,y,ww,wh) for y in ys for x in xs if mask[y:y+wh,x:x+ww].mean()>=min_fraction]

def inpaint(image,mask):
    mask=np.asarray(mask,dtype=bool)
    if image.dtype!=np.uint8 or mask.shape!=image.shape[:2] or not mask.any() or mask.all(): raise ValueError('Invalid image/mask')
    result=cv2.inpaint(image,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
    result[~mask]=image[~mask]
    return result

def window_statistics(heat,regions):
    """Descriptive clean sensitivity statistics, in spatial order (no selection)."""
    heat=np.asarray(heat,dtype=float)
    if heat.ndim!=2 or not np.isfinite(heat).all(): raise ValueError('Invalid heatmap')
    rows=[]
    for x,y,w,h in regions:
        if min(x,y)<0 or min(w,h)<=0 or x+w>heat.shape[1] or y+h>heat.shape[0]: raise ValueError('Window out of bounds')
        part=heat[y:y+h,x:x+w]
        rows.append(dict(x=x,y=y,width=w,height=h,mean=float(part.mean()),maximum=float(part.max()),total=float(part.sum())))
    return rows

def consistency(clean,edited,mask):
    a=np.asarray(clean,dtype=float)/255.; b=np.asarray(edited,dtype=float)/255.
    mask=np.asarray(mask,dtype=bool)
    if a.shape!=b.shape or mask.shape!=a.shape[:2] or not mask.any(): raise ValueError('Invalid arrays/mask')
    m=mask[...,None]; score=float(structural_similarity(a*m,b*m,data_range=1.,channel_axis=2,win_size=7))
    return dict(masked_array_mse=float(np.mean(((a-b)*m)**2)),selected_pixel_mse=float(np.mean((a[mask]-b[mask])**2)),
                masked_array_ssim=score,masked_array_ssim_dissimilarity=1-score,selected_pixels=int(mask.sum()))

def road_sam2(image,checkpoint,points,labels,device='cpu'):
    from pathlib import Path
    import sys
    # The CUDA environment reuses CPU dependencies but does not process their
    # editable-install .pth files. Import the pinned local SAM2 checkout explicitly.
    sam2_root=Path(__file__).resolve().parents[2]/'third_party/sam2'
    if str(sam2_root) not in sys.path: sys.path.insert(0,str(sam2_root))
    if not Path(checkpoint).is_file(): raise FileNotFoundError(checkpoint)
    if len(points)!=len(labels) or not points or 1 not in labels: raise ValueError('Need positive road prompts')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),device=device))
    with torch.inference_mode():
        predictor.set_image(image)
        masks,scores,_=predictor.predict(point_coords=np.asarray(points),point_labels=np.asarray(labels),multimask_output=True)
    index=int(np.argmax(scores))
    return masks[index].astype(bool),float(scores[index])
