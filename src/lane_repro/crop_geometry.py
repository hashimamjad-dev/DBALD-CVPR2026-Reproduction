"""NumPy-only crop and schedule helpers for the training-free speedup ladder."""
import numpy as np

def tight_box(mask,size,limit=512):
    """Top-left corner of a size x size window inside the limit-sized crop that contains the whole mask."""
    if size%8 or size>limit: raise ValueError('Window must be a multiple of 8 (VAE stride) and fit the crop')
    yy,xx=np.where(mask>0)
    if not len(yy): raise ValueError('Empty mask')
    y0=int(np.clip((yy.min()+yy.max()+1)//2-size//2,0,limit-size))
    x0=int(np.clip((xx.min()+xx.max()+1)//2-size//2,0,limit-size))
    if y0>yy.min() or x0>xx.min() or y0+size<yy.max()+1 or x0+size<xx.max()+1: raise ValueError('Mask does not fit in window')
    return y0,x0

def paste(base,generated,y0,x0):
    out=base.copy(); h,w=generated.shape[:2]; out[y0:y0+h,x0:x0+w]=generated
    return out

def schedule_indices(steps,total=20):
    """Indices into the 20-step DDIM schedule; steps=4 gives 0,5,10,15,20 as in the V1/V2 teacher."""
    if not 1<=steps<=total: raise ValueError('Steps outside schedule')
    return [round(i*total/steps) for i in range(steps+1)]
