import numpy as np
from PIL import Image

def preprocess(image, size=256, low=1.0, high=99.0):
    x=np.asarray(image,dtype=np.float32)
    if x.ndim==3: x=x[0] if x.shape[0] < x.shape[-1] else x[...,0]
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
    lo,hi=np.percentile(x,[low,high])
    if hi<=lo: hi=lo+1e-6
    x=np.clip(x,lo,hi); x=(x-lo)/(hi-lo)
    x=np.asarray(Image.fromarray((x*255).astype(np.uint8)).resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255.0
    return x[None,...], {'clip_low':float(lo),'clip_high':float(hi),'size':size}

def preprocess_mask(mask,size=256):
    m=np.asarray(mask)
    if m.ndim>2: m=m[...,0]
    m=(m>0).astype(np.uint8)
    m=np.asarray(Image.fromarray(m).resize((size,size),Image.Resampling.NEAREST),dtype=np.uint8)
    return m[None,...]
