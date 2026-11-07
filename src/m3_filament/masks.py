"""COCO column-major RLE with validation before entering the native decoder."""
import numpy as np
from pycocotools import mask as coco


def binary(mask):
    a=np.asarray(mask)
    if a.ndim!=2 or min(a.shape)<=0 or not np.isin(a,[0,1]).all():
        raise ValueError('mask must be a nonempty H×W binary array')
    return a.astype(np.uint8)


def encode(mask):
    r=coco.encode(np.asfortranarray(binary(mask)))
    return {'size':list(r['size']),'counts':r['counts'].decode('ascii')}


def _runs(counts):
    if isinstance(counts,list):
        if any(type(v) is not int or v<0 for v in counts):raise ValueError('invalid RLE runs')
        return counts
    if not isinstance(counts,str):raise ValueError('RLE counts must be an ASCII string or integer list')
    runs=[];p=0
    while p<len(counts):
        x=0;k=0
        while True:
            if p>=len(counts) or k>12:raise ValueError('invalid compressed RLE')
            c=ord(counts[p])-48;p+=1
            if not 0<=c<=63:raise ValueError('invalid compressed RLE character')
            x|=(c&31)<<(5*k);k+=1
            if not c&32:
                if c&16:x|=-1<<(5*k)
                break
        if len(runs)>2:x+=runs[-2]
        if x<0:raise ValueError('negative decoded RLE run')
        runs.append(x)
    return runs


def decode(rle):
    shape=rle.get('size')
    if not isinstance(shape,list) or len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape):
        raise ValueError('invalid RLE size')
    if shape[0]*shape[1]>100_000_000:raise ValueError('RLE is larger than supported solar images')
    runs=_runs(rle.get('counts'))
    if not runs or sum(runs)!=shape[0]*shape[1]:raise ValueError('RLE counts do not cover H×W')
    safe=coco.frPyObjects({'size':shape,'counts':runs},*shape)
    return coco.decode(safe).astype(np.uint8)


def bbox(mask):
    y,x=np.nonzero(binary(mask))
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else [0,0,0,0]
