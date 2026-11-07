"""Method-only ROI support retains cells partially covered by a thin target box."""
import torch


def coverage_crop(masks,boxes):
    lower=torch.floor(boxes[:,:2]);upper=torch.ceil(boxes[:,2:])
    x1,y1=lower[:,0,None,None],lower[:,1,None,None]
    x2,y2=upper[:,0,None,None],upper[:,1,None,None]
    y=torch.arange(masks.shape[-2],device=masks.device)[None,:,None]
    x=torch.arange(masks.shape[-1],device=masks.device)[None,None,:]
    return masks*((x>=x1)&(x<x2)&(y>=y1)&(y<y2))
