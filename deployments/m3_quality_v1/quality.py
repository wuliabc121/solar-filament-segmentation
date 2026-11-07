"""Feature-based mask quality learning; inspired by, not a reproduction of, MS R-CNN."""
from dataclasses import dataclass
import copy
import math
import random
import cv2
import numpy as np
import torch
from torch import nn
from pycocotools import mask as coco
from m3_filament.evaluation import native_rle, pairwise_iou
from m3_filament.schema import crop_metadata

FEATURES=['det_score','log_area','log_width','log_height','extent','aspect','compactness','components','largest_component_fraction','radial_position','edge_distance','mean_inside','std_inside','q10_inside','q50_inside','q90_inside','mean_ring','std_ring','contrast','relative_contrast','gradient_inside','gradient_boundary','image_mean','image_std']


def quality_targets(predictions, annotations):
    """Expected per-annotator best IoU and Pr(IoU > .5); empty GT excluded as in metric."""
    groups=[g for g in annotations.values() if g]
    if not groups: raise ValueError('no nonempty annotator record')
    if not predictions:return np.empty((0,2),np.float32)
    best=np.stack([pairwise_iou(g,predictions).max(axis=0) for g in groups])
    return np.stack([best.mean(axis=0),(best>.5).mean(axis=0)],axis=1).astype(np.float32)


def extract_features(gray,record):
    if gray.ndim!=2 or list(gray.shape)!=record['image_size']:raise ValueError('image shape mismatch')
    h,w=gray.shape;rows=[];immean=float(gray.mean())/255;imstd=float(gray.std())/255
    for obj in record['instances']:
        mask=coco.decode(native_rle(obj['segmentation'])).astype(bool)
        x0,y0,x1,y1=obj['bbox'];a=int(mask.sum());bw=x1-x0;bh=y1-y0
        if not a or min(bw,bh)<=0:raise ValueError('empty mask')
        pad=8;xa=max(0,x0-pad);ya=max(0,y0-pad);xb=min(w,x1+pad);yb=min(h,y1+pad)
        m=mask[ya:yb,xa:xb].astype(np.uint8);im=gray[ya:yb,xa:xb].astype(np.float32)/255
        dil=cv2.dilate(m,np.ones((7,7),np.uint8));ring=(dil>0)&(m==0)
        inner=im[m>0];outer=im[ring] if ring.any() else inner
        cs,_=cv2.findContours(m,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        per=sum(cv2.arcLength(c,True) for c in cs)
        n,_,stats,_=cv2.connectedComponentsWithStats(m,8)
        er=cv2.erode(m,np.ones((3,3),np.uint8));boundary=(m>0)&(er==0)
        gx=cv2.Sobel(im,cv2.CV_32F,1,0);gy=cv2.Sobel(im,cv2.CV_32F,0,1);grad=np.hypot(gx,gy)
        yy,xx=np.where(m);cx=xa+xx.mean();cy=ya+yy.mean();mi=float(inner.mean());mo=float(outer.mean())
        rows.append([obj['score'],math.log1p(a),math.log1p(bw),math.log1p(bh),a/(bw*bh),math.log(bw/bh),4*math.pi*a/max(1,per*per),math.log1p(n-1),max(stats[1:,cv2.CC_STAT_AREA])/a,math.hypot((cx-w/2)/(w/2),(cy-h/2)/(h/2)),min(x0,y0,w-x1,h-y1)/max(h,w),mi,float(inner.std()),*np.quantile(inner,[.1,.5,.9]).tolist(),mo,float(outer.std()),mo-mi,(mo-mi)/max(.01,mo),float(grad[m>0].mean()),float(grad[boundary].mean()) if boundary.any() else 0,immean,imstd])
    result=np.asarray(rows,dtype=np.float32).reshape(-1,len(FEATURES))
    if not np.isfinite(result).all():raise ValueError('nonfinite features')
    return result


def filter_record(record,scores,threshold,run_id,metadata,min_area=32):
    scores=np.asarray(scores,dtype=float)
    if scores.shape!=(len(record['instances']),) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():raise ValueError('invalid scores')
    if not 0<=threshold<=1:raise ValueError('invalid threshold')
    out=copy.deepcopy(record);source=out['source'];source.update(run_id=run_id,quality_scoring=metadata,score_threshold=threshold,min_area=min_area,benchmark_eligible=False,is_oof=False)
    # Masks already exclusive. Filtering never changes geometry or claims new pixels.
    items=[]
    for obj,score in zip(record['instances'],scores):
        if score<threshold or float(coco.area(native_rle(obj['segmentation'])))<min_area:continue
        item=copy.deepcopy(obj);item.update(instance_id=len(items)+1,detector_score=obj['score'],score=float(score),score_kind='learned_mask_quality' if metadata.get('head_sha256') else 'detector_confidence',source=copy.deepcopy(source),source_run_id=run_id)
        item['crop']=crop_metadata(item,record['image_size']);items.append(item)
    out['instances']=items;return out


def split_groups(ids,hashes,n_fit,n_tune,seed):
    """Date OR exact-content connected components stay together. Requested counts are minima."""
    ids=sorted(ids);parent={s:s for s in ids}
    def find(s):
        while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
        return s
    index={}
    for s in ids:
        for k in [('date',s[:8]),('hash',hashes[s])]:
            if k in index:parent[find(s)]=find(index[k])
            else:index[k]=s
    groups={}
    for s in ids:groups.setdefault(find(s),[]).append(s)
    g=list(groups.values());random.Random(seed).shuffle(g);fit=[];tune=[]
    for group in g:
        if len(tune)<n_tune:tune.extend(group)
        elif len(fit)<n_fit:fit.extend(group)
    if len(fit)<n_fit or len(tune)<n_tune:raise ValueError('insufficient independent groups')
    return sorted(fit),sorted(tune)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    @classmethod
    def fit(cls,x):
        if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('invalid fit matrix')
        return cls(x.mean(0),np.maximum(x.std(0),1e-5))
    def transform(self,x):return (x-self.mean)/self.std


class QualityHead(nn.Module):
    def __init__(self,n_features=len(FEATURES)):
        super().__init__();self.net=nn.Sequential(nn.Linear(n_features,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,2))
    def forward(self,x):return self.net(x)


def head_scores(head,normalizer,x):
    with torch.inference_mode():return torch.sigmoid(head(torch.as_tensor(normalizer.transform(x),dtype=torch.float32))).numpy()


def score_mode(predicted,detector,mode):
    if mode=='iou':return predicted[:,0]
    if mode=='match':return predicted[:,1]
    if mode=='joint':return np.sqrt(np.asarray(detector)*predicted[:,0])
    if mode=='detector':return np.asarray(detector)
    raise ValueError('unknown score mode')
