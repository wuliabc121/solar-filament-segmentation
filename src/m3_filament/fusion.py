"""E5/E5R candidate fusion. Only genuine semantic probabilities are accepted."""
from dataclasses import dataclass,asdict
import math
import numpy as np
from scipy import ndimage
from .masks import decode
from .schema import make_record,make_instance,validate_record,crop_metadata
from .io import content_hash


@dataclass(frozen=True)
class FusionConfig:
    alpha:float=.5
    threshold:float=.5
    radius:int=0
    min_area:int=8
    fallback:bool=False
    fallback_score:float=.8
    min_coverage:float=0.
    residual:bool=False
    endpoint_guard:int=2

    def __post_init__(self):
        for key in ('alpha','threshold','fallback_score','min_coverage'):
            x=getattr(self,key)
            if not math.isfinite(x) or not 0<=x<=1:raise ValueError(f'invalid {key}')
        for key in ('radius','min_area','endpoint_guard'):
            if type(getattr(self,key))is not int or getattr(self,key)<0:raise ValueError(f'invalid {key}')


def probability(value,shape):
    p=np.asarray(value,dtype=np.float32)
    if p.shape!=tuple(shape) or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
        raise ValueError('semantic probability must be finite H×W in [0,1], in original coordinates')
    return p


def expand(mask,radius):
    return ndimage.binary_dilation(mask,structure=np.ones((3,3)),iterations=radius) if radius else mask.copy()


def fuse_record(record,pc,pt,config,run_id):
    validate_record(record)
    shape=record['image_size'];p=config.alpha*probability(pc,shape)+(1-config.alpha)*probability(pt,shape)
    ordered=sorted(record['instances'],key=lambda i:(-i['score'],i['instance_id']))
    masks=[decode(i['segmentation']).astype(bool) for i in ordered]
    owner=np.full(shape,-1,np.int32)
    for rank,m in enumerate(masks):owner[m&(owner<0)]=rank
    used=np.zeros(shape,bool);out=[];rejected=[]
    source={'run_id':run_id,'parent_instance_run_id':record['source']['run_id'],
            'fusion_config':asdict(config),'fusion_config_sha256':content_hash(asdict(config))}
    for rank,(old,m) in enumerate(zip(ordered,masks)):
        available=(owner<0)|(owner==rank)
        new=expand(m,config.radius)&(p>=config.threshold)&available&~used
        fallback=False
        if not new.any() or (new&m).sum()/m.sum()<config.min_coverage:
            if config.fallback and old['score']>=config.fallback_score:
                new=m&available&~used;fallback=True
            else:
                rejected.append({'instance_id':old['instance_id'],'reason':'empty_or_low_semantic_coverage'})
                continue
        if not new.any():continue
        inst=make_instance(new,len(out)+1,old['score'],source)
        inst.update({'source_instance_id':old['instance_id'],'parent_source_run_id':old['source_run_id'],
                     'candidate_source':'instance_guided','fallback_used':fallback,
                     'requires_refiner':False,'semantic_coverage':float((new&m).sum()/m.sum())})
        inst['crop']=crop_metadata(inst,shape);out.append(inst);used|=new
    if config.residual:
        labels,n=ndimage.label((p>=config.threshold)&~used,structure=np.ones((3,3)))
        guarded=expand(used,config.endpoint_guard)
        for label in range(1,n+1):
            m=labels==label
            if m.sum()<config.min_area:continue
            if (m&guarded).any():
                rejected.append({'residual_component':label,'reason':'possible_existing_instance_endpoint'})
                continue
            inst=make_instance(m,len(out)+1,float(p[m].mean()),source)
            inst.update({'candidate_source':'semantic_residual','score_kind':'mean_semantic_probability',
                         'requires_refiner':True,'source_instance_id':None})
            inst['crop']=crop_metadata(inst,shape);out.append(inst);used|=m
    result=make_record(record['observation_id'],shape,out,source)
    result['candidate_rejections']=rejected
    result['artifact_kind']='coarse_candidates'
    validate_record(result,exclusive=True)
    return result
