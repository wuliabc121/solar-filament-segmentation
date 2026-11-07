"""Instance-seeded local semantic reconstruction; no GT is used at inference."""
from dataclasses import dataclass,asdict
import math
import numpy as np
from scipy import ndimage
from pycocotools import mask as coco
from m3_filament.evaluation import native_rle
from m3_filament.schema import make_record,crop_metadata
from m3_filament.fusion import probability
from m3_filament.io import content_hash
from team_pipeline.contracts import validate_meta
from .metrics import validate_fast_record

@dataclass(frozen=True)
class AdaptConfig:
 min_detector:float=.3
 grow_threshold:float=.2
 seed_threshold:float=.5
 radius:int=16
 min_area:int=32
 min_seed_fraction:float=.02
 min_seed_pixels:int=4
 fallback_detector:float=.7
 method:str='reconstruct'
 residual:bool=False
 residual_min_area:int=400
 residual_threshold:float=.35
 endpoint_guard:int=2
 def __post_init__(self):
  for k in ('min_detector','grow_threshold','seed_threshold','min_seed_fraction','fallback_detector','residual_threshold'):
   x=getattr(self,k)
   if type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1:raise ValueError('invalid '+k)
  for k in ('radius','min_area','min_seed_pixels','residual_min_area','endpoint_guard'):
   if type(getattr(self,k))is not int or getattr(self,k)<0:raise ValueError('invalid '+k)
  if self.grow_threshold>self.seed_threshold or self.method not in ('reconstruct','gate_only'):raise ValueError('invalid reconstruction thresholds/method')
  if self.radius>256:raise ValueError('radius too large')

def combine_semantics(semantics,mode,observation_id,shape):
 names={'m1':['m1'],'m2':['m2'],'both':['m1','m2'],'guided_both':['m1','m2']}.get(mode)
 if names is None:raise ValueError('unknown semantic mode')
 values=[];meta={}
 for name in names:
  a,m=semantics[name];validate_meta(m)
  if m['model']!=name or m['observation_id']!=observation_id:raise ValueError('semantic identity mismatch')
  if m['purpose']=='smoke':raise ValueError('synthetic semantics cannot be used for real adaptation')
  values.append(probability(a,shape));meta[name]=m
 if mode=='guided_both':
  pc,pt=values;p=pt+.25*np.maximum(pc-pt,0)*(pt>=.1);rule='P_T+.25*max(P_C-P_T,0)*I(P_T>=.1)'
 else:p=sum(values)/len(values);rule='single_model' if len(values)==1 else 'equal_mean'
 meta={k:dict(v,fusion_rule=rule) for k,v in meta.items()}
 return p,meta

def prepare_record(record):
 validate_fast_record(record);items=[]
 for inst in sorted(record['instances'],key=lambda i:(-i['score'],i['instance_id'])):
  x0,y0,x1,y1=inst['bbox'];m=coco.decode(native_rle(inst['segmentation']))[y0:y1,x0:x1].astype(bool)
  items.append({'instance':inst,'box':(x0,y0,x1,y1),'roi':m,'area':int(m.sum())})
 return {'record':record,'items':items}

def roi_instance(mask,box,shape,old,source,instance_id):
 x0,y0,x1,y1=box;ys,xs=np.nonzero(mask)
 if not len(xs):raise ValueError('empty ROI instance')
 full=np.zeros(shape,np.uint8,order='F');full[y0:y1,x0:x1]=mask
 r=coco.encode(full);r={'size':list(r['size']),'counts':r['counts'].decode('ascii')}
 inst={'instance_id':instance_id,'segmentation':r,'bbox':[int(x0+xs.min()),int(y0+ys.min()),int(x0+xs.max()+1),int(y0+ys.max()+1)],'score':float(old['score']),'score_kind':'detector_confidence','source_run_id':source['run_id'],'source':dict(source),'source_instance_id':old['instance_id'],'candidate_source':'M3_seeded_semantic_reconstruction','requires_refiner':False}
 inst['crop']=crop_metadata(inst,shape);return inst

def adapt_prepared(prepared,p,components,config,run_id,validate_output=True):
 record=prepared['record'];shape=tuple(record['image_size']);p=probability(p,shape)
 if not components or any(m.get('observation_id')!=record['observation_id'] for m in components.values()):raise ValueError('missing/wrong semantic components')
 for m in components.values():validate_meta(m)
 source={'run_id':run_id,'parent_instance_source':record['source'],'semantic_components':components,'adaptation_config':asdict(config),'adaptation_config_sha256':content_hash(asdict(config)),'coordinate_space':'original','is_oof':False,'benchmark_eligible':False,'artifact_kind':'coarse_candidates'}
 candidates=[x for x in prepared['items'] if x['instance']['score']>=config.min_detector];owner=np.full(shape,-1,np.int16);used=np.zeros(shape,bool)
 for rank,item in enumerate(candidates):
  x0,y0,x1,y1=item['box'];area=owner[y0:y1,x0:x1];area[item['roi']&(area<0)]=rank
 out=[];rejections=[];h,w=shape
 for rank,item in enumerate(candidates):
  old=item['instance'];x0,y0,x1,y1=item['box'];r=config.radius;xL,yT,xR,yB=max(0,x0-r),max(0,y0-r),min(w,x1+r),min(h,y1+r)
  original=np.zeros((yB-yT,xR-xL),bool);original[y0-yT:y1-yT,x0-xL:x1-xL]=item['roi'];pr=p[yT:yB,xL:xR];avail=((owner[yT:yB,xL:xR]<0)|(owner[yT:yB,xL:xR]==rank))&~used[yT:yB,xL:xR]
  seed=original&(pr>=config.seed_threshold)&avail;fraction=float(seed.sum()/max(1,item['area']));fallback=False
  if int(seed.sum())<config.min_seed_pixels or fraction<config.min_seed_fraction:
   if old['score']>=config.fallback_detector:new=original&avail;fallback=True
   else:rejections.append({'instance_id':old['instance_id'],'reason':'weak_semantic_seed','seed_fraction':fraction});continue
  elif config.method=='gate_only':new=original&avail
  else:
   domain=ndimage.maximum_filter(original,size=2*r+1,mode='constant') if r else original
   support=domain&(pr>=config.grow_threshold)&avail
   new=ndimage.binary_propagation(seed,structure=np.ones((3,3)),mask=support)
  if int(new.sum())<config.min_area:rejections.append({'instance_id':old['instance_id'],'reason':'small_reconstruction'});continue
  inst=roi_instance(new,(xL,yT,xR,yB),shape,old,source,len(out)+1);inst.update(seed_fraction=fraction,fallback_used=fallback,semantic_mean=float(pr[new].mean()),original_detector_score=old['score']);out.append(inst);used[yT:yB,xL:xR]|=new
 if config.residual:
  labels,n=ndimage.label((p>=config.residual_threshold)&~used,structure=np.ones((3,3)));sizes=np.bincount(labels.ravel());guard=ndimage.binary_dilation(used,iterations=config.endpoint_guard,structure=np.ones((3,3))) if config.endpoint_guard else used
  slices=ndimage.find_objects(labels)
  for label in np.flatnonzero(sizes>=config.residual_min_area):
   if label==0:continue
   yy,xx=slices[label-1];m=labels[yy,xx]==label
   if (guard[yy,xx]&m).any():continue
   score=float(p[yy,xx][m].mean());inst=roi_instance(m,(xx.start,yy.start,xx.stop,yy.stop),shape,{'score':score,'instance_id':None},source,len(out)+1);inst.update(candidate_source='semantic_residual',requires_refiner=True,score_kind='mean_semantic_probability',source_instance_id=None);out.append(inst)
 result=make_record(record['observation_id'],shape,out,source);result.update(artifact_kind='coarse_candidates',candidate_rejections=rejections)
 if validate_output:validate_fast_record(result)
 return result
