"""Semantic components anchored to M3 detector evidence, including one-to-many repair."""
from dataclasses import dataclass,asdict
import math
import numpy as np
from scipy import ndimage
from pycocotools import mask as coco
from m3_filament.schema import make_record,crop_metadata
from m3_filament.fusion import probability
from m3_filament.io import content_hash
from team_pipeline.contracts import validate_meta
from .metrics import validate_fast_record

@dataclass(frozen=True)
class AssociationConfig:
 min_detector:float=.15
 min_iou:float=.15
 threshold:float=.35
 min_area:int=400
 def __post_init__(self):
  for k in ('min_detector','min_iou','threshold'):
   x=getattr(self,k)
   if type(x)not in (int,float) or not math.isfinite(x) or not 0<=x<=1:raise ValueError('invalid '+k)
  if type(self.min_area)is not int or self.min_area<1:raise ValueError('invalid min_area')

def prepare_components(p,threshold,min_area):
 p=probability(p,p.shape);labels,n=ndimage.label(p>=threshold,structure=np.ones((3,3)));sizes=np.bincount(labels.ravel());slices=ndimage.find_objects(labels);components=[]
 for j in np.flatnonzero(sizes>=min_area):
  if j==0:continue
  yy,xx=slices[j-1];mask=labels[yy,xx]==j;full=np.zeros(p.shape,np.uint8,order='F');full[yy,xx]=mask;rr=coco.encode(full)
  components.append({'box':(xx.start,yy.start,xx.stop,yy.stop),'roi':mask,'area':int(mask.sum()),'segmentation':{'size':list(rr['size']),'counts':rr['counts'].decode('ascii')},'semantic_mean':float(p[yy,xx][mask].mean())})
 return {'shape':p.shape,'threshold':threshold,'min_area':min_area,'components':components}

def mask_iou(a,b):
 ax0,ay0,ax1,ay1=a['box'];bx0,by0,bx1,by1=b['box'];x0,y0,x1,y1=max(ax0,bx0),max(ay0,by0),min(ax1,bx1),min(ay1,by1)
 if x0>=x1 or y0>=y1:return 0.
 inter=int((a['roi'][y0-ay0:y1-ay0,x0-ax0:x1-ax0]&b['roi'][y0-by0:y1-by0,x0-bx0:x1-bx0]).sum())
 return inter/(a['area']+b['area']-inter) if inter else 0.

def associate(prepared,semantic,metadata,config,run_id):
 rec=prepared['record'];shape=rec['image_size']
 if tuple(shape)!=tuple(semantic['shape']) or semantic['threshold']!=config.threshold or semantic['min_area']!=config.min_area:raise ValueError('component preparation/config mismatch')
 for key,m in metadata.items():
  validate_meta(m)
  if m['observation_id']!=rec['observation_id'] or m['model']!=key:raise ValueError('semantic identity mismatch')
 if not metadata:raise ValueError('semantic provenance required')
 src={'run_id':run_id,'parent_instance_source':rec['source'],'semantic_components':metadata,'association_config':asdict(config),'association_config_sha256':content_hash(asdict(config)),'artifact_kind':'coarse_candidates','is_oof':False,'benchmark_eligible':False}
 eligible=[x for x in prepared['items'] if x['instance']['score']>=config.min_detector];out=[];rejected=0
 for component in semantic['components']:
  matches=[(mask_iou(x,component),x) for x in eligible];matches=[(iou,x) for iou,x in matches if iou>0 and iou>=config.min_iou]
  if not matches:rejected+=1;continue
  iou,parent=max(matches,key=lambda v:(v[0]*v[1]['instance']['score'],v[1]['instance']['score']));old=parent['instance']
  # Ambiguity is independent of anchor acceptance; a second strong partial seed still matters.
  strong=[x['instance']['instance_id'] for x in prepared['items'] if x['instance']['score']>=.45 and mask_iou(x,component)>=.1]
  inst={'instance_id':len(out)+1,'segmentation':component['segmentation'],'bbox':list(component['box']),'score':float(old['score']),'score_kind':'detector_confidence','source_run_id':run_id,'source':dict(src),'source_instance_id':old['instance_id'],'supporting_M3_instance_ids':[x['instance']['instance_id'] for i,x in matches],'anchor_iou':iou,'semantic_mean':component['semantic_mean'],'candidate_source':'semantic_component_with_M3_anchor','requires_refiner':len(strong)>1,'refiner_reason':'multiple_strong_M3_seeds' if len(strong)>1 else None,'ambiguous_strong_M3_instance_ids':strong,'ambiguity_rule':{'detector_score_gte':.45,'component_iou_gte':.1,'independent_of_anchor_acceptance':True}}
  inst['crop']=crop_metadata(inst,shape);out.append(inst)
 result=make_record(rec['observation_id'],shape,out,src);result.update(artifact_kind='coarse_candidates',unanchored_semantic_components=rejected);validate_fast_record(result);return result
