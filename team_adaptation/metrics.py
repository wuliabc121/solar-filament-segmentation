"""Official-aligned arithmetic; no benchmark eligibility is inferred."""
import numpy as np
from pycocotools import mask as coco
from m3_filament.schema import CONVENTION
import math
from m3_filament.evaluation import pairwise_iou,pq_counts,aggregate_counts,native_rle

def score_records(records,gt,observations):
 by={}
 for r in records:
  validate_fast_record(r)
  if r['observation_id'] in by:raise ValueError('duplicate observation')
  by[r['observation_id']]=r
 if set(by)!=set(observations) or set(gt)!=set(observations):raise ValueError('cohort coverage mismatch')
 rows=[];excluded=0
 for obs in observations:
  pred=by[obs]['instances'];masks=[i['segmentation'] for i in pred]
  for rid,g in gt[obs].items():
   if not g:excluded+=1;continue
   mat=pairwise_iou(g,masks);c=pq_counts(mat);matched=(mat>.5).any(axis=1);small=np.array([float(coco.area(native_rle(x)))<1000 for x in g])
   rows.append({'observation_id':obs,'annotation_id':rid,**c,'n_gt':len(g),'n_pred':len(masks),'small_gt':int(small.sum()),'small_matched':int((small&matched).sum()),'max_iou_gt':mat.max(axis=1).tolist() if masks else [0.]*len(g)})
 return aggregate_rows(rows,len(observations),excluded),rows


def aggregate_rows(rows,n_observations,excluded=0):
 total=aggregate_counts(rows);tp,fp,fn=total['tp'],total['fp'],total['fn'];ng=sum(x['n_gt'] for x in rows);sm=sum(x['small_gt'] for x in rows)
 total.update(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.,n_gt=ng,small_gt=sm,small_matched=sum(x['small_matched'] for x in rows),small_recall=sum(x['small_matched'] for x in rows)/sm if sm else None,observations=n_observations,annotation_records=len(rows),zero_gt_excluded=excluded)
 return total


def validate_fast_record(record):
 """Equivalent original-mask contract via native RLE area/bbox/IoU, without dense decoding."""
 if record.get('schema_version')!='m3.instances.v1' or record.get('coordinate_convention')!=CONVENTION:raise ValueError('schema/coordinates mismatch')
 size=record.get('image_size')
 if not isinstance(size,list) or len(size)!=2 or any(type(x)is not int or x<=0 for x in size) or size[0]*size[1]>100_000_000:raise ValueError('invalid image_size')
 if not isinstance(record.get('observation_id'),str) or not record['observation_id'] or not record.get('source',{}).get('run_id'):raise ValueError('missing identity')
 if not isinstance(record.get('instances'),list):raise ValueError('invalid instances')
 ids=set();rles=[]
 for i in record['instances']:
  idx=i.get('instance_id');score=i.get('score')
  if type(idx)is not int or idx<1 or idx in ids:raise ValueError('invalid instance ID')
  ids.add(idx)
  if type(score)not in (int,float) or not math.isfinite(score) or not 0<=score<=1 or not i.get('source_run_id'):raise ValueError('invalid score/source')
  if i['segmentation']['size']!=size:raise ValueError('mask size mismatch')
  rr=native_rle(i['segmentation']);area=float(coco.area(rr));x,y,w,h=coco.toBbox(rr).tolist()
  if area<=0 or [int(x),int(y),int(x+w),int(y+h)]!=i.get('bbox'):raise ValueError('empty mask/bbox mismatch')
  rles.append(rr)
 if len(rles)>1:
  mat=np.asarray(coco.iou(rles,rles,[0]*len(rles)));np.fill_diagonal(mat,0)
  if (mat>0).any():raise ValueError('overlapping prediction instances')
 return record
