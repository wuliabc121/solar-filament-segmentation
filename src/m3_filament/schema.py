"""Versioned original-coordinate instance contract, shared with M4."""
import math
import numpy as np
from .masks import encode,decode,bbox

CONVENTION='xyxy_half_open;mask_hw;coco_fortran_rle'


def make_instance(mask,instance_id,score,source):
    if not np.asarray(mask).any():raise ValueError('empty instance mask')
    return {'instance_id':int(instance_id),'segmentation':encode(mask),'bbox':bbox(mask),
            'score':float(score),'score_kind':'detector_confidence','source_run_id':source['run_id'],
            'source':dict(source),'candidate_source':'yolo'}


def make_record(observation_id,image_size,instances,source):
    return {'schema_version':'m3.instances.v1','observation_id':str(observation_id),
            'image_size':list(image_size),'coordinate_convention':CONVENTION,
            'source':dict(source),'instances':instances}


def validate_record(record,exclusive=False):
    if record.get('schema_version')!='m3.instances.v1':raise ValueError('unknown schema_version')
    if not isinstance(record.get('observation_id'),str) or not record['observation_id']:
        raise ValueError('missing observation_id')
    size=record.get('image_size')
    if not isinstance(size,list) or len(size)!=2 or any(type(v)is not int or v<=0 for v in size):
        raise ValueError('invalid image_size')
    if record.get('coordinate_convention')!=CONVENTION:raise ValueError('coordinate convention mismatch')
    if not record.get('source',{}).get('run_id'):raise ValueError('missing source run_id')
    ids=set();used=np.zeros(size,bool)
    if not isinstance(record.get('instances'),list):raise ValueError('instances must be a list')
    for inst in record['instances']:
        i=inst.get('instance_id');score=inst.get('score')
        if type(i)is not int or i<1 or i in ids:raise ValueError('invalid or duplicate instance_id')
        ids.add(i)
        if type(score)not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('invalid instance score')
        if not inst.get('source_run_id'):raise ValueError('missing source_run_id')
        mask=decode(inst['segmentation'])
        if list(mask.shape)!=size or not mask.any():raise ValueError('empty or incorrect original mask shape')
        if bbox(mask)!=inst.get('bbox'):raise ValueError('bbox does not match mask in xyxy half-open coordinates')
        if exclusive and (used&mask.astype(bool)).any():raise ValueError('overlap between final instance masks')
        used|=mask.astype(bool)
    return record


def crop_metadata(inst,shape,context=.2):
    x0,y0,x1,y1=inst['bbox'];h,w=shape
    dx=max(2,math.ceil((x1-x0)*context));dy=max(2,math.ceil((y1-y0)*context))
    return {'xyxy':[max(0,x0-dx),max(0,y0-dy),min(w,x1+dx),min(h,y1+dy)],
            'original_shape':[h,w],'padding_ltrb':[0,0,0,0],
            'target_instance_id':inst['instance_id'],'coordinate_convention':CONVENTION}
