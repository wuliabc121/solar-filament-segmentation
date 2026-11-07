"""Auditable cached binary-mask postprocessing; no semantic probabilities invented."""
import argparse
import copy
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as coco
from .io import content_hash,read_jsonl,write_jsonl,write_json,sha256
from .masks import decode
from .schema import make_instance,crop_metadata
from .evaluation import native_rle,evaluate_records
from .model import source_manifest_hash


def transform_record(record,run_id,min_area=0,closing_radius=0,score_floor=0.):
    if type(min_area) is not int or min_area<0 or type(closing_radius) is not int or not 0<=closing_radius<=3 or not 0<=score_floor<=1:
        raise ValueError('invalid postprocess parameters')
    out=copy.deepcopy(record);source=out['source'];original_config=copy.deepcopy(source['inference_config'])
    source['parent_inference_config_sha256']=content_hash(original_config)
    source['parent_source_manifest_sha256']=source.get('source_manifest_sha256')
    source['inference_config']['postprocess']={'min_area':min_area,'closing_radius':closing_radius,'score_floor':score_floor}
    source['inference_config_sha256']=content_hash(source['inference_config'])
    source['source_manifest_sha256']=source_manifest_hash();source['run_id']=run_id
    instances=[];shape=tuple(record['image_size']);used=np.zeros(shape,bool) if closing_radius else None
    kernel=np.ones((2*closing_radius+1,2*closing_radius+1),np.uint8)
    for old in sorted(record['instances'],key=lambda i:(-i['score'],i['instance_id'])):
        if old['score']<score_floor:
            continue
        if not closing_radius:
            if int(coco.area(native_rle(old['segmentation'])))<min_area:
                continue
            new=copy.deepcopy(old);new.update(instance_id=len(instances)+1,source_run_id=run_id,source=copy.deepcopy(source))
        else:
            mask=cv2.morphologyEx(decode(old['segmentation']),cv2.MORPH_CLOSE,kernel).astype(bool)&~used
            if not mask.any() or int(mask.sum())<min_area:
                continue
            new=make_instance(mask,len(instances)+1,old['score'],source);used|=mask
        new['crop']=crop_metadata(new,shape)
        instances.append(new)
    out['instances']=instances
    return out


def process_and_score(instances,dataset,output,min_area,closing_radius,thresholds,score_floor=.05):
    if min(thresholds)<score_floor:
        raise ValueError('score threshold below postprocessing cache floor')
    output=Path(output);pred=output.with_name(output.name+'_predictions')
    if output.exists() or pred.exists():
        raise FileExistsError(output)
    rows=[transform_record(r,pred.name,min_area,closing_radius,score_floor) for r in read_jsonl(instances)]
    pred.mkdir(parents=True)
    write_jsonl(pred/'instances.jsonl',rows)
    write_json(pred/'run_manifest.json',{'input_instances':str(instances),'input_sha256':sha256(instances),
               'min_area':min_area,'closing_radius':closing_radius,'score_floor':score_floor,
               'source_manifest_sha256':source_manifest_hash(),'postprocessed_images':len(rows)})
    return evaluate_records(rows,dataset,thresholds,output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instances',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--min-area',type=int,default=0);p.add_argument('--closing-radius',type=int,default=0)
    p.add_argument('--score-floor',type=float,default=.05);p.add_argument('--thresholds',type=float,nargs='+',required=True)
    a=p.parse_args();r=process_and_score(a.instances,a.dataset,a.output,a.min_area,a.closing_radius,a.thresholds,a.score_floor)
    print(r['selected'])


if __name__=='__main__':main()
