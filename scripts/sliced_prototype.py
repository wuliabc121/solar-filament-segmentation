"""SAHI-inspired experimental mask slicing; tuning-only, not the frozen model result."""
import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as coco
from m3_filament.io import read_jsonl,write_jsonl,write_json,sha256,content_hash
from m3_filament.masks import decode
from m3_filament.schema import make_record,crop_metadata,validate_record
from m3_filament.evaluation import native_rle,evaluate_records,ensure_eval_images
from m3_filament.model import load_training_provenance,check_unseen,source_manifest_hash,mask_priority_indices


def tile_windows(h,w,size=1024,overlap=.5):
    if not 0<=overlap<1:raise ValueError('overlap out of range')
    def starts(length):
        if length<=size:return [0]
        step=max(1,round(size*(1-overlap)))
        return sorted(set(list(range(0,length-size+1,step))+[length-size]))
    return [[x,y,min(w,x+size),min(h,y+size)] for y in starts(h) for x in starts(w)]


def rle(mask):
    value=coco.encode(np.asfortranarray(mask,dtype=np.uint8))
    return {'size':[int(x) for x in value['size']],'counts':value['counts'].decode('ascii')}


def place_mask(mask,box,shape):
    x0,y0,x1,y1=box;h,w=shape
    if mask.shape!=(y1-y0,x1-x0) or not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('invalid tile coordinates')
    canvas=np.zeros((h,w),np.uint8);canvas[y0:y1,x0:x1]=mask
    return rle(canvas)


def mask_groups(rles,origins,threshold):
    if not 0<threshold<=1 or len(rles)!=len(origins):raise ValueError('invalid merge arguments')
    native=[native_rle(r) for r in rles];n=len(rles)
    if not n:return []
    areas=np.array([float(coco.area(x)) for x in native]);iou=np.asarray(coco.iou(native,native,[0]*n))
    intersection=iou*(areas[:,None]+areas[None,:])/(1+iou)
    ios=intersection/np.maximum(1,np.minimum(areas[:,None],areas[None,:]))
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in np.argwhere(np.triu(ios>=threshold,1)):
        if origins[int(i)]!=origins[int(j)]:parent[find(int(j))]=find(int(i))
    groups={}
    for i in range(n):groups.setdefault(find(i),[]).append(i)
    return sorted(groups.values(),key=lambda g:g[0])


def instance(segmentation,idx,score,source):
    native=native_rle(segmentation);x,y,w,h=coco.toBbox(native).tolist()
    result={'instance_id':idx,'segmentation':segmentation,'bbox':[int(x),int(y),int(x+w),int(y+h)],
       'score':float(score),'score_kind':'max_component_detector_confidence','source_run_id':source['run_id'],
       'source':copy.deepcopy(source),'candidate_source':'experimental_yolo_sliced'}
    result['crop']=crop_metadata(result,segmentation['size']);return result


def cache(weights,dataset,full_instances,output):
    import torchvision  # compiled CPU NMS, explicit in the recorded config
    from ultralytics import YOLO
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    images=ensure_eval_images(dataset);full={r['observation_id']:r for r in read_jsonl(full_instances)}
    provenance=load_training_provenance(weights);check_unseen(full,provenance['training_observation_ids'])
    digest=sha256(weights)
    if any(r['source']['checkpoint_sha256']!=digest for r in full.values()):raise ValueError('mixed full-image checkpoint')
    cfg={'device':'cpu','imgsz':1024,'conf':.05,'iou':.95,'retina_masks':True,'max_det':20,'verbose':False,'save':False}
    source=copy.deepcopy(next(iter(full.values()))['source']);source.update(run_id=out.name,
       inference_config={**cfg,'full_parent_sha256':sha256(full_instances),'tile_size':1024,'overlap':.5,
                         'nms_backend':'torchvision_cpu_for_tiles','external_script_sha256':sha256(__file__)},
       source_manifest_sha256=source_manifest_hash(),pipeline_stage='overlapping_candidates_not_M4_handoff',
       experimental_development_only=True)
    source['inference_config_sha256']=content_hash(source['inference_config'])
    model=YOLO(weights);records=[]
    for obs,original in sorted(full.items()):
        path=next(p for p in images.iterdir() if p.stem==obs);image=cv2.imread(str(path));h,w=image.shape[:2]
        objects=[]
        for old in original['instances']:
            if old['score']<.05:continue
            item=instance(old['segmentation'],len(objects)+1,old['score'],source);item['origin']=-1;objects.append(item)
        for origin,box in enumerate(tile_windows(h,w)):
            x0,y0,x1,y1=box;result=model.predict(image[y0:y1,x0:x1],**cfg)[0]
            if result.masks is None:continue
            masks=result.masks.data.cpu().numpy().astype(bool);scores=result.boxes.conf.cpu().numpy()
            used=np.zeros((y1-y0,x1-x0),bool)
            for k in mask_priority_indices(masks,scores):
                mask=masks[k]&~used
                if not mask.any():continue
                used|=mask;seg=place_mask(mask,box,[h,w]);item=instance(seg,len(objects)+1,scores[k],source)
                item['origin']=origin;item['tile_xyxy']=box;objects.append(item)
        record=make_record(obs,[h,w],objects,source);record['image_sha256']=original['image_sha256'];records.append(record)
        print(json.dumps({'observation':obs,'raw_candidates':len(objects)}),flush=True)
    out.mkdir(parents=True);write_jsonl(out/'instances.jsonl',records)
    write_json(out/'run_manifest.json',{'images':len(records),'candidates':sum(len(r['instances']) for r in records),
       'checkpoint_sha256':digest,'source':source,'independent_validation':False})


def merge_and_score(raw,dataset,output,ios):
    out=Path(output);pred=out.with_name(out.name+'_predictions')
    if out.exists() or pred.exists():raise FileExistsError(out)
    records=[]
    for original in read_jsonl(raw):
        source=copy.deepcopy(original['source']);source.update(run_id=pred.name,pipeline_stage='exclusive_fused_experimental_instances')
        source['inference_config'].update(mask_merge_ios=ios,min_area=32,max_final_instances=100)
        source['inference_config_sha256']=content_hash(source['inference_config'])
        items=original['instances'];groups=mask_groups([i['segmentation'] for i in items],[i['origin'] for i in items],ios)
        groups.sort(key=lambda g:(-max(items[i]['score'] for i in g),g[0]))
        used=np.zeros(original['image_size'],bool);instances=[]
        for group in groups:
            union=coco.merge([native_rle(items[i]['segmentation']) for i in group]);mask=coco.decode(union).astype(bool)&~used
            if int(mask.sum())<32:continue
            used|=mask;seg=rle(mask);item=instance(seg,len(instances)+1,max(items[i]['score'] for i in group),source)
            item['component_origins']=sorted({items[i]['origin'] for i in group});instances.append(item)
            if len(instances)>=100:break
        record=make_record(original['observation_id'],original['image_size'],instances,source)
        record['image_sha256']=original['image_sha256'];records.append(record)
    pred.mkdir(parents=True);write_jsonl(pred/'instances.jsonl',records)
    write_json(pred/'run_manifest.json',{'parent_sha256':sha256(raw),'merge_ios':ios,'development_only':True})
    thresholds=[.1,.15,.2,.25,.3,.35,.4,.45,.5]
    result=evaluate_records(records,dataset,thresholds,out)
    print(json.dumps(result['selected'],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['cache','score'])
    p.add_argument('--weights');p.add_argument('--full-instances');p.add_argument('--raw')
    p.add_argument('--dataset',default='data/overnight176');p.add_argument('--output',required=True)
    p.add_argument('--ios',type=float,default=.75);a=p.parse_args()
    if a.mode=='cache':cache(a.weights,a.dataset,a.full_instances,a.output)
    else:merge_and_score(a.raw,a.dataset,a.output,a.ios)


if __name__=='__main__':main()
