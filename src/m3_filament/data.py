"""Data auditing and explicitly provisional, observation-grouped development data."""
import json
import random
import shutil
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as coco
from .io import sha256,content_hash,write_json,write_jsonl,write_text
from .masks import decode,bbox,encode


ANNOTATION='train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'


def observation_groups(rows):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for r in rows:
        obs='o:'+r['observation_id']
        for key in ('date','sha256'):
            if not r.get(key):raise ValueError(f'missing grouping field {key}')
            union(obs,key+':'+r[key])
    groups=defaultdict(set)
    for r in rows:groups[find('o:'+r['observation_id'])].add(r['observation_id'])
    return [sorted(v) for _,v in sorted(groups.items())]


def grouped_split(rows,seed):
    groups=observation_groups(rows)
    if len(groups)<3:raise ValueError('at least three independent date/content groups are required')
    random.Random(seed).shuffle(groups)
    n=len(groups);ncal=max(1,round(n*.15));nhold=max(1,round(n*.15));ntrain=n-ncal-nhold
    result={}
    for index,group in enumerate(groups):
        split='train' if index<ntrain else 'calibration' if index<ntrain+ncal else 'holdout'
        result.update((obs,split) for obs in group)
    return result


def annotation_mask(ann,height,width):
    seg=ann['segmentation']
    if isinstance(seg,dict):return decode(seg)
    if not seg:return np.zeros((height,width),np.uint8)
    if any(len(s)<6 or len(s)%2 or not np.isfinite(s).all() for s in seg):
        raise ValueError('invalid COCO polygon')
    r=coco.merge(coco.frPyObjects(seg,height,width))
    return coco.decode(r).astype(np.uint8)


def polygon_roundtrip(ann,height,width,resample=False,min_iou=.98):
    """Convert a single COCO object via its raster contour, without filling holes."""
    original=annotation_mask(ann,height,width)
    area=int(original.sum())
    result={'coco_iou':1.0,'topology_supported':False,'native_iou':0.,'accepted':False,
            'original_area':area,'reason':None,'polygon':None}
    if area==0:result['reason']='empty_mask';return result
    if isinstance(ann['segmentation'],list) and len(ann['segmentation'])!=1:
        result['reason']='multiple_original_polygons';return result
    x0,y0,x1,y1=bbox(original)
    roi=original[y0:y1,x0:x1]
    contours,hierarchy=cv2.findContours(roi,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(contours)!=1 or hierarchy is None or hierarchy[0,0,3]!=-1:
        result['reason']='holes_or_disconnected_fragments';return result
    points=contours[0].reshape(-1,2).astype(np.float32)+[x0,y0]
    if len(points)<3 or np.ptp(points[:,0])<=0 or np.ptp(points[:,1])<=0:
        result['reason']='degenerate_training_polygon';return result
    result['topology_supported']=True
    # A fractional offset keeps serialized normalized vertices safely inside
    # the same integer raster cell when Ultralytics truncates to int32.
    normalized=np.round((points+.25)/[width,height],10)
    training_points=(normalized*[width,height]).astype(np.float32)
    if resample:
        from ultralytics.utils.ops import resample_segments
        training_points=resample_segments([training_points],n=max(1000,len(points)+1))[0]
    recovered=np.zeros((height,width),np.uint8)
    cv2.fillPoly(recovered,[training_points.astype(np.int32)],1)
    union=np.logical_or(original,recovered).sum()
    iou=float(np.logical_and(original,recovered).sum()/union)
    result.update({'native_iou':iou,'accepted':iou>=min_iou,'polygon':normalized.tolist(),
                   'reason':None if iou>=min_iou else 'raster_iou_below_gate'})
    return result


def load_data(root):
    root=Path(root);path=root/ANNOTATION
    data=json.loads(path.read_text())
    ids=[i['id'] for i in data['images']]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate annotation image record ID')
    return root,data


def audit_dataset(root,output,resample=True):
    root,data=load_data(root);out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    image_records={i['id']:i for i in data['images']}
    known_categories={x['id'] for x in data['categories']}
    files=sorted(root.rglob('*.jpeg'));file_rows=[];errors=[]
    byname={}
    for path in files:
        try:
            with Image.open(path) as im:
                im.load();w,h=im.size
                if (h,w)!=(2048,2048):raise ValueError(f'unexpected solar image dimensions {(h,w)}')
            row={'relative_path':str(path.relative_to(root)),'observation_id':path.stem,
                 'height':h,'width':w,'sha256':sha256(path),'date':path.stem[:8],
                 'partition':'test' if 'test_images' in path.parts else 'train'}
            file_rows.append(row)
            if row['partition']=='train':byname[path.name]=row
        except Exception as e:errors.append({'path':str(path),'error':str(e)})
    for im in data['images']:
        row=byname.get(im['file_name'])
        if row is None or (im['height'],im['width'])!=(row['height'],row['width']):
            errors.append({'annotation_id':im['id'],'error':'missing image or dimension mismatch'})
    mask_rows=[];seen=set()
    for index,ann in enumerate(data['annotations']):
        try:
            if ann['id'] in seen:raise ValueError('duplicate instance annotation ID')
            seen.add(ann['id'])
            if ann['category_id'] not in known_categories:raise ValueError('unknown category')
            im=image_records[ann['image_id']]
            stats=polygon_roundtrip(ann,im['height'],im['width'],resample=resample)
            stats.pop('polygon')
            mask_rows.append({'annotation_id':ann['id'],'image_record_id':ann['image_id'],
                              'observation_id':Path(im['file_name']).stem,**stats})
        except Exception as e:errors.append({'annotation_id':ann.get('id'),'error':str(e)})
        if (index+1)%1000==0:print(f'audited {index+1}/{len(data["annotations"])} instances',flush=True)
    write_jsonl(out/'data_manifest.jsonl',file_rows)
    write_jsonl(out/'mask_conversion_audit.jsonl',mask_rows)
    write_jsonl(out/'annotation_passes.jsonl',[
        {'annotation_id':i['id'],'observation_id':Path(i['file_name']).stem,'file_name':i['file_name']}
        for i in data['images']])
    ious=[x['native_iou'] for x in mask_rows if x['topology_supported']]
    summary={'images':len(file_rows),'train_images':sum(r['partition']=='train' for r in file_rows),
             'test_images':sum(r['partition']=='test' for r in file_rows),
             'annotation_passes':len(data['images']),'instances':len(data['annotations']),
             'annotation_sha256':sha256(root/ANNOTATION),'dataset_hash':content_hash(file_rows),
             'categories':data['categories'],'training_class_policy':'all categories -> filament class 0',
             'annotation_policy':'one annotation pass per sample, no cross-annotator unions',
             'conversion_gate':.98,'ultralytics_resample_checked':resample,
             'conversion_accepted':sum(r['accepted'] for r in mask_rows),
             'conversion_rejections':dict(Counter(r['reason'] for r in mask_rows if not r['accepted'])),
             'supported_native_iou_mean':float(np.mean(ious)) if ious else None,
             'supported_native_iou_min':min(ious) if ious else None,
             'errors':errors,'file_decode_passed':not errors,
             'production_conversion_passed':not errors and all(r['accepted'] for r in mask_rows),
             'near_time_note':'same date and identical content grouped; broader temporal correlation remains for M1 review'}
    write_json(out/'audit.json',summary)
    return summary


def select_development_observations(assignments,count,seed):
    if type(count)is not int or count<1:raise ValueError('positive development observation count required')
    chosen=set()
    for split,cap in [('train',count),('calibration',max(2,count//4))]:
        available=[obs for obs,p in assignments.items() if p==split]
        available.sort(key=lambda obs:content_hash([seed,obs]))
        chosen.update(available[:cap])
    return chosen


def validate_subset_selection(selection,assignments):
    selected=set()
    for role in ('train','calibration'):
        items=selection.get(role,[])
        if len(items)!=len(set(items)):raise ValueError('duplicate subset observation')
        for obs in items:
            if assignments.get(obs)!=role or obs in selected:raise ValueError('subset violates original observation split')
            selected.add(obs)
    return selected


def prepare_dataset(root,out,split_path=None,max_observations=None,seed=42,backend='native',selection='legacy',selection_path=None):
    root,data=load_data(root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    byname={i['file_name']:i for i in data['images']}
    rows=[{'observation_id':Path(name).stem,'date':name[:8],
           'sha256':sha256(root/'train/train_images'/name)} for name in sorted(byname)]
    if split_path:
        supplied=json.loads(Path(split_path).read_text());assignments=supplied['assignments']
        if set(assignments)!={r['observation_id'] for r in rows}:raise ValueError('supplied split must cover every observation exactly')
        if set(assignments.values())!={'train','calibration','holdout'}:raise ValueError('invalid supplied split names')
        for g in observation_groups(rows):
            if len({assignments[o] for o in g})!=1:raise ValueError('supplied split leaks date/content groups')
        if supplied['split_hash']!=content_hash(assignments):raise ValueError('split_hash mismatch')
        split_status='provided_by_team'
    else:
        assignments=grouped_split(rows,seed);split_status='provisional_development_only'
    split={'assignments':assignments,'split_hash':content_hash(assignments),'seed':seed,
           'status':split_status,'grouping':'connected components of observation/date/content sha256'}
    write_json(out/'split_manifest.json',split)
    chosen=select_development_observations(assignments,max_observations,seed) if selection=='hash' and max_observations else None
    if selection_path:
        declared=json.loads(Path(selection_path).read_text())
        if declared['source_split_hash']!=split['split_hash']:raise ValueError('subset source split_hash mismatch')
        chosen=validate_subset_selection(declared,assignments)
    anns=defaultdict(list)
    for a in data['annotations']:anns[a['image_id']].append(a)
    sample_rows=[];rejected=[];selected=set();lists=defaultdict(list)
    # Never access holdout for smoke model selection or even label conversion here.
    for im in sorted(data['images'],key=lambda i:i['id']):
        obs=Path(im['file_name']).stem;partition=assignments[obs]
        if partition=='holdout':continue
        if chosen is not None and obs not in chosen:continue
        if max_observations is not None:
            cap=max_observations if partition=='train' else max(2,max_observations//4)
            already={r['observation_id'] for r in sample_rows if r['split']==partition}
            if obs not in already and len(already)>=cap:continue
        labels=[];bad=[];native=[];native_masks=[]
        for a in anns[im['id']]:
            if backend=='native':
                m=annotation_mask(a,im['height'],im['width'])
                if not m.any():raise ValueError(f'empty annotated instance: {a["id"]}')
                rle=encode(m)
                if not np.array_equal(m,decode(rle)):raise ValueError('native mask RLE roundtrip failed')
                native_masks.append(rle);native.append(1.)
                continue
            audit=polygon_roundtrip(a,im['height'],im['width'],resample=True)
            if not audit['accepted']:bad.append({'instance_annotation_id':a['id'],'reason':audit['reason'],'iou':audit['native_iou']})
            else:
                labels.append('0 '+' '.join(f'{v:.10f}' for pair in audit['polygon'] for v in pair))
                native.append(audit['native_iou'])
        if bad:
            rejected.append({'image_record_id':im['id'],'observation_id':obs,'reasons':bad});continue
        safe=im['id']
        if not all(c.isalnum() or c in '-_' for c in safe):raise ValueError('unsafe annotation record ID')
        img=out/'images'/partition/(safe+'.jpg');lab=out/'labels'/partition/(safe+'.txt')
        img.parent.mkdir(parents=True,exist_ok=True);lab.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'train/train_images'/im['file_name'],img)
        target_path=None
        if backend=='native':
            target_path=(out/'targets'/partition/(safe+'.json')).resolve()
            write_json(target_path,{'size':[im['height'],im['width']],'masks':native_masks,
                                   'instance_annotation_ids':[a['id'] for a in anns[im['id']]]})
        else:write_text(lab,'\n'.join(labels)+'\n' if labels else '')
        lists[partition].append(str(img.resolve()));selected.add(obs)
        sample_rows.append({'sample_id':safe,'annotation_id':im['id'],'observation_id':obs,
                            'split':partition,'instance_count':len(native_masks) if backend=='native' else len(labels),
                            'image_path':str(img.resolve()),'target_path':str(target_path) if target_path else None,
                            'min_conversion_iou':min(native) if native else 1.})
    if not lists['train'] or not lists['calibration']:raise ValueError('not enough conversion-safe train/calibration samples')
    for key,paths in lists.items():write_text(out/(key+'.txt'),'\n'.join(paths)+'\n')
    write_text(out/'dataset.yaml',yaml.safe_dump({'path':str(out.resolve()),'train':'train.txt','val':'calibration.txt',
                                                 'names':{0:'filament'},'m3_target_backend':backend},sort_keys=False))
    write_jsonl(out/'samples.jsonl',sample_rows);write_jsonl(out/'rejected_samples.jsonl',rejected)
    info={'status':'selected_development_subset' if selection_path else 'smoke_subset' if max_observations else 'full_development_export',
          'split_status':split_status,'split_hash':split['split_hash'],'annotation_sha256':sha256(root/ANNOTATION),
          'samples':dict(Counter(r['split'] for r in sample_rows)),'unique_observations':len(selected),
          'rejected_annotation_passes':len(rejected),'conversion_gate':.98,
          'target_backend':backend,'native_roundtrip_iou':1. if backend=='native' else None,
          'subset_selection':selection,
          'production_training_allowed':max_observations is None and not rejected and split_path is not None and not selection_path,
          'selection_manifest_sha256':sha256(selection_path) if selection_path else None,
          'annotation_policy':'each annotation pass is a separate sample; held-out observations never exported',
          'sample_manifest_sha256':sha256(out/'samples.jsonl')}
    write_json(out/'preparation.json',info)
    return info
