"""Fast RLE implementation aligned to the organizer's public notebook v6.

This deliberately preserves the notebook's all-IoU>0.5-pairs matching,
including its behavior for overlapping objects; it is not Hungarian matching.
Scores describe the explicitly selected development observations only.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as coco
from .io import read_jsonl,write_json,write_jsonl,sha256,content_hash
from .masks import _runs
from .schema import validate_record


def native_rle(rle):
    h,w=rle['size'];runs=_runs(rle['counts'])
    if min(h,w)<=0 or sum(runs)!=h*w:raise ValueError('invalid evaluation RLE')
    return coco.frPyObjects({'size':[h,w],'counts':runs},h,w)


def pairwise_iou(gt,pred):
    if not gt or not pred:return np.zeros((len(gt),len(pred)),np.float32)
    shape=gt[0]['size']
    if any(r['size']!=shape for r in gt+pred):raise ValueError('evaluation mask shape mismatch')
    g=[native_rle(r) for r in gt];p=[native_rle(r) for r in pred]
    return np.asarray(coco.iou(p,g,[0]*len(g)),np.float32).T


def pq_counts(matrix):
    a=np.asarray(matrix,dtype=np.float32)
    if a.ndim!=2 or not np.isfinite(a).all() or (a<0).any() or (a>1).any():raise ValueError('invalid IoU matrix')
    hits=a>.5
    return {'tp':int(hits.sum()),'fp':int((~hits.any(axis=0)).sum()),
            'fn':int((~hits.any(axis=1)).sum()),'tp_iou_sum':sum(a[hits].tolist())}


def aggregate_counts(rows):
    counts={k:sum(r[k] for r in rows) for k in ('tp','fp','fn','tp_iou_sum')}
    denom=counts['tp']+.5*(counts['fp']+counts['fn'])
    return {**counts,'pq':counts['tp_iou_sum']/denom if denom else 0.,
            'sq':counts['tp_iou_sum']/counts['tp'] if counts['tp'] else 0.,
            'rq':counts['tp']/denom if denom else 0.}


def validate_evaluation_provenance(records,info,manifest,thresholds):
    if not records:raise ValueError('no prediction records')
    source=records[0]['source']
    fields=('checkpoint_sha256','inference_config_sha256','source_manifest_sha256','training_run_id')
    known=set(manifest.get('checkpoint_hashes',{}).values())|{manifest.get('checkpoint_sha256')}
    if not source.get('checkpoint_sha256') or source['checkpoint_sha256'] not in known:
        raise ValueError('checkpoint provenance is not in detector training manifest')
    if source.get('training_run_id')!=manifest.get('run_id'):raise ValueError('training run provenance mismatch')
    trained=set(manifest['training_observation_ids'])
    for r in records:
        s=r['source']
        if any(not s.get(k) or s[k]!=source.get(k) for k in fields):raise ValueError('mixed evaluation checkpoint/config/code provenance')
        if s.get('split_hash')!=info['split_hash']:raise ValueError('evaluation split_hash mismatch')
        if s.get('unseen_check_passed')is not True:raise ValueError('unseen prediction check was not verified')
        if r['observation_id'] in trained:raise ValueError('detector was trained on evaluation observation')
        if content_hash(s.get('inference_config'))!=s.get('inference_config_sha256'):raise ValueError('inference configuration hash mismatch')
    floor=source['inference_config'].get('conf')
    if type(floor)not in (int,float) or any(t<floor for t in thresholds):
        raise ValueError('calibration threshold is below the cached inference confidence floor')
    return {k:source[k] for k in fields}|{'inference_floor':floor,'inference_config':source['inference_config']}


def ensure_eval_images(dataset):
    dataset=Path(dataset);out=dataset/'eval_images';expected={}
    for r in read_jsonl(dataset/'samples.jsonl'):
        if r['split']!='calibration':continue
        obs=r['observation_id']
        if not obs or not all(c.isalnum() or c in '-_' for c in obs):raise ValueError('unsafe observation ID')
        path=Path(r['image_path']);digest=sha256(path)
        if obs in expected and expected[obs][1]!=digest:raise ValueError('annotation passes disagree on original image')
        expected[obs]=(path,digest)
    if not expected:raise ValueError('no calibration observations')
    if out.exists():
        files={p.stem:p for p in out.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
        if set(files)!=set(expected) or any(sha256(files[o])!=v[1] for o,v in expected.items()):
            raise ValueError('existing evaluation images differ from selected calibration source')
        return out
    out.mkdir()
    for obs,(path,digest) in expected.items():shutil.copy2(path,out/(obs+'.jpg'))
    return out


def evaluate_records(records,dataset,thresholds,output):
    dataset=Path(dataset);out=Path(output)
    if out.exists():raise FileExistsError(out)
    sample_rows=[r for r in read_jsonl(dataset/'samples.jsonl') if r['split']=='calibration']
    expected={r['observation_id'] for r in sample_rows}
    byobs={}
    for record in records:
        validate_record(record,exclusive=True)
        obs=record['observation_id']
        if obs in byobs:raise ValueError('duplicate observation in evaluation')
        byobs[obs]=record
    if set(byobs)!=expected:raise ValueError('evaluation must cover exactly the selected calibration observations')
    if any(type(x)not in (int,float) or not 0<=x<=1 for x in thresholds):raise ValueError('invalid confidence threshold')
    info=json.loads((dataset/'preparation.json').read_text())
    run_id=records[0]['source'].get('training_run_id','')
    if not run_id or not all(c.isalnum() or c in '-_' for c in run_id):raise ValueError('safe detector training run_id required')
    manifest=json.loads((Path('runs')/run_id/'run_manifest.json').read_text())
    provenance=validate_evaluation_provenance(records,info,manifest,thresholds)
    prepared=[];empty_gt=[]
    for row in sample_rows:
        gt=json.loads(Path(row['target_path']).read_text())['masks']
        if not gt:
            # The official dataframe loop iterates annotator IDs present in
            # GT filament rows, so records with zero GT rows are absent.
            empty_gt.append(row['sample_id']);continue
        instances=byobs[row['observation_id']]['instances']
        matrix=pairwise_iou(gt,[i['segmentation'] for i in instances])
        prepared.append((row,gt,instances,matrix))
    grid=[];all_rows=[]
    for threshold in thresholds:
        rows=[]
        for sample,gt,instances,matrix in prepared:
            keep=np.array([i['score']>=threshold for i in instances],bool)
            a=matrix[:,keep];counts=pq_counts(a)
            areas=np.array([float(coco.area(native_rle(r))) for r in gt])
            matched=(a>.5).any(axis=1);overlap=a>0
            small=areas<1000
            row={'confidence':threshold,'observation_id':sample['observation_id'],
                 'annotation_id':sample['annotation_id'],'n_gt':len(gt),'n_pred':int(keep.sum()),**counts,
                 'small_gt':int(small.sum()),'small_matched':int((small&matched).sum()),
                 'fragmented_gt':int((overlap.sum(axis=1)>1).sum()),
                 'merged_pred':int((overlap.sum(axis=0)>1).sum())}
            rows.append(row);all_rows.append(row)
        total=aggregate_counts(rows)
        total.update({'confidence':threshold,'n_gt':sum(r['n_gt'] for r in rows),
                      'n_pred_across_annotators':sum(r['n_pred'] for r in rows),
                      'small_gt':sum(r['small_gt'] for r in rows),'small_matched':sum(r['small_matched'] for r in rows),
                      'fragmented_gt':sum(r['fragmented_gt'] for r in rows),'merged_pred':sum(r['merged_pred'] for r in rows)})
        grid.append(total)
    best=max(grid,key=lambda r:(r['pq'],-r['fp'],r['confidence']))
    source_path=Path('references/official-self-evaluation-v6.py')
    report={'metric':'organizer_notebook_v6_aligned_development_pq','official_leaderboard_score':False,
            'official_notebook_url':'https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook',
            'official_source_sha256':sha256(source_path),'matching':'all pairs with float32 IoU > 0.5; pooled per annotator',
            'split_hash':json.loads((dataset/'preparation.json').read_text())['split_hash'],
            'sample_manifest_sha256':sha256(dataset/'samples.jsonl'),'observations':sorted(expected),
            'annotation_passes_scored':len(prepared),'zero_gt_records_excluded_like_source':empty_gt,
            'checkpoint_sha256':provenance['checkpoint_sha256'],'validated_provenance':provenance,
            'calibration_selection':True,'holdout_used':False,'grid':grid,'selected':best}
    out.mkdir(parents=True);write_json(out/'metrics.json',report);write_jsonl(out/'per_annotation.jsonl',all_rows)
    grouped=defaultdict(list)
    for row in all_rows:
        if row['confidence']==best['confidence']:grouped[row['observation_id']].append(row)
    write_jsonl(out/'per_observation.jsonl',[{'observation_id':o,**aggregate_counts(rows)} for o,rows in sorted(grouped.items())])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--instances');inputs.add_argument('--weights')
    parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--imgsz',type=int,default=512);parser.add_argument('--device',default='cpu')
    parser.add_argument('--thresholds',type=float,nargs='+',default=[.001,.003,.01,.03,.1])
    args=parser.parse_args()
    instances=args.instances
    if args.weights:
        from .model import predict
        prediction_dir=str(Path(args.output).with_name(Path(args.output).name+'_predictions'))
        predict(args.weights,str(ensure_eval_images(args.dataset)),prediction_dir,
                device=args.device,imgsz=args.imgsz,conf=min(args.thresholds),require_unseen=True,max_det=100)
        instances=str(Path(prediction_dir)/'instances.jsonl')
    result=evaluate_records(list(read_jsonl(instances)),args.dataset,args.thresholds,args.output)
    print(json.dumps({'selected':result['selected'],'observations':len(result['observations']),'annotations':result['annotation_passes_scored']},indent=2))


if __name__=='__main__':main()
