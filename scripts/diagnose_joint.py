"""Post-hoc diagnosis of frozen results; no selection or threshold changes."""
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as coco
from m3_filament.evaluation import pairwise_iou,native_rle
from m3_filament.io import read_jsonl,write_json,sha256

OUT=Path('reports/balanced_20261104')


def score_masks(gt,pred):
    matrix=pairwise_iou(gt,pred)
    best_gt=matrix.max(axis=1) if pred else np.zeros(len(gt))
    best_pred=matrix.max(axis=0) if gt else np.zeros(len(pred))
    g=coco.merge([native_rle(r) for r in gt]) if gt else None
    p=coco.merge([native_rle(r) for r in pred]) if pred else None
    ga=float(coco.area(g)) if g else 0.;pa=float(coco.area(p)) if p else 0.
    inter=float(coco.area(coco.merge([g,p],intersect=True))) if g and p else 0.
    union=ga+pa-inter
    result={'gt_instances':len(gt),'pred_instances':len(pred),'gt_matched':int((best_gt>.5).sum()),
      'fn_zero_overlap':int((best_gt==0).sum()),'fn_overlap_to_0p1':int(((best_gt>0)&(best_gt<=.1)).sum()),
      'fn_0p1_to_0p3':int(((best_gt>.1)&(best_gt<=.3)).sum()),
      'fn_0p3_to_0p5':int(((best_gt>.3)&(best_gt<=.5)).sum()),
      'fp_zero_overlap':int((best_pred==0).sum()),'fp_partial_overlap':int(((best_pred>0)&(best_pred<=.5)).sum()),
      'sum_best_gt_iou':float(best_gt.sum()),'gt_union_area':ga,'pred_union_area':pa,'intersection_area':inter,
      'foreground_iou':inter/union if union else 1.,'foreground_dice':2*inter/(ga+pa) if ga+pa else 1.}
    return result


def main():
    result=json.loads((OUT/'results.json').read_text());checks=result['new_check']
    if len(checks)!=3:raise RuntimeError('three frozen comparisons required; no new inference performed')
    sample_rows=list(read_jsonl('data/balanced_check24/samples.jsonl'));summary={}
    for name,data in checks.items():
        path=Path(data['metrics_path']);pred_path=path.parent.with_name(path.parent.name+'_predictions')/'instances.jsonl'
        # Raw reference evaluation keeps predictions in that sibling folder;
        # postprocessed candidate uses the same published contract/path.
        records={r['observation_id']:r for r in read_jsonl(pred_path)}
        confidence=data['selected']['confidence'];rows=[]
        for sample in sample_rows:
            gt=json.loads(Path(sample['target_path']).read_text())['masks']
            pred=[i['segmentation'] for i in records[sample['observation_id']]['instances'] if i['score']>=confidence]
            rows.append(score_masks(gt,pred)|{'observation_id':sample['observation_id'],'annotation_id':sample['annotation_id']})
        totals={key:sum(row[key] for row in rows) for key in rows[0] if key not in ('observation_id','annotation_id','foreground_iou','foreground_dice')}
        ga=totals['gt_union_area'];pa=totals['pred_union_area'];inter=totals['intersection_area']
        summary[name]={'counts':totals,'foreground_micro_iou':inter/(ga+pa-inter) if ga+pa-inter else 1.,
           'foreground_micro_dice':2*inter/(ga+pa) if ga+pa else 1.,
           'mean_annotation_iou':float(np.mean([x['foreground_iou'] for x in rows])),
           'mean_annotation_dice':float(np.mean([x['foreground_dice'] for x in rows])),
           'mean_best_gt_iou_including_misses':totals['sum_best_gt_iou']/totals['gt_instances'],
           'annotation_passes_including_zero_GT':len(rows),'predictions_sha256':sha256(pred_path),
           'threshold_frozen':confidence}
    write_json(OUT/'auxiliary_metrics.json',{'diagnostic_only':True,'used_for_selection':False,
        'scope':'same already-evaluated frozen check; pixel union metrics are auxiliary, not official PQ',
        'frozen_selection_sha256':sha256(OUT/'frozen_selection.json'),'models':summary})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
