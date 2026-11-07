"""Post-freeze diagnostic crops: fixed hash-selected observations, no model selection."""
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from m3_filament.io import read_jsonl,content_hash,write_json,sha256
from m3_filament.masks import decode,bbox
from m3_filament.evaluation import pairwise_iou,pq_counts
from summarize_structure import check_paths,NAMES

OUT=Path('reports/structure_20261104')


def overlay(image,masks,gt,color):
 canvas=image.astype(float).copy()
 for mask in masks:canvas[mask]=.65*canvas[mask]+.35*np.array(color)
 canvas=np.uint8(np.clip(canvas,0,255))
 for mask in gt:
  contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(canvas,contours,-1,(0,255,255),1)
 return canvas


def main():
 result=json.loads((OUT/'results.json').read_text());checks=result['fresh_check']
 assert len(checks)==3 and (OUT/'frozen_selection.json').exists()
 samples=list(read_jsonl('data/structure_fresh_check12/samples.jsonl'));byobs={}
 for sample in sorted(samples,key=lambda r:r['annotation_id']):byobs.setdefault(sample['observation_id'],sample)
 chosen=sorted(byobs,key=lambda o:content_hash({'purpose':'structure-visual-diagnostic','observation_id':o}))[:3]
 caches={name:{r['observation_id']:r for r in read_jsonl(check_paths(name)[1])} for name in checks}
 colors={'latest_quality_reference':[255,60,60],'previous_recall_reference':[255,165,0],'new_candidate':[220,60,255]}
 fig,axes=plt.subplots(len(chosen),4,figsize=(14,3.7*len(chosen)),squeeze=False);metadata=[]
 for row,obs in enumerate(chosen):
  sample=byobs[obs];gt=json.loads(Path(sample['target_path']).read_text())['masks']
  image=np.asarray(Image.open(sample['image_path']).convert('RGB'));h,w=image.shape[:2]
  masks=[decode(rle).astype(bool) for rle in gt]
  if masks:
   focus=min(masks,key=lambda m:int(m.sum()));x0,y0,x1,y1=bbox(focus);cx=(x0+x1)//2;cy=(y0+y1)//2
  else:cx=w//2;cy=h//2
  size=512;left=max(0,min(w-size,cx-size//2));top=max(0,min(h-size,cy-size//2));crop=[left,top,left+size,top+size]
  view=image[top:top+size,left:left+size];cropped_gt=[m[top:top+size,left:left+size] for m in masks]
  axes[row,0].imshow(overlay(view,[],cropped_gt,[0,0,0]));axes[row,0].set_title(obs+'\nGT outline (one annotation)',fontsize=8)
  item={'observation_id':obs,'annotation_id':sample['annotation_id'],'crop_xyxy':crop,'image_sha256':sha256(sample['image_path']),'scope':'fixed hash image selection; crop centered on smallest GT for posthoc visualization only'}
  for col,(name,data) in enumerate(checks.items(),1):
   instances=[i for i in caches[name][obs]['instances'] if i['score']>=data['selected']['confidence']]
   pred=[i['segmentation'] for i in instances];cropped=[decode(r).astype(bool)[top:top+size,left:left+size] for r in pred]
   counts=pq_counts(pairwise_iou(gt,pred));axes[row,col].imshow(overlay(view,cropped,cropped_gt,colors[name]));axes[row,col].set_title(NAMES[name]+f"\nfull-image TP/FP/FN {counts['tp']}/{counts['fp']}/{counts['fn']}",fontsize=8)
   item[name]=counts
  metadata.append(item)
 for ax in axes.flat:ax.axis('off')
 fig.suptitle('Post-freeze diagnostic crops; cyan = independent GT outline, color = predicted binary masks',fontsize=12);fig.tight_layout();fig.savefig(OUT/'diagnostic_crops.png',dpi=180);plt.close(fig)
 write_json(OUT/'diagnostic_crops.json',{'diagnostic_only':True,'used_for_selection':False,'images':metadata,'selection':'first3 by fixed content_hash(purpose, observation_id), not ranked by improvement'})
 print(json.dumps({'images':len(chosen),'observations':chosen}))


if __name__=='__main__':main()
