import json
from collections import defaultdict
from pathlib import Path
from . import ROOT
from team_pipeline.vendor import m2_modules

def ground_truth(observations):
 # Read only selected annotator masks; records remain independent.
 raw=json.loads((ROOT/'data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text());lookup={str(i['id']):Path(i['file_name']).stem for i in raw['images']};selected=set(observations);out={s:{} for s in observations}
 for i in raw['images']:
  if lookup[str(i['id'])] in selected:out[lookup[str(i['id'])]][str(i['id'])]=[]
 train,_=m2_modules()
 for ann in raw['annotations']:
  rid=str(ann['image_id']);obs=lookup[rid]
  if obs not in selected:continue
  r=train.polygon_rle(ann);out[obs][rid].append({'size':list(r['size']),'counts':r['counts'].decode('ascii') if isinstance(r['counts'],bytes) else r['counts']})
 return out
