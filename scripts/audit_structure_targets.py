"""Training-label-only supervision audit; never reads calibration/final-check labels."""
import json
from pathlib import Path
import cv2
import numpy as np
from m3_filament.io import read_jsonl,write_json,sha256
from m3_filament.masks import decode


def main():
 rows=[r for r in read_jsonl('data/overnight176/samples.jsonl') if r['split']=='train']
 stats={'annotation_passes':len(rows),'instances':0,'original_positive_pixels':0,'lost_at_input1024_nearest':0,
 'input_positive_but_lost_at_proto256_nearest':0,'input_positive_but_lost_at_proto256_area':0,'small_original_instances':0,
 'small_lost_at_proto256_nearest':0,'small_lost_at_proto256_area':0}
 examples=[]
 for row in rows:
  for index,rle in enumerate(json.loads(Path(row['target_path']).read_text())['masks']):
   mask=decode(rle);original_area=int(mask.sum());image_target=cv2.resize(mask,(1024,1024),interpolation=cv2.INTER_NEAREST)
   nn=cv2.resize(image_target,(256,256),interpolation=cv2.INTER_NEAREST)
   area=image_target.reshape(256,4,256,4).mean(axis=(1,3))
   stats['instances']+=1;stats['original_positive_pixels']+=original_area
   input_count=int(image_target.sum());missing_nn=not nn.any();missing_area=not area.any()
   stats['lost_at_input1024_nearest']+=int(input_count==0)
   stats['input_positive_but_lost_at_proto256_nearest']+=int(input_count>0 and missing_nn)
   stats['input_positive_but_lost_at_proto256_area']+=int(input_count>0 and missing_area)
   if original_area<1000:
    stats['small_original_instances']+=1;stats['small_lost_at_proto256_nearest']+=int(missing_nn);stats['small_lost_at_proto256_area']+=int(missing_area)
   if input_count>0 and missing_nn and len(examples)<12:examples.append({'sample':row['sample_id'],'instance_index':index,'original_area':original_area,'input_positive':input_count,'area_mass_rescaled':float(area.sum()*16)})
 result={'scope':'training GT only; deterministic full-image resizing, not random crop training distribution','stats':stats,
  'examples':examples,'sample_manifest_sha256':sha256('data/overnight176/samples.jsonl'),
  'interpretation':'Area occupancy preserves input foreground mass; it cannot restore pixels already lost at input resize, nor guarantee improved detector recall/PQ.'}
 write_json('reports/structure_20261104/target_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
