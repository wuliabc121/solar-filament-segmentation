"""Independent-annotator agreement on training images only; not a model ceiling."""
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from m3_filament.io import read_jsonl,write_json,sha256
from m3_filament.evaluation import pairwise_iou,pq_counts,aggregate_counts


def main():
 groups=defaultdict(list)
 for row in read_jsonl('data/overnight176/samples.jsonl'):
  if row['split']=='train':groups[row['observation_id']].append(row)
 rows=[]
 for obs,passes in groups.items():
  if len(passes)<2:continue
  for a,b in combinations(passes,2):
   ga=json.loads(Path(a['target_path']).read_text())['masks'];gb=json.loads(Path(b['target_path']).read_text())['masks']
   if not ga or not gb:continue
   matrix=pairwise_iou(ga,gb);counts=pq_counts(matrix)
   rows.append({'observation_id':obs,'annotation_a':a['annotation_id'],'annotation_b':b['annotation_id'],**counts,'n_a':len(ga),'n_b':len(gb)})
 total=aggregate_counts(rows)
 result={'scope':'training176 only; pairwise annotation agreement diagnostic, not model prediction, not upper bound or official score',
  'overlapping_observations':len({r['observation_id'] for r in rows}),'annotation_pairs':len(rows),'pooled_pairwise_agreement':total,
  'sample_manifest_sha256':sha256('data/overnight176/samples.jsonl'),'interpretation':'Annotation disagreement exists; this does not establish a performance ceiling or explain away model errors. Never union annotations for training/evaluation.'}
 write_json('reports/structure_20261104/training_agreement.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
