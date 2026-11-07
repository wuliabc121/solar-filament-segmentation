import math,json,random
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone
from . import ROOT,REPORT
from m3_filament.io import sha256

def make_split(images,ratio=.2,seed=42):
 if type(ratio) not in (int,float) or not math.isfinite(ratio) or not 0<ratio<1:raise ValueError('ratio must be in (0,1)')
 records=defaultdict(list);seen=set()
 for row in images:
  rid=str(row['id']);obs=Path(row['file_name']).stem
  if rid in seen:raise ValueError('duplicate annotation record id')
  seen.add(rid);records[obs].append(rid)
 ids=sorted(records);n=round(len(ids)*ratio)
 if n<1 or n>=len(ids):raise ValueError('empty train/validation partition')
 selected=sorted(random.Random(seed).sample(ids,n));train=sorted(set(ids)-set(selected))
 return {'ratio_requested':ratio,'seed':seed,'sampler':'random.Random(seed).sample(sorted(unique_image_stems),round(n*ratio))','sampling_unit':'observation/image, all annotator records stay together','validation_ids':selected,'train_ids':train,'annotation_records':{s:sorted(records[s]) for s in ids},'assignments':{s:'validation' if s in selected else 'train' for s in ids},'actual_ratio':len(selected)/len(ids)}

def overlap_report(ids,models,data):
 out={}
 for name,model in models.items():
  trained=set(model['training_observations']);dates={data[s]['date'] for s in trained};hashes={data[s]['sha256'] for s in trained};exact=sorted(set(ids)&trained);group=sorted(s for s in ids if data[s]['date'] in dates or data[s]['sha256'] in hashes)
  out[name]={'scored_observations':len(ids),'exact_training_overlap_count':len(exact),'exact_training_overlap_ids':exact,'date_or_content_training_overlap_count':len(group),'date_or_content_training_overlap_ids':group,'training_membership_status':model['training_membership_status']}
 return out

def load():
 p=json.loads((REPORT/'protocol.json').read_text());split=json.loads((ROOT/p['split_file']).read_text())
 if sha256(ROOT/p['split_file'])!=p['split_sha256']:raise ValueError('split changed after freeze')
 if split['validation_ids']!=p['validation_ids']:raise ValueError('validation IDs changed')
 return p

def deadline(p):
 if datetime.now(timezone.utc)>=datetime.fromisoformat(p['deadline_utc']):raise TimeoutError('random20 campaign wallclock deadline reached')
