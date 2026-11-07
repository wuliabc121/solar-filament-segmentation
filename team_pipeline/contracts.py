"""Original-coordinate probabilities with explicit, unmodified training provenance."""
import json,os,tempfile
from pathlib import Path
import numpy as np
from m3_filament.fusion import probability,fuse_record
from m3_filament.io import content_hash

REQUIRED=('model','observation_id','run_id','checkpoint_hash','split_hash','coordinate_space','is_oof','purpose')

def validate_meta(meta):
 if any(k not in meta for k in REQUIRED):raise ValueError('missing probability provenance')
 if meta['model'] not in ('m1','m2') or meta['coordinate_space']!='original':raise ValueError('invalid semantic model/coordinates')
 if not all(isinstance(meta[k],str) and meta[k] for k in ('observation_id','run_id','checkpoint_hash','split_hash')):raise ValueError('empty probability provenance')
 if len(meta['checkpoint_hash'])!=64:raise ValueError('invalid checkpoint SHA256')
 if meta['is_oof'] is not False:raise ValueError('OOF evidence is not supplied by these projects')
 return meta

def save_probability(path,values,meta):
 validate_meta(meta);a=np.asarray(values,dtype=np.float32)
 if a.ndim!=2 or min(a.shape)<1:raise ValueError('expected HxW probability')
 probability(a,a.shape);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=None
 try:
  with tempfile.NamedTemporaryFile(dir=p.parent,suffix='.npz',delete=False) as f:
   tmp=Path(f.name)
   np.savez_compressed(f,probabilities=a,__meta__=json.dumps(meta,sort_keys=True),**{k:meta[k] for k in ('observation_id','run_id','checkpoint_hash','split_hash')})
   f.flush();os.fsync(f.fileno())
  os.link(tmp,p)
 finally:
  if tmp:tmp.unlink(missing_ok=True)

def load_probability(path,observation_id=None,shape=None):
 with np.load(path,allow_pickle=False) as z:
  m=validate_meta(json.loads(str(z['__meta__'])));a=z['probabilities'].astype(np.float32)
  for k in ('observation_id','run_id','checkpoint_hash','split_hash'):
   if str(z[k])!=m[k]:raise ValueError('conflicting probability metadata')
 if observation_id is not None and m['observation_id']!=observation_id:raise ValueError('observation mismatch')
 probability(a,shape or a.shape)
 if a.ndim!=2:raise ValueError('expected HxW probability')
 return a,m

def check_evaluation_eligibility(observations,models):
 for name,m in models.items():
  if m['training_membership_status'] not in ('manifest_verified','local_logged'):raise ValueError('evaluation blocked: unattested training membership for '+name)
  if set(observations)&set(m['training_observations']):raise ValueError('evaluation blocked: trained-on observations for '+name)

def fuse_team(record,cnn,transformer,config,run_id,purpose):
 if purpose not in ('smoke','application'):raise ValueError('evaluation fusion is blocked until a common attested protocol is provided')
 components={}
 for expected,(a,m) in zip(('m1','m2'),(cnn,transformer)):
  validate_meta(m)
  if m['model']!=expected or m['observation_id']!=record['observation_id']:raise ValueError('semantic component identity mismatch')
  if purpose=='smoke' and m['purpose']!='smoke':raise ValueError('smoke requires synthetic components')
  if purpose=='application' and m['purpose']=='smoke':raise ValueError('synthetic cache cannot be used for application')
  probability(a,record['image_size']);components[expected]=m
 out=fuse_record(record,cnn[0],transformer[0],config,run_id)
 out['source'].update(semantic_components=components,instance_component=record['source'],is_oof=False,purpose=purpose,benchmark_eligible=False,inference_cohort_hash=content_hash([record['observation_id']]))
 for inst in out['instances']:inst['source']=dict(out['source'])
 return out
