import json,time,copy
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
import torch
from . import ROOT,REPORT
from .protocol import load,deadline
from .caches import original_m2
from m3_filament.io import read_jsonl,write_json,sha256,content_hash
from m3_filament.model import predict
from team_pipeline.integrity import verified_checkpoint
from team_pipeline.models import load_model,predict_m1,predict_m2
from team_pipeline.contracts import load_probability,save_probability

def state(job,status,**extra):
 path=REPORT/'status'/(job+'.json');path.parent.mkdir(parents=True,exist_ok=True);value={'job':job,'status':status,'updated_utc':datetime.now(timezone.utc).isoformat(),**extra};tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def m3_inference(key):
 p=load();deadline(p);torch.set_num_threads(4);torch.manual_seed(42);model=p['models'][key];weights,actual=verified_checkpoint(model);out=REPORT/'cache'/key;state(key,'running',completed_forward=0,total_images=len(p['validation_ids']),checkpoint_sha256=actual)
 from ultralytics import YOLO
 original=YOLO.predict;count=0;start=time.monotonic()
 def traced(self,*args,**kwargs):
  nonlocal count
  deadline(p);result=original(self,*args,**kwargs);count+=1;state(key,'running',completed_forward=count,total_images=len(p['validation_ids']),seconds=time.monotonic()-start);return result
 YOLO.predict=traced
 try:summary=predict(str(weights),str(REPORT/'input_images'),str(out),device='cpu',imgsz=2048,conf=.05,max_det=100,require_unseen=False)
 finally:YOLO.predict=original
 records=list(read_jsonl(out/'instances.jsonl'))
 if sorted(r['observation_id'] for r in records)!=p['validation_ids']:raise ValueError('M3 cohort mismatch')
 if any(r['source']['checkpoint_sha256']!=actual or r['source']['inference_config']!=p['M3_raw_config'] or r['source']['inference_config_sha256']!=content_hash(p['M3_raw_config']) for r in records):raise ValueError('M3 raw inference identity/config mismatch')
 metrics={'status':'complete','model':key,'checkpoint_sha256':actual,'cohort_sha256':p['split_sha256'],'images':len(records),'instance_count':summary['instances'],'seconds':time.monotonic()-start,'known_training_overlap_count':p['training_overlap'][key]['exact_training_overlap_count'],'diagnostic_not_unseen':True,'original_predict_function_mathematics_unchanged':True};write_json(out/'metrics.json',metrics);state(key,'complete',**{k:v for k,v in metrics.items() if k!='status'});print(json.dumps(metrics,indent=2))

def semantic_inference():
 p=load();deadline(p);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);data=json.loads((REPORT/'data_inventory.json').read_text());device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');out=REPORT/'cache';out.mkdir(exist_ok=True);(out/'m1').mkdir();(out/'m2').mkdir();counts={'m1_fresh':0,'m1_reused':0,'m2_fresh':0,'m2_reused':0};manifest=[];start=time.monotonic();state('semantics','running',phase='m1',device=str(device),completed_images=0,total_images=len(p['validation_ids']))
 # M1 existing verified canonical probabilities, otherwise fresh exact RGB512 pipeline.
 m1,info=load_model('m1',device)
 for index,s in enumerate(p['validation_ids']):
  deadline(p);old=ROOT/'reports/team_adaptation_20261104/cache/m1'/(s+'.npz');target=out/'m1'/(s+'.npz');reused=False
  if old.exists():
   a,m=load_probability(old,s,(data[s]['height'],data[s]['width']))
   if m['checkpoint_hash']!=info['checkpoint_sha256'] or m.get('image_sha256')!=data[s]['sha256']:raise ValueError('M1 reused cache identity mismatch')
   m=dict(m,purpose='random20_mixed_training_development_diagnostic',reused_probability_file=str(old.relative_to(ROOT)),reused_probability_sha256=sha256(old));reused=True
  else:
   image=cv2.imread(str(ROOT/'data/raw'/data[s]['relative_path']),cv2.IMREAD_COLOR)
   if image is None:raise ValueError('unreadable image')
   a=predict_m1(m1,cv2.cvtColor(image,cv2.COLOR_BGR2RGB),device);m={'model':'m1','observation_id':s,'run_id':'random20_M1_fresh','checkpoint_hash':info['checkpoint_sha256'],'split_hash':info['source_split_hash'],'coordinate_space':'original','is_oof':False,'purpose':'random20_mixed_training_development_diagnostic','training_membership_status':info['training_membership_status'],'image_sha256':data[s]['sha256'],'inference':{'RGB':True,'resize':[512,512],'normalization':'ImageNet','probability':'sigmoid','restore':'cv2 bilinear','device':str(device)}}
  save_probability(target,a,m);counts['m1_reused' if reused else 'm1_fresh']+=1;manifest.append({'model':'m1','observation_id':s,'path':str(target.relative_to(ROOT)),'sha256':sha256(target),'reused':reused});state('semantics','running',phase='m1',completed_images=index+1,total_images=len(p['validation_ids']),device=str(device),counts=counts,seconds=time.monotonic()-start)
 del m1
 if device.type=='mps':torch.mps.empty_cache()
 # Adopt actual original M2 calib AND holdout caches; user now requested random-all-labeled diagnostic.
 packed=list((ROOT/'team/m2/artifacts').rglob('semantic_probs_*.npz'));banks=[(f,np.load(f,allow_pickle=False),sha256(f)) for f in packed];lookup={s:(f,z,h) for f,z,h in banks for s in z.files if s!='__meta__'};missing=[]
 try:
  for s in p['validation_ids']:
   deadline(p)
   if s not in lookup:missing.append(s);continue
   f,z,h=lookup[s];a,m=original_m2(z,s,p['models']['m2'],(data[s]['height'],data[s]['width']));m.update(source_cache_sha256=h,source_cache_path=str(f.relative_to(ROOT)),image_identity_evidence='source observation stem and annotation dataset match; source image SHA not supplied');target=out/'m2'/(s+'.npz');save_probability(target,a,m);counts['m2_reused']+=1;manifest.append({'model':'m2','observation_id':s,'path':str(target.relative_to(ROOT)),'sha256':sha256(target),'reused':True,'original_subset':m['original_subset_preserved']})
 finally:
  for _,z,_ in banks:z.close()
 m2,info=load_model('m2',device)
 for index,s in enumerate(missing):
  deadline(p);path=ROOT/'data/raw'/data[s]['relative_path'];gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
  if gray is None:raise ValueError('unreadable image')
  a=predict_m2(m2,gray,device,tile=512,overlap=128,batch=1);m={'model':'m2','observation_id':s,'run_id':'random20_M2_native_fresh','checkpoint_hash':info['checkpoint_sha256'],'split_hash':info['source_split_hash'],'coordinate_space':'original','is_oof':False,'purpose':'random20_mixed_training_development_diagnostic','training_membership_status':info['training_membership_status'],'image_sha256':data[s]['sha256'],'inference':{'read':'cv2.IMREAD_GRAYSCALE','tile':512,'overlap':128,'batch':1,'blend':'Hann','device':str(device),'dtype':'float32','zero_below':None}}
  target=out/'m2'/(s+'.npz');save_probability(target,a,m);counts['m2_fresh']+=1;manifest.append({'model':'m2','observation_id':s,'path':str(target.relative_to(ROOT)),'sha256':sha256(target),'reused':False});state('semantics','running',phase='m2',completed_images=counts['m2_reused']+index+1,total_images=len(p['validation_ids']),device=str(device),counts=counts,seconds=time.monotonic()-start)
 del m2
 for key in ('m1','m2'):
  if sorted(x.stem for x in (out/key).glob('*.npz'))!=p['validation_ids']:raise ValueError('semantic cohort mismatch')
 write_json(out/'semantic_manifest.json',{'files':manifest,'counts':counts,'source_precision_mixed_and_preserved':True,'source_subset_never_renamed':True,'no_OOF_or_unseen_claim':True,'device':str(device)})
 metrics={'status':'complete','images_per_model':len(p['validation_ids']),**counts,'seconds':time.monotonic()-start,'device':str(device),'actual_model_checkpoints':{k:p['models'][k]['checkpoint_sha256'] for k in ('m1','m2')},'diagnostic_not_unseen':True};write_json(out/'semantic_metrics.json',metrics);state('semantics','complete',**{k:v for k,v in metrics.items() if k!='status'});print(json.dumps(metrics,indent=2))
