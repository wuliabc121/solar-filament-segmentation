import argparse,json,os,time
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
import torch
from . import ROOT
from .protocol import load_protocol,check_deadline
from .data import ground_truth
from .metrics import score_records
from .baselines import frozen_m3,semantic_components,initial_fusion
from m3_filament.io import read_jsonl,write_jsonl,write_json,sha256
from m3_filament.model import predict
from team_pipeline.models import load_model,predict_m1,predict_m2
from team_pipeline.cache import convert_m2_cache
from team_pipeline.contracts import save_probability,load_probability

REPORT=ROOT/'reports/team_adaptation_20261104'

def update_state(phase,status,**fields):
 p=REPORT/'state.json';old=json.loads(p.read_text()) if p.exists() else {};old.update(phase=phase,status=status,updated_utc=datetime.now(timezone.utc).isoformat(),**fields);p.write_text(json.dumps(old,indent=2)+'\n')

def prepare_predictions(protocol,data):
 output=REPORT/'cache';output.mkdir(exist_ok=True);images=output/'images';images.mkdir(exist_ok=True)
 for s in protocol['primary_ids']:
  p=ROOT/'data/raw'/data[s]['relative_path'];target=images/p.name
  if sha256(p)!=data[s]['sha256']:raise ValueError('input image differs from audited data')
  if not target.exists():os.link(p,target)
 update_state('A','running_M3_inference',active=True)
 m3out=output/'m3_raw'
 if not m3out.exists():
  check_deadline(protocol);predict(str(ROOT/protocol['models']['m3']['checkpoint']),str(images),str(m3out),device='cpu',imgsz=2048,conf=.05,require_unseen=True,max_det=100)
 # Cache conversion preserves true external split hash/quantization.
 cache=next((ROOT/'team/m2/artifacts').rglob('semantic_probs_calib*.npz'));registry=json.loads((ROOT/'team/registry.json').read_text())
 if not (output/'m2').exists():convert_m2_cache(cache,output/'m2',protocol['primary_ids'],registry)
 update_state('A','running_M1_inference',active=True)
 dest=output/'m1';dest.mkdir(exist_ok=True);dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model,info=load_model('m1',dev)
 for s in protocol['primary_ids']:
  check_deadline(protocol);p=ROOT/'data/raw'/data[s]['relative_path'];out=dest/(s+'.npz')
  if out.exists():continue
  image=cv2.imread(str(p),cv2.IMREAD_COLOR);prob=predict_m1(model,cv2.cvtColor(image,cv2.COLOR_BGR2RGB),dev)
  save_probability(out,prob,{'model':'m1','observation_id':s,'run_id':'team_adaptation_M1','checkpoint_hash':info['checkpoint_sha256'],'split_hash':info['source_split_hash'],'coordinate_space':'original','is_oof':False,'purpose':'development_diagnostic','training_membership_status':info['training_membership_status'],'image_sha256':data[s]['sha256']})
 del model
 # One native original-image M2 pass checks actual local implementation against packed source probabilities.
 s=protocol['tune_ids'][0];check_deadline(protocol);model,info=load_model('m2',dev);gray=cv2.imread(str(ROOT/'data/raw'/data[s]['relative_path']),cv2.IMREAD_GRAYSCALE)
 started=time.monotonic();prob=predict_m2(model,gray,dev);cached,meta=load_probability(output/'m2'/(s+'.npz'),s,gray.shape);diff=np.abs(prob-cached)
 save_probability(output/'m2_local_verification.npz',prob,{'model':'m2','observation_id':s,'run_id':'team_adaptation_M2_native_verify','checkpoint_hash':info['checkpoint_sha256'],'split_hash':info['source_split_hash'],'coordinate_space':'original','is_oof':False,'purpose':'development_diagnostic','training_membership_status':info['training_membership_status']})
 write_json(output/'M2_local_cache_agreement.json',{'observation_id':s,'device':str(dev),'native_input':'cv2.IMREAD_GRAYSCALE','source_cache_dtype':'float16','source_zero_below':.001,'mean_abs':float(diff.mean()),'max_abs':float(diff.max()),'p99_abs':float(np.quantile(diff,.99)),'pixels_threshold035_differ':int(((prob>=.35)!=(cached>=.35)).sum()),'seconds':time.monotonic()-started,'not_bitwise_equivalence':True})
 if float(diff.mean())>.01:raise ValueError('M2 local/cache mean difference exceeds .01; inspect before evaluation')
 write_json(output/'manifest.json',{'protocol_sha256':sha256(REPORT/'protocol.json'),'M2_source_cache_sha256':sha256(cache),'models':{k:v['checkpoint_sha256'] for k,v in protocol['models'].items()},'data_ids':protocol['primary_ids'],'local_device':str(dev),'M1_overlapping_training_on_most_cohort':True,'no_GT_masks_read':True})

def baseline_phase(protocol,data):
 prepare_predictions(protocol,data);check_deadline(protocol);update_state('A','scoring_tune_baselines',active=True);ids=protocol['tune_ids'];g=ground_truth(ids);raw={r['observation_id']:r for r in read_jsonl(REPORT/'cache/m3_raw/instances.jsonl')};outputs={k:[] for k in ('M3_frozen','M2_semantic','E5_M3_M2_initial','E5_both_M1_contaminated_diagnostic')}
 for s in ids:
  rec=raw[s];pt,_=load_probability(REPORT/'cache/m2'/(s+'.npz'),s,rec['image_size']);pc,_=load_probability(REPORT/'cache/m1'/(s+'.npz'),s,rec['image_size'])
  outputs['M3_frozen'].append(frozen_m3(rec));outputs['M2_semantic'].append(semantic_components(rec,pt));outputs['E5_M3_M2_initial'].append(initial_fusion(rec,pt));outputs['E5_both_M1_contaminated_diagnostic'].append(initial_fusion(rec,pt,pc))
 out=REPORT/'baseline_tune';out.mkdir();metrics={}
 for k,records in outputs.items():
  m,rows=score_records(records,g,ids);metrics[k]=m;write_jsonl(out/(k+'.jsonl'),records);write_jsonl(out/(k+'_per_annotation.jsonl'),rows)
 metrics['scope']='tune_only_reused_M2_calibration; M1contaminated route not eligible performance evidence';write_json(out/'metrics.json',metrics);update_state('A','complete',active=False,check_scores_read=False);print(json.dumps(metrics,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['baseline','tune','associate-tune','check','apply','export-corrected']);p.add_argument('--protocol',default=str(REPORT/'protocol.json'));p.add_argument('--instances');p.add_argument('--m1');p.add_argument('--m2');p.add_argument('--profile',default=str(REPORT/'frozen_adaptation_v2.json'));p.add_argument('--mode',choices=['m1','m2','both','guided_both'],default='m2');p.add_argument('--run-id',default='M3_semantic_adapted');p.add_argument('--output');args=p.parse_args()
 if args.phase=='apply':
  if not args.instances or not args.output:p.error('apply requires --instances and --output')
  from .application import apply_files
  apply_files(args);return
 protocol,data=load_protocol(args.protocol);check_deadline(protocol);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
 if args.phase=='baseline':baseline_phase(protocol,data)
 elif args.phase=='tune':
  from .tuning import tune
  tune(protocol)
 elif args.phase=='associate-tune':
  from .association_tuning import tune_association
  tune_association(protocol)
 elif args.phase=='check':
  from .checking import final_check
  final_check(protocol)
 elif args.phase=='export-corrected':
  from .postreview import export_corrected
  export_corrected(protocol)
