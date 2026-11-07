"""One final internal check of previously frozen configurations, never a tuning loop."""
import copy,json,time
from dataclasses import replace
from datetime import datetime,timezone
import numpy as np
from .experiment import REPORT,update_state
from .protocol import check_deadline
from .data import ground_truth
from .metrics import score_records,aggregate_rows,validate_fast_record
from .reconstruction import AdaptConfig,prepare_record,adapt_prepared,combine_semantics
from .application import apply_one
from .association import prepare_components
from .baselines import frozen_m3,initial_fusion,semantic_components
from m3_filament.io import sha256,read_jsonl,write_json,write_jsonl
from m3_filament.schema import make_record,crop_metadata
from team_pipeline.contracts import load_probability
from team_pipeline.integrity import verified_checkpoint

def validate_frozen(profile,protocol):
 if profile['protocol_sha256']!=sha256(REPORT/'protocol.json') or profile['check_ids']!=protocol['check_ids']:raise ValueError('frozen profile/protocol mismatch')
 grid=REPORT/('association_grid.json' if profile['family']=='association' else 'adaptation_grid.json')
 if profile['grid_sha256']!=sha256(grid):raise ValueError('grid changed after freeze')
 points=json.loads(grid.read_text())['points']
 if profile['config']!=points[profile['point']]:raise ValueError('frozen configuration differs from registered point')
 for k,v in protocol['models'].items():
  _,actual=verified_checkpoint(v)
  if profile['models'][k]!=actual:raise ValueError('checkpoint changed after freeze')

def gate_geometry_ablation(raw,associated):
 ids={i['source_instance_id'] for i in associated['instances']};rec=copy.deepcopy(raw);rec['source']=dict(associated['source'],run_id='association_gate_without_semantic_geometry',ablation='retain original detector shapes for accepted anchors');out=[]
 for old in rec['instances']:
  if old['instance_id'] not in ids:continue
  i=copy.deepcopy(old);i.update(instance_id=len(out)+1,source_run_id=rec['source']['run_id'],source=rec['source']);i['crop']=crop_metadata(i,rec['image_size']);out.append(i)
 rec['instances']=out;validate_fast_record(rec);return rec

def add_association_residual(raw,selected,pt,meta):
 result=copy.deepcopy(selected);components=prepare_components(pt,.35,400)['components'];accepted={i['segmentation']['counts'] for i in result['instances']}
 for comp in components:
  if comp['segmentation']['counts'] in accepted:continue
  i={'instance_id':len(result['instances'])+1,'segmentation':comp['segmentation'],'bbox':list(comp['box']),'score':comp['semantic_mean'],'score_kind':'mean_semantic_probability','source_run_id':result['source']['run_id'],'source':result['source'],'candidate_source':'semantic_residual','source_instance_id':None,'requires_refiner':True,'refiner_reason':'no_accepted_M3_anchor'};i['crop']=crop_metadata(i,result['image_size']);result['instances'].append(i)
 result['source']=dict(result['source'],variant='M4_unreviewed_residual_candidates');validate_fast_record(result);return result

def paired_bootstrap(rows_a,rows_b,n=5000):
 groups=sorted({r['observation_id'][:8] for r in rows_a});keys=('tp','fp','fn','tp_iou_sum');aa=np.array([[sum(r[k] for r in rows_a if r['observation_id'][:8]==g) for k in keys] for g in groups]);bb=np.array([[sum(r[k] for r in rows_b if r['observation_id'][:8]==g) for k in keys] for g in groups]);idx=np.random.default_rng(42).integers(0,len(groups),(n,len(groups)));a=aa[idx].sum(1);b=bb[idx].sum(1)
 def metrics(x):
  tp,fp,fn,s=x.T;den=tp+.5*(fp+fn)
  return {'pq':np.divide(s,den,out=np.zeros_like(s),where=den>0),'recall':np.divide(tp,tp+fn,out=np.zeros_like(tp),where=tp+fn>0),'fp':fp}
 a,b=metrics(a),metrics(b);return {'groups':len(groups),'draws':n,'seed':42,'paired_group_bootstrap_not_training_seed_variance':True,'development_calibration_not_unbiased_test':True,'delta_selected_minus_M3':{k:{'mean':float((b[k]-a[k]).mean()),'ci95':np.quantile(b[k]-a[k],[.025,.975]).tolist()} for k in a}}

def final_check(protocol):
 profile_path=REPORT/'frozen_adaptation_v2.json';profile=json.loads(profile_path.read_text());validate_frozen(profile,protocol);out=REPORT/'final_check';out.mkdir(exist_ok=False);write_json(out/'started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'profile_sha256':sha256(profile_path),'profile_frozen_utc':profile['frozen_utc'],'no_reselection_after_check':True})
 update_state('C','scoring_frozen_internal_check',active=True,check_scores_read=True);ids=protocol['check_ids'];gt=ground_truth(ids);raw={r['observation_id']:r for r in read_jsonl(REPORT/'cache/m3_raw/instances.jsonl')};v1=json.loads((REPORT/'frozen_adaptation.json').read_text());outputs={k:[] for k in ('M3_frozen','M2_semantic','E5_M3_M2_initial','M3_M2_reconstruction_v1','M3_M2_selected','M3_M2_geometry_gate_ablation','M4_residual_candidates_only_not_final')};diag={k:[] for k in ('M3_frozen','M2_semantic','M1_semantic','M3_M2_selected','M3_M1_transfer','M3_both_mean_transfer','M3_both_guided_transfer')};diagnostic=set(protocol['three_model_reconstructed_group_safe_diagnostic_ids']);started=time.monotonic()
 for s in ids:
  check_deadline(protocol);rec=raw[s];pt,mt=load_probability(REPORT/'cache/m2'/(s+'.npz'),s,rec['image_size']);selected=apply_one(rec,{'m2':(pt,mt)},profile,'m2','frozen_M3_M2_selected');prep=prepare_record(rec);p,components=combine_semantics({'m2':(pt,mt)},'m2',s,rec['image_size'])
  values={'M3_frozen':frozen_m3(rec),'M2_semantic':semantic_components(rec,pt),'E5_M3_M2_initial':initial_fusion(rec,pt),'M3_M2_selected':selected,'M3_M2_reconstruction_v1':adapt_prepared(prep,p,components,AdaptConfig(**v1['config']),'reconstruction_v1')}
  if profile['family']=='association':
   values['M3_M2_geometry_gate_ablation']=gate_geometry_ablation(rec,selected);values['M4_residual_candidates_only_not_final']=add_association_residual(rec,selected,pt,mt)
  else:
   values['M3_M2_geometry_gate_ablation']=adapt_prepared(prep,p,components,AdaptConfig(**dict(profile['config'],method='gate_only')),'gate_ablation');values['M4_residual_candidates_only_not_final']=adapt_prepared(prep,p,components,AdaptConfig(**dict(profile['config'],residual=True)),'residual_candidates')
  for k,v in values.items():outputs[k].append(v)
  if s in diagnostic:
   pc,mc=load_probability(REPORT/'cache/m1'/(s+'.npz'),s,rec['image_size'])
   for k in ('M3_frozen','M2_semantic','M3_M2_selected'):diag[k].append(values[k])
   diag['M1_semantic'].append(semantic_components(rec,pc,.5,400,'m1'))
   for mode,k in [('m1','M3_M1_transfer'),('both','M3_both_mean_transfer'),('guided_both','M3_both_guided_transfer')]:diag[k].append(apply_one(rec,{'m1':(pc,mc),'m2':(pt,mt)} if mode!='m1' else {'m1':(pc,mc)},profile,mode,k))
 metrics={};allrows={}
 for name,recs in outputs.items():
  m,ann=score_records(recs,gt,ids);m['requires_refiner_instances']=sum(sum(bool(i.get('requires_refiner')) for i in r['instances']) for r in recs);metrics[name]=m;allrows[name]=ann;write_jsonl(out/(name+'.jsonl'),recs);write_jsonl(out/(name+'_per_annotation.jsonl'),ann)
 diagmetrics={};diaggt={s:gt[s] for s in diagnostic}
 for name,recs in diag.items():
  m,ann=score_records(recs,diaggt,sorted(diagnostic));diagmetrics[name]=m;write_jsonl(out/(name+'_diagnostic2.jsonl'),recs)
 baseline=metrics['M3_frozen'];chosen=metrics['M3_M2_selected'];gate={'pq_above_035':chosen['pq']>.35,'pq_improved':chosen['pq']>baseline['pq'],'recall_improved':chosen['recall']>baseline['recall'],'FP_reduced':chosen['fp']<baseline['fp'],'SQ_no_lower':chosen['sq']>=baseline['sq'],'small_recall_no_lower':chosen['small_recall']>=baseline['small_recall']}
 result={'status':'complete','scope':'19observations18dategroups internal check inside already-used external M2calibration, notpristine/official','primary':metrics,'M1_diagnostic2_only':diagmetrics,'M1_warning':'reconstructed untrained membership, only2cases, cannot establish generalization','gate':gate,'no_postcheck_selection':True,'selected_profile_sha256':sha256(profile_path),'seconds':time.monotonic()-started,'no_training':True,'original_M3_holdout_used':False,'M2_holdout_used':False}
 write_json(out/'metrics.json',result);write_json(out/'bootstrap.json',paired_bootstrap(allrows['M3_frozen'],allrows['M3_M2_selected']));update_state('C','complete',active=False,check_scores_read=True,gate=gate,quality_claim_scope=result['scope']);print(json.dumps(result,indent=2))
