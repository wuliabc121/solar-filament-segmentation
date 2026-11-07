"""Two-hour sequential method campaign with fixed inference and one fresh check."""
import datetime as dt
import fcntl
import json
import os
import sys
import time
import yaml
from pathlib import Path
from m3_filament.io import sha256,write_json,read_jsonl
from m3_filament.queueing import run_wall_limited
from m3_filament.joint import derive_metrics
from m3_filament.model import source_manifest_hash

OUT=Path('reports/structure_20261104');ROOT=Path.cwd()
PLAN=json.loads((OUT/'campaign.json').read_text());DEADLINE=PLAN['deadline_utc']
LOGGER=str(Path.home()/'.agents/skills/research-guardian/scripts/experiment_logger.py')
SESSION='01a1a144-84a8-7f81-8664-65c6a24ea02b'
GRID=PLAN['tuning_confidences']
STATE={'pid':os.getpid(),'status':'starting','deadline_utc':DEADLINE,'jobs':[],'failures':[]}


def now():return dt.datetime.now(dt.timezone.utc)
def remaining():return (dt.datetime.fromisoformat(DEADLINE)-now()).total_seconds()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def save(status):STATE.update(status=status,updated_utc=now().isoformat());atomic(OUT/'state.json',STATE)


def record_for(name):
 records=[]
 for p in Path('.research-guardian/experiments').glob('*/record.json'):
  d=json.loads(p.read_text())
  if d.get('model')==name:records.append((d['started_at'],p,d))
 return max(records) if records else None


def job(name,command,metrics,seconds,config=None,checkpoints=(),final=False,reserve_seconds=1050):
 path=OUT/'jobs'/(name+'.json');spec={'command':command,'source_sha256':source_manifest_hash(),
 'external_training_sha256':sha256('scripts/structure_training.py'),'config_sha256':sha256(config or OUT/'campaign.json')}
 if path.exists():
  old=json.loads(path.read_text())
  if old['spec']!=spec:raise RuntimeError('job specification changed: '+name)
  if old.get('returncode')==0 and Path(metrics).exists():return
  raise RuntimeError('existing partial job requires explicit inspection: '+name)
 seconds=min(seconds,remaining()-reserve_seconds)
 if seconds<15:raise TimeoutError('reserved final budget: '+name)
 if Path(metrics).exists():raise FileExistsError(metrics)
 dataset_label='fresh_check12_once' if final else 'old_tune24_only'
 if config:dataset_label=Path(yaml.safe_load(Path(config).read_text())['data']).parent.name+'_train_plus_oldtune24'
 if name.startswith('structure_checkpoint_export_'):dataset_label='saved_checkpoint_only_no_labels'
 cmd=[sys.executable,LOGGER,'run','--project',str(ROOT),'--session-id',SESSION,'--model',name,
 '--dataset',dataset_label,'--seed','42','--config',str(config or OUT/'campaign.json'),
 '--metrics',metrics,'--timeout',str(seconds),'--grace','1']
 for checkpoint in checkpoints:cmd+=['--checkpoint',checkpoint]
 cmd+=['--param','objective=training_and_method_not_threshold_only','--']+command
 save('running_'+name);result=run_wall_limited(cmd,OUT/(name+'.log'),DEADLINE,seconds)
 detail={'name':name,'spec':spec,'metrics_path':metrics,**result};atomic(path,detail);STATE['jobs'].append(detail);save('finished_'+name)
 if result['returncode']!=0 or not Path(metrics).exists():raise RuntimeError('job failed '+name+' code '+str(result['returncode']))


def recover_training(run):
 rec=record_for(run)
 if rec and rec[2]['status']!='running' and not Path('runs',run,'metrics.json').exists():
  from recover_budget_checkpoints import recover
  note=recover(Path('runs')/run,rec[2]);atomic(OUT/(run+'_recovery.json'),note)
  STATE['failures'].append({'run':run,'runner_status':rec[2]['status'],'recovery':note})
 return Path('runs',run,'run_manifest.json').exists()


def wait_control():
 save('waiting_existing_structure_control_e10')
 while remaining()>750:
  record=record_for('structure_control_e10')
  if not record:raise RuntimeError('expected explicitly started control runner missing')
  if record[2]['status']!='running':
   STATE['control_record']=str(record[1]);STATE['control_status']=record[2]['status']
   if record[2]['status']!='success':recover_training('structure_control_e10')
   return
  time.sleep(2)
 raise TimeoutError('control exhausted scoring reserve')




def inference_ready(run,reserve_seconds):
 metric=Path('runs',run,'metrics.json')
 if not metric.exists():return False
 info=json.loads(metric.read_text())
 if not info.get('recovered_for_inference'):return True
 try:
  job('structure_checkpoint_export_'+run,[str(ROOT/'run.sh'),'scripts/export_structure_checkpoint.py','--run',run],
      'runs/'+run+'/inference_exports.json',90,checkpoints=['runs/'+run+'/weights/best_inference.pt','runs/'+run+'/weights/last_inference.pt'],reserve_seconds=reserve_seconds)
  return True
 except Exception as e:
  STATE['failures'].append({'stage':'inference_export_'+run,'error':repr(e)});return False


def resolved_checkpoint(run,checkpoint):
 path=Path('runs',run,'inference_exports.json')
 if path.exists():
  item=json.loads(path.read_text())['exports'].get(checkpoint)
  if item:return Path(item['inference_path']).name
 return checkpoint


def equivalent_checkpoints(run):
 import torch,hashlib
 hashes={}
 for name in ['best.pt','last.pt']:
  path=Path('runs',run,'weights',resolved_checkpoint(run,name))
  if not path.exists():return False
  ckpt=torch.load(path,map_location='cpu',weights_only=False)
  model=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt['model'];h=hashlib.sha256()
  for key,value in sorted(model.state_dict().items()):
   h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(list(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
  hashes[name]=h.hexdigest()
 same=len(set(hashes.values()))==1
 atomic(OUT/('checkpoint_equivalence_'+run+'.json'),{'run':run,'state_dict_sha256':hashes,'identical_parameters':same,'last_inference_skipped_if_identical':same})
 return same


def choose(points,refs):
 refs=[derive_metrics(r) for r in refs];lower_recall=max(r['recall'] for r in refs);upper_fp=min(r['fp'] for r in refs);lower_sq=max(r['sq'] for r in refs)
 def violations(p):
  return (['pq'] if p['pq']<=.35 else [])+(['recall'] if p['recall']<=lower_recall else [])+(['fp'] if p['fp']>=upper_fp else [])+(['sq'] if p['sq']<lower_sq else [])
 def margin(p):return min((p['pq']-.35)/.35,(p['recall']-lower_recall)/max(lower_recall,.01),(upper_fp-p['fp'])/max(upper_fp,1),(p['sq']-lower_sq)/max(lower_sq,.01))
 feasible=[p for p in points if not violations(p)]
 candidate=max(feasible,key=lambda p:p['pq']) if feasible else max(points,key=lambda p:(margin(p),p['pq']))
 return {'candidate':candidate,'goal_met':bool(feasible),'violations':violations(candidate),'feasible_points':len(feasible),
 'bounds':{'pq_strictly_greater':.35,'recall_strictly_greater':lower_recall,'fp_strictly_less':upper_fp,'sq_at_least':lower_sq},
 'selection':'PQ ranks feasible; otherwise maximin diagnostic only'}


def score_model(run,label,dataset,thresholds,imgsz=2048,min_area=32,final=False,after_seconds=1050,checkpoint="best.pt"):
 raw=label+'_raw';path='runs/'+raw+'/metrics.json';weights='runs/'+run+'/weights/'+checkpoint
 command=[str(ROOT/'run.sh'),'-m','m3_filament.evaluation','--weights',weights,'--dataset',dataset,'--output','runs/'+raw,
 '--imgsz',str(imgsz),'--device','cpu','--thresholds',*[str(x) for x in thresholds]]
 job(raw,command,path,210,checkpoints=[weights],final=final,reserve_seconds=after_seconds+(150 if min_area else 0))
 if not min_area:return json.loads(Path(path).read_text())
 processed=label+'_area32';path='runs/'+processed+'/metrics.json'
 command=[str(ROOT/'run.sh'),'-m','m3_filament.postprocess','--instances','runs/'+raw+'_predictions/instances.jsonl',
 '--dataset',dataset,'--output','runs/'+processed,'--min-area','32','--closing-radius','0','--score-floor',str(min(thresholds)),
 '--thresholds',*[str(x) for x in thresholds]]
 job(processed,command,path,150,final=final,reserve_seconds=after_seconds)
 return json.loads(Path(path).read_text())


def main():
 lock=open(OUT/'driver.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 wait_control()
 existing_method=record_for('structure_dicecl_e10')
 method_already_ended=bool(existing_method and existing_method[2]['status']!='running' and (OUT/'jobs/structure_dicecl_e10.json').exists())
 if remaining()>4300 or method_already_ended:
  run='structure_dicecl_e10';command=[str(ROOT/'run.sh'),'scripts/verified_structure_train.py','--config','configs/'+run+'.yaml','--run-id',run,'--arm','structure']
  try:job(run,command,'runs/'+run+'/metrics.json',2700,config='configs/'+run+'.yaml',checkpoints=['runs/'+run+'/weights/best.pt'],reserve_seconds=2500)
  except Exception as e:
   record=record_for(run)
   STATE['failures'].append({'stage':run,'error':repr(e),'original_runner_status':record[2]['status'] if record else None,'original_exit_code':record[2].get('exit_code') if record else None});recover_training(run)
 else:STATE['failures'].append({'stage':'structure_dicecl_e10','error':'insufficient protected UTC budget'})
 refs=[json.loads(Path('reports/balanced_20261104/results.json').read_text())['selection']['candidate'],
       json.loads(Path('reports/balanced_20261104/references.json').read_text())['references'][1]]
 results={'references_tune':refs,'tuning':{},'fresh_check':{},'goal_met':False};points=[]
 evaluated_primary=0
 for run in ['structure_control_e10','structure_dicecl_e10']:
  if not Path('runs',run,'run_manifest.json').exists():continue
  if not inference_ready(run,2200):continue
  if run.endswith('dicecl_e10') and not Path('runs',run,'criterion_actual.json').exists():
   STATE['failures'].append({'stage':run,'error':'missing actual loss verification'});continue
  same_parameters=equivalent_checkpoints(run)
  for checkpoint in ['best.pt','last.pt']:
   if checkpoint=='last.pt' and same_parameters:
    evaluated_primary+=1;continue
   tag=run+'_'+checkpoint.split('.')[0]
   checkpoint=resolved_checkpoint(run,checkpoint)
   try:
    label='structure_tune_'+tag
    metrics=score_model(run,label,'data/overnight176',GRID,after_seconds=1050+360*(3-evaluated_primary),checkpoint=checkpoint)
    training_metrics=json.loads(Path('runs',run,'metrics.json').read_text())
    metrics['training_status']='completed10' if training_metrics.get('actual_epochs')==10 and not training_metrics.get('recovered_for_inference') else 'partial_unequal_diagnostic_only'
    metrics['model_run']=run;metrics['checkpoint_name']=checkpoint
    results['tuning'][tag]=metrics
    for row in metrics['grid']:points.append(derive_metrics(row)|{'run':run,'checkpoint':checkpoint,'imgsz':2048,'min_area':32,'closing_radius':0})
   except Exception as e:STATE['failures'].append({'stage':'tune_'+tag,'error':repr(e)})
   evaluated_primary+=1
   atomic(OUT/'results.json',results)
 if not points:raise RuntimeError('no actual method candidate scored')
 selection=choose(points,refs)
 primary_metrics=[json.loads(Path('runs',run,'metrics.json').read_text()) for run in ['structure_control_e10','structure_dicecl_e10'] if Path('runs',run,'metrics.json').exists()]
 results['controlled_loss_comparison_complete']=(len(primary_metrics)==2 and all(m.get('actual_epochs')==10 and not m.get('recovered_for_inference') and m.get('optimizer_updates') is not None for m in primary_metrics) and len({m['optimizer_updates'] for m in primary_metrics})==1)
 if not selection['goal_met'] and remaining()>2400:
  run='structure496_transfer';parent=selection['candidate']['run'];arm='structure' if parent=='structure_dicecl_e10' else 'control'
  cfg=yaml.safe_load(Path('configs/'+parent+'.yaml').read_text())
  transfer_epochs=6 if remaining()>3900 else 4 if remaining()>3300 else 1
  cfg.update(model='runs/'+parent+'/weights/'+selection['candidate'].get('checkpoint','best.pt'),data='data/balanced496/dataset.yaml',epochs=transfer_epochs,lr0=.0001,time=.7 if transfer_epochs==6 else .5 if transfer_epochs==4 else .333,name=run)
  config='configs/'+run+'.yaml';Path(config).write_text(yaml.safe_dump(cfg,sort_keys=False))
  atomic(OUT/'conditional_transfer_started.json',{'declared_utc':now().isoformat(),'parent':parent,'parent_checkpoint_sha256':sha256(cfg['model']),'arm':arm,'remaining_seconds':remaining(),'configuration':cfg,'selection_before_transfer':selection})
  entry='scripts/verified_structure_train.py' if arm=='structure' else 'scripts/structure_training.py'
  try:
   job(run,[str(ROOT/'run.sh'),entry,'--config',config,'--run-id',run,'--arm',arm],'runs/'+run+'/metrics.json',2400 if transfer_epochs==6 else 1800 if transfer_epochs==4 else 1200,config=config,checkpoints=['runs/'+run+'/weights/best.pt'],reserve_seconds=1500)
  except Exception as e:
   record=record_for(run)
   STATE['failures'].append({'stage':run,'error':repr(e),'original_runner_status':record[2]['status'] if record else None,'original_exit_code':record[2].get('exit_code') if record else None});recover_training(run)
  if Path('runs',run,'run_manifest.json').exists() and inference_ready(run,1410):
   try:
    metrics=score_model(run,'structure_tune_'+run,'data/overnight176',GRID,after_seconds=1050,checkpoint=resolved_checkpoint(run,'best.pt'))
    metrics['training_status']='additional_data_transfer_not_equal_compute_ablation'
    results['tuning'][run]=metrics
    for row in metrics['grid']:points.append(derive_metrics(row)|{'run':run,'checkpoint':resolved_checkpoint(run,'best.pt'),'imgsz':2048,'min_area':32,'closing_radius':0})
    selection=choose(points,refs)
   except Exception as e:STATE['failures'].append({'stage':'tune_'+run,'error':repr(e)})
 results['selection']=selection
 freeze={'frozen_utc':now().isoformat(),'selection':selection,'fresh_check_sha256':sha256(OUT/'fresh_check.json'),
 'training_script_sha256':sha256('scripts/structure_training.py'),'driver_sha256':sha256(__file__),'source_sha256':source_manifest_hash()}
 atomic(OUT/'frozen_selection.json',freeze);atomic(OUT/'results.json',results)
 # Do not even export fresh GT until parameters/checkpoint have been frozen.
 if remaining()<1050:
  STATE['failures'].append({'stage':'fresh_check','error':'insufficient protected 930 scoring +120 closeout seconds; labels remain unexported'})
  save('completed_without_fresh_check');return
 declaration=json.loads((OUT/'fresh_check.json').read_text())
 from m3_filament.data import prepare_dataset
 from m3_filament.io import read_jsonl,write_jsonl,write_json
 dataset=Path('data/structure_fresh_check12')
 if dataset.exists():raise FileExistsError('fresh check already exported; do not repeat')
 # Existing prepare requires both roles; use raw GT mapping but no training images.
 from m3_filament.data import load_data,ANNOTATION,annotation_mask
 from m3_filament.masks import encode
 root,data=load_data('data/raw');byimage={im['id']:im for im in data['images']};annotations={}
 from collections import defaultdict,Counter
 annotations=defaultdict(list)
 for ann in data['annotations']:annotations[ann['image_id']].append(ann)
 selected=set(declaration['calibration']);samples=[]
 for im in data['images']:
  obs=Path(im['file_name']).stem
  if obs not in selected:continue
  target=dataset/'targets'/(im['id']+'.json');target.parent.mkdir(parents=True,exist_ok=True)
  masks=[encode(annotation_mask(a,im['height'],im['width'])) for a in annotations[im['id']]]
  write_json(target,{'size':[im['height'],im['width']],'masks':masks})
  samples.append({'sample_id':im['id'],'annotation_id':im['id'],'observation_id':obs,'split':'calibration','instance_count':len(masks),
  'image_path':str((root/'train/train_images'/im['file_name']).resolve()),'target_path':str(target.resolve())})
 write_jsonl(dataset/'samples.jsonl',samples)
 write_json(dataset/'preparation.json',{'split_hash':declaration['source_split_hash'],'split_status':'provisional_development_only',
 'samples':{'calibration':len(samples)},'sample_manifest_sha256':sha256(dataset/'samples.jsonl'),'target_backend':'native','production_training_allowed':False,
 'annotation_policy':'each independent annotation pass; final check only'})
 checks=[('latest_quality_reference','balanced_maskmetric_ft4',2048,.4,32,'best.pt'),('previous_recall_reference','resume176_mixedcrop_e6',1536,.15,0,'best.pt'),
 ('new_candidate',selection['candidate']['run'],2048,selection['candidate']['confidence'],32,selection['candidate'].get('checkpoint','best.pt'))]
 for key,run,size,confidence,area,checkpoint in checks:
  try:
   data=score_model(run,'structure_check_'+key,str(dataset),[confidence],size,area,final=True,after_seconds={'latest_quality_reference':690,'previous_recall_reference':480,'new_candidate':120}[key],checkpoint=checkpoint)
   results['fresh_check'][key]=data
  except Exception as e:STATE['failures'].append({'stage':key,'error':repr(e)})
  atomic(OUT/'results.json',results)
 if len(results['fresh_check'])==3:
  ref=[r['selected'] for k,r in results['fresh_check'].items() if k!='new_candidate'];candidate=derive_metrics(results['fresh_check']['new_candidate']['selected'])
  verdict=choose([candidate],ref);results['final_gate']=verdict;results['goal_met']=verdict['goal_met']
 atomic(OUT/'results.json',results);save('completed')


if __name__=='__main__':
 try:main()
 except BaseException as e:
  STATE['failures'].append({'stage':'driver','error':repr(e)});save('failed');raise
