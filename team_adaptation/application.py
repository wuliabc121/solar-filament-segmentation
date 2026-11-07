"""Inference-only M3 adapter; validates cache/checkpoint/floor compatibility."""
import json,copy
from pathlib import Path
from .reconstruction import combine_semantics,prepare_record,adapt_prepared,AdaptConfig
from .association import prepare_components,associate,AssociationConfig
from m3_filament.io import read_jsonl,write_jsonl,sha256,content_hash
from team_pipeline.contracts import load_probability

def validate_input(record,profile):
 source=record['source']
 if source.get('checkpoint_sha256')!=profile['models']['m3']:raise ValueError('M3 checkpoint differs from frozen profile')
 config=source.get('inference_config',{})
 expected=profile.get('raw_proposal_contract',{'device':'cpu','imgsz':2048,'conf':.05,'iou':.95,'retina_masks':True,'max_det':100,'verbose':False,'save':False})
 if source.get('inference_config_sha256')!=content_hash(config):raise ValueError('M3 inference configuration hash is inconsistent')
 if 'parent_inference_config_sha256' in source or 'postprocess' in config:raise ValueError('adapter requires original raw proposals; postprocessed history may be irreversible')
 if config!=expected:raise ValueError('raw M3 proposal contract differs from frozen adapter (CPU/2048/.05/.95/retina/maxdet100)')
 if source.get('source_manifest_sha256')!='564c3b3eaa1d93b89003dac0f37d64d5a03670349d0f231eef3694d62cb2b007' or source.get('ultralytics_version')!='8.3.221':raise ValueError('M3 inference code/runtime differs from tested raw proposal generator')
 if config['conf']>profile['config']['min_detector']:raise ValueError('M3 raw confidence floor too high')

def apply_one(record,semantics,profile,mode,run_id):
 validate_input(record,profile)
 for name,(p,meta) in semantics.items():
  if meta['checkpoint_hash']!=profile['models'][name]:raise ValueError('semantic checkpoint differs from frozen profile')
 p,meta=combine_semantics(semantics,mode,record['observation_id'],record['image_size']);prepared=prepare_record(record)
 if profile['family']=='association':
  config=AssociationConfig(**profile['config']);result=associate(prepared,prepare_components(p,config.threshold,config.min_area),meta,config,run_id)
 else:result=adapt_prepared(prepared,p,meta,AdaptConfig(**profile['config']),run_id)
 result['source'].update(frozen_profile_point=profile['point'],semantic_mode=mode,selection_status='development_candidate_not_official_score',M1_effect_validation='two_reconstructed_unseen_cases_only' if mode!='m2' else None)
 for i in result['instances']:i['source']=dict(result['source'])
 return result

def apply_files(args):
 profile_path=Path(args.profile);profile=json.loads(profile_path.read_text());records=list(read_jsonl(args.instances));out=[];seen=set()
 for rec in records:
  s=rec['observation_id']
  if s in seen or Path(s).name!=s or s in ('.','..'):raise ValueError('duplicate/unsafe observation')
  seen.add(s);sem={}
  if args.mode in ('m1','both','guided_both'):
   if not args.m1:raise ValueError('--m1 is required for chosen mode')
   sem['m1']=load_probability(Path(args.m1)/(s+'.npz'),s,rec['image_size'])
  if args.mode in ('m2','both','guided_both'):
   if not args.m2:raise ValueError('--m2 is required for chosen mode')
   sem['m2']=load_probability(Path(args.m2)/(s+'.npz'),s,rec['image_size'])
  r=apply_one(rec,sem,profile,args.mode,args.run_id);r['source']['frozen_profile_sha256']=sha256(profile_path)
  for inst in r['instances']:inst['source']=dict(r['source'])
  out.append(r)
 write_jsonl(args.output,out);print(json.dumps({'observations':len(out),'instances':sum(len(r['instances']) for r in out),'output':args.output,'mode':args.mode,'benchmark_eligible':False}))
