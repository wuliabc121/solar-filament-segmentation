"""Fixed all-model comparison on exactly the user's frozen random20 images."""
import json,copy,time
from dataclasses import asdict
from . import ROOT,REPORT
from .protocol import load,deadline
from .inference import state
from m3_filament.io import read_jsonl,write_json,write_jsonl,sha256
from m3_filament.postprocess import transform_record
from m3_filament.schema import make_record,crop_metadata
from team_pipeline.contracts import load_probability
from team_adaptation.data import ground_truth
from team_adaptation.metrics import score_records,aggregate_rows,validate_fast_record
from team_adaptation.application import validate_input
from team_adaptation.reconstruction import prepare_record,combine_semantics
from team_adaptation.association import prepare_components,associate,AssociationConfig
from team_pipeline.vendor import m2_modules

DEPENDENCIES={'M1_CC_050':['m1'],'M1_CC_035':['m1'],'M2_CC_035':['m2'],'M3_C_quality_reference':['M3_C'],'M3_S_structure_candidate':['M3_S'],'M3_C_plus_M1':['M3_C','m1'],'M3_S_plus_M1':['M3_S','m1'],'M3_C_plus_M2':['M3_C','m2'],'M3_S_plus_M2':['M3_S','m2'],'M3_C_guided_both':['M3_C','m1','m2'],'M3_S_guided_both':['M3_S','m1','m2']}

def cc_record(obs,prepared,meta,label):
 source={'run_id':label,'semantic_components':meta,'threshold':prepared['threshold'],'min_area':prepared['min_area'],'is_oof':False,'benchmark_eligible':False};items=[];shape=list(prepared['shape'])
 for c in prepared['components']:
  i={'instance_id':len(items)+1,'segmentation':c['segmentation'],'bbox':list(c['box']),'score':c['semantic_mean'],'score_kind':'mean_semantic_probability','source_run_id':label,'source':source,'candidate_source':'semantic_connected_component','requires_refiner':False};i['crop']=crop_metadata(i,shape);items.append(i)
 rec=make_record(obs,shape,items,source);validate_fast_record(rec);return rec

def comparison():
 p=load();deadline(p);ids=p['validation_ids'];out=REPORT/'evaluation';out.mkdir(exist_ok=False);profiles=json.loads((REPORT/'adapter_profiles.json').read_text());raw={k:{r['observation_id']:r for r in read_jsonl(REPORT/'cache'/k/'instances.jsonl')} for k in ('M3_C','M3_S')}
 for k in raw:
  if sorted(raw[k])!=ids:raise ValueError('incomplete model output: '+k)
 gt=ground_truth(ids);rows={k:[] for k in DEPENDENCIES};excluded={k:0 for k in DEPENDENCIES};counts={k:0 for k in DEPENDENCIES};refiner={k:0 for k in DEPENDENCIES};actual_seen={k:set(p['models'][k]['training_observations']) for k in p['models']};handles={k:(out/(k+'.jsonl')).open('x') for k in DEPENDENCIES};started=time.monotonic();state('scoring','running',completed_images=0,total_images=len(ids));gtrows=[];csvpred={k:[] for k in DEPENDENCIES}
 try:
  for index,s in enumerate(ids):
   deadline(p);shape=raw['M3_C'][s]['image_size'];pc,mc=load_probability(REPORT/'cache/m1'/(s+'.npz'),s,shape);pt,mt=load_probability(REPORT/'cache/m2'/(s+'.npz'),s,shape)
   if mc['checkpoint_hash']!=p['models']['m1']['checkpoint_sha256'] or mt['checkpoint_hash']!=p['models']['m2']['checkpoint_sha256']:raise ValueError('semantic checkpoint mismatch')
   m1,mm1=combine_semantics({'m1':(pc,mc)},'m1',s,shape);m2,mm2=combine_semantics({'m2':(pt,mt)},'m2',s,shape);both,mmb=combine_semantics({'m1':(pc,mc),'m2':(pt,mt)},'guided_both',s,shape)
   components={'m1':prepare_components(m1,.35,400),'m2':prepare_components(m2,.35,400),'guided_both':prepare_components(both,.35,400)};metas={'m1':mm1,'m2':mm2,'guided_both':mmb};recs={'M1_CC_050':cc_record(s,prepare_components(pc,.5,400),mm1,'M1_CC_050'),'M1_CC_035':cc_record(s,components['m1'],mm1,'M1_CC_035'),'M2_CC_035':cc_record(s,components['m2'],mm2,'M2_CC_035')}
   for key in ('M3_C','M3_S'):
    record=raw[key][s];validate_input(record,profiles[key]);base=p['M3_baselines'][key];name='M3_C_quality_reference' if key=='M3_C' else 'M3_S_structure_candidate';recs[name]=transform_record(record,name,base['min_area'],0,base['conf']);prepared=prepare_record(record)
    for mode,suffix in [('m1','plus_M1'),('m2','plus_M2'),('guided_both','guided_both')]:
     label=key+'_'+suffix;recs[label]=associate(prepared,components[mode],metas[mode],AssociationConfig(**p['association_fixed']),label)
   for label,rec in recs.items():
    seen_models=[k for k in DEPENDENCIES[label] if s in actual_seen[k]];rec['source'].update(evaluation_protocol='random20_mixed_training_development_diagnostic',random_split_sha256=p['split_sha256'],random_protocol_sha256=sha256(REPORT/'protocol.json'),model_training_overlap_on_this_image=seen_models,semantic_mode='guided_both' if label.endswith('guided_both') else ('m1' if label.endswith('M1') else ('m2' if label.endswith('M2') else None)),benchmark_eligible=False,is_oof=False)
    for i in rec['instances']:i['source']=copy.deepcopy(rec['source'])
    metric,rr=score_records([rec],{s:gt[s]},[s]);rows[label].extend(dict(x,dependency_exact_training_seen=bool(seen_models),seen_models=seen_models) for x in rr);excluded[label]+=metric['zero_gt_excluded'];counts[label]+=len(rec['instances']);refiner[label]+=sum(bool(i.get('requires_refiner')) for i in rec['instances']);handles[label].write(json.dumps(rec)+'\n')
    if label in csvpred:csvpred[label].extend((f"{s}_{i['instance_id']}",i['segmentation']['counts']) for i in rec['instances'])
   for rid,rs in gt[s].items():
    for k,g in enumerate(rs,1):gtrows.append((f'{rid}_{k}',g['counts']))
   state('scoring','running',completed_images=index+1,total_images=len(ids),seconds=time.monotonic()-started)
 finally:
  for f in handles.values():f.close()
 table={};subgroups={}
 for label in DEPENDENCIES:
  total=aggregate_rows(rows[label],len(ids),excluded[label]);total.update(unique_image_prediction_count=counts[label],requires_refiner_instances=refiner[label],dependencies=DEPENDENCIES[label]);table[label]=total;write_jsonl(out/(label+'_per_annotation.jsonl'),rows[label]);subgroups[label]={}
  for seen in (True,False):
   subset=[s for s in ids if any(s in actual_seen[k] for k in DEPENDENCIES[label])==seen];selected=[r for r in rows[label] if r['observation_id'] in set(subset)];subgroups[label]['dependency_exact_seen' if seen else 'not_in_dependency_training_lists']=aggregate_rows(selected,len(subset));subgroups[label]['dependency_exact_seen' if seen else 'not_in_dependency_training_lists']['warning']='different subsets across methods; reconstructed M1 membership/date overlap/history prevents independent validation claim'
 cross=validate_csv_routes(gtrows,csvpred,table)
 result={'status':'complete','validation_images':len(ids),'annotation_records':p['annotation_records'],'scope':p['diagnostic_status'],'split_sha256':p['split_sha256'],'protocol_sha256':sha256(REPORT/'protocol.json'),'models_explicitly_named_and_versioned':True,'methods':table,'no_training':True,'no_parameter_tuning_on_random20':True,'source_precision_mixed':True,'seconds':time.monotonic()-started}
 write_json(out/'metrics.json',result);write_json(out/'training_overlap_subgroups.json',subgroups);write_json(out/'M2_CSV_agreement.json',cross);state('scoring','complete',completed_images=len(ids),total_images=len(ids),seconds=time.monotonic()-started);print(json.dumps(result,indent=2))


def validate_csv_routes(gtrows,predictions,table,h=2048,w=2048):
 if set(predictions)!=set(DEPENDENCIES) or set(table)!=set(DEPENDENCIES):raise ValueError('CSV route coverage must include every frozen method')
 _,ev=m2_modules();cross={}
 for label,rr in predictions.items():
  c=ev.score_csv(gtrows,rr,h,w,False)
  for k in ('tp','fp','fn'):
   if c[k]!=table[label][k]:raise ValueError('CSV count mismatch '+label+': '+k)
  for x,y in [('PQ','pq'),('SQ','sq'),('RQ','rq')]:
   if abs(c[x]-table[label][y])>=1e-7:raise ValueError('CSV score mismatch '+label+': '+x)
  cross[label]=c
 return cross
