import json,time,shutil
from dataclasses import asdict
from datetime import datetime,timezone
from .experiment import REPORT,update_state
from .protocol import check_deadline
from .data import ground_truth
from .metrics import score_records,aggregate_rows
from .reconstruction import prepare_record,combine_semantics
from .association import AssociationConfig,prepare_components,associate
from team_pipeline.contracts import load_probability
from m3_filament.io import read_jsonl,write_json,write_jsonl,sha256

def tune_association(protocol):
 cfgs=json.loads((REPORT/'association_grid.json').read_text())['points'];points={k:AssociationConfig(**v) for k,v in cfgs.items()};ids=protocol['tune_ids'];gt=ground_truth(ids);raw={r['observation_id']:r for r in read_jsonl(REPORT/'cache/m3_raw/instances.jsonl')};out=REPORT/'association_tuning';out.mkdir();pred=out/'predictions';pred.mkdir();rows={k:[] for k in points};secs={k:0. for k in points};handles={k:(pred/(k+'.jsonl')).open('x') for k in points};started=time.monotonic();update_state('B2','tuning_association',active=True,completed_observations=0)
 try:
  for index,s in enumerate(ids):
   check_deadline(protocol);rec=raw[s];prep=prepare_record(rec);pt,m=load_probability(REPORT/'cache/m2'/(s+'.npz'),s,rec['image_size']);p,meta=combine_semantics({'m2':(pt,m)},'m2',s,rec['image_size']);components=prepare_components(p,.35,400)
   for name,cfg in points.items():
    check_deadline(protocol);t=time.monotonic();r=associate(prep,components,meta,cfg,name);secs[name]+=time.monotonic()-t;_,ann=score_records([r],{s:gt[s]},[s]);rows[name].extend(ann);handles[name].write(json.dumps(r)+'\n')
   update_state('B2','tuning_association',active=True,completed_observations=index+1,total_observations=len(ids),seconds=time.monotonic()-started)
 finally:
  for f in handles.values():f.close()
 table={k:{'config':asdict(points[k]),'metrics':aggregate_rows(rows[k],len(ids)),'association_seconds':secs[k]} for k in points};baseline=json.loads((REPORT/'baseline_tune/metrics.json').read_text())['M3_frozen']
 feasible=[k for k,v in table.items() if v['metrics']['pq']>baseline['pq'] and v['metrics']['recall']>=baseline['recall'] and v['metrics']['fp']<=baseline['fp']]
 allmetrics=[k for k in feasible if table[k]['metrics']['sq']>=baseline['sq'] and table[k]['metrics']['small_recall']>=baseline['small_recall']]
 selected=max(allmetrics or feasible or list(table),key=lambda k:(table[k]['metrics']['pq'],-table[k]['metrics']['fp'],table[k]['metrics']['recall']));previous=json.loads((REPORT/'frozen_adaptation.json').read_text());selected_metrics=table[selected]['metrics'];choose_new=bool(feasible) and selected_metrics['pq']>previous['tune_metrics']['pq']
 frozen={'frozen_utc':datetime.now(timezone.utc).isoformat(),'family':'association' if choose_new else 'reconstruct','point':selected if choose_new else previous['point'],'config':table[selected]['config'] if choose_new else previous['config'],'tune_metrics':selected_metrics if choose_new else previous['tune_metrics'],'baseline_tune':baseline,'gate_met_on_tune':bool(feasible) if choose_new else previous['gate_met_on_tune'],'all_reported_metrics_gate_on_tune':bool(allmetrics) if choose_new else False,'selection_uses':'M2 only tune37; neither check19 nor diagnostic2 scored','protocol_sha256':sha256(REPORT/'protocol.json'),'grid_sha256':sha256(REPORT/'association_grid.json') if choose_new else previous['grid_sha256'],'prior_frozen_profile_sha256':sha256(REPORT/'frozen_adaptation.json'),'models':previous['models'],'check_ids':protocol['check_ids'],'M1_transfer_predeclared_modes':['m1','both','guided_both'],'M1_guided_formula':'P_T + .25 max(P_C-P_T,0) I(P_T>=.1); fixed, not tuned on M1 contaminated observations','not_pristine_test':True}
 write_json(out/'grid_metrics.json',table);write_jsonl(out/'selected_per_annotation.jsonl',rows[selected]);write_json(REPORT/'frozen_adaptation_v2.json',frozen);result={'status':'complete','selected':frozen,'candidate_best':selected,'candidate_best_metrics':selected_metrics,'points':len(points),'seconds':time.monotonic()-started,'check_scored':False};write_json(out/'metrics.json',result);update_state('B2','complete',active=False,selected=frozen['point'],check_scores_read=False);print(json.dumps(result,indent=2))
