"""Tune on predeclared development IDs; freeze before final-check masks are read."""
import json,time,shutil
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from . import ROOT
from .protocol import check_deadline
from .data import ground_truth
from .metrics import score_records,aggregate_rows
from .reconstruction import AdaptConfig,prepare_record,adapt_prepared,combine_semantics
from .experiment import REPORT,update_state
from m3_filament.io import read_jsonl,write_json,write_jsonl,sha256
from m3_filament.postprocess import transform_record
from team_pipeline.contracts import load_probability

def tune(protocol):
 config=json.loads((REPORT/'adaptation_grid.json').read_text());points={k:AdaptConfig(**v) for k,v in config['points'].items()};ids=protocol['tune_ids'];g=ground_truth(ids)
 raw={r['observation_id']:r for r in read_jsonl(REPORT/'cache/m3_raw/instances.jsonl')};out=REPORT/'tuning';out.mkdir(exist_ok=False);predictions=out/'predictions';predictions.mkdir();rows={k:[] for k in points};excluded={k:0 for k in points};times={k:0. for k in points};rawrows=[]
 started=time.monotonic();update_state('B','running_adaptation_tune',active=True,points=len(points),completed_observations=0)
 handles={k:(predictions/(k+'.jsonl')).open('x') for k in points}
 try:
  for idx,s in enumerate(ids):
   check_deadline(protocol);rec=raw[s];prepared=prepare_record(rec);pt,meta=load_probability(REPORT/'cache/m2'/(s+'.npz'),s,rec['image_size']);p,components=combine_semantics({'m2':(pt,meta)},'m2',s,rec['image_size'])
   _,rr=score_records([transform_record(rec,'raw_lowconf_ablation',32,0,.05)],{s:g[s]},[s]);rawrows.extend(rr)
   for name,cfg in points.items():
    check_deadline(protocol);t=time.monotonic();result=adapt_prepared(prepared,p,components,cfg,name,validate_output=False);times[name]+=time.monotonic()-t
    m,annotations=score_records([result],{s:g[s]},[s]);rows[name].extend(annotations);excluded[name]+=m['zero_gt_excluded'];handles[name].write(json.dumps(result)+'\n')
   update_state('B','running_adaptation_tune',active=True,points=len(points),completed_observations=idx+1,total_observations=len(ids),seconds=time.monotonic()-started)
 finally:
  for f in handles.values():f.close()
 table={k:{'config':asdict(points[k]),'metrics':aggregate_rows(rows[k],len(ids),excluded[k]),'reconstruction_seconds':times[k]} for k in points}
 baseline=json.loads((REPORT/'baseline_tune/metrics.json').read_text())['M3_frozen'];feasible=[k for k,v in table.items() if v['metrics']['pq']>baseline['pq'] and v['metrics']['recall']>=baseline['recall'] and v['metrics']['fp']<=baseline['fp']]
 selected=max(feasible or list(table),key=lambda k:(table[k]['metrics']['pq'],-table[k]['metrics']['fp'],table[k]['metrics']['recall']))
 frozen={'frozen_utc':datetime.now(timezone.utc).isoformat(),'point':selected,'config':table[selected]['config'],'gate_met_on_tune':bool(feasible),'tune_metrics':table[selected]['metrics'],'baseline_tune':baseline,'selection_uses':'M2_only; tune37 only; no M1 outputs/check19 metrics','protocol_sha256':sha256(REPORT/'protocol.json'),'grid_sha256':sha256(REPORT/'adaptation_grid.json'),'selected_tune_prediction_sha256':sha256(predictions/(selected+'.jsonl')),'models':{k:v['checkpoint_sha256'] for k,v in protocol['models'].items()},'check_ids':protocol['check_ids'],'not_pristine_test':True}
 write_json(out/'grid_metrics.json',table);write_json(out/'raw_lowconf_ablation.json',aggregate_rows(rawrows,len(ids)));write_jsonl(out/'selected_per_annotation.jsonl',rows[selected]);shutil.copy2(predictions/(selected+'.jsonl'),out/'selected_predictions.jsonl');write_json(REPORT/'frozen_adaptation.json',frozen)
 result={'status':'complete','selected':selected,'gate_met_on_tune':bool(feasible),'selected_metrics':table[selected]['metrics'],'baseline_metrics':baseline,'points':len(table),'seconds':time.monotonic()-started,'check_scored':False}
 write_json(out/'metrics.json',result);update_state('B','complete',active=False,selected=selected,check_scores_read=False);print(json.dumps(result,indent=2))
