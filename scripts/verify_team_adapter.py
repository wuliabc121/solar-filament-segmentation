"""Cross-check frozen scores against supplied M2 evaluator and exercise apply CLI."""
import os,sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')];os.environ['PYTHONNOUSERSITE']='1';os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib');os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')
from m3_filament.io import read_jsonl,write_jsonl,write_json,sha256
from m3_filament.submission import write_submission
from team_pipeline.vendor import m2_modules
from team_adaptation.data import ground_truth
from team_adaptation.postreview import signature

report=ROOT/'reports/team_adaptation_20261104';out=report/'crosscheck';out.mkdir(exist_ok=False);protocol=json.loads((report/'protocol.json').read_text());frozen=json.loads((report/'final_check/metrics.json').read_text());gt=ground_truth(protocol['check_ids']);records=list(read_jsonl(report/'postreview/M3_M2_check19_corrected.jsonl'));_,ev=m2_modules();gtrows=[];predrows=[]
for obs,passes in gt.items():
 for rid,masks in passes.items():
  for k,m in enumerate(masks,1):gtrows.append((f'{rid}_{k}',m['counts']))
for rec in records:
 for i in rec['instances']:predrows.append((f"{rec['observation_id']}_{i['instance_id']}",i['segmentation']['counts']))
ev.write_rows(out/'ground_truth_check19.csv',gtrows);ev.write_rows(out/'predicted_check19_coarse.csv',predrows);score=ev.score_csv(gtrows,predrows,2048,2048,True);target=frozen['primary']['M3_M2_selected']
for k in ('tp','fp','fn'):assert score[k]==target[k],k
for k,other in [('PQ','pq'),('SQ','sq'),('RQ','rq')]:assert abs(score[k]-target[other])<1e-7,k
write_json(out/'M2_score_csv.json',score)
try:write_submission(records,out/'blocked_final_submission.csv')
except ValueError as e:blocked=str(e)
else:raise AssertionError('M4-required instances were accepted as final submission')
assert not (out/'blocked_final_submission.csv').exists()
# Actual CLI accepts only full raw proposal contract, preserves exact frozen geometry.
subprocess.run([str(ROOT/'adapt.sh'),'apply','--instances',str(report/'cache/m3_raw/instances.jsonl'),'--m2',str(report/'cache/m2'),'--profile',str(report/'deployment_profile.json'),'--mode','m2','--run-id','verified_application_M2','--output',str(out/'application_M2_all56.jsonl')],check=True,cwd=ROOT)
actual=list(read_jsonl(out/'application_M2_all56.jsonl'));expected={r['observation_id']:r for r in read_jsonl(report/'postreview/M3_M2_all56_corrected.jsonl')};profilehash=sha256(report/'deployment_profile.json')
for r in actual:
 assert signature(r)==signature(expected[r['observation_id']])
 assert r['source']['frozen_profile_sha256']==profilehash
 for i in r['instances']:assert i['source']['frozen_profile_sha256']==profilehash
raw=[r for r in read_jsonl(report/'cache/m3_raw/instances.jsonl') if r['observation_id'] in protocol['three_model_reconstructed_group_safe_diagnostic_ids']];write_jsonl(out/'raw_diagnostic2.jsonl',raw)
subprocess.run([str(ROOT/'adapt.sh'),'apply','--instances',str(out/'raw_diagnostic2.jsonl'),'--m1',str(report/'cache/m1'),'--m2',str(report/'cache/m2'),'--profile',str(report/'deployment_profile.json'),'--mode','guided_both','--run-id','verified_application_guided_both','--output',str(out/'application_guided_both_diagnostic2.jsonl')],check=True,cwd=ROOT)
old={r['observation_id']:r for r in read_jsonl(report/'postreview/M3_both_guided_transfer_diagnostic2_corrected.jsonl')}
for r in read_jsonl(out/'application_guided_both_diagnostic2.jsonl'):assert signature(r)==signature(old[r['observation_id']])
result={'status':'passed','M2_CSV_matches_core_metrics':True,'tp':score['tp'],'fp':score['fp'],'fn':score['fn'],'PQ':score['PQ'],'all56_actual_apply_CLI_geometry_identical':True,'guided_both_actual_CLI_diagnostic2_identical':True,'M4_required_instances_block_final_submission':True,'submission_rejection':blocked,'no_new_model_inference':True,'no_reselection_or_new_quality_tuning':True};write_json(out/'metrics.json',result);print(json.dumps(result,indent=2))
