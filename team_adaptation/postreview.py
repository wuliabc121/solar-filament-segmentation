"""Export corrected provenance/refiner metadata; prove frozen masks/scores are unchanged."""
import json,time
from datetime import datetime,timezone
from .experiment import REPORT,update_state
from .protocol import check_deadline
from .application import apply_one
from m3_filament.io import read_jsonl,write_jsonl,write_json,sha256
from team_pipeline.contracts import load_probability

def signature(record):return [(i['segmentation'],i['bbox'],i['score'],i.get('source_instance_id')) for i in record['instances']]

def export_corrected(protocol):
 check_deadline(protocol);profile=json.loads((REPORT/'frozen_adaptation_v2.json').read_text());out=REPORT/'postreview';out.mkdir();raw={r['observation_id']:r for r in read_jsonl(REPORT/'cache/m3_raw/instances.jsonl')};old_tune={r['observation_id']:r for r in read_jsonl(REPORT/'association_tuning/predictions/assoc_07.jsonl')};old_check={r['observation_id']:r for r in read_jsonl(REPORT/'final_check/M3_M2_selected.jsonl')};baseline={**old_tune,**old_check};records=[];count=0;started=time.monotonic();update_state('postreview','exporting_metadata_correction',active=True)
 for s in protocol['primary_ids']:
  check_deadline(protocol);rec=raw[s];pt,mt=load_probability(REPORT/'cache/m2'/(s+'.npz'),s,rec['image_size']);got=apply_one(rec,{'m2':(pt,mt)},profile,'m2','M3_M2_association_corrected_metadata')
  if signature(got)!=signature(baseline[s]):raise ValueError('metadata-only correction changed frozen masks/scores: '+s)
  count+=sum(bool(i.get('requires_refiner')) for i in got['instances']);records.append(got)
 write_jsonl(out/'M3_M2_all56_corrected.jsonl',records);write_jsonl(out/'M3_M2_check19_corrected.jsonl',[r for r in records if r['observation_id'] in protocol['check_ids']]);write_jsonl(out/'M4_ambiguity_handoff.jsonl',[dict(r,instances=[i for i in r['instances'] if i.get('requires_refiner')]) for r in records if any(i.get('requires_refiner') for i in r['instances'])])
 # Verify the two diagnostic transfer routes also preserve masks/scores after metadata correction.
 diag_counts={}
 for mode,name in [('m1','M3_M1_transfer'),('both','M3_both_mean_transfer'),('guided_both','M3_both_guided_transfer')]:
  prior={r['observation_id']:r for r in read_jsonl(REPORT/'final_check'/(name+'_diagnostic2.jsonl'))};rs=[]
  for s in protocol['three_model_reconstructed_group_safe_diagnostic_ids']:
   rec=raw[s];sem={k:load_probability(REPORT/'cache'/k/(s+'.npz'),s,rec['image_size']) for k in (['m1'] if mode=='m1' else ['m1','m2'])};got=apply_one(rec,sem,profile,mode,name+'_corrected');assert signature(got)==signature(prior[s]);rs.append(got)
  write_jsonl(out/(name+'_diagnostic2_corrected.jsonl'),rs);diag_counts[name]=sum(sum(bool(i.get('requires_refiner')) for i in r['instances']) for r in rs)
 deployment=dict(profile,selected_source_profile_sha256=sha256(REPORT/'frozen_adaptation_v2.json'),exported_utc=datetime.now(timezone.utc).isoformat(),raw_proposal_contract=json.loads((REPORT/'cache/m3_raw/run_manifest.json').read_text())['inference_config'],m3_source_manifest_sha256='564c3b3eaa1d93b89003dac0f37d64d5a03670349d0f231eef3694d62cb2b007',status='candidate_PQ_recall_improved_but_FP_SQ_gate_failed_not_default',default_semantic_mode='m2',guided_both_status='experimental_transfer_only2_cases',ambiguity_metadata_rule={'detector_score_gte':.45,'component_iou_gte':.1,'independent_of_anchor_acceptance':True})
 write_json(REPORT/'deployment_profile.json',deployment)
 checkcount=sum(sum(bool(i.get('requires_refiner')) for i in r['instances']) for r in records if r['observation_id'] in protocol['check_ids']);metrics={'status':'passed','all56_masks_scores_boxes_parent_ids_unchanged':True,'diagnostic2_all3_modes_masks_scores_unchanged':True,'fixed_profile_no_reselection':True,'coarse_quality_metrics_unchanged':True,'total_56_refiner_required':count,'check19_refiner_required':checkcount,'diagnostic_refiner_required':diag_counts,'previous_final_check_flag_counts_superseded':True,'original_final_check_files_preserved':True,'new_quality_scoring':False,'seconds':time.monotonic()-started}
 write_json(out/'metrics.json',metrics);update_state('postreview','complete',active=False,refiner_flag_counts_superseded=True,refiner_required_check19=checkcount,old_default_replaced=False);print(json.dumps(metrics,indent=2))
