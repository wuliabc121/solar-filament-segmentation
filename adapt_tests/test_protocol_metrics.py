import numpy as np
import pytest
from team_adaptation.protocol import validate_cohort
from team_adaptation.metrics import score_records
from m3_filament.schema import make_record,make_instance
from m3_filament.masks import encode

def test_validate_cohort_rejects_original_holdout_and_group_overlap():
 info={'ok':{'date':'1','sha256':'a'},'trained':{'date':'2','sha256':'b'},'same_date':{'date':'2','sha256':'c'},'same_content':{'date':'3','sha256':'b'},'held':{'date':'4','sha256':'d'}}
 model={'training_observations':['trained']};validate_cohort(['ok'],[model],info,{'held'})
 for obs in ('trained','same_date','same_content','held'):
  with pytest.raises(ValueError):validate_cohort([obs],[model],info,{'held'})

def test_metrics_pools_independent_records_and_handles_empty_prediction():
 a=np.zeros((12,15),np.uint8);a[2:5,3:8]=1
 src={'run_id':'unit'};rec=make_record('obs',a.shape,[make_instance(a,1,.9,src)],src)
 gt={'obs':{'annotator1':[encode(a)],'annotator2':[encode(a)],'empty':[]}}
 result,rows=score_records([rec],gt,['obs']);assert result['pq']==1 and result['tp']==2 and result['n_gt']==2 and result['small_recall']==1
 assert result['annotation_records']==2 and result['zero_gt_excluded']==1
 rec['instances']=[];r,_=score_records([rec],gt,['obs']);assert r['fn']==2 and r['fp']==0 and r['recall']==0
 with pytest.raises(ValueError):score_records([rec,rec],gt,['obs'])

def test_fast_validation_agrees_with_original_and_rejects_invalid_masks():
 import copy
 from team_adaptation.metrics import validate_fast_record
 from m3_filament.schema import validate_record
 a=np.zeros((12,15),np.uint8);a[2:5,3:8]=1;src={'run_id':'unit'};r=make_record('obs',a.shape,[make_instance(a,1,.9,src)],src)
 validate_fast_record(r);validate_record(r,exclusive=True)
 variants=[];x=copy.deepcopy(r);x['instances'][0]['bbox']=[0,0,15,12];variants.append(x)
 x=copy.deepcopy(r);x['instances'][0]['score']=float('nan');variants.append(x)
 x=copy.deepcopy(r);second=copy.deepcopy(x['instances'][0]);second['instance_id']=2;x['instances'].append(second);variants.append(x)
 x=copy.deepcopy(r);x['image_size']=[13,15];variants.append(x)
 x=copy.deepcopy(r);x['coordinate_convention']='unknown';variants.append(x)
 for x in variants:
  with pytest.raises(ValueError):validate_fast_record(x)
