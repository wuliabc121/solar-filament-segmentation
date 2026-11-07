import copy
import pytest
from team_adaptation.application import validate_input

def test_application_refuses_missing_low_confidence_proposals_or_wrong_checkpoint():
 profile={'models':{'m3':'c'*64},'family':'association','config':{'min_detector':.15}}
 rec={'source':valid_raw_source()}
 validate_input(rec,profile)
 for change in ({'checkpoint_sha256':'d'*64},{'inference_config':{'conf':.45,'imgsz':2048}},{'inference_config':{'conf':.05,'imgsz':1024}},{'inference_config':{'conf':.05,'imgsz':2048,'postprocess':{'score_floor':.45}}}):
  bad=copy.deepcopy(rec);bad['source'].update(change)
  with pytest.raises(ValueError):validate_input(bad,profile)

def valid_raw_source():
 from m3_filament.io import content_hash
 cfg={'device':'cpu','imgsz':2048,'conf':.05,'iou':.95,'retina_masks':True,'max_det':100,'verbose':False,'save':False}
 return {'run_id':'raw','checkpoint_sha256':'c'*64,'inference_config':cfg,'inference_config_sha256':content_hash(cfg),'source_manifest_sha256':'564c3b3eaa1d93b89003dac0f37d64d5a03670349d0f231eef3694d62cb2b007','ultralytics_version':'8.3.221'}

@pytest.mark.parametrize('change',[{'max_det':1},{'iou':.1},{'postprocess':{'min_area':10000,'closing_radius':3,'score_floor':.05}}])
def test_apply_rejects_other_raw_proposal_contracts(change):
 from m3_filament.io import content_hash
 source=valid_raw_source();source['inference_config'].update(change);source['inference_config_sha256']=content_hash(source['inference_config'])
 profile={'models':{'m3':'c'*64},'family':'association','config':{'min_detector':.15}}
 with pytest.raises(ValueError):validate_input({'source':source},profile)

def test_apply_rejects_inconsistent_configuration_hash():
 source=valid_raw_source();source['inference_config_sha256']='0'*64
 with pytest.raises(ValueError):validate_input({'source':source},{'models':{'m3':'c'*64},'config':{'min_detector':.15}})

def test_repeated_postprocessing_cannot_hide_irreversible_truncation():
 import numpy as np
 from m3_filament.schema import make_record,make_instance
 from m3_filament.postprocess import transform_record
 a=np.zeros((12,15),np.uint8);b=a.copy();a[2:5,2:5]=1;b[7:10,8:12]=1;s=valid_raw_source();r=make_record('unit',a.shape,[make_instance(a,1,.8,s),make_instance(b,2,.3,s)],s)
 high=transform_record(r,'high',0,0,.45);low=transform_record(high,'low',0,0,.05);assert len(low['instances'])==1
 with pytest.raises(ValueError):validate_input(low,{'models':{'m3':'c'*64},'config':{'min_detector':.15}})
