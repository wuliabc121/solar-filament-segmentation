import json,sys
from pathlib import Path
import numpy as np
import pytest
from team_pipeline.contracts import save_probability,load_probability,fuse_team,check_evaluation_eligibility
from team_pipeline.vendor import m2_modules
from m3_filament.evaluation import pq_counts
from m3_filament.schema import make_instance,make_record,validate_record
from m3_filament.fusion import FusionConfig

def meta(model='m1',obs='synthetic'):
 return {'model':model,'observation_id':obs,'run_id':'unit','checkpoint_hash':'a'*64,'split_hash':'split-'+model,'coordinate_space':'original','is_oof':False,'training_membership_status':'synthetic_only','purpose':'smoke'}

def test_probability_roundtrip_and_refuse_overwrite(tmp_path):
 p=tmp_path/'a.npz';a=np.full((8,9),.4,np.float32);save_probability(p,a,meta());q,m=load_probability(p,'synthetic',(8,9));assert np.array_equal(a,q);assert m['split_hash']=='split-m1'
 with pytest.raises(FileExistsError):save_probability(p,a,meta())
 with pytest.raises(ValueError):load_probability(p,'other',(8,9))
 with pytest.raises(ValueError):load_probability(p,'synthetic',(9,8))
 for bad in (np.full((8,9),np.nan),np.full((8,9),1.1)):
  with pytest.raises(ValueError):save_probability(tmp_path/'bad.npz',bad,meta())

def test_fusion_keeps_distinct_training_protocols_and_m4(tmp_path):
 a=np.zeros((16,20),np.float32);a[3:8,4:12]=.9;a[11:14,15:18]=.9
 mask=np.zeros_like(a,np.uint8);mask[3:8,4:12]=1
 source={'run_id':'coarse','split_hash':'split-m3'};r=make_record('synthetic',a.shape,[make_instance(mask,1,.8,source)],source)
 r2=fuse_team(r,(a,meta()),(a,meta('m2')),FusionConfig(residual=True,min_area=2,endpoint_guard=0),'fused','smoke')
 validate_record(r2,exclusive=True);assert len(r2['instances'])==2
 assert r2['source']['semantic_components']['m1']['split_hash']=='split-m1'
 assert r2['source']['semantic_components']['m2']['split_hash']=='split-m2'
 assert r2['source']['is_oof'] is False
 assert any(i['requires_refiner'] for i in r2['instances'])
 assert all(i['crop']['original_shape']==[16,20] for i in r2['instances'])
 with pytest.raises(ValueError,match='evaluation'):fuse_team(r,(a,meta()),(a,meta('m2')),FusionConfig(),'bad','evaluation')

def test_evaluation_blocks_unknown_or_seen_membership():
 models={'m1':{'training_observations':['x'],'training_membership_status':'reconstructed_not_attested'}}
 with pytest.raises(ValueError):check_evaluation_eligibility(['y'],models)
 models['m1']['training_membership_status']='manifest_verified'
 with pytest.raises(ValueError):check_evaluation_eligibility(['x'],models)
 check_evaluation_eligibility(['y'],models)

def test_vendor_scoped_and_float32_alignment():
 sentinel=object();old=sys.modules.get('train');sys.modules['train']=sentinel
 try:
  train,evaluator=m2_modules();assert sys.modules['train'] is sentinel
  for a in (np.array([[.5,.50000001,.6]]),np.empty((0,2)),np.empty((2,0)),np.array([[.6,.7],[.8,.9]])):
   expected=pq_counts(a);got=evaluator.official_counts(a)
   assert (got['tp'],got['fp'],got['fn'])==(expected['tp'],expected['fp'],expected['fn'])
   assert got['iou_sum']==pytest.approx(expected['tp_iou_sum'],abs=1e-6)
 finally:
  if old is None:sys.modules.pop('train',None)
  else:sys.modules['train']=old

def test_csv_independent_annotation_records():
 from pycocotools import mask as cm
 _,ev=m2_modules();a=np.zeros((8,9),np.uint8);a[2:5,3:6]=1;c=cm.encode(np.asfortranarray(a))['counts'].decode()
 r=ev.score_csv([('ann1-obs_1',c),('ann2-obs_1',c)],[('obs_1',c)],8,9,per_record=True)
 assert r['PQ']==1 and r['tp']==2 and len(r['records'])==2
