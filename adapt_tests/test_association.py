import numpy as np
from team_adaptation.association import AssociationConfig,prepare_components,associate
from team_adaptation.reconstruction import prepare_record
from m3_filament.schema import make_record,make_instance,validate_record
from m3_filament.masks import decode

def test_detector_anchor_accepts_full_component_but_rejects_unanchored_false_positive():
 m=np.zeros((32,40),np.uint8);m[5:9,4:10]=1;p=np.zeros(m.shape,np.float32);p[5:9,4:20]=.8;p[20:25,28:35]=.8
 src={'run_id':'coarse','checkpoint_sha256':'c'*64};rec=make_record('obs',m.shape,[make_instance(m,1,.3,src)],src)
 meta={'model':'m2','observation_id':'obs','run_id':'sem','checkpoint_hash':'b'*64,'split_hash':'m2','coordinate_space':'original','is_oof':False,'purpose':'development_diagnostic'}
 out=associate(prepare_record(rec),prepare_components(p,.35,4),{'m2':meta},AssociationConfig(min_detector=.15,min_iou=.15,min_area=4),'associated')
 validate_record(out,exclusive=True);assert len(out['instances'])==1
 got=decode(out['instances'][0]['segmentation']);assert got[5:9,4:20].all() and not got[20:25,28:35].any()
 assert out['instances'][0]['source_instance_id']==1

def test_one_merged_detector_can_anchor_two_disconnected_semantic_instances():
 m=np.zeros((24,32),np.uint8);m[4:16,4:24]=1;p=np.zeros(m.shape,np.float32);p[5:10,5:12]=.9;p[10:15,17:23]=.9
 src={'run_id':'coarse'};rec=make_record('obs',m.shape,[make_instance(m,1,.8,src)],src)
 out=associate(prepare_record(rec),prepare_components(p,.35,4),{'m2':{'model':'m2','observation_id':'obs','run_id':'sem','checkpoint_hash':'b'*64,'split_hash':'m2','coordinate_space':'original','is_oof':False,'purpose':'development_diagnostic'}},AssociationConfig(min_detector=.1,min_iou=.05,min_area=4),'split')
 validate_record(out,exclusive=True);assert len(out['instances'])==2
 assert [i['source_instance_id'] for i in out['instances']]==[1,1]

def test_second_strong_seed_is_flagged_even_when_below_anchor_acceptance_iou():
 a=np.zeros((16,20),np.uint8);b=a.copy();a[4:12,4:10]=1;b[4:12,10:14]=1;p=np.zeros(a.shape,np.float32);p[4:12,4:14]=.9
 s={'run_id':'coarse'};r=make_record('obs',a.shape,[make_instance(a,1,.9,s),make_instance(b,2,.8,s)],s)
 meta={'model':'m2','observation_id':'obs','run_id':'sem','checkpoint_hash':'b'*64,'split_hash':'m2','coordinate_space':'original','is_oof':False,'purpose':'development_diagnostic'}
 got=associate(prepare_record(r),prepare_components(p,.35,4),{'m2':meta},AssociationConfig(min_detector=.15,min_iou=.5,min_area=4),'anchored')
 assert len(got['instances'])==1
 inst=got['instances'][0];assert inst['requires_refiner'] is True
 assert inst['ambiguous_strong_M3_instance_ids']==[1,2]
 assert inst['source_instance_id']==1 and inst['score']==.9
 assert decode(inst['segmentation'])[4:12,4:14].all()
