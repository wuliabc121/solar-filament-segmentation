import numpy as np
from random_validation.scoring import cc_record
from team_adaptation.association import prepare_components
from team_adaptation.metrics import score_records
from m3_filament.masks import encode

def test_semantic_component_baseline_has_real_geometry_and_mean_probability():
 a=np.zeros((12,15),np.float32);a[2:6,3:8]=.8
 metadata={'m1':{'model':'m1','split_hash':'real-M1-source','checkpoint_hash':'a'*64}}
 r=cc_record('obs',prepare_components(a,.5,2),metadata,'M1_CC_050')
 assert r['instances'][0]['score_kind']=='mean_semantic_probability'
 assert r['instances'][0]['bbox']==[3,2,8,6]
 assert abs(r['instances'][0]['score']-.8)<1e-6
 m,_=score_records([r],{'obs':{'ann-obs':[encode((a>.5).astype(np.uint8))]}},['obs']);assert m['pq']==1
 assert r['source']['semantic_components']['m1']['split_hash']=='real-M1-source'

def test_csv_verification_requires_every_route_including_M2_and_structure_baseline():
 import pytest
 from random_validation.scoring import validate_csv_routes,DEPENDENCIES
 mask=np.zeros((8,9),np.uint8);mask[2:5,3:7]=1;counts=encode(mask)['counts'];gt=[('ann-obs_1',counts)];pred={k:[('obs_1',counts)] for k in DEPENDENCIES};metrics={k:{'tp':1,'fp':0,'fn':0,'pq':1.,'sq':1.,'rq':1.} for k in DEPENDENCIES}
 result=validate_csv_routes(gt,pred,metrics,8,9);assert set(result)==set(DEPENDENCIES)
 del pred['M2_CC_035']
 with pytest.raises(ValueError,match='coverage'):validate_csv_routes(gt,pred,metrics,8,9)
