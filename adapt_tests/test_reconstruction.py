import numpy as np
import pytest
from team_adaptation.reconstruction import AdaptConfig,prepare_record,adapt_prepared,combine_semantics
from m3_filament.schema import make_record,make_instance,validate_record
from m3_filament.masks import decode

def record(masks,scores=None):
 src={'run_id':'coarse','checkpoint_sha256':'c'*64,'split_hash':'m3-source'};scores=scores or [.8]*len(masks)
 return make_record('obs',masks[0].shape,[make_instance(m,i+1,s,src) for i,(m,s) in enumerate(zip(masks,scores))],src)

def metadata(model):return {'model':model,'observation_id':'obs','checkpoint_hash':('a' if model=='m1' else 'b')*64,'split_hash':model+'-protocol','run_id':model+'-run','coordinate_space':'original','is_oof':False,'purpose':'development_diagnostic'}

def test_reconstruction_grows_connected_support_and_preserves_hole_excludes_unseeded_island():
 m=np.zeros((32,36),np.uint8);m[10:12,10:13]=1;p=np.zeros(m.shape,np.float32);p[10:12,10:21]=.8;p[16:18,15:18]=.8
 cfg=AdaptConfig(min_detector=.1,grow_threshold=.3,seed_threshold=.6,radius=10,min_area=1,min_seed_fraction=.1,fallback_detector=1.)
 out=adapt_prepared(prepare_record(record([m])),p,{'m2':metadata('m2')},cfg,'adapted')
 validate_record(out,exclusive=True);got=decode(out['instances'][0]['segmentation']);assert got[10:12,10:21].all();assert not got[16:18,15:18].any()
 assert out['instances'][0]['bbox']==[10,10,21,12];assert out['source']['semantic_components']['m2']['split_hash']=='m2-protocol'
 assert out['instances'][0]['crop']['original_shape']==[32,36]

def test_owner_protection_and_fallback_never_overlap():
 a=np.zeros((16,24),np.uint8);b=a.copy();a[5:9,3:7]=1;b[5:9,10:14]=1;p=np.zeros(a.shape,np.float32);p[5:9,3:14]=.9
 cfg=AdaptConfig(min_detector=.1,grow_threshold=.3,seed_threshold=.5,radius=10,min_area=1,min_seed_fraction=.1)
 out=adapt_prepared(prepare_record(record([a,b],[.9,.8])),p,{'m2':metadata('m2')},cfg,'adapted');validate_record(out,exclusive=True)
 assert len(out['instances'])==2
 assert (decode(out['instances'][1]['segmentation'])&b).sum()==b.sum()

def test_empty_semantic_drops_weak_and_retains_declared_strong_fallback():
 a=np.zeros((12,16),np.uint8);a[3:7,4:8]=1;cfg=AdaptConfig(min_detector=.1,min_area=1,fallback_detector=.7)
 low=adapt_prepared(prepare_record(record([a],[.3])),np.zeros(a.shape,np.float32),{'m2':metadata('m2')},cfg,'low');assert not low['instances']
 high=adapt_prepared(prepare_record(record([a],[.9])),np.zeros(a.shape,np.float32),{'m2':metadata('m2')},cfg,'high');assert high['instances'][0]['fallback_used']

def test_semantic_identity_range_and_distinct_protocols():
 shape=(12,16);a=np.full(shape,.4,np.float32);b=np.full(shape,.8,np.float32)
 p,meta=combine_semantics({'m1':(a,metadata('m1')),'m2':(b,metadata('m2'))},'both','obs',shape);assert np.allclose(p,.6);assert meta['m1']['split_hash']!=meta['m2']['split_hash']
 wrong=metadata('m1');wrong['observation_id']='other'
 with pytest.raises(ValueError):combine_semantics({'m1':(a,wrong)},'m1','obs',shape)
 with pytest.raises(ValueError):AdaptConfig(grow_threshold=.8,seed_threshold=.5)
 with pytest.raises(ValueError):AdaptConfig(radius=-1)

def test_guided_cnn_assist_never_suppresses_transformer_and_preserves_uncertain_boundary():
 pt=np.array([[.05,.3,.8]],np.float32);pc=np.array([[.9,.9,.2]],np.float32)
 got,components=combine_semantics({'m1':(pc,metadata('m1')),'m2':(pt,metadata('m2'))},'guided_both','obs',pt.shape)
 assert np.allclose(got,[[.05,.45,.8]])
 assert (got>=pt).all() and (got<=1).all()
 assert components['m1']['split_hash']=='m1-protocol' and components['m2']['split_hash']=='m2-protocol'
 assert 'fusion_rule' in components['m1']
