import json
import numpy as np
import pytest
from team_pipeline.cache import convert_m2_cache
from team_pipeline.contracts import load_probability

def test_convert_preserves_quantization_and_source_protocol(tmp_path):
 src=tmp_path/'packed.npz';m={'producer':'M2','split_hash':'different','checkpoint_sha256':'c'*64,'run_id':'external','subset':'calib','size':[8,9],'dtype':'float16','zero_below':.001,'coords':'original image pixels, origin top-left, array HxW = (row, col)'}
 np.savez_compressed(src,__meta__=json.dumps(m),obs=np.full((8,9),.25,np.float16))
 registry={'models':{'m2':{'source_split_hash':'different','checkpoint_sha256':'c'*64}}}
 files=convert_m2_cache(src,tmp_path/'converted',['obs'],registry)
 a,info=load_probability(files[0],'obs',(8,9));assert info['split_hash']=='different' and info['original_metadata']['zero_below']==.001
 assert np.all(a==.25) and info['purpose']=='imported_cache'
 with pytest.raises(ValueError):convert_m2_cache(src,tmp_path/'other',['missing'],registry)
 m['subset']='holdout';np.savez_compressed(tmp_path/'hold.npz',__meta__=json.dumps(m),obs=np.ones((8,9)))
 with pytest.raises(ValueError,match='holdout'):convert_m2_cache(tmp_path/'hold.npz',tmp_path/'blocked',['obs'],registry)

@pytest.mark.parametrize('change,actual_dtype',[
 ({'coords':'resized 512px'},np.float16),
 ({'dtype':'float32'},np.float16),
 ({'zero_below':.1},np.float16),
 ({},np.float32),
])
def test_reject_incompatible_source_cache(tmp_path,change,actual_dtype):
 m={'producer':'M2','split_hash':'s','checkpoint_sha256':'c'*64,'run_id':'external','subset':'calib','size':[8,9],'dtype':'float16','zero_below':.001,'coords':'original image pixels, origin top-left, array HxW = (row, col)'};m.update(change)
 src=tmp_path/'bad.npz';np.savez_compressed(src,__meta__=json.dumps(m),obs=np.full((8,9),.2,actual_dtype))
 registry={'models':{'m2':{'source_split_hash':'s','checkpoint_sha256':'c'*64}}}
 with pytest.raises(ValueError):convert_m2_cache(src,tmp_path/'out',['obs'],registry)
 assert not (tmp_path/'out/obs.npz').exists()
