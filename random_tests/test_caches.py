import json
import numpy as np
import pytest
from random_validation.caches import original_m2

def test_original_holdout_cache_remains_labeled_holdout_in_random_diagnostic(tmp_path):
 meta={'producer':'M2','checkpoint_sha256':'b'*64,'split_hash':'original-source','subset':'holdout','dtype':'float16','zero_below':.001,'coords':'original image pixels, origin top-left, array HxW = (row, col)','size':[8,9],'run_id':'original'}
 p=tmp_path/'source.npz';np.savez_compressed(p,__meta__=json.dumps(meta),obs=np.full((8,9),.4,np.float16));info={'checkpoint_sha256':'b'*64,'source_split_hash':'original-source'}
 with np.load(p,allow_pickle=False) as z:
  a,m=original_m2(z,'obs',info,(8,9));assert m['original_metadata']['subset']=='holdout' and m['split_hash']=='original-source'
  assert m['purpose']=='random20_mixed_training_development_diagnostic' and m['is_oof'] is False
  assert np.array_equal(a,np.full((8,9),.4,np.float16).astype(np.float32))
  with pytest.raises(ValueError):original_m2(z,'obs',dict(info,checkpoint_sha256='c'*64),(8,9))
