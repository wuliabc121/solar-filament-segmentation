"""Adopt true source probabilities without renaming old split/subset provenance."""
import json
import numpy as np
from m3_filament.fusion import probability

def original_m2(z,obs,info,shape):
 m=json.loads(str(z['__meta__']))
 if m.get('producer')!='M2' or m.get('checkpoint_sha256')!=info['checkpoint_sha256'] or m.get('split_hash')!=info['source_split_hash']:raise ValueError('M2 source identity mismatch')
 if m.get('dtype')!='float16' or m.get('zero_below')!=.001 or m.get('coords')!='original image pixels, origin top-left, array HxW = (row, col)':raise ValueError('M2 source probability contract mismatch')
 a=z[obs]
 if a.dtype!=np.float16 or tuple(a.shape)!=tuple(shape) or list(a.shape)!=m['size']:raise ValueError('M2 source dtype/size mismatch')
 a=a.astype(np.float32);probability(a,shape)
 meta={'model':'m2','observation_id':obs,'checkpoint_hash':m['checkpoint_sha256'],'split_hash':m['split_hash'],'run_id':m['run_id'],'coordinate_space':'original','is_oof':False,'purpose':'random20_mixed_training_development_diagnostic','original_metadata':m,'training_membership_status':'manifest_verified','original_subset_preserved':m['subset']}
 return a,meta
