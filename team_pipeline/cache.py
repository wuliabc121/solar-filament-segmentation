"""Convert supplied M2 cache without changing its source experiment identity."""
import json
from pathlib import Path
import numpy as np
from m3_filament.io import sha256
from .contracts import save_probability

def convert_m2_cache(source,output,observations,registry):
 if not observations or len(set(observations))!=len(observations):raise ValueError('explicit unique observations required')
 if any(Path(s).name!=s or s in ('.','..') for s in observations):raise ValueError('unsafe observation id')
 digest=sha256(source);paths=[]
 with np.load(source,allow_pickle=False) as z:
  original=json.loads(str(z['__meta__']));info=registry['models']['m2']
  if original.get('subset') in ('holdout','test'):raise ValueError('holdout/test cache is archived only; conversion blocked')
  if original.get('producer')!='M2' or original['checkpoint_sha256']!=info['checkpoint_sha256'] or original['split_hash']!=info['source_split_hash']:raise ValueError('M2 cache identity mismatch')
  if original.get('coords')!='original image pixels, origin top-left, array HxW = (row, col)' or original.get('dtype')!='float16' or original.get('zero_below')!=.001:raise ValueError('unsupported M2 coordinate/quantization contract')
  if not all(s in z.files for s in observations):raise ValueError('missing observation in cache')
  # Validate all selected arrays before publishing any output.
  for s in observations:
   if z[s].dtype!=np.float16 or list(z[s].shape)!=original['size']:raise ValueError('source cache dtype/shape mismatch')
  for s in observations:
   a=z[s].astype(np.float32)
   if list(a.shape)!=original['size']:raise ValueError('source cache shape mismatch')
   meta={'model':'m2','observation_id':s,'run_id':original['run_id'],'checkpoint_hash':original['checkpoint_sha256'],'split_hash':original['split_hash'],'coordinate_space':'original','is_oof':False,'purpose':'imported_cache','training_membership_status':'manifest_verified','source_cache_sha256':digest,'original_metadata':original}
   dest=Path(output)/(s+'.npz');save_probability(dest,a,meta);paths.append(dest)
 return paths
