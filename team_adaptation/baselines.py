import numpy as np
from scipy import ndimage
from m3_filament.schema import make_record,make_instance,crop_metadata,validate_record
from m3_filament.postprocess import transform_record
from m3_filament.fusion import fuse_record,FusionConfig

def frozen_m3(record):return transform_record(record,'frozen_m3_045_area32',32,0,.45)

def semantic_components(record,prob,threshold=.35,min_area=400,model='m2'):
 src={'run_id':f'{model}_semantic_cc','semantic_model':model,'threshold':threshold,'min_area':min_area,'benchmark_status':'reused_calibration_diagnostic'}
 lab,n=ndimage.label(prob>=threshold,structure=np.ones((3,3)));sizes=np.bincount(lab.ravel());out=[]
 for i in np.flatnonzero(sizes>=min_area):
  if i==0:continue
  mask=lab==i;inst=make_instance(mask,len(out)+1,float(prob[mask].mean()),src);inst['score_kind']='mean_semantic_probability';inst['crop']=crop_metadata(inst,record['image_size']);out.append(inst)
 rec=make_record(record['observation_id'],record['image_size'],out,src);validate_record(rec,exclusive=True);return rec

def initial_fusion(record,pt,pc=None):
 m=frozen_m3(record);out=fuse_record(m,pt if pc is None else pc,pt,FusionConfig(alpha=.5,threshold=.5,min_area=8),'initial_E5_m2' if pc is None else 'initial_E5_both_diagnostic')
 out['source'].update(semantic_weights={'m2':1.} if pc is None else {'m1':.5,'m2':.5},benchmark_eligible=False,diagnostic=True)
 return out
