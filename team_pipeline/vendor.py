"""Load reviewed vendor modules without polluting generic train/models imports."""
import importlib.util,sys
from functools import lru_cache
from . import ROOT
from m3_filament.evaluation import pq_counts

def aligned_counts(matrix):
 c=pq_counts(matrix)
 return {'tp':c['tp'],'fp':c['fp'],'fn':c['fn'],'iou_sum':c['tp_iou_sum']}

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@lru_cache(maxsize=1)
def m2_modules():
 train=module('_team_m2_train',ROOT/'team/m2/source/train.py')
 train.ROOT=ROOT;train.DEFAULT_ANN=ROOT/'data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
 train.TRAIN_IMG_DIR=ROOT/'data/raw/train/train_images';train.TEST_IMG_DIR=ROOT/'data/raw/test/test_images'
 train.official_counts=aligned_counts
 sentinel=object();old=sys.modules.get('train',sentinel)
 try:
  sys.modules['train']=train
  evaluator=module('_team_m2_evaluator',ROOT/'team/m2/source/test.py')
 finally:
  if old is sentinel:sys.modules.pop('train',None)
  else:sys.modules['train']=old
 return train,evaluator
