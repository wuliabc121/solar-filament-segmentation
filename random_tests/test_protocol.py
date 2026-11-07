from pathlib import Path
import pytest
from random_validation.protocol import make_split,overlap_report

def images():return [{'id':f'{a}-img{i}','file_name':f'img{i}.jpeg'} for i in range(10) for a in range(2)]

def test_random_image_split_is_stable_and_keeps_all_annotation_records():
 a=make_split(images(),.2,42);b=make_split(list(reversed(images())),.2,42)
 assert a['validation_ids']==b['validation_ids'] and len(a['validation_ids'])==2 and len(a['train_ids'])==8
 assert set(a['validation_ids']).isdisjoint(a['train_ids'])
 assert sum(len(a['annotation_records'][s]) for s in a['validation_ids'])==4
 assert make_split(images(),.2,43)['validation_ids']!=a['validation_ids']
 for ratio in (0,1,-.2,float('nan')):
  with pytest.raises(ValueError):make_split(images(),ratio,42)

def test_training_overlap_is_disclosed_not_filtered():
 data={'a':{'date':'1','sha256':'x'},'b':{'date':'1','sha256':'y'},'c':{'date':'2','sha256':'z'}}
 models={'m1':{'training_observations':['a'],'training_membership_status':'reconstructed_not_attested'}}
 result=overlap_report(['a','b','c'],models,data)['m1']
 assert result['exact_training_overlap_count']==1 and result['date_or_content_training_overlap_count']==2
 assert result['scored_observations']==3 and result['training_membership_status']=='reconstructed_not_attested'
