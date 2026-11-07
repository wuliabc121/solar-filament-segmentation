import sys
from pathlib import Path


def test_absolute_group_interval_for_identical_groups_is_exact():
 sys.path.insert(0,str(Path('scripts').resolve()))
 from summarize_structure import candidate_intervals
 rows=[{'observation_id':name,'tp':1,'fp':0,'fn':0,'tp_iou_sum':.5,'n_gt':1,'n_pred':1,'small_gt':1,'small_matched':1} for name in ['a','b']]
 result=candidate_intervals(rows,{'a':'g1','b':'g2'},draws=100)
 assert result['independent_groups']==2 and result['intervals']['pq']==[.5,.5]
 assert result['intervals']['recall']==[1.,1.]
