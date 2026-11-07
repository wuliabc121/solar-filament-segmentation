import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    spec=importlib.util.spec_from_file_location('summarize_balanced',Path('scripts/summarize_balanced.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def rows(tp,fp,fn,iou,small):
    return [{'observation_id':o,'annotation_id':'ann-'+o,'tp':tp,'fp':fp,'fn':fn,'tp_iou_sum':iou,
             'n_gt':2,'n_pred':2,'small_gt':1,'small_matched':small} for o in ['a','b','c']]


def test_bootstrap_pairs_complete_date_content_groups():
    result=module().bootstrap_comparison(rows(1,1,1,.6,0),rows(2,0,0,1.4,1),{'a':'g1','b':'g1','c':'g2'},draws=200)
    assert result['independent_groups']==2
    assert np.allclose(result['intervals']['pq'],[.4,.4])
    assert np.allclose(result['intervals']['small_recall'],[1.,1.])


def test_bootstrap_rejects_unpaired_annotation_passes():
    b=rows(1,1,1,.6,0);c=rows(2,0,0,1.4,1);c[0]['annotation_id']='different'
    with pytest.raises(ValueError,match='annotation'):
        module().bootstrap_comparison(b,c,{'a':'a','b':'b','c':'c'},draws=10)


def test_profile_detector_floor_matches_actual_final_evaluation():
    m=module()
    assert m.pipeline_score_floor({'confidence':.25,'min_area':0,'closing_radius':0})==.25
    assert m.pipeline_score_floor({'confidence':.25,'min_area':32,'closing_radius':0})==.05
