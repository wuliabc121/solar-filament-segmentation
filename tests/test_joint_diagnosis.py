import importlib.util
from pathlib import Path
import numpy as np
from m3_filament.masks import encode


def module():
    spec=importlib.util.spec_from_file_location('diagnose_joint',Path('scripts/diagnose_joint.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_diagnosis_separates_unmatched_objects_and_pixel_union():
    gt=np.zeros((12,12),np.uint8);gt[1:6,1:6]=1
    fp=np.zeros_like(gt);fp[9:11,9:11]=1
    result=module().score_masks([encode(gt)],[encode(gt),encode(fp)])
    assert result['gt_matched']==1 and result['fn_zero_overlap']==0
    assert result['fp_zero_overlap']==1
    assert np.isclose(result['foreground_iou'],25/29)
    assert np.isclose(result['foreground_dice'],50/54)
