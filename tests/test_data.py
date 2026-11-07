import json
import numpy as np
import pytest
from m3_filament.data import grouped_split, annotation_mask, polygon_roundtrip
from m3_filament.data import select_development_observations,validate_subset_selection


def test_same_observation_annotation_pass_and_date_do_not_leak():
    rows=[{'observation_id':f'obs{i}','date':f'2020-01-{i//2+1:02d}','sha256':str(i)} for i in range(20)]
    rows.append(dict(rows[0]))
    s=grouped_split(rows,42)
    bydate={}
    for r in rows:bydate.setdefault(r['date'],set()).add(s[r['observation_id']])
    assert all(len(v)==1 for v in bydate.values())
    assert set(s.values())=={'train','calibration','holdout'}
    assert s==grouped_split(list(reversed(rows)),42)


def test_duplicate_content_links_different_dates_into_one_group():
    rows=[{'observation_id':str(i),'date':str(i),'sha256':str(i)} for i in range(12)]
    rows[-1]['sha256']=rows[0]['sha256']
    s=grouped_split(rows,42)
    assert s['0']==s['11']


def test_single_annotation_polygon_is_kept_in_coco_raster_convention():
    ann={'segmentation':[[1,1,4,1,4,3,1,3]],'iscrowd':0}
    m=annotation_mask(ann,5,7)
    assert m.sum()==6
    audit=polygon_roundtrip(ann,5,7)
    assert audit['coco_iou']==1.0


def test_unrepresentable_multiple_polygons_are_blocked():
    a={'segmentation':[[0,0,2,0,2,2,0,2],[5,3,7,3,7,5,5,5]]}
    assert not polygon_roundtrip(a,6,8)['topology_supported']


def test_development_subset_is_order_independent_and_partition_safe():
    split={str(i):'train' if i<30 else 'calibration' if i<40 else 'holdout' for i in range(45)}
    selected=select_development_observations(split,8,42)
    assert selected==select_development_observations(dict(reversed(list(split.items()))),8,42)
    assert len([o for o in selected if split[o]=='train'])==8
    assert len([o for o in selected if split[o]=='calibration'])==2
    assert not any(split[o]=='holdout' for o in selected)


def test_explicit_subset_cannot_promote_calibration_or_holdout_to_training():
    assignments={'a':'train','b':'calibration','c':'holdout'}
    assert validate_subset_selection({'train':['a'],'calibration':['b']},assignments)=={'a','b'}
    for bad in [{'train':['b'],'calibration':[]},{'train':['a'],'calibration':['c']}]:
        with pytest.raises(ValueError):validate_subset_selection(bad,assignments)
