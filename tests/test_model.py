import pytest
from m3_filament.model import validate_training_request, load_training_provenance,mask_priority_indices
import numpy as np
from m3_filament.io import write_json,sha256


def info(**kw):
    return {'production_training_allowed':False,'split_status':'provisional_development_only',
            'status':'smoke_subset','samples':{'train':8,'calibration':2},**kw}


def test_production_training_refuses_provisional_or_lossy_data():
    with pytest.raises(ValueError,match='production'):validate_training_request({'epochs':50,'imgsz':1024},info(),False)


def test_smoke_budget_cannot_be_expanded_accidentally():
    with pytest.raises(ValueError,match='smoke'):validate_training_request({'epochs':50,'imgsz':256},info(),True)
    with pytest.raises(ValueError,match='smoke'):validate_training_request({'epochs':1,'imgsz':1024},info(),True)
    assert validate_training_request({'epochs':1,'imgsz':256},info(),True) is None


def test_replaced_checkpoint_cannot_inherit_training_lineage(tmp_path):
    path=tmp_path/'weights/best.pt';path.parent.mkdir();path.write_bytes(b'new weight')
    write_json(tmp_path/'run_manifest.json',{'checkpoint_sha256':'different','training_observation_ids':['obs']})
    with pytest.raises(ValueError,match='checkpoint'):load_training_provenance(path)


def test_matching_checkpoint_can_use_verified_manifest(tmp_path):
    path=tmp_path/'weights/best.pt';path.parent.mkdir();path.write_bytes(b'weight')
    write_json(tmp_path/'run_manifest.json',{'checkpoint_sha256':sha256(path),'training_observation_ids':['obs']})
    assert load_training_provenance(path)['training_observation_ids']==['obs']


def test_development_mode_is_explicit_and_time_bounded():
    cfg={'epochs':8,'imgsz':512,'time':.25}
    assert validate_training_request(cfg,info(),False,development=True) is None
    with pytest.raises(ValueError,match='production'):validate_training_request(cfg,info(),False)
    with pytest.raises(ValueError,match='development'):validate_training_request({**cfg,'time':1.},info(),False,development=True)
    with pytest.raises(ValueError,match='development'):validate_training_request({'epochs':8,'imgsz':512},info(),False,development=True)


def test_mask_dedup_uses_actual_overlap_and_stable_confidence_order():
    a=np.zeros((9,17),np.uint8);a[1:5,1:8]=1
    b=np.zeros_like(a);b[4:8,8:15]=1
    masks=np.stack([a,a,b])
    assert mask_priority_indices(masks,np.array([.7,.9,.8]))==[1,2]
    assert mask_priority_indices(masks,np.array([.9,.9,.8]))==[0,2]


def test_mask_dedup_does_not_merge_fragments_with_overlapping_boxes():
    a=np.zeros((9,17),np.uint8);a[1,1]=1;a[7,15]=1
    b=np.zeros_like(a);b[1,15]=1;b[7,1]=1
    assert mask_priority_indices(np.stack([a,b]),np.array([.9,.8]))==[0,1]


def test_overnight_budget_requires_explicit_profile_and_live_deadline():
    import datetime
    live=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=2)).isoformat()
    cfg={'epochs':24,'imgsz':1024,'time':2.,'budget_profile':'overnight','deadline_utc':live}
    large=info(samples={'train':800,'calibration':40})
    assert validate_training_request(cfg,large,False,True) is None
    with pytest.raises(ValueError):validate_training_request({**cfg,'budget_profile':'short'},large,False,True)
    with pytest.raises(ValueError):validate_training_request({**cfg,'deadline_utc':'2020-01-01T00:00:00Z'},large,False,True)
