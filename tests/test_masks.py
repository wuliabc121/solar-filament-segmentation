import numpy as np
import pytest

from m3_filament.masks import encode, decode, bbox
from m3_filament.schema import make_instance, validate_record, make_record


def test_asymmetric_mask_hole_and_fragments_survive_rle():
    mask = np.array([[1,1,1,0,0,0,1], [1,0,1,0,0,0,0], [1,1,1,0,1,0,0]], dtype=np.uint8)
    rle = encode(mask)
    assert rle['size'] == [3,7]
    assert isinstance(rle['counts'], str)
    assert np.array_equal(decode(rle), mask)
    assert bbox(mask) == [0,0,7,3]


def test_empty_mask_roundtrip():
    assert not decode(encode(np.zeros((3,7),np.uint8))).any()


@pytest.mark.parametrize('rle', [
    {'size':[3,7], 'counts':[20]},
    {'size':[3,7], 'counts':'invalid!!!'},
    {'size':[3,7], 'counts':[-1,22]},
])
def test_malformed_rle_is_rejected_before_native_decoder(rle):
    with pytest.raises(ValueError): decode(rle)


def test_schema_checks_bbox_identity_and_original_size():
    m=np.zeros((3,7),np.uint8); m[1,4:6]=1
    inst=make_instance(m, 1, .7, {'run_id':'test'})
    r=make_record('obs', [3,7], [inst], {'run_id':'test'})
    validate_record(r)
    assert inst['bbox']==[4,1,6,2]
    inst['bbox']=[1,4,2,6]
    with pytest.raises(ValueError,match='bbox'): validate_record(r)


def test_overlapping_final_instances_are_rejected():
    m=np.ones((3,7),np.uint8)
    r=make_record('obs',[3,7],[make_instance(m,1,.8,{'run_id':'x'}),make_instance(m,2,.7,{'run_id':'x'})],{'run_id':'x'})
    with pytest.raises(ValueError,match='overlap'): validate_record(r,exclusive=True)
