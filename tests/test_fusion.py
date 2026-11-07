import csv
import numpy as np
import pytest

from m3_filament.masks import decode
from m3_filament.schema import make_instance, make_record
from m3_filament.fusion import FusionConfig, fuse_record
from m3_filament.submission import write_submission


def sample():
    a=np.zeros((5,9),np.uint8);a[2,1:3]=1
    b=np.zeros_like(a);b[2,4:6]=1
    return make_record('20110120105534Ch',[5,9],[
        make_instance(a,1,.9,{'run_id':'det'}), make_instance(b,2,.8,{'run_id':'det'})],{'run_id':'det'})


def test_dilation_cannot_steal_another_original_mask():
    r=fuse_record(sample(),np.ones((5,9)),np.ones((5,9)),FusionConfig(radius=3), 'fusion')
    m=[decode(x['segmentation']) for x in r['instances']]
    assert m[1][2,4] and not m[0][2,4]
    assert not (m[0]&m[1]).any()
    assert r['instances'][0]['source_instance_id']==1


def test_zero_radius_equals_semantic_intersection():
    p=np.zeros((5,9));p[2,1]=1
    r=fuse_record(sample(),p,p,FusionConfig(radius=0), 'fusion')
    assert len(r['instances'])==1
    assert decode(r['instances'][0]['segmentation']).sum()==1


def test_empty_instance_input_and_residual_switch():
    record=make_record('obs',[5,9],[],{'run_id':'det'})
    p=np.zeros((5,9));p[1:3,5:8]=.9
    assert not fuse_record(record,p,p,FusionConfig(), 'f')['instances']
    r=fuse_record(record,p,p,FusionConfig(residual=True,min_area=2),'f')
    assert len(r['instances'])==1
    assert r['instances'][0]['candidate_source']=='semantic_residual'
    assert r['instances'][0]['requires_refiner'] is True


def test_residual_endpoint_guard_does_not_create_adjacent_object():
    r=sample();p=np.zeros((5,9));p[2,1:4]=.9
    out=fuse_record(r,p,p,FusionConfig(residual=True,min_area=1,endpoint_guard=1),'f')
    assert len(out['instances'])==1


@pytest.mark.parametrize('p',[np.zeros((9,5)),np.full((5,9),np.nan),np.full((5,9),1.1)])
def test_bad_semantic_arrays_fail(p):
    with pytest.raises(ValueError):fuse_record(sample(),p,p,FusionConfig(),'f')


def test_fallback_is_explicit_and_keeps_original_id():
    p=np.zeros((5,9))
    out=fuse_record(sample(),p,p,FusionConfig(fallback=True,fallback_score=.85),'f')
    assert len(out['instances'])==1
    assert out['instances'][0]['fallback_used']
    assert out['instances'][0]['source_instance_id']==1


def test_submission_roundtrip_and_size_gate(tmp_path):
    p=tmp_path/'submission.csv'
    with pytest.raises(ValueError,match='2048'):write_submission([sample()],p)
    write_submission([sample()],p,expected_shape=(5,9))
    rows=list(csv.DictReader(p.open()))
    assert rows[0]['filament_id']=='20110120105534Ch_1'
    assert decode({'size':[5,9],'counts':rows[0]['segmentation_rle']}).sum()==2
    with pytest.raises(FileExistsError):write_submission([sample()],p,expected_shape=(5,9))


def test_minimum_coverage_measures_retained_original_pixels_not_dilation():
    m=np.zeros((5,9),np.uint8);m[2,3]=1
    record=make_record('obs',[5,9],[make_instance(m,1,.9,{'run_id':'d'})],{'run_id':'d'})
    p=np.ones((5,9));p[2,3]=0
    r=fuse_record(record,p,p,FusionConfig(radius=1,min_coverage=.5),'f')
    assert r['instances']==[]
    fallback=fuse_record(record,p,p,FusionConfig(radius=1,min_coverage=.5,fallback=True),'f')
    assert fallback['instances'][0]['fallback_used']
    assert np.array_equal(decode(fallback['instances'][0]['segmentation']),m)


def test_submission_normalizes_uncompressed_rle_to_compressed_counts(tmp_path):
    r=sample();r['instances'][0]['segmentation']['counts']=[7,1,4,1,32]
    path=tmp_path/'out.csv';write_submission([r],path,expected_shape=(5,9))
    row=next(csv.DictReader(path.open()))
    assert not row['segmentation_rle'].startswith('[')
    assert decode({'size':[5,9],'counts':row['segmentation_rle']}).sum()==2
