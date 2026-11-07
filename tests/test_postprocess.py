import numpy as np
from m3_filament.postprocess import transform_record
from m3_filament.schema import make_instance,make_record,validate_record
from m3_filament.masks import decode


def fixture():
    source={'run_id':'original','checkpoint_sha256':'a','training_run_id':'training','inference_config':{'conf':.001}}
    a=np.zeros((12,12),np.uint8);a[2:8,2:8]=1;a[4,4]=0
    b=np.zeros((12,12),np.uint8);b[10:12,10:12]=1
    r=make_record('obs',[12,12],[make_instance(a,1,.8,source),make_instance(b,2,.2,source)],source)
    return r,a


def test_area_filter_preserves_retained_native_mask_and_provenance():
    r,a=fixture();out=transform_record(r,'filtered',min_area=5)
    assert len(out['instances'])==1
    assert np.array_equal(decode(out['instances'][0]['segmentation']),a)
    assert out['source']['checkpoint_sha256']=='a'
    assert out['source']['inference_config']['postprocess']['min_area']==5
    assert len(r['instances'])==2
    validate_record(out,exclusive=True)


def test_closing_changes_prediction_only_and_retains_exclusivity():
    r,a=fixture();out=transform_record(r,'closed',closing_radius=1)
    assert decode(out['instances'][0]['segmentation'])[4,4]==1
    assert np.array_equal(decode(r['instances'][0]['segmentation']),a)
    validate_record(out,exclusive=True)
