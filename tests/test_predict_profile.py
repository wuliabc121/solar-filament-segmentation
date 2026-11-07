import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from m3_filament.schema import make_instance,make_record,validate_record
from m3_filament.io import write_jsonl,read_jsonl


def module():
    spec=importlib.util.spec_from_file_location('predict_joint_profile',Path('scripts/predict_joint_profile.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_weight_hash_mismatch_stops_before_inference(tmp_path,monkeypatch):
    m=module();weights=tmp_path/'model.pt';weights.write_bytes(b'weights')
    profile=tmp_path/'profile.json';profile.write_text(json.dumps({'weights':str(weights),'checkpoint_sha256':'wrong'}))
    monkeypatch.setattr(m,'predict',lambda *a,**k:pytest.fail('inference should not run'))
    with pytest.raises(ValueError,match='hash'):
        m.run_profile(profile,tmp_path/'images',tmp_path/'out')


def test_profile_applies_mask_filter_without_ground_truth(tmp_path,monkeypatch):
    m=module();weights=tmp_path/'model.pt';weights.write_bytes(b'weights')
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    profile=tmp_path/'profile.json';profile.write_text(json.dumps({'weights':str(weights),'checkpoint_sha256':digest,
        'imgsz':1536,'conf':.25,'min_area':5,'closing_radius':0,'device':'cpu','max_det':100,'score_floor':.05,
        'source_manifest_sha256':m.source_manifest_hash(),'requirements_lock_sha256':m.sha256(m.ROOT/'requirements-lock.txt'),
        'inference_script_sha256':m.sha256(m.Path(m.__file__)),'runtime_versions':m.runtime_versions()}))
    def fake_predict(weights,images,output,**kwargs):
        assert kwargs['conf']==.05 and kwargs['require_unseen']
        source={'run_id':'raw','checkpoint_sha256':digest,'inference_config':{'conf':.05}}
        a=np.zeros((12,12),np.uint8);a[2:8,2:8]=1
        b=np.zeros_like(a);b[10:12,10:12]=1
        record=make_record('obs',[12,12],[make_instance(a,1,.8,source),make_instance(b,2,.3,source)],source)
        write_jsonl(Path(output)/'instances.jsonl',[record])
    monkeypatch.setattr(m,'predict',fake_predict)
    result=m.run_profile(profile,tmp_path/'images',tmp_path/'out')
    records=list(read_jsonl(tmp_path/'out/instances.jsonl'))
    assert len(records[0]['instances'])==1 and result['images']==1
    assert records[0]['source']['inference_config']['postprocess']['min_area']==5
    validate_record(records[0],exclusive=True)


def test_changed_source_is_rejected_before_inference(tmp_path,monkeypatch):
    m=module();weights=tmp_path/'model.pt';weights.write_bytes(b'weights')
    profile=tmp_path/'profile.json';profile.write_text(json.dumps({'weights':str(weights),
        'checkpoint_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'source_manifest_sha256':'old-code'}))
    monkeypatch.setattr(m,'predict',lambda *a,**k:pytest.fail('stale profile must not run'))
    with pytest.raises(ValueError,match='source'):
        m.run_profile(profile,tmp_path/'images',tmp_path/'out')
