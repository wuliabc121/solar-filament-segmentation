import importlib.util
from pathlib import Path
import pytest
from m3_filament.io import sha256


def load():
 spec=importlib.util.spec_from_file_location('close_structure',Path('scripts/close_structure.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_failed_missing_checkpoint_is_retained_without_hashing(tmp_path):
 path=tmp_path/'never-created.pt';m=load()
 result=m.verify_checkpoint_evidence([{'id':'failed-smoke','status':'failure','checkpoints':[{'path':str(path),'missing':True}]}])
 assert len(result)==1 and result[0]['experiment_id']=='failed-smoke' and not path.exists()


def test_successful_checkpoint_must_have_matching_bytes(tmp_path):
 path=tmp_path/'best.pt';path.write_bytes(b'checkpoint-test');m=load()
 assert m.verify_checkpoint_evidence([{'id':'success','status':'success','checkpoints':[{'path':str(path),'sha256':sha256(path)}]}])==[]
 with pytest.raises(AssertionError):m.verify_checkpoint_evidence([{'id':'bad','status':'success','checkpoints':[{'path':str(tmp_path/'missing.pt'),'missing':True}]}])
