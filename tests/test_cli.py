import json
import numpy as np
import pytest
import subprocess
import sys
import os
from pathlib import Path
from m3_filament.cli import load_semantic


def test_semantic_cache_requires_matching_identity_and_provenance(tmp_path):
    p=tmp_path/'p.npz'
    np.savez(p,probabilities=np.zeros((3,7)),observation_id='wrong',run_id='cnn',checkpoint_hash='abc',split_hash='s')
    with pytest.raises(ValueError,match='observation'):load_semantic(p,'obs',(3,7))
    np.savez(p,probabilities=np.zeros((3,7)),observation_id='obs')
    with pytest.raises(ValueError,match='metadata'):load_semantic(p,'obs',(3,7))


def test_semantic_cache_is_original_coordinate_float_array(tmp_path):
    p=tmp_path/'p.npz'
    np.savez(p,probabilities=np.full((3,7),.4,dtype=np.float32),observation_id='obs',
             run_id='cnn',checkpoint_hash='abc',split_hash='s')
    array,meta=load_semantic(p,'obs',(3,7))
    assert array.shape==(3,7)
    assert meta['run_id']=='cnn'


def test_runner_creates_local_configuration_directories_before_import():
    project=Path(__file__).resolve().parents[1]
    p=subprocess.run([str(project/'run.sh'),'-c',
        'import os,pathlib; assert pathlib.Path(os.environ["YOLO_CONFIG_DIR"]).is_dir()'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_runner_does_not_inherit_other_project_python_paths(tmp_path):
    project=Path(__file__).resolve().parents[1]
    code='import sys; assert '+repr(str(tmp_path))+' not in sys.path'
    p=subprocess.run([str(project/'run.sh'),'-c',code],env={**os.environ,'PYTHONPATH':str(tmp_path)},capture_output=True,text=True)
    assert p.returncode==0,p.stderr
