import hashlib,json
from pathlib import Path
import pytest
from team_pipeline.integrity import write_import_manifest,verified_checkpoint

def test_manifest_never_hashes_itself_and_verifies(tmp_path):
 team=tmp_path/'team';team.mkdir();(team/'import_manifest.json').write_text('old');(team/'a.py').write_text('hello')
 target=write_import_manifest(tmp_path)
 files=json.loads(target.read_text());assert 'team/import_manifest.json' not in files
 for k,v in files.items():assert hashlib.sha256((tmp_path/k).read_bytes()).hexdigest()==v
 assert files['team/a.py']==hashlib.sha256(b'hello').hexdigest()

def test_checkpoint_hash_checked_before_any_model_load(tmp_path):
 p=tmp_path/'checkpoint.pt';p.write_bytes(b'original');info={'checkpoint':p.name,'checkpoint_sha256':hashlib.sha256(b'original').hexdigest()}
 path,digest=verified_checkpoint(info,tmp_path);assert path==p and digest==info['checkpoint_sha256']
 p.write_bytes(b'changed')
 with pytest.raises(ValueError,match='checkpoint hash mismatch'):verified_checkpoint(info,tmp_path)

def test_final_import_manifest_matches_every_file():
 from team_pipeline import ROOT
 m=json.loads((ROOT/'team/import_manifest.json').read_text())
 assert 'team/import_manifest.json' not in m
 for k,v in m.items():assert hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v,k
