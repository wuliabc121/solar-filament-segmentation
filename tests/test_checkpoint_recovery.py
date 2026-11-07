import importlib.util
import json
from pathlib import Path
import torch
import pytest


def module():
    spec=importlib.util.spec_from_file_location('recover_budget_checkpoints',Path('scripts/recover_budget_checkpoints.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_running_training_is_never_finalized(tmp_path):
    assert module().recover(tmp_path,{'status':'running'})['status']=='still_running'
    assert not (tmp_path/'metrics.json').exists()


def test_recovery_uses_saved_updates_and_marks_lower_bound(tmp_path):
    m=module();weights=tmp_path/'weights';weights.mkdir()
    (tmp_path/'run_manifest_started.json').write_text(json.dumps({'run_id':'sample','training_observation_ids':['train']}))
    ckpt={'epoch':2,'updates':360,'train_metrics':{'fitness':.2}}
    torch.save(ckpt,weights/'best.pt');torch.save(ckpt,weights/'last.pt')
    result=m.recover(tmp_path,{'status':'interrupted','duration_seconds':600,'id':'EXP-test'})
    metrics=json.loads((tmp_path/'metrics.json').read_text())
    assert result['status']=='recovered_for_inference'
    assert metrics['optimizer_updates']==360 and metrics['actual_epochs']==3
    assert metrics['optimizer_updates_scope']=='last saved checkpoint lower bound; unsaved updates unknown'
    assert json.loads((tmp_path/'run_manifest.json').read_text())['status']=='interrupted_checkpoint_recovered'


def test_stripped_checkpoint_uses_csv_epochs_without_inventing_updates(tmp_path):
    m=module();weights=tmp_path/'weights';weights.mkdir()
    (tmp_path/'run_manifest_started.json').write_text(json.dumps({'run_id':'sample'}))
    torch.save({'epoch':-1,'updates':None,'train_metrics':{}},weights/'best.pt')
    (tmp_path/'results.csv').write_text('epoch,time\n8,1700\n')
    assert m.recover(tmp_path,{'status':'interrupted','duration_seconds':1800,'id':'EXP-test'})['status']=='recovered_for_inference'
    metrics=json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['actual_epochs']==8 and metrics['optimizer_updates'] is None
    assert 'unavailable' in metrics['optimizer_updates_scope']


def test_interrupted_metadata_publication_is_retryable(tmp_path,monkeypatch):
    m=module();weights=tmp_path/'weights';weights.mkdir()
    (tmp_path/'run_manifest_started.json').write_text(json.dumps({'run_id':'sample'}))
    torch.save({'epoch':2,'updates':100,'train_metrics':{}},weights/'best.pt')
    original=m.atomic_json
    def interrupted(path,value):
        if Path(path).name=='metrics.json':raise OSError('simulated interruption')
        return original(path,value)
    monkeypatch.setattr(m,'atomic_json',interrupted)
    record={'status':'interrupted','duration_seconds':100,'id':'EXP-test'}
    with pytest.raises(OSError):m.recover(tmp_path,record)
    assert not (tmp_path/'metrics.json').exists()
    monkeypatch.setattr(m,'atomic_json',original)
    assert m.recover(tmp_path,record)['status']=='recovered_for_inference'
