import importlib.util
import json
from pathlib import Path
import pytest


def driver():
    spec=importlib.util.spec_from_file_location('balanced_driver',Path('scripts/balanced_driver.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_completed_reference_checks_survive_candidate_failure(monkeypatch):
    module=driver();monkeypatch.setattr(module,'export_check',lambda:'test-dataset')
    def scoring(run,size,output,*args):
        if output=='joint_check_candidate':raise RuntimeError('inference interrupted')
        return {'selected':{'result':output}}
    monkeypatch.setattr(module,'score_detector',scoring)
    checks={};candidate={'run':'model','imgsz':1024,'confidence':.2,'min_area':0,'closing_radius':0}
    with pytest.raises(RuntimeError,match='interrupted'):
        module.frozen_check(candidate,checks)
    assert set(checks)=={'old_baseline','previous_recall_candidate'}


def test_restart_preserves_only_completed_scoring_time(tmp_path,monkeypatch):
    module=driver();monkeypatch.setattr(module,'OUT',tmp_path);jobs=tmp_path/'jobs';jobs.mkdir()
    for i,row in enumerate([{'kind':'score','elapsed_seconds':20},{'kind':'score','elapsed_seconds':30},
                             {'kind':'training','elapsed_seconds':200},{'kind':'score'}]):
        (jobs/(str(i)+'.json')).write_text(json.dumps(row))
    restored=module.restore_score_accounting()
    assert restored['score_seconds']==50
    assert len(restored['jobs'])==3


def test_user_selection_prefers_strict_improvement_over_higher_pq_with_tied_recall():
    module=driver()
    base={'tp':10,'fp':5,'fn':10,'n_gt':20,'n_pred_across_annotators':15,
          'small_gt':10,'small_matched':2,'pq':.3,'sq':.6,'rq':.5}
    tied=base|{'name':'pq_only','fp':3,'n_pred_across_annotators':13,'pq':.6,'sq':.8,'rq':.7}
    better=base|{'name':'strict','tp':12,'fp':2,'fn':8,'n_pred_across_annotators':14,
                   'small_matched':3,'pq':.4,'sq':.65,'rq':.6}
    choice=module.choose_for_user([tied,better],[base])
    assert choice['joint_gate_passed'] and choice['candidate']['name']=='strict'
    failed=module.choose_for_user([tied],[base])
    assert not failed['joint_gate_passed'] and 'recall' in failed['violations']
    below_target=better|{'pq':.34}
    failed=module.choose_for_user([below_target],[base])
    assert not failed['joint_gate_passed'] and 'pq' in failed['violations']
