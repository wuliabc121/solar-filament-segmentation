import importlib.util
import json
from pathlib import Path
import pytest


def driver():
    spec=importlib.util.spec_from_file_location('resume_driver',Path('scripts/resume_driver.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_report_replaces_stage_preview_and_keeps_failure(tmp_path):
    module=driver();module.REPORT=tmp_path
    module.report({},None,{})
    module.report({},None,{},'interrupted after preview')
    assert json.loads((tmp_path/'results.json').read_text())['error']=='interrupted after preview'
    assert 'interrupted after preview' in (tmp_path/'report.md').read_text()


def test_resume_rejects_metrics_from_modified_training_config(tmp_path):
    module=driver()
    config=tmp_path/'config.yaml';config.write_text('epochs: 6')
    weights=tmp_path/'weights';weights.mkdir();(weights/'best.pt').write_bytes(b'checkpoint')
    (tmp_path/'metrics.json').write_text(json.dumps({'checkpoint_sha256':module.sha256(weights/'best.pt')}))
    (tmp_path/'run_manifest.json').write_text(json.dumps({'config_file_sha256':module.sha256(config)}))
    module.validate_reuse([],str(tmp_path/'metrics.json'),str(config))
    config.write_text('epochs: 12')
    with pytest.raises(RuntimeError,match='provenance'):
        module.validate_reuse([],str(tmp_path/'metrics.json'),str(config))
