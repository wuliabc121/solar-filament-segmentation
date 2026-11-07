import importlib.util
from pathlib import Path
import torch


def module():
    spec=importlib.util.spec_from_file_location('train_mask_metric',Path('scripts/train_mask_metric.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_mask_selection_does_not_inherit_combined_fitness_best():
    m=module();trainer=m.MaskMetricTrainer.__new__(m.MaskMetricTrainer)
    trainer.ema=None;trainer.world_size=1;trainer.loss=torch.tensor(1.);trainer.epoch=0
    trainer.best_fitness=.1
    trainer.validator=lambda _: {'fitness':.9,'metrics/mAP50-95(M)':.05}
    _,fitness=trainer.validate()
    assert fitness==.05 and trainer.best_fitness==.1
    trainer.validator=lambda _: {'fitness':.8,'metrics/mAP50-95(M)':.2}
    _,fitness=trainer.validate()
    assert fitness==.2 and trainer.best_fitness==.2


def test_conditional_training_keeps_budget_to_evaluate_its_checkpoint():
    m=module()
    assert m.has_scoring_budget({'score_seconds':1000},limit=1800,reserve=300)
    assert not m.has_scoring_budget({'score_seconds':1450},limit=1800,reserve=300)
