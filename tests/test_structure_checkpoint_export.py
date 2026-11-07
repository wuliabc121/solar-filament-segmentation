import importlib.util
from pathlib import Path
import subprocess
import sys
import torch
from m3_filament.io import sha256


class TrainingOnlyCriterion:
 pass


def load():
 spec=importlib.util.spec_from_file_location('export_structure_checkpoint',Path('scripts/export_structure_checkpoint.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_export_keeps_original_and_exact_ema_parameters_without_custom_loss(tmp_path):
 m=load();model=torch.nn.Linear(3,2).half();model.criterion=TrainingOnlyCriterion()
 source=tmp_path/'raw.pt';destination=tmp_path/'inference.pt'
 torch.save({'model':None,'ema':model,'epoch':2,'updates':27,'train_args':{},'optimizer':None},source)
 before=sha256(source)
 raw_load=subprocess.run([sys.executable,'-c','import torch,sys; torch.load(sys.argv[1],map_location="cpu",weights_only=False)',str(source)],cwd=str(tmp_path),capture_output=True)
 assert raw_load.returncode!=0 and b'TrainingOnlyCriterion' in raw_load.stderr or raw_load.returncode!=0 and b'test_structure_checkpoint_export' in raw_load.stderr
 evidence=m.export_one(source,destination)
 assert sha256(source)==before and evidence['ema_parameters_identical']
 exported=torch.load(destination,map_location='cpu',weights_only=False)
 assert exported['model'].criterion is None and evidence['original_updates']==27 and evidence['original_epoch_zero_index']==2
 subprocess.run([sys.executable,'-c','import torch,sys; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert x["model"].criterion is None',str(destination)],cwd=str(tmp_path),check=True,capture_output=True)
