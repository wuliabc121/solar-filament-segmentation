"""Non-destructive, numerically identical inference export after training has ended.

Interrupted raw EMA checkpoints may pickle the external training-only criterion.
Preserve those optimizer/epoch-bearing originals; strip into new *_inference.pt
files and register their hashes in the neighboring provenance manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from ultralytics.utils.torch_utils import strip_optimizer
from m3_filament.io import sha256


def parameter_digest(model):
 h=hashlib.sha256()
 for name,value in sorted(model.state_dict().items()):
  h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(list(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def export_one(source,destination):
 source,destination=Path(source),Path(destination)
 if destination.exists():raise FileExistsError(destination)
 original=sha256(source);raw=torch.load(source,map_location='cpu',weights_only=False)
 model=raw['ema'] if raw.get('ema') is not None else raw['model'];before=parameter_digest(model)
 result=strip_optimizer(source,s=str(destination))
 if not result or not destination.is_file():raise RuntimeError('inference checkpoint export failed')
 exported=torch.load(destination,map_location='cpu',weights_only=False);after=parameter_digest(exported['model'])
 if before!=after or sha256(source)!=original:raise RuntimeError('export changed EMA parameters or original file')
 assert getattr(exported['model'],'criterion',None) is None
 return {'original_path':str(source),'original_sha256':original,'inference_path':str(destination),'inference_sha256':sha256(destination),
 'ema_state_sha256':before,'ema_parameters_identical':True,'original_epoch_zero_index':raw.get('epoch'),'original_updates':raw.get('updates'),
 'training_only_criterion_removed':True,'original_optimizer_checkpoint_preserved':True}


def atomic(path,obj):
 path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 if not all(c.isalnum() or c in '-_' for c in a.run):raise ValueError('unsafe run id')
 folder=Path('runs')/a.run;manifest_path=folder/'run_manifest.json';manifest=json.loads(manifest_path.read_text())
 for record_path in Path('.research-guardian/experiments').glob('*/record.json'):
  record=json.loads(record_path.read_text())
  if record.get('model')==a.run and record['status']=='running':raise RuntimeError('never export while original training runs')
 output=folder/'inference_exports.json'
 if output.exists():raise FileExistsError(output)
 exports={}
 for name in ['best.pt','last.pt']:
  source=folder/'weights'/name
  if not source.is_file():continue
  destination=source.with_name(source.stem+'_inference.pt');exports[name]=export_one(source,destination)
  # A fresh process at project root must not need scripts/ or custom loss imports.
  environment=os.environ.copy();environment['PYTHONPATH']=str(Path('src').resolve());environment['PYTHONNOUSERSITE']='1'
  subprocess.run([sys.executable,'-c','from ultralytics import YOLO; import sys; m=YOLO(sys.argv[1]); assert getattr(m.model,"criterion",None) is None',str(destination.resolve())],cwd=str(Path.cwd()),env=environment,check=True,capture_output=True)
  exports[name]['normal_inference_process_load_verified']=True
  manifest.setdefault('checkpoint_hashes',{})[destination.name]=exports[name]['inference_sha256']
 if 'best.pt' not in exports:raise RuntimeError('no saved best checkpoint')
 result={'run':a.run,'exports':exports,'export_script_sha256':sha256(__file__),'no_training_or_validation_performed':True,
 'original_run_status':manifest.get('status'),'note':'Export does not turn interrupted training into success; original record/epochs/updates remain authoritative.'}
 manifest['inference_export_manifest']=str(output);atomic(manifest_path,manifest);atomic(output,result)
 print(json.dumps(result,indent=2))


if __name__=='__main__':main()
