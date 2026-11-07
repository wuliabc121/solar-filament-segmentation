"""Native-mask warm start with checkpoint fitness aligned to mask mAP50-95."""
import argparse
import json
import math
from pathlib import Path
import shutil
import signal
from m3_filament.native import NativeSegmentationTrainer
from m3_filament.model import train_model
from m3_filament.io import sha256

TRACE=[]


def has_scoring_budget(state,limit=1800,reserve=300):
    return limit-state['score_seconds']-reserve>=300


def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def persist(folder):
    folder=Path(folder)
    if not folder.exists():return
    destination=folder/'code_snapshot/external/train_mask_metric.py';destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,destination)
    metadata={'checkpoint_selection_metric':'metrics/mAP50-95(M)',
              'external_training_script_sha256':sha256(__file__),
              'external_training_script_snapshot':str(destination.resolve()),
              'loss_changed':False,'base_source_manifest_scope':'src only; this external trainer is recorded separately'}
    atomic(folder/'external_training.json',metadata)
    atomic(folder/'selection_trace.json',TRACE)
    atomic(folder/'code_snapshot/external_source_manifest.json',{'external/train_mask_metric.py':sha256(destination)})
    for name in ('run_manifest_started.json','run_manifest.json','metrics.json'):
        path=folder/name
        if path.exists():
            obj=json.loads(path.read_text());obj['external_training']=metadata;atomic(path,obj)


class MaskMetricTrainer(NativeSegmentationTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        persist(self.save_dir)

    def validate(self):
        previous=self.best_fitness
        metrics,combined=super().validate()
        if metrics is None:return metrics,combined
        mask=float(metrics['metrics/mAP50-95(M)'])
        if not math.isfinite(mask) or not 0<=mask<=1:raise ValueError('invalid mask checkpoint fitness')
        self.best_fitness=max(previous if previous is not None else -math.inf,mask)
        TRACE.append({'epoch':int(self.epoch)+1,'mask_fitness':mask,'original_combined_fitness':float(combined),
                      'best_mask_fitness':self.best_fitness})
        return metrics,mask


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args()
    if a.run_id=='balanced_maskmetric_ft4':
        state=json.loads(Path('reports/balanced_20261104/state.json').read_text())
        if not has_scoring_budget(state):
            print('SKIPPED before model initialization: insufficient protected checkpoint scoring budget',flush=True)
            raise SystemExit(75)
    import m3_filament.native as native
    native.NativeSegmentationTrainer=MaskMetricTrainer
    def stop(signum,frame):raise KeyboardInterrupt('owned bounded training signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    try:
        result=train_model(a.config,a.run_id,False,development=True)
        print(json.dumps(result,indent=2),flush=True)
    finally:persist(Path('runs')/a.run_id)


if __name__=='__main__':main()
