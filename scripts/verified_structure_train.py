"""Audit the actual per-batch criterion without changing numerical training."""
import json
import shutil
import sys
from pathlib import Path
import structure_training as training
import structure_method_v2
training.differentiable_crop=structure_method_v2.coverage_crop
from m3_filament.io import sha256

original_init=training.FocusedTrainer.__init__


def audited_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self._criterion_logged=False
    def verify(trainer):
        if trainer._criterion_logged:return
        criterion=trainer.model.criterion
        assert isinstance(criterion,training.StructureLoss),'structure experiment did not activate StructureLoss'
        folder=Path(trainer.save_dir)
        (folder/'criterion_actual.json').write_text(json.dumps({'actual_criterion':type(criterion).__module__+'.'+type(criterion).__name__,
          'epoch':int(trainer.epoch)+1,'verified_after_backward':True,'entry_script_sha256':sha256(__file__)},indent=2)+'\n')
        trainer._criterion_logged=True
    self.add_callback('on_train_batch_end',verify)


if __name__=='__main__':
    training.FocusedTrainer.__init__=audited_init
    run=sys.argv[sys.argv.index('--run-id')+1]
    try:training.main()
    finally:
        folder=Path('runs')/run
        if folder.exists():
            destination=folder/'code_snapshot/external/verified_structure_train.py';destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(__file__,destination)
            support=folder/'code_snapshot/external/structure_method_v2.py'
            shutil.copy2(structure_method_v2.__file__,support)
            p=folder/'code_snapshot/external_source_manifest.json'
            data=json.loads(p.read_text()) if p.exists() else {};data['external/verified_structure_train.py']=sha256(destination);data['external/structure_method_v2.py']=sha256(support);p.write_text(json.dumps(data,indent=2)+'\n')
