"""Recover inference provenance after a bounded training interruption, without retraining."""
import datetime as dt
import csv
import json
from pathlib import Path
import time
import os
from m3_filament.io import sha256


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def recover(folder,record):
    folder=Path(folder)
    if record['status']=='running':return {'status':'still_running'}
    if (folder/'metrics.json').exists():return {'status':'completed_metrics_present'}
    import torch
    valid={};errors={}
    for path in (folder/'weights').glob('*.pt'):
        try:valid[path.name]=(torch.load(path,map_location='cpu',weights_only=False),sha256(path))
        except Exception as exc:errors[path.name]=repr(exc)
    if 'best.pt' not in valid:return {'status':'no_valid_best_checkpoint','errors':errors}
    best,digest=valid['best.pt'];last=valid.get('last.pt',valid['best.pt'])[0]
    elapsed=record.get('duration_seconds')
    if type(elapsed) not in (int,float):return {'status':'missing_timer_evidence'}
    epoch=last.get('epoch');updates=last.get('updates')
    epoch_scope='last saved checkpoint lower bound; unsaved batches unknown'
    updates_scope='last saved checkpoint lower bound; unsaved updates unknown'
    if type(epoch) is not int or epoch<0:
        csv_path=folder/'results.csv'
        rows=list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
        if not rows:return {'status':'missing_saved_epoch_evidence'}
        epoch=int(float(rows[-1]['epoch']))-1
        epoch_scope='completed epochs recorded in results.csv; checkpoint epoch stripped'
        updates=None;updates_scope='unavailable after optimizer strip; not estimated from epoch count'
    original=folder/'run_manifest.json'
    manifest=json.loads((original if original.exists() else folder/'run_manifest_started.json').read_text())
    if not original.exists():
        manifest.update(status='interrupted_checkpoint_recovered',checkpoint_sha256=digest,
            checkpoint_hashes={name:data[1] for name,data in valid.items()},elapsed_seconds=elapsed,
            recovery_record_id=record['id'])
        atomic_json(original,manifest)
    else:
        assert manifest['checkpoint_sha256']==digest
    metrics={'run_id':manifest['run_id'],'purpose':'interrupted development checkpoint, inference evaluation only',
       'training_completed':False,'recovered_for_inference':True,'actual_epochs':epoch+1,
       'actual_epochs_scope':epoch_scope,
       'optimizer_updates':updates,'optimizer_updates_scope':updates_scope,
       'elapsed_seconds':elapsed,'elapsed_seconds_source':'experiment runner wall clock, not training model timer',
       'checkpoint_sha256':digest,'ultralytics_metrics':best.get('train_metrics',{}),
       'validation_checkpoint':'best.pt metadata, not a new validation pass','official_pq':None,
       'recovery_record_id':record['id']}
    note={'status':'recovered_for_inference','original_runner_status':record['status'],
          'runner_record_id':record['id'],'checkpoint_sha256':digest,'read_errors':errors,
          'note':'Original failed/interrupted experiment remains failed; no new training or score inferred.'}
    atomic_json(folder/'recovery.json',note)
    atomic_json(folder/'metrics.json',metrics)
    return note


def main():
    out=Path('reports/balanced_20261104');campaign=json.loads((out/'campaign.json').read_text())
    deadline=dt.datetime.fromisoformat(campaign['deadline_utc'])
    waiting={'balanced176_ft8':'balanced176-ft8','balanced496_ft4':'balanced496_ft4'};results={}
    while waiting and dt.datetime.now(dt.timezone.utc)<deadline:
        for run,name in list(waiting.items()):
            if Path('runs',run,'metrics.json').exists():results[run]={'status':'completed_metrics_present'};del waiting[run];continue
            matching=[]
            for path in Path('.research-guardian/experiments').glob('*/record.json'):
                record=json.loads(path.read_text())
                if record.get('model')==name:matching.append((record['started_at'],record))
            if matching:
                record=max(matching,key=lambda x:x[0])[1]
                if record['status']!='running':results[run]=recover(Path('runs')/run,record);del waiting[run]
        atomic_json(out/'recovery_watch.json',{'status':'waiting' if waiting else 'completed',
              'pid':__import__('os').getpid(),'waiting':list(waiting),'results':results,
              'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
        if waiting:time.sleep(1)
    if waiting:
        atomic_json(out/'recovery_watch.json',{'status':'deadline_reached','waiting':list(waiting),'results':results})


if __name__=='__main__':main()
