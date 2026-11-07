"""Reload selection criteria after the one registered conditional training ends."""
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path.cwd();OUT=ROOT/'reports/balanced_20261104'
plan=json.loads((OUT/'pq_handoff_plan.json').read_text());old=plan['old_driver_pid']
record_path=ROOT/plan['job_record'];record=json.loads(record_path.read_text())
assert record['model']==plan['job_name']=='balanced_maskmetric_ft4'
deadline=dt.datetime.fromisoformat(plan['deadline_utc'])
progress={'old_paused_driver_pid':old,'handoff_pid':os.getpid(),'job_record':str(record_path),
          'deadline_utc':plan['deadline_utc']}
def save(status):
    target=OUT/'pq_handoff.json';temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(progress|{'status':status,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2)+'\n');temp.replace(target)
save('waiting_owned_mask_metric_training')
while True:
    current=json.loads(record_path.read_text())
    if current['status']!='running':break
    if dt.datetime.now(dt.timezone.utc)>=deadline:
        try:os.kill(record['wrapper_pid'],signal.SIGTERM)
        except ProcessLookupError:pass
    time.sleep(1)
folder=ROOT/'runs'/record['model']
if not (folder/'metrics.json').exists():
    spec=importlib.util.spec_from_file_location('owned_recovery',ROOT/'scripts/recover_budget_checkpoints.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    progress['recovery']=module.recover(folder,current)
time.sleep(1)
try:os.kill(old,signal.SIGTERM);os.kill(old,signal.SIGCONT)
except ProcessLookupError:pass
save('waiting_old_selection_lock')
with (OUT/'driver.lock').open('a') as lock:
    for _ in range(100):
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);break
        except BlockingIOError:time.sleep(.1)
    else:raise RuntimeError('old selection driver lock remains; no duplicate launched')
save('reloading_PQ035_and_joint_strict_driver')
os.execv(sys.executable,[sys.executable,str(ROOT/'scripts/balanced_driver.py')])
