"""Replace only an idle-waiting controller after its explicitly owned method ends."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from m3_filament.io import sha256
from m3_filament.model import source_manifest_hash

OUT=Path('reports/structure_20261104');OLD=87487
RECORD=Path('.research-guardian/experiments/EXP-20261104T055130-b6e51ad74e4b/record.json')
DEADLINE=dt.datetime.fromisoformat('2026-11-04T07:22:00+00:00')


def atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
 state=json.loads((OUT/'state.json').read_text());assert state['pid']==OLD and state['status']=='running_structure_dicecl_e10'
 atomic(OUT/'method_handoff.json',{'status':'waiting_owned_method','pid':os.getpid(),'old_parent':OLD,'record':str(RECORD),'deadline_utc':DEADLINE.isoformat()})
 os.kill(OLD,signal.SIGSTOP)
 try:
  while dt.datetime.now(dt.timezone.utc)<DEADLINE:
   record=json.loads(RECORD.read_text())
   if record['status']!='running':break
   time.sleep(1)
  else:raise TimeoutError('handoff deadline; original controller resumed for cleanup')
  spec={'command':record['command'],'source_sha256':source_manifest_hash(),'external_training_sha256':sha256('scripts/structure_training.py'),
        'config_sha256':sha256('configs/structure_dicecl_e10.yaml')}
  metrics='runs/structure_dicecl_e10/metrics.json'
  if record['status']=='success':
   assert Path(metrics).exists() and sha256(metrics)==record['metrics_evidence']['sha256']
  atomic(OUT/'jobs/structure_dicecl_e10.json',{'name':'structure_dicecl_e10','spec':spec,'metrics_path':metrics,
   'command':record['command'],'started_utc':record['started_at'],'ended_utc':record['ended_at'],
   'pid':record['wrapper_pid'],'returncode':record['exit_code'],'timed_out':record['status']=='timeout',
   'elapsed_seconds':record['duration_seconds'],'elapsed_seconds_source':'original actual experiment runner wall clock',
   'controller_metadata_restored_after_successful_child':record['status']=='success','experiment_record':str(RECORD),'original_runner_status':record['status']})
  os.kill(OLD,signal.SIGTERM);os.kill(OLD,signal.SIGCONT)
  lock=open(OUT/'driver.lock','a')
  for _ in range(100):
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);break
   except BlockingIOError:time.sleep(.1)
  else:raise RuntimeError('old parent did not release lock')
  lock.close()
  atomic(OUT/'method_handoff.json',{'status':'completed','pid':os.getpid(),'old_parent':OLD,'record':str(RECORD),'original_runner_status':record['status'],'handoff_interrupted_training':False,'new_driver':'scripts/structure_driver.py'})
  os.execv(str(Path('run.sh').resolve()),['./run.sh','scripts/structure_driver.py'])
 except BaseException:
  try:os.kill(OLD,signal.SIGCONT)
  except ProcessLookupError:pass
  raise


if __name__=='__main__':main()
