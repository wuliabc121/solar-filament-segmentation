"""One owned-process handoff after the registered 496-observation training ends."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path.cwd();OUT=ROOT/'reports/balanced_20261104'
state=json.loads((OUT/'state.json').read_text());assert state['status']=='running_balanced496_ft4'
old=state['pid'];deadline=dt.datetime.fromisoformat(state['deadline_utc'])
records=[]
for path in (ROOT/'.research-guardian/experiments').glob('*/record.json'):
    r=json.loads(path.read_text())
    if r.get('model')=='balanced496_ft4':
        assert r['command'][-2:]==['balanced496_ft4','--development'];records.append((r['started_at'],path,r))
_,path,training=max(records)
progress={'old_paused_driver_pid':old,'handoff_pid':os.getpid(),'training_record':str(path),
          'deadline_utc':state['deadline_utc']}
(OUT/'queue_before_strict_reload.json').write_text(json.dumps(state,indent=2)+'\n')
def save(status):
    (OUT/'strict_handoff.json').write_text(json.dumps(progress|{'status':status,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2)+'\n')
save('waiting_owned_training')
while True:
    current=json.loads(path.read_text())
    if current['status']!='running':break
    if dt.datetime.now(dt.timezone.utc)>=deadline:
        try:os.kill(training['wrapper_pid'],signal.SIGTERM)
        except ProcessLookupError:pass
        save('deadline_stop_sent_to_runner')
    time.sleep(1)
# Give the runner time to exit and the checkpoint watcher time to recover any
# interrupted run; neither phase starts training or accesses evaluation labels.
for _ in range(12):
    if (ROOT/'runs/balanced496_ft4/metrics.json').exists():break
    time.sleep(1)
time.sleep(1)
try:
    os.kill(old,signal.SIGTERM);os.kill(old,signal.SIGCONT)
except ProcessLookupError:pass
save('waiting_old_driver_lock_release')
with (OUT/'driver.lock').open('a') as lock:
    for _ in range(100):
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);break
        except BlockingIOError:time.sleep(.1)
    else:raise RuntimeError('old driver did not release lock; refusing duplicate')
save('reloading_strict_driver')
os.execv(sys.executable,[sys.executable,str(ROOT/'scripts/balanced_driver.py')])
