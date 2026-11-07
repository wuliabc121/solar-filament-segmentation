"""One-off queue handoff after the already registered crop training finishes."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import sys
import time

root=Path.cwd();out=root/'reports/resume_20261104'
state=json.loads((out/'state.json').read_text())
assert state['status']=='running_resume176_mixedcrop_e6'
old_pid=state['pid']
deadline=dt.datetime.fromisoformat(state['deadline_utc'].replace('Z','+00:00'))
records=[]
for p in (root/'.research-guardian/experiments').glob('*/record.json'):
    r=json.loads(p.read_text())
    if r.get('model')=='resume176_mixedcrop_e6':
        assert r['command'][-2:] == ['resume176_mixedcrop_e6','--development']
        records.append((r['started_at'],p,r))
_,record,training=max(records)
progress={'old_paused_driver_pid':old_pid,'handoff_pid':os.getpid(),
          'training_record':str(record),'deadline_utc':state['deadline_utc']}
(out/'queue_before_adaptive_reload.json').write_text(json.dumps(state,indent=2)+'\n')
def save(status):
    (out/'handoff.json').write_text(json.dumps(progress|{'status':status,
       'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2)+'\n')
save('waiting_registered_crop_training')
while True:
    r=json.loads(record.read_text())
    if r.get('status') not in ('running','starting'):
        break
    if dt.datetime.now(dt.timezone.utc)>=deadline:
        try:
            os.kill(training['wrapper_pid'],signal.SIGTERM)
        except ProcessLookupError:
            pass
        save('deadline_stop_sent_to_owned_runner')
        break
    time.sleep(2)
# The old parent is explicitly paused; terminate it only after its runner ends
# (or after deadline-stop was sent). No scan of unrelated processes is used.
os.kill(old_pid,signal.SIGTERM)
os.kill(old_pid,signal.SIGCONT)
save('reloading_updated_driver')
os.execv(sys.executable,[sys.executable,str(root/'scripts/resume_driver.py')])
