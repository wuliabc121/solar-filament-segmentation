import datetime
import sys
import time
from m3_filament.queueing import run_wall_limited,choose_recall_candidate


def test_utc_timeout_stops_owned_process_and_retains_checkpoint(tmp_path):
    checkpoint=tmp_path/'checkpoint'
    code='from pathlib import Path; import time; Path('+repr(str(checkpoint))+').write_text("saved"); time.sleep(30)'
    deadline=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=.5)).isoformat()
    started=time.time()
    result=run_wall_limited([sys.executable,'-c',code],tmp_path/'log',deadline,10)
    assert result['returncode']==124 and result['timed_out']
    assert checkpoint.read_text()=='saved'
    assert time.time()-started<4


def test_recall_selection_enforces_pq_and_false_positive_limits():
    base={'pq':.15,'fn':80,'fp':50}
    rows=[{'name':'baseline',**base},{'name':'good','pq':.16,'fn':60,'fp':70},
          {'name':'too_noisy','pq':.1,'fn':30,'fp':200}]
    assert choose_recall_candidate(rows,base)['name']=='good'
