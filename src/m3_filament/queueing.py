"""Owned-process wall-clock limits and predeclared recall selection."""
import datetime as dt
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def run_wall_limited(command, log_path, deadline_utc, max_seconds):
    deadline = dt.datetime.fromisoformat(deadline_utc.replace('Z', '+00:00'))
    if deadline.tzinfo is None or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('aware UTC deadline and positive finite time budget required')
    started = dt.datetime.now(dt.timezone.utc)
    monotonic_start = time.monotonic()
    result = {'command': list(command), 'started_utc': started.isoformat(), 'pid': None}
    if started >= deadline:
        return result | {'returncode': 124, 'timed_out': True, 'elapsed_seconds': 0}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result['pid'] = child.pid
        timed_out = False
        try:
            while child.poll() is None:
                if dt.datetime.now(dt.timezone.utc) >= deadline or time.monotonic()-monotonic_start >= max_seconds:
                    timed_out = True
                    break
                time.sleep(.1)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        result.update(returncode=124 if timed_out else child.returncode, timed_out=timed_out,
                      ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-monotonic_start)
    return result


def choose_recall_candidate(rows, baseline):
    """Minimize FN subject to PQ >=95% baseline/90% best and FP <=2x baseline."""
    if not rows:
        raise ValueError('no measured candidates')
    floor = max(.95*baseline['pq'], .9*max(r['pq'] for r in rows))
    feasible = [r for r in rows if r['pq'] >= floor and r['fp'] <= 2*baseline['fp']]
    if not feasible:
        # Keep the measured baseline when the combined quality guard cannot be met.
        return dict(baseline, selection_feasible=False, pq_floor=floor)
    return dict(min(feasible, key=lambda r: (r['fn'], -r['pq'], r['fp'])),
                selection_feasible=True, pq_floor=floor)
