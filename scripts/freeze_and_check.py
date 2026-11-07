import json,time,datetime,subprocess,sys
from pathlib import Path
from m3_filament.io import write_json
p=Path('runs/eval_dev32_r1024_nbs2/metrics.json')
start=time.monotonic()
while not p.exists():
 if time.monotonic()-start>180:raise TimeoutError('candidate calibration not complete')
 time.sleep(1)
a=json.load(open('runs/eval_dev32_r1024/metrics.json'))
b=json.load(p.open())
run='dev32_r1024_nbs2_e8' if b['selected']['pq']>a['selected']['pq'] else 'dev32_r1024_e8'
m=b if run.endswith('nbs2_e8') else a
choice={'run_id':run,'checkpoint_sha256':m['checkpoint_sha256'],'confidence':m['selected']['confidence'],
 'calibration_pq':m['selected']['pq'],'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'criterion':'initial8 calibration PQ only; additional16 outcomes were not read',
 'nbs2_adopted':run.endswith('nbs2_e8')}
write_json('reports/frozen-candidate.json',choice);print(choice,flush=True)
if run.endswith('nbs2_e8'):
 cmd=[sys.executable,str(Path.home()/'.agents/skills/research-guardian/scripts/experiment_logger.py'),'run',
 '--project',str(Path('.').resolve()),'--session-id','01a1a144-84a8-7f81-8664-65c6a24ea02b',
 '--model','selected-r1024-nbs2-additional16','--dataset','dev32-check16','--seed','43',
 '--config',f'configs/{run}.yaml','--metrics','runs/check16_selected/metrics.json','--param','tuning_allowed=false','--timeout','180','--',
 str(Path('run.sh').resolve()),'-m','m3_filament.evaluation','--weights',f'runs/{run}/weights/best.pt',
 '--dataset','data/dev32_check16','--output','runs/check16_selected','--imgsz','1024','--device','cpu',
 '--thresholds',str(choice['confidence'])]
 subprocess.run(cmd,check=True)
