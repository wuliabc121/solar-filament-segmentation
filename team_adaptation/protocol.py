import json
from datetime import datetime,timezone
from pathlib import Path
from . import ROOT

def validate_cohort(observations,models,data,protected):
 if len(set(observations))!=len(observations) or not observations:raise ValueError('empty/duplicate cohort')
 if set(observations)&set(protected):raise ValueError('protected holdout requested')
 for m in models:
  trained=set(m['training_observations']);dates={data[s]['date'] for s in trained};hashes={data[s]['sha256'] for s in trained}
  for s in observations:
   if s in trained or data[s]['date'] in dates or data[s]['sha256'] in hashes:raise ValueError('training observation/date/content overlap: '+s)

def load_protocol(path):
 p=json.loads(Path(path).read_text());registry=json.loads((ROOT/'team/registry.json').read_text());data={x['observation_id']:x for x in (json.loads(s) for s in (ROOT/'reports/audit_001/data_manifest.jsonl').read_text().splitlines()) if x['partition']=='train'}
 assignments=json.loads((ROOT/'data/overnight176/split_manifest.json').read_text())['assignments'];protected={s for s,v in assignments.items() if v=='holdout'}
 m2=json.loads((ROOT/'team/protocols/m2_original.json').read_text())['observations'];protected|={s for s,v in m2.items() if v['split']=='holdout'}
 validate_cohort(p['primary_ids'],[registry['models']['m3'],registry['models']['m2']],data,protected)
 if set(p['tune_ids'])&set(p['check_ids']) or set(p['tune_ids'])|set(p['check_ids'])!=set(p['primary_ids']):raise ValueError('invalid tune/check partition')
 if {data[s]['date'] for s in p['tune_ids']}&{data[s]['date'] for s in p['check_ids']}:raise ValueError('tune/check date overlap')
 for name in ('m1','m2','m3'):
  if p['models'][name]['checkpoint_sha256']!=registry['models'][name]['checkpoint_sha256']:raise ValueError('registered checkpoint changed')
 return p,data

def check_deadline(protocol):
 if datetime.now(timezone.utc)>=datetime.fromisoformat(protocol['deadline_utc']):raise TimeoutError('campaign UTC deadline reached')
