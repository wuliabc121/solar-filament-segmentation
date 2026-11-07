"""Local evidence/archive closeout, after all own jobs and monitor have stopped."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import shlex
import zipfile
from m3_filament.io import sha256,read_jsonl
from m3_filament.model import source_manifest_hash
from m3_filament.joint import QUALITY

OUT=Path('reports/structure_20261104')


def dump(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')



def verify_checkpoint_evidence(records):
 missing=[]
 for record in records:
  for item in record.get('checkpoints',[]):
   if item.get('sha256'):
    assert Path(item['path']).is_file() and sha256(item['path'])==item['sha256'],item
   else:
    assert record['status']!='success' and not Path(item['path']).exists(),item
    missing.append({'experiment_id':record['id'],'status':record['status'],'path':item['path'],'evidence':'declared checkpoint never created; original failure retained'})
 return missing


def main():
 p=argparse.ArgumentParser();p.add_argument('--monitor-stopped',action='store_true',required=True);args=p.parse_args()
 state=json.loads((OUT/'state.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
 if state['status'].startswith(('running','waiting')):raise RuntimeError('controller not complete')
 if not state.get('driver_exit_verified'):raise RuntimeError('verify actual controller tool exit before closeout')
 for path in Path('.research-guardian/experiments').glob('*/record.json'):
  record=json.loads(path.read_text())
  if record.get('model','').startswith('structure_') and record['status']=='running':raise RuntimeError('own experiment still running: '+str(path))
 own_records=[json.loads(path.read_text()) for path in Path('.research-guardian/experiments').glob('*/record.json')]
 missing_checkpoint_evidence=verify_checkpoint_evidence([r for r in own_records if r.get('model','').startswith('structure_')])
 state.update(monitor_stopped=True,monitor_id='m3',closeout_utc=dt.datetime.now(dt.timezone.utc).isoformat(),goal_status=summary['goal_status']);dump(OUT/'state.json',state)
 ledger=[]
 for path in Path('.research-guardian/experiments').glob('*/record.json'):
  record=json.loads(path.read_text())
  if record.get('model','').startswith('structure_'):
   ledger.append({key:record.get(key) for key in ['id','model','status','started_at','ended_at','duration_seconds','exit_code','command']}|{'record_path':str(path),'record_sha256':sha256(path)})
 ledger.sort(key=lambda r:r['started_at']);dump(OUT/'experiment-ledger.json',ledger)
 commands=['# 全部实际实验命令','', '既有输出不能直接重跑；新训练需新run_id和新UTC预算，fresh12不能回流选参。']
 for record in ledger:commands+=['',record['id']+' '+record['model']+' '+record['status'],'','```sh',shlex.join(record['command']),'```']
 (OUT/'commands.md').write_text('\n'.join(commands)+'\n')
 metrics=summary['new_check_metrics'];lines=['## 2026-11-04 训练与结构损失实验（最新）','',
 '本轮约2小时预算，实际修改训练与分割损失：共同小目标加权裁剪/全图比例/同步旋转/亮度增强；方法组area监督缩小+BCE+Dice+clDice及保留相交格ROI。原始独立GT、隔离环境和原图JSONL/RLE/M4契约保持。',
 '', '**本轮状态：'+summary['goal_status']+'。** 验收为PQ>0.35、召回高于两个固定参照、误检更少、SQ不退步。以下为参数冻结后一次新开发检查（12观测/11日期内容组），不是排行榜：',
 '', '| 方案 | PQ | SQ | RQ | Precision | Recall | F1 | 小目标召回 | FP | FN |','|---|'+'---:|'*9]
 for name,row in metrics.items():lines.append('| '+name+' | '+' | '.join('N/A' if row[k] is None else f'{row[k]:.5f}' for k in QUALITY)+f" | {row['fp']} | {row['fn']} |")
 profile=json.loads((OUT/'candidate_profile.json').read_text()) if (OUT/'candidate_profile.json').exists() else None
 lines+=['', ('冻结候选：`'+profile['weights']+'`；imgsz=2048、conf='+str(profile['conf'])+'、area32、CPU、max_det100。完整哈希/代码/运行版本见 `reports/structure_20261104/candidate_profile.json`。旧默认未自动替换。' if profile else '未形成完成评估的冻结候选；保留实际模型与失败证据，不提供未经完成验证的采用建议。'),
 '', '训练实际轮数、EMA更新、耗时、原始失败、对照曲线、辅助IoU/Dice、11组配对bootstrap与可视化见 `reports/structure_20261104/report.md`。两主组只有完整10轮且更新数相等时才称等训练量对照；可选496接续增加数据/计算，单独解释。标签一致性分歧不是模型上限；无整标签缩小消失，不能据此解释漏检。',
 '', '新检查规模小，不能证明稳定泛化；剩余未用校准组已在本轮使用，后续不得再次称为未见检查。原holdout103和比赛test仍未触碰。M1正式split/M2正式入口及真实语义缓存限制仍在。监测已停止，所有自有任务结束。',
 '', '```sh','./run.sh scripts/predict_joint_profile.py --profile reports/structure_20261104/candidate_profile.json --images /新的原图目录 --output runs/new_structure_prediction','```','']
 readme=Path('README.md');text=readme.read_text();header='## 2026-11-04 综合指标实验（未达标）'
 if '## 2026-11-04 训练与结构损失实验（最新）' in text:raise RuntimeError('closeout already added README')
 text=text.replace(header,'\n'.join(lines)+'\n'+header,1)
 text=text.replace('最新综合实验见 `reports/balanced_20261104/decision.md` 和 `report.md`','最新训练/结构方法实验见 `reports/structure_20261104/report.md`；先前综合实验见 `reports/balanced_20261104/decision.md`')
 readme.write_text(text)
 (OUT/'closeout.md').write_text('# 用户要求暂停\n\n后续质量预测头仅完成论文调研，未编码或训练。暂停与恢复边界见项目RESUME.md。\n\nUTC硬截止07:22（北京时间15:22），未延长。任务实际状态：'+summary['goal_status']+'。原始holdout103/test未用；不把临时开发分数当排行榜。模型、更新数、源码及失败均保存。所有自有训练/评分进程已结束，driver退出已核对，automation m3已删除、监测停止。\n\n最终测试见tests-final.log，最终复核见review-final.md，源码包仅本地准备、不发布、不Git提交推送。验证清单在verification.json。\n')
 files=[Path(name) for name in ['README.md','RESUME.md','AGENTS.md','requirements-lock.txt','environment-explicit.txt','run.sh','setup.sh','pyproject.toml'] if Path(name).exists()]
 for folder in ['src','scripts','tests','configs']:
  files.extend(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.sh','.yaml','.yml','.toml'))
 for name in ['report.md','methodology-review.md','review-final.md','closeout.md','summary.json','candidate_profile.json','campaign.json','frozen_selection.json','review-revisions.md','checkpoint-plan-revision.json','runtime-plan-revision.json','commands.md']:
  path=OUT/name
  if path.exists():files.append(path)
 files.append(Path('docs/superpowers/plans/2026-11-04-structure-training.md'))
 release=Path('releases/structure_20261104_review');args=['python3',str(Path.home()/'.agents/skills/research-guardian/scripts/research_archive.py'),'--project',str(Path.cwd()),'--output',str(release.resolve())]
 for path in sorted(set(files)):args+=['--include',str(path)]
 subprocess.run(args,check=True)
 archive=Path('releases/structure_20261104_source.zip')
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for path in sorted(release.rglob('*')):
   if path.is_file():z.write(path,path.relative_to(release))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for entry in json.loads((release/'manifest.json').read_text())['files']:assert hashlib.sha256(z.read('files/'+entry['path'])).hexdigest()==entry['sha256']
 paths=set(files);paths.add(archive);paths.update(p for p in release.rglob('*') if p.is_file())
 paths.update(p for p in OUT.iterdir() if p.is_file() and p.name not in ('evidence-manifest.json','verification.json','summary-wrapper.log','closeout-wrapper.log','closeout.log'))
 # Include complete source snapshots, weights, prediction caches, scoring and actual records.
 for folder in Path('runs').glob('structure_*'):
  paths.update(p for p in folder.rglob('*') if p.is_file() and ('code_snapshot' in p.parts or p.suffix=='.pt' or p.name in ('metrics.json','run_manifest.json','run_manifest_started.json','external_training.json','selection_trace.json','criterion_actual.json','inference_exports.json','recovery.json','instances.jsonl','per_annotation.jsonl','results.csv')))
 for name in ['balanced_maskmetric_ft4','resume176_mixedcrop_e6']:
  paths.add(Path('runs',name,'weights/best.pt'));paths.add(Path('runs',name,'run_manifest.json'))
 for folder in ['data/overnight176','data/balanced496','data/structure_fresh_check12']:
  directory=Path(folder)
  paths.update(p for p in directory.glob('*.json*') if p.is_file())
  paths.update(p for p in (directory/'targets').rglob('*.json') if p.is_file())
 for path in Path('.research-guardian/experiments').glob('*/record.json'):
  record=json.loads(path.read_text())
  if record.get('model','').startswith('structure_'):
   paths.add(path)
   # Checkpoint hashes and recorded missing failures were verified before mutations.
   evidence=record.get('metrics_evidence')
   if evidence:assert sha256(evidence['path'])==evidence['sha256']
 manifest_path=Path('reports/audit_001/data_manifest.jsonl');paths.add(manifest_path)
 raw_checked=0;used=set()
 for run in ['structure_control_e10','structure_dicecl_e10','structure496_transfer']:
  path=Path('runs')/run/'run_manifest.json'
  if path.exists():
   data=json.loads(path.read_text());used.update(data.get('training_observation_ids',[]));used.update(data.get('calibration_observation_ids',[]))
 used.update(json.loads((OUT/'fresh_check.json').read_text())['calibration'] if summary['fresh_check_cases'] else [])
 raw_lookup={r['observation_id']:r for r in read_jsonl(manifest_path) if r['partition']=='train'}
 for obs in used:
  row=raw_lookup[obs];path=Path('data/raw')/row['relative_path'];assert sha256(path)==row['sha256'];paths.add(path);raw_checked+=1
 for directory in ['data/overnight176','data/balanced496','data/structure_fresh_check12']:
  path=Path(directory)/'samples.jsonl'
  if not path.exists():continue
  for sample in read_jsonl(path):
   if sample['observation_id'] in used:assert sha256(sample['image_path'])==raw_lookup[sample['observation_id']]['sha256']
 # Original holdout103 and competition test files are not opened by this closeout.
 annotation=Path('data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json');paths.add(annotation)
 for folder in ['data/overnight176','data/balanced496']:
  prep=json.loads(Path(folder,'preparation.json').read_text());assert sha256(annotation)==prep['annotation_sha256']
 for run in ['structure_control_e10','structure_dicecl_e10','structure496_transfer']:
  folder=Path('runs')/run/'code_snapshot'
  for filename in ['source_manifest.json','external_source_manifest.json']:
   path=folder/filename
   if path.exists():
    for relative,digest in json.loads(path.read_text()).items():assert sha256(folder/relative)==digest,(run,relative)
 if profile:
  assert profile['checkpoint_sha256']==sha256(profile['weights']) and profile['source_manifest_sha256']==source_manifest_hash()
  assert profile['inference_script_sha256']==sha256('scripts/predict_joint_profile.py')
  assert profile['requirements_lock_sha256']==sha256('requirements-lock.txt')
 manifest={str(path):sha256(path) for path in sorted(paths) if path.exists()};dump(OUT/'evidence-manifest.json',manifest)
 for path,digest in manifest.items():assert sha256(path)==digest,path
 dump(OUT/'verification.json',{'verified_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'files_hashed_and_rechecked':len(manifest),'raw_training_calibration_images_rechecked':raw_checked,'original_holdout_and_test_opened':False,
 'manifest_sha256':sha256(OUT/'evidence-manifest.json'),'archive_sha256':sha256(archive),'archive_crc':'passed',
 'training_source_snapshots_and_external_modules':'matched','registered_metrics_and_checkpoint_hashes':'matched where evidence exists','missing_checkpoint_failure_evidence':missing_checkpoint_evidence,'goal_status':summary['goal_status'],
 'selected_weights':profile['weights'] if profile else None,'selected_checkpoint_sha256':profile['checkpoint_sha256'] if profile else None,'monitor_stopped':True,
 'self_and_manifest_excluded_to_avoid_circular_hash':True})
 print((OUT/'verification.json').read_text())


if __name__=='__main__':main()
