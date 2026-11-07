"""Summarize completed structure experiments; no model selection or new inference."""
import csv
import json
import platform
import shlex
from pathlib import Path
from importlib.metadata import version
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools import mask as coco
from m3_filament.io import read_jsonl,write_json,sha256
from m3_filament.joint import derive_metrics,QUALITY
from m3_filament.data import observation_groups
from m3_filament.evaluation import native_rle,pairwise_iou
from m3_filament.model import source_manifest_hash
from summarize_balanced import bootstrap_comparison,vector_metrics,COUNTS

OUT=Path('reports/structure_20261104')
NAMES={'latest_quality_reference':'Previous quality','previous_recall_reference':'Previous recall','new_candidate':'New candidate'}


def check_paths(name):
 suffix='_raw' if name=='previous_recall_reference' else '_area32'
 folder=Path('runs')/('structure_check_'+name+suffix)
 return folder,folder.with_name(folder.name+'_predictions')/'instances.jsonl'


def auxiliary(checks):
 samples=list(read_jsonl('data/structure_fresh_check12/samples.jsonl'));summary={}
 for name,data in checks.items():
  _,prediction=check_paths(name);records={r['observation_id']:r for r in read_jsonl(prediction)}
  totals={'gt':0,'matched':0,'fn_zero':0,'fn_0_to_p1':0,'fn_p1_to_p3':0,'fn_p3_to_p5':0,'gt_area':0.,'pred_area':0.,'intersection':0.}
  ious=[];dices=[]
  for sample in samples:
   gt=json.loads(Path(sample['target_path']).read_text())['masks']
   pred=[i['segmentation'] for i in records[sample['observation_id']]['instances'] if i['score']>=data['selected']['confidence']]
   matrix=pairwise_iou(gt,pred);best=matrix.max(1) if pred else np.zeros(len(gt))
   totals['gt']+=len(gt);totals['matched']+=int((best>.5).sum());totals['fn_zero']+=int((best==0).sum())
   totals['fn_0_to_p1']+=int(((best>0)&(best<=.1)).sum());totals['fn_p1_to_p3']+=int(((best>.1)&(best<=.3)).sum());totals['fn_p3_to_p5']+=int(((best>.3)&(best<=.5)).sum())
   g=coco.merge([native_rle(x) for x in gt]) if gt else None;p=coco.merge([native_rle(x) for x in pred]) if pred else None
   ga=float(coco.area(g)) if g else 0.;pa=float(coco.area(p)) if p else 0.;inter=float(coco.area(coco.merge([g,p],intersect=True))) if g and p else 0.
   totals['gt_area']+=ga;totals['pred_area']+=pa;totals['intersection']+=inter
   ious.append(inter/(ga+pa-inter) if ga+pa-inter else 1.);dices.append(2*inter/(ga+pa) if ga+pa else 1.)
  ga,pa,inter=totals['gt_area'],totals['pred_area'],totals['intersection']
  summary[name]={'counts':totals,'foreground_micro_iou':inter/(ga+pa-inter) if ga+pa-inter else 1.,'foreground_micro_dice':2*inter/(ga+pa) if ga+pa else 1.,
   'mean_annotation_iou':float(np.mean(ious)),'mean_annotation_dice':float(np.mean(dices)),
   'predictions_sha256':sha256(prediction),'annotation_passes_including_zero_gt':len(samples)}
 return {'diagnostic_only':True,'used_for_selection':False,'note':'frozen outputs after threshold; pixels are auxiliary, no low-score proposal absence claim','models':summary}



def candidate_intervals(rows,groups,draws=5000):
 units=sorted({groups[r['observation_id']] for r in rows});index={g:i for i,g in enumerate(units)}
 if not units:raise ValueError('no scorable groups for absolute interval')
 values=np.zeros((len(units),len(COUNTS)))
 for row in rows:values[index[groups[row['observation_id']]]]+=np.array([row[key] for key in COUNTS])
 rng=np.random.default_rng(42);sample=rng.integers(0,len(units),size=(draws,len(units)))
 metrics=vector_metrics(values[sample].sum(1))
 return {'independent_groups':len(units),'draws':draws,'seed':42,'unit':'entire date/content connected group including all independent annotations',
 'intervals':{key:np.nanquantile(value,[.025,.975]).tolist() for key,value in metrics.items()},
 'scope':'fixed-model development data uncertainty, not training-seed variance or official leaderboard confidence'}


def row(label,m):
 m=derive_metrics(m)
 return '| '+label+' | '+' | '.join('N/A' if m[k] is None else f'{m[k]:.5f}' for k in QUALITY)+f" | {m['tp']} | {m['fp']} | {m['fn']} |"


def main():
 result=json.loads((OUT/'results.json').read_text());state=json.loads((OUT/'state.json').read_text());checks=result.get('fresh_check',{});tuning=result.get('tuning',{})
 goal='target_met_on_small_fresh_development_check' if result.get('goal_met') else 'not_met' if len(checks)==3 else 'not_fully_evaluated'
 rows=[]
 for name,data in tuning.items():
  checkpoint_name=data.get('checkpoint_name','best.pt')
  export_path=Path('runs',data.get('model_run',name),'inference_exports.json')
  if 'checkpoint_name' not in data and export_path.exists():checkpoint_name=Path(json.loads(export_path.read_text())['exports']['best.pt']['inference_path']).name
  for point in data['grid']:rows.append(derive_metrics(point)|{'candidate':name,'run':data.get('model_run',name),'checkpoint':checkpoint_name,'training_status':data.get('training_status'),'imgsz':2048,'min_area':32})
 with (OUT/'tuning_metrics.csv').open('w') as f:
  fields=['candidate','run','checkpoint','training_status','imgsz','confidence','min_area',*QUALITY,'tp','fp','fn','n_gt','small_gt','small_matched']
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 for name in tuning:
  r=[r for r in rows if r['candidate']==name]
  for ax,x,y in [(axes[0],'confidence','pq'),(axes[1],'fp','recall'),(axes[2],'confidence','sq')]:ax.plot([z[x] for z in r],[z[y] for z in r],'.-',label=name)
 for ax,x,y in [(axes[0],'Confidence','PQ'),(axes[1],'FP','Unique-instance recall'),(axes[2],'Confidence','SQ')]:ax.set(xlabel=x,ylabel=y);ax.grid(alpha=.2);ax.legend(fontsize=6)
 axes[0].axhline(.35,c='r',ls='--',lw=1);fig.suptitle('Original 24-observation tuning only, fixed 2048 / area32');fig.tight_layout();fig.savefig(OUT/'tuning.png',dpi=180);plt.close(fig)
 values={name:derive_metrics(data['selected']) for name,data in checks.items()};bootstrap={};aux=None
 if len(checks)==3:
  audit=[r for r in read_jsonl('reports/audit_001/data_manifest.jsonl') if r['partition']=='train'];groups={o:min(g) for g in observation_groups(audit) for o in g}
  candidate=list(read_jsonl(check_paths('new_candidate')[0]/'per_annotation.jsonl'))
  for name in checks:
   if name=='new_candidate':continue
   bootstrap[name]=bootstrap_comparison(list(read_jsonl(check_paths(name)[0]/'per_annotation.jsonl')),candidate,groups)
  bootstrap['candidate_absolute']=candidate_intervals(candidate,groups)
  write_json(OUT/'bootstrap.json',bootstrap);aux=auxiliary(checks);write_json(OUT/'auxiliary_metrics.json',aux)
  fig,axes=plt.subplots(3,3,figsize=(11,9));names=list(values)
  for ax,key in zip(axes.flat,[*QUALITY,'fp','fn']):
   numbers=[values[n][key] if values[n][key] is not None else 0 for n in names];bars=ax.bar(['Prev quality','Prev recall','New'],numbers,color=['tab:blue','tab:orange','purple']);ax.set_title(key+(' (lower better)' if key in ('fp','fn') else ' (higher better)'));ax.tick_params(axis='x',labelsize=8);ax.margins(y=.2)
   for b,v in zip(bars,numbers):
    if v is not None:ax.annotate(str(int(v)) if key in ('fp','fn') else f'{v:.3f}',(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='bottom',fontsize=8)
   if key=='pq':ax.axhline(.35,c='r',ls='--',lw=1)
  fig.suptitle('Frozen final development check: 12 observations / 11 date-content groups');fig.tight_layout();fig.savefig(OUT/'fresh_check.png',dpi=180);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(9,3.8))
  for ax,key in zip(axes,['foreground_micro_iou','foreground_micro_dice']):
   bars=ax.bar(['Prev quality','Prev recall','New'],[aux['models'][n][key] for n in names],color=['tab:blue','tab:orange','purple']);ax.set_title(key);ax.margins(y=.2)
   for b in bars:ax.annotate(f'{b.get_height():.3f}',(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='bottom')
  fig.suptitle('Auxiliary pixel-union metrics, independent annotation passes');fig.tight_layout();fig.savefig(OUT/'pixel_metrics.png',dpi=180);plt.close(fig)
 train=[]
 for name in ['structure_control_e10','structure_dicecl_e10','structure496_transfer']:
  p=Path('runs')/name/'metrics.json'
  if p.exists():
   d=json.loads(p.read_text());train.append({k:d.get(k) for k in ['run_id','actual_epochs','optimizer_updates','elapsed_seconds','checkpoint_sha256','recovered_for_inference','optimizer_updates_scope']}|{'metrics_sha256':sha256(p)})
 primary=[t for t in train if t['run_id'] in ['structure_control_e10','structure_dicecl_e10']]
 equal_updates=(len(primary)==2 and all(t['actual_epochs']==10 and not t.get('recovered_for_inference') and t['optimizer_updates'] is not None for t in primary) and len({t['optimizer_updates'] for t in primary})==1)
 selection=result.get('selection',{});candidate=selection.get('candidate');profile=None
 if candidate:
  weights='runs/'+candidate['run']+'/weights/'+candidate.get('checkpoint','best.pt')
  saved_epoch=None;saved_updates=None;selection_evidence='unavailable after strip'
  export_path=Path('runs',candidate['run'],'inference_exports.json')
  if export_path.exists():
   for item in json.loads(export_path.read_text())['exports'].values():
    if Path(item['inference_path']).name==candidate.get('checkpoint','best.pt'):
     raw_epoch=item.get('original_epoch_zero_index');saved_epoch=raw_epoch+1 if isinstance(raw_epoch,int) and raw_epoch>=0 else None
     saved_updates=item.get('original_updates');selection_evidence='original unstripped checkpoint metadata retained in inference_exports.json'
  trace_path=Path('runs',candidate['run'],'selection_trace.json')
  if saved_epoch is None and trace_path.exists():
   trace=json.loads(trace_path.read_text())
   if trace:
    chosen=max(trace,key=lambda x:(x['mask_ap95'],x['epoch'])) if 'best' in candidate.get('checkpoint','best.pt') else max(trace,key=lambda x:x['epoch'])
    saved_epoch=chosen['epoch'];selection_evidence='mask selection trace epoch; checkpoint updates unavailable after optimizer strip, not estimated'
  training_metadata=json.loads(Path('runs',candidate['run'],'metrics.json').read_text())
  if saved_updates is None and saved_epoch==training_metadata.get('actual_epochs') and not training_metadata.get('recovered_for_inference'):
   saved_updates=training_metadata.get('optimizer_updates')
   selection_evidence+='; same final epoch as directly observed trainer EMA update count'
  profile={'weights':weights,'checkpoint_sha256':sha256(weights),'checkpoint_name':candidate.get('checkpoint','best.pt'),'run':candidate['run'],'imgsz':2048,'conf':candidate['confidence'],
  'score_floor':candidate['confidence'],'min_area':32,'closing_radius':0,'device':'cpu','max_det':100,'status':goal,
  'source_manifest_sha256':source_manifest_hash(),'requirements_lock_sha256':sha256('requirements-lock.txt'),
  'inference_script_sha256':sha256('scripts/predict_joint_profile.py'),
  'runtime_versions':{'python':platform.python_version(),**{name:version(name) for name in ['torch','torchvision','ultralytics','numpy','pycocotools','opencv-python']}},
  'frozen_selection_sha256':sha256(OUT/'frozen_selection.json'),'independent_check_observations':12,'independent_check_groups':11,
  'official_leaderboard_score':False,'old_default_automatically_replaced':False,
  'selected_checkpoint_epoch_in_run':saved_epoch,'selected_checkpoint_optimizer_updates':saved_updates,'selection_evidence':selection_evidence}
  write_json(OUT/'candidate_profile.json',profile)
 ledger=[]
 for p in Path('.research-guardian/experiments').glob('*/record.json'):
  d=json.loads(p.read_text())
  if d.get('model','').startswith('structure_') and d.get('model')!='structure_final_analysis':
   ledger.append({k:d.get(k) for k in ['id','model','status','started_at','ended_at','duration_seconds','exit_code','command']}|{'record_path':str(p),'record_sha256':sha256(p)})
 ledger.sort(key=lambda d:d['started_at']);write_json(OUT/'experiment-ledger.json',ledger)
 lines=['# M3训练与结构损失实验','', '目标状态：**'+goal+'**。PQ>0.35、Recall高于两个固定参照、FP少于两个参照、SQ不低于两个参照。只在原调参24幅选模型与操作点；冻结后仅一次新的12幅检查。',
 '', '## 实际改动','', '相同C模型热启动，两主组各计划10轮：75%全图/25%小目标加权1024裁剪，90/180/270度旋转概率0.5、水平翻转、轻量对比度/亮度增强；几何变换同时作用所有独立实例mask，框按变换后mask重算。控制组原始BCE；方法组保覆盖率area目标缩小+BCE+0.5Dice+0.2foreground soft-clDice（3次骨架迭代），按正样本实例计算，ROI按相交格floor/ceil。原始缓存GT不改、不合并标注者、不填孔。',
 '', 'clDice参考[原论文](https://arxiv.org/abs/2003.07311)与[作者实现](https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch)。本项目是前景/逐实例/有限迭代改编，不声称暗条拓扑保证。损失、监督缩小与ROI一起变化，不能把方法组收益单独归因于clDice。若触发496观测接续，它改变训练数据和训练量，也不是等计算消融。',
 '', '训练标签审计1999实例、784小目标，没有整实例在固定1024→256最近邻流程中完全消失；不能将实际漏检归因为整标签消失。area和clDice改善部分覆盖/细结构是待验证假设，最终以质量指标验收。',
 '', '## 实际训练量','', '| Run | Epochs | EMA updates | Seconds | Best SHA256 |','|---|---:|---:|---:|---|']
 for t in train:lines.append(f"| {t['run_id']} | {t['actual_epochs']} | {t['optimizer_updates']} | {t['elapsed_seconds']:.1f} | {t['checkpoint_sha256']} |")
 lines+=['', '同轮次主对照实际完成：'+str(bool(result.get('controlled_loss_comparison_complete',False) and equal_updates))+'。中断恢复和更新数未知/下界按原记录保留，不能当作完整训练。外部训练脚本及method-only ROI/criterion审核脚本已各自快照；实际criterion_after_backward证据见run/criterion_actual.json。该表是整个run实际完成训练量，不一定等于被选best权重的保存时刻；候选保存epoch、可观测updates及来源见candidate_profile.json，strip后未知updates不估算。',
 '', '## 新检查（冻结后，不能回流选参）','', '| 方案 | '+' | '.join(QUALITY)+' | TP | FP | FN |','|---|'+'---:|'*10]
 for name,data in checks.items():lines.append(row(NAMES[name],data['selected']))
 if result.get('final_gate'):lines+=['','未满足项：'+str(result['final_gate']['violations'])]
 if 'candidate_absolute' in bootstrap:
  interval=bootstrap['candidate_absolute']['intervals']['pq'];lines+=['','候选PQ的95%整组bootstrap区间：'+str(interval)+'。目标状态使用本次检查点估计；若区间覆盖0.35，不能声称已稳健超过该水平。']
 if aux:
  lines+=['','### 辅助前景像素指标','', '| 方案 | micro IoU | micro Dice |','|---|---:|---:|']
  for name,data in aux['models'].items():lines.append(f"| {NAMES[name]} | {data['foreground_micro_iou']:.5f} | {data['foreground_micro_dice']:.5f} |")
  lines+=['','FN按最终输出IoU分桶（每位标注者独立计数）：'+json.dumps(aux['models']['new_candidate']['counts'],ensure_ascii=False)+'。零交集是在固定阈值之后，不证明没有低分原始候选。']
 lines+=['','## 可信边界与失败','', '新检查只有12观测/11日期内容组，统计不确定性较大；5000次整组配对bootstrap见bootstrap.json，不含训练随机种子方差。原始holdout103与比赛test从未使用。旧check24/reserve24/check16只作历史诊断，不用于本轮选参；原24校准仍重复使用，存在开发集过拟合风险。M1正式split/M2正式入口缺失，官方Notebook v6对齐的离线分数不能称排行榜。真实语义网络缓存未提供，不能归因于E5/E5R融合。',
 '', '失败记录：'+json.dumps(state.get('failures',[]),ensure_ascii=False)+'。早期smoke2因上游crop_mask原地改sigmoid导致反向错误已修复并回归；smoke3/4与MPS完整实例梯度验证通过。配置生成曾误用无yaml的系统Python，在登记前失败；没有把它算成已训练模型。所有失败保留。预算中断的自定义损失checkpoint以独立导出任务创建无训练criterion的*_inference.pt；原始optimizer/checkpoint不覆盖，核对EMA参数逐字节一致且正常新进程能加载。实际采用文件名与哈希以candidate_profile.json为准。',
 '', '## 使用与证据','', 'candidate_profile.json只是本轮冻结候选；旧默认未自动替换。用完整训练权重及冻结操作点对未训练原图推理：','', '```sh',
 './run.sh scripts/predict_joint_profile.py --profile reports/structure_20261104/candidate_profile.json --images /新的原图目录 --output runs/new_structure_prediction', '```',
 '', '输出原图JSONL/RLE及M4 crop元数据；分数是检测置信度，非像素概率。命令、实际记录和哈希见commands.md、experiment-ledger.json、evidence-manifest.json。环境沿用项目.env，未修改base/global；无比赛提交、发布、Git提交推送或外发消息。',
 '', '![tuning](tuning.png)']
 if len(checks)==3:lines+=['','![fresh check](fresh_check.png)','','![pixels](pixel_metrics.png)']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 commands=['# 实际命令与复现边界','', '每个新实验必须新run_id与新UTC预算；不能直接重用既有输出，也不能再次以fresh12选参。']
 for r in ledger:commands+=['',r['id']+' '+r['model']+' '+r['status'],'','```sh',shlex.join(r['command']),'```']
 (OUT/'commands.md').write_text('\n'.join(commands)+'\n')
 write_json(OUT/'summary.json',{'goal_status':goal,'goal_met':bool(result.get('goal_met')),'fresh_check_cases':len(checks),'tuning_points':len(rows),
 'training':train,'new_check_metrics':values,'controlled_loss_comparison_complete':bool(result.get('controlled_loss_comparison_complete',False) and equal_updates),'primary_update_counts_equal':equal_updates,
 'pq_95_group_bootstrap':bootstrap.get('candidate_absolute',{}).get('intervals',{}).get('pq'),'fresh_check_parameters_frozen_before_export':True,'original_holdout103_used':False,'competition_test_used':False,'default_automatically_replaced':False})
 print(json.dumps({'goal_status':goal,'fresh_check_cases':len(checks),'training_runs':len(train),'tuning_points':len(rows)},indent=2))


if __name__=='__main__':main()
