"""Build a versioned report from completed experiment artifacts only."""
import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from m3_filament.io import sha256,write_json,write_text

out=Path('reports/optimization_001')
if out.exists():raise FileExistsError(out)
out.mkdir()
rows=[]
for size in [256,512,768,1024]:
 train=Path(f'runs/dev32_r{size}_e8')
 score=Path(f'runs/calibrated_r{size}_e8' if size!=1024 else 'runs/eval_dev32_r1024')
 t=json.load((train/'metrics.json').open());m=json.load((score/'metrics.json').open())
 rows.append({'run_id':train.name,'resolution':size,'nbs':64,'estimated_optimizer_updates':7,
  'train_seconds':t['elapsed_seconds'],'score_path':str(score/'metrics.json'),**m['selected'],
  'checkpoint_sha256':m['checkpoint_sha256']})
p=Path('runs/eval_dev32_r1024_nbs2/metrics.json')
if p.exists():
 m=json.load(p.open());t=json.load(open('runs/dev32_r1024_nbs2_e8/metrics.json'))
 rows.append({'run_id':'dev32_r1024_nbs2_e8','resolution':1024,'nbs':2,
  'actual_epochs':t.get('actual_epochs'),'optimizer_updates':t.get('optimizer_updates'),
  'train_seconds':t['elapsed_seconds'],'score_path':str(p),**m['selected'],'checkpoint_sha256':m['checkpoint_sha256']})
selection=json.load(open('reports/frozen-candidate.json'))
check=json.load(open('runs/check16_r1024_default/metrics.json'))
if selection['run_id']!='dev32_r1024_e8':check=json.load(open('runs/check16_selected/metrics.json'))
summary={'scope':'one-hour provisional development experiment','calibration_observations':8,'annotation_passes':14,
 'instances':95,'rows':rows,'selected':selection,'additional16':{'observations':16,'annotation_passes':check['annotation_passes_scored'],
  'threshold_retuned':False,'result':check['selected']},'holdout_used':False,'official_leaderboard_score':False,
 'warning':'Calibration-selected configurations and confidence thresholds are not unbiased performance estimates; n=1 training seed.'}
write_json(out/'results.json',summary)
with (out/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['run_id','resolution','nbs','confidence','pq','tp','fp','fn','train_seconds']);w.writeheader()
 for r in rows:w.writerow({k:r[k] for k in w.fieldnames})
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
ax[0].plot([r['resolution'] for r in rows[:4]],[r['pq'] for r in rows[:4]],'o-');ax[0].set(xlabel='Input resolution',ylabel='Development PQ',title='Same 8 epochs / 32 training observations')
for r in rows[:4]:ax[0].annotate(f"{r['pq']:.3f}",(r['resolution'],r['pq']),xytext=(0,8),textcoords='offset points',ha='center')
for r in rows[2:]:
 m=json.load(open(r['score_path']));ax[1].plot([x['confidence'] for x in m['grid']],[x['pq'] for x in m['grid']],'o-',label=r['run_id'].replace('dev32_',''))
ax[1].set(xlabel='Confidence threshold',ylabel='Development PQ',title='Calibration only');ax[1].set_xscale('log');ax[1].legend(fontsize=7)
fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
bench=json.load(open('reports/mask-dedup-benchmark.json'))
table='\n'.join(f"| {r['run_id']} | {r['nbs']} | {r['confidence']} | {r['pq']:.4f} | {r['tp']} / {r['fp']} / {r['fn']} | {r['train_seconds']:.1f} |" for r in rows)
selected=selection['run_id'];s=next(r for r in rows if r['run_id']==selected);c=check['selected']
text=f'''# M3 本轮实验与优化结果

已完成一轮受预算约束的开发实验。使用相同 32 个训练观测（57 次独立标注）、8 个校准观测（14 次标注、95 个实例），固定 seed42、原始 mask、YOLO11n-seg、batch2、8 epochs。正式 M1 split 尚未提供；未使用 holdout 或测试图像选参。

## 实测比较

| 运行 | nbs | confidence | 原图开发 PQ | TP / FP / FN | 训练计时秒 |
|---|---:|---:|---:|---:|---| 
{table}

以上 PQ 按组织者公开 Self Evaluation Notebook v6 的严格 IoU>0.5、全部合格匹配、逐标注者累积统计实现，已有原函数对照测试。它是本地开发集分数，不是排行榜成绩。表中训练时间来自进程计时；CPU 评测与 GPU 训练有重叠，不作硬件隔离的速度结论。

初始五个阈值中，768 的最优值位于上边界0.1，因此有限追加0.2/0.3/0.5，并对所有模型使用同一最终八点网格。校准选择偏差仍存在，不能称为泛化提升。

## 采用与不采用

当前冻结候选为 `{selected}`，checkpoint SHA256 `{s['checkpoint_sha256']}`，confidence `{s['confidence']}`。权重位置见 `../../runs/{selected}/weights/best.pt`，对应完整配置在 `../../configs/{selected}.yaml`。这里只冻结本轮开发候选，不能替代团队最终方案。

nbs64 在 batch2、无 warmup 时累计32个 minibatch 才更新一次，8 epochs共232个 minibatch，约7次更新。另测 nbs2，实际更新次数见其 metrics.json；没有因为更新更频繁就自动采用，仍按初始8组校准结果决定。nbs8与20-epoch配置仅准备、未启动，避免超出预算。

## 额外检查

在读取额外检查结果前已冻结选择规则、候选及阈值。从剩余 calibration 池固定抽取16个不与原8组重叠的日期/内容组，保留全部标注者；这不是团队正式 holdout。冻结候选在 {check['annotation_passes_scored']} 次标注上的 PQ 为 **{c['pq']:.4f}**，TP/FP/FN为 **{c['tp']}/{c['fp']}/{c['fn']}**，没有据此继续调参。见 `../../reports/check16-protocol.json` 和对应运行报告。

## 推理效率优化

将逐候选的整图二值两两去重改为 COCO RLE IoU，保持相同0.8门槛、置信度顺序、像素归属与输出。80张1024²合成掩码基准：{bench['old_boolean_seconds']:.3f}s → {bench['new_rle_seconds']:.3f}s，约 **{bench['speedup']:.1f}倍**；该数字仅为此本机合成工作负载。真实校准图37个实例的 mask、bbox、score与顺序也逐项一致。见 `../../reports/mask-dedup-real-parity.json`。

## 可信度与局限

本轮仅一个训练 seed，训练观测32、选参观测8，模型与阈值均经过校准选择；额外16组是开发检查，不是竞赛测试或M1冻结holdout。不能承诺排行榜增益或统计显著性。暗条漏检仍多，nbs频率改变也可能过拟合。M1/M2真实语义缓存和OOF尚无，未报告E5/E5R融合性能。

官方评分函数来源、哈希及对照测试见 `../../references/official-source-provenance.json` 和 `../../tests/test_evaluation.py`。源码审查修复了混合来源缓存、低于缓存预测门槛的无效校准、以及新数据集缺少唯一观测图像入口的问题。最终测试日志见 `../optimization-tests-final.log`。所有实际训练和主要评分运行均由research-guardian留存；一次1024配置生成失败在训练启动前修正，未把未运行配置当作实验结果。
'''
write_text(out/'report.md',text)
print(json.dumps({'selected':selected,'calibration_pq':s['pq'],'additional16_pq':c['pq'],'report':str(out/'report.md')},indent=2))
