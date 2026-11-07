"""Create plots, paired uncertainty and evidence review from completed cached scores."""
import csv
import datetime as dt
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from m3_filament.io import read_jsonl, sha256
from m3_filament.evaluation import aggregate_counts

OUT=Path('reports/resume_20261104')


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def paired_bootstrap(base_path,candidate_path,base_conf,candidate_conf):
    def grouped(path,conf):
        groups={}
        for r in read_jsonl(path):
            if r['confidence']==conf:
                groups.setdefault(r['observation_id'],[]).append(r)
        return groups
    b=grouped(base_path,base_conf);c=grouped(candidate_path,candidate_conf)
    assert b.keys()==c.keys()
    # Frozen selection used distinct observation/date/content groups. Keep all
    # annotation passes of each observation together in every bootstrap draw.
    ids=sorted(b);rng=np.random.default_rng(42);deltas=[]
    for _ in range(5000):
        chosen=rng.choice(ids,size=len(ids),replace=True)
        rb=aggregate_counts([r for obs in chosen for r in b[obs]])
        rc=aggregate_counts([r for obs in chosen for r in c[obs]])
        deltas.append([rc['fn']-rb['fn'],rc['pq']-rb['pq']])
    ci=np.quantile(deltas,[.025,.975],axis=0)
    return {'unit':'observation (all annotation passes kept together)', 'observations':len(ids),
            'draws':5000,'seed':42,'fn_delta_95_percentile_interval':ci[:,0].tolist(),
            'pq_delta_95_percentile_interval':ci[:,1].tolist(),
            'scope':'within this fixed checkpoint pair; no across-training-seed uncertainty'}


def main():
    result=json.loads((OUT/'results.json').read_text())
    selected=result['frozen_selection'];tuning=result['tuning'];check=result['independent_check']
    if not selected:
        raise RuntimeError('selection not completed; keep partial report')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for name,data in tuning.items():
        grid=data['grid'];label=name.replace('resume176_','').replace('dev32_r1024_nbs2_e8','old baseline')
        axes[0,0].plot([r['confidence'] for r in grid],[r['fn'] for r in grid],'.-',label=label)
        axes[0,1].plot([r['confidence'] for r in grid],[r['pq'] for r in grid],'.-',label=label)
        axes[1,0].plot([r['fp'] for r in grid],[r['fn'] for r in grid],'.-',label=label)
        csv_path=Path('runs')/name/'results.csv'
        if csv_path.exists() and name in ('resume176_global_e6','resume176_mixedcrop_e6'):
            rows=list(csv.DictReader(csv_path.open()))
            axes[1,1].plot([int(r['epoch']) for r in rows],[float(r['val/seg_loss']) for r in rows],'.-',label=label)
    axes[0,0].set(xlabel='Confidence threshold',ylabel='FN (IoU > 0.5 match required)',xscale='log')
    axes[0,1].set(xlabel='Confidence threshold',ylabel='Development PQ',xscale='log')
    axes[1,0].set(xlabel='FP',ylabel='FN',title='Tuning trade-off (lower-left is better)')
    axes[1,1].set(xlabel='Epoch',ylabel='Validation segmentation loss (same 24 observations)')
    for ax in axes.flat:
        ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.suptitle('Fixed 24-observation tuning set; not competition test')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    appendix=['','## 图表与冻结检查结论','', '![comparison](comparison.png)','']
    uncertainty=None
    if len(check)==2:
        b=check['baseline_conf0.2']['selected'];c=check['selected_frozen']['selected']
        reduction=(b['fn']-c['fn'])/b['fn'] if b['fn'] else 0.
        uncertainty=paired_bootstrap(Path(check['baseline_conf0.2']['artifact_output'])/'per_annotation.jsonl',
                                    Path(check['selected_frozen']['artifact_output'])/'per_annotation.jsonl',.2,selected['confidence'])
        passes=c['fn']<b['fn'] and c['pq']>=.95*b['pq'] and c['fp']<=2*b['fp']
        recommendation={'run':selected['run'] if passes else 'dev32_r1024_nbs2_e8',
                        'confidence':selected['confidence'] if passes else .2,
                        'imgsz':selected['imgsz'] if passes else 1024,
                        'independent_check_guard_passed':passes,
                        'evaluated_candidate_fn_relative_reduction':reduction,'bootstrap':uncertainty,
                        'policy':'adopt_candidate' if passes else 'retain_conservative_default; candidate remains recall-priority experimental option',
                        'checkpoint_sha256':sha256(Path('runs')/(selected['run'] if passes else 'dev32_r1024_nbs2_e8')/'weights/best.pt')}
        write(OUT/'recommendation.json',recommendation)
        write(OUT/'recall_candidate.json',{'run':selected['run'],
              'weights':'runs/'+selected['run']+'/weights/best.pt',
              'checkpoint_sha256':sha256(Path('runs')/selected['run']/'weights/best.pt'),
              'imgsz':selected['imgsz'],'conf':selected['confidence'],'device':'cpu','max_det':100,
              'require_unseen':True,'automatic_adoption_guard_passed':passes,
              'reserved_check':c,'baseline_reserved_check':b,
              'note':'recall-priority candidate; FP guard failed' if not passes else 'passed fixed local check only'})
        appendix += [f"独立24幅检查：FN {b['fn']}→{c['fn']}（相对减少{100*reduction:.1f}%），FP {b['fp']}→{c['fp']}，PQ {b['pq']:.5f}→{c['pq']:.5f}。",
                     '',f"预设迁移检查（FN降低、PQ≥旧模型95%、FP≤2倍）{'通过' if passes else '未通过'}。建议运行 `{recommendation['run']}`、conf={recommendation['confidence']}、imgsz={recommendation['imgsz']}、device=cpu、max_det=100；这一结论只覆盖本轮开发检查样本。",
                     '',f"按观测配对bootstrap（5000次、seed42）：FN差值95%区间 {uncertainty['fn_delta_95_percentile_interval']}，PQ差值95%区间 {uncertainty['pq_delta_95_percentile_interval']}。该区间不包含训练seed不确定性。"]
    else:
        appendix+=['独立检查未完整完成，不能据此替换既有基线。']
    appendix+=['','FN在此指原始标注实例没有任何预测达到IoU>0.5，包含定位/轮廓偏差，不能全部解释为模型完全未发现目标。小目标以原图面积<1000像素定义。']
    report=OUT/'report.md';text=report.read_text().split('\n## 图表与冻结检查结论')[0]
    text=text.replace('无置信区间。','无跨训练seed置信区间。')
    report.write_text(text+'\n'.join(appendix)+'\n')
    evidence={}
    paths=[OUT/'campaign.json',OUT/'adoption_rule.json',OUT/'adoption_rule_revision.json',
           OUT/'adaptive_extension.json',OUT/'resolution_extension.json',OUT/'closeout_budget.json',
           OUT/'frozen_selection.json',OUT/'results.json',OUT/'recommendation.json',OUT/'recall_candidate.json',
           OUT/'tests.log',OUT/'mps_attempt/diagnosis.md',
           OUT/'report.md',OUT/'comparison.png',Path('scripts/resume_driver.py'),Path('scripts/summarize_resume.py'),
           Path('src/m3_filament/native.py'),Path('src/m3_filament/model.py'),Path('src/m3_filament/evaluation.py'),
           Path('requirements-lock.txt')]
    for name in tuning:
        paths += [Path('runs')/name/'metrics.json',Path('runs')/name/'run_manifest.json',
                  Path(tuning[name]['artifact_output'])/'metrics.json']
    for data in check.values():
        paths += [Path(data['artifact_output'])/'metrics.json',Path(data['artifact_output'])/'per_annotation.jsonl']
    for p in paths:
        if p.exists(): evidence[str(p)]=sha256(p)
    write(OUT/'evidence-manifest.json',evidence)
    review=['# M3方法与实验审阅','', '状态：证据审阅草稿；研究决策仍需团队复核。无Git提交或发布。','',
            '## 1. 可证伪假设','',
            '增加原图训练观测可能降低样本不足造成的漏检；同训练预算下混合原图1024裁剪可能改善小目标监督尺度。若混合分支在相同24幅调参集和固定独立检查中没有更低FN，不能宣称裁剪有效。','',
            '## 2. 实际方法','',
            '原始RGB和逐标注实例mask→保持实例独立的全图/随机目标中心裁剪→1024 letterbox→YOLO11n-seg。损失与优化器沿用固定Ultralytics 8.3.221分割训练实现；本轮没有新增拓扑损失。推理使用完整原图，保留置信度阈值以上mask，重叠像素按优先级处理，输出原图坐标JSONL/RLE。裁剪是训练数据方法，不是完整SAHI切片推理。直觉是让细小实例在输入中占更多像素；长实例截断和分布变化是反作用。','',
            '## 3. 实现一致性与证据','',
            '| 证据 | 位置 | 约束/观察 |','|---|---|---|',
            '| E1 | src/m3_filament/native.py:NativeMaskDataset.__getitem__ | mask与图像同步裁剪/翻转，native RLE保留拓扑 |',
            '| E2 | configs/resume176_*_e6.yaml | 两分支训练集、seed与优化配置相同；crop概率不同 |',
            '| E3 | reports/resume_20261104/frozen_selection.json | 选参发生在预留标签导出/评分之前，PQ与FP约束明确 |',
            '| E4 | runs/resume_tune_*/metrics.json | 实测阈值网格、TP/FP/FN/PQ与小目标统计 |',
            '| E5 | runs/resume_check_*/metrics.json | 冻结参数一次比较；不再回流调参 |',
            '| E6 | reports/resume_20261104/evidence-manifest.json | 精确文件SHA256 |','',
            '## 4. 实验设计','',
            '176训练观测/289独立标注，24调参观测/46独立标注。预留24来自原calibration的未用观测/date/content组；旧check16和holdout103不参与。本轮单seed，检查集比较可做观测级配对bootstrap，但无法估计重训方差。best.pt先由内部验证fitness选取，再按外部PQ/FN选择阈值；结果存在开发选择偏差。旧32幅模型与新增176幅模型对比同时改变数据量与训练步数，不能把提升全部归因于某个新方法；global/mixedcrop才是裁剪对照。','',
            '## 5. 结果可信边界','',
            '数值见report.md与results.json，不引用旧8图分数比较新24图。失败/中断由research-guardian保留。源指标遵循官方Notebook v6的float32 IoU>0.5全配对和逐标注池化，并非Kaggle线上成绩。实际训练epochs、更新次数与权重哈希见报告。','',
            '## 6. 风险与下一证据','',
            '优先验证M1冻结split、M2正式评测和多seed；在团队划分到位前不开展生产训练。若裁剪无收益，不能继续以论文动机代替实测。完整切片推理、形态学与拓扑loss均不在本轮已验证方法中。','',
            '论文依据：[SAHI](https://arxiv.org/abs/2202.06934)，已阅读方法部分3节；仅借鉴全图/局部混合训练动机，不移植其性能结论。']
    (OUT/'methodology-review.md').write_text('\n'.join(review)+'\n')
    print(json.dumps({'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'plots':str(OUT/'comparison.png'),
                      'independent_check_complete':len(check)==2,'bootstrap':uncertainty},indent=2))


if __name__=='__main__': main()
