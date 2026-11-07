"""Evidence and visual summary for the fixed joint-metric campaign, no model rerun."""
import csv
import datetime as dt
import json
import platform
from importlib.metadata import version
from pathlib import Path
import shlex
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from m3_filament.io import read_jsonl,sha256
from m3_filament.joint import QUALITY,derive_metrics,choose_joint_candidate
from m3_filament.data import observation_groups
from m3_filament.model import source_manifest_hash

OUT=Path('reports/balanced_20261104')
PQ_TARGET=.35
COUNTS=('tp','fp','fn','tp_iou_sum','n_gt','n_pred','small_gt','small_matched')


def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def pipeline_score_floor(candidate):
    return .05 if candidate['min_area'] or candidate['closing_radius'] else candidate['confidence']


def vector_metrics(a):
    tp,fp,fn,total,gt,pred,sg,sm=a.T
    def ratio(x,y):return np.divide(x,y,out=np.zeros_like(x,dtype=float),where=y>0)
    denom=tp+.5*(fp+fn);p=ratio(pred-fp,pred);r=ratio(gt-fn,gt)
    small=np.divide(sm,sg,out=np.full_like(sm,np.nan,dtype=float),where=sg>0)
    return {'pq':ratio(total,denom),'sq':ratio(total,tp),'rq':ratio(tp,denom),
            'precision':p,'recall':r,'f1':ratio(2*p*r,p+r),'small_recall':small,'fp':fp,'fn':fn}


def bootstrap_comparison(base_rows,candidate_rows,groups,draws=5000):
    signature=lambda rows:{(r['observation_id'],r['annotation_id'],r['n_gt'],r['small_gt']) for r in rows}
    if signature(base_rows)!=signature(candidate_rows):raise ValueError('unpaired annotation records')
    observed={r['observation_id'] for r in base_rows}
    if not observed<=groups.keys():raise ValueError('missing observation groups')
    units=sorted({groups[o] for o in observed});index={g:i for i,g in enumerate(units)}
    arrays=[]
    for rows in (base_rows,candidate_rows):
        a=np.zeros((len(units),len(COUNTS)))
        for row in rows:a[index[groups[row['observation_id']]]]+=np.array([row[k] for k in COUNTS])
        arrays.append(a)
    rng=np.random.default_rng(42);samples=rng.integers(0,len(units),size=(draws,len(units)))
    b=vector_metrics(arrays[0][samples].sum(axis=1));c=vector_metrics(arrays[1][samples].sum(axis=1))
    intervals={k:np.nanquantile(c[k]-b[k],[.025,.975]).tolist() for k in b}
    return {'independent_groups':len(units),'observations':len(observed),'draws':draws,'seed':42,
            'unit':'complete date/content connected group, all annotation passes together','intervals':intervals,
            'scope':'fixed checkpoint comparison; excludes training seed variance'}


def table_row(name,row):
    m=derive_metrics(row)
    return '| '+name+' | '+' | '.join('N/A' if m[k] is None else f'{m[k]:.5f}' for k in QUALITY)+f" | {m['fp']} | {m['fn']} |"


def main():
    r=json.loads((OUT/'results.json').read_text());scores=r['tuning'];selection=r['selection'];checks=r['new_check']
    rows=[]
    for alias,data in scores.items():
        for point in data['grid']:
            rows.append(derive_metrics(point)|{'alias':alias,'run':data['model_run'],'imgsz':data['imgsz'],
                 'min_area':data['min_area'],'closing_radius':data['closing_radius']})
    fields=['alias','run','imgsz','confidence','min_area','closing_radius',*QUALITY,'tp','fp','fn','n_gt','small_gt','small_matched']
    with (OUT/'joint_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    refs=json.loads((OUT/'references.json').read_text())['references']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row in rows:
        choice=choose_joint_candidate([row],refs)
        strict_point=row['pq']>PQ_TARGET+1e-12 and all(row[k] is not None and row[k]>bound+1e-12 for k,bound in choice['quality_lower_bounds'].items()) and all(row[k]<bound for k,bound in choice['count_upper_bounds'].items())
        color='tab:green' if strict_point else '.72'
        for ax,x,y in [(axes[0,0],'fp','fn'),(axes[0,1],'precision','recall'),(axes[1,0],'pq','small_recall')]:
            ax.scatter(row[x],row[y],s=12,c=color,alpha=.65)
    for label,point,color in [('Old baseline',refs[0],'tab:blue'),('Previous recall',refs[1],'tab:orange')]:
        m=derive_metrics(point)
        for ax,x,y in [(axes[0,0],'fp','fn'),(axes[0,1],'precision','recall'),(axes[1,0],'pq','small_recall')]:
            ax.scatter(m[x],m[y],s=80,c=color,marker='X',label=label)
    if selection:
        point=selection['candidate']
        for ax,x,y in [(axes[0,0],'fp','fn'),(axes[0,1],'precision','recall'),(axes[1,0],'pq','small_recall')]:
            ax.scatter(point[x],point[y],s=100,c='purple',marker='*',label='Frozen candidate')
    axes[0,0].set(xlabel='FP',ylabel='FN',title='Lower-left is better')
    axes[0,1].set(xlabel='Unique-instance precision',ylabel='Unique-instance recall')
    axes[1,0].set(xlabel='Development PQ',ylabel='Small-instance recall')
    axes[1,0].axvline(PQ_TARGET,color='red',linestyle='--',linewidth=1,label='PQ target 0.35')
    for run in ['resume176_mixedcrop_e6','balanced176_ft8','balanced496_ft4','balanced_maskmetric_ft4']:
        p=Path('runs')/run/'results.csv'
        if p.exists():
            with p.open() as f:history=list(csv.DictReader(f))
            axes[1,1].plot([int(float(x['epoch'])) for x in history],[float(x['metrics/mAP50(M)']) for x in history],'.-',label=run)
    axes[1,1].set(xlabel='Epoch in this run',ylabel='Internal mask mAP50 (same 24 validation observations)')
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Tuning: green passes full joint gate; grey does not. Not competition test.')
    fig.tight_layout();fig.savefig(OUT/'joint_tuning.png',dpi=180);plt.close(fig)
    final=r.get('final_joint_gate');strict=False;benchmark_met=False;bootstrap={}
    if len(checks)==3:
        values={name:derive_metrics(data['selected']) for name,data in checks.items()}
        candidate=values['new_candidate'];reference_values=[values['old_baseline'],values['previous_recall_candidate']]
        strict=all(candidate[k] is not None and candidate[k]>max(x[k] for x in reference_values if x[k] is not None)+1e-12 for k in QUALITY)
        strict &= all(candidate[k]<min(x[k] for x in reference_values) for k in ('fp','fn'))
        benchmark_met=candidate['pq']>PQ_TARGET+1e-12
        audit=[x for x in read_jsonl('reports/audit_001/data_manifest.jsonl') if x['partition']=='train']
        groups={o:min(g) for g in observation_groups(audit) for o in g}
        candidate_rows=list(read_jsonl(Path(checks['new_candidate']['metrics_path']).with_name('per_annotation.jsonl')))
        for key in ['old_baseline','previous_recall_candidate']:
            base=list(read_jsonl(Path(checks[key]['metrics_path']).with_name('per_annotation.jsonl')))
            bootstrap[key]=bootstrap_comparison(base,candidate_rows,groups)
        dump(OUT/'bootstrap.json',bootstrap)
        fig,axes=plt.subplots(3,3,figsize=(12,10));names=list(values)
        for ax,metric in zip(axes.flat,[*QUALITY,'fp','fn']):
            numbers=[values[n][metric] for n in names]
            bars=ax.bar(['Old','Prev recall','New'],numbers,color=['tab:blue','tab:orange','purple'])
            ax.set_title(metric+(' (lower better)' if metric in ('fp','fn') else ' (higher better)'))
            if metric=='pq':ax.axhline(PQ_TARGET,color='red',linestyle='--',linewidth=1,label='target 0.35');ax.legend(fontsize=7)
            for bar,value in zip(bars,numbers):ax.annotate(str(int(value)) if metric in ('fp','fn') else f'{value:.3f}',
                (bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',va='bottom',fontsize=8)
            ax.margins(y=.18);ax.grid(axis='y',alpha=.2)
        fig.suptitle('New fixed 24-observation check; no parameter changes after evaluation')
        fig.tight_layout();fig.savefig(OUT/'new_check.png',dpi=180);plt.close(fig)
    success=bool(strict and benchmark_met)
    goal=('target_met_all_metrics_strictly_improved' if success else 'metrics_improved_below_pq_target' if strict else 'not_met' if final else 'not_evaluated')
    profile=None
    pipeline_meta={'source_manifest_sha256':source_manifest_hash(),
       'requirements_lock_sha256':sha256('requirements-lock.txt'),
       'inference_script_sha256':sha256('scripts/predict_joint_profile.py'),
       'runtime_versions':{'python':platform.python_version(),**{name:version(name) for name in
           ('torch','torchvision','ultralytics','numpy','pycocotools','opencv-python')}}}
    if selection:
        c=selection['candidate'];weights=Path('runs')/c['run']/'weights/best.pt'
        profile={'weights':str(weights),'checkpoint_sha256':sha256(weights),'run':c['run'],'imgsz':c['imgsz'],
             'conf':c['confidence'],'min_area':c['min_area'],'closing_radius':c['closing_radius'],
             'score_floor':pipeline_score_floor(c),'device':'cpu','max_det':100,'status':goal,
             'joint_gate_passed_on_new_check':bool(final and final['joint_gate_passed']),
             'all_metrics_strictly_improved_on_new_check':bool(strict),
             'pq_target':PQ_TARGET,'pq_target_met_on_new_check':bool(benchmark_met)}|pipeline_meta
        dump(OUT/'candidate_profile.json',profile)
        default=profile if success else {'weights':'runs/dev32_r1024_nbs2_e8/weights/best.pt','checkpoint_sha256':
             sha256('runs/dev32_r1024_nbs2_e8/weights/best.pt'),'run':'dev32_r1024_nbs2_e8','imgsz':1024,'conf':.2,
             'min_area':0,'closing_radius':0,'score_floor':.2,'device':'cpu','max_det':100,'status':'retained_default'}|pipeline_meta
        dump(OUT/'recommended_profile.json',default)
    dump(OUT/'decision.json',{'goal_status':goal,'all_metrics_strictly_improved':bool(strict),
          'joint_gate_passed':bool(final and final['joint_gate_passed']),'strict_default_adoption':success,
          'pq_target':PQ_TARGET,'pq_target_met':bool(benchmark_met),
          'failed_constraints':final['violations'] if final else ['new check incomplete'],
          'candidate':profile,'original_holdout103_used':False,'competition_test_used':False})
    lines=['# 结果验收与使用','', '最终状态：`'+goal+'`。只有新检查PQ>0.35且其余各项严格改善才更新本轮推荐默认；否则保留既有配置，并逐项说明差距。',
       '', '## 新检查指标','', '| 方案 | '+ ' | '.join(QUALITY)+' | FP | FN |',
       '|---|'+'---:|'*9]
    for name,data in checks.items():lines.append(table_row(name,data['selected']))
    if final:lines+=['','退步/不满足项：'+str(final['violations'])]
    lines+=['', '官网公告所述有价值水平是PQ>0.35，不是3.5；本轮将其作为本地开发验收目标。来源：[主办方公告](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements)。',
       '', '主实验报告见report.md；全部阈值点见joint_metrics.csv。各GT标注独立计分，Precision/Recall使用唯一对象数量，SQ/RQ/PQ沿用官方Notebook v6匹配；不能把这些称为线上成绩。',
       '', '![joint tuning](joint_tuning.png)']
    if (OUT/'new_check.png').exists():lines+=['','![new check](new_check.png)']
    if profile:lines += ['', '使用完整冻结参数对新的原图目录预测（无需GT标签，输出目录须不存在）：','', '```sh',
       './run.sh scripts/predict_joint_profile.py --profile reports/balanced_20261104/candidate_profile.json --images /新原图目录完整路径 --output runs/new_joint_prediction',
       '```','', 'candidate_profile是被评估候选，不一定通过目标；recommended_profile只在严格逐项改善时采用新候选。推理会校验权重哈希、训练观测隔离，并导出原图JSONL/RLE与M4 crop元数据。']
    lines+=['','分组配对bootstrap见bootstrap.json，保留日期/内容连接组的全部图像和标注；不包含训练seed不确定性。训练日志mAP仅作同24幅内部验证的辅助诊断，不代替固定操作点和最终PQ联合验收。',
       '', '低分分解及像素IoU/Dice退步见[诊断](low-pq-diagnosis.md)；剩余预算内的切片方法仅用原调参集，实际为负结果，见[切片原型](experimental_slicing_report.md)。后者没有改变冻结候选或新检查结论。最终只读复核见review-final.md。']
    (OUT/'decision.md').write_text('\n'.join(lines)+'\n')
    ledger=[]
    for p in Path('.research-guardian/experiments').glob('*/record.json'):
        item=json.loads(p.read_text())
        if item.get('model','').startswith(('balanced','joint_','experimental_sliced')):
            ledger.append({k:item.get(k) for k in ['id','model','status','started_at','ended_at','duration_seconds','exit_code','metrics_state','command']}
                          |{'record':str(p),'record_sha256':sha256(p)})
    ledger.sort(key=lambda x:x['started_at']);dump(OUT/'experiment-ledger.json',ledger)
    commands=['# 实际命令','', '历史命令不能直接复用已有输出；新训练需新run_id和新UTC预算。新检查一经使用不得回流选参。']
    for item in ledger:commands+=['',item['id']+' '+item['status'],'','```sh',shlex.join(item['command']),'```']
    commands+=['','后处理分析不会重新训练或推理：','', '```sh','./run.sh scripts/summarize_balanced.py','./run.sh -m pytest -q','```']
    (OUT/'commands.md').write_text('\n'.join(commands)+'\n')
    review=['# 综合指标方法与证据审阅','',
      '## 假设与反证','', '继续训练/更广训练观测/预测清理能同时降低FP与FN并提高PQ及实例质量。任一固定主指标退步即不能称全面改善；没有用单一加权均分遮盖负面结果。',
      '', '## 实际方法与目标','', '训练仍是固定YOLO11n-seg原始实例mask监督与半概率1024目标中心裁剪，AdamW低学习率热启动；没有新增loss或语义网络。预测输入1024/1536/2048、置信度网格、可选原图min_area和二值closing。预测closing可能改变孔洞；它不改变训练标签，也没有被当成逐像素概率。',
      '', '质量向量q=(PQ,SQ,RQ,P,R,F1,SmallRecall)，每项都必须严格高于两个固定参照的最大值，FP/FN必须严格低于两个参照最小值。所有严格约束通过才按PQ选优；不通过时最差归一化差距只选诊断候选。持平不称提高。',
      '', '## 实现与证据','', 'E1 src/m3_filament/joint.py：口径、联合支配、退步项。E2 postprocess.py：只处理预测二值mask、来源链、RLE和M4重编号。E3 scripts/balanced_driver.py：UTC预算、固定参照、共享部分结果、冻结顺序。E4 new_check24.json/analysis_plan.json/training_budget_revision.json：预声明的组隔离与预算修订。E5 results.json/experiment-ledger.json：真实指标及成功/失败；evidence-manifest.json锁定SHA256。',
      '', '## 实验设计','', '本轮训练仅原train组；第一分支176观测8轮，第二分支496观测计划4轮，实际数量见metrics，训练量不同不能当等计算消融。两者从相同旧crop模型热启动且新建优化器。统一24幅调参图逐标注计分；新24检查在所有参数冻结后评两个参照和一个候选。旧reserve24/check16不参与本轮参数选择，原holdout103/test不触碰。',
      '', '## 可信边界','', '单seed，有限日期/内容组；配对bootstrap反映固定模型在本次组上的不确定性，不证明重训稳定。A/B best.pt由Ultralytics内部fitness选择，C按mask mAP50-95选择，均可能与外部PQ不完全一致。A/B/C均正常完成，没有使用中断恢复权重。M1正式split和M2正式入口仍缺失，Notebook v6对齐指标不是排行榜。',
      '', '## 结论与风险','', '严格目标状态：'+goal+'。若未达标必须保留差距，不以完成实验代替成功改善。没有发布、比赛提交或Git提交推送。GT原始孔洞和独立标注保持，真实CNN/Transformer缓存未提供，不能归因于E5融合。']
    review+=['','## 条件checkpoint选择对照','',
       ('已执行' if Path('runs/balanced_maskmetric_ft4').exists() else '未触发')+'：当A/B无严格联合解且剩余>30分钟时，允许从A模型低学习率接续4轮，并按mask mAP50-95选择checkpoint。该外部trainer只在新进程中注入，不改变B训练或原始loss。外部脚本SHA和快照、selection_trace与模型manifest共同保留；不能把接续训练加选择规则的组合效果单独归因于选择规则。']
    review+=['','## 切片原型与辅助诊断','',
       '主实验冻结检查完成后，使用剩余预算在旧调参24幅测试SAHI启发原型；不是新检查补试。原论文与作者代码见experimental_slicing_report.md，mask IoS传递并集不是作者box NMM忠实复现。三个合并阈值均未提高调参PQ，未采用。辅助像素IoU/Dice和最终mask覆盖分桶见low-pq-diagnosis.md/auxiliary_metrics.json，仅作解释，未参与选参。复核见review-final.md。']
    (OUT/'methodology-review.md').write_text('\n'.join(review)+'\n')
    paths=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.md','.png','.csv','.log') and p.name not in ('evidence-manifest.json','summary.log','summary-stderr.log')]
    paths += [Path('README.md'),Path('requirements-lock.txt'),Path('environment-explicit.txt')]
    for directory in ['src','tests','configs','scripts']:
        paths += [p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.sh')]
    for data in list(scores.values())+list(checks.values()):
        p=Path(data['metrics_path']);paths += [p,p.with_name('per_annotation.jsonl')]
    for run in ['balanced176_ft8','balanced496_ft4','balanced_maskmetric_ft4']:
        folder=Path('runs')/run
        paths += [folder/p for p in ['metrics.json','run_manifest.json','recovery.json','code_snapshot/source_manifest.json',
                  'external_training.json','selection_trace.json','code_snapshot/external_source_manifest.json',
                  'code_snapshot/external/train_mask_metric.py'] if (folder/p).exists()]
        for p in (folder/'weights').glob('*.pt'):paths.append(p)
    for p in [Path('data/balanced496/preparation.json'),Path('data/balanced496/samples.jsonl'),Path('data/balanced_check24/preparation.json'),Path('data/balanced_check24/samples.jsonl')]:
        if p.exists():paths.append(p)
    dump(OUT/'evidence-manifest.json',{str(p):sha256(p) for p in sorted(set(paths)) if p.exists()})
    print(json.dumps({'goal_status':goal,'joint_gate_passed':bool(final and final['joint_gate_passed']),
            'strict_all_improved':bool(strict),'pq_target_met':bool(benchmark_met),'tuning_rows':len(rows),'new_check_cases':len(checks)},indent=2))


if __name__=='__main__':main()
