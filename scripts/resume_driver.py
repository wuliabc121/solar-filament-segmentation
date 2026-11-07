"""Serial, deadline-bound MPS comparison. Run with project run.sh from project root."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

from m3_filament.io import read_jsonl, write_json, write_jsonl, sha256
from m3_filament.queueing import run_wall_limited, choose_recall_candidate

ROOT = Path.cwd()
REPORT = ROOT/'reports/resume_20261104'
CAMPAIGN = json.loads((REPORT/'campaign.json').read_text())
DEADLINE = CAMPAIGN['deadline_utc']
LOGGER = str(Path.home()/'.agents/skills/research-guardian/scripts/experiment_logger.py')
SESSION = '01a1a144-84a8-7f81-8664-65c6a24ea02b'
GRID = [.001,.003,.005,.01,.02,.03,.05,.075,.1,.125,.15,.175,.2,.25,.3,.4,.5]
BASE = 'dev32_r1024_nbs2_e8'
GLOBAL = 'resume176_global_e6'
CROP = 'resume176_mixedcrop_e6'
FINETUNE = 'resume176_global_ft6'
STATE = {'pid':os.getpid(), 'deadline_utc':DEADLINE, 'jobs':[], 'status':'waiting_global'}


def now():
    return dt.datetime.now(dt.timezone.utc)


def remaining():
    return (dt.datetime.fromisoformat(DEADLINE.replace('Z','+00:00'))-now()).total_seconds()


def save(status):
    STATE.update(status=status, updated_utc=now().isoformat())
    tmp=REPORT/'state.tmp'
    write_json(tmp,STATE)
    tmp.replace(REPORT/'state.json')


def replace_json(path, payload):
    """Explicitly replace this campaign's evolving report, preserving io's immutable defaults."""
    path=Path(path)
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def registered_job(name, command, metrics, seconds=1500, config=None, checkpoints=()):
    if Path(metrics).exists():
        validate_reuse(command, metrics, config)
        STATE['jobs'].append({'name':name,'status':'existing_metrics','metrics':metrics})
        save('reused_'+name)
        return
    if remaining()<120:
        raise TimeoutError('insufficient remaining UTC budget for '+name)
    # Existing output without metrics is an interruption, never silently overwritten.
    if Path(metrics).parent.exists():
        raise RuntimeError('incomplete existing output: '+metrics)
    cmd=[sys.executable,LOGGER,'run','--project',str(ROOT),'--session-id',SESSION,
         '--model',name,'--dataset',
         (Path(command[command.index('--dataset')+1]).name+'-provisional' if '--dataset' in command else 'overnight176-provisional'),'--seed','42',
         '--config',config or 'reports/resume_20261104/campaign.json',
         '--metrics',metrics,'--timeout',str(min(seconds,remaining()-15)), '--grace','1']
    for c in checkpoints:
        cmd+=['--checkpoint',c]
    cmd+=['--param','campaign=resume_recall_20261104','--']+command
    save('running_'+name)
    result=run_wall_limited(cmd,REPORT/(name+'.log'),DEADLINE,seconds)
    STATE['jobs'].append({'name':name,**result,'metrics':metrics})
    save('finished_'+name)
    if result['returncode'] or not Path(metrics).exists():
        raise RuntimeError('registered job failed: '+name+' '+str(result))


def validate_reuse(command, metrics, config):
    """Refuse stale artifacts before resuming a partially completed queue."""
    measured=json.loads(Path(metrics).read_text())
    if '--weights' in command:
        weights=Path(command[command.index('--weights')+1])
        dataset=Path(command[command.index('--dataset')+1])
        thresholds=list(map(float,command[command.index('--thresholds')+1:]))
        observations=sorted({r['observation_id'] for r in read_jsonl(dataset/'samples.jsonl') if r['split']=='calibration'})
        valid=(measured['checkpoint_sha256']==sha256(weights)
               and measured['sample_manifest_sha256']==sha256(dataset/'samples.jsonl')
               and measured['observations']==observations
               and measured['validated_provenance']['inference_config']['imgsz']==int(command[command.index('--imgsz')+1])
               and measured['validated_provenance']['inference_config']['device']==command[command.index('--device')+1]
               and measured['validated_provenance']['inference_config']['max_det']==100
               and [r['confidence'] for r in measured['grid']]==thresholds
               and measured['split_hash']==json.loads((dataset/'preparation.json').read_text())['split_hash'])
    else:
        manifest=json.loads(Path(metrics).with_name('run_manifest.json').read_text())
        valid=(manifest['config_file_sha256']==sha256(config)
               and measured['checkpoint_sha256']==sha256(Path(metrics).parent/'weights/best.pt'))
    if not valid:
        raise RuntimeError('existing metrics provenance differs from requested job: '+metrics)


def score(run, output, dataset='data/overnight176', thresholds=GRID, imgsz=1024, seconds=900):
    output=output+'_cpu'
    metrics='runs/'+output+'/metrics.json'
    command=[str(ROOT/'run.sh'),'-m','m3_filament.evaluation','--weights',
             'runs/'+run+'/weights/best.pt','--dataset',dataset,'--output','runs/'+output,
             '--imgsz',str(imgsz),'--device','cpu','--thresholds']+list(map(str,thresholds))
    registered_job(output,command,metrics,seconds=seconds)
    return json.loads(Path(metrics).read_text())|{'artifact_output':'runs/'+output,'inference_imgsz':imgsz}


def candidates(results):
    return [row|{'run':data['validated_provenance']['training_run_id'],
                 'imgsz':data['inference_imgsz'],'candidate':name}
            for name,data in results.items() for row in data['grid']]


def reserve_export():
    """Evaluation-only original masks. Called only after selection is frozen."""
    from m3_filament.data import load_data, annotation_mask
    from m3_filament.masks import encode
    selected=json.loads(Path(CAMPAIGN['reserved_check24']).read_text())
    ids=set(selected['calibration'])
    split=json.loads(Path('data/overnight176/split_manifest.json').read_text())
    assert selected['source_split_hash']==split['split_hash']
    assert all(split['assignments'][o]=='calibration' for o in ids)
    tuning={r['observation_id'] for r in read_jsonl('data/overnight176/samples.jsonl')}
    assert not ids&tuning
    out=Path('data/resume_reserved24')
    if out.exists():
        raise FileExistsError('reserve export already exists; inspect rather than reevaluate')
    root,data=load_data('data/raw')
    annotations=defaultdict(list)
    for ann in data['annotations']:
        annotations[ann['image_id']].append(ann)
    rows=[]
    for image in data['images']:
        obs=Path(image['file_name']).stem
        if obs not in ids:
            continue
        target=(out/'targets'/(image['id']+'.json')).resolve()
        masks=[encode(annotation_mask(a,image['height'],image['width'])) for a in annotations[image['id']]]
        write_json(target,{'size':[image['height'],image['width']],'masks':masks})
        rows.append({'sample_id':image['id'],'annotation_id':image['id'],'observation_id':obs,
                     'split':'calibration','image_path':str((root/'train/train_images'/image['file_name']).resolve()),
                     'target_path':str(target)})
    assert {r['observation_id'] for r in rows}==ids
    write_jsonl(out/'samples.jsonl',rows)
    write_json(out/'preparation.json',{'split_hash':split['split_hash'],'status':'one_time_frozen_comparison',
               'selection_sha256':sha256(REPORT/'frozen_selection.json'),
               'reserve_manifest_sha256':sha256(CAMPAIGN['reserved_check24'])})
    return str(out)


def report(results,selected,check,error=None):
    payload={'campaign':CAMPAIGN,'state':STATE,'tuning':results,'frozen_selection':selected,
             'independent_check':check,'error':error,'official_score':False,'holdout103_used':False}
    replace_json(REPORT/'results.json',payload)
    lines=['# 降低漏检实验（续跑）','',
           '实际开始：2026-11-04 09:55（北京时间）；硬截止11:25。开发划分不是M1冻结划分，PQ为官方Notebook v6对齐的本地指标。',
           '', '## 24幅调参集（独立标注分别计分）','',
           '| 模型 | conf | TP | FP | FN | PQ | 小目标匹配/总数 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,data in results.items():
        best=data['selected']
        lines.append(f"| {name}（最高PQ） | {best['confidence']} | {best['tp']} | {best['fp']} | {best['fn']} | {best['pq']:.5f} | {best['small_matched']}/{best['small_gt']} |")
    if selected:
        lines+=['', '预先声明的召回选择：在PQ≥旧基线95%且≥网格最高PQ的90%、FP≤旧基线2倍的条件下最小化FN。',
                '', '冻结选择：`'+json.dumps(selected,ensure_ascii=False)+'`。']
    if check:
        lines+=['','## 冻结参数后的独立24幅检查','',
                '| 模型 | conf | TP | FP | FN | PQ | 小目标匹配/总数 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for name,data in check.items():
            row=data['selected']
            lines.append(f"| {name} | {row['confidence']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['pq']:.5f} | {row['small_matched']}/{row['small_gt']} |")
    if error:
        lines+=['','未完成项/失败：'+error]
    for failure in STATE.get('optional_failures',[]):
        lines+=['','可选扩展失败（未纳入选参）：'+failure['stage']+'；'+failure['error']]
    lines+=['','## 实际训练成本','', '| run | 实际epochs | optimizer更新 | 秒 | best.pt SHA256 |',
            '|---|---:|---:|---:|---|']
    for run in (BASE,GLOBAL,CROP,FINETUNE):
        p=Path('runs')/run/'metrics.json'
        if p.exists():
            m=json.loads(p.read_text())
            lines.append(f"| {run} | {m.get('actual_epochs')} | {m.get('optimizer_updates')} | {m['elapsed_seconds']:.1f} | {m['checkpoint_sha256']} |")
    lines+=['','## 方法与限制','',
            '全图与混合裁剪计划使用相同176幅训练图、289次独立标注、seed42、6 epochs、1024输入、batch2、nbs2；实际训练量见上表。混合分支以0.5概率从原始掩膜裁1024窗口。预算内训练不代表收敛。训练best checkpoint仍由Ultralytics内部验证fitness选择；跨模型与置信度由本报告PQ/FN约束选择。',
            '', '裁剪动机参考[SAHI原论文](https://arxiv.org/abs/2202.06934)，本轮没有实现其完整切片推理，不宣称复现论文。原始mask孔洞/连通结构保留；裁剪会截断长目标。',
            '', '单seed、开发样本有限，无置信区间。原103幅holdout及比赛test未用于选参。保留原始日志、权重与实验runner记录，未提交比赛或发布。']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    results={};selected=None;check={};error=None
    save('waiting_global_training')
    try:
        record=Path('.research-guardian/experiments/EXP-20261104T015559-ed9fb1e2418d/record.json')
        while not Path('runs/'+GLOBAL+'/metrics.json').exists():
            state=json.loads(record.read_text())
            if state.get('status') not in ('running','starting'):
                raise RuntimeError('first global experiment ended without metrics: '+str(state.get('status')))
            if remaining()<120:
                raise TimeoutError('UTC deadline while waiting for global training')
            time.sleep(5)
        for run in (BASE,GLOBAL):
            results[run]=score(run,'resume_tune_'+run)
        command=[str(ROOT/'run.sh'),'-m','m3_filament','train','--config','configs/'+CROP+'.yaml',
                 '--run-id',CROP,'--development']
        registered_job(CROP,command,'runs/'+CROP+'/metrics.json',2100,'configs/'+CROP+'.yaml',
                       ['runs/'+CROP+'/weights/best.pt','runs/'+CROP+'/weights/last.pt'])
        results[CROP]=score(CROP,'resume_tune_'+CROP)
        # Inference resolution directly tests the thin-target downsampling hypothesis.
        # Use tuning observations only; reserve remains locked throughout.
        if remaining()>2100:
            for run in (BASE,GLOBAL,CROP):
                if remaining()<900:
                    break
                key=run+'_r1536'
                try:
                    results[key]=score(run,'resume_tune_'+key,imgsz=1536,seconds=min(450,remaining()-600))
                except (RuntimeError,TimeoutError) as exc:
                    STATE.setdefault('optional_failures',[]).append({'stage':key,'error':repr(exc)})
                    save('optional_resolution_failed')
        baseline=next(r for r in results[BASE]['grid'] if r['confidence']==.2)|{'run':BASE,'imgsz':1024,'candidate':BASE}
        rows=candidates(results)
        selected=choose_recall_candidate(rows,baseline)
        extension_needed=(not selected['selection_feasible'] or selected['fn']>.85*baseline['fn'])
        if extension_needed and remaining()>1800:
            command=[str(ROOT/'run.sh'),'-m','m3_filament','train','--config','configs/'+FINETUNE+'.yaml',
                     '--run-id',FINETUNE,'--development']
            try:
                registered_job(FINETUNE,command,'runs/'+FINETUNE+'/metrics.json',min(1200,remaining()-600),'configs/'+FINETUNE+'.yaml',
                               ['runs/'+FINETUNE+'/weights/best.pt','runs/'+FINETUNE+'/weights/last.pt'])
                results[FINETUNE]=score(FINETUNE,'resume_tune_'+FINETUNE,seconds=min(450,remaining()-300))
            except (RuntimeError,TimeoutError) as exc:
                STATE.setdefault('optional_failures',[]).append({'stage':FINETUNE,'error':repr(exc)})
                save('optional_finetune_failed')
            rows=candidates(results)
            selected=choose_recall_candidate(rows,baseline)
        STATE['adaptive_extension']={'triggered':extension_needed,'executed':FINETUNE in results,
                                     'remaining_seconds':remaining()}
        write_json(REPORT/'frozen_selection.json',{'selected':selected,'frozen_utc':now().isoformat(),
                   'tuning_metrics_sha256':{run:sha256(Path(results[run]['artifact_output'])/'metrics.json') for run in results}})
        save('selection_frozen')
        report(results,selected,check)
        if remaining()>300:
            dataset=reserve_export()
            check['baseline_conf0.2']=score(BASE,'resume_check_baseline',dataset,[.2])
            check['selected_frozen']=score(selected['run'],'resume_check_selected',dataset,[selected['confidence']],selected['imgsz'])
        save('completed' if len(check)==2 else 'completed_without_independent_check')
    except Exception as exc:
        error=repr(exc)
        STATE['error']=error
        save('failed_or_budget_stopped')
    finally:
        report(results,selected,check,error)
        files=[*Path('src').rglob('*.py'),*Path('configs').glob('resume*.yaml'),Path('scripts/resume_driver.py')]
        replace_json(REPORT/'source-manifest.json',{str(p):sha256(p) for p in files})
        print(json.dumps(STATE,indent=2),flush=True)
    if error:
        raise SystemExit(1)


if __name__=='__main__':
    main()
