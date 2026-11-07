"""Two-hour joint-metric campaign; only completed measured candidates may be selected."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import yaml
from collections import defaultdict
from m3_filament.io import read_jsonl,write_json,write_jsonl,sha256,content_hash
from m3_filament.joint import QUALITY,derive_metrics,choose_joint_candidate
from m3_filament.queueing import run_wall_limited
from m3_filament.model import source_manifest_hash

ROOT=Path.cwd();OUT=ROOT/'reports/balanced_20261104'
CAMPAIGN=json.loads((OUT/'campaign.json').read_text());DEADLINE=CAMPAIGN['deadline_utc']
SESSION='01a1a144-84a8-7f81-8664-65c6a24ea02b'
LOGGER=str(Path.home()/'.agents/skills/research-guardian/scripts/experiment_logger.py')
GRID=json.loads((OUT/'analysis_plan.json').read_text())['tuning_thresholds']
BASE='dev32_r1024_nbs2_e8';PREV='resume176_mixedcrop_e6'
FIRST='balanced176_ft8';SECOND='balanced496_ft4'
MASK_METRIC='balanced_maskmetric_ft4'
SCORE_LIMIT=1800
PQ_TARGET=.35
STATE={'pid':os.getpid(),'status':'starting','deadline_utc':DEADLINE,'jobs':[],'failures':[],'score_seconds':0.}


def now():return dt.datetime.now(dt.timezone.utc)
def remaining():return (dt.datetime.fromisoformat(DEADLINE)-now()).total_seconds()
def replace_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def save(status):
    STATE.update(status=status,updated_utc=now().isoformat());replace_json(OUT/'state.json',STATE)


def restore_score_accounting():
    jobs=[]
    for path in (OUT/'jobs').glob('*.json'):
        record=json.loads(path.read_text())
        if 'elapsed_seconds' in record:jobs.append(record)
    return {'jobs':jobs,'score_seconds':sum(j['elapsed_seconds'] for j in jobs if j['kind']=='score')}


def choose_for_user(rows,references):
    refs=[derive_metrics(r) for r in references]
    lower={k:max(r[k] for r in refs if r[k] is not None) for k in QUALITY
           if any(r[k] is not None for r in refs)}
    lower['pq']=max(lower['pq'],PQ_TARGET)
    upper={k:min(r[k] for r in refs) for k in ('fp','fn')}
    def missing(raw):
        r=derive_metrics(raw)
        return [k for k,bound in lower.items() if r[k] is None or r[k]<=bound+1e-12]+[k for k,bound in upper.items() if r[k]>=bound]
    def margin(raw):
        r=derive_metrics(raw)
        return min([((r[k]-bound)/max(bound,.05) if r[k] is not None else -1e12) for k,bound in lower.items()]+
                   [(bound-r[k])/max(bound,1) for k,bound in upper.items()])
    strict=[r for r in rows if not missing(r)]
    diagnostic=max(rows,key=lambda r:(margin(r),r['pq'])) if not strict else None
    choice=choose_joint_candidate(strict or [diagnostic],references)
    choice.update(joint_gate_passed=bool(strict),all_metrics_strictly_improved=bool(strict),
                  feasible_candidates=len(strict),violations=[] if strict else missing(choice['candidate']),
                  quality_lower_bounds=lower,count_upper_bounds=upper,pq_target=PQ_TARGET,
                  pq_target_met=choice['candidate']['pq']>PQ_TARGET+1e-12,
                  candidates_measured=len(rows),worst_normalized_margin=margin(choice['candidate']),
                  selection_policy='PQ>0.35 and every quality metric strictly higher, FP/FN strictly lower than both references; PQ ranks feasible')
    return choice

def runner_record(name):
    records=[]
    for p in (ROOT/'.research-guardian/experiments').glob('*/record.json'):
        r=json.loads(p.read_text())
        if r.get('model')==name:records.append((r['started_at'],p,r))
    return max(records) if records else None


def run_job(name,command,metrics,seconds=420,config=None,checkpoints=(),kind='score',final=False):
    metric=Path(metrics);jobfile=OUT/'jobs'/(name+'.json')
    inputs={}
    for flag in ('--weights','--instances','--dataset'):
        if flag not in command:continue
        path=Path(command[command.index(flag)+1])
        if flag=='--dataset':
            inputs[str(path/'samples.jsonl')]=sha256(path/'samples.jsonl')
            inputs[str(path/'preparation.json')]=sha256(path/'preparation.json')
        else:inputs[str(path)]=sha256(path)
    for token in command:
        if token.endswith('.py') and Path(token).is_file():inputs[token]=sha256(token)
    if kind=='training':
        training_config=yaml.safe_load(Path(config).read_text())
        weights=Path(training_config['model']);data_dir=Path(training_config['data']).parent
        inputs[str(weights)]=sha256(weights)
        for filename in ('preparation.json','samples.jsonl'):
            inputs[str(data_dir/filename)]=sha256(data_dir/filename)
    specification={'command':command,'config_sha256':sha256(config or OUT/'campaign.json'),
                   'source_manifest_sha256':source_manifest_hash(),'input_hashes':inputs,
                   'requirements_lock_sha256':sha256('requirements-lock.txt')}
    key=content_hash(specification)
    if jobfile.exists():
        old=json.loads(jobfile.read_text())
        if old['specification_sha256']!=key:raise RuntimeError('job source mismatch '+name)
        if metric.exists() and old.get('returncode')==0:return
        active=runner_record(name)
        if metric.exists() and active and active[2]['status']!='running':
            previous=active[2]
            if previous['status']=='success':
                evidence=previous.get('metrics_evidence',{})
                if evidence.get('sha256')!=sha256(metric):raise RuntimeError('runner metric hash mismatch '+name)
            elif not json.loads(metric.read_text()).get('recovered_for_inference'):
                raise RuntimeError('failed run has unverified metrics '+name)
            old.update(returncode=previous.get('exit_code'),elapsed_seconds=previous['duration_seconds'],
                       recovered_wrapper_metadata=True,original_runner_status=previous['status'])
            replace_json(jobfile,old);STATE['jobs'].append(old)
            if kind=='score':STATE['score_seconds']+=old['elapsed_seconds']
            if previous['status']!='success':STATE['failures'].append({'stage':name,'error':'interrupted training checkpoint recovered, not a successful run'})
            return
        if active and active[2]['status']=='running':
            save('waiting_existing_'+name)
            while remaining()>30:
                r=json.loads(active[1].read_text())
                if r['status']!='running':break
                time.sleep(2)
            if metric.exists():return
        raise RuntimeError('existing incomplete job requires inspection: '+name)
    if remaining()<120:raise TimeoutError('insufficient UTC budget '+name)
    if metric.parent.exists():raise RuntimeError('refusing incomplete output '+str(metric.parent))
    if kind=='score':
        pool=SCORE_LIMIT-STATE['score_seconds']-(0 if final else 300)
        if pool<30:raise TimeoutError('reserved score budget '+name)
        seconds=min(seconds,pool)
    seconds=min(seconds,remaining()-(20 if final else 330))
    if seconds<10:raise TimeoutError('reserved UTC closeout budget '+name)
    record={'name':name,'specification_sha256':key,**specification,'kind':kind,'metrics':metrics,'started_utc':now().isoformat()}
    replace_json(jobfile,record)
    cmd=[sys.executable,LOGGER,'run','--project',str(ROOT),'--session-id',SESSION,'--model',name,
         '--dataset',('balanced-check24' if final else 'balanced-provisional'),'--seed','42',
         '--config',str(config or OUT/'campaign.json'),'--metrics',metrics,'--timeout',str(seconds),'--grace','1']
    for checkpoint in checkpoints:cmd+=['--checkpoint',checkpoint]
    cmd+=['--param','objective=joint_metrics','--']+command
    save('running_'+name)
    result=run_wall_limited(cmd,OUT/(name+'.log'),DEADLINE,seconds)
    record.update(result);replace_json(jobfile,record);STATE['jobs'].append(record)
    if kind=='score':STATE['score_seconds']+=result['elapsed_seconds']
    save('finished_'+name)
    if result['returncode']!=0 or not metric.exists():raise RuntimeError('failed '+name+': '+str(result['returncode']))


def load_score(path,alias,minimum_area=0,closing_radius=0,raw_cache=None,expect_tuning=True):
    path=Path(path);data=json.loads(path.read_text());config=data['validated_provenance']['inference_config']
    if expect_tuning and data['observations']!=TUNING_OBS:
        raise ValueError('tuning observations differ')
    return data|{'alias':alias,'metrics_path':str(path),'model_run':data['validated_provenance']['training_run_id'],
          'imgsz':config['imgsz'],'min_area':minimum_area,'closing_radius':closing_radius,
          'raw_cache':str(raw_cache or path.parent.with_name(path.parent.name+'_predictions')/'instances.jsonl')}


def candidate_rows(scores):
    return [row|{'alias':alias,'run':data['model_run'],'imgsz':data['imgsz'],
            'min_area':data['min_area'],'closing_radius':data['closing_radius'],'metrics_path':data['metrics_path']}
            for alias,data in scores.items() for row in data['grid']]


def score_detector(run,size,output,dataset='data/overnight176',thresholds=GRID,final=False):
    metrics='runs/'+output+'/metrics.json'
    command=[str(ROOT/'run.sh'),'-m','m3_filament.evaluation','--weights','runs/'+run+'/weights/best.pt',
       '--dataset',dataset,'--output','runs/'+output,'--imgsz',str(size),'--device','cpu','--thresholds']+list(map(str,thresholds))
    run_job(output,command,metrics,seconds=420,final=final)
    return load_score(metrics,output,expect_tuning=not final)


def score_post(parent,area,radius,output,dataset='data/overnight176',thresholds=GRID,final=False):
    metrics='runs/'+output+'/metrics.json'
    command=[str(ROOT/'run.sh'),'-m','m3_filament.postprocess','--instances',parent['raw_cache'],
       '--dataset',dataset,'--output','runs/'+output,'--min-area',str(area),'--closing-radius',str(radius),
       '--score-floor','.05','--thresholds']+list(map(str,thresholds))
    run_job(output,command,metrics,seconds=180,final=final)
    return load_score(metrics,output,area,radius,parent['raw_cache'],expect_tuning=not final)


def optional_post(scores,parent,prefix,max_profiles=7):
    profiles=[(32,0),(64,0),(128,0),(0,1),(32,1),(64,1),(128,1)][:max_profiles]
    for area,radius in profiles:
        if any(d['min_area']==area and d['closing_radius']==radius and
               Path(d['raw_cache']).resolve()==Path(parent['raw_cache']).resolve() for d in scores.values()):
            continue
        if remaining()<900 or STATE['score_seconds']>SCORE_LIMIT-330:break
        name=prefix+'_a'+str(area)+'_c'+str(radius)
        try:scores[name]=score_post(parent,area,radius,name)
        except (RuntimeError,TimeoutError) as exc:
            STATE['failures'].append({'stage':name,'error':repr(exc)});save('optional_post_failed')
        persist_partial(scores)


def persist_partial(scores):
    choice=choose_for_user(candidate_rows(scores),REFERENCES)
    replace_json(OUT/'tuning_progress.json',{'choice':choice,'variants':list(scores),'score_seconds':STATE['score_seconds']})


def export_check():
    from m3_filament.data import load_data,annotation_mask
    from m3_filament.masks import encode
    declaration=json.loads((OUT/'new_check24.json').read_text());ids=set(declaration['calibration'])
    split=json.loads(Path('data/overnight176/split_manifest.json').read_text())
    assert declaration['source_split_hash']==split['split_hash']
    assert not ids&set(declaration['used_ids_excluded'])
    assert all(split['assignments'][o]=='calibration' for o in ids)
    out=Path('data/balanced_check24')
    if out.exists():
        preparation=json.loads((out/'preparation.json').read_text())
        if preparation['frozen_selection_sha256']!=sha256(OUT/'frozen_selection.json'):
            raise RuntimeError('frozen check provenance mismatch')
        return str(out)
    root,data=load_data('data/raw');annotations=defaultdict(list)
    for a in data['annotations']:annotations[a['image_id']].append(a)
    rows=[]
    for image in data['images']:
        obs=Path(image['file_name']).stem
        if obs not in ids:continue
        target=(out/'targets'/(image['id']+'.json')).resolve()
        masks=[encode(annotation_mask(a,image['height'],image['width'])) for a in annotations[image['id']]]
        write_json(target,{'size':[image['height'],image['width']],'masks':masks})
        rows.append({'sample_id':image['id'],'annotation_id':image['id'],'observation_id':obs,'split':'calibration',
             'image_path':str((root/'train/train_images'/image['file_name']).resolve()),'target_path':str(target)})
    assert {r['observation_id'] for r in rows}==ids
    write_jsonl(out/'samples.jsonl',rows)
    write_json(out/'preparation.json',{'split_hash':split['split_hash'],'status':'frozen_joint_check',
               'frozen_selection_sha256':sha256(OUT/'frozen_selection.json'),'check_ids_sha256':sha256(OUT/'new_check24.json')})
    return str(out)


def frozen_check(candidate,checks):
    dataset=export_check()
    checks['old_baseline']=score_detector(BASE,1024,'joint_check_old',dataset,[.2],True)
    checks['previous_recall_candidate']=score_detector(PREV,1536,'joint_check_previous',dataset,[.15],True)
    if candidate['min_area'] or candidate['closing_radius']:
        output='runs/joint_check_raw_prediction'
        command=[str(ROOT/'run.sh'),'-m','m3_filament','predict','--weights','runs/'+candidate['run']+'/weights/best.pt',
                 '--images',str(Path(dataset)/'eval_images'),'--output',output,'--imgsz',str(candidate['imgsz']),
                 '--device','cpu','--conf','.05','--max-det','100','--require-unseen']
        run_job('joint_check_raw_prediction',command,output+'/run_manifest.json',seconds=240,final=True)
        parent={'raw_cache':output+'/instances.jsonl'}
        checks['new_candidate']=score_post(parent,candidate['min_area'],candidate['closing_radius'],
                           'joint_check_candidate',dataset,[candidate['confidence']],True)
    else:
        checks['new_candidate']=score_detector(candidate['run'],candidate['imgsz'],'joint_check_candidate',dataset,[candidate['confidence']],True)
    final=choose_for_user([checks['new_candidate']['selected']],
          [checks['old_baseline']['selected'],checks['previous_recall_candidate']['selected']])
    return checks,final


def report(scores,selection,checks,final,error):
    replace_json(OUT/'results.json',{'campaign':CAMPAIGN,'state':STATE,'tuning':scores,
           'selection':selection,'new_check':checks,'final_joint_gate':final,'error':error,'official_score':False})
    lines=['# 综合指标优化实验','', '目标是PQ>0.35且各项质量严格提高；官网0.35不是3.5，不改变指标量纲。',
      '', '## 同一24幅调参图的固定参照','', '| 方案 | PQ | SQ | RQ | Precision | Recall | F1 | Small recall | FP | FN |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def table(name,row):
        r=derive_metrics(row)
        return '| '+name+' | '+' | '.join(f'{r[k]:.5f}' if r[k] is not None else 'N/A' for k in ('pq','sq','rq','precision','recall','f1','small_recall'))+f" | {r['fp']} | {r['fn']} |"
    lines += [table('旧保守方案',REFERENCES[0]),table('上一轮召回候选',REFERENCES[1])]
    if selection:
        lines+=['',table('本轮冻结候选',selection['candidate']),'',
           '调参联合约束：'+('通过' if selection['joint_gate_passed'] else '未通过；仅作诊断候选')+'；退步项：'+str(selection['violations'])]
    if checks:
        lines += ['', '## 新的24幅冻结检查','', '| 方案 | PQ | SQ | RQ | Precision | Recall | F1 | Small recall | FP | FN |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        lines += [table(name,data['selected']) for name,data in checks.items()]
    if final:lines+=['','最终联合约束：'+('通过' if final['joint_gate_passed'] else '未通过')+'；退步项：'+str(final['violations'])]
    lines += ['', '## 实际训练量','', '| Run | Epochs | Optimizer updates | Training timer seconds | Best SHA256 |',
           '|---|---:|---:|---:|---|']
    for run in (FIRST,SECOND,MASK_METRIC):
        p=Path('runs')/run/'metrics.json'
        if p.exists():
            m=json.loads(p.read_text());lines += [f"| {run} | {m['actual_epochs']} | {m['optimizer_updates']} | {m['elapsed_seconds']:.1f} | {m['checkpoint_sha256']} |"]
    if error:lines+=['','未完成/错误：'+error]
    for failure in STATE['failures']:
        lines+=['','阶段未完成（不计作成功）：'+failure['stage']+'；'+failure['error']]
    lines += ['', '## 口径与限制','',
       'Precision=(预测数−FP)/预测数，Recall=(GT数−FN)/GT数；与官方全IoU>0.5匹配对计数的SQ/RQ/PQ分别列出。小目标面积<1000原图像素，所有标注批次分别评分。',
       '', '两个参照均固定；各质量指标必须严格高于较高参照值，FP/FN必须严格低于较低参照值。持平也不称提高。无可行解时，按最差归一化差距选择的结果仅用于诊断，不称为全面提升。',
       '', '本轮仍为临时开发划分、单seed。本轮新检查来自未用于先前模型评分的calibration日期/内容组；先前预留24和check16不用于本轮调参，原holdout103与比赛test未用。任何已评估新检查都不回流本轮选参。',
       '', '训练使用原始独立实例mask；后处理只作用于预测的二值mask，不伪造语义概率。面积过滤保留存活mask原状；closing会改变预测边界和孔洞，收益必须由指标验证。',
       '', '所有重大评分与训练由research-guardian记录，失败不删除。依赖、base和全局环境未修改。不发布、不提交比赛。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


TUNING_OBS=json.loads(Path('runs/resume_tune_dev32_r1024_nbs2_e8_cpu/metrics.json').read_text())['observations']
HISTORY=json.loads(Path('reports/resume_20261104/results.json').read_text())['tuning']
REFERENCES=[next(r for r in HISTORY[BASE]['grid'] if r['confidence']==.2),
            next(r for r in HISTORY[PREV+'_r1536']['grid'] if r['confidence']==.15)]


def main():
    lock=(OUT/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    STATE.update(restore_score_accounting())
    scores={};selection=None;checks={};final=None;error=None
    save('starting')
    try:
        if (OUT/'frozen_selection.json').exists():
            frozen=json.loads((OUT/'frozen_selection.json').read_text());selection=frozen['selection'];scores=frozen['tuning']
        else:
            reference_data={'references':[derive_metrics(r) for r in REFERENCES],
                       'history_results_sha256':sha256('reports/resume_20261104/results.json')}
            if (OUT/'references.json').exists():
                if json.loads((OUT/'references.json').read_text())!=reference_data:
                    raise RuntimeError('fixed references changed')
            else:write_json(OUT/'references.json',reference_data)
            for alias,data in HISTORY.items():
                path=Path(data['artifact_output'])/'metrics.json'
                scores['history_'+alias]=load_score(path,'history_'+alias)
            # Completed transforms are reusable after a selection-policy reload;
            # validate their input/config/code hashes rather than rerunning them.
            for jobpath in (OUT/'jobs').glob('*.json'):
                job=json.loads(jobpath.read_text());command=job.get('command',[])
                if job.get('returncode')!=0 or '-m' not in command:continue
                if command[command.index('-m')+1]!='m3_filament.postprocess':continue
                name=job['name'];parent={'raw_cache':command[command.index('--instances')+1]}
                area=int(command[command.index('--min-area')+1]);radius=int(command[command.index('--closing-radius')+1])
                scores[name]=score_post(parent,area,radius,name)
            persist_partial(scores)
            optional_post(scores,scores['history_'+PREV+'_r1536'],'joint_seed',max_profiles=7)
            save('waiting_first_training')
            record=runner_record('balanced176-ft8')
            if not record:raise RuntimeError('first training registration missing')
            while not Path('runs/'+FIRST+'/metrics.json').exists():
                r=json.loads(record[1].read_text())
                if r['status']!='running':
                    STATE['failures'].append({'stage':FIRST,'error':'external training ended '+r['status']});break
                if remaining()<600:raise TimeoutError('deadline waiting first training')
                time.sleep(3)
            if Path('runs/'+SECOND+'/metrics.json').exists() or remaining()>4500:
                command=[str(ROOT/'run.sh'),'-m','m3_filament','train','--config','configs/'+SECOND+'.yaml','--run-id',SECOND,'--development']
                try:run_job(SECOND,command,'runs/'+SECOND+'/metrics.json',seconds=3000,config='configs/'+SECOND+'.yaml',
                      checkpoints=['runs/'+SECOND+'/weights/best.pt','runs/'+SECOND+'/weights/last.pt'],kind='training')
                except (RuntimeError,TimeoutError) as exc:STATE['failures'].append({'stage':SECOND,'error':repr(exc)})
            else:STATE['failures'].append({'stage':SECOND,'error':'insufficient protected UTC budget; not launched'})
            for run in (FIRST,SECOND):
                if not Path('runs/'+run+'/metrics.json').exists():continue
                for size in (1024,1536):
                    if remaining()<900:break
                    name='joint_tune_'+run+'_r'+str(size)
                    try:scores[name]=score_detector(run,size,name)
                    except (RuntimeError,TimeoutError) as exc:STATE['failures'].append({'stage':name,'error':repr(exc)})
                    persist_partial(scores)
            new_scores={key:value for key,value in scores.items() if key.startswith('joint_tune_')}
            if new_scores and remaining()>1200:
                native_parent=choose_for_user(candidate_rows(new_scores),REFERENCES)['candidate']
                name='joint_tune_'+native_parent['run']+'_r2048'
                try:
                    scores[name]=score_detector(native_parent['run'],2048,name)
                    new_scores[name]=scores[name]
                    persist_partial(scores)
                except (RuntimeError,TimeoutError) as exc:STATE['failures'].append({'stage':name,'error':repr(exc)})
            if new_scores and remaining()>900:
                preliminary=choose_for_user(candidate_rows(new_scores),REFERENCES)['candidate']
                optional_post(scores,new_scores[preliminary['alias']],'joint_new_post_'+preliminary['alias'],max_profiles=7)
            selection=choose_for_user(candidate_rows(scores),REFERENCES)
            if (not selection['joint_gate_passed'] and remaining()>1800) or Path('runs/'+MASK_METRIC).exists():
                command=[str(ROOT/'run.sh'),'scripts/train_mask_metric.py','--config','configs/'+MASK_METRIC+'.yaml','--run-id',MASK_METRIC]
                try:
                    run_job(MASK_METRIC,command,'runs/'+MASK_METRIC+'/metrics.json',seconds=1000,
                      config='configs/'+MASK_METRIC+'.yaml',checkpoints=['runs/'+MASK_METRIC+'/weights/best.pt','runs/'+MASK_METRIC+'/weights/last.pt'],kind='training')
                except (RuntimeError,TimeoutError) as exc:
                    STATE['failures'].append({'stage':MASK_METRIC,'error':repr(exc)})
                    record=runner_record(MASK_METRIC)
                    if record and record[2]['status']!='running':
                        import importlib.util
                        spec=importlib.util.spec_from_file_location('owned_checkpoint_recovery',ROOT/'scripts/recover_budget_checkpoints.py')
                        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                        STATE['conditional_recovery']=module.recover(Path('runs')/MASK_METRIC,record[2])
                if Path('runs/'+MASK_METRIC+'/metrics.json').exists() and remaining()>600:
                    conditional_scores={}
                    for size in (1536,2048):
                        if remaining()<600:break
                        name='joint_tune_'+MASK_METRIC+'_r'+str(size)
                        try:
                            scores[name]=score_detector(MASK_METRIC,size,name)
                            conditional_scores[name]=scores[name]
                        except (RuntimeError,TimeoutError) as exc:STATE['failures'].append({'stage':name,'error':repr(exc)})
                    if conditional_scores:
                        best=choose_for_user(candidate_rows(conditional_scores),REFERENCES)['candidate']
                        optional_post(scores,conditional_scores[best['alias']],'joint_maskmetric_post_'+best['alias'],max_profiles=3)
                selection=choose_for_user(candidate_rows(scores),REFERENCES)
            STATE['conditional_mask_metric_executed']=Path('runs/'+MASK_METRIC).exists()
            write_json(OUT/'frozen_selection.json',{'frozen_utc':now().isoformat(),'selection':selection,'tuning':scores,
                 'new_check_manifest_sha256':sha256(OUT/'new_check24.json'),'code_sha256':sha256('scripts/balanced_driver.py')})
            save('selection_frozen')
        report(scores,selection,checks,final,None)
        if remaining()>300:
            checks,final=frozen_check(selection['candidate'],checks)
        save('completed' if len(checks)==3 else 'completed_without_new_check')
    except BaseException as exc:
        error=repr(exc);STATE['error']=error;save('failed_or_budget_stopped')
    finally:
        report(scores,selection,checks,final,error)
        print(json.dumps(STATE,indent=2),flush=True)
    if error:raise SystemExit(1)


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('owned driver signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    main()
