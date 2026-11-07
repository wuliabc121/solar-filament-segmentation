"""Bounded training and original-resolution prediction with reproducible manifests."""
import csv
import json
import platform
import shutil
import time
import datetime
import functools
from pathlib import Path
import numpy as np
import yaml
from PIL import Image,ImageDraw
from pycocotools import mask as coco
from .io import sha256,content_hash,write_json,write_jsonl,read_jsonl
from .schema import make_record,make_instance,validate_record,crop_metadata
from .masks import decode


def validate_training_request(config,data_info,smoke,development=False):
    epochs=config.get('epochs',0);size=config.get('imgsz',0)
    if type(epochs)is not int or epochs<1 or type(size)is not int or size<32:
        raise ValueError('epochs/imgsz must be positive integers')
    if smoke:
        if epochs>2 or size>640 or sum(data_info['samples'].values())>48:
            raise ValueError('smoke budget: <=2 epochs, <=640 input, <=48 annotation-pass samples')
    elif development:
        hours=config.get('time')
        overnight=config.get('budget_profile')=='overnight'
        if overnight:
            try:
                deadline=datetime.datetime.fromisoformat(config['deadline_utc'].replace('Z','+00:00'))
                remaining=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            except Exception as e:raise ValueError('overnight development requires UTC deadline') from e
            if not 0<remaining<=8*3600:raise ValueError('overnight deadline expired or exceeds authorization window')
        if epochs>(40 if overnight else 20) or size>(1536 if overnight else 1024) or sum(data_info['samples'].values())>(1800 if overnight else 256) or type(hours)not in (int,float) or not 0<hours<=(3 if overnight else .5):
            raise ValueError('development budget: <=20 epochs, <=1024 input, <=256 samples, explicit time <=0.5h')
    elif not data_info.get('production_training_allowed'):
        raise ValueError('production training blocked: requires complete conversion audit and team split')


def doctor(output=None):
    import torch,ultralytics,torchvision
    available=torch.backends.mps.is_available()
    device='mps' if available else 'cpu'
    t=time.perf_counter()
    a=torch.ones((16,16),device=device,requires_grad=True)
    loss=(a@a).mean();loss.backward()
    if device=='mps':torch.mps.synchronize()
    result={'python':platform.python_version(),'machine':platform.machine(),
            'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,
            'mps_built':torch.backends.mps.is_built(),'mps_available':available,
            'tested_device':device,'forward_backward_passed':bool(torch.isfinite(a.grad).all().item()),
            'probe_seconds':time.perf_counter()-t,
            'isolation':{'prefix':__import__('sys').prefix,'user_site_disabled':__import__('site').ENABLE_USER_SITE is False}}
    if output:write_json(output,result)
    return result


def choose_device(requested):
    import torch
    if requested=='auto':return 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS not available in this process; choose CPU explicitly')
    return requested


def train_model(config_path,run_id,smoke,development=False):
    from ultralytics import YOLO
    cfg=yaml.safe_load(Path(config_path).read_text())
    data_path=Path(cfg['data']).resolve()
    info=json.loads((data_path.parent/'preparation.json').read_text())
    validate_training_request(cfg,info,smoke,development)
    if not all(c.isalnum() or c in '-_' for c in run_id):raise ValueError('unsafe run_id')
    run=Path('runs')/run_id
    if run.exists():raise FileExistsError(run)
    model_path=Path(cfg.pop('model')).resolve()
    if not model_path.is_file():raise FileNotFoundError(f'download official model weights first: {model_path}')
    source_hash=source_manifest_hash()
    cfg.update({'device':choose_device(cfg.get('device','auto')),'project':str(Path('runs').resolve()),
                'name':run_id,'exist_ok':True,'single_cls':True,'verbose':False})
    samples=list(read_jsonl(data_path.parent/'samples.jsonl'))
    manifest={'run_id':run_id,'run_kind':'technical_smoke' if smoke else 'development_experiment' if development else 'production',
              'split_hash':info['split_hash'],'split_status':info['split_status'],
              'annotation_sha256':info['annotation_sha256'],'sample_manifest_sha256':info['sample_manifest_sha256'],
              'training_observation_ids':sorted({r['observation_id'] for r in samples if r['split']=='train'}),
              'calibration_observation_ids':sorted({r['observation_id'] for r in samples if r['split']=='calibration'}),
              'initial_checkpoint_sha256':sha256(model_path),'model_name':model_path.name,
              'config_file_sha256':sha256(config_path),'resolved_config':cfg,
              'source_manifest_sha256':source_hash,'requirements_lock_sha256':sha256('requirements-lock.txt'),
              'target_backend':info.get('target_backend','polygon'),
              'official_evaluator':'not provided by M2; no official PQ claimed'}
    # Ultralytics' time option dynamically changes the number of epochs.
    # Keep fixed-epoch comparisons; our callback plus the experiment runner
    # enforce the independently declared wall-clock budget instead.
    budget_hours=cfg.pop('time',None) if development else None
    deadline_utc=cfg.pop('deadline_utc',None);budget_profile=cfg.pop('budget_profile',None)
    crop_size=cfg.pop('training_crop_size',0);crop_probability=cfg.pop('training_crop_probability',0.)
    manifest['budget_profile']=budget_profile;manifest['deadline_utc']=deadline_utc
    manifest['training_crop']={'size':crop_size,'probability':crop_probability}
    if development:manifest['development_budget_hours']=budget_hours
    run.mkdir(parents=True)
    snapshot=run/'code_snapshot'
    shutil.copytree('src',snapshot/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('configs',snapshot/'configs')
    shutil.copy2('requirements-lock.txt',snapshot/'requirements-lock.txt')
    write_json(snapshot/'source_manifest.json',{str(p.relative_to(snapshot)):sha256(p) for p in sorted(snapshot.rglob('*')) if p.is_file()})
    write_json(run/'run_manifest_started.json',manifest)
    started=time.perf_counter()
    try:
        model=YOLO(str(model_path))
        if development:
            def stop_at_budget(trainer):
                expired=deadline_utc and datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(deadline_utc.replace('Z','+00:00'))
                if time.perf_counter()-started>=budget_hours*3600 or expired:trainer.stop=True
            model.add_callback('on_train_epoch_end',stop_at_budget)
            model.add_callback('on_train_batch_end',stop_at_budget)
        if info.get('target_backend')=='native':
            from .native import NativeSegmentationTrainer
            trainer=functools.partial(NativeSegmentationTrainer,training_crop_size=crop_size,training_crop_probability=crop_probability)
            validation_result=model.train(trainer=trainer,**cfg)
        else:validation_result=model.train(**cfg)
    except Exception as e:
        write_json(run/'failure.json',{'run_id':run_id,'error':repr(e),'seconds':time.perf_counter()-started})
        raise
    checkpoint=run/'weights/best.pt'
    if not checkpoint.exists():raise RuntimeError('training returned without best.pt')
    manifest.update({'checkpoint_sha256':sha256(checkpoint),
                     'checkpoint_hashes':{p.name:sha256(p) for p in (run/'weights').glob('*.pt')},
                     'elapsed_seconds':time.perf_counter()-started,
                     'status':'completed','checkpoint':str(checkpoint.resolve())})
    write_json(run/'run_manifest.json',manifest)
    with (run/'results.csv').open() as f:last=list(csv.DictReader(f))[-1]
    last_values={k.strip():float(v) for k,v in last.items() if v.strip()}
    values={k:float(v) for k,v in validation_result.results_dict.items()}
    metrics={'run_id':run_id,'purpose':'technical smoke' if smoke else 'provisional development comparison' if development else 'production',
             'elapsed_seconds':manifest['elapsed_seconds'],'device':cfg['device'],
             'checkpoint_sha256':manifest['checkpoint_sha256'],'ultralytics_metrics':values,
             'validation_checkpoint':'best.pt','last_epoch_csv':last_values,
             'actual_epochs':int(last_values['epoch']),
             'optimizer_updates':int(model.trainer.ema.updates) if model.trainer.ema else None,
             'gradient_accumulation':int(model.trainer.accumulate),
             'minibatches_per_epoch':len(model.trainer.train_loader),
             'official_pq':None,'official_evaluator_aligned':False}
    write_json(run/'metrics.json',metrics)
    return metrics


def source_manifest_hash():
    paths=sorted(Path('src').rglob('*.py'))
    return content_hash({str(p):sha256(p) for p in paths})


def check_unseen(observation_ids,training_ids):
    overlap=set(observation_ids)&set(training_ids)
    if overlap:raise ValueError(f'OOF leakage: detector trained on {sorted(overlap)}')


def load_training_provenance(weights):
    w=Path(weights);path=w.parent.parent/'run_manifest.json'
    if not path.exists():return None
    manifest=json.loads(path.read_text())
    known=set(manifest.get('checkpoint_hashes',{}).values())
    if manifest.get('checkpoint_sha256'):known.add(manifest['checkpoint_sha256'])
    if sha256(w) not in known:raise ValueError('checkpoint hash does not match neighboring training manifest')
    return manifest


def mask_priority_indices(masks,scores,threshold=.8):
    if len(masks)!=len(scores) or not np.isfinite(scores).all():raise ValueError('invalid detector masks/scores')
    accepted=[];indices=[]
    for k in sorted(range(len(scores)),key=lambda k:(-float(scores[k]),k)):
        rle=coco.encode(np.asfortranarray(masks[k].astype(np.uint8)))
        if accepted and (coco.iou([rle],accepted,[0]*len(accepted))>=threshold).any():continue
        accepted.append(rle);indices.append(k)
    return indices


def predict(weights,images,output,device='auto',imgsz=640,conf=.25,max_images=None,require_unseen=False,max_det=20):
    from ultralytics import YOLO
    import torch
    w=Path(weights);out=Path(output)
    if out.exists():raise FileExistsError(out)
    paths=[Path(images)] if Path(images).is_file() else sorted(p for p in Path(images).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if max_images:paths=paths[:max_images]
    if not paths:raise ValueError('no images to predict')
    observations=[p.stem for p in paths]
    if len(set(observations))!=len(observations):raise ValueError('duplicate observation_id from filenames')
    train_manifest=load_training_provenance(w)
    if require_unseen:
        if not train_manifest:raise ValueError('unseen/OOF check requires detector training manifest')
        check_unseen(observations,train_manifest['training_observation_ids'])
    run_id=out.name
    cfg={'device':choose_device(device),'imgsz':imgsz,'conf':conf,'iou':.95,'retina_masks':True,
         'max_det':max_det,'verbose':False,'save':False}
    source={'run_id':run_id,'checkpoint_sha256':sha256(w),'inference_config':cfg,
            'source_manifest_sha256':source_manifest_hash(),
            'inference_config_sha256':content_hash(cfg),'ultralytics_version':__import__('ultralytics').__version__,
            'training_run_id':train_manifest['run_id'] if train_manifest else None,
            'split_hash':train_manifest['split_hash'] if train_manifest else None,
            'unseen_check_passed':require_unseen,'is_oof':False,
            'note':'unseen check alone does not establish training-internal OOF; M1 fold provenance is also required'}
    model=YOLO(str(w));records=[];timings=[];out.mkdir(parents=True)
    for path in paths:
        with Image.open(path) as im:rgb=im.convert('RGB');width,height=im.size
        started=time.perf_counter();result=model.predict(str(path),**cfg)[0]
        if cfg['device']=='mps':torch.mps.synchronize()
        timings.append(time.perf_counter()-started)
        insts=[];used=np.zeros((height,width),bool);dropped=0
        if result.masks is not None:
            masks=result.masks.data.detach().cpu().numpy()
            scores=result.boxes.conf.detach().cpu().numpy()
            if masks.shape[1:]!=(height,width):raise ValueError('retina masks were not restored to original image shape')
            candidates=mask_priority_indices(masks,scores)
            dropped+=len(scores)-len(candidates)
            for k in candidates:
                m=masks[k].astype(bool)
                new=m&~used
                if not new.any():dropped+=1;continue
                inst=make_instance(new,len(insts)+1,float(scores[k]),source)
                inst['crop']=crop_metadata(inst,(height,width));insts.append(inst);used|=new
        rec=make_record(path.stem,[height,width],insts,source)
        rec['image_sha256']=sha256(path);rec['dropped_duplicate_or_occluded_masks']=dropped
        validate_record(rec,exclusive=True);records.append(rec)
        if len(records)==1:save_overlay(rgb,rec,out/'prediction-preview.jpg')
    write_jsonl(out/'instances.jsonl',records)
    write_json(out/'run_manifest.json',{**source,'images':observations,'image_count':len(records),
                                      'instance_count':sum(len(r['instances']) for r in records),
                                      'seconds_per_image':timings,'total_predict_seconds':sum(timings)})
    return {'output':str(out),'images':len(records),'instances':sum(len(r['instances']) for r in records)}


def save_overlay(image,record,path):
    array=np.array(image).copy();colors=[(255,70,70),(20,220,190),(250,190,30),(100,100,255)]
    for j,inst in enumerate(record['instances']):
        m=decode(inst['segmentation']).astype(bool)
        array[m]=(array[m]*.5+np.array(colors[j%len(colors)])*.5).astype(np.uint8)
    im=Image.fromarray(array);draw=ImageDraw.Draw(im)
    for inst in record['instances']:
        draw.rectangle(inst['bbox'],outline='yellow',width=2)
        draw.text(inst['bbox'][:2],f"{inst['instance_id']} {inst['score']:.4f}",fill='yellow')
    im.thumbnail((1024,1024));im.save(path)
