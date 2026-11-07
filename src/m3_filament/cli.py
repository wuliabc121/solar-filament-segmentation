"""Independent entry points for data, model, fusion and submission stages."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import yaml
from .io import read_jsonl,write_json,write_jsonl,sha256
from .fusion import FusionConfig,fuse_record,probability


def load_semantic(path,observation_id,shape):
    with np.load(path,allow_pickle=False) as d:
        required={'probabilities','observation_id','run_id','checkpoint_hash','split_hash'}
        if not required.issubset(d.files):raise ValueError('semantic cache metadata missing')
        meta={k:str(d[k].item()) for k in required-{'probabilities'}}
        if meta['observation_id']!=observation_id:raise ValueError('semantic cache observation mismatch')
        if any(not meta[k] for k in ('run_id','checkpoint_hash','split_hash')):raise ValueError('empty semantic metadata')
        p=probability(d['probabilities'],shape).copy()
    meta['file_sha256']=sha256(path)
    return p,meta


def main(argv=None):
    parser=argparse.ArgumentParser(description='M3 filament instance segmentation; all masks use original H×W coordinates')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('doctor');p.add_argument('--output')
    p=sub.add_parser('audit');p.add_argument('--root',default='data/raw');p.add_argument('--output',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',default='data/raw');p.add_argument('--output',required=True)
    p.add_argument('--split');p.add_argument('--max-observations',type=int);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--backend',choices=['native','polygon'],default='native')
    p.add_argument('--selection',choices=['legacy','hash'],default='legacy')
    p.add_argument('--selection-manifest')
    p=sub.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--smoke',action='store_true');mode.add_argument('--production',action='store_true');mode.add_argument('--development',action='store_true')
    p=sub.add_parser('predict');p.add_argument('--weights',required=True);p.add_argument('--images',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--imgsz',type=int,default=640);p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--max-images',type=int);p.add_argument('--require-unseen',action='store_true')
    p.add_argument('--max-det',type=int,default=20)
    p=sub.add_parser('fuse');p.add_argument('--instances',required=True);p.add_argument('--cnn',required=True);p.add_argument('--transformer',required=True)
    p.add_argument('--config',default='configs/fusion.yaml');p.add_argument('--output',required=True);p.add_argument('--run-id',required=True)
    p=sub.add_parser('submit');p.add_argument('--instances',required=True);p.add_argument('--expected-images',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='doctor':
        from .model import doctor
        result=doctor(args.output)
    elif args.command=='audit':
        from .data import audit_dataset
        result=audit_dataset(args.root,args.output)
    elif args.command=='prepare':
        from .data import prepare_dataset
        result=prepare_dataset(args.root,args.output,args.split,args.max_observations,args.seed,args.backend,args.selection,args.selection_manifest)
    elif args.command=='train':
        from .model import train_model
        result=train_model(args.config,args.run_id,args.smoke,args.development)
    elif args.command=='predict':
        from .model import predict
        result=predict(args.weights,args.images,args.output,args.device,args.imgsz,args.conf,args.max_images,args.require_unseen,args.max_det)
    elif args.command=='fuse':
        cfg=FusionConfig(**yaml.safe_load(Path(args.config).read_text()));out=Path(args.output)
        if out.exists():raise FileExistsError(out)
        result_records=[];seen=set()
        for r in read_jsonl(args.instances):
            obs=r['observation_id']
            if obs in seen:raise ValueError('duplicate observation_id in instances')
            seen.add(obs)
            pc,mc=load_semantic(Path(args.cnn)/(obs+'.npz'),obs,r['image_size'])
            pt,mt=load_semantic(Path(args.transformer)/(obs+'.npz'),obs,r['image_size'])
            if mc['split_hash']!=mt['split_hash'] or (r['source'].get('split_hash') and r['source']['split_hash']!=mc['split_hash']):
                raise ValueError('semantic and detector split_hash mismatch')
            fused=fuse_record(r,pc,pt,cfg,args.run_id)
            fused['source']['semantic_caches']={'cnn':mc,'transformer':mt};result_records.append(fused)
        out.mkdir(parents=True)
        write_jsonl(out/'coarse_instances.jsonl',result_records)
        write_json(out/'run_manifest.json',{'run_id':args.run_id,'config':asdict(cfg),'instance_cache_sha256':sha256(args.instances)})
        result={'images':len(result_records),'output':str(out)}
    else:
        from .submission import write_submission
        records=list(read_jsonl(args.instances))
        expected={p.stem for p in Path(args.expected_images).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
        if not expected or {r['observation_id'] for r in records}!=expected:raise ValueError('prediction coverage differs from expected image set')
        write_submission(records,args.output);result={'submission':args.output,'images':len(records),'uploaded':False}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
