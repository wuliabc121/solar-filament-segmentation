"""Active M3 standalone prediction with a versioned, verified deployment profile."""
import argparse,importlib,json,sys,tempfile
from contextlib import contextmanager
from pathlib import Path
from m3_filament.io import sha256,read_jsonl,write_json,write_jsonl
from m3_filament.model import predict
from m3_filament.postprocess import transform_record
from m3_filament.schema import validate_record
ROOT=Path(__file__).resolve().parents[1]


def load_deployment(pointer):
    pointer=Path(pointer).resolve();active=json.loads(pointer.read_text());path=pointer.parent/active['deployment']
    if sha256(path)!=active['sha256']:raise ValueError('active deployment hash mismatch')
    p=json.loads(path.read_text())
    for name,digest in p['artifacts'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('deployment artifact hash mismatch: '+name)
    p['_deployment_sha256']=active['sha256'];p['_pointer']=str(pointer)
    return p



@contextmanager
def quality_image_directory(images):
    images=Path(images).resolve()
    if images.is_dir():
        yield images
    else:
        with tempfile.TemporaryDirectory(prefix='m3-single-image-') as staging:
            link=Path(staging)/images.name;link.symlink_to(images)
            yield Path(staging)


def run(images,output,pointer,require_unseen=False):
    import torch
    torch.set_num_threads(4)
    p=load_deployment(pointer);images=Path(images).resolve();out=Path(output).resolve();raw=out.with_name(out.name+'_raw')
    if out.exists() or raw.exists():raise FileExistsError('output or raw output already exists')
    if require_unseen:
        paths=[images] if images.is_file() else sorted(x for x in images.iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png'))
        excluded=set(p['development_exposure_dates']);excluded_hashes=set(p.get('development_exposure_hashes',[]))
        if any(x.stem[:8] in excluded or sha256(x) in excluded_hashes for x in paths):raise ValueError('image overlaps backbone/head training or head tuning by date/content; unseen request rejected')
    predict(str(ROOT/p['backbone']),str(images),str(raw),device='cpu',imgsz=2048,conf=.05,max_det=100,require_unseen=require_unseen)
    if p['kind']=='quality':
        sys.path.insert(0,str(ROOT/p['bundle']))
        application=importlib.import_module('apply_quality')
        with quality_image_directory(images) as image_directory:
            application.apply(ROOT/p['quality_profile'],raw/'instances.jsonl',image_directory,out)
    else:
        records=[transform_record(r,out.name,32,0,.4) for r in read_jsonl(raw/'instances.jsonl')]
        out.mkdir(parents=True);write_jsonl(out/'instances.jsonl',records)
    records=list(read_jsonl(out/'instances.jsonl'))
    for r in records:validate_record(r,exclusive=True)
    summary={'deployment_id':p['id'],'deployment_sha256':p['_deployment_sha256'],'active_pointer':p['_pointer'],'images':len(records),'instances':sum(len(r['instances']) for r in records),'raw_sha256':sha256(raw/'instances.jsonl'),'instances_sha256':sha256(out/'instances.jsonl'),'backbone_sha256':sha256(ROOT/p['backbone']),'head_sha256':p.get('head_sha256'),'score_kind':'learned_mask_quality' if p['kind']=='quality' else 'detector_confidence','requires_ground_truth':False,'benchmark_eligible':False,'output':str(out),'require_unseen':require_unseen,'adoption':'user_authorized_default'}
    write_json(out/'run_manifest.json',summary);return summary


def main():
    a=argparse.ArgumentParser(description='Active M3: frozen S + learned quality head; original-coordinate instances.')
    a.add_argument('--images',required=True);a.add_argument('--output',required=True)
    a.add_argument('--deployment',default=str(ROOT/'deployments/active_m3.json'),help='Active pointer or preserved old-C rollback pointer')
    a.add_argument('--require-unseen',action='store_true');args=a.parse_args()
    print(json.dumps(run(args.images,args.output,args.deployment,args.require_unseen),indent=2))


if __name__=='__main__':main()
