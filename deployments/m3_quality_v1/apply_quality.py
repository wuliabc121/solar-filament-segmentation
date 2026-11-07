"""Apply frozen standalone quality head to verified M3 raw masks; no GT or M1/M2 inputs."""
import argparse,json,time
from pathlib import Path
import cv2
import torch
from quality import QualityHead,Standardizer,FEATURES,extract_features,head_scores,score_mode,filter_record
from m3_filament.io import sha256,read_jsonl,write_json,write_jsonl
from team_adaptation.metrics import validate_fast_record


def validate_raw(record,profile):
    if record['source'].get('checkpoint_sha256')!=profile['checkpoint_sha256']:raise ValueError('wrong M3 backbone checkpoint')
    if record['source'].get('inference_config')!=profile['raw_config']:raise ValueError('raw inference settings differ from head training')
    if record['source'].get('quality_scoring') or record['source'].get('score_threshold') is not None:raise ValueError('input already filtered/rescored')
    validate_fast_record(record)


def apply(profile_path,raw,images,output):
    profile_path=Path(profile_path);p=json.loads(profile_path.read_text());out=Path(output)
    if out.exists():raise FileExistsError(out)
    checkpoint=profile_path.parent/p['head_file']
    if sha256(checkpoint)!=p['head_sha256'] or sha256(Path(__file__).with_name('quality.py'))!=p['quality_code_sha256']:raise ValueError('head or feature code hash changed')
    pack=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if pack['features']!=FEATURES:raise ValueError('feature order changed')
    head=QualityHead();head.load_state_dict(pack['state_dict'],strict=True);head.eval();torch.set_num_threads(2);norm=Standardizer(pack['mean'].numpy(),pack['std'].numpy())
    source={k:p[k] for k in ['head_sha256','checkpoint_sha256','quality_code_sha256']};source.update(profile_sha256=sha256(profile_path),mode=p['selected']['mode'])
    results=[];seen=set();start=time.monotonic()
    paths={}
    for path in Path(images).iterdir():
        if path.suffix.lower() in ('.jpg','.jpeg','.png'):
            if path.stem in paths:raise ValueError('duplicate image stem')
            paths[path.stem]=path
    for record in read_jsonl(raw):
        validate_raw(record,p);s=record['observation_id']
        if s in seen:raise ValueError('duplicate raw observation')
        seen.add(s);path=paths[s]
        if sha256(path)!=record.get('image_sha256'):raise ValueError('raw masks do not match image bytes')
        x=extract_features(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE),record);q=head_scores(head,norm,x);scores=score_mode(q,x[:,0],p['selected']['mode'])
        results.append(filter_record(record,scores,p['selected']['threshold'],'M3_quality_frozen',source,min_area=p['min_area']))
    if not results:raise ValueError('empty prediction file')
    out.mkdir(parents=True);write_jsonl(out/'instances.jsonl',results);write_json(out/'manifest.json',{'profile_sha256':sha256(profile_path),'raw_sha256':sha256(raw),'images':len(results),'instances':sum(len(r['instances']) for r in results),'seconds':time.monotonic()-start,'GT_used':False,'M1_M2_used':False})
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['profile','raw','images','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();print(apply(a.profile,a.raw,a.images,a.output))
