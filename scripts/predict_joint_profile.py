"""Apply a frozen joint-metric profile to new original images, without evaluation labels."""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from m3_filament.io import read_jsonl,write_jsonl,write_json,sha256
from m3_filament.model import predict,source_manifest_hash
from m3_filament.postprocess import transform_record
from m3_filament.schema import validate_record

ROOT=Path(__file__).resolve().parents[1]


def runtime_versions():
    return {'python':platform.python_version(),**{name:version(name) for name in
            ('torch','torchvision','ultralytics','numpy','pycocotools','opencv-python')}}


def run_profile(profile_path,images,output):
    profile_path=Path(profile_path);profile=json.loads(profile_path.read_text())
    weights=Path(profile['weights'])
    if not weights.is_absolute():weights=ROOT/weights
    if sha256(weights)!=profile['checkpoint_sha256']:raise ValueError('profile weight hash mismatch')
    if profile.get('source_manifest_sha256')!=source_manifest_hash():raise ValueError('profile source manifest mismatch')
    if profile.get('requirements_lock_sha256')!=sha256(ROOT/'requirements-lock.txt'):raise ValueError('profile dependency lock mismatch')
    if profile.get('inference_script_sha256')!=sha256(Path(__file__)):raise ValueError('profile inference script mismatch')
    if profile.get('runtime_versions')!=runtime_versions():raise ValueError('profile runtime version mismatch')
    out=Path(output).resolve();raw=out.with_name(out.name+'_raw')
    if out.exists() or raw.exists():raise FileExistsError(out)
    floor=profile.get('score_floor',.05)
    if not 0<=floor<=profile['conf']<=1:raise ValueError('invalid profile score floor')
    predict(str(weights),str(Path(images).resolve()),str(raw),device=profile.get('device','cpu'),
            imgsz=profile['imgsz'],conf=floor,max_det=profile.get('max_det',100),require_unseen=True)
    records=[transform_record(row,out.name,profile.get('min_area',0),profile.get('closing_radius',0),profile['conf'])
             for row in read_jsonl(raw/'instances.jsonl')]
    for row in records:validate_record(row,exclusive=True)
    out.mkdir(parents=True)
    write_jsonl(out/'instances.jsonl',records)
    summary={'images':len(records),'instances':sum(len(r['instances']) for r in records),
             'profile_sha256':sha256(profile_path),'profile':profile,'parent_instances_sha256':sha256(raw/'instances.jsonl'),
             'output':str(out),'requires_ground_truth':False,'adoption_status':profile.get('status','unspecified')}
    write_json(out/'run_manifest.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',required=True);p.add_argument('--images',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=run_profile(a.profile,a.images,a.output)
    print(json.dumps({'images':result['images'],'instances':result['instances'],'output':result['output']},indent=2))


if __name__=='__main__':main()
