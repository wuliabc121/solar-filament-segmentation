"""Unified local entry points. Integration does not resume optimization."""
import argparse,json,time,sys,platform,importlib.metadata
from pathlib import Path
import cv2
import numpy as np
import torch
from . import ROOT
from .contracts import load_probability,save_probability,fuse_team
from .models import load_model,predict_m1,predict_m2,yolo_record
from .cache import convert_m2_cache
from .integrity import verified_checkpoint
from .vendor import m2_modules
from m3_filament.io import sha256,read_jsonl,write_json,write_jsonl,content_hash
from m3_filament.fusion import FusionConfig
from m3_filament.schema import make_record,make_instance,validate_record

def registry():return json.loads((ROOT/'team/registry.json').read_text())

def device(value):
 d=torch.device(value)
 if value=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable; use --device cpu')
 return d

def doctor(args):
 r=registry();out={'python':sys.version,'prefix':sys.prefix,'platform':platform.platform(),'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),'optimization_status':r['optimization_status'],'versions':{k:importlib.metadata.version(k) for k in ('torch','torchvision','numpy','transformers','segmentation-models-pytorch','timm')},'checkpoints':{}}
 for k,v in r['models'].items():
  actual=sha256(ROOT/v['checkpoint']);out['checkpoints'][k]={'sha256':actual,'matches_registry':actual==v['checkpoint_sha256']}
 if not all(v['matches_registry'] for v in out['checkpoints'].values()):raise ValueError('checkpoint integrity failure')
 print(json.dumps(out,indent=2))
 if args.output:write_json(args.output,out)

def predict(args):
 d=device(args.device);model,info=load_model(args.model,d)
 files=sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff'))
 if not files:raise ValueError('no supported input images')
 if len({p.stem for p in files})!=len(files):raise ValueError('duplicate observation stems')
 for p in files:
  image=cv2.imread(str(p),cv2.IMREAD_COLOR)
  if image is None:raise ValueError('unreadable image: '+str(p))
  prob=predict_m1(model,cv2.cvtColor(image,cv2.COLOR_BGR2RGB),d) if args.model=='m1' else predict_m2(model,cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),d,args.tile,args.overlap,args.batch)
  meta={'model':args.model,'observation_id':p.stem,'run_id':args.run_id,'checkpoint_hash':info['checkpoint_sha256'],'split_hash':info['source_split_hash'],'coordinate_space':'original','is_oof':False,'purpose':'application','training_membership_status':info['training_membership_status'],'image_sha256':sha256(p),'inference':{'device':args.device,'resize':[512,512] if args.model=='m1' else None,'tile':args.tile if args.model=='m2' else None,'overlap':args.overlap if args.model=='m2' else None,'batch':args.batch if args.model=='m2' else None}}
  save_probability(Path(args.output)/(p.stem+'.npz'),prob,meta)
 print(json.dumps({'images':len(files),'model':args.model,'benchmark_eligible':False}))

def fuse(args):
 config=FusionConfig(**json.loads(Path(args.config).read_text())) if args.config else FusionConfig()
 records=list(read_jsonl(args.instances));ids=[r['observation_id'] for r in records]
 if len(set(ids))!=len(ids):raise ValueError('duplicate instance observations')
 out=[]
 for r in records:
  obs=r['observation_id']
  if Path(obs).name!=obs or obs in ('.','..'):raise ValueError('unsafe observation id')
  pc=load_probability(Path(args.cnn)/(obs+'.npz'),obs,r['image_size']);pt=load_probability(Path(args.transformer)/(obs+'.npz'),obs,r['image_size'])
  out.append(fuse_team(r,pc,pt,config,args.run_id,args.purpose))
 write_jsonl(args.output,out)
 print(json.dumps({'observations':len(out),'coarse_instances':sum(len(r['instances']) for r in out),'output':args.output,'benchmark_eligible':False}))

def smoke(args):
 start=time.monotonic();np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4);d=device(args.device)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 gray=np.tile(np.linspace(0,255,80,dtype=np.uint8),(64,1));rgb=np.repeat(gray[:,:,None],3,axis=2);probs={};details={}
 for name in ('m1','m2'):
  t=time.monotonic();model,info=load_model(name,d)
  p=predict_m1(model,rgb,d) if name=='m1' else predict_m2(model,gray,d,tile=512,overlap=128,batch=1)
  meta={'model':name,'observation_id':'synthetic','run_id':'team-integration-synthetic','checkpoint_hash':info['checkpoint_sha256'],'split_hash':info['source_split_hash'],'coordinate_space':'original','is_oof':False,'purpose':'smoke','training_membership_status':'synthetic_only'}
  save_probability(out/(name+'.npz'),p,meta);probs[name]=(p,meta);details[name]={'strict_load':True,'shape':list(p.shape),'finite':bool(np.isfinite(p).all()),'range':[float(p.min()),float(p.max())],'seconds':time.monotonic()-t,'checkpoint_sha256':info['checkpoint_sha256'],'checkpoint_details':{k:v for k,v in info['details'].items() if k!='hf_config'}}
  del model
 from ultralytics import YOLO
 r=registry();mi=r['models']['m3'];m3path,m3hash=verified_checkpoint(mi);y=YOLO(str(m3path));pred=y.predict(rgb,imgsz=64,device=args.device,verbose=False,retina_masks=True)[0]
 details['m3']={'normal_checkpoint_load':True,'synthetic_detected_instances':len(pred.boxes),'checkpoint_sha256':m3hash}
 actual_coarse=yolo_record(pred,'synthetic',{'run_id':'synthetic-M3-real-checkpoint','split_hash':mi['source_split_hash'],'checkpoint_hash':mi['checkpoint_sha256'],'purpose':'smoke'})
 write_jsonl(out/'actual_m3_coarse.jsonl',[actual_coarse])
 fullchain=fuse_team(actual_coarse,probs['m1'],probs['m2'],FusionConfig(threshold=.35,residual=True,min_area=4),'actual-three-model-chain','smoke')
 write_jsonl(out/'actual_three_model_fusion.jsonl',[fullchain]);validate_record(fullchain,exclusive=True);del y
 # A deterministic coarse fixture exercises M4/fusion even when synthetic image has no YOLO proposals.
 mask=np.zeros(gray.shape,np.uint8);mask[12:20,10:50]=1
 src={'run_id':'synthetic-coarse-fixture','split_hash':'synthetic-fixture','purpose':'smoke'}
 coarse=make_record('synthetic',gray.shape,[make_instance(mask,1,.9,src)],src);write_jsonl(out/'coarse_fixture.jsonl',[coarse])
 actual=fuse_team(coarse,probs['m1'],probs['m2'],FusionConfig(threshold=.35,residual=True,min_area=4), 'synthetic-real-model-fusion','smoke');write_jsonl(out/'real_model_fusion.jsonl',[actual])
 # Fixture probabilities verify residual generation and preserve independent source hashes.
 fixture=np.zeros(gray.shape,np.float32);fixture[12:20,10:50]=.9;fixture[40:47,60:70]=.9
 pc=(fixture,probs['m1'][1]);pt=(fixture,probs['m2'][1]);checked=fuse_team(coarse,pc,pt,FusionConfig(threshold=.5,residual=True,min_area=4,endpoint_guard=0),'synthetic-contract-fusion','smoke')
 validate_record(checked,exclusive=True);assert len(checked['instances'])==2 and any(i['requires_refiner'] for i in checked['instances']);write_jsonl(out/'contract_fusion_m4.jsonl',[checked])
 # Test both original RLE CSV evaluator and aligned helper on synthetic independent annotators.
 _,ev=m2_modules();counts=ev.mask_to_counts(mask);score=ev.score_csv([('a-synthetic_1',counts),('b-synthetic_1',counts)],[('synthetic_1',counts)],64,80,True);assert score['PQ']==1 and score['tp']==2
 result={'status':'passed','kind':'technical_synthetic_only_not_model_quality','seed':42,'device':str(d),'models':details,'actual_three_model_chain_verified':True,'actual_three_model_fusion_instance_count':len(fullchain['instances']),'fusion_real_model_instance_count':len(actual['instances']),'fixture_fusion_instances':2,'m4_contract_verified':True,'synthetic_evaluator_pq':score['PQ'],'no_real_dataset_inference':True,'optimization_still_paused':True,'seconds':time.monotonic()-start}
 write_json(out/'metrics.json',result);print(json.dumps(result,indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 d=s.add_parser('doctor');d.add_argument('--output');d.set_defaults(func=doctor)
 a=s.add_parser('audit');a.set_defaults(func=lambda _:print((ROOT/'reports/team_integration_20261104/protocol_audit.json').read_text()))
 a=s.add_parser('predict-semantic');a.add_argument('--model',choices=['m1','m2'],required=True);a.add_argument('--images',required=True);a.add_argument('--output',required=True);a.add_argument('--run-id',required=True);a.add_argument('--device',choices=['cpu','mps'],default='cpu');a.add_argument('--tile',type=int,default=512);a.add_argument('--overlap',type=int,default=128);a.add_argument('--batch',type=int,default=1);a.set_defaults(func=predict)
 a=s.add_parser('import-cache');a.add_argument('--source',required=True);a.add_argument('--observations',nargs='+',required=True);a.add_argument('--output',required=True);a.set_defaults(func=lambda a:print([str(x) for x in convert_m2_cache(a.source,a.output,a.observations,registry())]))
 a=s.add_parser('fuse');a.add_argument('--instances',required=True);a.add_argument('--cnn',required=True);a.add_argument('--transformer',required=True);a.add_argument('--config');a.add_argument('--run-id',required=True);a.add_argument('--purpose',choices=['smoke','application'],required=True);a.add_argument('--output',required=True);a.set_defaults(func=fuse)
 a=s.add_parser('score-csv');a.add_argument('--gt',required=True);a.add_argument('--pred',required=True);a.add_argument('--height',type=int,default=2048);a.add_argument('--width',type=int,default=2048);a.add_argument('--output',required=True)
 def score(a):
  _,ev=m2_modules();r=ev.score_csv(ev.read_rows(a.gt),ev.read_rows(a.pred),a.height,a.width,True);r.update(evaluator='M2 logic with official float32 IoU alignment',benchmark_status='caller_supplied_CSVs_no_protocol_eligibility_attested');write_json(a.output,r);print(json.dumps(r))
 a.set_defaults(func=score)
 a=s.add_parser('verify-integration');a.add_argument('--device',choices=['cpu','mps'],default='cpu');a.add_argument('--output',required=True);a.set_defaults(func=smoke)
 args=p.parse_args();args.func(args)
