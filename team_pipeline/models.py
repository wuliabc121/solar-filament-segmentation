"""Offline, strict-weight semantic model adapters; no pretrained downloads."""
import json,os
import cv2
import numpy as np
import torch
from . import ROOT
from .vendor import m2_modules
from .integrity import verified_checkpoint

MEAN=np.array([.485,.456,.406],np.float32);STD=np.array([.229,.224,.225],np.float32)

def load_model(name,device):
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['DISABLE_SAFETENSORS_CONVERSION']='1'
 registry=json.loads((ROOT/'team/registry.json').read_text());info=registry['models'][name];path,actual_hash=verified_checkpoint(info)
 ck=torch.load(path,map_location='cpu',weights_only=True)
 if name=='m1':
  import segmentation_models_pytorch as smp
  model=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=1,activation=None)
  state=ck['model_state_dict'];details={'saved_best_epoch_one_based':ck['epoch']+1,'saved_loss':ck['loss']}
 elif name=='m2':
  train,_=m2_modules();model=train.build_model(hf_config=ck['hf_config']);state=ck['model']
  details={'epoch':ck['epoch'],'step':ck['step'],'hf_config':ck['hf_config']}
 else:raise ValueError('semantic model must be m1 or m2')
 model.load_state_dict(state,strict=True);model.to(device).eval()
 return model,dict(info,strict_load=True,model=name,details=details)

@torch.inference_mode()
def predict_m1(model,rgb,device):
 if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('M1 expects uint8 RGB')
 h,w=rgb.shape[:2];x=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
 x=(x-MEAN)/STD;x=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0).to(device)
 p=model(x).sigmoid()[0,0].float().cpu().numpy()
 return cv2.resize(p,(w,h),interpolation=cv2.INTER_LINEAR)

@torch.inference_mode()
def predict_m2(model,gray,device,tile=512,overlap=128,batch=1):
 if type(tile)is not int or tile<32 or tile%32 or type(overlap)is not int or not 0<=overlap<tile or type(batch)is not int or batch<1:raise ValueError('invalid tile/overlap/batch')
 if gray.dtype!=np.uint8 or gray.ndim!=2:raise ValueError('M2 expects uint8 grayscale')
 h,w=gray.shape
 padded=np.pad(gray,((0,max(0,tile-h)),(0,max(0,tile-w))),mode='edge')
 train,_=m2_modules()
 return train.predict_prob(model,padded,device,tile=tile,overlap=overlap,batch=batch)[:h,:w]

def yolo_record(result,observation_id,source):
 """Ultralytics result predicted with retina_masks=True -> original RLE contract."""
 from m3_filament.schema import make_record,make_instance,validate_record
 shape=tuple(result.orig_shape);instances=[]
 if result.masks is not None:
  masks=result.masks.data.cpu().numpy()
  if masks.shape[1:]!=shape:raise ValueError('YOLO masks must be original coordinates; set retina_masks=True')
  if len(masks)!=len(result.boxes.conf):raise ValueError('YOLO mask/score count mismatch')
  for m,score in zip(masks,result.boxes.conf.cpu().numpy()):
   mask=m>.5
   if mask.any():instances.append(make_instance(mask,len(instances)+1,float(score),source))
 out=make_record(observation_id,shape,instances,source);validate_record(out);return out
