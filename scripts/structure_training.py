"""Controlled native-mask training and foreground topology-loss experiment.

Soft skeleton equations follow Shit et al., arXiv:2003.07311 and author
jocpae/clDice code. Here terms are per assigned instance, not batch-global;
foreground only, three erosion iterations, no topology guarantee claimed.
"""
import argparse
import copy
import json
import math
import random
import shutil
import signal
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask
from m3_filament.native import NativeMaskDataset,NativeSegmentationTrainer
from m3_filament.model import train_model
from m3_filament.masks import decode,bbox
from m3_filament.io import sha256

TRACE=[]
ARM='control'


def coverage_targets(masks,size):
    return F.interpolate(masks[:,None].float(),size=size,mode='area')[:,0]


def small_weights(masks):
    areas=np.array([int(m.sum()) for m in masks],float)
    return np.clip(np.sqrt(max(areas.max(),1)/np.maximum(areas,1)),1,4).tolist()


def rotate_pair(image,masks,k):
    return np.ascontiguousarray(np.rot90(image,k)),[np.ascontiguousarray(np.rot90(m,k)) for m in masks]


def soft_skeleton(x,iterations=3):
    def erode(a):return torch.minimum(-F.max_pool2d(-a,(3,1),1,(1,0)),-F.max_pool2d(-a,(1,3),1,(0,1)))
    def opening(a):return F.max_pool2d(erode(a),3,1,1)
    skeleton=F.relu(x-opening(x))
    for _ in range(iterations):
        x=erode(x);delta=F.relu(x-opening(x));skeleton=skeleton+F.relu(delta-skeleton*delta)
    return skeleton


def structure_terms(probability,target):
    dims=(-3,-2,-1);smooth=1.
    dice=1-(2*(probability*target).sum(dims)+smooth)/(probability.sum(dims)+target.sum(dims)+smooth)
    sp=soft_skeleton(probability);st=soft_skeleton(target)
    precision=((sp*target).sum(dims)+smooth)/(sp.sum(dims)+smooth)
    sensitivity=((st*probability).sum(dims)+smooth)/(st.sum(dims)+smooth)
    cl=1-2*precision*sensitivity/(precision+sensitivity).clamp_min(1e-7)
    return dice,cl


def differentiable_crop(masks,boxes):
    # Pinned upstream crop_mask mutates inputs for N<50, which invalidates
    # SigmoidBackward and can mutate GT. Keep its rounding convention, but
    # use an out-of-place ROI multiplication for training gradients.
    boxes=boxes.round() if len(masks)<50 else boxes
    x1,y1,x2,y2=boxes[:, :,None].chunk(4,1)
    y=torch.arange(masks.shape[-2],device=masks.device)[None,:,None]
    x=torch.arange(masks.shape[-1],device=masks.device)[None,None,:]
    return masks*((x>=x1)&(x<x2)&(y>=y1)&(y<y2))


class StructureLoss(v8SegmentationLoss):
    def __call__(self,preds,batch):
        proto=(preds if len(preds)==3 else preds[1])[2]
        prepared=dict(batch)
        if tuple(batch['masks'].shape[-2:])!=tuple(proto.shape[-2:]):
            prepared['masks']=coverage_targets(batch['masks'],proto.shape[-2:])
        return super().__call__(preds,prepared)

    @staticmethod
    def single_mask_loss(gt_mask,pred,proto,xyxy,area):
        logits=torch.einsum('in,nhw->ihw',pred,proto)
        bce=(differentiable_crop(F.binary_cross_entropy_with_logits(logits,gt_mask,reduction='none'),xyxy).mean((1,2))/area.clamp_min(1e-8)).sum()
        probability=differentiable_crop(logits.sigmoid(),xyxy)[:,None]
        target=differentiable_crop(gt_mask,xyxy)[:,None]
        dice,cl=structure_terms(probability,target)
        return bce+.5*dice.sum()+.2*cl.sum()


class FocusedDataset(NativeMaskDataset):
    def __init__(self,*args,training=False,**kwargs):
        super().__init__(*args,**kwargs);self.training=training

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row['image_path']) as im:image=np.asarray(im.convert('RGB'))
        h,w=image.shape[:2];source_shape=(h,w)
        originals=[decode(rle) for rle in json.loads(Path(row['target_path']).read_text())['masks']]
        if any(m.shape!=(h,w) for m in originals):raise ValueError('target shape mismatch')
        crop=[0,0,w,h]
        if self.crop_size and originals and random.random()<self.crop_probability:
            size=min(self.crop_size,h,w);m=random.choices(originals,weights=small_weights(originals),k=1)[0]
            x0,y0,x1,y1=bbox(m);cx=(x0+x1)/2+random.uniform(-.2,.2)*size;cy=(y0+y1)/2+random.uniform(-.2,.2)*size
            left=max(0,min(w-size,round(cx-size/2)));top=max(0,min(h-size,round(cy-size/2)))
            crop=[left,top,left+size,top+size];image=image[top:top+size,left:left+size]
            originals=[m[top:top+size,left:left+size] for m in originals];originals=[m for m in originals if m.any()]
        turns=random.randrange(1,4) if self.training and random.random()<.5 else 0
        image,originals=rotate_pair(image,originals,turns);h,w=image.shape[:2]
        if self.training:
            image=np.clip(image.astype(np.float32)*random.uniform(.9,1.1)+random.uniform(-8,8),0,255).astype(np.uint8)
        s=self.imgsz;gain=min(s/h,s/w);nh,nw=round(h*gain),round(w*gain);top=(s-nh)//2;left=(s-nw)//2
        canvas=np.full((s,s,3),114,np.uint8);canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))
        masks=[];boxes=[];flip=random.random()<self.fliplr
        for mask in originals:
            resized=np.zeros((s,s),np.uint8);resized[top:top+nh,left:left+nw]=cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
            if flip:resized=np.ascontiguousarray(resized[:,::-1])
            if not resized.any():continue
            x0,y0,x1,y1=bbox(resized);masks.append(resized)
            boxes.append([(x0+x1)/(2*s),(y0+y1)/(2*s),(x1-x0)/s,(y1-y0)/s])
        if flip:canvas=np.ascontiguousarray(canvas[:,::-1])
        n=len(masks)
        return {'img':torch.from_numpy(np.ascontiguousarray(canvas.transpose(2,0,1))),
          'cls':torch.zeros((n,1)),'bboxes':torch.tensor(boxes,dtype=torch.float32).reshape(n,4),
          'masks':torch.from_numpy(np.stack(masks) if masks else np.zeros((0,s,s),np.uint8)),
          'batch_idx':torch.zeros(n),'im_file':row['image_path'],'ori_shape':(h,w),'resized_shape':(s,s),
          'ratio_pad':((gain,gain),(left,top)),'source_shape':source_shape,'crop_xyxy':crop,'quarter_turns':turns}


def persist(folder):
    folder=Path(folder)
    if not folder.exists():return
    destination=folder/'code_snapshot/external/structure_training.py';destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,destination)
    metadata={'arm':ARM,'external_script_sha256':sha256(__file__),'external_snapshot':str(destination),
      'loss':'BCE+.5Dice+.2clDice(area targets)' if ARM=='structure' else 'original Ultralytics BCE(nearest targets)',
      'checkpoint_selection':'metrics/mAP50-95(M)','foreground_only':True,'skeleton_iterations':3,
      'crop_probability':.25,'small_target_weights':'inverse sqrt area capped [1,4]',
      'quarter_turn_probability':.5,'photometric_contrast':[.9,1.1],'brightness':[-8,8],
      'native_original_GT_unchanged':True,'source_hash_scope':'src; external code separately hashed'}
    def save(path,d):
        temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(path)
    save(folder/'external_training.json',metadata);save(folder/'selection_trace.json',TRACE)
    save(folder/'code_snapshot/external_source_manifest.json',{'external/structure_training.py':sha256(destination)})
    for name in ['run_manifest_started.json','run_manifest.json','metrics.json']:
        path=folder/name
        if path.exists():d=json.loads(path.read_text());d['external_training']=metadata;save(path,d)


class FocusedTrainer(NativeSegmentationTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);persist(self.save_dir)

    def build_dataset(self,img_path,mode='train',batch=None):
        if self.args.overlap_mask or self.args.mask_ratio!=1:raise ValueError('native masks required')
        return FocusedDataset(self.data['path'],'train' if mode=='train' else 'calibration',self.args.imgsz,
          self.args.fliplr if mode=='train' else 0.,self.training_crop_size if mode=='train' else 0,
          self.training_crop_probability if mode=='train' else 0.,training=mode=='train')

    def validate(self):
        previous=self.best_fitness;metrics,combined=super().validate()
        if metrics is None:return metrics,combined
        value=float(metrics['metrics/mAP50-95(M)'])
        if not math.isfinite(value):raise ValueError('nonfinite validation')
        self.best_fitness=max(previous if previous is not None else -math.inf,value)
        TRACE.append({'epoch':int(self.epoch)+1,'mask_ap95':value,'combined_fitness':float(combined),'best_mask_ap95':self.best_fitness})
        return metrics,value


def main():
    global ARM
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--arm',choices=['control','structure'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();ARM=a.arm
    import m3_filament.native as native
    from ultralytics.nn.tasks import SegmentationModel
    native.NativeSegmentationTrainer=FocusedTrainer
    if ARM=='structure':SegmentationModel.init_criterion=lambda self:StructureLoss(self)
    signal.signal(signal.SIGTERM,lambda n,f:(_ for _ in ()).throw(KeyboardInterrupt('owned experiment timeout')))
    try:print(json.dumps(train_model(a.config,a.run_id,a.smoke,development=not a.smoke),indent=2),flush=True)
    finally:persist(Path('runs')/a.run_id)


if __name__=='__main__':main()
