"""Mask-native target loader for the pinned Ultralytics segmentation trainer.

COCO masks retain topology in the training cache. Only the explicitly recorded
network resize/letterbox and optional horizontal flip transform the targets.
"""
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from ultralytics.models.yolo.segment import SegmentationTrainer
from .io import read_jsonl
from .masks import decode,bbox


class NativeMaskDataset(Dataset):
    rect=False
    mosaic=False

    def __init__(self,root,split,imgsz,fliplr=0.,crop_size=0,crop_probability=0.):
        self.rows=[r for r in read_jsonl(Path(root)/'samples.jsonl') if r['split']==split]
        if not self.rows:raise ValueError(f'empty native-mask dataset: {split}')
        self.imgsz=imgsz;self.fliplr=fliplr
        self.crop_size=crop_size;self.crop_probability=crop_probability
        self.im_files=[r['image_path'] for r in self.rows]
        self.labels=[{'cls':np.zeros((r['instance_count'],1),np.float32)} for r in self.rows]

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row['image_path']) as im:image=np.asarray(im.convert('RGB'))
        h,w=image.shape[:2];source_shape=(h,w)
        target=json.loads(Path(row['target_path']).read_text())
        originals=[decode(rle) for rle in target['masks']]
        if any(m.shape!=(h,w) for m in originals):raise ValueError('native target differs from original image dimensions')
        crop=[0,0,w,h]
        if self.crop_size and originals and random.random()<self.crop_probability:
            size=min(self.crop_size,h,w);m=random.choice(originals);x0,y0,x1,y1=bbox(m)
            cx=(x0+x1)/2+random.uniform(-.2,.2)*size;cy=(y0+y1)/2+random.uniform(-.2,.2)*size
            left=max(0,min(w-size,round(cx-size/2)));top=max(0,min(h-size,round(cy-size/2)))
            crop=[left,top,left+size,top+size]
            image=image[top:top+size,left:left+size]
            originals=[m[top:top+size,left:left+size] for m in originals]
            originals=[m for m in originals if m.any()]
            h,w=image.shape[:2]
        s=self.imgsz;gain=min(s/h,s/w)
        nh,nw=round(h*gain),round(w*gain);top=(s-nh)//2;left=(s-nw)//2
        canvas=np.full((s,s,3),114,np.uint8)
        canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
        masks=[];boxes=[]
        flip=random.random()<self.fliplr
        for m in originals:
            x0,y0,x1,y1=bbox(m)
            resized=np.zeros((s,s),np.uint8)
            resized[top:top+nh,left:left+nw]=cv2.resize(m,(nw,nh),interpolation=cv2.INTER_NEAREST)
            bx0=x0*gain+left;bx1=x1*gain+left;by0=y0*gain+top;by1=y1*gain+top
            if flip:resized=np.ascontiguousarray(resized[:,::-1]);bx0,bx1=s-bx1,s-bx0
            masks.append(resized)
            boxes.append([(bx0+bx1)/(2*s),(by0+by1)/(2*s),(bx1-bx0)/s,(by1-by0)/s])
        if flip:canvas=np.ascontiguousarray(canvas[:,::-1])
        n=len(masks)
        return {'img':torch.from_numpy(np.ascontiguousarray(canvas.transpose(2,0,1))),
                'cls':torch.zeros((n,1),dtype=torch.float32),
                'bboxes':torch.tensor(boxes,dtype=torch.float32).reshape(n,4),
                'masks':torch.from_numpy(np.stack(masks) if masks else np.zeros((0,s,s),np.uint8)),
                'batch_idx':torch.zeros(n,dtype=torch.float32),'im_file':row['image_path'],
                'ori_shape':(h,w),'resized_shape':(s,s),'ratio_pad':((gain,gain),(left,top)),
                'source_shape':source_shape,'crop_xyxy':crop}

    @staticmethod
    def collate_fn(batch):
        output={}
        for k in batch[0]:
            values=[b[k] for b in batch]
            if k=='img':output[k]=torch.stack(values)
            elif k=='batch_idx':output[k]=torch.cat([v+i for i,v in enumerate(values)])
            elif k in ('masks','cls','bboxes'):output[k]=torch.cat(values)
            else:output[k]=tuple(values)
        return output


class NativeSegmentationTrainer(SegmentationTrainer):
    def __init__(self,*args,training_crop_size=0,training_crop_probability=0.,**kwargs):
        self.training_crop_size=training_crop_size;self.training_crop_probability=training_crop_probability
        super().__init__(*args,**kwargs)

    def build_dataset(self,img_path,mode='train',batch=None):
        if self.args.overlap_mask or self.args.mask_ratio!=1:
            raise ValueError('native targets require overlap_mask=False and mask_ratio=1')
        unsupported=('mosaic','mixup','copy_paste','degrees','translate','scale','flipud','shear','perspective')
        if any(getattr(self.args,k,0) for k in unsupported) or self.args.multi_scale:
            raise ValueError('native mask loader supports only fixed letterbox and horizontal flip; disable unsupported augmentations')
        return NativeMaskDataset(self.data['path'],'train' if mode=='train' else 'calibration',
                                 self.args.imgsz,self.args.fliplr if mode=='train' else 0.,
                                 self.training_crop_size if mode=='train' else 0,
                                 self.training_crop_probability if mode=='train' else 0.)
