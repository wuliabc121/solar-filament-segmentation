import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils


class FilamentDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, image_ids=None):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.images = {img['id']: img for img in self.coco['images']}
        self.anns = {}
        for ann in self.coco['annotations']:
            if ann['image_id'] not in self.anns:
                self.anns[ann['image_id']] = []
            self.anns[ann['image_id']].append(ann)
        if image_ids is None:
            self.image_ids = list(self.images.keys())
        else:
            self.image_ids = image_ids
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create binary mask from all annotations
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.anns.get(img_id, []):
            if isinstance(ann['segmentation'], list):
                rles = maskUtils.frPyObjects(ann['segmentation'], h, w)
                rle = maskUtils.merge(rles)
            elif isinstance(ann['segmentation'], dict):
                if isinstance(ann['segmentation']['counts'], list):
                    rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                else:
                    rle = ann['segmentation']
            else:
                continue
            m = maskUtils.decode(rle)
            if len(m.shape) == 3:
                m = m[:, :, 0]
            mask[m > 0] = 1
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            if isinstance(image, torch.Tensor):
                if image.ndim == 3 and image.shape[0] != 3:
                    image = image.permute(2, 0, 1)
                image = image.float()
            else:
                image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        elif isinstance(mask, torch.Tensor):
            mask = mask.float().unsqueeze(0) if mask.ndim == 2 else mask.float()
        else:
            mask = torch.tensor(mask).float().unsqueeze(0)
        return image, mask, img_id, img_info['file_name']
