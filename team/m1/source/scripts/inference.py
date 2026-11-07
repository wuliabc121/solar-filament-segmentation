#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(cfg['encoder'], cfg['encoder_weights']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    transform = A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    
    img_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))])
    for img_name in tqdm(img_files):
        img_path = os.path.join(args.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        aug = transform(image=img)
        x = aug['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x)
            pred = torch.sigmoid(pred)[0,0].cpu().numpy()
        pred = cv2.resize(pred, (w,h))
        out_name = os.path.splitext(img_name)[0] + '.npy'
        np.save(os.path.join(args.output, out_name), pred)
    print(f'Saved {len(img_files)} predictions to {args.output}')

if __name__ == '__main__':
    main()
