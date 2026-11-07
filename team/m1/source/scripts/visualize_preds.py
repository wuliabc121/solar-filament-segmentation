#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--thresh', type=float, default=0.5)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    img_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))])
    
    for img_name in tqdm(img_files):
        img_path = os.path.join(args.img_dir, img_name)
        pred_path = os.path.join(args.pred_dir, os.path.splitext(img_name)[0] + '.npy')
        if not os.path.exists(pred_path):
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = np.load(pred_path)
        mask = (pred > args.thresh).astype(np.uint8) * 255
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(pred, cmap='hot')
        axes[1].set_title('Pred Prob')
        axes[1].axis('off')
        axes[2].imshow(img)
        axes[2].imshow(mask, alpha=0.5, cmap='Reds')
        axes[2].set_title(f'Overlay (th={args.thresh})')
        axes[2].axis('off')
        plt.tight_layout()
        out_name = os.path.splitext(img_name)[0] + '.png'
        plt.savefig(os.path.join(args.output, out_name), dpi=150, bbox_inches='tight')
        plt.close()
    print(f'Saved visualizations to {args.output}')

if __name__ == '__main__':
    main()
