import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import FilamentDataset
from models.unet import get_model, get_loss


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    os.makedirs(cfg['output_dir'], exist_ok=True)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(cfg['log_dir'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_transform = A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    
    train_dataset = FilamentDataset(
        img_dir=cfg['train_img_dir'],
        ann_file=cfg['train_ann'],
        transform=train_transform
    )
    all_ids = list(train_dataset.image_ids)
    val_size = 50
    if len(all_ids) > val_size:
        val_ids = all_ids[-val_size:]
        train_ids = all_ids[:-val_size]
    else:
        val_ids = all_ids
        train_ids = all_ids
    val_dataset = FilamentDataset(
        img_dir=cfg['train_img_dir'],
        ann_file=cfg['train_ann'],
        transform=val_transform,
        image_ids=val_ids
    )
    train_dataset = FilamentDataset(
        img_dir=cfg['train_img_dir'],
        ann_file=cfg['train_ann'],
        transform=train_transform,
        image_ids=train_ids
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_model(cfg['encoder'], cfg['encoder_weights']).to(device)
    criterion = get_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    best_loss = float('inf')
    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0
        for imgs, masks, _, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["epochs"]} [Train]'):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, _, _ in tqdm(val_loader, desc=f'Epoch {epoch+1}/{cfg["epochs"]} [Val]'):
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(cfg['checkpoint_dir'], 'best_model.pth'))
            print(f'Saved best model')
        
        scheduler.step()
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main()
