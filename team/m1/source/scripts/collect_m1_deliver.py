#!/usr/bin/env python3
import json
import os
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
M1_CODE = ROOT / 'M1_code'
DELIV = ROOT / 'M1_deliver'
DATA_ROOT = ROOT / 'MAGFiLO_1.0_Kaggle_2026'

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    # Copy weights
    weights_src = M1_CODE / 'weights' / 'best_model.pth'
    if weights_src.exists():
        shutil.copy2(weights_src, DELIV / 'weights' / 'best_model.pth')
        print(f'Copied weights: {weights_src}')
    
    # Copy logs
    for logf in (M1_CODE / 'logs').glob('*'):
        if logf.is_file():
            shutil.copy2(logf, DELIV / 'logs' / logf.name)
    print('Copied logs')
    
    # Copy configs
    for cfg in (M1_CODE / 'configs').glob('*.yaml'):
        shutil.copy2(cfg, DELIV / 'configs' / cfg.name)
    if (ROOT / 'install_env.sh').exists():
        shutil.copy2(ROOT / 'install_env.sh', DELIV / 'scripts' / 'install_env.sh')
    
    # Load config
    with open(M1_CODE / 'configs' / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    # Build split manifest (last 50 as val as per train.py)
    ann_file = DATA_ROOT / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    img_dir = DATA_ROOT / 'train' / 'train_images'
    
    split_info = {
        'created_at': datetime.now().isoformat(),
        'method': 'last_50_val',
        'val_count': 50,
        'train_count': None,
        'img_dir': str(img_dir),
        'ann_file': str(ann_file),
        'split_hash': None,
        'note': 'As per M1_code/scripts/train.py: val_ids = train_ids[-50:]'
    }
    
    # Compute hash of config
    split_info['split_hash'] = sha256_file(M1_CODE / 'configs' / 'config.yaml')[:16]
    
    with open(DELIV / 'manifest' / 'split_manifest.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    print('Created split manifest')
    
    # Generate OOF/probs placeholder info
    oof_info = {
        'description': 'OOF probabilities for training folds - generate with scripts/generate_oof.py using fold CV',
        'status': 'pending',
        'format': 'npz with keys: probs, image_ids, coords, meta',
        'target': str(DELIV / 'oof')
    }
    with open(DELIV / 'oof' / 'README.json', 'w') as f:
        json.dump(oof_info, f, indent=2)
    
    probs_info = {
        'description': 'Inference probabilities on calibration/test sets',
        'status': 'pending',
        'format': 'npz',
        'target': str(DELIV / 'probs')
    }
    with open(DELIV / 'probs' / 'README.json', 'w') as f:
        json.dump(probs_info, f, indent=2)
    
    # Create deliver readme
    readme = f"""# M1 Deliverables - CNN Semantic Segmentation (U-Net + ResNet34)

Generated: {datetime.now().isoformat()}

## Model
- Architecture: segmentation-models-pytorch Unet
- Encoder: resnet34 (ImageNet pretrained)
- Input size: {cfg.get('img_size', 512)}
- Loss: BCEWithLogitsLoss + DiceLoss
- Optimizer: AdamW (lr={cfg.get('lr')}, weight_decay={cfg.get('weight_decay')})

## Training
- Config: configs/config.yaml
- Weights: weights/best_model.pth
- Logs: logs/ (TensorBoard)
- GPU: CUDA_VISIBLE_DEVICES=2,3 (RTX 3090)

## Data
- Train images: {DATA_ROOT}/train/train_images
- Annotations: {DATA_ROOT}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
- Split: manifest/split_manifest.json (last 50 for validation)

## For M3/M4
- OOF probabilities: oof/ (to be generated)
- Inference probabilities: probs/

## Reproduction
```bash
cd M1_code
source $(conda info --base)/etc/profile.d/conda.sh && conda activate shape-for-motion
CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py configs/config.yaml
```
"""
    with open(DELIV / 'reports' / 'm1_deliver_readme.md', 'w') as f:
        f.write(readme)
    
    # Deliver manifest
    deliver_manifest = {
        'module': 'M1',
        'task': 'CNN semantic segmentation with U-Net + ResNet34',
        'generated_at': datetime.now().isoformat(),
        'files': {
            'weights': ['weights/best_model.pth'],
            'configs': ['configs/config.yaml'],
            'logs': [f.name for f in (DELIV / 'logs').glob('*') if f.is_file()],
            'manifest': ['manifest/split_manifest.json'],
            'scripts': ['scripts/install_env.sh'],
            'reports': ['reports/m1_deliver_readme.md']
        },
        'environment': {
            'pytorch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None
        }
    }
    with open(DELIV / 'reports' / 'deliver_manifest.json', 'w') as f:
        json.dump(deliver_manifest, f, indent=2)
    
    print('M1 deliverables collected to', DELIV)

if __name__ == '__main__':
    main()
