"""
10.8 Train EfficientNet-Lite0 on PlantVillage (PyTorch / timm, GPU).

Fine-tunes timm `tf_efficientnet_lite0` (ImageNet pretrained) on PlantVillage
with the specified label-alignment strategy.

EfficientNet-Lite0 is the EdgeTPU-compatible variant of EfficientNet-B0:
  - ReLU6 activation (NOT Swish)
  - No Squeeze-Excitation blocks
  - All ops are EdgeTPU-safe

Reference: Tan et al., "EfficientNet: Rethinking Model Scaling" (ICML 2019),
EfficientNet-Lite: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite

Prerequisites:
    - splits/{strategy}/pv_train.csv
    - splits/{strategy}/pv_val.csv

Output:
    models/efficientnet_lite0_pv_{strategy}_best.pt

Usage:
    poetry run python scripts/10.8_train_efficientnet_lite0_pv.py --strategy hybrid
    poetry run python scripts/10.8_train_efficientnet_lite0_pv.py --strategy Fuzzy
    poetry run python scripts/10.8_train_efficientnet_lite0_pv.py --strategy sbert
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

CONFIG = Path('configs/train_resnet50.yaml')
SPLITS = Path('splits')
MODELS = Path('models')
MODELS.mkdir(exist_ok=True)


class CSVDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r['path']).convert('RGB')
        return self.transform(img), int(r['label_id'])


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune EfficientNet-Lite0 (timm) on PlantVillage'
    )
    parser.add_argument(
        '--strategy', required=True, choices=['hybrid', 'Fuzzy', 'sbert']
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.open())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device}')

    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])

    strategy = args.strategy
    batch_size = cfg['training']['batch_size']

    # -- Data --
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CSVDataset(SPLITS / strategy / 'pv_train.csv', train_tf)
    val_ds = CSVDataset(SPLITS / strategy / 'pv_val.csv', val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    num_classes = train_ds.df['label_id'].nunique()
    print(f'[INFO] Training EfficientNet-Lite0 for strategy={strategy}')
    print(f'       train={len(train_ds)}, val={len(val_ds)}, classes={num_classes}')

    # -- Model --
    model = timm.create_model(
        'tf_efficientnet_lite0', pretrained=True, num_classes=num_classes
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'       params={total_params:,}')

    # -- Optimizer + Scheduler --
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['training']['epochs']
    )
    ce = nn.CrossEntropyLoss()

    best_f1 = 0.0
    patience_counter = 0
    patience = cfg['training']['early_stopping_patience']

    # -- Train --
    for epoch in range(cfg['training']['epochs']):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        scheduler.step()

        # -- Validate --
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(device))
                preds.extend(out.argmax(1).cpu().tolist())
                ys.extend(y.tolist())

        f1 = f1_score(ys, preds, average='macro', zero_division=0)
        avg_loss = running_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]
        print(
            f'[Epoch {epoch:02d}] loss={avg_loss:.4f}  val_macro_f1={f1:.4f}  lr={lr_now:.6f}'
        )

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            save_path = MODELS / f'efficientnet_lite0_pv_{strategy}_best.pt'
            torch.save(model.state_dict(), save_path)
            print(f'         -> new best, saved {save_path}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[INFO] Early stopping at epoch {epoch} (patience={patience})')
                break

    print(f'[OK] EfficientNet-Lite0 training done, best_f1={best_f1:.4f}')


if __name__ == '__main__':
    main()
