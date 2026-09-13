import argparse
import csv
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import yaml

# =====================================================
# Paths
# =====================================================
CONFIG = Path('configs/train_resnet50.yaml')
SPLITS = Path('splits')
MODELS = Path('models')
RESULTS = Path('results')

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


# =====================================================
# Dataset
# =====================================================
class CSVDataset(Dataset):
    def __init__(self, csv_path: Path, train: bool):
        self.df = pd.read_csv(csv_path)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        tf = [
            transforms.Resize((224, 224)),
        ]

        if train:
            tf.append(transforms.RandomHorizontalFlip())

        tf += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        self.tf = transforms.Compose(tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        return self.tf(img), int(row['label_id'])


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--target', choices=['plantdoc', 'fcdd'], default='plantdoc')
    parser.add_argument('--freeze_epochs', type=int, default=3)
    args = parser.parse_args()

    strategy = args.strategy
    target = args.target

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device}')

    cfg = yaml.safe_load(CONFIG.open())

    # reproducibility
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])

    # -------------------------------------------------
    # Datasets
    # -------------------------------------------------
    train_ds = CSVDataset(
        SPLITS / strategy / f'target_{target}_train.csv',
        train=True,
    )
    val_ds = CSVDataset(
        SPLITS / strategy / f'target_{target}_val.csv',
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        num_workers=4,
        pin_memory=True,
    )

    num_classes = train_ds.df['label_id'].nunique()

    # -------------------------------------------------
    # Load PV-pretrained model
    # -------------------------------------------------
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(
        torch.load(
            MODELS / f'resnet50_pv_{strategy}_best.pt',
            map_location=device,
        )
    )
    model.to(device)

    # -------------------------------------------------
    # Optimizer & loss
    # -------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'] * 0.1,  # smaller LR for fine-tuning
        weight_decay=cfg['training']['weight_decay'],
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0.0
    patience = 0

    out_model = MODELS / f'resnet50_target_{target}_{strategy}_best.pt'
    log_csv = RESULTS / f'train_log_target_{target}_{strategy}.csv'

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    with log_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_macro_f1'])

        for epoch in range(cfg['training']['epochs']):
            # Freeze backbone for first N epochs
            if epoch < args.freeze_epochs:
                for name, p in model.named_parameters():
                    p.requires_grad = name.startswith('fc')
            else:
                for p in model.parameters():
                    p.requires_grad = True

            # ---- train ----
            model.train()
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            # ---- validate ----
            model.eval()
            preds, targets = [], []

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    out = model(x)
                    preds.extend(out.argmax(1).cpu().tolist())
                    targets.extend(y.cpu().tolist())

            f1 = f1_score(targets, preds, average='macro', zero_division=0)
            writer.writerow([epoch, f1])
            f.flush()

            if f1 > best_f1:
                best_f1 = f1
                patience = 0
                torch.save(model.state_dict(), out_model)
            else:
                patience += 1

            if patience >= cfg['training']['early_stopping_patience']:
                print('[INFO] Early stopping')
                break

    print(f'[OK] Fine-tuning completed: {out_model}')


if __name__ == '__main__':
    main()
