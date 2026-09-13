import argparse
import csv
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision.transforms as T
import yaml

CONFIG = Path('configs/train_resnet50.yaml')
SPLITS = Path('splits')
MODELS = Path('models')
RESULTS = Path('results')

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


class CSVDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        return self.transform(img), int(row['label_id'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()
    strategy = args.strategy

    # ✅ CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    cfg = yaml.safe_load(CONFIG.open())
    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])

    # ----------------------------
    # Transforms
    # ----------------------------
    train_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(cfg['augmentation']['train']['rotation_deg']),
            T.ColorJitter(**cfg['augmentation']['train']['color_jitter']),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    # ----------------------------
    # Datasets & loaders
    # ----------------------------
    train_ds = CSVDataset(SPLITS / strategy / 'pv_train.csv', train_tf)
    val_ds = CSVDataset(SPLITS / strategy / 'pv_val.csv', eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training'].get('num_workers', 4),
        pin_memory=(device.type == 'cuda'),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training'].get('num_workers', 4),
        pin_memory=(device.type == 'cuda'),
    )

    num_classes = train_ds.df['label_id'].nunique()

    # ----------------------------
    # Model: MobileNetV2
    # ----------------------------
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0.0
    patience = 0

    run_name = f'mobilenet_baseline_pv_{strategy}'
    model_path = MODELS / f'{run_name}_best.pt'
    log_path = RESULTS / f'train_log_{run_name}.csv'

    # ----------------------------
    # Training loop
    # ----------------------------
    with log_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'epoch',
                'train_loss',
                'val_loss',
                'val_precision_macro',
                'val_recall_macro',
                'val_f1_macro',
            ]
        )

        for epoch in range(cfg['training']['epochs']):
            # ---- train ----
            model.train()
            train_loss = 0.0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # ---- validation ----
            model.eval()
            val_loss = 0.0
            preds, targets = [], []

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    out = model(x)
                    loss = criterion(out, y)

                    val_loss += loss.item()
                    preds.extend(out.argmax(1).cpu().tolist())
                    targets.extend(y.cpu().tolist())

            val_precision = precision_score(
                targets, preds, average='macro', zero_division=0
            )
            val_recall = recall_score(targets, preds, average='macro', zero_division=0)
            val_f1 = f1_score(targets, preds, average='macro', zero_division=0)

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    val_precision,
                    val_recall,
                    val_f1,
                ]
            )
            f.flush()

            print(f'[Epoch {epoch:02d}] val_f1={val_f1:.4f}')

            # ---- early stopping ----
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience += 1

            if patience >= cfg['training']['early_stopping_patience']:
                print('[INFO] Early stopping triggered')
                break

    print(f"[OK] MobileNetV2 baseline training finished for strategy='{strategy}'")


if __name__ == '__main__':
    main()
