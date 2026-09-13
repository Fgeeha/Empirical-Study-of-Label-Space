import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch import nn
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
    def __init__(self, csv_path: Path, labeled: bool):
        self.df = pd.read_csv(csv_path)
        self.labeled = labeled

        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')

        if self.labeled:
            return self.tf(img), int(row['label_id'])
        return self.tf(img)


# =====================================================
# CORAL loss
# =====================================================
def coral_loss(source, target):
    d = source.size(1)

    source = source - source.mean(dim=0)
    target = target - target.mean(dim=0)

    cs = (source.T @ source) / (source.size(0) - 1)
    ct = (target.T @ target) / (target.size(0) - 1)

    return ((cs - ct) ** 2).sum() / (4 * d * d)


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--lambda_coral', type=float, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None, help='Override config seed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device}')

    cfg = yaml.safe_load(CONFIG.open())
    seed = args.seed if args.seed is not None else cfg['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------
    source_ds = CSVDataset(
        SPLITS / args.strategy / 'pv_train.csv',
        labeled=True,
    )
    target_ds = CSVDataset(
        SPLITS / args.strategy / f'target_{args.target}_unlabeled.csv',
        labeled=False,
    )

    source_loader = DataLoader(
        source_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        drop_last=True,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        drop_last=True,
    )
    target_iter = iter(target_loader)

    num_classes = source_ds.df['label_id'].nunique()

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    backbone = models.resnet50()
    backbone.fc = nn.Identity()

    classifier = nn.Linear(2048, num_classes)

    state = torch.load(
        MODELS / f'resnet50_pv_{args.strategy}_best.pt',
        map_location=device,
    )

    backbone.load_state_dict(
        {k.replace('fc.', ''): v for k, v in state.items() if not k.startswith('fc')}
    )
    classifier.load_state_dict(
        {k.replace('fc.', ''): v for k, v in state.items() if k.startswith('fc')}
    )

    backbone.to(device)
    classifier.to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    ce = nn.CrossEntropyLoss()

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    backbone.train()
    classifier.train()

    for epoch in range(args.epochs):
        total_loss = 0.0

        for xs, ys in source_loader:
            try:
                xt = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt = next(target_iter)

            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            fs = backbone(xs)
            ft = backbone(xt)

            logits = classifier(fs)

            loss = ce(logits, ys) + args.lambda_coral * coral_loss(fs, ft)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'[Epoch {epoch:02d}] loss={total_loss / len(source_loader):.4f}')

    seed_suffix = f'_seed{seed}' if seed != 42 else ''
    out_model = (
        MODELS
        / f'resnet50_coral_lambda_{args.lambda_coral}_{args.strategy}{seed_suffix}.pt'
    )
    torch.save(
        {
            'backbone': backbone.state_dict(),
            'classifier': classifier.state_dict(),
        },
        out_model,
    )

    print(f'[OK] CORAL model saved → {out_model}')


if __name__ == '__main__':
    main()
