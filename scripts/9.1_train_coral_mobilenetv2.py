import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import yaml

CONFIG = Path('configs/train_resnet50.yaml')
SPLITS = Path('splits')
MODELS = Path('models')


class CSVDataset(Dataset):
    def __init__(self, csv_path, labeled):
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
        r = self.df.iloc[idx]
        img = Image.open(r['path']).convert('RGB')
        if self.labeled:
            return self.tf(img), int(r['label_id'])
        return self.tf(img)


def coral_loss(xs, xt):
    xs = xs - xs.mean(0)
    xt = xt - xt.mean(0)
    cs = (xs.T @ xs) / (xs.size(0) - 1)
    ct = (xt.T @ xt) / (xt.size(0) - 1)
    d = xs.size(1)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--strategy', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--lambda_coral', type=float, required=True)
    p.add_argument('--epochs', type=int, default=20)
    args = p.parse_args()

    cfg = yaml.safe_load(CONFIG.open())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_ds = CSVDataset(SPLITS / args.strategy / 'pv_train.csv', True)
    tgt_ds = CSVDataset(
        SPLITS / args.strategy / f'target_{args.target}_unlabeled.csv', False
    )

    src_loader = DataLoader(
        src_ds, batch_size=cfg['training']['batch_size'], shuffle=True, drop_last=True
    )
    tgt_loader = DataLoader(
        tgt_ds, batch_size=cfg['training']['batch_size'], shuffle=True, drop_last=True
    )
    tgt_iter = iter(tgt_loader)

    num_classes = src_ds.df['label_id'].nunique()

    base = models.mobilenet_v2(weights=None)
    base.classifier = nn.Identity()
    clf = nn.Linear(1280, num_classes)

    state = torch.load(
        MODELS / f'mobilenetv2_pv_{args.strategy}_best.pt', map_location=device
    )
    base.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith('classifier')}, strict=False
    )
    clf.load_state_dict(
        {
            k.replace('classifier.1.', ''): v
            for k, v in state.items()
            if k.startswith('classifier.1')
        }
    )

    base.to(device)
    clf.to(device)

    opt = torch.optim.AdamW(
        list(base.parameters()) + list(clf.parameters()),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    ce = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        base.train()
        clf.train()
        for xs, ys in src_loader:
            try:
                xt = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt = next(tgt_iter)

            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            fs = base(xs)
            ft = base(xt)
            logits = clf(fs)

            loss = ce(logits, ys) + args.lambda_coral * coral_loss(fs, ft)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f'[Epoch {epoch:02d}] CORAL done')

    out = MODELS / f'mobilenetv2_coral_lambda_{args.lambda_coral}_{args.strategy}.pt'
    torch.save({'backbone': base.state_dict(), 'classifier': clf.state_dict()}, out)
    print(f'[OK] Saved {out}')


if __name__ == '__main__':
    main()
