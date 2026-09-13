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
RESULTS = Path('results')

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


class CSVDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
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
        return self.tf(img), int(r['label_id'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.open())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device}')

    train_ds = CSVDataset(SPLITS / args.strategy / 'pv_train.csv')
    val_ds = CSVDataset(SPLITS / args.strategy / 'pv_val.csv')

    train_loader = DataLoader(
        train_ds, batch_size=cfg['training']['batch_size'], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'])

    num_classes = train_ds.df['label_id'].nunique()

    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )
    ce = nn.CrossEntropyLoss()

    best_f1 = 0.0

    for epoch in range(cfg['training']['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()

        # simple val macro-F1
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(device))
                preds.extend(out.argmax(1).cpu().tolist())
                ys.extend(y.tolist())

        from sklearn.metrics import f1_score

        f1 = f1_score(ys, preds, average='macro', zero_division=0)
        print(f'[Epoch {epoch:02d}] val_macro_f1={f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                model.state_dict(), MODELS / f'mobilenetv2_pv_{args.strategy}_best.pt'
            )

    print('[OK] MobileNetV2 PV training done')


if __name__ == '__main__':
    main()
