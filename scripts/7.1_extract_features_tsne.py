import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SPLITS = Path('splits')
MODELS = Path('models')
RESULTS = Path('results')

RESULTS.mkdir(exist_ok=True)


# =========================
# Dataset
# =========================
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


# =========================
# Feature extraction
# =========================
@torch.no_grad()
def extract_features(backbone, loader, device):
    feats, labels = [], []

    for x, y in loader:
        x = x.to(device)
        f = backbone(x)
        feats.append(f.cpu())
        labels.extend(y.tolist())

    return torch.cat(feats).numpy(), labels


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--target', default='plantdoc')
    parser.add_argument('--mode', choices=['baseline', 'coral'], required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device}')

    dataset = CSVDataset(SPLITS / args.strategy / f'target_{args.target}_test.csv')
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # ---------------------
    # Backbone
    # ---------------------
    backbone = models.resnet50()
    backbone.fc = nn.Identity()

    if args.mode == 'baseline':
        state = torch.load(
            MODELS / f'resnet50_pv_{args.strategy}_best.pt',
            map_location=device,
        )
        backbone.load_state_dict(
            {
                k.replace('fc.', ''): v
                for k, v in state.items()
                if not k.startswith('fc')
            }
        )
    else:
        ckpt = torch.load(
            MODELS / f'resnet50_coral_lambda_0.1_{args.strategy}.pt',
            map_location=device,
        )
        backbone.load_state_dict(ckpt['backbone'])

    backbone.to(device).eval()

    # ---------------------
    # Extract & save
    # ---------------------
    X, y = extract_features(backbone, loader, device)

    out = RESULTS / f'tsne_features_{args.mode}_{args.strategy}.npz'
    np.savez(out, X=X, y=y)

    print(f'[OK] Features saved → {out}')


if __name__ == '__main__':
    main()
