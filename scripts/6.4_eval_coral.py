import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# =====================================================
# Paths
# =====================================================
SPLITS = Path('splits')
MODELS = Path('models')
RESULTS = Path('results')

RESULTS.mkdir(exist_ok=True)


# =====================================================
# Dataset
# =====================================================
class CSVDataset(Dataset):
    def __init__(self, csv_path: Path):
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
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        return self.tf(img), int(row['label_id'])


# =====================================================
# Eval helper (WITH PRED DUMP)
# =====================================================
@torch.no_grad()
def eval_and_dump_preds(backbone, classifier, loader, device, out_csv):
    rows = []
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device)
        feats = backbone(x)
        logits = classifier(feats)
        p = logits.argmax(1).cpu().tolist()

        preds.extend(p)
        targets.extend(y.tolist())

        for i in range(len(p)):
            rows.append(
                {
                    'y_true': int(y[i]),
                    'y_pred': int(p[i]),
                }
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    return acc, f1


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--target', choices=['plantdoc', 'fcdd'], default='plantdoc')
    parser.add_argument('--lambda_coral', type=float, required=True)
    args = parser.parse_args()

    strategy = args.strategy
    target = args.target
    lam = args.lambda_coral

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device}')

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------
    pv_test = CSVDataset(SPLITS / strategy / 'pv_test.csv')
    tgt_test = CSVDataset(SPLITS / strategy / f'target_{target}_test.csv')

    pv_loader = DataLoader(pv_test, batch_size=64)
    tgt_loader = DataLoader(tgt_test, batch_size=64)

    num_classes = pv_test.df['label_id'].nunique()

    # -------------------------------------------------
    # Rebuild CORAL model (EXACT as in 6.3)
    # -------------------------------------------------
    backbone = models.resnet50()
    backbone.fc = nn.Identity()

    classifier = nn.Linear(2048, num_classes)

    model_path = MODELS / f'resnet50_coral_lambda_{lam}_{strategy}.pt'
    ckpt = torch.load(model_path, map_location=device)

    backbone.load_state_dict(ckpt['backbone'])
    classifier.load_state_dict(ckpt['classifier'])

    backbone.to(device).eval()
    classifier.to(device).eval()

    # -------------------------------------------------
    # Eval + dump preds
    # -------------------------------------------------
    pv_acc, pv_f1 = eval_and_dump_preds(
        backbone,
        classifier,
        pv_loader,
        device,
        RESULTS / f'preds_pv_coral_lambda_{lam}_{strategy}.csv',
    )

    tgt_acc, tgt_f1 = eval_and_dump_preds(
        backbone,
        classifier,
        tgt_loader,
        device,
        RESULTS / f'preds_target_coral_lambda_{lam}_{strategy}.csv',
    )

    metrics = {
        'lambda': lam,
        'strategy': strategy,
        'pv_accuracy': pv_acc,
        'pv_macro_f1': pv_f1,
        'target_accuracy': tgt_acc,
        'target_macro_f1': tgt_f1,
        'delta_f1': pv_f1 - tgt_f1,
    }

    out = RESULTS / f'coral_lambda_{lam}_{strategy}_metrics.json'
    out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    print(f'[OK] CORAL eval done: λ={lam}')


if __name__ == '__main__':
    main()
