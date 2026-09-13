#!/usr/bin/env python3
"""Audit of the supervised upper bound: evaluate the fine-tuned target checkpoint
(scripts/5.5.2) separately on the images it was trained on (target_*_train),
the images used for early stopping (target_*_val) and the full test list
(target_*_test, which is the union of the two). Output:
results/ub_audit/ub_{target}_{strategy}__{split}.json (metrics + per-sample preds).

    python scripts/5.5.4_audit_ub_splits.py --strategy hybrid
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
from torchvision import models, transforms

MODELS = Path('models')
SPLITS = Path('splits')
OUT = Path('results/ub_audit')
TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@torch.no_grad()
def evaluate(model: torch.nn.Module, csv_path: Path, device: str) -> dict:
    df = pd.read_csv(csv_path)
    preds: list[int] = []
    for i in range(0, len(df), 64):
        x = torch.stack(
            [TF(Image.open(p).convert('RGB')) for p in df.path.iloc[i : i + 64]]
        )
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
    y = df.label_id.tolist()
    return {
        'split': csv_path.name,
        'n': len(df),
        'accuracy': accuracy_score(y, preds),
        'macro_f1': f1_score(y, preds, average='macro', zero_division=0),
        'y_true': y,
        'y_pred': preds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='hybrid')
    ap.add_argument('--target', default='plantdoc')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(SPLITS / args.strategy / f'target_{args.target}_test.csv')
    train = pd.read_csv(SPLITS / args.strategy / f'target_{args.target}_train.csv')
    val = pd.read_csv(SPLITS / args.strategy / f'target_{args.target}_val.csv')
    overlap = {
        'n_test': len(test),
        'n_train': len(train),
        'n_val': len(val),
        'train_in_test': int(test.path.isin(train.path).sum()),
        'val_in_test': int(test.path.isin(val.path).sum()),
        'train_val_overlap': int(train.path.isin(val.path).sum()),
    }
    print('split overlap:', overlap)
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, test.label_id.nunique())
    model.load_state_dict(
        torch.load(
            MODELS / f'resnet50_target_{args.target}_{args.strategy}_best.pt',
            map_location='cpu',
            weights_only=False,
        )
    )
    model = model.to(device).eval()
    for split in ('train', 'val', 'test'):
        res = evaluate(
            model, SPLITS / args.strategy / f'target_{args.target}_{split}.csv', device
        )
        res.update(
            {'strategy': args.strategy, 'target': args.target, 'overlap': overlap}
        )
        (OUT / f'ub_{args.target}_{args.strategy}__{split}.json').write_text(
            json.dumps(res, indent=1)
        )
        print(
            f'{split:6s} n={res["n"]:5d} acc={res["accuracy"]:.4f} macro-F1={res["macro_f1"]:.4f}'
        )


if __name__ == '__main__':
    main()
