"""Collect multi-seed CORAL results and compute mean±std.

Run after run_coral_multiseed.sh completes.
Updates result_registry.csv with per-seed and aggregate rows.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MODELS = Path('models')
SPLITS = Path('splits')
RESULTS = Path('results')


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class CSVDataset(Dataset):
    def __init__(self, csv_path):
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.tf = TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        return self.tf(img), int(row['label_id'])


def load_coral_checkpoint(ckpt_path: Path, num_classes: int):
    model = models.resnet50()
    model.fc = torch.nn.Identity()
    classifier = torch.nn.Linear(2048, num_classes)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'backbone' in state and 'classifier' in state:
        model.load_state_dict(state['backbone'])
        classifier.load_state_dict(state['classifier'])
    return model, classifier


def evaluate_on_plantdoc(
    strategy: str, lambda_val: float, seed: int, device: str
) -> dict:
    seed_suffix = f'_seed{seed}' if seed != 42 else ''
    ckpt = MODELS / f'resnet50_coral_lambda_{lambda_val}_{strategy}{seed_suffix}.pt'
    if not ckpt.exists():
        print(f'  MISSING: {ckpt.name}')
        return None

    test_csv = SPLITS / strategy / 'target_plantdoc_test.csv'
    ds = CSVDataset(test_csv)
    num_classes = ds.df['label_id'].nunique()
    loader = DataLoader(ds, batch_size=64, num_workers=4)

    backbone, classifier = load_coral_checkpoint(ckpt, num_classes)
    backbone = backbone.to(device).eval()
    classifier = classifier.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)
            preds = classifier(feats).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    print(
        f'  {strategy} lambda={lambda_val} seed={seed}: F1={macro_f1:.4f}  Acc={accuracy:.4f}'
    )
    return {'macro_f1': float(macro_f1), 'accuracy': float(accuracy), 'seed': seed}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    configs = [
        ('Fuzzy', 1.0),
        ('hybrid', 0.01),
        ('sbert', 0.01),
    ]
    seeds = [42, 43, 44]

    all_results = {}
    for strategy, lam in configs:
        key = f'{strategy}_lambda{lam}'
        all_results[key] = {}
        print(f'\n--- {strategy} lambda={lam} ---')
        for seed in seeds:
            r = evaluate_on_plantdoc(strategy, lam, seed, device)
            if r:
                all_results[key][seed] = r

    # Summary
    print('\n=== Multi-Seed CORAL Summary ===')
    summary = {}
    for key, seed_results in all_results.items():
        f1_vals = [v['macro_f1'] for v in seed_results.values()]
        if f1_vals:
            mean_f1 = float(np.mean(f1_vals))
            std_f1 = float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0
            n_seeds = len(f1_vals)
            print(f'  {key}: F1={mean_f1:.4f}±{std_f1:.4f} (n={n_seeds} seeds)')
            summary[key] = {
                'macro_f1_mean': mean_f1,
                'macro_f1_std': std_f1,
                'per_seed': seed_results,
                'n_seeds': n_seeds,
            }

    out = RESULTS / 'coral_multiseed_summary.json'
    out.write_text(
        json.dumps({'configs': configs_list(configs), 'results': summary}, indent=2)
    )
    print(f'\nSaved {out}')


def configs_list(configs):
    return [{'strategy': s, 'lambda': lam} for s, lam in configs]


if __name__ == '__main__':
    main()
