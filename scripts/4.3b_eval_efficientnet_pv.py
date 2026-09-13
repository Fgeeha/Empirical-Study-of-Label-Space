import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SPLITS = Path('splits')
MODELS = Path('models')
RESULTS = Path('results')
FIGURES = Path('figures')

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


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
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        return self.tf(img), int(row['label_id'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()
    strategy = args.strategy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    test_ds = CSVDataset(SPLITS / strategy / 'pv_test.csv')
    loader = DataLoader(
        test_ds,
        batch_size=32,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
    )

    num_classes = test_ds.df['label_id'].nunique()

    # Load EfficientNet-B3
    model = models.efficientnet_b3()
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(
        torch.load(
            MODELS / f'efficientnet_pv_{strategy}_best.pt',
            map_location=device,
        )
    )
    model.to(device).eval()

    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            preds.extend(out.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())

    # ----------------------------
    # Metrics
    # ----------------------------
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'precision_macro': precision_score(
            targets, preds, average='macro', zero_division=0
        ),
        'recall_macro': recall_score(targets, preds, average='macro', zero_division=0),
        'f1_macro': f1_score(targets, preds, average='macro', zero_division=0),
        'precision_micro': precision_score(
            targets, preds, average='micro', zero_division=0
        ),
        'recall_micro': recall_score(targets, preds, average='micro', zero_division=0),
        'f1_micro': f1_score(targets, preds, average='micro', zero_division=0),
    }

    out_json = RESULTS / f'efficientnet_pv_metrics_{strategy}.json'
    out_json.write_text(
        json.dumps(metrics, indent=2),
        encoding='utf-8',
    )

    # ----------------------------
    # Classification report
    # ----------------------------
    report = classification_report(
        targets,
        preds,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULTS / f'classification_report_efficientnet_pv_{strategy}.csv'
    )

    # ----------------------------
    # Confusion matrix
    # ----------------------------
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title(f'EfficientNet-B3 PV ({strategy})')
    plt.tight_layout()
    plt.savefig(FIGURES / f'cm_efficientnet_pv_{strategy}.png')
    plt.close()

    print(f"[OK] EfficientNet evaluation completed for strategy='{strategy}'")


if __name__ == '__main__':
    main()
