import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SPLITS = Path('splits')
MODELS = Path('models')
FIGURES = Path('figures')

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


def load_model(name, num_classes, ckpt, device):
    """Load model architecture and weights."""
    if name == 'resnet50':
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == 'efficientnet':
        model = models.efficientnet_b3()
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    elif name == 'mobilenet_baseline':
        model = models.mobilenet_v2()
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f'Unknown model: {name}')

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.eval()


def get_predictions(model, loader, device):
    """Run inference and collect predictions."""
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            preds.extend(out.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())

    return targets, preds


def plot_cm(y_true, y_pred, title, out_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        cm,
        cmap='Blues',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Count'},
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'[OK] Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='hybrid', help='Which strategy to plot')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    strategy = args.strategy
    test_csv = SPLITS / strategy / 'pv_test.csv'

    if not test_csv.exists():
        print(f'[ERROR] Missing test CSV: {test_csv}')
        return

    dataset = CSVDataset(test_csv)
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    num_classes = dataset.df['label_id'].nunique()

    # ----------------------------
    # Plot for each model
    # ----------------------------
    configs = [
        ('resnet50', f'resnet50_pv_{strategy}_best.pt', 'ResNet-50'),
        ('efficientnet', f'efficientnet_pv_{strategy}_best.pt', 'EfficientNet-B3'),
        (
            'mobilenet_baseline',
            f'mobilenet_baseline_pv_{strategy}_best.pt',
            'MobileNetV2',
        ),
    ]

    for model_name, ckpt_name, display_name in configs:
        ckpt_path = MODELS / ckpt_name

        if not ckpt_path.exists():
            print(f'[WARN] Skipping {model_name}: checkpoint not found')
            continue

        print(f'[INFO] Processing {model_name}...')
        model = load_model(model_name, num_classes, ckpt_path, device).to(device)

        y_true, y_pred = get_predictions(model, loader, device)

        plot_cm(
            y_true,
            y_pred,
            title=f'{display_name} on PlantVillage ({strategy})',
            out_path=FIGURES / f'cm_baseline_{model_name}_{strategy}.png',
        )

    print('[OK] All confusion matrices plotted')


if __name__ == '__main__':
    main()
