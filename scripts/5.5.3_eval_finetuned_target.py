import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SPLITS = Path('splits')
MODELS = Path('models')
RESULTS = Path('results')


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
    parser.add_argument('--target', choices=['plantdoc', 'fcdd'], default='plantdoc')
    args = parser.parse_args()

    strategy = args.strategy
    target = args.target

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_ds = CSVDataset(SPLITS / strategy / f'target_{target}_test.csv')
    loader = DataLoader(test_ds, batch_size=32)

    num_classes = test_ds.df['label_id'].nunique()

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(
            MODELS / f'resnet50_target_{target}_{strategy}_best.pt',
            map_location=device,
        )
    )
    model.to(device).eval()

    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())

    metrics = {
        'accuracy_target_finetuned': accuracy_score(targets, preds),
        'macro_f1_target_finetuned': f1_score(
            targets, preds, average='macro', zero_division=0
        ),
    }

    out = RESULTS / f'metrics_target_finetuned_{target}_{strategy}.json'
    out.write_text(json.dumps(metrics, indent=2))

    print(f'[OK] Fine-tuned evaluation saved: {out}')


if __name__ == '__main__':
    main()
