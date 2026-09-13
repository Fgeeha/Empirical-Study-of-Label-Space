import argparse
import csv
from pathlib import Path

import pandas as pd
from PIL import Image
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
        return row['path'], self.tf(img), int(row['label_id'])


def load_model(name, num_classes, ckpt):
    if name == 'resnet50':
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == 'efficientnet':
        model = models.efficientnet_b3()
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    elif name in ('mobilenet', 'mobilenet_baseline'):
        model = models.mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    else:
        raise ValueError(name)

    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument(
        '--model',
        choices=['resnet50', 'efficientnet', 'mobilenet', 'mobilenet_baseline'],
        required=True,
    )
    args = parser.parse_args()

    strategy = args.strategy
    model_name = args.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device}')

    test_csv = SPLITS / strategy / 'target_plantdoc_test.csv'
    ds = CSVDataset(test_csv)
    loader = DataLoader(ds, batch_size=32)

    num_classes = ds.df['label_id'].nunique()
    ckpt = MODELS / f'{model_name}_pv_{strategy}_best.pt'

    model = load_model(model_name, num_classes, ckpt).to(device)

    out_csv = RESULTS / f'preds_target_{model_name}_{strategy}.csv'

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'y_true', 'y_pred', 'prob_top1'])

        with torch.no_grad():
            for paths, x, y in loader:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1)
                top1 = probs.max(dim=1)

                for p, yt, yp, pr in zip(
                    paths,
                    y.tolist(),
                    top1.indices.tolist(),
                    top1.values.tolist(),
                ):
                    writer.writerow([p, yt, yp, pr])

    print(f'[OK] Predictions saved: {out_csv}')


if __name__ == '__main__':
    main()
