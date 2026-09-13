"""E5b Phase 3: Extract aligned PlantWild v2 classes and evaluate baseline/CORAL models."""

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PLANTWILD_ZIP = Path(
    '/home/nkolesnikov/.cache/huggingface/hub/datasets--uqtwei2--PlantWild'
    '/snapshots/527a72eb8f00c95e41698bb09d982c7d4625dce3/plantwild_v2.zip'
)
MODELS_DIR = Path('models')
OUT_DIR = Path('experiments/E5b_plantwild')
EXTRACT_DIR = OUT_DIR / 'images'

# PlantWild class -> (hybrid label_id, PV original)
# Only include classes that map to the 18 hybrid model classes
PW_TO_HYBRID = {
    'apple scab': (0, 'Apple___Apple_scab'),
    'apple rust': (1, 'Apple___Cedar_apple_rust'),
    'bell pepper bacterial spot': (3, 'Pepper,_bell___Bacterial_spot'),
    'corn gray leaf spot': (4, 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'),
    'corn northern leaf blight': (5, 'Corn_(maize)___Northern_Leaf_Blight'),
    'corn rust': (6, 'Corn_(maize)___Common_rust_'),
    'potato early blight': (7, 'Potato___Early_blight'),
    'potato late blight': (8, 'Potato___Late_blight'),
    'squash powdery mildew': (9, 'Squash___Powdery_mildew'),
    'tomato early blight': (10, 'Tomato___Early_blight'),
    'tomato septoria leaf spot': (11, 'Tomato___Septoria_leaf_spot'),
    'tomato bacterial leaf spot': (12, 'Tomato___Bacterial_spot'),
    'tomato late blight': (13, 'Tomato___Late_blight'),
    'tomato mosaic virus': (14, 'Tomato___Tomato_mosaic_virus'),
    'tomato yellow leaf curl virus': (15, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'),
    'tomato leaf mold': (16, 'Tomato___Leaf_Mold'),
    'grape black rot': (17, 'Grape___Black_rot'),
}

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class PathDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r['path']).convert('RGB')
        return TRANSFORM(img), r['label_id']


def extract_images() -> list[dict]:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    z = zipfile.ZipFile(PLANTWILD_ZIP)
    print('Extracting aligned PlantWild v2 images...')

    for name in z.namelist():
        parts = name.split('/')
        if len(parts) < 3 or not parts[2]:
            continue
        pw_class = parts[1]
        if pw_class not in PW_TO_HYBRID:
            continue
        label_id, pv_class = PW_TO_HYBRID[pw_class]
        out_path = EXTRACT_DIR / pw_class / parts[2]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            out_path.write_bytes(z.read(name))
        records.append(
            {'path': str(out_path), 'label_id': label_id, 'pw_class': pw_class}
        )

    print(f'  Extracted {len(records)} images across {len(PW_TO_HYBRID)} classes')
    return records


def load_model(ckpt_path: Path) -> torch.nn.Module:
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 18)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # CORAL checkpoints save backbone/classifier separately
    if isinstance(state, dict) and 'backbone' in state and 'classifier' in state:
        merged = {
            **state['backbone'],
            'fc.weight': state['classifier']['weight'],
            'fc.bias': state['classifier']['bias'],
        }
        model.load_state_dict(merged)
    else:
        model.load_state_dict(state)
    return model.eval()


def run_inference(model: torch.nn.Module, records: list[dict], device: str) -> tuple:
    dataset = PathDataset(records)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
    model = model.to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def evaluate(preds: np.ndarray, labels: np.ndarray) -> dict:
    # Only evaluate on the 17 label_ids that appear in PlantWild
    present_ids = sorted(set(labels))
    macro_f1 = f1_score(
        labels, preds, labels=present_ids, average='macro', zero_division=0
    )
    accuracy = accuracy_score(labels, preds)
    per_class = f1_score(
        labels, preds, labels=present_ids, average=None, zero_division=0
    )
    return {
        'macro_f1': float(macro_f1),
        'accuracy': float(accuracy),
        'n_classes': len(present_ids),
        'n_samples': len(labels),
        'per_class_f1': {
            str(lid): float(f1) for lid, f1 in zip(present_ids, per_class)
        },
    }


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    records = extract_images()

    # Save test CSV for reference
    pd.DataFrame(records).to_csv(OUT_DIR / 'plantwild_test.csv', index=False)

    checkpoints = {
        'baseline': MODELS_DIR / 'resnet50_pv_hybrid_best.pt',
        'coral_0.1': MODELS_DIR / 'resnet50_coral_lambda_0.1_hybrid.pt',
        'coral_0.01': MODELS_DIR / 'resnet50_coral_lambda_0.01_hybrid.pt',
        'dann_seed42': MODELS_DIR / 'dann_hybrid_seed42.pt',
        'dann_seed43': MODELS_DIR / 'dann_hybrid_seed43.pt',
        'dann_seed44': MODELS_DIR / 'dann_hybrid_seed44.pt',
    }

    results = {}
    for name, ckpt in checkpoints.items():
        if not ckpt.exists():
            print(f'  Skipping {name}: {ckpt} not found')
            continue
        print(f'\nEvaluating {name} ({ckpt.name})...')
        model = load_model(ckpt)
        preds, labels = run_inference(model, records, device)
        metrics = evaluate(preds, labels)
        results[name] = metrics
        print(f'  Macro-F1: {metrics["macro_f1"]:.4f}  Acc: {metrics["accuracy"]:.4f}')
        print(f'  N classes: {metrics["n_classes"]}, N samples: {metrics["n_samples"]}')

    # Compute DANN mean±std across seeds
    dann_seeds = [
        results[k]['macro_f1']
        for k in ('dann_seed42', 'dann_seed43', 'dann_seed44')
        if k in results
    ]
    dann_summary = None
    if dann_seeds:
        dann_mean = float(np.mean(dann_seeds))
        dann_std = float(np.std(dann_seeds, ddof=1)) if len(dann_seeds) > 1 else 0.0
        dann_summary = {
            'macro_f1_mean': dann_mean,
            'macro_f1_std': dann_std,
            'seeds': dann_seeds,
        }

    print('\n=== Summary ===')
    for name, m in results.items():
        print(f'  {name:20s}: macro_f1={m["macro_f1"]:.4f}  acc={m["accuracy"]:.4f}')
    if dann_summary:
        print(
            f'\n  DANN mean±std: {dann_summary["macro_f1_mean"]:.4f} ± {dann_summary["macro_f1_std"]:.4f}'
        )

    output = {
        'experiment': 'E5b',
        'dataset': 'PlantWild v2',
        'n_classes': len(PW_TO_HYBRID),
        'n_samples': len(records),
        'model': 'resnet50 hybrid',
        'results': results,
        'dann_summary': dann_summary,
        'class_mapping': {
            k: {'label_id': v[0], 'pv_class': v[1]} for k, v in PW_TO_HYBRID.items()
        },
    }
    (OUT_DIR / 'eval_results.json').write_text(json.dumps(output, indent=2))
    print('\nSaved eval_results.json')


if __name__ == '__main__':
    main()
