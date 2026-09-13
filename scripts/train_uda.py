"""Unified UDA training script with source-validation model selection.

Usage:
  python scripts/train_uda.py --method coral --target plantdoc --seed 42
  python scripts/train_uda.py --method dann --target plantdoc --seed 42
  python scripts/train_uda.py --method cdan --target plantdoc --seed 42
  python scripts/train_uda.py --method mcc  --target plantdoc --seed 42
  python scripts/train_uda.py --method adabn --target plantdoc --seed 42

Protocol: checkpoint selected by source-validation macro-F1.
Target test labels are never used during training or model selection.
"""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import yaml

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.adaptation.adabn import AdaBN  # noqa: E402
from src.adaptation.cdan import CDAN  # noqa: E402
from src.adaptation.coral import CORAL  # noqa: E402
from src.adaptation.dann import DANN  # noqa: E402
from src.adaptation.mcc import MCC  # noqa: E402

SPLITS = ROOT / 'splits'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results' / 'uda_benchmark'
CONFIGS = ROOT / 'configs' / 'uda'

RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / 'raw').mkdir(exist_ok=True)
(RESULTS / 'predictions').mkdir(exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(method: str) -> dict:
    cfg_path = CONFIGS / f'{method}.yaml'
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text())
    return {}


def evaluate(
    backbone: torch.nn.Module,
    classifier: torch.nn.Module,
    test_csv: Path,
    device: torch.device,
    out_preds_csv: Path,
) -> dict:
    from torch.utils.data import DataLoader

    from src.adaptation.base import TRANSFORM_VAL, LabeledDataset

    ds = LabeledDataset(test_csv, TRANSFORM_VAL)
    loader = DataLoader(
        ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    backbone.eval()
    classifier.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.extend(classifier(backbone(x)).argmax(1).cpu().tolist())
            labels.extend(y.tolist())

    # Save predictions
    pd.DataFrame({'y_true': labels, 'y_pred': preds}).to_csv(
        str(out_preds_csv), index=False
    )

    return {
        'macro_f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(labels, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)),
        'macro_recall': float(
            f1_score(labels, preds, average='macro', zero_division=0)
        ),
        'n_samples': len(labels),
    }


def main():
    parser = argparse.ArgumentParser(description='Unified UDA training')
    parser.add_argument(
        '--method', required=True, choices=['coral', 'dann', 'cdan', 'mcc', 'adabn']
    )
    parser.add_argument('--target', required=True, choices=['plantdoc', 'plantwild'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--strategy', default='hybrid', choices=['hybrid', 'Fuzzy', 'sbert']
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_coral', type=float, default=0.1)
    parser.add_argument('--mcc_temperature', type=float, default=2.5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f'[INFO] method={args.method}  target={args.target}  seed={args.seed}  device={device}'
    )

    # Git hash for reproducibility
    try:
        git_hash = (
            subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'], cwd=str(ROOT)
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = 'unknown'

    load_config(args.method)  # load for future config-based overrides

    # Paths
    strategy = args.strategy
    pv_ckpt = MODELS / f'resnet50_pv_{strategy}_best.pt'
    source_train_csv = SPLITS / strategy / 'pv_train.csv'
    source_val_csv = SPLITS / strategy / 'pv_val.csv'

    if args.target == 'plantdoc':
        target_unlabeled_csv = SPLITS / strategy / 'target_plantdoc_unlabeled.csv'
        target_test_csv = SPLITS / strategy / 'target_plantdoc_test.csv'
    else:
        raise ValueError(f'Target {args.target} not yet supported in this script')

    # Determine number of classes
    pv_df = pd.read_csv(str(SPLITS / strategy / 'pv_train.csv'))
    num_classes = pv_df['label_id'].nunique()
    print(f'[INFO] strategy={strategy}  num_classes={num_classes}')

    # Instantiate and train method
    if args.method == 'adabn':
        adapter = AdaBN(pv_ckpt, num_classes, device, batch_size=64)
        adapter.adapt(target_unlabeled_csv)
        backbone, classifier = adapter.get_model()
        val_f1_history = []

    elif args.method == 'coral':
        adapter = CORAL(
            pv_ckpt,
            num_classes,
            device,
            lambda_coral=args.lambda_coral,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        val_f1_history = adapter.fit(
            source_train_csv, target_unlabeled_csv, source_val_csv
        )
        backbone, classifier = adapter.get_model()

    elif args.method == 'dann':
        adapter = DANN(
            pv_ckpt,
            num_classes,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        val_f1_history = adapter.fit(
            source_train_csv, target_unlabeled_csv, source_val_csv
        )
        backbone, classifier = adapter.get_model()

    elif args.method == 'cdan':
        adapter = CDAN(
            pv_ckpt,
            num_classes,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            entropy_conditioning=True,
        )
        val_f1_history = adapter.fit(
            source_train_csv, target_unlabeled_csv, source_val_csv
        )
        backbone, classifier = adapter.get_model()

    elif args.method == 'mcc':
        adapter = MCC(
            pv_ckpt,
            num_classes,
            device,
            temperature=args.mcc_temperature,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        val_f1_history = adapter.fit(
            source_train_csv, target_unlabeled_csv, source_val_csv
        )
        backbone, classifier = adapter.get_model()

    # Evaluate on target test
    preds_path = (
        RESULTS
        / 'predictions'
        / f'{args.method}_{strategy}_{args.target}_seed{args.seed}.csv'
    )
    metrics = evaluate(backbone, classifier, target_test_csv, device, preds_path)
    print(
        f'\n[RESULT] {args.method} seed={args.seed}: macro_f1={metrics["macro_f1"]:.4f}  acc={metrics["accuracy"]:.4f}'
    )

    # Save metrics
    result = {
        'method': args.method,
        'target': args.target,
        'strategy': strategy,
        'seed': args.seed,
        'git_hash': git_hash,
        'args': vars(args),
        'model_selection': 'source_val_macro_f1',
        'best_src_val_f1': getattr(adapter, 'best_src_f1', None),
        'val_f1_history': val_f1_history,
        'target_metrics': metrics,
    }

    out_json = (
        RESULTS / 'raw' / f'{args.method}_{strategy}_{args.target}_seed{args.seed}.json'
    )
    out_json.write_text(json.dumps(result, indent=2))
    print(f'[OK] Saved → {out_json}')

    # Save checkpoint for downstream evaluation (e.g. PlantWild v2)
    ckpt_dir = ROOT / 'models' / 'uda_fair'
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f'{args.method}_{strategy}_seed{args.seed}.pt'
    if hasattr(adapter, 'best_state') and adapter.best_state:
        torch.save(adapter.best_state, str(ckpt_path))
        print(f'[OK] Checkpoint → {ckpt_path}')
    elif args.method == 'adabn':
        # AdaBN: save adapted backbone state
        torch.save(
            {'backbone': backbone.state_dict(), 'classifier': classifier.state_dict()},
            str(ckpt_path),
        )
        print(f'[OK] Checkpoint → {ckpt_path}')


if __name__ == '__main__':
    main()
