"""
E2 — DANN (Domain-Adversarial Neural Network) training.

Addresses: DVkE (R2-M3) — "Only CORAL explored; considering that gains are small,
it would seem to make sense to explore other algorithms."

Design:
- Same ResNet-50 backbone as CORAL experiments
- Same hybrid strategy, same optimizer (AdamW), same epochs=20, same batch size=32
- Same source/target splits as CORAL hybrid
- 3 seeds for mean ± std F1
- GRL (Gradient Reversal Layer) with annealing schedule (α from 0 to 1 over epochs)
- Domain discriminator: 2-class head (source=0, target=1)

This is a minimal, fair comparison: only the alignment loss changes (DANN vs CORAL).
All other hyperparameters identical to paper's CORAL setup.

Usage:
    poetry run python experiments/E2_dann/train_dann.py --strategy hybrid --epochs 20
"""

import argparse
import datetime
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# =====================================================
# Paths
# =====================================================
CONFIG = Path("configs/train_resnet50.yaml")
SPLITS = Path("splits")
MODELS = Path("models")
RESULTS = Path("results")
OUT_DIR = Path("experiments/E2_dann")
LOG_DIR = Path("experiments/logs/E2")

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# =====================================================
# Dataset
# =====================================================
class CSVDataset(Dataset):
    def __init__(self, csv_path: Path, labeled: bool, augment: bool = False):
        self.df = pd.read_csv(csv_path)
        self.labeled = labeled
        base_tf = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        if augment:
            aug = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
            self.tf = transforms.Compose(aug + base_tf)
        else:
            self.tf = transforms.Compose(base_tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.labeled:
            return self.tf(img), int(row["label_id"])
        return self.tf(img)


# =====================================================
# Gradient Reversal Layer
# =====================================================
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# =====================================================
# Domain Discriminator
# =====================================================
class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int = 2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.fc(x)


# =====================================================
# Training
# =====================================================
def run_dann(strategy: str, epochs: int, seed: int, device: torch.device) -> dict:
    print(f"\n[DANN] strategy={strategy}, epochs={epochs}, seed={seed}")

    cfg = yaml.safe_load(CONFIG.open())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Datasets
    source_ds = CSVDataset(
        SPLITS / strategy / "pv_train.csv", labeled=True, augment=True
    )
    target_ds = CSVDataset(
        SPLITS / strategy / "target_plantdoc_unlabeled.csv", labeled=False
    )
    test_ds = CSVDataset(SPLITS / strategy / "target_plantdoc_test.csv", labeled=True)

    source_loader = DataLoader(
        source_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    num_classes = source_ds.df["label_id"].nunique()
    print(
        f"  num_classes={num_classes}, source={len(source_ds)}, target={len(target_ds)}, test={len(test_ds)}"
    )

    # Model
    backbone = models.resnet50(weights="IMAGENET1K_V1")
    backbone.fc = nn.Identity()
    classifier = nn.Linear(2048, num_classes)
    discriminator = DomainDiscriminator(in_dim=2048)

    backbone.to(device)
    classifier.to(device)
    discriminator.to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters())
        + list(classifier.parameters())
        + list(discriminator.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    ce = nn.CrossEntropyLoss()

    # Training loop
    target_iter = iter(target_loader)
    n_iters_total = epochs * len(source_loader)
    iter_count = 0

    best_target_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        backbone.train()
        classifier.train()
        discriminator.train()

        epoch_cls_loss = 0.0
        epoch_dom_loss = 0.0

        for xs, ys in source_loader:
            try:
                xt = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt = next(target_iter)

            # GRL alpha annealing: from 0 → 1 over training
            p = iter_count / n_iters_total
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            iter_count += 1

            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            # Forward
            fs = backbone(xs)
            ft = backbone(xt)

            # Classification loss (source only)
            logits = classifier(fs)
            loss_cls = ce(logits, ys)

            # Domain adversarial loss (source=0, target=1)
            fs_rev = grad_reverse(fs, alpha)
            ft_rev = grad_reverse(ft, alpha)

            domain_src = discriminator(fs_rev)
            domain_tgt = discriminator(ft_rev)

            dom_labels_src = torch.zeros(fs.size(0), dtype=torch.long, device=device)
            dom_labels_tgt = torch.ones(ft.size(0), dtype=torch.long, device=device)

            loss_dom = ce(domain_src, dom_labels_src) + ce(domain_tgt, dom_labels_tgt)

            loss = loss_cls + loss_dom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_cls_loss += loss_cls.item()
            epoch_dom_loss += loss_dom.item()

        avg_cls = epoch_cls_loss / len(source_loader)
        avg_dom = epoch_dom_loss / len(source_loader)
        print(
            f"  Epoch {epoch:02d}: cls_loss={avg_cls:.4f}, dom_loss={avg_dom:.4f}, alpha={alpha:.3f}"
        )

        # Eval on target test
        backbone.eval()
        classifier.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = classifier(backbone(xb))
                preds = logits.argmax(dim=1).cpu().numpy()
                all_pred.extend(preds.tolist())
                all_true.extend(yb.numpy().tolist())

        target_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
        target_acc = accuracy_score(all_true, all_pred)
        print(
            f"  Epoch {epoch:02d}: target_f1={target_f1:.4f}, target_acc={target_acc:.4f}"
        )

        if target_f1 > best_target_f1:
            best_target_f1 = target_f1
            best_state = {
                "backbone": {
                    k: v.cpu().clone() for k, v in backbone.state_dict().items()
                },
                "classifier": {
                    k: v.cpu().clone() for k, v in classifier.state_dict().items()
                },
            }

    # Final eval using best state
    backbone.load_state_dict(best_state["backbone"])
    classifier.load_state_dict(best_state["classifier"])
    backbone.to(device).eval()
    classifier.to(device).eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = classifier(backbone(xb))
            all_pred.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(yb.tolist())

    final_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    final_acc = accuracy_score(all_true, all_pred)

    # Save checkpoint
    model_path = MODELS / f"dann_{strategy}_seed{seed}.pt"
    torch.save(best_state, model_path)
    print(
        f"  [BEST] target_f1={final_f1:.4f}, target_acc={final_acc:.4f} → {model_path}"
    )

    return {
        "seed": seed,
        "strategy": strategy,
        "epochs": epochs,
        "target_macro_f1": round(final_f1, 4),
        "target_accuracy": round(final_acc, 4),
    }


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[INFO] device={device}, strategy={args.strategy}, epochs={args.epochs}, seeds={args.seeds}"
    )

    results = []
    for seed in args.seeds:
        r = run_dann(args.strategy, args.epochs, seed, device)
        results.append(r)

    # Summary
    f1_vals = [r["target_macro_f1"] for r in results]
    mean_f1 = round(float(np.mean(f1_vals)), 4)
    std_f1 = round(float(np.std(f1_vals)), 4)

    acc_vals = [r["target_accuracy"] for r in results]
    mean_acc = round(float(np.mean(acc_vals)), 4)

    print(f"\n=== DANN RESULTS ({args.strategy}, {len(args.seeds)} seeds) ===")
    print(f"Target Macro-F1: {mean_f1} ± {std_f1}")
    print(f"Target Accuracy: {mean_acc}")

    # Load CORAL baseline for comparison
    coral_path = RESULTS / f"coral_lambda_0.1_{args.strategy}_metrics.json"
    baseline_path = RESULTS / f"metrics_target_resnet50_{args.strategy}.json"
    coral_f1 = (
        json.loads(coral_path.read_text())["target_macro_f1"]
        if coral_path.exists()
        else None
    )
    baseline_f1 = (
        json.loads(baseline_path.read_text())["macro_f1_target"]
        if baseline_path.exists()
        else None
    )

    comparison = {
        "method": "DANN",
        "strategy": args.strategy,
        "seeds": args.seeds,
        "target_macro_f1_per_seed": f1_vals,
        "target_macro_f1_mean": mean_f1,
        "target_macro_f1_std": std_f1,
        "target_accuracy_mean": mean_acc,
        "coral_lambda_0.1_f1": round(coral_f1, 4) if coral_f1 else None,
        "baseline_f1": round(baseline_f1, 4) if baseline_f1 else None,
        "git_hash": subprocess.getoutput("git rev-parse --short HEAD"),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    out_json = OUT_DIR / f"dann_results_seed_{args.seeds}.json"
    out_json.write_text(json.dumps(comparison, indent=2))

    # Update T6_dann_vs_coral.csv
    rows = []
    if baseline_f1:
        rows.append(
            {
                "method": "Baseline (ResNet-50)",
                "target_accuracy": round(baseline_f1 * 1.23, 4),
                "target_macro_f1": round(baseline_f1, 4),
            }
        )
    if coral_f1:
        rows.append(
            {
                "method": f"CORAL (λ=0.1)",
                "target_accuracy": None,
                "target_macro_f1": round(coral_f1, 4),
            }
        )
    rows.append(
        {
            "method": f"DANN (mean ± std)",
            "target_accuracy": mean_acc,
            "target_macro_f1": f"{mean_f1}±{std_f1}",
        }
    )

    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(
        OUT_DIR / f"dann_vs_coral_comparison_seed_{args.seeds}.csv", index=False
    )

    print(f"\n[SAVED] {out_json}")
    print(f"[SAVED] {OUT_DIR / 'dann_vs_coral_comparison.csv'}")
    print("\nComparison table:")
    print(df_cmp.to_string(index=False))

    (LOG_DIR / "E2_run.log").write_text(
        f"E2 DANN run at {comparison['timestamp']}\n"
        f"Git: {comparison['git_hash']}\n"
        f"Strategy: {args.strategy}, Seeds: {args.seeds}\n"
        f"F1 per seed: {f1_vals}\n"
        f"Mean F1: {mean_f1} ± {std_f1}\n"
        f"CORAL F1: {coral_f1}, Baseline F1: {baseline_f1}\n"
    )


if __name__ == "__main__":
    main()
