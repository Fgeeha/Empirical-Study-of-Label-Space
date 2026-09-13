"""Deep CORAL: Correlation Alignment for Deep Domain Adaptation.

Sun & Saenko, ECCV Workshop 2016.
Fixed protocol: checkpoint selected by source validation F1.
"""

from pathlib import Path

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import (
    load_backbone_from_pv_checkpoint,
    make_loader,
)


def coral_loss(source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """Frobenius norm between source and target covariance matrices."""
    d = source_feat.size(1)
    src = source_feat - source_feat.mean(dim=0)
    tgt = target_feat - target_feat.mean(dim=0)
    cs = (src.T @ src) / (src.size(0) - 1)
    ct = (tgt.T @ tgt) / (tgt.size(0) - 1)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


class CORAL:
    """CORAL with source-validation checkpoint selection."""

    name = 'coral'

    def __init__(
        self,
        pv_ckpt: Path,
        num_classes: int,
        device: torch.device,
        lambda_coral: float = 0.1,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        self.device = device
        self.lambda_coral = lambda_coral
        self.epochs = epochs
        self.batch_size = batch_size
        self.backbone, self.classifier = load_backbone_from_pv_checkpoint(
            pv_ckpt, num_classes, device
        )
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.ce = nn.CrossEntropyLoss()
        self.best_state = None
        self.best_src_f1 = -1.0

    def fit(
        self,
        source_train_csv: Path,
        target_unlabeled_csv: Path,
        source_val_csv: Path,
    ) -> list[float]:
        """Train and select checkpoint by source-val F1."""
        src_loader = make_loader(
            source_train_csv, self.batch_size, labeled=True, shuffle=True
        )
        tgt_loader = make_loader(
            target_unlabeled_csv, self.batch_size, labeled=False, shuffle=True
        )
        val_loader = make_loader(source_val_csv, 64, labeled=True, shuffle=False)

        target_iter = iter(tgt_loader)
        val_f1_history = []

        for epoch in range(self.epochs):
            self.backbone.train()
            self.classifier.train()
            total_loss = 0.0

            for xs, ys in src_loader:
                try:
                    xt = next(target_iter)
                except StopIteration:
                    target_iter = iter(tgt_loader)
                    xt = next(target_iter)

                xs, ys = xs.to(self.device), ys.to(self.device)
                xt = xt.to(self.device)

                fs = self.backbone(xs)
                ft = self.backbone(xt)
                logits = self.classifier(fs)
                loss = self.ce(logits, ys) + self.lambda_coral * coral_loss(fs, ft)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # Source-val evaluation for checkpoint selection
            src_f1 = self._eval_f1(val_loader)
            val_f1_history.append(src_f1)
            print(
                f'  [CORAL] Epoch {epoch:02d}: loss={total_loss / len(src_loader):.4f}  src_val_f1={src_f1:.4f}'
            )

            if src_f1 > self.best_src_f1:
                self.best_src_f1 = src_f1
                self.best_state = {
                    'backbone': {
                        k: v.cpu().clone()
                        for k, v in self.backbone.state_dict().items()
                    },
                    'classifier': {
                        k: v.cpu().clone()
                        for k, v in self.classifier.state_dict().items()
                    },
                }

        self._load_best()
        return val_f1_history

    def _eval_f1(self, loader: DataLoader) -> float:
        self.backbone.eval()
        self.classifier.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                preds.extend(self.classifier(self.backbone(x)).argmax(1).cpu().tolist())
                labels.extend(y.tolist())
        self.backbone.train()
        self.classifier.train()
        return f1_score(labels, preds, average='macro', zero_division=0)

    def _load_best(self):
        if self.best_state:
            self.backbone.load_state_dict(self.best_state['backbone'])
            self.classifier.load_state_dict(self.best_state['classifier'])
        self.backbone.eval()
        self.classifier.eval()

    def get_model(self):
        return self.backbone, self.classifier
