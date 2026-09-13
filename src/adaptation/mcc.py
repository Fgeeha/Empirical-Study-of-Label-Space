"""MCC: Minimum Class Confusion for Versatile Domain Adaptation.

Jin et al., ECCV 2020.
No discriminator. Minimizes pairwise class confusion on target domain.
"""

from pathlib import Path

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import load_backbone_from_pv_checkpoint, make_loader


def mcc_loss(target_logits: torch.Tensor, temperature: float = 2.5) -> torch.Tensor:
    """Minimum Class Confusion loss.

    For a batch of target predictions (B, K):
    - Compute soft predictions with temperature T
    - Build normalized confusion matrix C (K, K)
    - Loss = sum of off-diagonal elements of C

    Args:
        target_logits: (B, K) raw logits on target batch
        temperature: temperature for softmax scaling
    """
    soft_pred = F.softmax(target_logits / temperature, dim=1)  # (B, K)

    # Confusion matrix: C[i,j] = (1/B) * sum_b p_i(x_b) * p_j(x_b)
    # Matrix form: C = soft_pred.T @ soft_pred / B
    confusion = soft_pred.T @ soft_pred / soft_pred.size(0)  # (K, K)

    # Normalize so diagonal = 1 (avoids degenerate solution of predicting one class)
    diag = confusion.diag().clamp(min=1e-8)
    norm = diag.unsqueeze(0) * diag.unsqueeze(1)
    # C_hat[i,j] = C[i,j] / sqrt(C[i,i] * C[j,j])
    confusion_norm = confusion / norm.sqrt()

    # Loss = sum of off-diagonal elements
    identity = torch.eye(confusion_norm.size(0), device=confusion_norm.device)
    return (confusion_norm * (1 - identity)).sum()


class MCC:
    """MCC with source-validation checkpoint selection."""

    name = 'mcc'

    def __init__(
        self,
        pv_ckpt: Path,
        num_classes: int,
        device: torch.device,
        temperature: float = 2.5,
        mcc_weight: float = 1.0,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        self.device = device
        self.temperature = temperature
        self.mcc_weight = mcc_weight
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
            epoch_cls_loss = 0.0
            epoch_mcc_loss = 0.0

            for xs, ys in src_loader:
                try:
                    xt = next(target_iter)
                except StopIteration:
                    target_iter = iter(tgt_loader)
                    xt = next(target_iter)

                xs, ys = xs.to(self.device), ys.to(self.device)
                xt = xt.to(self.device)

                # Source classification loss
                fs = self.backbone(xs)
                logits_s = self.classifier(fs)
                loss_cls = self.ce(logits_s, ys)

                # Target MCC loss (no labels used)
                ft = self.backbone(xt)
                logits_t = self.classifier(ft)
                loss_mcc = mcc_loss(logits_t, self.temperature)

                loss = loss_cls + self.mcc_weight * loss_mcc
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_cls_loss += loss_cls.item()
                epoch_mcc_loss += loss_mcc.item()

            src_f1 = self._eval_f1(val_loader)
            val_f1_history.append(src_f1)
            print(
                f'  [MCC] Epoch {epoch:02d}: cls={epoch_cls_loss / len(src_loader):.4f}'
                f'  mcc={epoch_mcc_loss / len(src_loader):.4f}'
                f'  src_val_f1={src_f1:.4f}'
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
