"""DANN: Domain-Adversarial Neural Networks.

Ganin et al., JMLR 2016.
Fixed protocol: checkpoint selected by source validation F1 (not target test F1).
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.autograd import Function
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import load_backbone_from_pv_checkpoint, make_loader


class GradReverse(Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradReverse.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class DANN:
    """DANN with source-validation checkpoint selection (fair protocol)."""

    name = 'dann'

    def __init__(
        self,
        pv_ckpt: Path,
        num_classes: int,
        device: torch.device,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

        self.backbone, self.classifier = load_backbone_from_pv_checkpoint(
            pv_ckpt, num_classes, device
        )
        self.discriminator = DomainDiscriminator().to(device)
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters())
            + list(self.classifier.parameters())
            + list(self.discriminator.parameters()),
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
        n_iters_total = self.epochs * len(src_loader)
        iter_count = 0
        val_f1_history = []

        for epoch in range(self.epochs):
            self.backbone.train()
            self.classifier.train()
            self.discriminator.train()
            epoch_cls_loss = 0.0
            epoch_dom_loss = 0.0

            for xs, ys in src_loader:
                try:
                    xt = next(target_iter)
                except StopIteration:
                    target_iter = iter(tgt_loader)
                    xt = next(target_iter)

                p = iter_count / n_iters_total
                alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
                iter_count += 1

                xs, ys = xs.to(self.device), ys.to(self.device)
                xt = xt.to(self.device)

                fs = self.backbone(xs)
                ft = self.backbone(xt)

                logits = self.classifier(fs)
                loss_cls = self.ce(logits, ys)

                fs_rev = grad_reverse(fs, alpha)
                ft_rev = grad_reverse(ft, alpha)
                dom_src = self.discriminator(fs_rev)
                dom_tgt = self.discriminator(ft_rev)

                dom_labels_s = torch.zeros(
                    fs.size(0), dtype=torch.long, device=self.device
                )
                dom_labels_t = torch.ones(
                    ft.size(0), dtype=torch.long, device=self.device
                )
                loss_dom = self.ce(dom_src, dom_labels_s) + self.ce(
                    dom_tgt, dom_labels_t
                )

                loss = loss_cls + loss_dom
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_cls_loss += loss_cls.item()
                epoch_dom_loss += loss_dom.item()

            # Source-val evaluation for checkpoint selection (fair protocol)
            src_f1 = self._eval_f1(val_loader)
            val_f1_history.append(src_f1)
            print(
                f'  [DANN] Epoch {epoch:02d}: cls={epoch_cls_loss / len(src_loader):.4f}'
                f'  dom={epoch_dom_loss / len(src_loader):.4f}'
                f'  alpha={alpha:.3f}  src_val_f1={src_f1:.4f}'
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
