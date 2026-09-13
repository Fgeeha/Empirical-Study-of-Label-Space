"""CDAN: Conditional Adversarial Domain Adaptation.

Long et al., NeurIPS 2018.
Discriminator operates on multilinear map of features x predictions.
Entropy conditioning weights samples by prediction confidence.

Used in agriculture DA: Jeon et al. (2025) applied CDAN+E to plant disease.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import load_backbone_from_pv_checkpoint, make_loader
from .dann import grad_reverse


class ConditionalDomainDiscriminator(nn.Module):
    """Domain discriminator on outer product of features and predictions.

    Input dimension: feature_dim (d) — we apply random linear projection
    from d*K → d before the discriminator to keep memory manageable.
    """

    def __init__(
        self, feature_dim: int = 2048, num_classes: int = 18, hidden_dim: int = 1024
    ):
        super().__init__()
        # Random multilinear map: project d*K → d using fixed random matrix
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        projected_dim = feature_dim  # project back to d

        # Fixed random projection matrix (not trained)
        self.register_buffer(
            'projection', torch.randn(feature_dim * num_classes, projected_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(projected_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self, features: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        features: (B, d)
        predictions: (B, K) — softmax probabilities
        returns: (B, 2) domain logits
        """
        # Outer product: (B, d, K) → flatten to (B, d*K)
        outer = torch.bmm(
            features.unsqueeze(2),  # (B, d, 1)
            predictions.unsqueeze(1),  # (B, 1, K)
        ).view(features.size(0), -1)  # (B, d*K)

        # Random projection: (B, d*K) @ (d*K, d) → (B, d)
        projected = outer @ self.projection
        return self.net(projected)


def entropy_weight(predictions: torch.Tensor) -> torch.Tensor:
    """Entropy conditioning weight: exp(-H(p)) where H is entropy."""
    ent = -(predictions * (predictions + 1e-8).log()).sum(dim=1)
    return torch.exp(-ent)


class CDAN:
    """CDAN+E with source-validation checkpoint selection."""

    name = 'cdan'

    def __init__(
        self,
        pv_ckpt: Path,
        num_classes: int,
        device: torch.device,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        entropy_conditioning: bool = True,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_conditioning = entropy_conditioning

        self.backbone, self.classifier = load_backbone_from_pv_checkpoint(
            pv_ckpt, num_classes, device
        )
        self.discriminator = ConditionalDomainDiscriminator(
            feature_dim=2048, num_classes=num_classes
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters())
            + list(self.classifier.parameters())
            + [p for p in self.discriminator.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.ce = nn.CrossEntropyLoss(reduction='none')
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

                # Forward pass
                fs = self.backbone(xs)
                ft = self.backbone(xt)

                logits_s = self.classifier(fs)
                logits_t = self.classifier(ft)

                # Classification loss (source only)
                loss_cls = self.ce(logits_s, ys).mean()

                # Softmax predictions for conditional discriminator
                pred_s = F.softmax(logits_s.detach(), dim=1)
                pred_t = F.softmax(logits_t.detach(), dim=1)

                # GRL features
                fs_rev = grad_reverse(fs, alpha)
                ft_rev = grad_reverse(ft, alpha)

                # Conditional domain discrimination
                dom_logits_s = self.discriminator(fs_rev, pred_s)
                dom_logits_t = self.discriminator(ft_rev, pred_t)

                dom_labels_s = torch.zeros(
                    fs.size(0), dtype=torch.long, device=self.device
                )
                dom_labels_t = torch.ones(
                    ft.size(0), dtype=torch.long, device=self.device
                )

                if self.entropy_conditioning:
                    w_s = entropy_weight(pred_s)
                    w_t = entropy_weight(pred_t)
                    loss_dom_s = (self.ce(dom_logits_s, dom_labels_s) * w_s).mean()
                    loss_dom_t = (self.ce(dom_logits_t, dom_labels_t) * w_t).mean()
                    loss_dom = loss_dom_s + loss_dom_t
                else:
                    loss_dom = (
                        self.ce(dom_logits_s, dom_labels_s).mean()
                        + self.ce(dom_logits_t, dom_labels_t).mean()
                    )

                loss = loss_cls + loss_dom
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_cls_loss += loss_cls.item()
                epoch_dom_loss += loss_dom.item()

            src_f1 = self._eval_f1(val_loader)
            val_f1_history.append(src_f1)
            print(
                f'  [CDAN] Epoch {epoch:02d}: cls={epoch_cls_loss / len(src_loader):.4f}'
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
