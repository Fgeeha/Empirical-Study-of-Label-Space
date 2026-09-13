"""AdaBN: Revisiting Batch Normalization For Practical Domain Adaptation.

Li et al., ICLR 2018 (workshop).
No training — only updates BN running statistics from target images.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import TRANSFORM_VAL, UnlabeledDataset, load_backbone_from_pv_checkpoint


def adapt_bn_statistics(
    backbone: nn.Module,
    target_loader: DataLoader,
    device: torch.device,
) -> nn.Module:
    """Update BN running mean/var from target images (no grad updates)."""
    # Set all BN layers to train mode to update running stats
    backbone.eval()
    for m in backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()
            m.momentum = None  # cumulative moving average from scratch

    with torch.no_grad():
        for imgs in target_loader:
            imgs = imgs.to(device)
            backbone(imgs)

    backbone.eval()
    return backbone


class AdaBN:
    """Adapter that updates BN statistics from target unlabeled images."""

    name = 'adabn'

    def __init__(
        self,
        pv_ckpt: Path,
        num_classes: int,
        device: torch.device,
        batch_size: int = 64,
    ):
        self.device = device
        self.backbone, self.classifier = load_backbone_from_pv_checkpoint(
            pv_ckpt, num_classes, device
        )
        self.batch_size = batch_size

    def adapt(self, target_unlabeled_csv: Path) -> None:
        """Forward pass on target images to update BN statistics."""
        ds = UnlabeledDataset(target_unlabeled_csv, TRANSFORM_VAL)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        self.backbone = adapt_bn_statistics(self.backbone, loader, self.device)

    def get_model(self):
        return self.backbone, self.classifier
