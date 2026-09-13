"""Base dataset and model utilities shared across all UDA methods."""

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

TRANSFORM_VAL = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class LabeledDataset(Dataset):
    def __init__(self, csv_path: Path, transform=TRANSFORM_TRAIN):
        self.df = pd.read_csv(str(csv_path))
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        return self.transform(img), int(row['label_id'])


class UnlabeledDataset(Dataset):
    def __init__(self, csv_path: Path, transform=TRANSFORM_TRAIN):
        self.df = pd.read_csv(str(csv_path))
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        return self.transform(img)


def build_backbone(num_classes: int, device: torch.device):
    """ResNet-50 split into backbone (2048-dim) and classifier head."""
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    classifier = nn.Linear(2048, num_classes)
    return backbone.to(device), classifier.to(device)


def load_backbone_from_pv_checkpoint(
    ckpt_path: Path,
    num_classes: int,
    device: torch.device,
):
    """Load a PlantVillage-trained checkpoint into backbone+classifier."""
    backbone, classifier = build_backbone(num_classes, device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    backbone.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith('fc.')}
    )
    classifier.load_state_dict({'weight': state['fc.weight'], 'bias': state['fc.bias']})
    return backbone, classifier


def make_loader(
    csv_path: Path, batch_size: int, labeled: bool, shuffle: bool, num_workers: int = 4
) -> DataLoader:
    transform = TRANSFORM_TRAIN if shuffle else TRANSFORM_VAL
    if labeled:
        ds = LabeledDataset(csv_path, transform)
    else:
        ds = UnlabeledDataset(csv_path, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
        pin_memory=True,
    )
