"""Deterministic stratified data splits for CIFAR-10 and EuroSAT RGB."""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from config import ExperimentConfig

# RGB ImageNet statistics are conventional normalization constants;
# using identical preprocessing for all model variants ensures fairness.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TransformSubset(Dataset):
    """Apply a distinct transform to a fixed subset of the original dataset."""

    def __init__(self, dataset: Dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(map(int, indices))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), int(label)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    class_names: list[str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _transforms(image_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, eval_transform


def create_dataloaders(config: ExperimentConfig) -> DataBundle:
    """CIFAR-10: official 50k/10k train/test; stratified val from train.

    EuroSAT RGB: stratified 80/10/10 split by default. No augmentations
    are applied to validation or test data. Indices depend only on seed.
    """
    seed_everything(config.seed)
    data_dir = str(Path(config.data_dir))
    train_transform, eval_transform = _transforms(config.image_size)

    if config.dataset == "cifar10":
        base_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
        base_test = datasets.CIFAR10(root=data_dir, train=False, download=True)
        labels = np.asarray(base_train.targets)
        all_indices = np.arange(len(base_train))
        train_idx, val_idx = train_test_split(
            all_indices, test_size=config.val_fraction,
            random_state=config.seed, stratify=labels,
        )
        train_ds = TransformSubset(base_train, train_idx, train_transform)
        val_ds = TransformSubset(base_train, val_idx, eval_transform)
        test_ds = TransformSubset(base_test, np.arange(len(base_test)), eval_transform)
        class_names = list(base_train.classes)
    else:
        # torchvision EuroSAT downloads the RGB version (not 13-band Sentinel-2).
        base = datasets.EuroSAT(root=data_dir, download=True)
        labels = np.asarray(base.targets)
        all_indices = np.arange(len(base))
        train_val_idx, test_idx = train_test_split(
            all_indices, test_size=config.test_fraction,
            random_state=config.seed, stratify=labels,
        )
        relative_val_fraction = config.val_fraction / (1 - config.test_fraction)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=relative_val_fraction,
            random_state=config.seed, stratify=labels[train_val_idx],
        )
        train_ds = TransformSubset(base, train_idx, train_transform)
        val_ds = TransformSubset(base, val_idx, eval_transform)
        test_ds = TransformSubset(base, test_idx, eval_transform)
        class_names = list(base.classes)

    generator = torch.Generator().manual_seed(config.seed)
    kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return DataBundle(train_loader, val_loader, test_loader, len(class_names), class_names)
