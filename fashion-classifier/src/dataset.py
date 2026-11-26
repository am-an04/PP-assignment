"""Dataset loading, train/val splitting, and transforms for Fashion-MNIST."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Official Fashion-MNIST channel mean and standard deviation (computed over 60k train images).
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform) for Fashion-MNIST.

    Augmentations are applied only during training so validation and test metrics
    reflect unperturbed images.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ]
    )
    return train_transform, eval_transform


def get_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    val_size: int = 6000,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Download Fashion-MNIST and build train (54k), val (6k), and test (10k) DataLoaders."""
    data_path = Path(data_dir)
    train_transform, eval_transform = get_transforms()

    # Load the training split twice with different transforms so validation images
    # never receive random flips or rotations.
    train_full = datasets.FashionMNIST(
        root=data_path, train=True, download=True, transform=train_transform
    )
    val_full = datasets.FashionMNIST(
        root=data_path, train=True, download=True, transform=eval_transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_path, train=False, download=True, transform=eval_transform
    )

    total_train = len(train_full)
    train_size = total_train - val_size
    if train_size <= 0:
        raise ValueError(f"val_size ({val_size}) must be smaller than {total_train}.")

    # Use an isolated generator so the 54k/6k split is identical on every run.
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_train, generator=generator).tolist()
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
