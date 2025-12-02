"""
Dataset utilities for Oxford-IIIT Pet classification.

This module provides dataset classes and utilities for loading and
preprocessing the Oxford-IIIT Pet dataset.
"""

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image


class PetDataset(Dataset):
    """
    Wrapper for Oxford-IIIT Pet Dataset with custom transforms.

    Args:
        root: Root directory of dataset
        split: 'trainval' or 'test'
        transform: Optional transform to be applied on images
        target_transform: Optional transform to be applied on labels
    """

    def __init__(self, root="./data", split="trainval", transform=None, target_transform=None):
        self.dataset = OxfordIIITPet(
            root=root, split=split, download=False, target_types="category"
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    @property
    def classes(self):
        """Return class names."""
        return self.dataset.classes


def get_transforms(train=True):
    """
    Get image transforms for training or validation.

    Args:
        train: If True, returns training transforms with augmentation.
               If False, returns validation transforms without augmentation.

    Returns:
        torchvision.transforms.Compose object
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def create_train_val_split(dataset, val_split=0.2, seed=42):
    """
    Split dataset into training and validation sets.

    Args:
        dataset: PyTorch Dataset object
        val_split: Fraction of data to use for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    return train_dataset, val_dataset


def get_dataloaders(
    root="./data",
    batch_size=32,
    val_split=0.2,
    seed=42,
    num_workers=4,
):
    """
    Create train and validation dataloaders.

    Args:
        root: Root directory of dataset
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Create datasets with transforms
    full_dataset = PetDataset(root=root, split="trainval", transform=None)

    # Split into train and validation
    train_dataset, val_dataset = create_train_val_split(full_dataset, val_split, seed)

    # Apply transforms
    train_dataset.dataset.transform = get_transforms(train=True)
    val_dataset.dataset.transform = get_transforms(train=False)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset.classes
