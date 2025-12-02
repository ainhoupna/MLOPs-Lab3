"""
Download and prepare the Oxford-IIIT Pet Dataset.

This script downloads the Oxford-IIIT Pet Dataset and organizes it
for training and validation.
"""

import os
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torchvision.datasets import OxfordIIITPet


def download_progress(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    print(f"\rDownloading: {percent:.1f}%", end="")


def download_oxford_pet_dataset(data_dir="./data"):
    """
    Download the Oxford-IIIT Pet Dataset using torchvision.

    Args:
        data_dir: Directory to store the dataset

    Returns:
        Path to the dataset directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("Downloading Oxford-IIIT Pet Dataset...")
    print("This may take a few minutes...")

    try:
        # Download training set
        print("\nDownloading training set...")
        train_dataset = OxfordIIITPet(
            root=data_dir, split="trainval", download=True, target_types="category"
        )

        # Download test set
        print("\nDownloading test set...")
        test_dataset = OxfordIIITPet(
            root=data_dir, split="test", download=True, target_types="category"
        )

        print(f"\n✓ Dataset downloaded successfully to {data_path.absolute()}")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")
        print(f"  - Number of classes: {len(train_dataset.classes)}")

        # Print class names
        print("\nClasses:")
        for idx, class_name in enumerate(train_dataset.classes):
            print(f"  {idx}: {class_name}")

        return data_path

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        raise


if __name__ == "__main__":
    dataset_path = download_oxford_pet_dataset()
    print(f"\nDataset ready at: {dataset_path.absolute()}")
