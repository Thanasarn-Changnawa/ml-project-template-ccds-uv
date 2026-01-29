"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom PyTorch dataset."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """Initialize dataset.

        Args:
            data: Input data
            labels: Target labels
            transform: Optional transform to be applied
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index.

        Args:
            idx: Index

        Returns:
            Tuple of data and label
        """
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def load_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from file.

    Args:
        data_path: Path to data file

    Returns:
        Tuple of data and labels
    """
    # Placeholder implementation
    # Replace with actual data loading logic
    data = np.random.rand(100, 10)
    labels = np.random.randint(0, 2, 100)
    return data, labels


def preprocess_data(data: np.ndarray) -> np.ndarray:
    """Preprocess data.

    Args:
        data: Input data

    Returns:
        Preprocessed data
    """
    # Example preprocessing
    # Normalize data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / (std + 1e-8)
    return data
