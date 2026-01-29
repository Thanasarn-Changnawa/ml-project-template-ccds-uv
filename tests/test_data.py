"""Tests for data utilities."""

import numpy as np
import torch

from src.data import CustomDataset, preprocess_data


def test_custom_dataset(sample_data):
    """Test CustomDataset."""
    data, labels = sample_data
    dataset = CustomDataset(data, labels)

    assert len(dataset) == len(data)

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_preprocess_data(sample_data):
    """Test data preprocessing."""
    data, _ = sample_data
    processed = preprocess_data(data)

    assert processed.shape == data.shape
    assert np.abs(processed.mean()) < 0.1  # Should be close to 0
    assert np.abs(processed.std() - 1.0) < 0.1  # Should be close to 1
