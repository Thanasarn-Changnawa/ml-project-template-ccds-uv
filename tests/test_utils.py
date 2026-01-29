"""Tests for utility functions."""

import torch

from src.utils import get_device, set_seed


def test_set_seed():
    """Test seed setting."""
    set_seed(42)
    x1 = torch.rand(10)

    set_seed(42)
    x2 = torch.rand(10)

    assert torch.allclose(x1, x2)


def test_get_device():
    """Test device detection."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cpu", "cuda"]
