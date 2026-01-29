"""Test configuration."""

import numpy as np
import pytest


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    data = np.random.rand(100, 10)
    labels = np.random.randint(0, 2, 100)
    return data, labels
