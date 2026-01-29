"""Test configuration."""

import pytest


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    import numpy as np

    data = np.random.rand(100, 10)
    labels = np.random.randint(0, 2, 100)
    return data, labels
