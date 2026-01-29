"""Tests for models."""

import torch

from src.models import ConvNet, SimpleNN


def test_simple_nn():
    """Test SimpleNN model."""
    model = SimpleNN(input_size=10, hidden_size=64, num_classes=2)
    x = torch.randn(32, 10)
    output = model(x)
    assert output.shape == (32, 2)


def test_conv_net():
    """Test ConvNet model."""
    model = ConvNet(num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    output = model(x)
    assert output.shape == (8, 10)


def test_model_training():
    """Test model can be trained."""
    model = SimpleNN(input_size=10, hidden_size=64, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))

    # Forward pass
    output = model(x)
    loss = criterion(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
