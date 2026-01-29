"""Training script for PyTorch models."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import CustomDataset, load_data, preprocess_data
from src.models import SimpleNN
from src.utils import get_device, save_checkpoint, set_seed


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for data, target in tqdm(dataloader, desc="Training"):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    """Main training function."""
    # Set random seed
    set_seed(42)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load and preprocess data
    data_path = Path("data/raw")
    data, labels = load_data(data_path)
    data = preprocess_data(data)

    # Create dataset and dataloader
    dataset = CustomDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    input_size = 10
    hidden_size = 64
    num_classes = 2
    model = SimpleNN(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            save_path = Path("models/best_model.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, train_loss, save_path)
            print(f"Model saved to {save_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
