"""
Transfer Learning Training Script with MLflow Experiment Tracking.

This script trains image classification models using transfer learning
on the Oxford-IIIT Pet dataset and logs all experiments to MLflow.
"""

import json
import os
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from mylib.dataset import get_dataloaders


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes, pretrained=True):
    """
    Get a pre-trained model and modify the classifier for transfer learning.

    Args:
        model_name: Name of the model ('resnet18', 'resnet50', 'vgg16', 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        Modified model with frozen feature extractor
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        # Freeze all layers except the final classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier[6].parameters():
            param.requires_grad = True

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier[1].parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@click.command()
@click.option("--model-name", default="resnet18", help="Model architecture")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--epochs", default=10, help="Number of epochs")
@click.option("--lr", default=0.001, help="Learning rate")
@click.option("--seed", default=42, help="Random seed")
@click.option("--data-dir", default="./data", help="Data directory")
@click.option("--experiment-name", default="pet-classification", help="MLflow experiment name")
def train(model_name, batch_size, epochs, lr, seed, data_dir, experiment_name):
    """Train a model with transfer learning and log to MLflow."""
    # Set seed for reproducibility
    set_seed(seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create run name
    run_name = f"{model_name}_bs{batch_size}_lr{lr}_ep{epochs}"

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("seed", seed)
        mlflow.log_param("dataset", "Oxford-IIIT-Pet")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "CrossEntropyLoss")
        mlflow.log_param("device", str(device))
        mlflow.log_param("pretrained", True)
        mlflow.log_param("transfer_learning", True)

        # Get dataloaders
        print("Loading data...")
        train_loader, val_loader, class_names = get_dataloaders(
            root=data_dir, batch_size=batch_size, val_split=0.2, seed=seed, num_workers=4
        )

        num_classes = len(class_names)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("train_samples", len(train_loader.dataset))
        mlflow.log_param("val_samples", len(val_loader.dataset))

        # Save class labels
        class_labels = {i: name for i, name in enumerate(class_names)}
        class_labels_path = "class_labels.json"
        with open(class_labels_path, "w", encoding="utf-8") as f:
            json.dump(class_labels, f, indent=2)
        mlflow.log_artifact(class_labels_path)

        # Get model
        print(f"Creating model: {model_name}")
        model = get_model(model_name, num_classes, pretrained=True)
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # Training loop
        print(f"Training for {epochs} epochs...")
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Log metrics per epoch
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # Log final metrics
        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("final_val_loss", val_losses[-1])
        mlflow.log_metric("final_train_accuracy", train_accs[-1])
        mlflow.log_metric("final_val_accuracy", val_accs[-1])
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # Plot and save training curves
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / f"{run_name}_curves.png"
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
        mlflow.log_artifact(plot_path)

        # Log model
        print("Logging model to MLflow...")
        mlflow.pytorch.log_model(model, "model", registered_model_name="pet-classifier")

        print(f"\n✓ Training complete!")
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")
        print(f"  MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train()
