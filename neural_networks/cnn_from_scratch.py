"""
CNN implementation from scratch in pytorch

This is a simple CNN implementation for the MNIST dataset.
It is a simple CNN with 2 convolutional layers and 2 fully connected layers.
It is trained for 5 epochs and the best model is saved.
"""
import os
import random
import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For extra determinism (slower, optional):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (64, 7, 7)
        x = torch.flatten(x, 1)                # -> (64*7*7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def compute_confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    preds: (N,) predicted class indices
    targets: (N,) true class indices
    """
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=preds.device)
    for t, p in zip(targets, preds):
        cm[t.long(), p.long()] += 1
    return cm

def precision_recall_f1_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    """
    Macro and weighted precision/recall/F1 from confusion matrix
    cm shape: (C, C), rows = true, cols = pred
    """
    eps = 1e-12
    tp = cm.diag().to(torch.float64)                  # (C,)
    per_class_support = cm.sum(dim=1).to(torch.float64)   # true counts (row sum)
    pred_pos = cm.sum(dim=0).to(torch.float64)            # predicted counts (col sum)

    per_class_precision = tp / (pred_pos + eps)
    per_class_recall = tp / (per_class_support + eps)
    per_class_f1 = 2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall + eps)

    # Macro (unweighted mean across classes)
    macro_precision = per_class_precision.mean().item()
    macro_recall = per_class_recall.mean().item()
    macro_f1 = per_class_f1.mean().item()

    # Weighted (weighted by support)
    weights = per_class_support / (per_class_support.sum() + eps)
    weighted_precision = (per_class_precision * weights).sum().item()
    weighted_recall = (per_class_recall * weights).sum().item()
    weighted_f1 = (per_class_f1 * weights).sum().item()

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

def per_class_accuracy_from_cm(cm: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    tp = cm.diag().to(torch.float64)
    support = cm.sum(dim=1).to(torch.float64)
    return (tp / (support + eps)).cpu()

# -----------------------------
# Train & Evaluate
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion=nn.CrossEntropyLoss(),
):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    criterion=nn.CrossEntropyLoss(),
) -> Dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)

        loss_sum += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += y.size(0)

        all_preds.append(preds)
        all_targets.append(y)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    cm = compute_confusion_matrix(all_preds, all_targets, num_classes=num_classes)
    metrics = precision_recall_f1_from_cm(cm)
    per_class_acc = per_class_accuracy_from_cm(cm)

    results = {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
    }
    return results | {"confusion_matrix": cm.cpu(), "per_class_accuracy": per_class_acc}

# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),                      # [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
    ])

    root = os.environ.get("MNIST_ROOT", "./data")
    full_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # Split train into train/val
    val_size = 10_000
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Model/opt
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 5
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_stats = evaluate(model, val_loader, device, num_classes=10, criterion=criterion)

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} | "
              f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} "
              f"val_macroF1={val_stats['macro_f1']:.4f}")

        # Track best
        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            best_state = {
                "model": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
            }

    # Load best (if any)
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        print(f"Loaded best model from epoch {best_state['epoch']} with val_acc={best_state['val_acc']:.4f}")

    # Final test evaluation
    test_stats = evaluate(model, test_loader, device, num_classes=10, criterion=criterion)
    print("\n==== Test Metrics ====")
    print(f"loss:               {test_stats['loss']:.4f}")
    print(f"accuracy:           {test_stats['accuracy']:.4f}")
    print(f"macro_precision:    {test_stats['macro_precision']:.4f}")
    print(f"macro_recall:       {test_stats['macro_recall']:.4f}")
    print(f"macro_f1:           {test_stats['macro_f1']:.4f}")
    print(f"weighted_precision: {test_stats['weighted_precision']:.4f}")
    print(f"weighted_recall:    {test_stats['weighted_recall']:.4f}")
    print(f"weighted_f1:        {test_stats['weighted_f1']:.4f}")

    # Confusion matrix + per-class accuracy
    cm = test_stats["confusion_matrix"].numpy()
    per_class_acc = test_stats["per_class_accuracy"].numpy()
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nPer-class accuracy (class: acc):")
    for cls, acc in enumerate(per_class_acc):
        print(f"{cls}: {acc:.4f}")

if __name__ == "__main__":
    main()
