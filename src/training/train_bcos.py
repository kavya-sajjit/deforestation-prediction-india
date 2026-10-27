"""
Training and Evaluation Loop for B-cos and Baseline Deforestation CNNs.
"""

import time
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from .metrics import compute_classification_metrics, get_confusion_matrix


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, List[int], List[int], List[float]]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            total += labels.size(0)

            probs = torch.softmax(outputs, dim=-1)
            preds = outputs.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            # Probability of positive class (deforested = 1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    val_loss = total_loss / total if total > 0 else 0.0
    metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
    cm = get_confusion_matrix(all_targets, all_preds)

    return val_loss, metrics, cm, all_targets, all_preds, all_probs


def train_deforestation_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """Complete training procedure with learning rate scheduling and checkpointing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    start_time = time.time()
    print(f"Training on device: {device} | Total Epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.1%} F1: {val_metrics['f1']:.3f}"
        )

        if checkpoint_path and val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, checkpoint_path)

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s | Best Val F1: {best_val_f1:.3f}")

    return history


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """Plots loss and accuracy/F1 learning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Metrics
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="royalblue", lw=2)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="forestgreen", lw=2)
    ax2.plot(epochs, history["val_f1"], label="Val F1-Score", color="darkorange", lw=2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Accuracy & F1 Progression")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
