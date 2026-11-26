"""Test evaluation, terminal tables, and matplotlib visualization helpers."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import FASHION_MNIST_MEAN, FASHION_MNIST_STD


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    num_samples: int = 16,
) -> dict[str, Any]:
    """Evaluate a trained model on the test set without modifying weights."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    sample_images: list[np.ndarray] = []
    sample_preds: list[int] = []
    sample_labels: list[int] = []

    mean = FASHION_MNIST_MEAN[0]
    std = FASHION_MNIST_STD[0]

    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device, non_blocking=True)
            logits = model(images_dev)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = labels.numpy()

            # Accumulate confusion matrix directly with NumPy (rows=true, cols=pred).
            np.add.at(cm, (targets, preds), 1)

            if len(sample_images) < num_samples:
                needed = num_samples - len(sample_images)
                # Undo normalization so images display with natural grayscale contrast.
                imgs_np = np.clip(images[:needed, 0].numpy() * std + mean, 0.0, 1.0)
                sample_images.extend(list(imgs_np))
                sample_preds.extend(preds[:needed].tolist())
                sample_labels.extend(targets[:needed].tolist())

    total = int(cm.sum())
    correct = int(np.trace(cm))
    test_acc = 100.0 * correct / total

    class_totals = cm.sum(axis=1)
    per_class_acc = [
        100.0 * cm[i, i] / class_totals[i] if class_totals[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    return {
        "test_acc": test_acc,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
        "sample_images": np.array(sample_images),
        "sample_preds": sample_preds,
        "sample_labels": sample_labels,
    }


def plot_training_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: str | Path,
) -> None:
    """Plot train/validation loss and accuracy curves for all compared models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"mlp": "#1f77b4", "cnn": "#d62728"}

    for model_name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        color = colors.get(model_name.lower(), None)
        label = model_name.upper()

        axes[0].plot(
            epochs, history["train_loss"], linestyle="--", color=color, label=f"{label} Train"
        )
        axes[0].plot(
            epochs, history["val_loss"], linestyle="-", marker="o", markersize=4, color=color, label=f"{label} Val"
        )

        axes[1].plot(
            epochs, history["train_acc"], linestyle="--", color=color, label=f"{label} Train"
        )
        axes[1].plot(
            epochs, history["val_acc"], linestyle="-", marker="o", markersize=4, color=color, label=f"{label} Val"
        )

    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str | Path,
) -> None:
    """Render and save an annotated confusion matrix PNG."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            text_color = "white" if cm[row, col] > threshold else "black"
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    fig.tight_layout()
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_predictions(
    images: np.ndarray,
    preds: list[int],
    labels: list[int],
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """Plot a 4x4 grid of 16 test images with green (correct) / red (wrong) titles."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            ax.axis("off")
            continue

        ax.imshow(images[idx], cmap="gray")
        pred_name = class_names[preds[idx]]
        true_name = class_names[labels[idx]]
        is_correct = preds[idx] == labels[idx]
        color = "green" if is_correct else "red"

        ax.set_title(f"Pred: {pred_name}\nTrue: {true_name}", color=color, fontsize=9)
        ax.axis("off")

    fig.suptitle("16 Test Set Predictions (Green = Correct, Red = Wrong)", fontsize=13)
    fig.tight_layout()
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_per_class_accuracy(
    per_class_acc: list[float],
    class_names: list[str],
    model_label: str,
) -> None:
    """Print formatted per-class test accuracy in the terminal."""
    print(f"\n--- Per-Class Test Accuracy ({model_label}) ---")
    for name, acc in zip(class_names, per_class_acc):
        print(f"  {name:<14}: {acc:6.2f}%")


def print_comparison_table(summary_rows: list[dict[str, Any]]) -> None:
    """Print a clean final comparison table between MLP and CNN."""
    print("\n================ Model Comparison ================")
    print(f"{'Model':<14} | {'Params':>10} | {'Best Val':>9} | {'Test Acc':>9}")
    print("-" * 52)
    for row in summary_rows:
        print(
            f"{row['name']:<14} | "
            f"{row['params']:>10,} | "
            f"{row['val_acc']:>8.2f}% | "
            f"{row['test_acc']:>8.2f}%"
        )
    print("=" * 52)
