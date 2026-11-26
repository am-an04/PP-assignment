"""Training and validation loops with Adam, StepLR, and tqdm progress bars."""

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import save_checkpoint


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> tuple[float, float]:
    """Run one pass over a DataLoader (training if optimizer is given, else eval)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=desc, leave=False, mininterval=1.0) if desc else loader
    with torch.set_grad_enabled(is_train):
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += batch_size

            if desc:
                progress.set_postfix(
                    loss=f"{total_loss / total_samples:.4f}",
                    acc=f"{100.0 * correct / total_samples:.2f}%",
                )

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * correct / total_samples
    return avg_loss, avg_acc


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 12,
    lr: float = 0.001,
    step_size: int = 4,
    gamma: float = 0.5,
    checkpoint_path: str | Path = "outputs/best_model.pth",
) -> tuple[nn.Module, dict[str, list[float]], float]:
    """Train a model and keep the weights with the highest validation accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"\n=== Training {model_name.upper()} for {epochs} epochs ===")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            desc=f"{model_name.upper()} Epoch {epoch}/{epochs} [Train]",
        )
        val_loss, val_acc = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            desc=f"{model_name.upper()} Epoch {epoch}/{epochs} [Val]  ",
        )

        # Step the LR scheduler at epoch end so each epoch sees a consistent learning rate.
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            save_checkpoint(model, model_name, best_val_acc, checkpoint_path)
            marker = " * (saved best)"

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:6.2f}% | "
            f"Next LR: {current_lr:.5f}{marker}"
        )

    # Restore best validation weights before test evaluation so late-epoch overfitting
    # cannot degrade final test accuracy.
    model.load_state_dict(best_weights)
    return model, history, best_val_acc
