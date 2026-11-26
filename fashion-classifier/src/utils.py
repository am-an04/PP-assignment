"""Shared utilities: seeding, device selection, and checkpoint I/O."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models import build_model

CLASS_NAMES: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN algorithms ensure identical results across runs on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Select CUDA GPU if available, otherwise fall back to CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")
    return device


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    model_name: str,
    val_acc: float,
    path: str | Path,
) -> None:
    """Save model weights along with architecture metadata to disk."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "val_acc": float(val_acc),
        "class_names": CLASS_NAMES,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, save_path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved checkpoint and reconstruct the matching model architecture."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    model_name = checkpoint["model_name"]
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    model = build_model(model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
