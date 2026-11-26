"""Model architectures: a baseline MLP and an improved 3-block CNN."""

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Simple Multi-Layer Perceptron baseline for 28x28 grayscale images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FashionCNN(nn.Module):
    """Small 3-block CNN with BatchNorm, ReLU, MaxPool, and Dropout."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Three conv blocks progressively double channels while halving spatial size:
        # 28x28 -> 14x14 -> 7x7 -> 3x3.
        self.features = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.1),
            # Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.15),
            # Block 3: 64x7x7 -> 128x3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """Instantiate a model by its short registry name ('mlp' or 'cnn')."""
    registry: dict[str, type[nn.Module]] = {
        "mlp": BaselineMLP,
        "cnn": FashionCNN,
    }
    key = model_name.lower()
    if key not in registry:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {list(registry)}.")
    return registry[key](num_classes=num_classes)
