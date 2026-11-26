"""End-to-end training and evaluation pipeline for Fashion-MNIST."""

import argparse
from pathlib import Path

from src.dataset import get_dataloaders
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_training_curves,
    print_comparison_table,
    print_per_class_accuracy,
)
from src.models import build_model
from src.train import train_model
from src.utils import CLASS_NAMES, count_parameters, get_device, save_checkpoint, set_seed

# --- Configurable Hyperparameters ---
SEED = 42
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 0.001
STEP_SIZE = 4
GAMMA = 0.5
VAL_SIZE = 6000
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def parse_args() -> argparse.Namespace:
    """Allow overriding key training hyperparameters from the command line."""
    parser = argparse.ArgumentParser(description="Train and compare MLP vs CNN on Fashion-MNIST.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs per model.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial Adam learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        val_size=VAL_SIZE,
        seed=SEED,
    )
    print(
        f"Dataset splits -> Train: {len(train_loader.dataset):,}, "
        f"Val: {len(val_loader.dataset):,}, Test: {len(test_loader.dataset):,}"
    )

    histories: dict[str, dict[str, list[float]]] = {}
    eval_results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    best_overall_val_acc = -1.0
    best_overall_name = ""
    best_overall_model = None

    for model_name in ("mlp", "cnn"):
        # Re-seed before each model so both start with clean, reproducible initialization.
        set_seed(SEED)
        model = build_model(model_name, num_classes=len(CLASS_NAMES)).to(device)
        params = count_parameters(model)
        print(f"\nInitialized {model_name.upper()} with {params:,} trainable parameters.")

        model_ckpt = OUTPUT_DIR / f"{model_name}_best.pth"
        trained_model, history, best_val_acc = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            step_size=STEP_SIZE,
            gamma=GAMMA,
            checkpoint_path=model_ckpt,
        )

        histories[model_name] = history
        test_metrics = evaluate_model(trained_model, test_loader, device, num_classes=len(CLASS_NAMES))
        eval_results[model_name] = test_metrics
        print_per_class_accuracy(test_metrics["per_class_acc"], CLASS_NAMES, model_name.upper())

        summary_rows.append(
            {
                "name": model_name.upper(),
                "params": params,
                "val_acc": best_val_acc,
                "test_acc": test_metrics["test_acc"],
            }
        )

        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_overall_name = model_name
            best_overall_model = trained_model

    # Save the winning architecture across both models to outputs/best_model.pth.
    best_model_path = OUTPUT_DIR / "best_model.pth"
    assert best_overall_model is not None
    save_checkpoint(best_overall_model, best_overall_name, best_overall_val_acc, best_model_path)
    print(f"\nSaved overall best model ({best_overall_name.upper()}) to: {best_model_path}")

    # Generate all required plots in outputs/.
    plot_training_curves(histories, OUTPUT_DIR / "training_curves.png")
    best_eval = eval_results[best_overall_name]
    plot_confusion_matrix(
        best_eval["confusion_matrix"],
        CLASS_NAMES,
        title=f"Confusion Matrix - {best_overall_name.upper()} (Test Acc: {best_eval['test_acc']:.2f}%)",
        save_path=OUTPUT_DIR / "confusion_matrix.png",
    )
    plot_sample_predictions(
        best_eval["sample_images"],
        best_eval["sample_preds"],
        best_eval["sample_labels"],
        CLASS_NAMES,
        save_path=OUTPUT_DIR / "sample_predictions.png",
    )

    print_comparison_table(summary_rows)
    print(f"\nArtifacts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
