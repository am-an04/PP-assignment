"""Optional Gradio web app for interactive clothing image classification."""

from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps
from torchvision import datasets

from predict import predict_pil_image
from src.utils import get_device, load_checkpoint

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "outputs" / "best_model.pth"
EXAMPLES_DIR = BASE_DIR / "outputs" / "examples"

DEVICE = get_device()
MODEL, CHECKPOINT = load_checkpoint(CHECKPOINT_PATH, DEVICE)
CLASS_NAMES: list[str] = CHECKPOINT["class_names"]

BACKGROUND_MODES = {
    "Auto (Recommended)": None,
    "Force Invert (Light background)": True,
    "No Invert (Dark background)": False,
}


def _ensure_example_images() -> list[list]:
    """Save a few sample test-set images (white-bg & native) so users can test in 1 click."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # Indices chosen from official test set: Ankle boot (0), Pullover (1), Trouser (2), Dress (13), Bag (18)
    sample_indices = [0, 1, 2, 13, 18]
    example_rows: list[list] = []
    test_ds = datasets.FashionMNIST(BASE_DIR / "data", train=False, download=False)
    for idx in sample_indices:
        img, label = test_ds[idx]
        slug = CLASS_NAMES[label].lower().replace("/", "_").replace(" ", "_")
        out_path = EXAMPLES_DIR / f"{slug}_{idx}.png"
        if not out_path.exists():
            # Save upscaled white-background silhouette like a real product cutout.
            ImageOps.invert(img).resize((168, 168), Image.Resampling.BILINEAR).save(out_path)
        example_rows.append([str(out_path), "Auto (Recommended)", True])
    return example_rows


def classify_image(
    image: Image.Image | None,
    background_mode: str = "Auto (Recommended)",
    auto_crop: bool = True,
) -> tuple[dict[str, float], Image.Image | None]:
    """Predict top-3 Fashion-MNIST classes and show the preprocessed 28x28 model input."""
    if image is None:
        return {}, None

    invert = BACKGROUND_MODES.get(background_mode, None)
    preds, processed_pil = predict_pil_image(
        image=image,
        model=MODEL,
        class_names=CLASS_NAMES,
        device=DEVICE,
        invert=invert,
        auto_crop=auto_crop,
        top_k=3,
    )
    preview = processed_pil.resize((224, 224), Image.Resampling.NEAREST)
    return {label: prob / 100.0 for label, prob in preds}, preview


demo = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="pil", label="Upload Clothing Image"),
        gr.Radio(
            choices=list(BACKGROUND_MODES.keys()),
            value="Auto (Recommended)",
            label="Background Mode",
        ),
        gr.Checkbox(
            value=True,
            label="Auto-crop & center garment (preserves aspect ratio)",
        ),
    ],
    outputs=[
        gr.Label(num_top_classes=3, label="Top-3 Predictions"),
        gr.Image(type="pil", label="What the Model Sees (28x28 Grayscale Input)"),
    ],
    examples=_ensure_example_images(),
    title="Fashion-MNIST Clothing Classifier",
    description=(
        f"Loaded model: {CHECKPOINT['model_name'].upper()} "
        f"(Val Acc: {CHECKPOINT['val_acc']:.2f}%). "
        "Click an example below or upload a plain-background clothing image."
    ),
)

if __name__ == "__main__":
    demo.launch()
