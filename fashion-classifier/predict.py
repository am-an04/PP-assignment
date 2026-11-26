"""Predict the clothing category of a single image using the saved best model."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from src.dataset import FASHION_MNIST_MEAN, FASHION_MNIST_STD
from src.utils import get_device, load_checkpoint

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "outputs" / "best_model.pth"

TENSOR_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ]
)


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Convert any PIL image (including transparent PNGs) into a clean RGB image."""
    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    arr_rgba = np.asarray(rgba, dtype=np.float32)
    alpha = arr_rgba[:, :, 3] / 255.0
    rgb_raw = arr_rgba[:, :, :3]

    # Composite onto black if the garment itself is bright, else onto white so
    # transparent cutouts stay visually separated from the background.
    opaque = alpha > 0.5
    bg_val = 0.0 if (opaque.any() and rgb_raw[opaque].mean() >= 200.0) else 255.0
    composited = rgb_raw * alpha[:, :, None] + bg_val * (1.0 - alpha[:, :, None])
    return Image.fromarray(np.clip(composited, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _extract_foreground_mask(work_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use border-connected RGB flood-fill to isolate the garment from plain/gradient backgrounds."""
    wh, ww, _ = work_rgb.shape
    border_rgb = np.concatenate(
        [work_rgb[0, :, :], work_rgb[-1, :, :], work_rgb[:, 0, :], work_rgb[:, -1, :]],
        axis=0,
    )
    bg_rgb = np.median(border_rgb, axis=0)
    dist = np.linalg.norm(work_rgb - bg_rgb[None, None, :], axis=2)
    border_dist = np.concatenate([dist[0, :], dist[-1, :], dist[:, 0], dist[:, -1]])
    bg_thresh = max(22.0, float(np.percentile(border_dist, 95)) * 1.6)

    visited = np.zeros((wh, ww), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(ww):
        for y in (0, wh - 1):
            if dist[y, x] <= bg_thresh and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for y in range(wh):
        for x in (0, ww - 1):
            if dist[y, x] <= bg_thresh and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < wh and 0 <= nx < ww and not visited[ny, nx] and dist[ny, nx] <= bg_thresh:
                visited[ny, nx] = True
                queue.append((ny, nx))

    return ~visited, dist


def preprocess_pil_image(
    image: Image.Image,
    invert: bool | None = None,
    auto_crop: bool = True,
) -> tuple[torch.Tensor, Image.Image]:
    """Preprocess a PIL image into a 1x1x28x28 normalized tensor and 28x28 preview image.

    Args:
        image: Input PIL image of any size or color mode.
        invert: True to force inversion, False to keep polarity, or None to auto-detect
            light backgrounds and convert them to Fashion-MNIST's light-on-black style.
        auto_crop: If True, crop tightly around the garment and center it inside 28x28
            while preserving aspect ratio.
    """
    rgb_pil = _flatten_to_rgb(image)
    width, height = rgb_pil.size
    arr_gray = np.asarray(rgb_pil.convert("L"), dtype=np.float32)

    pad = max(1, min(height, width) // 20)
    border_g = np.concatenate(
        [
            arr_gray[:pad, :].ravel(),
            arr_gray[-pad:, :].ravel(),
            arr_gray[:, :pad].ravel(),
            arr_gray[:, -pad:].ravel(),
        ]
    )
    border_med = float(np.median(border_g))

    # Fast path: preserve native 28x28 Fashion-MNIST inputs without re-cropping.
    if (width, height) == (28, 28) and (
        invert is False or (invert is None and border_med <= 15.0)
    ):
        processed_pil = Image.fromarray(arr_gray.astype(np.uint8), mode="L")
        return TENSOR_TRANSFORM(processed_pil).unsqueeze(0), processed_pil

    # Work at <=96px for fast border flood-fill before final 28x28 framing.
    scale_work = min(1.0, 96.0 / max(width, height))
    ww = max(28, int(round(width * scale_work)))
    wh = max(28, int(round(height * scale_work)))
    work_pil = rgb_pil.resize((ww, wh), Image.Resampling.BILINEAR)
    work_rgb = np.asarray(work_pil, dtype=np.float32)
    work_gray = np.asarray(work_pil.convert("L"), dtype=np.float32)

    do_invert = invert is True or (invert is None and border_med > 127.0)
    intensity = (255.0 - work_gray) if do_invert else work_gray.copy()

    fg_mask, dist = _extract_foreground_mask(work_rgb)
    # Only rely on flood-fill silhouette when it isolates a clear foreground region.
    if fg_mask.any() and fg_mask.mean() < 0.90:
        dist_norm = np.clip(dist * (255.0 / max(float(dist[fg_mask].max()), 1.0)), 0.0, 255.0)
        # Blend brightness and color distance so light details (e.g. white soles) stay solid.
        sil = np.where(fg_mask, np.maximum(intensity, np.maximum(dist_norm, 90.0)), 0.0)
        hi = float(np.percentile(sil[fg_mask], 99.0))
        if hi > 10.0:
            sil = np.where(fg_mask, np.clip(sil * (255.0 / hi), 0.0, 255.0), 0.0)
    else:
        b_int = np.concatenate(
            [intensity[0, :], intensity[-1, :], intensity[:, 0], intensity[:, -1]]
        )
        sil = np.clip(intensity - float(np.percentile(b_int, 85)), 0.0, 255.0)
        hi = float(np.percentile(sil, 99.5))
        if hi > 10.0:
            sil = np.clip(sil * (255.0 / hi), 0.0, 255.0)
        fg_mask = sil > max(20.0, 0.08 * float(sil.max()))

    if auto_crop and fg_mask.any():
        ys, xs = np.where(fg_mask)
        crop = sil[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        ch, cw = crop.shape
        scale = 27.0 / max(ch, cw)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        pil_crop = Image.fromarray(crop.astype(np.uint8), mode="L").resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        )
        processed_pil = Image.new("L", (28, 28), 0)
        processed_pil.paste(pil_crop, ((28 - new_w) // 2, (28 - new_h) // 2))
    else:
        processed_pil = Image.fromarray(sil.astype(np.uint8), mode="L").resize(
            (28, 28), Image.Resampling.LANCZOS
        )

    return TENSOR_TRANSFORM(processed_pil).unsqueeze(0), processed_pil


def predict_pil_image(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
    invert: bool | None = None,
    auto_crop: bool = True,
    top_k: int = 3,
) -> tuple[list[tuple[str, float]], Image.Image]:
    """Run top-k inference on an in-memory PIL image and return predictions + 28x28 input."""
    tensor, processed_pil = preprocess_pil_image(image, invert=invert, auto_crop=auto_crop)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_names)))
    preds = [
        (class_names[idx.item()], prob.item() * 100.0)
        for idx, prob in zip(top_indices, top_probs)
    ]
    return preds, processed_pil


def predict_single_image(
    image_path: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    invert: bool | None = None,
    auto_crop: bool = True,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Load a saved model checkpoint and return top-k (class_name, probability %) pairs."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run main.py first to train the model."
        )

    device = get_device()
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    class_names: list[str] = checkpoint["class_names"]
    with Image.open(image_path) as image:
        preds, _ = predict_pil_image(
            image, model, class_names, device, invert=invert, auto_crop=auto_crop, top_k=top_k
        )
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Fashion-MNIST class for a single image.")
    parser.add_argument("image_path", type=Path, help="Path to the input image file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the saved model checkpoint.",
    )
    bg_group = parser.add_mutually_exclusive_group()
    bg_group.add_argument(
        "--invert",
        action="store_true",
        help="Force color inversion (useful for dark clothing on a white background).",
    )
    bg_group.add_argument(
        "--no-invert",
        action="store_true",
        help="Disable automatic background inversion.",
    )
    args = parser.parse_args()

    invert_mode: bool | None = True if args.invert else (False if args.no_invert else None)
    predictions = predict_single_image(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        invert=invert_mode,
    )
    top_label, top_conf = predictions[0]
    print(f"\nPrediction: {top_label} ({top_conf:.2f}%)")
    print("Top-3 probabilities:")
    for rank, (label, conf) in enumerate(predictions, start=1):
        print(f"  {rank}. {label:<14} {conf:6.2f}%")


if __name__ == "__main__":
    main()
