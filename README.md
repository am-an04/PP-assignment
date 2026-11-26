# Fashion-MNIST Clothing Classifier (PyTorch)

A clean, beginner-friendly PyTorch project that trains and compares a baseline **Multi-Layer Perceptron (MLP)** against a **3-block Convolutional Neural Network (CNN)** on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset (10 clothing categories, 28×28 grayscale images).

---

## Project Abstract

This project presents an end-to-end clothing image classification system built in PyTorch and evaluated on the Fashion-MNIST benchmark, which contains 70,000 grayscale images across ten apparel categories. To demonstrate the practical advantage of spatial feature extraction over flat dense representations, two neural architectures are implemented, trained, and compared under identical optimization conditions: a baseline Multi-Layer Perceptron (MLP) and a three-block Convolutional Neural Network (CNN) incorporating Batch Normalization, ReLU activations, Max Pooling, and Dropout. The 60,000 training images are split into 54,000 training and 6,000 validation samples, with horizontal flip and rotation augmentations applied strictly to the training split. Trained using Cross-Entropy loss, Adam, and StepLR scheduling for 12 epochs, the MLP achieves 87.89% test accuracy while the CNN reaches 92.12%. Comprehensive evaluation includes training curves, a test-set confusion matrix, per-class accuracy, single-image inference, and an interactive Gradio interface, with the best checkpoint saved automatically.

---

## Folder Structure

```text
fashion-classifier/
├── data/                   # Auto-downloaded Fashion-MNIST files
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Train/val/test split (54k / 6k / 10k) & augmentations
│   ├── models.py           # BaselineMLP and FashionCNN architectures
│   ├── train.py            # Training/validation loops, StepLR, checkpointing
│   ├── evaluate.py         # Test metrics, confusion matrix, and PNG plots
│   └── utils.py            # Seed, device selection, checkpoint load/save
├── outputs/
│   ├── best_model.pth      # Overall best model checkpoint (CNN)
│   ├── mlp_best.pth        # Best MLP checkpoint
│   ├── cnn_best.pth        # Best CNN checkpoint
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
├── main.py                 # Full training & evaluation pipeline
├── predict.py              # Single-image CLI inference
├── app.py                  # Optional Gradio web interface
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install Dependencies

Python 3.10+ is required.

```bash
cd fashion-classifier
pip install -r requirements.txt
```

### 2. Train and Evaluate Both Models

Runs data download (into `./data`), trains `BaselineMLP` and `FashionCNN` for 12 epochs each (configurable at the top of `main.py` or via CLI flags), saves the best model to `outputs/best_model.pth`, prints per-class accuracy + comparison table, and saves all PNG plots in `outputs/`.

```bash
python main.py
```

Optional quick smoke test (1 epoch):

```bash
python main.py --epochs 1
```

### 3. Predict on a Single Image

`predict.py` automatically handles transparent PNGs, detects light/white backgrounds to match Fashion-MNIST's dark-background format, and centers the garment while preserving its aspect ratio.

```bash
python predict.py path/to/clothing_image.png
```

You can also explicitly control background inversion with `--invert` or `--no-invert`:

```bash
python predict.py path/to/clothing_image.png --invert
```

### 4. Launch the Optional Gradio App

```bash
python app.py
```

---

## Model Architectures (In Simple Words)

### 1. Baseline: `BaselineMLP` (235,146 parameters)
- Flattens each `28×28` image into a 1D list of `784` pixel numbers.
- Passes those numbers through two fully connected hidden layers (`784 → 256 → 128 → 10`) with `ReLU` activations and `Dropout(0.2)`.
- **Limitation:** Flattening discards 2D spatial layout—neighboring pixels above, below, left, and right are no longer grouped together.

### 2. Improved: `FashionCNN` (391,370 parameters)
- Keeps the 2D image structure intact and learns local visual patterns (edges, collars, sleeves, soles) using three convolutional blocks:
  - **Block 1:** `Conv2d(1 → 32, 3×3)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)` → `Dropout2d(0.1)` (`28×28 → 14×14`)
  - **Block 2:** `Conv2d(32 → 64, 3×3)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)` → `Dropout2d(0.15)` (`14×14 → 7×7`)
  - **Block 3:** `Conv2d(64 → 128, 3×3)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)` → `Dropout2d(0.2)` (`7×7 → 3×3`)
- Flattens the `128×3×3` feature maps into a classifier head (`1152 → 256 → 10`) with `BatchNorm1d`, `ReLU`, and `Dropout(0.4)`.

---

## Results

### Model Comparison (12 Epochs, Batch Size 64, Adam `lr=0.001` + `StepLR`)

| Model | Trainable Parameters | Best Validation Accuracy | Official Test Accuracy |
| :--- | :---: | :---: | :---: |
| **BaselineMLP** | 235,146 | 88.67% | **87.89%** |
| **FashionCNN** | 391,370 | 92.70% | **92.12%** |

### Per-Class Test Accuracy (`FashionCNN`)

| Class | MLP Test Acc | CNN Test Acc |
| :--- | :---: | :---: |
| T-shirt/top | 82.70% | 87.80% |
| Trouser | 96.90% | 98.30% |
| Pullover | 79.00% | 91.40% |
| Dress | 89.00% | 94.50% |
| Coat | 79.90% | 88.00% |
| Sandal | 95.70% | 97.10% |
| Shirt | 69.50% | 71.90% |
| Sneaker | 92.10% | 96.80% |
| Bag | 97.50% | 98.70% |
| Ankle boot | 96.60% | 96.70% |

### Plots

**Training & Validation Curves**

![Training and Validation Loss and Accuracy](outputs/training_curves.png)

**Confusion Matrix (CNN)**

![Confusion Matrix on Official Test Set](outputs/confusion_matrix.png)

**16 Sample Test Predictions**

![16 Test Set Predictions](outputs/sample_predictions.png)

---

## 6 Likely Viva Questions & Short Answers

1. **Why did we use BatchNorm in the CNN?**
   Batch Normalization normalizes layer activations within each mini-batch to have zero mean and unit variance (plus learnable scale/shift). This stabilizes gradient flow, prevents exploding/vanishing activations, allows faster convergence, and adds mild regularization.

2. **Why do we split the training set into 54,000 train and 6,000 validation instead of tuning on the test set?**
   The validation set acts as an unbiased checkpoint during training for early model selection and hyperparameter tuning. Keeping the official 10,000-image test set untouched until final evaluation prevents "data leakage" and ensures the reported test accuracy reflects true generalization.

3. **What is overfitting, and how did we prevent it here?**
   Overfitting occurs when a model memorizes training examples (high train accuracy) instead of learning general patterns (stagnating or worsening validation accuracy). We reduced overfitting using data augmentation (`RandomHorizontalFlip`, `RandomRotation`), `Dropout`/`Dropout2d`, `BatchNorm`, and saving only the checkpoint with the highest validation accuracy.

4. **Why do we apply data augmentation only to the training set and not to validation or test sets?**
   Augmentations create slightly altered versions of training clothes so the model generalizes across small shifts and orientations. Validation and test sets must remain clean and consistent so we can measure model performance fairly on real, unaltered images.

5. **Why does the CNN outperform the MLP on Fashion-MNIST?**
   An MLP flattens the 2D image into a 1D vector, destroying spatial relationships between neighboring pixels and requiring separate weights per pixel position. A CNN uses shared 3×3 filters that preserve 2D geometry, detect local features (collars, hems, shoe soles) anywhere in the image (translation equivariance), and build hierarchical shapes across layers.

6. **Why did we use `CrossEntropyLoss` and a `StepLR` scheduler?**
   `CrossEntropyLoss` combines `LogSoftmax` and negative log-likelihood loss in a numerically stable way, making it the standard loss for multi-class single-label classification. `StepLR` halves the learning rate every 4 epochs so the model learns quickly at first and makes finer, more accurate weight updates near convergence.
