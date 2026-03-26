"""
Model training script for the Garbage Classification project.

This module orchestrates the full training pipeline: dataset loading,
MobileNetV2 fine-tuning, validation, and checkpoint saving.

Usage::

    python train.py --data_dir data/ --epochs 15 --batch_size 32

The script writes the best model weights (measured by validation accuracy)
to ``--output`` (default ``best_model.pt``).
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training runs.

    Sets Python's built-in ``random``, NumPy, and all PyTorch random
    number generators (CPU and every CUDA device).  Also disables
    cuDNN's non-deterministic auto-tuner.

    Args:
        seed: Integer seed value.  Defaults to ``42``.

    Example:
        >>> set_seed(0)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy from raw model logits.

    Converts logits to class predictions via ``argmax`` and compares
    them element-wise against ground-truth labels.

    Args:
        logits: Float tensor of shape ``(N, C)`` — the raw (unnormalised)
            class scores produced by the model.
        targets: Long tensor of shape ``(N,)`` — ground-truth class indices
            in the range ``[0, C)``.

    Returns:
        Top-1 accuracy as a Python float in the range ``[0.0, 1.0]``.

    Example:
        >>> logits = torch.tensor([[2.0, 0.5], [0.1, 3.0]])
        >>> targets = torch.tensor([0, 1])
        >>> accuracy_from_logits(logits, targets)
        1.0
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Hyper-parameter and path configuration for a training run.

    All fields map directly to CLI arguments parsed in :func:`main`.

    Attributes:
        data_dir: Root directory that contains ``train/``, ``val/``, and
            ``test/`` sub-folders in ImageFolder layout.
        epochs: Total number of training epochs.
        batch_size: Mini-batch size used for all data splits.
        lr: Initial Adam learning rate.
        num_workers: Number of worker processes for :class:`DataLoader`.
        img_size: Height and width (in pixels) images are resized to.
        seed: Random seed forwarded to :func:`set_seed`.
        freeze_backbone: If ``True``, all MobileNetV2 feature layers are
            frozen and only the classifier head is trained.
        output_path: File path where the best model checkpoint is saved.
    """

    data_dir: str
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    img_size: int
    seed: int
    freeze_backbone: bool
    output_path: str


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    """Create ImageNet-normalised torchvision transform pipelines.

    Returns separate pipelines for training (with data augmentation)
    and evaluation / testing (deterministic resize + normalise only).

    The training pipeline applies:
    * Random horizontal flip (p = 0.5)
    * Colour jitter (brightness, contrast, saturation, hue)

    Args:
        img_size: Target spatial resolution.  Both height and width are
            set to this value (square crops).

    Returns:
        A dict with keys ``"train"``, ``"val"``, and ``"test"``, each
        mapping to a :class:`transforms.Compose` instance.

    Example:
        >>> tfms = build_transforms(224)
        >>> img_tensor = tfms["val"](some_pil_image)
        >>> img_tensor.shape
        torch.Size([3, 224, 224])
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return {"train": train_tfms, "val": eval_tfms, "test": eval_tfms}


def build_dataloaders(cfg: TrainConfig) -> Tuple[Dict[str, DataLoader], int, Dict[str, int]]:
    """Construct PyTorch DataLoaders for train, validation, and test splits.

    Expects the dataset to follow the ``ImageFolder`` convention::

        <cfg.data_dir>/
            train/
                class_a/  img1.jpg  img2.jpg …
                class_b/  …
            val/
                …
            test/
                …

    Args:
        cfg: A :class:`TrainConfig` instance with at minimum ``data_dir``,
            ``batch_size``, ``num_workers``, and ``img_size`` populated.

    Returns:
        A 3-tuple ``(loaders, num_classes, class_to_idx)`` where:

        - *loaders*: dict mapping ``"train"`` / ``"val"`` / ``"test"`` to
          :class:`DataLoader`.
        - *num_classes*: Integer count of unique classes.
        - *class_to_idx*: Mapping from class name string to integer index.

    Raises:
        FileNotFoundError: If any of the three split directories are absent.
    """
    data_root = Path(cfg.data_dir)
    transforms_map = build_transforms(cfg.img_size)

    datasets_map = {
        split: datasets.ImageFolder(
            root=data_root / split,
            transform=transforms_map[split],
        )
        for split in ["train", "val", "test"]
    }

    class_to_idx = datasets_map["train"].class_to_idx

    loaders = {
        split: DataLoader(
            datasets_map[split],
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        for split in ["train", "val", "test"]
    }

    return loaders, len(class_to_idx), class_to_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    """Build a MobileNetV2 model with a custom classification head.

    Downloads ImageNet-pretrained weights, optionally freezes all
    convolutional feature layers, and replaces the final linear layer
    with a new ``nn.Linear(in_features, num_classes)`` layer.

    Args:
        num_classes: Number of output classes for the classifier head.
        freeze_backbone: When ``True``, gradients are disabled for all
            layers in ``model.features``, making only the head trainable.

    Returns:
        An ``nn.Module`` ready to be moved to a device and trained.

    Example:
        >>> model = build_model(num_classes=5, freeze_backbone=False)
        >>> model(torch.randn(1, 3, 224, 224)).shape
        torch.Size([1, 5])
    """
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run a single training epoch over the provided data loader.

    Sets the model to training mode, iterates over mini-batches,
    computes cross-entropy loss, back-propagates gradients, and
    updates weights via the given optimizer.

    Args:
        model: The neural network to train.
        loader: DataLoader yielding ``(images, labels)`` mini-batches.
        optimizer: Gradient-based optimiser (e.g. :class:`torch.optim.Adam`).
        device: Compute device for tensor operations.

    Returns:
        A 2-tuple ``(mean_loss, mean_accuracy)`` averaged over all samples
        in *loader*.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy_from_logits(logits, labels) * bs
        total += bs

    return running_loss / total, running_acc / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate a model on the provided data loader without gradient tracking.

    Sets the model to evaluation mode and computes cross-entropy loss
    and top-1 accuracy over the full dataset split.

    Args:
        model: The neural network to evaluate.
        loader: DataLoader yielding ``(images, labels)`` mini-batches.
        device: Compute device for tensor operations.

    Returns:
        A 2-tuple ``(mean_loss, mean_accuracy)`` averaged over all samples
        in *loader*.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy_from_logits(logits, labels) * bs
        total += bs

    return running_loss / total, running_acc / total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments, run training, and save the best checkpoint.

    Parses command-line arguments into a :class:`TrainConfig`, seeds the
    RNG, builds dataloaders and model, trains for the requested number of
    epochs, and evaluates on the test set at the end.

    The checkpoint that achieves the highest validation accuracy during
    training is saved to ``--output`` in the format::

        {
            "model_state_dict": <OrderedDict>,
            "class_to_idx": <Dict[str, int]>,
        }
    """
    parser = argparse.ArgumentParser(
        description="Garbage classification with MobileNetV2 (PyTorch)"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory with train/val/test subfolders.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze MobileNetV2 feature layers.")
    parser.add_argument("--output", type=str, default="best_model.pt",
                        help="Output path for the best checkpoint.")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
        freeze_backbone=args.freeze_backbone,
        output_path=args.output,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loaders, num_classes, class_to_idx = build_dataloaders(cfg)
    print("Classes:", class_to_idx)

    model = build_model(num_classes, cfg.freeze_backbone)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=cfg.lr)

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], device)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "class_to_idx": class_to_idx},
                cfg.output_path,
            )
            print(f"✅ Saved best model to {cfg.output_path}")

    test_loss, test_acc = evaluate(model, loaders["test"], device)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
