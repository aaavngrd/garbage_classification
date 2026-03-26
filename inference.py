"""
CLI inference script for the Garbage Classification project.

Loads a trained MobileNetV2 checkpoint and classifies a single image
supplied via the command line.

Usage::

    python inference.py --image path/to/photo.jpg --model best_model.pt

Output example::

    Using device: cpu
    Predicted class: plastic
    Confidence: 97.31%
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE: int = 224
"""Input spatial resolution expected by the model (pixels)."""

IMAGENET_MEAN = [0.485, 0.456, 0.406]
"""Per-channel mean used for ImageNet normalisation."""

IMAGENET_STD = [0.229, 0.224, 0.225]
"""Per-channel standard deviation used for ImageNet normalisation."""


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def build_transform() -> transforms.Compose:
    """Create the deterministic image pre-processing pipeline for inference.

    The pipeline resizes any input image to ``IMG_SIZE × IMG_SIZE`` pixels,
    converts it to a float tensor, and applies standard ImageNet
    normalisation.

    Returns:
        A :class:`torchvision.transforms.Compose` instance that accepts a
        :class:`PIL.Image.Image` and produces a float tensor of shape
        ``(3, IMG_SIZE, IMG_SIZE)``.

    Example:
        >>> tfm = build_transform()
        >>> img = Image.open("photo.jpg").convert("RGB")
        >>> tensor = tfm(img)
        >>> tensor.shape
        torch.Size([3, 224, 224])
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    num_classes: int,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, int]]:
    """Load a fine-tuned MobileNetV2 checkpoint for inference.

    Constructs the MobileNetV2 architecture, replaces the classifier
    head to match *num_classes*, loads the saved weights, and switches
    the model to evaluation mode.

    Args:
        model_path: Path to the ``.pt`` checkpoint file produced by
            ``train.py``.
        num_classes: Number of output classes.  Must match the value used
            during training.
        device: Torch device to load the model onto.

    Returns:
        A 2-tuple ``(model, class_to_idx)`` where *model* is the
        ``nn.Module`` ready for inference and *class_to_idx* maps class
        name strings to integer indices.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
        RuntimeError: If the checkpoint's state-dict is incompatible with
            the reconstructed architecture.

    Example:
        >>> ckpt = torch.load("best_model.pt", map_location="cpu")
        >>> model, c2i = load_model(
        ...     "best_model.pt", len(ckpt["class_to_idx"]), torch.device("cpu")
        ... )
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=device)  # nosec B614
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["class_to_idx"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    image_path: str,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    device: torch.device,
) -> Tuple[str, float]:
    """Classify a single image file using a loaded model.

    Opens the image, applies the standard pre-processing pipeline, runs
    a forward pass, and returns the top-1 predicted class and its
    softmax confidence.

    Args:
        image_path: Filesystem path to the input image.  Accepted formats
            include JPEG, PNG, and WebP.
        model: A loaded, evaluation-mode ``nn.Module``.
        class_to_idx: Mapping from class name strings to integer indices,
            as returned by :func:`load_model`.
        device: Torch device to run inference on.

    Returns:
        A 2-tuple ``(predicted_class, confidence)`` where
        *predicted_class* is a string (e.g. ``"plastic"``) and
        *confidence* is a float in ``[0, 1]``.

    Raises:
        FileNotFoundError: If *image_path* does not exist.
        UnidentifiedImageError: If the file is not a valid image.

    Example:
        >>> pred, conf = predict("photo.jpg", model, class_to_idx, device)
        >>> print(f"{pred}: {conf:.2%}")
        plastic: 97.31%
    """
    transform = build_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class[pred_idx], confidence


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run single-image inference.

    Loads the checkpoint specified by ``--model``, classifies the image
    given by ``--image``, and prints the predicted class and confidence
    to *stdout*.

    Exits with code 1 if required arguments are missing.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Garbage classification inference")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument("--model", type=str, default="best_model.pt",
                        help="Path to the trained model checkpoint.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint = torch.load(args.model, map_location="cpu")  # nosec B614
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    model, class_to_idx = load_model(
        model_path=args.model,
        num_classes=num_classes,
        device=device,
    )

    pred_class, confidence = predict(args.image, model, class_to_idx, device)
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
