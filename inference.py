import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(model_path: str, num_classes: int, device: torch.device):
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, checkpoint["class_to_idx"]


# Inference
@torch.no_grad()
def predict(image_path: str, model, class_to_idx, device):
    transform = build_transform()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    logits = model(image)
    probs = torch.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred_class = idx_to_class[pred_idx]

    return pred_class, confidence


# Main
def main():
    parser = argparse.ArgumentParser(description="Garbage classification inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="best_model.pt", help="Path to trained model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, class_to_idx = load_model(
        model_path=args.model,
        num_classes=len(class_to_idx := None) if False else None, 
        device=device
    )


if __name__ == "__main__":
    checkpoint = torch.load("best_model.pt", map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_to_idx = load_model(
        model_path="best_model.pt",
        num_classes=num_classes,
        device=device
    )

    import sys
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    if image_path is None:
        print("Please provide image path: --image path/to/image.jpg")
        sys.exit(1)

    pred_class, confidence = predict(image_path, model, class_to_idx, device)

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
