import argparse

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform():
    """Build image preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(model_path: str, num_classes: int, device: torch.device):
    """Load trained MobileNetV2 model from checkpoint file."""
    net = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=device)  # nosec B614
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()
    return net, checkpoint["class_to_idx"]


@torch.no_grad()
def predict(image_path: str, net, class_to_idx: dict, device: torch.device):
    """Run inference on a single image and return predicted class and confidence."""
    transform = build_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    logits = net(tensor)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred_class = idx_to_class[pred_idx]
    return pred_class, confidence


def main():
    """Parse CLI arguments and run garbage classification inference."""
    parser = argparse.ArgumentParser(description="Garbage classification inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="best_model.pt",
                        help="Path to trained model")
    args = parser.parse_args()

    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", run_device)

    checkpoint = torch.load(args.model, map_location=run_device)  # nosec B614
    num_classes = len(checkpoint["class_to_idx"])

    run_model, run_class_to_idx = load_model(args.model, num_classes, run_device)

    pred_class, confidence = predict(args.image, run_model, run_class_to_idx, run_device)
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
