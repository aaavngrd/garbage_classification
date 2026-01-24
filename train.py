import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# Utils
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


# Config
@dataclass
class TrainConfig:
    data_dir: str
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    img_size: int
    seed: int
    freeze_backbone: bool
    output_path: str


# Data
def build_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
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


def build_dataloaders(cfg: TrainConfig):
    data_root = Path(cfg.data_dir)
    transforms_map = build_transforms(cfg.img_size)

    datasets_map = {
        split: datasets.ImageFolder(
            root=data_root / split,
            transform=transforms_map[split]
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


# Model
def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# Train / Eval
def train_one_epoch(model, loader, optimizer, device):
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
def evaluate(model, loader, device):
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


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Garbage classification with MobileNetV2 (PyTorch)"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--output", type=str, default="best_model.pt")

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
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, loaders["val"], device
        )

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                },
                cfg.output_path,
            )
            print(f"✅ Saved best model to {cfg.output_path}")

    test_loss, test_acc = evaluate(
        model, loaders["test"], device
    )
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
