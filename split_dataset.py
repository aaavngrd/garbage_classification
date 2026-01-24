import random
import shutil
from pathlib import Path

SOURCE_DIR = "raw_dataset"
TARGET_DIR = "data/garbage5"

SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

SEED = 42
random.seed(SEED)

CLASSES = ["plastic", "paper", "glass", "metal", "organic"]

def main():
    for split in SPLIT:
        for cls in CLASSES:
            Path(TARGET_DIR, split, cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        images = list(Path(SOURCE_DIR, cls).glob("*.jpg"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["val"])

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, files in splits.items():
            for img in files:
                shutil.copy2(img, Path(TARGET_DIR, split, cls, img.name))

    print("Dataset successfully split")

if __name__ == "__main__":
    main()
