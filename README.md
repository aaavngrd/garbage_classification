# Garbage Classification System

## Project Overview

This project is a machine learning–based system for automatic classification of waste images.
The system analyzes an input image and predicts the type of garbage,
helping to improve waste sorting and recycling processes.

The project was developed as part of an academic assignment and includes:
- Model training using PyTorch (MobileNetV2 fine-tuning)
- REST API for inference using FastAPI
- CLI inference script

---

## Technologies Used

- Python 3.10+
- PyTorch 2.x
- FastAPI
- torchvision
- Sphinx + autodoc (documentation generation)

---

## Dataset

The project uses an image dataset of household waste with the following classes:
- paper
- plastic
- glass
- metal
- organic

The dataset is split into training, validation, and test subsets.

> **Note:** Due to size limitations, the dataset is not included in the repository.
> Dataset directories (`data/`, `raw_dataset/`, `inference_images/`) are excluded via `.gitignore`.

Expected local dataset structure:

```
data/
├── train/
├── val/
└── test/
```

---

## Project Structure

```
garbage_classification/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── docs/
│   ├── generate_docs.md      # Instructions for generating Sphinx docs
│   ├── conf.py               # Sphinx configuration
│   └── index.rst             # Sphinx root document
├── tests/
├── server.py                 # FastAPI server for inference
├── inference.py              # CLI inference script
├── train.py                  # Model training script
├── split_dataset.py          # Dataset splitting utility
├── README.md
└── .gitignore
```

---

## Quick Start

### Training

```bash
python train.py \
  --data_dir data/ \
  --epochs 15 \
  --batch_size 32 \
  --lr 0.001 \
  --output best_model.pt
```

### Running the API server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Interactive API docs available at: http://localhost:8000/docs

### CLI Inference

```bash
python inference.py --image path/to/photo.jpg --model best_model.pt
```

---

## Documentation

### Standards used

All Python source files in this project follow **Google-style docstrings**
(compatible with Sphinx + `napoleon` extension).

Every public function, class, and module must include:

| Element | Required sections |
|---------|-------------------|
| Module | One-line summary, description, usage example |
| Function / method | Summary, `Args`, `Returns`, `Raises` (if applicable), `Example` |
| Dataclass | Summary, `Attributes` for every field |
| Constants | Inline comment or module-level docstring |

### How to write a compliant docstring

```python
def load_model(model_path: str, num_classes: int, device: torch.device):
    """Load a fine-tuned MobileNetV2 checkpoint for inference.

    Constructs the MobileNetV2 architecture, replaces the classifier
    head to match *num_classes*, loads the saved weights, and switches
    the model to evaluation mode.

    Args:
        model_path: Path to the ``.pt`` checkpoint file produced by
            ``train.py``.
        num_classes: Number of output classes. Must match training value.
        device: Torch device to load the model onto.

    Returns:
        A 2-tuple ``(model, class_to_idx)`` where *model* is the
        ``nn.Module`` ready for inference.

    Raises:
        FileNotFoundError: If *model_path* does not exist.

    Example:
        >>> model, c2i = load_model("best_model.pt", 5, torch.device("cpu"))
    """
```

### Generating documentation

See [`docs/generate_docs.md`](docs/generate_docs.md) for full instructions.

Quick start:

```bash
pip install sphinx sphinx-autodoc-typehints furo
cd docs
make html
# Open docs/_build/html/index.html
```

### Documentation maintenance rules

When contributing to this project, follow these rules:

1. **New function / method** → add a complete Google-style docstring before merging.
2. **Changed signature** → update `Args` / `Returns` in the docstring in the same commit.
3. **New module** → add a module-level docstring with a usage example.
4. **API endpoint added or changed** → update the FastAPI `summary`, `description`, and `responses` in `server.py`. The OpenAPI spec at `/docs` regenerates automatically.
5. **Do not merge** pull requests that introduce undocumented public interfaces.

---

## API Reference

The FastAPI server auto-generates interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check — returns server status and class list |
| POST | `/predict` | Classify an uploaded image |

#### POST /predict

**Request**: `multipart/form-data` with field `file` (JPEG/PNG image).

**Response**:
```json
{
  "predicted_class": "plastic",
  "confidence": 0.9731
}
```

---

## Running Tests

```bash
pytest tests/ -v --cov=.
```
