"""
Garbage Classification API Server.

This module provides a FastAPI-based REST API for garbage image classification
using a pre-trained MobileNetV2 model fine-tuned on waste categories.

Example:
    Start the server with::

        uvicorn server:app --host 0.0.0.0 --port 8000

    Then navigate to http://localhost:8000/docs for the interactive Swagger UI.
"""

import io
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pt"

app = FastAPI(
    title="Garbage Classification API",
    description=(
        "REST API for automatic garbage image classification. "
        "Accepts a waste image and returns the predicted category "
        "with a confidence score. Supports classes: paper, plastic, "
        "glass, metal, organic."
    ),
    version="1.0.0",
    contact={
        "name": "Garbage Classification Project",
        "url": "https://github.com/aaavngrd/garbage_classification",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response schema for the health-check endpoint.

    Attributes:
        status: Human-readable server status string.
        device: Torch device currently used for inference (``cpu`` or ``cuda``).
        classes: Ordered list of class names the model can predict.
    """

    status: str
    device: str
    classes: List[str]


class PredictionResponse(BaseModel):
    """Response schema for the prediction endpoint.

    Attributes:
        predicted_class: The waste category predicted by the model.
        confidence: Softmax confidence score in the range [0, 1].
    """

    predicted_class: str
    confidence: float


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_classes(
    model_path: str = MODEL_PATH,
    device: torch.device = DEVICE,
):
    """Load a fine-tuned MobileNetV2 checkpoint from disk.

    Reads the checkpoint saved by ``train.py``, reconstructs the
    MobileNetV2 architecture with the correct number of output classes,
    loads the saved weights, and sets the model to evaluation mode.

    Args:
        model_path: Filesystem path to the ``.pt`` checkpoint file.
            Defaults to ``best_model.pt`` in the working directory.
        device: Torch device to map the model and weights to.
            Defaults to the module-level ``DEVICE`` constant.

    Returns:
        A tuple ``(model, class_names)`` where *model* is the loaded
        ``torch.nn.Module`` ready for inference, and *class_names* is a
        list of string labels ordered by class index.

    Raises:
        FileNotFoundError: If *model_path* does not exist on disk.
        KeyError: If the checkpoint is missing expected keys
            (``model_state_dict`` or ``class_to_idx``).

    Example:
        >>> model, classes = load_model_and_classes("best_model.pt")
        >>> print(classes)
        ['glass', 'metal', 'organic', 'paper', 'plastic']
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)  # nosec B614
    class_to_idx: Dict[str, int] = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


model, CLASS_NAMES = load_model_and_classes()

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

#: Standard ImageNet normalisation transform applied before inference.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Utility"],
)
def root() -> HealthResponse:
    """Return server status and available prediction classes.

    This endpoint can be used as a liveness probe to confirm the API is
    running and the model has loaded successfully.

    Returns:
        A :class:`HealthResponse` containing the status string, the active
        compute device, and the list of recognisable waste categories.

    Example response::

        {
          "status": "API is running",
          "device": "cpu",
          "classes": ["glass", "metal", "organic", "paper", "plastic"]
        }
    """
    return HealthResponse(
        status="API is running",
        device=str(DEVICE),
        classes=CLASS_NAMES,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a waste image",
    tags=["Inference"],
    responses={
        200: {
            "description": "Successful classification",
            "content": {
                "application/json": {
                    "example": {
                        "predicted_class": "plastic",
                        "confidence": 0.9731,
                    }
                }
            },
        },
        500: {"description": "Internal server error during inference"},
    },
)
async def predict(file: UploadFile = File(..., description="Waste image file (JPEG or PNG)")) -> JSONResponse:
    """Classify an uploaded waste image into one of the known categories.

    Reads the uploaded image, applies EXIF orientation correction,
    runs it through the MobileNetV2 classifier, and returns the top-1
    prediction with its softmax confidence.

    Args:
        file: Multipart image upload.  Accepted formats: JPEG, PNG, WebP.
            Images are automatically resized to 224 × 224 pixels.

    Returns:
        A JSON object with fields:

        - ``predicted_class`` (*str*): One of ``glass``, ``metal``,
          ``organic``, ``paper``, ``plastic``.
        - ``confidence`` (*float*): Softmax probability rounded to 4
          decimal places; range [0, 1].

    Raises:
        HTTPException: 500 if the image cannot be decoded or inference fails.

    Example:
        Upload ``photo.jpg`` with curl::

            curl -X POST http://localhost:8000/predict \\
                 -F "file=@photo.jpg"

        Expected response::

            {"predicted_class": "plastic", "confidence": 0.9731}
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        x = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

        return JSONResponse({
            "predicted_class": CLASS_NAMES[int(idx)],
            "confidence": round(float(conf), 4),
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
