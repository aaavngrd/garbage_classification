import io
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torchvision import models, transforms

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pt"


app = FastAPI(
    title="Garbage Classification API",
    version="1.0"
)


# Model + classes
def load_model_and_classes():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE) # nosec B614

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


model, CLASS_NAMES = load_model_and_classes()


# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Endpoints
@app.get("/")
def root():
    return {
        "status": "API is running",
        "device": str(DEVICE),
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
            "confidence": round(float(conf), 4)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
