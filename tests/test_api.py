import io
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))


# Helpers
def make_image_bytes(fmt: str = "JPEG", size=(96, 96)) -> bytes:
    img = Image.new("RGB", size, (120, 30, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes: int = 5, winner_idx: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.winner_idx = winner_idx

    def forward(self, x):
        logits = torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32)
        logits[:, self.winner_idx] = 10.0
        return logits


# Fixtures
@pytest.fixture(scope="session")
def server_module():
    os.environ["UNIT_TESTING"] = "1"

    import importlib
    import server as server_mod
    importlib.reload(server_mod)

    server_mod.CLASS_NAMES = ["paper", "plastic", "glass", "metal", "organic"]
    server_mod.model = DummyModel(num_classes=5, winner_idx=1)

    return server_mod


@pytest.fixture(scope="session")
def client(server_module):
    return TestClient(server_module.app)


# TC-01 | FR-01, FR-02
# Upload valid JPG -> returns class + confidence
def test_tc01_upload_valid_jpg_returns_class_and_confidence(client):
    files = {"file": ("img.jpg", make_image_bytes("JPEG"), "image/jpeg")}
    r = client.post("/predict", files=files)

    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert isinstance(data["predicted_class"], str)
    assert isinstance(data["confidence"], float)

# TC-02 | FR-01
# Upload PNG -> classification success
def test_tc02_upload_png_success(client):
    files = {"file": ("img.png", make_image_bytes("PNG"), "image/png")}
    r = client.post("/predict", files=files)

    assert r.status_code == 200


# TC-03 | FR-07
# Upload text file -> returns error message
def test_tc03_upload_text_file_returns_error(client):
    files = {"file": ("file.txt", b"hello", "text/plain")}
    r = client.post("/predict", files=files)

    assert r.status_code in (400, 415, 500)
    assert "error" in r.json()


# TC-04 | FR-04
# Confidence exists and numeric
def test_tc04_confidence_present_and_numeric(client):
    files = {"file": ("img.jpg", make_image_bytes("JPEG"), "image/jpeg")}
    r = client.post("/predict", files=files)

    assert r.status_code == 200
    data = r.json()
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


# TC-05 | FR-08
# Simulate server error -> returns error message
def test_tc05_simulate_server_error_returns_error(client, server_module, monkeypatch):
    def broken_forward(*args, **kwargs):
        raise RuntimeError("Simulated server error")

    monkeypatch.setattr(server_module.model, "forward", broken_forward, raising=True)

    files = {"file": ("img.jpg", make_image_bytes("JPEG"), "image/jpeg")}
    r = client.post("/predict", files=files)

    assert r.status_code >= 500
    assert "error" in r.json()


# TC-06 | FR-05
# Send image via API -> valid JSON response
def test_tc06_send_image_via_api_returns_valid_json(client):
    files = {"file": ("api.jpg", make_image_bytes("JPEG"), "image/jpeg")}
    r = client.post("/predict", files=files)

    assert r.status_code == 200
    assert isinstance(r.json(), dict)
