"""HTTP inference service for CPU or GPU cloud deployment."""

from __future__ import annotations

import base64
import io
import os

import joblib
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from fusion_model import FUSION_DIR, load_models, predict_fusion_image
from pipeline import ARTIFACTS, CLASSES, load_backbone, load_json, predict_image

app = FastAPI(title="Alliage panoramic positioning inference API", version="1.0")
SELECTED_MODEL = os.environ.get("ALLIAGE_MODEL", "fusion" if (FUSION_DIR / "training.json").exists() else "baseline")
if SELECTED_MODEL not in {"fusion", "baseline"}:
    raise ValueError("ALLIAGE_MODEL deve ser 'fusion' ou 'baseline'.")
REPORT_DIR = FUSION_DIR if SELECTED_MODEL == "fusion" else ARTIFACTS
training = load_json(REPORT_DIR / "training.json")
FEATURE_MODE = training["feature_mode"]
BACKBONE_KEY = training["backbone_key"]
TTA = training.get("tta", False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSIFIER = joblib.load(REPORT_DIR / "classifier.joblib")
BACKBONE = load_models(DEVICE, training["backbones"]) if SELECTED_MODEL == "fusion" else load_backbone(DEVICE, BACKBONE_KEY)
API_KEY = os.environ.get("ALLIAGE_API_KEY", "")


def authorize(provided_key: str | None) -> None:
    if API_KEY and provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API ausente ou inválida")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "device": DEVICE, "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:  # noqa: B008
    authorize(x_api_key)
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Envie uma imagem JPG ou PNG")
    payload = await file.read()
    if len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="A imagem excede o limite de 25 MB")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="O arquivo não é uma imagem válida") from exc
    result = (predict_fusion_image(image, CLASSIFIER, BACKBONE, DEVICE) if SELECTED_MODEL == "fusion"
              else predict_image(image, CLASSIFIER, BACKBONE, DEVICE, FEATURE_MODE, BACKBONE_KEY, TTA))
    overlay = io.BytesIO()
    result["overlay"].save(overlay, format="PNG")
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "scores": result["scores"],
        "elapsed_ms": result["elapsed_ms"],
        "device": DEVICE,
        "overlay_png": base64.b64encode(overlay.getvalue()).decode("ascii"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
