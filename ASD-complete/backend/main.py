from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

BACKEND_DIR = Path(__file__).resolve().parent
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def resolve_model_path() -> Path:
    env_override = os.getenv("ASD_MODEL_PATH")
    if env_override:
        return Path(env_override)

    preferred_names = [
        "model.pth",
        "model.pt",
        "vit_asd_best.pth",
        "vit_asd_best.pt",
    ]

    for file_name in preferred_names:
        candidate = BACKEND_DIR / file_name
        if candidate.exists():
            return candidate

    available_models = sorted(list(BACKEND_DIR.glob("*.pth")) + list(BACKEND_DIR.glob("*.pt")))
    if len(available_models) == 1:
        return available_models[0]

    return BACKEND_DIR / "model.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = None
    app.state.model_error = None
    app.state.preprocess_image = None
    app.state.predict_image = None
    app.state.InvalidImageError = None
    app.state.model_path = None

    try:
        model_path = resolve_model_path()
        app.state.model_path = model_path

        try:
            from .model import load_trained_model, predict_image
            from .utils import InvalidImageError, preprocess_image
        except ImportError:  # pragma: no cover - allows running from backend/ directly
            from model import load_trained_model, predict_image
            from utils import InvalidImageError, preprocess_image

        app.state.preprocess_image = preprocess_image
        app.state.predict_image = predict_image
        app.state.InvalidImageError = InvalidImageError
        app.state.model_bundle = load_trained_model(model_path)
    except Exception as exc:  # pragma: no cover - startup safety path
        app.state.model_error = str(exc)

    yield


app = FastAPI(
    title="ASD Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    model_bundle = getattr(app.state, "model_bundle", None)
    model_error = getattr(app.state, "model_error", None)
    model_path = getattr(app.state, "model_path", None)

    return {
        "message": "ASD Detection API is running.",
        "model_loaded": bool(model_bundle),
        "architecture": getattr(model_bundle, "architecture", None),
        "model_path": getattr(model_path, "name", None),
        "class_names": getattr(model_bundle, "class_names", None),
        "input_size": getattr(model_bundle, "input_size", None),
        "health_url": "/health",
        "docs_url": "/docs",
        "detail": model_error,
    }


@app.get("/health")
def health_check():
    model_bundle = getattr(app.state, "model_bundle", None)
    model_error = getattr(app.state, "model_error", None)
    model_path = getattr(app.state, "model_path", None)

    return {
        "status": "ready" if model_bundle else "error",
        "model_loaded": bool(model_bundle),
        "model_path": getattr(model_path, "name", None),
        "architecture": getattr(model_bundle, "architecture", None),
        "class_names": getattr(model_bundle, "class_names", None),
        "input_size": getattr(model_bundle, "input_size", None),
        "detail": model_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model_bundle: Any | None = getattr(app.state, "model_bundle", None)
    model_error: str | None = getattr(app.state, "model_error", None)
    preprocess_image = getattr(app.state, "preprocess_image", None)
    predict_image = getattr(app.state, "predict_image", None)
    InvalidImageError = getattr(app.state, "InvalidImageError", None)

    if model_bundle is None or preprocess_image is None or predict_image is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=model_error or "Model is not available. Check the backend/model.pth file.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a valid image file.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        input_tensor = preprocess_image(image_bytes, input_size=model_bundle.input_size)
        prediction, confidence = predict_image(input_tensor, model_bundle)
    except Exception as exc:
        if InvalidImageError is not None and isinstance(exc, InvalidImageError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        if isinstance(exc, ValueError):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {exc}",
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected inference error: {exc}",
        ) from exc

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }
