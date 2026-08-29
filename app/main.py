from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthResponse, PredictionResponse
from src.predictor import VQAPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

predictor_state: dict[str, VQAPredictor | None] = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor_state["predictor"] = VQAPredictor()
        logger.info("model loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("model artifacts not found, /predict will return 503: %s", exc)
        predictor_state["predictor"] = None
    yield
    predictor_state.clear()


app = FastAPI(
    title="Visual Question Answering API",
    description="Yes/No Visual Question Answering service (CNN + BiLSTM backbone).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root():
    return {"service": "vqa-api", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok", model_loaded=predictor_state["predictor"] is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    image: UploadFile = File(..., description="JPEG/PNG image"),
    question: str = Form(..., description="Yes/No question about the image"),
):
    predictor = predictor_state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and place a checkpoint first.")

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {image.content_type}")

    if not question or not question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty")

    try:
        result = predictor.predict(image_bytes, question)
    except Exception as exc:  # noqa: BLE001 - surface as a clean 400 to the client
        logger.exception("prediction failed")
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    return PredictionResponse(**result)
