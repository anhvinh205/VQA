from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthResponse, PredictionResponse, ReadinessResponse
from src.predictor import VQAPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
API_KEY = os.environ.get("VQA_API_KEY")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("VQA_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

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
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%d duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/", tags=["meta"])
def root():
    return {"service": "vqa-api", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    loaded = predictor_state["predictor"] is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.get("/live", response_model=HealthResponse, tags=["meta"])
def live():
    return HealthResponse(status="alive", model_loaded=predictor_state["predictor"] is not None)


@app.get("/ready", response_model=ReadinessResponse, tags=["meta"])
def ready(response: Response):
    loaded = predictor_state["predictor"] is not None
    if not loaded:
        response.status_code = 503
    return ReadinessResponse(status="ready" if loaded else "not_ready", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    image: UploadFile = File(..., description="JPEG/PNG image"),
    question: str = Form(..., description="Yes/No question about the image"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if API_KEY is not None and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Valid X-API-Key header required")

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
