from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    answer: str = Field(..., description="Predicted answer: 'yes' or 'no'")
    confidence: float = Field(..., description="Softmax probability of the predicted answer")
    probabilities: dict[str, float] = Field(..., description="Per-class probabilities")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str
