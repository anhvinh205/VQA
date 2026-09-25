from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionResponse(BaseModel):
    answer: Literal["yes", "no"] = Field(..., description="Predicted answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Predicted-class probability")
    probabilities: dict[Literal["yes", "no"], float] = Field(
        ..., description="Per-class probabilities for yes and no"
    )

    @model_validator(mode="after")
    def validate_probabilities(self) -> "PredictionResponse":
        if set(self.probabilities) != {"yes", "no"}:
            raise ValueError("probabilities must contain exactly 'yes' and 'no'")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-3:
            raise ValueError("probabilities must sum to 1")
        if abs(self.confidence - self.probabilities[self.answer]) > 1e-3:
            raise ValueError("confidence must match the probability of answer")
        return self


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class ReadinessResponse(HealthResponse):
    pass


class ErrorResponse(BaseModel):
    detail: str
