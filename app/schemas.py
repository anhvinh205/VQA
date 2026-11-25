from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DetectedObject(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0, description="Object detector confidence score")


class PredictionResponse(BaseModel):
    answer: Literal["yes", "no"] = Field(..., description="Whether the requested object was detected")
    target: str = Field(..., description="Object category identified from the question")
    detection_score: float = Field(..., ge=0.0, le=1.0, description="Highest detector score for the target")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Score threshold used for the yes/no answer")
    model: str = Field(..., description="Detection model used")
    detected_objects: list[DetectedObject] = Field(..., description="Objects detected above the threshold")


class ClassifierPredictionResponse(BaseModel):
    answer: Literal["yes", "no"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[Literal["yes", "no"], float]

    @model_validator(mode="after")
    def validate_probabilities(self) -> "ClassifierPredictionResponse":
        if set(self.probabilities) != {"yes", "no"}:
            raise ValueError("probabilities must contain exactly 'yes' and 'no'")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-3:
            raise ValueError("probabilities must sum to 1")
        if abs(self.confidence - self.probabilities[self.answer]) > 1e-3:
            raise ValueError("confidence must match the probability of the answer")
        return self


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class ReadinessResponse(HealthResponse):
    pass


class ErrorResponse(BaseModel):
    detail: str
