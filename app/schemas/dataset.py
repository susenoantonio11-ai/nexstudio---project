"""Pydantic schemas for dataset endpoints."""
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime


class DatasetColumnOut(BaseModel):
    name: str
    inferred_type: str
    n_missing: int
    n_unique: int
    completeness_pct: float
    is_potential_target: bool
    target_score: Optional[float] = None
    stats: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class DatasetOut(BaseModel):
    id: int
    name: str
    original_filename: Optional[str] = None
    file_format: Optional[str] = None
    file_size_bytes: Optional[int] = None
    n_rows: Optional[int] = None
    n_columns: Optional[int] = None
    suggested_target: Optional[str] = None
    target_confidence: Optional[float] = None
    target_reasoning: Optional[str] = None
    status: str
    created_at: datetime
    columns: List[DatasetColumnOut] = []

    class Config:
        from_attributes = True


class DatasetPreview(BaseModel):
    columns: List[str]
    sample_rows: List[Dict[str, Any]]
    n_total_rows: int


class ExperimentRequest(BaseModel):
    dataset_id: int
    target_column: Optional[str] = None
    objective: str = "auto"
    use_case: Optional[str] = None
    name: Optional[str] = None


class ExperimentOut(BaseModel):
    id: int
    name: Optional[str]
    target_variable: Optional[str]
    model_type: Optional[str]
    selected_algorithm: Optional[str]
    status: str
    confidence_score: Optional[float]
    metrics: Optional[Dict[str, Any]]
    selection_reasoning: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class NexaInsight(BaseModel):
    """Structured Nexa AI Agent output."""
    DATA_SUMMARY: str
    DETECTED_TARGET_VARIABLE: str
    SELECTED_METHOD: str
    WHY_THIS_METHOD: str
    WHY_NOT_OTHER_METHODS: str
    MODEL_RESULT: str
    INSIGHT: str
    RECOMMENDATION: str
    CONFIDENCE_SCORE: str
