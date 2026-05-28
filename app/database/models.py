"""
NEXLYTICS Database Models
SQLAlchemy ORM models untuk semua entitas sistem

Schema mengikuti CRISP-DM methodology:
- Users (Business Understanding context)
- Datasets (Data Understanding)
- DatasetColumns (Data Understanding - schema metadata)
- DataQualityReports (Data Preparation)
- Experiments (Modeling)
- MethodLogs (Evaluation - Method Monitor)
- Insights (Deployment - Nexa output)
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean, DateTime,
    ForeignKey, JSON, Enum as SQLEnum
)
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


class UserTier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class DatasetStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROFILING = "profiling"
    PROFILED = "profiled"
    READY = "ready"
    ERROR = "error"


class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelType(str, enum.Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"


# =============================================================================
# USER MODEL - Business Understanding context owner
# =============================================================================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))

    tier = Column(SQLEnum(UserTier), default=UserTier.FREE, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # Business Understanding context
    business_objective = Column(Text)  # User's stated objective
    use_case = Column(String(100))  # ecommerce, finance, marketing, research, custom

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    datasets = relationship("Dataset", back_populates="owner", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="owner", cascade="all, delete-orphan")


# =============================================================================
# DATASET MODELS - Data Understanding layer
# =============================================================================
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    name = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    storage_path = Column(String(500))  # File location
    file_format = Column(String(20))  # csv, xlsx, json, parquet
    file_size_bytes = Column(Integer)

    # Profile metadata (auto-detected from Data Understanding step)
    n_rows = Column(Integer)
    n_columns = Column(Integer)
    encoding = Column(String(50))  # utf-8, latin-1, etc

    # Auto-suggested target variable (Y) - Data Understanding output
    suggested_target = Column(String(255))
    target_confidence = Column(Float)
    target_reasoning = Column(Text)

    # Detected use case (helps Goal Setup later)
    detected_use_case = Column(String(100))

    status = Column(SQLEnum(DatasetStatus), default=DatasetStatus.UPLOADED)
    error_message = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="datasets")
    columns = relationship("DatasetColumn", back_populates="dataset", cascade="all, delete-orphan")
    quality_reports = relationship("DataQualityReport", back_populates="dataset", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="dataset", cascade="all, delete-orphan")


class DatasetColumn(Base):
    __tablename__ = "dataset_columns"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)

    name = Column(String(255), nullable=False)
    position = Column(Integer)  # column order
    inferred_type = Column(String(50))  # numerical, categorical, datetime, text, boolean
    pandas_dtype = Column(String(50))  # int64, float64, object, datetime64

    n_missing = Column(Integer, default=0)
    n_unique = Column(Integer)
    completeness_pct = Column(Float)  # (1 - missing/total) * 100

    # Statistical properties (for numerical)
    mean_value = Column(Float)
    std_value = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    median_value = Column(Float)

    # Top values (for categorical) - stored as JSON
    top_values = Column(JSON)  # [{value: "X", count: 100}, ...]

    is_potential_target = Column(Boolean, default=False)
    target_score = Column(Float)  # heuristic score for target detection

    dataset = relationship("Dataset", back_populates="columns")


# =============================================================================
# DATA QUALITY REPORT - Data Preparation layer
# =============================================================================
class DataQualityReport(Base):
    __tablename__ = "data_quality_reports"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)

    # Wang & Strong (1996) data quality dimensions
    completeness_score = Column(Float)  # % non-missing
    consistency_score = Column(Float)   # format consistency
    accuracy_score = Column(Float)      # validity of values
    validity_score = Column(Float)      # type matches schema
    uniqueness_score = Column(Float)    # duplicate detection
    timeliness_score = Column(Float)    # data freshness

    overall_score = Column(Float)  # weighted average

    # Issues detected (stored as JSON list)
    issues = Column(JSON)  # [{column, issue_type, severity, description, suggestion}]

    # Recommendations
    cleaning_recommendations = Column(JSON)  # [{step, action, reason}]

    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="quality_reports")


# =============================================================================
# EXPERIMENT MODEL - Modeling + Evaluation layer
# =============================================================================
class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)

    name = Column(String(255))
    objective = Column(Text)  # Business objective from user

    target_variable = Column(String(255))  # Y
    feature_variables = Column(JSON)  # X list

    model_type = Column(SQLEnum(ModelType))
    selected_algorithm = Column(String(100))  # RandomForest, IsolationForest, etc

    # Method Monitor data - WHY this model was chosen
    selection_reasoning = Column(Text)
    alternatives_considered = Column(JSON)  # [{algorithm, score, reason_rejected}]

    # Hyperparameters
    hyperparameters = Column(JSON)

    # Evaluation metrics
    metrics = Column(JSON)  # {rmse, mae, accuracy, precision, recall, f1, ...}
    confidence_score = Column(Float)

    # Model artifact
    model_path = Column(String(500))  # Saved model file location

    status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.PENDING)
    error_message = Column(Text)

    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="experiments")
    dataset = relationship("Dataset", back_populates="experiments")
    method_logs = relationship("MethodLog", back_populates="experiment", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="experiment", cascade="all, delete-orphan")


# =============================================================================
# METHOD MONITOR - The CRITICAL Explainable AI layer
# =============================================================================
class MethodLog(Base):
    """
    The Method Monitor logs every step of the AI pipeline with full reasoning.
    This is the core differentiator of Nexlytics - explainability at every stage.
    """
    __tablename__ = "method_logs"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)

    step = Column(String(100), nullable=False)  # profiling, target_detection, model_selection, training, evaluation
    step_order = Column(Integer)

    # The "why" - critical for Explainable AI
    selected_method = Column(String(255))
    why_chosen = Column(Text)  # Reasoning behind selection
    why_not_alternatives = Column(JSON)  # [{alternative, reason_rejected}]
    benefits = Column(JSON)  # List of benefits
    limitations = Column(JSON)  # List of known limitations

    # Process detail
    input_summary = Column(JSON)
    output_summary = Column(JSON)
    intermediate_results = Column(JSON)

    # Goal alignment
    goal_alignment = Column(Text)  # How this step serves the final goal

    duration_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="method_logs")


# =============================================================================
# INSIGHTS - Nexa AI Agent output (Deployment layer)
# =============================================================================
class Insight(Base):
    """
    Output dari Nexa AI Agent - structured insight untuk user.
    Format mengikuti spec:
    [DATA SUMMARY][DETECTED TARGET][SELECTED METHOD][WHY/WHY NOT]
    [MODEL RESULT][INSIGHT][RECOMMENDATION][CONFIDENCE]
    """
    __tablename__ = "insights"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)

    insight_type = Column(String(50))  # alert, recommendation, summary, anomaly, trend
    severity = Column(String(20))  # info, warning, critical

    title = Column(String(500))
    summary = Column(Text)

    # Structured Nexa output
    data_summary = Column(Text)
    detected_target = Column(String(255))
    selected_method = Column(String(255))
    why_this_method = Column(Text)
    why_not_others = Column(Text)
    model_result = Column(Text)
    insight_text = Column(Text)
    recommendation = Column(Text)
    confidence_score = Column(Float)

    # Action data (untuk realtime decision engine)
    action_required = Column(Boolean, default=False)
    action_type = Column(String(100))  # restock, reallocate, investigate, ignore

    # Realtime context
    triggered_by_event = Column(String(100))  # revenue_drop, expense_spike, anomaly_detected
    affected_kpi = Column(String(100))
    metric_change_pct = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    experiment = relationship("Experiment", back_populates="insights")


# =============================================================================
# REALTIME EVENT LOG - for streaming analytics
# =============================================================================
class RealtimeEvent(Base):
    __tablename__ = "realtime_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    event_type = Column(String(100))  # transaction, kpi_update, alert
    payload = Column(JSON)

    processed = Column(Boolean, default=False)
    insight_generated = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
