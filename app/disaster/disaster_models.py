"""
SQLAlchemy ORM models untuk modul Disaster Prediction.

9 tabel inti:
    1. disaster_projects          - project research aktif
    2. disaster_event_catalog     - katalog event historis (gempa, banjir, dll)
    3. disaster_risk_assessments  - hasil skoring risiko
    4. disaster_warnings          - log peringatan yang dihasilkan platform
    5. disaster_models            - registry model terlatih
    6. disaster_benchmarks        - hasil benchmark akurasi multi-model
    7. disaster_predictions       - log prediksi runtime
    8. disaster_explanations      - hasil SHAP explainer per prediksi
    9. disaster_audit_log         - audit trail untuk reproducibility
"""

from __future__ import annotations
from datetime import datetime

try:
    from sqlalchemy import (
        Column, Integer, String, Float, DateTime, Boolean,
        ForeignKey, Text, JSON,
    )
    from sqlalchemy.orm import declarative_base, relationship
    HAVE_SQLALCHEMY = True
    Base = declarative_base()
except Exception:
    HAVE_SQLALCHEMY = False
    Base = object  # type: ignore


if HAVE_SQLALCHEMY:

    class DisasterProject(Base):
        __tablename__ = "disaster_projects"
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, nullable=False, index=True)
        name = Column(String(200), nullable=False)
        region = Column(String(120), nullable=False)
        hazard_types = Column(JSON, nullable=False)  # list of 8 jenis bencana
        status = Column(String(40), default="active", nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
        updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


    class DisasterEventCatalog(Base):
        __tablename__ = "disaster_event_catalog"
        id = Column(Integer, primary_key=True)
        project_id = Column(Integer, ForeignKey("disaster_projects.id"), index=True)
        hazard_type = Column(String(40), nullable=False, index=True)
        event_time = Column(DateTime, nullable=False, index=True)
        latitude = Column(Float)
        longitude = Column(Float)
        magnitude = Column(Float)
        depth_km = Column(Float)
        intensity = Column(Float)
        rainfall_mm = Column(Float)
        impact_score = Column(Float)
        source = Column(String(80))
        raw_payload = Column(JSON)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


    class DisasterRiskAssessment(Base):
        __tablename__ = "disaster_risk_assessments"
        id = Column(Integer, primary_key=True)
        project_id = Column(Integer, ForeignKey("disaster_projects.id"), index=True)
        hazard_type = Column(String(40), nullable=False, index=True)
        composite_risk = Column(Float, nullable=False)
        hazard_component = Column(Float, nullable=False)
        exposure_component = Column(Float, nullable=False)
        vulnerability_component = Column(Float, nullable=False)
        model_probability = Column(Float, nullable=False)
        aggregation = Column(String(20), default="arithmetic")
        weights_json = Column(JSON)
        explanation = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


    class DisasterWarning(Base):
        __tablename__ = "disaster_warnings"
        id = Column(Integer, primary_key=True)
        project_id = Column(Integer, ForeignKey("disaster_projects.id"), index=True)
        risk_assessment_id = Column(Integer, ForeignKey("disaster_risk_assessments.id"))
        level = Column(String(20), nullable=False, index=True)  # NORMAL..CRITICAL
        risk_score = Column(Float, nullable=False)
        confidence = Column(Float, default=1.0)
        recommended_actions = Column(JSON)
        disclaimer = Column(Text, nullable=False)
        explanation = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


    class DisasterModel(Base):
        __tablename__ = "disaster_models"
        id = Column(Integer, primary_key=True)
        project_id = Column(Integer, ForeignKey("disaster_projects.id"), index=True)
        name = Column(String(120), nullable=False)
        algorithm = Column(String(60), nullable=False)  # lstm/xgboost/bayesian/ensemble
        task = Column(String(40), nullable=False)
        hyperparameters = Column(JSON)
        artifact_path = Column(String(400))
        is_active = Column(Boolean, default=True)
        trained_at = Column(DateTime, default=datetime.utcnow, nullable=False)


    class DisasterBenchmark(Base):
        __tablename__ = "disaster_benchmarks"
        id = Column(Integer, primary_key=True)
        project_id = Column(Integer, ForeignKey("disaster_projects.id"), index=True)
        model_id = Column(Integer, ForeignKey("disaster_models.id"))
        task = Column(String(40), nullable=False)
        metrics = Column(JSON, nullable=False)
        n_samples = Column(Integer)
        confusion = Column(JSON)
        notes = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


    class DisasterPrediction(Base):
        __tablename__ = "disaster_predictions"
        id = Column(Integer, primary_key=True)
        project_id = Column(Integer, ForeignKey("disaster_projects.id"), index=True)
        model_id = Column(Integer, ForeignKey("disaster_models.id"))
        hazard_type = Column(String(40), nullable=False)
        input_features = Column(JSON)
        probability = Column(Float, nullable=False)
        components = Column(JSON)
        confidence = Column(Float)
        latitude = Column(Float)
        longitude = Column(Float)
        target_time = Column(DateTime)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


    class DisasterExplanation(Base):
        __tablename__ = "disaster_explanations"
        id = Column(Integer, primary_key=True)
        prediction_id = Column(Integer, ForeignKey("disaster_predictions.id"), index=True)
        backend = Column(String(40), nullable=False)
        base_value = Column(Float)
        contributions = Column(JSON, nullable=False)
        ranked_contributors = Column(JSON, nullable=False)
        note = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


    class DisasterAuditLog(Base):
        __tablename__ = "disaster_audit_log"
        id = Column(Integer, primary_key=True)
        project_id = Column(Integer, ForeignKey("disaster_projects.id"), index=True)
        actor_user_id = Column(Integer)
        action = Column(String(80), nullable=False)
        entity_type = Column(String(60))
        entity_id = Column(Integer)
        payload = Column(JSON)
        ip_address = Column(String(60))
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
