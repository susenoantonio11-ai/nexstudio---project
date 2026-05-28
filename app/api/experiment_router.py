"""Experiment endpoints: run AI pipeline, get method monitor, list experiments."""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import pandas as pd

from app.database.session import get_db
from app.database.models import (
    User, Dataset, Experiment, ExperimentStatus, ModelType,
    MethodLog, Insight
)
from app.schemas.dataset import ExperimentRequest, ExperimentOut, NexaInsight
from app.core.security import get_current_user

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ml_engine.pipeline import NexlyticsPipeline

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _run_experiment(experiment_id: int, db_factory):
    """Background task: run the actual ML pipeline."""
    db = db_factory()
    try:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not exp:
            return

        ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
        if not ds:
            exp.status = ExperimentStatus.FAILED
            exp.error_message = "Dataset not found"
            db.commit()
            return

        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.utcnow()
        db.commit()

        # Read dataset
        from app.api.dataset_router import _read_dataset_file
        df = _read_dataset_file(Path(ds.storage_path), ds.file_format)

        # Run pipeline
        pipeline = NexlyticsPipeline(goal=exp.objective or "Generate insight from dataset")
        result = pipeline.run(
            df=df,
            target_column=exp.target_variable,
            objective="auto",
            use_case=exp.dataset.detected_use_case if hasattr(exp.dataset, 'detected_use_case') else None,
        )

        if result["status"] != "success":
            exp.status = ExperimentStatus.FAILED
            exp.error_message = result.get("error", "Unknown error")
            exp.completed_at = datetime.utcnow()
            db.commit()
            return

        # Save results
        sel = result["model_selection"]
        mr = result["model_result"]
        exp.target_variable = result["target_detection"]["suggested_target"]
        exp.selected_algorithm = sel["selected_model"]
        exp.model_type = sel["task"]
        exp.selection_reasoning = sel["reasoning"]
        exp.alternatives_considered = sel["alternatives_considered"]
        exp.hyperparameters = sel.get("hyperparameters", {})
        exp.metrics = mr.get("metrics", {})
        exp.confidence_score = mr.get("confidence_score", 0)
        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.utcnow()
        if exp.started_at:
            exp.duration_seconds = (exp.completed_at - exp.started_at).total_seconds()

        # Save method monitor logs
        for log in result["method_monitor"]["steps"]:
            db.add(MethodLog(
                experiment_id=exp.id,
                step=log["step"],
                step_order=log["step_order"],
                selected_method=log["selected_method"],
                why_chosen=log["why_chosen"],
                why_not_alternatives=log.get("why_not_alternatives", []),
                benefits=log.get("benefits", []),
                limitations=log.get("limitations", []),
                input_summary=log.get("input_summary", {}),
                output_summary=log.get("output_summary", {}),
                intermediate_results=log.get("intermediate_results", {}),
                goal_alignment=log.get("goal_alignment"),
                duration_ms=log.get("duration_ms", 0),
            ))

        # Save insight (Nexa structured output)
        nexa = result.get("nexa_output", {})
        db.add(Insight(
            experiment_id=exp.id,
            insight_type="summary",
            severity="info",
            title=f"AI Analysis: {exp.target_variable}",
            summary=result.get("insight", ""),
            data_summary=nexa.get("DATA_SUMMARY", ""),
            detected_target=nexa.get("DETECTED_TARGET_VARIABLE", ""),
            selected_method=nexa.get("SELECTED_METHOD", ""),
            why_this_method=nexa.get("WHY_THIS_METHOD", ""),
            why_not_others=nexa.get("WHY_NOT_OTHER_METHODS", ""),
            model_result=nexa.get("MODEL_RESULT", ""),
            insight_text=nexa.get("INSIGHT", ""),
            recommendation=nexa.get("RECOMMENDATION", ""),
            confidence_score=float(nexa.get("CONFIDENCE_SCORE", "0").replace("%", "") or 0),
        ))

        db.commit()
    except Exception as e:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            exp.status = ExperimentStatus.FAILED
            exp.error_message = str(e)
            exp.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.post("/run", response_model=ExperimentOut, status_code=status.HTTP_202_ACCEPTED)
def run_experiment(
    req: ExperimentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Start a new experiment. Returns immediately with status='pending',
    pipeline runs in background. Poll /api/experiments/{id} for status.
    """
    ds = db.query(Dataset).filter(
        Dataset.id == req.dataset_id,
        Dataset.user_id == current_user.id
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    exp = Experiment(
        user_id=current_user.id,
        dataset_id=ds.id,
        name=req.name or f"Experiment on {ds.name}",
        objective=req.objective,
        target_variable=req.target_column or ds.suggested_target,
        status=ExperimentStatus.PENDING,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)

    # Run in background
    from app.database.session import SessionLocal
    background_tasks.add_task(_run_experiment, exp.id, SessionLocal)

    return ExperimentOut.model_validate(exp)


@router.get("/", response_model=List[ExperimentOut])
def list_experiments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all experiments by current user."""
    exps = db.query(Experiment).filter(
        Experiment.user_id == current_user.id
    ).order_by(Experiment.created_at.desc()).all()
    return [ExperimentOut.model_validate(e) for e in exps]


@router.get("/{experiment_id}", response_model=ExperimentOut)
def get_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    exp = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == current_user.id
    ).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return ExperimentOut.model_validate(exp)


@router.get("/{experiment_id}/method-monitor")
def get_method_monitor(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get full Method Monitor trace for an experiment.
    This is the CRITICAL Explainable AI feature of Nexlytics.
    """
    exp = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == current_user.id
    ).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    logs = db.query(MethodLog).filter(
        MethodLog.experiment_id == experiment_id
    ).order_by(MethodLog.step_order).all()

    return {
        "experiment_id": exp.id,
        "status": exp.status,
        "goal": exp.objective,
        "selected_algorithm": exp.selected_algorithm,
        "model_type": exp.model_type,
        "selection_reasoning": exp.selection_reasoning,
        "alternatives_considered": exp.alternatives_considered,
        "metrics": exp.metrics,
        "confidence_score": exp.confidence_score,
        "n_steps": len(logs),
        "steps": [
            {
                "step": log.step,
                "step_order": log.step_order,
                "selected_method": log.selected_method,
                "why_chosen": log.why_chosen,
                "why_not_alternatives": log.why_not_alternatives,
                "benefits": log.benefits,
                "limitations": log.limitations,
                "input_summary": log.input_summary,
                "output_summary": log.output_summary,
                "goal_alignment": log.goal_alignment,
                "duration_ms": log.duration_ms,
            }
            for log in logs
        ],
    }


@router.get("/{experiment_id}/insights")
def get_insights(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get Nexa AI Agent structured insights for an experiment."""
    exp = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == current_user.id
    ).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    insights = db.query(Insight).filter(
        Insight.experiment_id == experiment_id
    ).order_by(Insight.created_at.desc()).all()

    return [
        {
            "id": i.id,
            "type": i.insight_type,
            "severity": i.severity,
            "title": i.title,
            "summary": i.summary,
            "nexa_format": {
                "DATA_SUMMARY": i.data_summary,
                "DETECTED_TARGET_VARIABLE": i.detected_target,
                "SELECTED_METHOD": i.selected_method,
                "WHY_THIS_METHOD": i.why_this_method,
                "WHY_NOT_OTHER_METHODS": i.why_not_others,
                "MODEL_RESULT": i.model_result,
                "INSIGHT": i.insight_text,
                "RECOMMENDATION": i.recommendation,
                "CONFIDENCE_SCORE": str(i.confidence_score or 0),
            },
            "created_at": i.created_at.isoformat() if i.created_at else None,
        }
        for i in insights
    ]
