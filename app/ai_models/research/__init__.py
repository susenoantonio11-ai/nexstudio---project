"""
Research Pipeline — Multi-Province Flood Prediction Indonesia
=============================================================
End-to-end implementation of the thesis pipeline:

    "Multi-Province Flood Prediction in Indonesia
     Using a Hybrid LSTM-XGBoost Model with SHAP-Based Interpretability
     from Google Earth Engine and BNPB Data 2016-2025"

Components:
    indonesia_provinces.py     38-province registry (Permendagri 137/2017 + 2022 DOB)
    panel_builder.py           Province × time panel data joiner + lag features
    hybrid_lstm_xgb.py         Hybrid LSTM + XGBoost trainer (torch-free fallback)
    shap_explainer.py          SHAP TreeExplainer + permutation fallback
    research_orchestrator.py   End-to-end pipeline: ingest → panel → train → SHAP → save
    gee_service.py             Google Earth Engine stub (real client when ee+creds present)
    bnpb_service.py            BNPB DIBI CSV ingestion (data.bnpb.go.id)
"""
from .indonesia_provinces import INDONESIA_PROVINCES, list_province_ids
from .panel_builder import FloodPanelBuilder
from .hybrid_lstm_xgb import HybridLSTMXGBoost
from .shap_explainer import HybridSHAPExplainer
from .research_orchestrator import FloodResearchOrchestrator

__all__ = [
    "INDONESIA_PROVINCES",
    "list_province_ids",
    "FloodPanelBuilder",
    "HybridLSTMXGBoost",
    "HybridSHAPExplainer",
    "FloodResearchOrchestrator",
]
