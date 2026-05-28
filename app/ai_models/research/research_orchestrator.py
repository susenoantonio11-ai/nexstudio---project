"""
FloodResearchOrchestrator
=========================
End-to-end pipeline that mirrors the thesis abstract one-to-one:

  1. INGEST       — pull GEE features (CHIRPS / MODIS / GLDAS / SRTM / JRC /
                    WorldCover) per province per day  (gee_service.pull_panel).
  2. INGEST       — load BNPB flood event records       (bnpb_service.load_*).
  3. PANEL        — join + lag features + temporal split (FloodPanelBuilder).
  4. TRAIN        — Hybrid LSTM-XGBoost soft voting    (HybridLSTMXGBoost).
  5. EVALUATE     — Train / Val / Test metrics + confusion matrix +
                    individual branch AUCs.
  6. EXPLAIN      — SHAP TreeExplainer + permutation   (HybridSHAPExplainer).
  7. EMIT         — JSON envelope for the frontend Research Lab page +
                    Method Monitor lineage entries.

The orchestrator returns ONE big envelope so the frontend (or a notebook)
can persist the run and replay it at thesis defense. Every step writes a
Method Monitor entry tagged `page='research-lab'` so the timeline drawer
already shows the full reproducible audit trail.

ASSUMPTIONS
  * Province ids follow indonesia_provinces.py.
  * GEE service is in stub mode unless creds present — pipeline still runs
    end-to-end.
  * BNPB ingestion is either a real CSV download (preferred — load_csv)
    or synthetic-demo (synthesize_demo_events) for first-pass review.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .panel_builder import FloodPanelBuilder
from .hybrid_lstm_xgb import HybridLSTMXGBoost
from .shap_explainer import HybridSHAPExplainer
from . import gee_service, bnpb_service
from .indonesia_provinces import list_province_ids


class FloodResearchOrchestrator:
    name = "FloodResearchOrchestrator"
    domain = "research_pipeline"

    def __init__(
        self,
        provinces: Optional[List[str]] = None,
        start: str = "2016-01-01",
        end: str = "2025-12-31",
        prediction_horizon_days: int = 7,
        seq_window: int = 30,
        lstm_weight: float = 0.5,
        xgb_weight: float = 0.5,
    ) -> None:
        self.provinces = provinces or list_province_ids()
        self.start = start
        self.end = end
        self.prediction_horizon_days = prediction_horizon_days
        self.seq_window = seq_window
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight

    # ------------------------------------------------------------------
    def run(self, bnpb_csv_path: Optional[str] = None,
            bnpb_folder: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        log: List[Dict[str, Any]] = []

        # 1. GEE ingest
        gee = gee_service.pull_panel(self.provinces, self.start, self.end)
        if gee["status"] != "success":
            return {"status": "error", "stage": "gee_ingest", "details": gee}
        log.append({"stage": "gee_ingest", "mode": gee["mode"], "rows": gee["rows"]})

        # 2. BNPB ingest
        if bnpb_csv_path:
            bnpb = bnpb_service.load_csv(bnpb_csv_path)
        elif bnpb_folder:
            bnpb = bnpb_service.load_directory(bnpb_folder)
        else:
            bnpb = bnpb_service.synthesize_demo_events(self.provinces, self.start, self.end)
        if bnpb["status"] != "success":
            return {"status": "error", "stage": "bnpb_ingest", "details": bnpb}
        log.append({"stage": "bnpb_ingest",
                    "rows": bnpb["rows"],
                    "mode": bnpb.get("mode", "csv")})

        # 3. Panel build
        builder = FloodPanelBuilder(prediction_horizon_days=self.prediction_horizon_days)
        panel_envelope = builder.build(gee["data"], bnpb["data"])
        if panel_envelope["status"] != "success":
            return {"status": "error", "stage": "panel_build", "details": panel_envelope}
        log.append({"stage": "panel_build", "stats": panel_envelope["stats"]})

        train = panel_envelope["splits"]["train"]
        val = panel_envelope["splits"]["val"]
        test = panel_envelope["splits"]["test"]

        # 4. Train hybrid model
        model = HybridLSTMXGBoost(
            seq_window=self.seq_window,
            lstm_weight=self.lstm_weight,
            xgb_weight=self.xgb_weight,
        )
        fit_info = model.fit(train)
        if fit_info["status"] != "success":
            return {"status": "error", "stage": "train", "details": fit_info}
        log.append({"stage": "train",
                    "lstm_backend": fit_info["lstm_backend"],
                    "xgb_backend": fit_info["xgb_backend"],
                    "n_train": fit_info["n_train"]})

        # 5. Evaluate
        try:
            train_metrics = model.evaluate(train)
        except Exception as e:
            train_metrics = {"error": f"train evaluate failed: {e}"}
        try:
            val_metrics = model.evaluate(val) if len(val) > self.seq_window * 2 else {"skipped": "val too small"}
        except Exception as e:
            val_metrics = {"error": f"val evaluate failed: {e}"}
        try:
            test_metrics = model.evaluate(test) if len(test) > self.seq_window * 2 else {"skipped": "test too small"}
        except Exception as e:
            test_metrics = {"error": f"test evaluate failed: {e}"}
        log.append({"stage": "evaluate", "train": train_metrics, "val": val_metrics, "test": test_metrics})

        # 6. SHAP / permutation explain on TEST set
        explainer = HybridSHAPExplainer()
        try:
            explain_envelope = explainer.explain(model, test if len(test) > self.seq_window * 2 else val)
        except Exception as e:
            explain_envelope = {"status": "error", "message": f"SHAP failed: {e}"}
        log.append({"stage": "explain",
                    "backend": explain_envelope.get("shap_backend", "unknown"),
                    "n_features": len(explain_envelope.get("global_importance_ranked") or [])})

        duration_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "success",
            "model_name": self.name,
            "config": {
                "provinces_count": len(self.provinces),
                "start": self.start, "end": self.end,
                "prediction_horizon_days": self.prediction_horizon_days,
                "seq_window": self.seq_window,
                "fusion_weights": {"lstm": self.lstm_weight, "xgb": self.xgb_weight},
            },
            "panel_stats": panel_envelope["stats"],
            "fit": fit_info,
            "metrics": {
                "train": train_metrics, "val": val_metrics, "test": test_metrics,
            },
            "explainability": explain_envelope,
            "stage_log": log,
            "duration_ms": duration_ms,
            "method_monitor": {
                "method": ("Pipeline: GEE pull → BNPB ingest → panel build → "
                           "Hybrid LSTM-XGBoost (soft voting) → SHAP explanation"),
                "why_used": "End-to-end reproducible workflow that maps 1:1 to the thesis abstract.",
                "formulas": [
                    f"flood_label_t = 1[∃ BNPB event in (t, t+{self.prediction_horizon_days}d]]",
                    "API_t = 0.85·API_{t-1} + rainfall_t  (Kohler & Linsley, 1951)",
                    f"final_p = {self.lstm_weight}·p_LSTM + {self.xgb_weight}·p_XGB  (soft voting)",
                ],
                "limitations": [
                    "Stub mode for GEE / BNPB synthesises plausible data — final paper figures must use real downloads.",
                    "Province-level granularity; for kabupaten resolution rerun with kabupaten geometry.",
                    "Imbalanced labels — consider focal loss or SMOTE in a follow-up.",
                ],
                "citations": [
                    "Hochreiter & Schmidhuber (1997) Neural Computation 9(8).",
                    "Chen & Guestrin (2016) KDD '16 — XGBoost.",
                    "Lundberg & Lee (2017) NeurIPS 30 — SHAP.",
                    "Funk et al. (2015) Sci. Data 2:150066 — CHIRPS.",
                    "Tehrany et al. (2014) J. Hydrol. 512:332-343.",
                    "Bergmeir & Benítez (2012) Inf. Sci. 191 — temporal CV.",
                    "Gorelick et al. (2017) Remote Sens. Environ. 202 — GEE.",
                ],
            },
        }
