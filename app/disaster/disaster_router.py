"""
DISASTER PREDICTION FASTAPI ROUTER
==================================

10 endpoint utama:
    POST /api/disaster/projects                  - buat project
    GET  /api/disaster/projects                  - list project user
    POST /api/disaster/risk/assess               - hitung composite risk
    POST /api/disaster/warning/classify          - klasifikasi level peringatan
    POST /api/disaster/hazard/analyze            - analisis 8 jenis bencana (router)
    POST /api/disaster/predict                   - inference ensemble model
    POST /api/disaster/explain                   - SHAP explanation per prediksi
    POST /api/disaster/benchmark                 - benchmark akurasi multi-model
    GET  /api/disaster/predictions/{project_id}  - list prediksi project
    GET  /api/disaster/audit/{project_id}        - audit trail

Semua endpoint mengembalikan disclaimer wajib.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field
    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False

from ml_engine.domains.disaster_prediction import (
    RiskScoreEngine, RiskComponents,
    WarningLevelClassifier, WARNING_DISCLAIMER,
    DisasterAccuracyBenchmark,
    EarthquakeAnalyzer, TsunamiAnalyzer, FloodAnalyzer, LandslideAnalyzer,
    WildfireAnalyzer, DroughtAnalyzer, RainfallAnalyzer, ClimateRiskAnalyzer,
    TemporalLSTMModel, GeospatialXGBoostModel, BayesianRiskModel,
    HybridEnsemblePipeline, SHAPExplainer,
)


if HAVE_FASTAPI:

    router = APIRouter(prefix="/api/disaster", tags=["disaster_prediction"])

    # ===== Schemas =====
    class CreateProjectRequest(BaseModel):
        name: str = Field(..., max_length=200)
        region: str = Field(..., max_length=120)
        hazard_types: List[str] = Field(..., min_items=1)


    class RiskAssessRequest(BaseModel):
        hazard: float = Field(..., ge=0.0, le=1.0)
        exposure: float = Field(..., ge=0.0, le=1.0)
        vulnerability: float = Field(..., ge=0.0, le=1.0)
        model_probability: float = Field(..., ge=0.0, le=1.0)
        weights: Optional[Dict[str, float]] = None
        aggregation: str = Field("arithmetic", pattern="^(arithmetic|geometric)$")


    class WarningClassifyRequest(BaseModel):
        risk_score: float = Field(..., ge=0.0, le=1.0)
        confidence: float = Field(1.0, ge=0.0, le=1.0)


    class HazardAnalyzeRequest(BaseModel):
        hazard_type: str
        params: Dict[str, Any]


    class PredictRequest(BaseModel):
        time_series: Optional[List[float]] = None
        spatial_features: Optional[List[List[float]]] = None
        bayesian_alpha: float = 1.0
        bayesian_beta: float = 1.0
        bayesian_updates: List[Dict[str, int]] = Field(default_factory=list)
        weights: Optional[Dict[str, float]] = None
        aggregation: str = "weighted_mean"


    class ExplainRequest(BaseModel):
        feature_names: List[str]
        instance: List[float]
        background: Optional[List[List[float]]] = None
        model_kind: str = "logistic_demo"  # demo predictor untuk fallback


    class BenchmarkRequest(BaseModel):
        task: str = Field(..., pattern="^(classification|regression)$")
        y_true: List[float]
        y_pred: List[float]
        y_proba: Optional[List[float]] = None
        model_name: str = "model"


    # ===== In-memory store sederhana (untuk demo / sandbox tanpa DB) =====
    _projects: Dict[int, Dict] = {}
    _predictions: Dict[int, List[Dict]] = {}
    _audit: Dict[int, List[Dict]] = {}
    _next_project_id = 1

    def _audit_log(project_id: int, action: str, payload: Dict) -> None:
        _audit.setdefault(project_id, []).append({
            "action": action, "payload": payload,
        })


    # ===== Endpoints =====
    @router.post("/projects")
    def create_project(req: CreateProjectRequest):
        global _next_project_id
        pid = _next_project_id
        _next_project_id += 1
        proj = {
            "id": pid,
            "name": req.name,
            "region": req.region,
            "hazard_types": req.hazard_types,
            "status": "active",
        }
        _projects[pid] = proj
        _audit_log(pid, "create_project", proj)
        return {"project": proj, "disclaimer": WARNING_DISCLAIMER}


    @router.get("/projects")
    def list_projects():
        return {
            "projects": list(_projects.values()),
            "disclaimer": WARNING_DISCLAIMER,
        }


    @router.post("/risk/assess")
    def assess_risk(req: RiskAssessRequest):
        engine = RiskScoreEngine(weights=req.weights, aggregation=req.aggregation)
        comp = RiskComponents(
            hazard=req.hazard, exposure=req.exposure,
            vulnerability=req.vulnerability,
            model_probability=req.model_probability,
        )
        result = engine.compute(comp)
        return {"assessment": result.to_dict(), "disclaimer": WARNING_DISCLAIMER}


    @router.post("/warning/classify")
    def classify_warning(req: WarningClassifyRequest):
        wl = WarningLevelClassifier()
        out = wl.classify(req.risk_score, req.confidence)
        return out.to_dict()


    @router.post("/hazard/analyze")
    def analyze_hazard(req: HazardAnalyzeRequest):
        ht = req.hazard_type.lower()
        params = req.params
        try:
            if ht == "earthquake":
                res = EarthquakeAnalyzer(params.get("region", "Unknown")).analyze(
                    params.get("catalog", []),
                    period_years=params.get("period_years", 1.0),
                )
            elif ht == "tsunami":
                res = TsunamiAnalyzer().analyze(
                    params["magnitude"], params["depth_km"],
                    params.get("is_undersea", True),
                    params.get("distance_to_coast_km"),
                    params.get("water_depth_m", 1000.0),
                )
            elif ht == "flood":
                res = FloodAnalyzer().analyze(
                    params["rainfall_mm"], params["soil_moisture"],
                    params["catchment_area_km2"], params.get("slope", 0.05),
                    params.get("impervious_fraction", 0.3),
                    params.get("duration_hours", 1.0),
                )
            elif ht == "landslide":
                res = LandslideAnalyzer().analyze(
                    params["slope_deg"], params.get("soil_cohesion_kpa", 5.0),
                    params.get("friction_angle_deg", 30.0),
                    params.get("soil_moisture", 0.4),
                    params.get("rainfall_24h_mm", 0.0),
                    params.get("depth_to_failure_m", 1.5),
                )
            elif ht == "wildfire":
                res = WildfireAnalyzer().analyze(
                    params["temperature_c"], params["relative_humidity_pct"],
                    params["wind_speed_kmh"],
                    params.get("rainfall_24h_mm", 0.0),
                    params.get("fuel_moisture_prev", 0.15),
                )
            elif ht == "drought":
                res = DroughtAnalyzer().analyze(
                    params["precipitation_history_mm"],
                    params.get("temperature_c_history", []),
                    params.get("timescale_months", 3),
                )
            elif ht == "rainfall":
                res = RainfallAnalyzer().analyze(
                    params["annual_max_rainfall_mm"],
                    params.get("current_event_mm"),
                )
            elif ht == "climate":
                res = ClimateRiskAnalyzer().analyze(
                    params["temperature_history_c"],
                    params.get("baseline_mean_c", 27.0),
                    params.get("sst_nino34_anomaly_c", 0.0),
                    params.get("years_per_record", 1.0),
                )
            else:
                raise HTTPException(400, f"hazard_type tidak dikenal: {req.hazard_type}")
        except KeyError as e:
            raise HTTPException(400, f"Parameter wajib hilang: {e}")
        out = res.__dict__ if hasattr(res, "__dict__") else dict(res)
        return {"hazard_type": ht, "result": out, "disclaimer": WARNING_DISCLAIMER}


    @router.post("/predict")
    def predict(req: PredictRequest):
        # Build minimal trained models (untuk demo). Pada produksi load dari registry.
        lstm = TemporalLSTMModel(input_size=1, horizon=3)
        xgb = GeospatialXGBoostModel(task="classification", feature_names=["a","b","c"])
        # train demo kecil bila spatial_features ada
        if req.spatial_features:
            d = len(req.spatial_features[0])
            X = [[(i % 2)] * d for i in range(20)]
            y = [i % 2 for i in range(20)]
            xgb.fit(X, y)

        bayes = BayesianRiskModel(req.bayesian_alpha, req.bayesian_beta)
        for upd in req.bayesian_updates:
            bayes.update(int(upd.get("n_events", 0)), int(upd.get("n_observations", 0)))

        ens = HybridEnsemblePipeline(
            lstm=lstm, xgboost=xgb, bayesian=bayes,
            weights=req.weights, aggregation=req.aggregation,
        )
        result = ens.predict(req.time_series, req.spatial_features)
        return {"prediction": result.to_dict(), "disclaimer": WARNING_DISCLAIMER}


    @router.post("/explain")
    def explain(req: ExplainRequest):
        # Demo predictor: weighted sum kemudian sigmoid
        import math
        def predict_fn(records):
            outs = []
            for r in records:
                z = sum((i + 1) * r[i] for i in range(len(r))) / max(1, len(r))
                outs.append(1.0 / (1.0 + math.exp(-z)))
            return outs
        explainer = SHAPExplainer(
            predict_fn, req.feature_names, req.background or None,
        )
        report = explainer.explain(req.instance)
        return {"explanation": report.to_dict(), "disclaimer": WARNING_DISCLAIMER}


    @router.post("/benchmark")
    def benchmark(req: BenchmarkRequest):
        bench = DisasterAccuracyBenchmark()
        if req.task == "classification":
            rep = bench.evaluate_classification(
                [int(v) for v in req.y_true],
                [int(v) for v in req.y_pred],
                req.y_proba,
                req.model_name,
            )
        else:
            rep = bench.evaluate_regression(req.y_true, req.y_pred, req.model_name)
        return {"report": rep.to_dict()}


    @router.get("/predictions/{project_id}")
    def get_predictions(project_id: int):
        return {
            "project_id": project_id,
            "predictions": _predictions.get(project_id, []),
            "disclaimer": WARNING_DISCLAIMER,
        }


    @router.get("/audit/{project_id}")
    def get_audit(project_id: int):
        return {
            "project_id": project_id,
            "audit_log": _audit.get(project_id, []),
        }


# =================================================================
# AI Agent endpoint - menerima parameter language sesuai i18n
# =================================================================
if HAVE_FASTAPI:

    from .ai_agent_request import AIAgentRequest, normalize_language

    class AIInsightRequest(BaseModel):
        task:       str = Field(..., max_length=80)
        data:       Dict[str, Any] = Field(default_factory=dict)
        language:   str = Field("en", max_length=10)
        project_id: Optional[int] = None
        max_tokens: int = 1024

    @router.post("/ai/insight")
    def generate_insight(req: AIInsightRequest):
        """
        Generate insight dari AI agent. Output bahasa mengikuti
        parameter `language` (default English).
        """
        normalized = normalize_language(req.language)
        agent_req = AIAgentRequest(
            task=req.task,
            data=req.data,
            language=normalized,
            project_id=req.project_id,
            max_tokens=req.max_tokens,
        )
        # Plug-in nyata akan memanggil LLM. Untuk sandbox kita stub.
        stub_output = {
            "en": "Risk is elevated due to recent rainfall and saturated soil.",
            "id": "Risiko meningkat karena hujan beberapa hari terakhir dan tanah jenuh air.",
            "ja": "最近の降雨と土壌の飽和によりリスクが上昇しています。",
            "zh": "由于近期降雨和土壤饱和, 风险升高。",
            "ar": "الخطر مرتفع بسبب الأمطار الأخيرة والتربة المشبعة بالماء.",
        }.get(normalized, "Risk is elevated due to recent rainfall and saturated soil.")
        return {
            "request": agent_req.to_dict(),
            "system_prompt_prefix": agent_req.to_system_prompt_prefix(),
            "response": {
                "task": req.task,
                "language": normalized,
                "output": stub_output,
                "model": "nexa-agent-v1",
            },
        }
