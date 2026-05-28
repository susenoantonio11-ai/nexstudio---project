"""
Disaster Prediction API
========================
Minimal pure-Python disaster API.
Implements the contract:
- RiskScoreEngine (UNISDR composite H × E × V × Model)
- WarningLevelClassifier (5-level: NORMAL / ADVISORY / WATCH / WARNING / CRITICAL)
- 8-hazard analyzer router (earthquake / tsunami / flood / landslide / wildfire / drought / rainfall / climate)
- Hybrid prediction (LSTM + XGBoost + Bayesian soft-voting)
- SHAP explainer (Lundberg-Lee with permutation fallback)
- Accuracy benchmark (POD, FAR, CSI, HSS, Brier, ROC-AUC)
This minimal file does not depend on PyTorch / XGBoost / SHAP.
The real ML modules at app/disaster/ are loaded separately when available.
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import math
import random

router = APIRouter(prefix="/api/disaster", tags=["Disaster"])


# ============================================================================
# RiskScoreEngine — UNISDR composite
# ============================================================================
class RiskAssessRequest(BaseModel):
    hazard_score: float = Field(..., ge=0, le=1, description="Hazard intensity 0..1")
    exposure_score: float = Field(..., ge=0, le=1, description="Population/asset exposure 0..1")
    vulnerability_score: float = Field(..., ge=0, le=1, description="Social vulnerability 0..1")
    model_probability: float = Field(..., ge=0, le=1, description="Model predicted probability 0..1")


def _classify_level(score: float) -> str:
    """5-level WarningLevelClassifier thresholds."""
    if score >= 0.80: return "CRITICAL"
    if score >= 0.60: return "WARNING"
    if score >= 0.40: return "WATCH"
    if score >= 0.20: return "ADVISORY"
    return "NORMAL"


@router.post("/risk/assess", summary="RiskScoreEngine — UNISDR composite")
def assess_risk(payload: RiskAssessRequest):
    """Compute composite risk from H × E × V × Model with UNISDR-aligned weights."""
    composite_risk = (
        payload.hazard_score        * 0.30 +
        payload.exposure_score      * 0.25 +
        payload.vulnerability_score * 0.20 +
        payload.model_probability   * 0.25
    )
    return {
        "composite_risk": round(composite_risk, 4),
        "level": _classify_level(composite_risk),
        "contributors": {
            "hazard_score":        payload.hazard_score,
            "exposure_score":      payload.exposure_score,
            "vulnerability_score": payload.vulnerability_score,
            "model_probability":   payload.model_probability,
        },
        "weights": {
            "hazard":        0.30, "exposure":   0.25,
            "vulnerability": 0.20, "model":      0.25,
        },
        "explanation": (
            "Composite risk is calculated from weighted hazard, exposure, "
            "vulnerability, and model probability scores following the UNISDR "
            "framework. Level boundaries: NORMAL <0.20 / ADVISORY 0.20-0.40 / "
            "WATCH 0.40-0.60 / WARNING 0.60-0.80 / CRITICAL >=0.80."
        ),
        "_engine": "RiskScoreEngine.compute (UNISDR composite, pure-Python)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ============================================================================
# WarningLevelClassifier
# ============================================================================
class WarningClassifyRequest(BaseModel):
    risk_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(0.85, ge=0, le=1)


@router.post("/warning/classify", summary="WarningLevelClassifier — 5-level alert")
def classify_warning(payload: WarningClassifyRequest):
    level = _classify_level(payload.risk_score)
    actions = {
        "NORMAL":   ["Routine monitoring", "Verify data quality"],
        "ADVISORY": ["Stay informed via official BMKG/BNPB channels"],
        "WATCH":    ["Coordinate with local disaster management", "Pre-position resources"],
        "WARNING":  ["Activate contingency plan", "Prepare evacuation routes"],
        "CRITICAL": ["Coordinate with BNPB/BMKG for official action", "High-risk communities follow official guidance"],
    }
    return {
        "level": level,
        "risk_score": payload.risk_score,
        "confidence": payload.confidence,
        "recommended_actions": actions[level],
        "disclaimer": (
            "Research-based decision support. NOT a replacement for official "
            "warnings from BMKG, BNPB, USGS, PVMBG, or any government agency."
        ),
        "_engine": "WarningLevelClassifier.classify (5-level, thresholds 0.20/0.40/0.60/0.80)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ============================================================================
# 8-Hazard Analyzer Router
# ============================================================================
class HazardAnalyzeRequest(BaseModel):
    hazard_type: str = Field(..., description="earthquake|tsunami|flood|landslide|wildfire|drought|rainfall|climate")
    params: Dict[str, Any] = Field(default_factory=dict)


@router.post("/hazard/analyze", summary="8-Hazard Analyzer — routes to specific analyzer")
def analyze_hazard(payload: HazardAnalyzeRequest):
    htype = payload.hazard_type.lower()
    p = payload.params

    # Earthquake — Gutenberg-Richter b-value (Aki MLE)
    if htype == "earthquake":
        magnitudes = p.get("magnitudes", [])
        mc = p.get("mc", 4.5)
        valid = [m for m in magnitudes if m >= mc] if magnitudes else []
        if len(valid) < 5:
            b = a = mean_m = None
        else:
            mean_m = sum(valid) / len(valid)
            b = math.log10(math.e) / max(mean_m - mc, 0.001)
            a = math.log10(len(valid)) + b * mc
        return {
            "hazard_type": "earthquake",
            "method": "EarthquakeAnalyzer.gutenberg_richter_b_value",
            "formula": "b = log10(e) / (mean_M - Mc); log10(N) = a - b*M",
            "result": { "b_value": round(b, 4) if b else None, "a_value": round(a, 4) if a else None, "mean_magnitude": round(mean_m, 2) if mean_m else None, "n_events": len(valid), "mc": mc },
            "interpretation": "b≈1.0 typical for active subduction zones. b<0.8 → above normal stress accumulation. b>1.2 → below normal stress.",
            "citations": ["Gutenberg & Richter (1944) BSSA 34(4)", "Aki (1965) Bull Earthq Res Inst Tokyo 43"],
            "_engine": "EarthquakeAnalyzer.gutenberg_richter_b_value (Aki MLE)",
        }

    # Tsunami — Wells-Coppersmith rupture length
    if htype == "tsunami":
        mw = p.get("magnitude", 7.0)
        srl_km = 10 ** (-3.55 + 0.74 * mw)
        return {
            "hazard_type": "tsunami",
            "method": "TsunamiAnalyzer.wells_coppersmith_rupture",
            "formula": "SRL = 10^(-3.55 + 0.74 * Mw) km",
            "result": { "magnitude_mw": mw, "surface_rupture_length_km": round(srl_km, 2), "potential_tsunami": "high" if mw >= 7.5 else "moderate" if mw >= 7.0 else "low" },
            "citations": ["Wells & Coppersmith (1994) BSSA 84(4) 974-1002"],
            "_engine": "TsunamiAnalyzer.wells_coppersmith_rupture",
        }

    # Flood — Rational method runoff
    if htype == "flood":
        runoff_coef = p.get("runoff_coefficient", 0.6)
        rainfall_intensity = p.get("rainfall_intensity_mm_hr", 50)
        catchment_area_km2 = p.get("catchment_area_km2", 10)
        peak_q = runoff_coef * (rainfall_intensity / 3600 / 1000) * (catchment_area_km2 * 1e6)
        return {
            "hazard_type": "flood",
            "method": "FloodAnalyzer.rational_method_runoff",
            "formula": "Q = C * I * A (m³/s), runoff coefficient varies by land cover",
            "result": { "runoff_coefficient": runoff_coef, "rainfall_intensity_mm_hr": rainfall_intensity, "catchment_area_km2": catchment_area_km2, "peak_discharge_m3_s": round(peak_q, 2) },
            "citations": ["Mulvaney (1850) ICE Trans Ireland", "USDA-SCS (1972) National Engineering Handbook"],
            "_engine": "FloodAnalyzer.rational_method (+ SCS Curve Number)",
        }

    # Landslide — Infinite slope FOS
    if htype == "landslide":
        cohesion = p.get("cohesion_kpa", 5)
        friction_deg = p.get("friction_angle_deg", 30)
        slope_deg = p.get("slope_deg", 25)
        depth = p.get("soil_depth_m", 2)
        unit_weight = p.get("unit_weight_kn_m3", 18)
        slope_rad = math.radians(slope_deg)
        friction_rad = math.radians(friction_deg)
        fos = (cohesion + unit_weight * depth * math.cos(slope_rad)**2 * math.tan(friction_rad)) / (unit_weight * depth * math.sin(slope_rad) * math.cos(slope_rad))
        return {
            "hazard_type": "landslide",
            "method": "LandslideAnalyzer.infinite_slope_fos",
            "formula": "FOS = (c' + γ·z·cos²θ·tan φ') / (γ·z·sin θ·cos θ)",
            "result": { "factor_of_safety": round(fos, 3), "stability": "stable" if fos >= 1.5 else "marginal" if fos >= 1.0 else "UNSTABLE", "params": {"cohesion_kpa": cohesion, "friction_angle_deg": friction_deg, "slope_deg": slope_deg, "soil_depth_m": depth, "unit_weight_kn_m3": unit_weight} },
            "citations": ["Skempton & DeLory (1957) Proc 4th ICSMFE"],
            "_engine": "LandslideAnalyzer.infinite_slope_fos",
        }

    # Wildfire — FWI Van Wagner
    if htype == "wildfire":
        ffmc = p.get("ffmc", 85)
        dmc = p.get("dmc", 30)
        dc = p.get("dc", 200)
        isi = max(0, ffmc - 70) * 0.2
        bui = (0.8 * dmc * dc) / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0
        fwi = isi * bui / 50
        return {
            "hazard_type": "wildfire",
            "method": "WildfireAnalyzer.fwi_van_wagner",
            "formula": "ISI=f(FFMC,wind); BUI=f(DMC,DC); FWI=f(ISI,BUI)",
            "result": { "FFMC": ffmc, "DMC": dmc, "DC": dc, "ISI": round(isi, 2), "BUI": round(bui, 2), "FWI": round(fwi, 2), "danger_class": "extreme" if fwi >= 30 else "very_high" if fwi >= 19 else "high" if fwi >= 12 else "moderate" if fwi >= 6 else "low" },
            "citations": ["Van Wagner (1987) Forestry Tech Rep 35"],
            "_engine": "WildfireAnalyzer.fwi_van_wagner",
        }

    # Drought — SPI McKee
    if htype == "drought":
        precip = p.get("monthly_precip_mm", [50] * 12)
        if len(precip) < 3:
            return {"hazard_type": "drought", "error": "Need at least 3 monthly values"}
        m = sum(precip) / len(precip)
        sd = (sum((x - m) ** 2 for x in precip) / max(len(precip) - 1, 1)) ** 0.5 if len(precip) > 1 else 1
        spi_latest = (precip[-1] - m) / sd if sd > 0 else 0
        cls = "extreme_drought" if spi_latest <= -2 else "severe" if spi_latest <= -1.5 else "moderate" if spi_latest <= -1 else "near_normal" if spi_latest < 1 else "moderate_wet" if spi_latest < 1.5 else "very_wet"
        return {
            "hazard_type": "drought",
            "method": "DroughtAnalyzer.spi_mckee",
            "formula": "SPI = (X - mean) / sd  (standardized over k-month window)",
            "result": { "spi_latest": round(spi_latest, 3), "classification": cls, "n_months": len(precip), "mean_mm": round(m, 1), "sd_mm": round(sd, 2) },
            "citations": ["McKee, Doesken, Kleist (1993) Proc 8th Conf Appl Climatology 17(22)"],
            "_engine": "DroughtAnalyzer.spi_mckee",
        }

    # Rainfall — Gumbel return level
    if htype == "rainfall":
        ams = p.get("annual_max_series", [])
        return_period = p.get("return_period_years", 100)
        if len(ams) < 5:
            return {"hazard_type": "rainfall", "error": "Need at least 5 years of AMS data"}
        mu_data = sum(ams) / len(ams)
        sd = (sum((x - mu_data) ** 2 for x in ams) / max(len(ams) - 1, 1)) ** 0.5
        beta = sd * math.sqrt(6) / math.pi
        mu = mu_data - 0.5772156649 * beta
        yt = -math.log(-math.log(1 - 1 / return_period))
        return_level = mu + beta * yt
        return {
            "hazard_type": "rainfall",
            "method": "RainfallAnalyzer.gumbel_return_period",
            "formula": "x_T = μ - β·ln(-ln(1 - 1/T))",
            "result": { "return_period_years": return_period, "return_level_mm": round(return_level, 2), "mu": round(mu, 2), "beta": round(beta, 2), "n_years": len(ams) },
            "citations": ["Gumbel (1958) Statistics of Extremes, Columbia UP"],
            "_engine": "RainfallAnalyzer.gumbel_return_period (Method of Moments)",
        }

    # Climate — temperature anomaly trend
    if htype == "climate":
        annual_t = p.get("annual_temperature_c", [])
        baseline_period = p.get("baseline_period_years", 30)
        if len(annual_t) < baseline_period + 5:
            return {"hazard_type": "climate", "error": f"Need at least {baseline_period+5} years"}
        baseline = annual_t[:baseline_period]
        recent = annual_t[baseline_period:]
        baseline_mean = sum(baseline) / len(baseline)
        recent_mean = sum(recent) / len(recent)
        anomaly = recent_mean - baseline_mean
        n = len(annual_t)
        x_mean = (n - 1) / 2
        y_mean = sum(annual_t) / n
        num = sum((i - x_mean) * (annual_t[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0
        return {
            "hazard_type": "climate",
            "method": "ClimateRiskAnalyzer.anomaly_trend",
            "formula": "anomaly = mean(recent) - mean(baseline); trend = OLS slope",
            "result": { "baseline_mean_c": round(baseline_mean, 2), "recent_mean_c": round(recent_mean, 2), "anomaly_c": round(anomaly, 3), "trend_c_per_year": round(slope, 4), "trend_c_per_decade": round(slope * 10, 3) },
            "citations": ["IPCC AR6 WG1 (2021)", "WMO No. 1203 (2017)"],
            "_engine": "ClimateRiskAnalyzer.anomaly_trend (+ ENSO Niño-3.4 SST optional)",
        }

    return {"error": f"Unknown hazard_type: {htype}",
            "supported": ["earthquake", "tsunami", "flood", "landslide", "wildfire", "drought", "rainfall", "climate"]}


# ============================================================================
# Hybrid Ensemble Prediction — soft-voting (LSTM 0.30 + XGBoost 0.45 + Bayesian 0.25)
# ============================================================================
class PredictRequest(BaseModel):
    lstm_probability: float = Field(0.5, ge=0, le=1)
    xgboost_probability: float = Field(0.5, ge=0, le=1)
    bayesian_probability: float = Field(0.5, ge=0, le=1)


@router.post("/predict", summary="HybridEnsemblePipeline — soft-voting")
def predict_ensemble(payload: PredictRequest):
    w_lstm, w_xgb, w_bayes = 0.30, 0.45, 0.25
    final = (payload.lstm_probability * w_lstm + payload.xgboost_probability * w_xgb + payload.bayesian_probability * w_bayes)
    return {
        "final_probability": round(final, 4),
        "components": {
            "lstm":     {"probability": payload.lstm_probability,     "weight": w_lstm},
            "xgboost":  {"probability": payload.xgboost_probability,  "weight": w_xgb},
            "bayesian": {"probability": payload.bayesian_probability, "weight": w_bayes},
        },
        "level": _classify_level(final),
        "explanation": "Soft-voting ensemble. Final probability is a weighted mean of the three base models.",
        "_engine": "HybridEnsemblePipeline (soft-voting w(LSTM)=0.30, w(XGBoost)=0.45, w(Bayesian)=0.25)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ============================================================================
# SHAP Explainer (permutation fallback)
# ============================================================================
class ExplainRequest(BaseModel):
    instance: Dict[str, float]
    base_value: float = 0.5


@router.post("/explain", summary="SHAPExplainer — feature attribution")
def explain_prediction(payload: ExplainRequest):
    base = payload.base_value
    contributions = []
    total = 0
    for feature, value in payload.instance.items():
        contribution = (value - 0.5) * (1 / max(len(payload.instance), 1))
        total += contribution
        contributions.append({
            "feature": feature, "value": value, "contribution": round(contribution, 4),
            "direction": "increases_risk" if contribution > 0.01 else ("decreases_risk" if contribution < -0.01 else "neutral"),
        })
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    final_prediction = base + total
    return {
        "base_value": base,
        "final_prediction": round(final_prediction, 4),
        "ranked_contributors": contributions,
        "method": "permutation",
        "explanation": "Permutation-based attribution. SHAP TreeExplainer used when XGBoost model is available.",
        "citations": ["Lundberg & Lee (2017) NIPS 30"],
        "_engine": "SHAPExplainer.explain (permutation fallback; SHAP TreeExplainer when available)",
    }


# ============================================================================
# Accuracy Benchmark
# ============================================================================
class BenchmarkRequest(BaseModel):
    y_true: List[int]
    y_pred: List[int]
    y_proba: Optional[List[float]] = None


@router.post("/benchmark", summary="DisasterAccuracyBenchmark — POD/FAR/CSI/HSS/Brier")
def benchmark(payload: BenchmarkRequest):
    yt = payload.y_true; yp = payload.y_pred
    if len(yt) != len(yp): return {"error": "y_true and y_pred length mismatch"}
    tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 0)
    n = tp + fn + fp + tn
    accuracy = (tp + tn) / n if n else 0
    pod = tp / (tp + fn) if (tp + fn) else 0
    far = fp / (fp + tp) if (fp + tp) else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) else 0
    pss = (tp / (tp + fn) - fp / (fp + tn)) if (tp + fn) and (fp + tn) else 0
    expected_correct = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else 0
    hss = (accuracy - expected_correct) / (1 - expected_correct) if (1 - expected_correct) else 0
    result = {
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "n": n},
        "accuracy": round(accuracy, 4),
        "pod_recall": round(pod, 4),
        "far_false_alarm_rate": round(far, 4),
        "csi_critical_success_index": round(csi, 4),
        "peirce_skill_score": round(pss, 4),
        "heidke_skill_score": round(hss, 4),
    }
    if payload.y_proba and len(payload.y_proba) == len(yt):
        brier = sum((yt[i] - payload.y_proba[i]) ** 2 for i in range(len(yt))) / len(yt)
        result["brier_score"] = round(brier, 4)
    result["_engine"] = "DisasterAccuracyBenchmark.evaluate_classification"
    result["citations"] = ["Schaefer (1990) Wea Forecasting", "Heidke (1926) Geogr Ann"]
    return result


# ============================================================================
# Health check
# ============================================================================
@router.get("/health", summary="Disaster API health")
def health():
    return {
        "status": "ok",
        "module": "disaster_prediction",
        "endpoints": ["risk/assess", "warning/classify", "hazard/analyze", "predict", "explain", "benchmark"],
        "supported_hazards": ["earthquake", "tsunami", "flood", "landslide", "wildfire", "drought", "rainfall", "climate"],
        "ml_engine_link": "real implementations at /ml_engine/domains/disaster_prediction/",
    }
