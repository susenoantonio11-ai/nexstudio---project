"""
Auto-register all known Nexlytics AI models on startup.
Importing this module triggers registration of every model in the platform.
"""
from .model_registry import register_class

def register_all():
    # === Disaster Prediction Domain ===
    register_class(
        id="risk_score_engine", name="RiskScoreEngine", domain="disaster",
        category="composite_risk_scoring",
        description="UNISDR composite Hazard×Exposure×Vulnerability×Model risk score (0..1).",
        formula="R = wH·H + wE·E + wV·V + wF·model_prob (default w = [0.30,0.25,0.20,0.25])",
        citations=["UNISDR (2015) GAR", "Cardona et al. (2012) IPCC SREX Ch.2"],
        api_endpoint="/api/disaster/risk/assess",
        method_monitor={
            "method": "Weighted composite (UNISDR framework)",
            "why_used": "Industry-standard risk decomposition with explainable contributors",
            "limitations": ["Linear weighted; assumes independence between H/E/V"],
        },
        integration_targets=["EWC", "RiskIntelligence", "PrimaryMonitor"],
    )
    register_class(
        id="warning_level_classifier", name="WarningLevelClassifier", domain="disaster",
        category="classifier",
        description="5-level warning classifier — Normal / Advisory / Watch / Warning / Critical.",
        formula="thresholds = [0.20, 0.40, 0.60, 0.80]",
        api_endpoint="/api/disaster/warning/classify",
        method_monitor={"method": "Threshold-based 5-level classifier"},
        integration_targets=["EWC", "PrimaryMonitor"],
    )
    register_class(
        id="hybrid_ensemble_pipeline", name="HybridEnsemblePipeline", domain="disaster",
        category="ensemble_predictor",
        description="Soft-voting ensemble: LSTM (temporal) + XGBoost (spatial) + Bayesian (posterior).",
        formula="final = 0.30·LSTM + 0.45·XGBoost + 0.25·Bayesian",
        citations=["Lundberg & Lee (2017) NIPS 30", "Friedman (2001) Ann Stats"],
        api_endpoint="/api/disaster/predict",
        dependencies=["torch", "xgboost"],
        fallback_available=True,
        method_monitor={
            "method": "Soft-voting weighted ensemble",
            "why_used": "Combines strengths: temporal (LSTM), spatial (XGB), uncertainty (Bayesian)",
            "limitations": ["Static weights; ideally learn weights via stacking on validation set"],
        },
        integration_targets=["EWC", "ResearchLab"],
    )
    register_class(
        id="shap_explainer", name="SHAPExplainer", domain="disaster",
        category="explainability",
        description="Lundberg-Lee SHAP feature attribution with permutation fallback.",
        formula="φ_i = E[f(x)|x_S∪{i}] - E[f(x)|x_S]",
        citations=["Lundberg & Lee (2017) NIPS 30"],
        api_endpoint="/api/disaster/explain",
        dependencies=["shap"],
        fallback_available=True,
        method_monitor={"method": "SHAP TreeExplainer (permutation fallback)"},
        integration_targets=["EWC", "ResearchLab", "ModelMonitor"],
    )
    register_class(
        id="disaster_accuracy_benchmark", name="DisasterAccuracyBenchmark", domain="disaster",
        category="evaluation",
        description="Classification + regression metrics: POD, FAR, CSI, HSS, Brier, ROC-AUC, MAE, RMSE, R².",
        api_endpoint="/api/disaster/benchmark",
        method_monitor={"method": "Forecast-verification metrics suite"},
        integration_targets=["ResearchLab"],
    )

    # === 8 Hazard Analyzers ===
    hazards = [
        ("earthquake_analyzer", "EarthquakeAnalyzer", "earthquake",
         "Gutenberg-Richter b-value via Aki MLE + ETAS",
         "log10 N(M) = a - bM; b ≈ log10(e) / (mean_M - Mc)",
         ["Gutenberg & Richter (1944)", "Aki (1965)"]),
        ("tsunami_analyzer", "TsunamiAnalyzer", "tsunami",
         "Wells-Coppersmith surface rupture length",
         "SRL = 10^(-3.55 + 0.74·Mw) km",
         ["Wells & Coppersmith (1994) BSSA 84(4)"]),
        ("flood_analyzer", "FloodAnalyzer", "flood",
         "Rational method peak discharge + SCS Curve Number",
         "Q = C · I · A (m³/s)",
         ["Mulvaney (1850)", "USDA-SCS (1972)"]),
        ("landslide_analyzer", "LandslideAnalyzer", "landslide",
         "Infinite slope factor of safety",
         "FOS = (c' + γ·z·cos²θ·tan φ') / (γ·z·sin θ·cos θ)",
         ["Skempton & DeLory (1957)"]),
        ("wildfire_analyzer", "WildfireAnalyzer", "wildfire",
         "FWI Van Wagner — FFMC, DMC, DC, ISI, BUI, FWI",
         "FWI = f(ISI, BUI)",
         ["Van Wagner (1987) Forestry Tech Rep 35"]),
        ("drought_analyzer", "DroughtAnalyzer", "drought",
         "SPI / SPEI McKee standardized precipitation",
         "SPI = (X - mean) / sd",
         ["McKee, Doesken & Kleist (1993)"]),
        ("rainfall_analyzer", "RainfallAnalyzer", "rainfall",
         "Gumbel Type-I extreme value return period",
         "x_T = μ - β·ln(-ln(1 - 1/T))",
         ["Gumbel (1958) Statistics of Extremes"]),
        ("climate_analyzer", "ClimateRiskAnalyzer", "climate",
         "Anomaly + OLS trend + ENSO Niño-3.4 SST",
         "trend = OLS slope(year, T)",
         ["IPCC AR6 WG1 (2021)"]),
    ]
    for hid, hname, hkey, desc, formula, cites in hazards:
        register_class(
            id=hid, name=hname, domain="disaster", category="hazard_analyzer",
            description=desc, formula=formula, citations=cites,
            api_endpoint=f"/api/disaster/hazard/analyze (type={hkey})",
            method_monitor={"method": desc},
            integration_targets=["GeoDisaster", "EWC"],
        )

    # === Quality Engine Domain ===
    register_class(
        id="analysis_quality_engine", name="AnalysisQualityEngine", domain="quality",
        category="quality_control",
        description="Quality control orchestrator: data-quality + model-quality + CV + uncertainty + ensemble + scientific + explainability.",
        formula="accuracy_score = 0.30·data_q + 0.40·model_q + 0.30·scientific_plausibility",
        citations=["Wong & Lee (2003)", "Lundberg & Lee (2017)", "James et al. (2013) ISLR"],
        api_endpoint="/api/ai/quality/assess",
        method_monitor={
            "method": "Composite quality scorecard with 8 sub-validators",
            "why_used": "Ensures every analysis result is validated, explainable, and reliable before reaching the user.",
            "limitations": ["Pure-Python heuristics; install scikit-learn/statsmodels for exact CV."],
        },
        integration_targets=["DataWorkspace", "DataScience", "ResearchLab", "GeoDisaster", "EWC", "PrimaryMonitor"],
    )
    register_class(
        id="data_quality_validator", name="DataQualityValidator", domain="quality",
        category="data_validator",
        description="Validates dataset quality across 6 dimensions: missing, duplicate, outlier, imbalance, leakage, bias.",
        api_endpoint="/api/ai/quality/validate-dataset",
        integration_targets=["DataWorkspace", "DataScience"],
    )
    register_class(
        id="model_quality_validator", name="ModelQualityValidator", domain="quality",
        category="model_validator",
        description="Computes accuracy/precision/recall/F1/ROC-AUC/MAE/RMSE/R² with 95% Wilson CI.",
        formula="Wilson CI: ((p + z²/2n) ± z√(p(1-p)/n + z²/4n²)) / (1 + z²/n)",
        citations=["Wilson (1927) JASA 22(158)"],
        api_endpoint="/api/ai/quality/validate-model",
        integration_targets=["ResearchLab", "DataScience"],
    )
    register_class(
        id="uncertainty_estimator", name="UncertaintyEstimator", domain="quality",
        category="uncertainty",
        description="Confidence + prediction interval + error margin + uncertainty score.",
        formula="margin = z · sd / √n; CI = estimate ± margin",
        api_endpoint="/api/ai/quality/uncertainty",
        integration_targets=["EWC", "ResearchLab"],
    )
    register_class(
        id="explainability_checker", name="ExplainabilityChecker", domain="quality",
        category="explainability",
        description="Permutation importance + feature ranking + decision trace.",
        api_endpoint="/api/ai/quality/explain",
        integration_targets=["EWC", "ResearchLab"],
    )

    # === Geospatial domain ===
    register_class(
        id="time_series_flood_tracker", name="TimeSeriesFloodTracker", domain="geospatial",
        category="temporal_analysis",
        description="Multi-temporal flood evolution tracker via NDWI/MNDWI + raster differencing.",
        api_endpoint="/api/geo/timeseries/flood",
        integration_targets=["EWC", "GeoDisaster"],
    )

    # === Complex Data Science Engine (8 new models) ===
    register_class(
        id="complex_dataset_analyzer", name="ComplexDatasetAnalyzer", domain="data_science",
        category="dataset_analysis",
        description="Joint detection of structure, target, leakage, hidden patterns for complex tabular datasets.",
        formula="Complexity = 0.30·tanh(cols/25) + 0.20·tanh(log10(rows)/7) + 0.20·H(types) + 0.30·tanh(miss%/25)",
        citations=["Kraskov et al. (2004) Phys. Rev. E 69:066138", "Tukey (1977)"],
        api_endpoint="/api/ai/complex/analyze-dataset",
        method_monitor={
            "method": "EDA composite + mutual_info_classif + IQR + Pearson relationship graph",
            "why_used": "End-to-end profile for complex datasets including hidden-pattern discovery.",
            "limitations": ["Mutual info is for numeric features; categoricals require encoding."],
        },
        integration_targets=["DataScience", "DataWorkspace", "PrimaryMonitor"],
    )
    register_class(
        id="large_dataset_processing_engine", name="LargeDatasetProcessingEngine", domain="data_science",
        category="streaming_io",
        description="Memory-safe chunked profiler for CSV / Parquet / Excel / JSON files of arbitrary size.",
        formula="Welford running variance: μₙ = μₙ₋₁ + (xₙ − μₙ₋₁)/n",
        citations=["Welford (1962) Technometrics 4(3)", "Knuth TAOCP Vol.2"],
        api_endpoint="/api/ai/complex/profile-large-dataset",
        dependencies=["pandas", "pyarrow"], fallback_available=True,
        method_monitor={"method": "Streaming chunked profiler with Welford-Knuth running statistics"},
        integration_targets=["DataScience", "DataWorkspace"],
    )
    register_class(
        id="multimodal_data_science_engine", name="MultimodalDataScienceEngine", domain="data_science",
        category="multimodal_fusion",
        description="Early-fusion concatenation of tabular + image + text + time-series features with baseline classifier/regressor.",
        formula="X_fused = [X_tab | X_img | X_text | X_ts]; standardize; train baseline.",
        citations=["Baltrušaitis et al. (2019) IEEE TPAMI 41(2)", "Ngiam et al. (2011) ICML"],
        api_endpoint="/api/ai/complex/multimodal-fuse",
        method_monitor={"method": "Early-fusion with handcrafted modality encoders"},
        integration_targets=["DataScience", "ResearchLab"],
    )
    register_class(
        id="image_analysis_ai_model", name="ImageAnalysisAIModel", domain="computer_vision",
        category="image_analysis",
        description="Multi-task image analyzer (classification, segmentation, object detection, quality check).",
        formula="Otsu σ²_B(t) = ω₀(t)ω₁(t)·(μ₀(t) − μ₁(t))²",
        citations=["Otsu (1979) IEEE Trans. SMC 9(1)", "Pertuz et al. (2013) Pattern Recognition 46(5)"],
        api_endpoint="/api/ai/image/analyze",
        dependencies=["Pillow", "numpy"], fallback_available=True,
        method_monitor={"method": "Otsu thresholding + connected components + handcrafted features"},
        integration_targets=["DataScience", "PrimaryMonitor", "GeoDisaster"],
    )
    register_class(
        id="computer_vision_pipeline", name="ComputerVisionPipeline", domain="computer_vision",
        category="orchestrator",
        description="End-to-end CV pipeline: preprocess → feature extraction → analysis → explain.",
        api_endpoint="/api/ai/image/analyze",
        method_monitor={"method": "Linear pipeline orchestrator"},
        integration_targets=["DataScience"],
    )
    register_class(
        id="image_dataset_builder", name="ImageDatasetBuilder", domain="computer_vision",
        category="dataset_builder",
        description="Walks a folder, deduplicates via pHash, detects blur and corruption, performs stratified split.",
        formula="pHash bit i = 1 iff I(i_left) > I(i_right) on a 9×8 luminance grid",
        citations=["Zauner (2010) — pHash"],
        api_endpoint="/api/ai/image/build-dataset",
        method_monitor={"method": "Folder walk + pHash dedup + Laplacian blur + stratified split"},
        integration_targets=["DataScience", "DataWorkspace"],
    )
    register_class(
        id="visual_feature_extractor", name="VisualFeatureExtractor", domain="computer_vision",
        category="feature_extraction",
        description="44-dim handcrafted descriptor: color hist (24) + Sobel stats (3) + LBP (10) + Hu moments (7).",
        citations=["Swain & Ballard (1991)", "Ojala et al. (2002)", "Hu (1962)"],
        api_endpoint="/api/ai/image/extract-features",
        method_monitor={"method": "Color histogram + Sobel + LBP + Hu invariant moments"},
        integration_targets=["DataScience", "Multimodal"],
    )
    register_class(
        id="image_explainability_engine", name="ImageExplainabilityEngine", domain="computer_vision",
        category="explainability",
        description="Region occlusion + edge saliency + textual rationale.",
        citations=["Selvaraju et al. (2017) ICCV", "Zeiler & Fergus (2014) ECCV"],
        api_endpoint="/api/ai/image/explain",
        method_monitor={"method": "Saliency = mean(edge_grad, occlusion_drop)"},
        integration_targets=["DataScience", "ResearchLab"],
    )

    # === ADVANCED AI ECOSYSTEM (100 next-generation models) =============
    try:
        import sys, pathlib
        backend_path = str(pathlib.Path(__file__).resolve().parent.parent.parent / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        from app.ai_models.advanced.registry import register_all_advanced
        n_advanced = register_all_advanced(register_class)
        print(f"[Registry] Registered {n_advanced} advanced AI models")
    except Exception as _e:
        print(f"[Registry] Advanced model registration skipped: {_e}")

    # === Multisource Geo Fusion ==========================================
    # === Thesis pipeline ===========================================
    register_class(
        id="flood_panel_builder", name="FloodPanelBuilder", domain="research_pipeline",
        category="panel_construction",
        description="Joins GEE features + BNPB events into province×time panel; lag features + Antecedent Precipitation Index + monsoon phase + temporal split (2016-2022/2023/2024-2025).",
        formula="API_t = 0.85·API_{t-1} + rainfall_t (Kohler & Linsley 1951)",
        citations=["Tehrany et al. (2014) J. Hydrol. 512", "Bergmeir & Benítez (2012) Inf. Sci. 191"],
        api_endpoint="/api/research/flood/build-panel",
        method_monitor={"method": "Province-time panel with strict temporal split"},
        integration_targets=["ResearchLab", "EWC"],
    )
    register_class(
        id="hybrid_lstm_xgboost", name="HybridLSTMXGBoost", domain="research_pipeline",
        category="hybrid_model",
        description="LSTM (temporal sequence) + XGBoost (static + lag features) soft-voting ensemble. Auto-fallback to MLPClassifier + GradientBoosting when torch/xgboost unavailable.",
        formula="final_p = w_LSTM·p_LSTM + w_XGB·p_XGB (Wolpert 1992)",
        citations=["Hochreiter & Schmidhuber (1997)", "Chen & Guestrin (2016)", "Wolpert (1992) Neural Networks 5"],
        api_endpoint="/api/research/flood/run",
        dependencies=["torch", "xgboost"], fallback_available=True,
        method_monitor={"method": "Soft-voting LSTM + tree boosting"},
        integration_targets=["ResearchLab"],
    )
    register_class(
        id="hybrid_shap_explainer", name="HybridSHAPExplainer", domain="research_pipeline",
        category="explainability",
        description="SHAP TreeExplainer (XGB branch) + permutation (LSTM branch) fused by ensemble weights. Produces global, per-province, and local waterfall explanations.",
        citations=["Lundberg & Lee (2017) NeurIPS 30", "Lundberg, Erion & Lee (2018) Nat. Mach. Intell. 2"],
        api_endpoint="/api/research/flood/run",
        dependencies=["shap"], fallback_available=True,
        method_monitor={"method": "Hybrid SHAP + permutation"},
        integration_targets=["ResearchLab", "EWC"],
    )
    register_class(
        id="flood_research_orchestrator", name="FloodResearchOrchestrator", domain="research_pipeline",
        category="orchestrator",
        description="End-to-end reproducible pipeline: GEE pull → BNPB ingest → panel build → Hybrid LSTM-XGBoost → SHAP. Mirrors the thesis abstract 1:1.",
        api_endpoint="/api/research/flood/run",
        method_monitor={"method": "End-to-end thesis pipeline"},
        integration_targets=["ResearchLab"],
    )

    # === Reasoning ===========================================
    register_class(
        id="dynamic_model_selection_engine", name="DynamicModelSelectionEngine",
        domain="reasoning", category="model_selection",
        description="Algorithm-agnostic, dataset-aware ranking. Scores N candidate algorithms per problem domain (tabular/timeseries/NLP/CV/geospatial) with explicit reasoning, computational cost, risks, and confidence — replaces hardcoded model picks.",
        formula="score_i = Σ w_k · f_k(characteristics);  confidence = 0.5 + 0.4·top + 0.1·(top − mean_top3)",
        citations=["Wolpert (1996) Neural Computation 8(7) — No Free Lunch", "Olson et al. (2017) PSB"],
        api_endpoint="/api/ai/reasoning/select-model",
        method_monitor={
            "method": "Heuristic per-candidate scoring across N algorithms",
            "why_used": "Prevents algorithm bias. Enables transparent recommendations the user can audit.",
            "limitations": ["Heuristic, not learned. Replace with learning-to-rank as telemetry accumulates."],
        },
        integration_targets=["Analytics", "DataWorkspace", "AIStudio", "ResearchLab"],
    )

    register_class(
        id="multisource_flood_fusion", name="MultisourceFloodFusion", domain="geo_disaster",
        category="multisource_fusion",
        description="Stacks JRC + S1/S2 + MODIS + GLDAS + CHIRPS + SRTM + HydroSHEDS + WorldCover + SoilGrids + BNPB into a feature cube and trains a Random Forest flood classifier.",
        formula="TWI = ln(a/tan β); NDWI = (G−NIR)/(G+NIR); MNDWI = (G−SWIR)/(G+SWIR); RF: ŷ = mode(b₁..b_T)",
        citations=[
            "Pekel et al. (2016) Nature 540", "Funk et al. (2015) Sci Data 2",
            "Beven & Kirkby (1979) Hydrol Sci Bull 24", "Twele et al. (2016) IJRS 37",
            "Tehrany et al. (2014) J Hydrol 512", "Lewis et al. (2017) RSE 202",
            "Breiman (2001) Mach Learn 45",
        ],
        api_endpoint="/api/ai/complex/multisource-flood-fuse",
        dependencies=["sklearn"], fallback_available=False,
        method_monitor={
            "method": "Per-source feature engineering + early-fusion + Random Forest",
            "why_used": "Standard practice in flood susceptibility ML literature.",
            "limitations": ["Caller must pre-resample to common grid; D8 flow accumulation not yet integrated."],
        },
        integration_targets=["GeoDisaster", "EWC", "ResearchLab"],
    )

print(f"[Registry] Registered {len(__import__('ml_engine.registry', fromlist=['registry']).registry.list_all())} models")


# Auto-execute on import
register_all()
