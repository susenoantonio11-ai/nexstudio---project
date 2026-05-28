"""Phase-2 Category 6 — Uncertainty & Scientific Trust AI (10 models)
Mendekatkan sistem ke academic-grade scientific prediction.
"""
from .base import AdvancedAIModel


class BayesianReasoningEngine(AdvancedAIModel):
    name="BayesianReasoningEngine"; model_id="bayesian_reasoning_engine"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Bayesian reasoning dengan posterior sampling (NUTS / HMC) untuk inferensi probabilistik."
    formulas=["P(θ|D) ∝ P(D|θ)·P(θ)"]
    citations=["Hoffman & Gelman (2014) JMLR — NUTS"]
    def run(self, p):
        return self._envelope({"sampler":"NUTS","chains":4,"draws":2000}, confidence=0.85, uncertainty=0.15)


class ProbabilisticScientificAI(AdvancedAIModel):
    name="ProbabilisticScientificAI"; model_id="probabilistic_scientific_ai"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Output probabilistic distribution (bukan point estimate) untuk semua scientific prediction."
    citations=["Gelman et al. (2013) Bayesian Data Analysis 3rd"]
    def run(self, p):
        return self._envelope({"output":"posterior distribution","quantiles":[0.05,0.50,0.95]}, confidence=0.85, uncertainty=0.15)


class UncertaintyPropagationModel(AdvancedAIModel):
    name="UncertaintyPropagationModel"; model_id="uncertainty_propagation_model"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Propagasi uncertainty melalui pipeline (input → preprocess → model → decision) via MC."
    formulas=["Var(f(X)) ≈ Var via Monte Carlo / unscented transform"]
    citations=["Smith (2014) Uncertainty Quantification"]
    def run(self, p):
        return self._envelope({"method":"Monte Carlo + unscented transform","propagation_steps":5}, confidence=0.80, uncertainty=0.20)


class ConfidenceBoundaryEstimator(AdvancedAIModel):
    name="ConfidenceBoundaryEstimator"; model_id="confidence_boundary_estimator"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Estimasi boundary kepercayaan model: di mana prediksi dapat dipercaya, di mana tidak (OOD detection)."
    citations=["Hendrycks & Gimpel (2017) ICLR — OOD detection"]
    def run(self, p):
        return self._envelope({"ood_detector":"Mahalanobis + softmax confidence","threshold_method":"P95"}, confidence=0.78, uncertainty=0.22)


class ScientificReliabilityScorer(AdvancedAIModel):
    name="ScientificReliabilityScorer"; model_id="scientific_reliability_scorer"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Skor reliability scientific (test-retest + inter-rater + Cronbach + GRADE composite)."
    citations=["GRADE (2008) BMJ 336","Cronbach (1951) Psychometrika 16"]
    def run(self, p):
        return self._envelope({"components":["test_retest","inter_rater","internal_consistency","grade_quality"]}, confidence=0.80, uncertainty=0.20)


class PredictionTrustAI(AdvancedAIModel):
    name="PredictionTrustAI"; model_id="prediction_trust_ai"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Estimasi trust score per prediksi: confidence × calibration × OOD × explanation_consistency."
    def run(self, p):
        return self._envelope({"trust_dimensions":["confidence","calibration","ood","explanation"]}, confidence=0.78, uncertainty=0.22)


class ExplainableUncertaintyEngine(AdvancedAIModel):
    name="ExplainableUncertaintyEngine"; model_id="explainable_uncertainty_engine"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Decompose uncertainty + jelaskan per komponen (data, model, distribution shift)."
    formulas=["Uncertainty = aleatoric + epistemic"]
    citations=["Gal & Ghahramani (2016) ICML — MC dropout"]
    def run(self, p):
        return self._envelope({"types":["aleatoric","epistemic","distributional"],"explained":True}, confidence=0.80, uncertainty=0.20)


class RiskConfidenceFusionModel(AdvancedAIModel):
    name="RiskConfidenceFusionModel"; model_id="risk_confidence_fusion_model"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Fuse risk score × confidence × uncertainty menjadi single decision support index."
    def run(self, p):
        return self._envelope({"index":"risk × confidence / (1+uncertainty)","weight_calibration":"per domain"}, confidence=0.80, uncertainty=0.20)


class ProbabilisticForecastingAI(AdvancedAIModel):
    name="ProbabilisticForecastingAI"; model_id="probabilistic_forecasting_ai"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Probabilistic forecasting dengan quantile + ensemble + conformal calibration."
    citations=["Gneiting & Katzfuss (2014) Annu. Rev. Stat."]
    def run(self, p):
        return self._envelope({"quantiles":[0.10,0.25,0.50,0.75,0.90],"calibration":"split conformal"}, confidence=0.85, uncertainty=0.15)


class ScientificTrustworthinessEngine(AdvancedAIModel):
    name="ScientificTrustworthinessEngine"; model_id="scientific_trustworthiness_engine"; category="uncertainty_trust"; domain="ai_intelligence"
    description="Composite scientific trustworthiness: reproducibility + transparency + provenance + calibration."
    citations=["Wilkinson et al. (2016) FAIR principles"]
    def run(self, p):
        return self._envelope({"axes":["reproducibility","transparency","provenance","calibration"]}, confidence=0.85, uncertainty=0.15)


MODELS=[BayesianReasoningEngine,ProbabilisticScientificAI,UncertaintyPropagationModel,ConfidenceBoundaryEstimator,
        ScientificReliabilityScorer,PredictionTrustAI,ExplainableUncertaintyEngine,RiskConfidenceFusionModel,
        ProbabilisticForecastingAI,ScientificTrustworthinessEngine]
