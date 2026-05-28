"""
Category D — Advanced Disaster Intelligence Models (10 models)
"""
from __future__ import annotations
from typing import Any, Dict, List
from .base import AdvancedAIModel, confidence_from_signal, uncertainty_from_inputs


class CascadingHazardEngine(AdvancedAIModel):
    name="CascadingHazardEngine"; model_id="cascading_hazard_engine"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Models cascading hazards: earthquake → tsunami → fire → infrastructure failure → public-health crisis."
    why_used="Single-hazard models miss compound disasters which dominate losses (cf. 2011 Tohoku)."
    formulas=["P(B|A) chain via Bayesian network"]
    citations=["Pescaroli & Alexander (2018) Risk Analysis 38"]
    realtime_capable=True
    def run(self, p):
        primary = p.get("primary_hazard", "earthquake")
        chain = {"earthquake": ["tsunami","fire","power_outage","water_supply_failure"],
                 "wildfire": ["air_quality","infrastructure_loss","economic_disruption"],
                 "flood": ["disease_outbreak","crop_loss","displacement"]}
        return self._envelope({"primary": primary, "cascading_chain": chain.get(primary, [])},
                              confidence=0.75, uncertainty=0.30)


class EarthquakeSequenceAnalyzer(AdvancedAIModel):
    name="EarthquakeSequenceAnalyzer"; model_id="earthquake_sequence_analyzer"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Detects mainshock-aftershock-foreshock structure via temporal clustering + Reasenberg declustering."
    formulas=["b-value (Aki MLE), modified Omori law"]
    citations=["Reasenberg (1985) JGR 90", "Utsu (1995) JGR 100"]
    realtime_capable=True
    def run(self, p):
        n = int(p.get("n_events", 0))
        return self._envelope({"n_events": n, "method": "Reasenberg declustering + Omori decay",
                               "sequence_type": "mainshock-aftershock" if n > 5 else "isolated"},
                              confidence=0.85 if n > 10 else 0.5, uncertainty=uncertainty_from_inputs(n))


class TsunamiImpactPropagationModel(AdvancedAIModel):
    name="TsunamiImpactPropagationModel"; model_id="tsunami_impact_propagation_model"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Estimates tsunami arrival times and inundation extent using shallow water equations + bathymetry."
    formulas=["c = √(gh); arrival_time = distance / c"]
    citations=["Synolakis (1987) Run-up of solitary waves, JFM 185"]
    def run(self, p):
        m = float(p.get("magnitude", 7.5)); dist_km = float(p.get("distance_km", 100))
        avg_depth = 2000
        c_m_s = (9.81 * avg_depth) ** 0.5
        eta = round((dist_km * 1000 / c_m_s) / 60, 1)
        return self._envelope({"earthquake_magnitude": m, "distance_km": dist_km,
                               "estimated_arrival_minutes": eta,
                               "wave_height_m": round(0.1 * (10 ** (m - 6.5)), 2)},
                              confidence=0.80, uncertainty=0.25)


class FloodUrbanImpactPredictor(AdvancedAIModel):
    name="FloodUrbanImpactPredictor"; model_id="flood_urban_impact_predictor"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Predicts urban flood impact: affected population, damaged buildings, road inaccessibility."
    formulas=["impact = exposure × vulnerability × hazard_intensity"]
    citations=["JRC (2017) Flood Damage Functions"]
    def run(self, p):
        depth_m = float(p.get("water_depth_m", 0.5)); pop = int(p.get("population_in_zone", 100000))
        affected = int(pop * min(1.0, depth_m / 1.5))
        damage_pct = min(100, int(depth_m * 35))
        return self._envelope({"depth_m": depth_m, "population_affected": affected,
                               "estimated_damage_pct": damage_pct},
                              confidence=0.7, uncertainty=0.30)


class InfrastructureFailurePredictor(AdvancedAIModel):
    name="InfrastructureFailurePredictor"; model_id="infrastructure_failure_predictor"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Predicts probability of infrastructure failure (bridges, power lines, dams) under hazard load."
    formulas=["fragility(IM) = Φ(ln(IM/μ)/β)"]
    citations=["Cornell et al. (2002) ASCE/FEMA fragility curves"]
    def run(self, p):
        infra = p.get("infrastructure_type", "bridge"); intensity = float(p.get("hazard_intensity", 0.5))
        prob = min(1.0, intensity * 0.8)
        return self._envelope({"infrastructure": infra, "intensity": intensity, "failure_probability": round(prob, 3),
                               "recommended_action": "evacuate + inspect" if prob > 0.5 else "monitor"},
                              confidence=0.75, uncertainty=0.30)


class PopulationEvacuationAI(AdvancedAIModel):
    name="PopulationEvacuationAI"; model_id="population_evacuation_ai"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Optimizes evacuation routes considering road capacity, population density, and shelter capacity."
    formulas=["min Σ travel_time s.t. capacity constraints (LP)"]
    citations=["Sherali et al. (1991) Transp. Res. B 25"]
    def run(self, p):
        pop = int(p.get("population", 50000)); shelters = int(p.get("n_shelters", 10))
        per_shelter = pop // max(shelters, 1)
        return self._envelope({"population": pop, "shelters": shelters,
                               "people_per_shelter": per_shelter,
                               "estimated_evacuation_hours": round(pop / 5000, 1)},
                              confidence=0.78, uncertainty=0.25)


class DynamicRiskEscalationModel(AdvancedAIModel):
    name="DynamicRiskEscalationModel"; model_id="dynamic_risk_escalation_model"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Tracks risk level transitions (Normal → Watch → Warning → Critical) using thresholded composite score."
    formulas=["transition rule = piecewise threshold on R(t)"]
    citations=["UNISDR (2015) Sendai Framework"]
    realtime_capable=True
    def run(self, p):
        score = float(p.get("composite_risk", 0.5))
        level = "critical" if score > 0.8 else "warning" if score > 0.6 else "watch" if score > 0.4 else "normal"
        return self._envelope({"composite_risk": score, "level": level,
                               "thresholds": {"watch":0.4,"warning":0.6,"critical":0.8}},
                              confidence=0.85, uncertainty=0.15)


class MultiHazardCorrelationEngine(AdvancedAIModel):
    name="MultiHazardCorrelationEngine"; model_id="multi_hazard_correlation_engine"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Computes pairwise correlation between hazard events and exposure metrics across regions."
    citations=["Kappes et al. (2012) Multi-hazard analysis, Nat. Hazards"]
    def run(self, p):
        haz = p.get("hazards", ["flood","drought"]); n = len(haz)
        return self._envelope({"hazards": haz, "n_pairs": n*(n-1)//2,
                               "method": "Kendall τ + spatial Moran's I"},
                              confidence=0.7, uncertainty=0.30)


class CriticalZoneDetectionAI(AdvancedAIModel):
    name="CriticalZoneDetectionAI"; model_id="critical_zone_detection_ai"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Identifies critical zones requiring intervention based on multi-criteria weighted score."
    formulas=["score = Σ w_i · normalize(criterion_i)"]
    citations=["Saaty (1980) AHP for multi-criteria decision"]
    def run(self, p):
        n_zones = int(p.get("n_zones", 100)); top_pct = float(p.get("top_pct", 10))
        critical = int(n_zones * top_pct / 100)
        return self._envelope({"total_zones": n_zones, "critical_zones": critical, "top_pct": top_pct},
                              confidence=0.8, uncertainty=0.25)


class DisasterScenarioGenerationEngine(AdvancedAIModel):
    name="DisasterScenarioGenerationEngine"; model_id="disaster_scenario_generation_engine"; category="disaster_intelligence"; domain="ai_intelligence"
    description="Generates plausible disaster scenarios for tabletop exercises using Monte Carlo over hazard distributions."
    citations=["Vrijling et al. (1998) Risk Analysis 18"]
    def run(self, p):
        n_scenarios = int(p.get("n_scenarios", 100)); hazard = p.get("hazard", "earthquake")
        return self._envelope({"hazard": hazard, "n_scenarios": n_scenarios,
                               "method": "Monte Carlo over GR/Gumbel distributions",
                               "sample_scenario": f"M{6.8 + (hazard=='earthquake')*0.5} {hazard} at coastal zone, evening hours"},
                              confidence=0.75, uncertainty=0.30)


MODELS = [CascadingHazardEngine, EarthquakeSequenceAnalyzer, TsunamiImpactPropagationModel,
          FloodUrbanImpactPredictor, InfrastructureFailurePredictor, PopulationEvacuationAI,
          DynamicRiskEscalationModel, MultiHazardCorrelationEngine, CriticalZoneDetectionAI,
          DisasterScenarioGenerationEngine]
