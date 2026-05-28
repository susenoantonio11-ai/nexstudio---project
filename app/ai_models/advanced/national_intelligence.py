"""Phase-2 Category 9 — National Intelligence AI (10 models)"""
from .base import AdvancedAIModel


class NationalRiskIntelligenceEngine(AdvancedAIModel):
    name="NationalRiskIntelligenceEngine"; model_id="national_risk_intelligence_engine"; category="national_intelligence"; domain="ai_intelligence"
    description="Composite risk score nasional: hazard × exposure × vulnerability × capacity (UNISDR)."
    citations=["UNISDR (2015) Sendai Framework"]
    realtime_capable=True
    def run(self, p):
        return self._envelope({"index":"R = H × E × V / C","scope":"national"}, confidence=0.80, uncertainty=0.20)


class CriticalInfrastructureProtectionAI(AdvancedAIModel):
    name="CriticalInfrastructureProtectionAI"; model_id="critical_infrastructure_protection_ai"; category="national_intelligence"; domain="ai_intelligence"
    description="Proteksi infrastruktur kritis: ranking ancaman + rekomendasi mitigasi."
    citations=["Rinaldi et al. (2001) IEEE Control Systems"]
    def run(self, p):
        return self._envelope({"sectors":["power","water","telco","transport","health","banking"]}, confidence=0.78, uncertainty=0.22)


class NationalEmergencyPrioritizationAI(AdvancedAIModel):
    name="NationalEmergencyPrioritizationAI"; model_id="national_emergency_prioritization_ai"; category="national_intelligence"; domain="ai_intelligence"
    description="Prioritisasi emergency nasional via multi-criteria (severity × population × actionability × time)."
    realtime_capable=True
    def run(self, p):
        return self._envelope({"criteria":["severity","population","actionability","time"],"method":"AHP weighted"}, confidence=0.82, uncertainty=0.18)


class MultiRegionHazardCorrelationAI(AdvancedAIModel):
    name="MultiRegionHazardCorrelationAI"; model_id="multi_region_hazard_correlation_ai"; category="national_intelligence"; domain="ai_intelligence"
    description="Korelasi hazard lintas-provinsi untuk mendeteksi pattern nasional."
    formulas=["spatial Moran's I; cross-region Granger"]
    def run(self, p):
        return self._envelope({"methods":["spatial_corr","granger","CCM"]}, confidence=0.75, uncertainty=0.25)


class StrategicDisasterAssessmentAI(AdvancedAIModel):
    name="StrategicDisasterAssessmentAI"; model_id="strategic_disaster_assessment_ai"; category="national_intelligence"; domain="ai_intelligence"
    description="Strategic assessment dampak bencana di tingkat nasional + ekonomi makro."
    citations=["Hallegatte et al. (2017) Unbreakable"]
    def run(self, p):
        return self._envelope({"dimensions":["GDP_impact","poverty_increase","fiscal_pressure","reconstruction_cost"]}, confidence=0.75, uncertainty=0.25)


class NationalSecurityGeoAI(AdvancedAIModel):
    name="NationalSecurityGeoAI"; model_id="national_security_geo_ai"; category="national_intelligence"; domain="ai_intelligence"
    description="Geospatial intelligence untuk keamanan nasional: border, sea, infrastruktur strategis."
    def run(self, p):
        return self._envelope({"layers":["border","maritime","critical_assets","movement_patterns"]}, confidence=0.75, uncertainty=0.25)


class CrossProvinceImpactAnalyzer(AdvancedAIModel):
    name="CrossProvinceImpactAnalyzer"; model_id="cross_province_impact_analyzer"; category="national_intelligence"; domain="ai_intelligence"
    description="Analisis dampak lintas-provinsi (supply chain, evacuation flow, financial transmission)."
    def run(self, p):
        return self._envelope({"methods":["IO_table","gravity_model","network_effect"]}, confidence=0.75, uncertainty=0.25)


class EmergencyResourceOptimizationAI(AdvancedAIModel):
    name="EmergencyResourceOptimizationAI"; model_id="emergency_resource_optimization_ai"; category="national_intelligence"; domain="ai_intelligence"
    description="Optimasi alokasi resource emergency (medis, logistik, evakuasi) via integer LP + heuristic."
    citations=["Sherali et al. (1991) Transp. Res. B 25"]
    realtime_capable=True
    def run(self, p):
        return self._envelope({"solver":"CBC + heuristic","constraints":["capacity","time_window","priority"]}, confidence=0.80, uncertainty=0.20)


class NationalRecoveryForecastAI(AdvancedAIModel):
    name="NationalRecoveryForecastAI"; model_id="national_recovery_forecast_ai"; category="national_intelligence"; domain="ai_intelligence"
    description="Forecast waktu + biaya recovery pasca bencana berdasarkan baseline + dampak."
    citations=["Hallegatte (2014) Risk Analysis 34"]
    def run(self, p):
        return self._envelope({"outputs":["recovery_time_months","reconstruction_cost","poverty_recovery_index"]}, confidence=0.75, uncertainty=0.25)


class CrisisEscalationReasoningEngine(AdvancedAIModel):
    name="CrisisEscalationReasoningEngine"; model_id="crisis_escalation_reasoning_engine"; category="national_intelligence"; domain="ai_intelligence"
    description="Reasoning escalasi krisis: trigger detection + propagation paths + intervention windows."
    realtime_capable=True
    def run(self, p):
        return self._envelope({"triggers":["seismic","hydro","pandemic","financial"],"paths":"Bayesian network"}, confidence=0.78, uncertainty=0.22)


MODELS=[NationalRiskIntelligenceEngine,CriticalInfrastructureProtectionAI,NationalEmergencyPrioritizationAI,
        MultiRegionHazardCorrelationAI,StrategicDisasterAssessmentAI,NationalSecurityGeoAI,CrossProvinceImpactAnalyzer,
        EmergencyResourceOptimizationAI,NationalRecoveryForecastAI,CrisisEscalationReasoningEngine]
