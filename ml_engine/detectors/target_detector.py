"""
Target Variable Auto-Detector
==============================
Heuristically detects the most likely Y (target) variable
from a dataset profile, providing reasoning and confidence score.

Heuristics combine:
- Column name signals (revenue, target, label, churn, sales, price, etc)
- Position (last column often target in many datasets)
- Type appropriateness (Y is rarely an ID or datetime)
- Variance / cardinality (Y has meaningful variation)
- Position bias for forecasting (numerical near time column)

Reference: AutoML literature - Hutter, Kotthoff & Vanschoren (2019)
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional


class TargetDetector:
    """
    Detects target variable Y from dataset profile.
    Returns suggestion with reasoning, confidence score, and alternatives.
    """

    # Strong target indicators (case-insensitive contains)
    # Revenue-related numerical names get HIGHEST priority because they are
    # the most common business targets in real-world analytics datasets.
    TARGET_KEYWORDS = {
        "high": [
            "target", "label", "y", "outcome", "result", "class",
            "total_amount", "total", "revenue", "sales_amount", "gross",
            "net", "profit", "income"
        ],
        "medium": [
            "sales", "price", "expense", "churn", "default", "score",
            "rating", "conversion", "click", "purchase", "amount", "value",
            "demand", "quantity", "rate"
        ],
        "low": ["status", "type", "category", "level", "tier", "count"]
    }

    # Anti-target indicators (Y is rarely these)
    # "is_*" boolean flags are usually features (fraud_flag, etc), not the actual target
    NOT_TARGET_KEYWORDS = [
        "id", "uuid", "key", "code", "name", "email", "phone",
        "address", "url", "description", "comment", "note",
        "created", "updated", "deleted", "timestamp",
        "is_suspect", "is_flag", "_flag", "_indicator"
    ]

    def detect(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            profile: Output from DataProfiler.profile()

        Returns:
            {
                "suggested_target": str,
                "confidence": float,
                "reasoning": str,
                "alternatives": [{"column": str, "score": float, "reason": str}],
                "all_scores": [{"column": str, "score": float, "factors": dict}]
            }
        """
        columns = profile.get("columns", [])
        if not columns:
            return {
                "suggested_target": None,
                "confidence": 0.0,
                "reasoning": "Dataset has no columns",
                "alternatives": [],
                "all_scores": [],
            }

        scored = []
        for col in columns:
            score, factors = self._score_column(col, columns)
            scored.append({
                "column": col["name"],
                "score": round(score, 3),
                "factors": factors,
                "type": col["inferred_type"],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[0]

        # Confidence is the gap between top and second
        if len(scored) > 1:
            gap = top["score"] - scored[1]["score"]
            confidence = min(1.0, top["score"] * (0.5 + gap))
        else:
            confidence = top["score"]

        reasoning = self._build_reasoning(top, scored)

        alternatives = [
            {
                "column": s["column"],
                "score": s["score"],
                "type": s["type"],
                "reason": self._summarize_factors(s["factors"])
            }
            for s in scored[1:4]  # Top 3 alternatives
        ]

        return {
            "suggested_target": top["column"] if top["score"] > 0.2 else None,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
            "alternatives": alternatives,
            "all_scores": scored,
        }

    def _score_column(
        self, col: Dict[str, Any], all_columns: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        score = 0.0
        factors = {}

        name_lower = col["name"].lower().strip()
        col_type = col["inferred_type"]
        n_unique = col["n_unique"]
        completeness = col["completeness_pct"] / 100.0

        # Factor 1: Anti-target check (negative score)
        for anti in self.NOT_TARGET_KEYWORDS:
            if anti in name_lower:
                factors["anti_target_penalty"] = -0.5
                score -= 0.5
                break

        # Factor 2: Strong keyword match
        keyword_score = 0.0
        for keyword in self.TARGET_KEYWORDS["high"]:
            if keyword == name_lower or f"_{keyword}" in name_lower or f"{keyword}_" in name_lower:
                keyword_score = max(keyword_score, 0.6)
        for keyword in self.TARGET_KEYWORDS["medium"]:
            if keyword in name_lower:
                keyword_score = max(keyword_score, 0.4)
        for keyword in self.TARGET_KEYWORDS["low"]:
            if keyword in name_lower:
                keyword_score = max(keyword_score, 0.2)
        if keyword_score > 0:
            factors["keyword_match"] = keyword_score
            score += keyword_score

        # Factor 3: Type appropriateness
        type_score = 0.0
        if col_type == "numerical":
            type_score = 0.25
        elif col_type == "categorical":
            type_score = 0.20
        elif col_type == "boolean":
            type_score = 0.30  # Binary classification target
        elif col_type == "datetime":
            type_score = -0.3  # Datetime almost never Y
        elif col_type == "text":
            type_score = -0.2

        factors["type_appropriateness"] = type_score
        score += type_score

        # Factor 4: Cardinality check (Y shouldn't be ID-like)
        cardinality_score = 0.0
        if n_unique <= 2 and col_type in ("boolean", "categorical"):
            cardinality_score = 0.15  # Binary classification
        elif 2 < n_unique <= 20 and col_type == "categorical":
            cardinality_score = 0.10  # Multi-class
        elif n_unique > 0.95 * (col.get("n_missing", 0) + n_unique):
            cardinality_score = -0.4  # Likely ID column

        factors["cardinality"] = cardinality_score
        score += cardinality_score

        # Factor 5: Completeness (Y must be mostly present)
        if completeness < 0.5:
            factors["completeness_penalty"] = -0.3
            score -= 0.3

        # Factor 6: Position - last column often is target in tabular data
        position = col.get("position", 0)
        if position == len(all_columns) - 1:
            factors["last_column_bonus"] = 0.10
            score += 0.10

        # Clamp score
        score = max(0.0, min(1.0, score))

        return score, factors

    def _build_reasoning(self, top: Dict, all_scored: List[Dict]) -> str:
        col = top["column"]
        score = top["score"]
        factors = top["factors"]

        if score < 0.2:
            return (
                f"No strong target candidate detected. The highest-scoring column "
                f"'{col}' only achieved {score:.2f} confidence. We recommend the user "
                f"manually specify the target variable."
            )

        parts = [f"Suggested target variable: '{col}' (score: {score:.2f})."]

        if factors.get("keyword_match", 0) > 0.4:
            parts.append(
                f"The column name strongly suggests a target variable based on "
                f"common naming conventions in machine learning datasets."
            )

        if factors.get("type_appropriateness", 0) > 0.2:
            parts.append(
                f"Its data type ({top['type']}) is suitable for prediction tasks."
            )

        if factors.get("cardinality", 0) > 0:
            parts.append(
                "The cardinality of this column matches typical target patterns."
            )

        if factors.get("last_column_bonus", 0) > 0:
            parts.append(
                "It is positioned as the last column, which is a common "
                "convention for target variables in tabular datasets."
            )

        if len(all_scored) > 1:
            second = all_scored[1]
            parts.append(
                f"Alternative candidates considered include '{second['column']}' "
                f"(score: {second['score']:.2f})."
            )

        return " ".join(parts)

    def _summarize_factors(self, factors: Dict[str, float]) -> str:
        positive = [k for k, v in factors.items() if v > 0]
        negative = [k for k, v in factors.items() if v < 0]
        parts = []
        if positive:
            parts.append("Positive: " + ", ".join(positive))
        if negative:
            parts.append("Negative: " + ", ".join(negative))
        return "; ".join(parts) if parts else "No significant factors"
