"""
DISASTER ACCURACY BENCHMARK
===========================

Membandingkan akurasi multi-model pada tugas prediksi bencana.
Mendukung tugas klasifikasi biner (event/non-event) dan regresi
intensitas (mis. estimasi magnitude, debit puncak).

Sitasi:
    Brier (1950). Verification of Forecasts Expressed in Terms of Probability.
    Hanssen & Kuipers (1965). On the relationship between the frequency of rain.
    Murphy (1993). What is a good forecast? An essay on the nature of goodness.
    Roebber (2009). Visualizing multiple measures of forecast quality.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math


@dataclass
class BenchmarkReport:
    model_name: str
    task: str
    metrics: Dict[str, float]
    n_samples: int
    n_positives: Optional[int] = None
    confusion: Optional[Dict[str, int]] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "metrics": {k: _safe_round(v) for k, v in self.metrics.items()},
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
            "confusion": self.confusion,
            "notes": self.notes,
        }


def _safe_round(x: float, digits: int = 4) -> float:
    try:
        if x is None or math.isnan(float(x)) or math.isinf(float(x)):
            return float("nan")
        return round(float(x), digits)
    except Exception:
        return float("nan")


class DisasterAccuracyBenchmark:
    """
    Evaluator metrik akurasi yang relevan untuk forecast bencana.

    Untuk klasifikasi biner kami menyertakan POD (Probability of Detection),
    FAR (False Alarm Ratio), CSI (Critical Success Index), HSS (Heidke Skill
    Score), Peirce Skill Score, dan Brier Score yang sudah lama menjadi
    standar verifikasi forecast cuaca/iklim sejak Brier (1950).
    """

    def evaluate_classification(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_proba: Optional[Sequence[float]] = None,
        model_name: str = "model",
    ) -> BenchmarkReport:
        if len(y_true) != len(y_pred):
            raise ValueError("Panjang y_true dan y_pred harus sama")

        tp = fp = tn = fn = 0
        for t, p in zip(y_true, y_pred):
            t = int(t)
            p = int(p)
            if t == 1 and p == 1:
                tp += 1
            elif t == 0 and p == 1:
                fp += 1
            elif t == 0 and p == 0:
                tn += 1
            else:
                fn += 1

        n = tp + fp + tn + fn
        n_pos = tp + fn
        n_neg = tn + fp

        accuracy = (tp + tn) / max(1, n)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)  # POD
        f1 = (
            2 * precision * recall / max(1e-9, precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        far = fp / max(1, tp + fp)  # False Alarm Ratio
        csi = tp / max(1, tp + fn + fp)  # Critical Success Index
        pofd = fp / max(1, fp + tn)  # Probability of False Detection
        peirce = recall - pofd  # Peirce / Hanssen-Kuipers

        # Heidke Skill Score
        expected_correct_random = (
            ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / max(1, n)
        )
        hss_denom = max(1e-9, n - expected_correct_random)
        hss = ((tp + tn) - expected_correct_random) / hss_denom

        metrics: Dict[str, float] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall_POD": recall,
            "f1": f1,
            "false_alarm_ratio_FAR": far,
            "critical_success_index_CSI": csi,
            "probability_false_detection_POFD": pofd,
            "peirce_skill_score_PSS": peirce,
            "heidke_skill_score_HSS": hss,
            "base_rate": n_pos / max(1, n),
        }

        if y_proba is not None and len(y_proba) == len(y_true):
            brier = sum((float(p) - int(t)) ** 2 for p, t in zip(y_proba, y_true)) / max(1, n)
            metrics["brier_score"] = brier
            metrics["log_loss"] = self._log_loss(y_true, y_proba)
            metrics["roc_auc"] = self._roc_auc(y_true, y_proba)

        return BenchmarkReport(
            model_name=model_name,
            task="binary_classification",
            metrics=metrics,
            n_samples=n,
            n_positives=n_pos,
            confusion={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            notes=(
                "Metrik mengikuti Brier (1950) dan Hanssen-Kuipers (1965). "
                "POD/FAR/CSI standar untuk verifikasi forecast bencana."
            ),
        )

    def evaluate_regression(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        model_name: str = "model",
    ) -> BenchmarkReport:
        if len(y_true) != len(y_pred):
            raise ValueError("Panjang y_true dan y_pred harus sama")
        n = len(y_true)
        if n == 0:
            raise ValueError("Tidak ada sample untuk evaluasi")

        diffs = [float(p) - float(t) for p, t in zip(y_pred, y_true)]
        abs_diffs = [abs(d) for d in diffs]
        mae = sum(abs_diffs) / n
        rmse = math.sqrt(sum(d * d for d in diffs) / n)
        bias = sum(diffs) / n
        mean_t = sum(float(t) for t in y_true) / n
        ss_res = sum(d * d for d in diffs)
        ss_tot = sum((float(t) - mean_t) ** 2 for t in y_true)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Nash-Sutcliffe Efficiency, standar di hidrologi
        nse = r2  # secara matematis identik untuk forecast deterministik

        # Index of Agreement (Willmott 1981)
        denom_ia = sum(
            (abs(float(p) - mean_t) + abs(float(t) - mean_t)) ** 2
            for p, t in zip(y_pred, y_true)
        )
        ia = 1.0 - ss_res / denom_ia if denom_ia > 0 else 0.0

        return BenchmarkReport(
            model_name=model_name,
            task="regression_intensity",
            metrics={
                "MAE": mae,
                "RMSE": rmse,
                "bias": bias,
                "R2": r2,
                "nash_sutcliffe_efficiency": nse,
                "index_of_agreement_willmott": ia,
            },
            n_samples=n,
            notes=(
                "NSE menjadi standar hidrologi sejak Nash & Sutcliffe (1970). "
                "IA mengikuti Willmott (1981)."
            ),
        )

    def compare(
        self,
        reports: List[BenchmarkReport],
        primary_metric: str = "f1",
    ) -> List[Dict]:
        if not reports:
            return []
        ranked = sorted(
            reports,
            key=lambda r: r.metrics.get(primary_metric, float("-inf")),
            reverse=True,
        )
        out = []
        best_score = ranked[0].metrics.get(primary_metric, 0.0)
        for r in ranked:
            score = r.metrics.get(primary_metric, 0.0)
            out.append({
                "rank": len(out) + 1,
                "model_name": r.model_name,
                "primary_metric": primary_metric,
                "score": _safe_round(score),
                "delta_to_best": _safe_round(score - best_score),
                "all_metrics": {k: _safe_round(v) for k, v in r.metrics.items()},
            })
        return out

    @staticmethod
    def _log_loss(y_true: Sequence[int], y_proba: Sequence[float], eps: float = 1e-15) -> float:
        n = len(y_true)
        s = 0.0
        for t, p in zip(y_true, y_proba):
            p = max(eps, min(1 - eps, float(p)))
            s += -(int(t) * math.log(p) + (1 - int(t)) * math.log(1 - p))
        return s / max(1, n)

    @staticmethod
    def _roc_auc(y_true: Sequence[int], y_proba: Sequence[float]) -> float:
        # Implementasi Mann-Whitney U sederhana (tanpa ties handling)
        pos = [float(p) for p, t in zip(y_proba, y_true) if int(t) == 1]
        neg = [float(p) for p, t in zip(y_proba, y_true) if int(t) == 0]
        if not pos or not neg:
            return float("nan")
        wins = 0.0
        for a in pos:
            for b in neg:
                if a > b:
                    wins += 1
                elif a == b:
                    wins += 0.5
        return wins / (len(pos) * len(neg))
