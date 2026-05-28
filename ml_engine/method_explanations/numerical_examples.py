"""
Numerical Example Engine
=========================
Generates concrete numerical examples for each method using actual computation.
Used to make math accessible: "given X, compute Y, get Z".
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import math


class NumericalExampleEngine:
    """Compute simple numerical examples on the fly."""

    def example_for(self, method_name: str) -> Optional[Dict[str, Any]]:
        method = method_name.lower().strip().replace("-", "_").replace(" ", "_")

        if "linear_regression" in method or method == "ols":
            return self._linear_regression_example()
        if "logistic" in method:
            return self._logistic_example()
        if "standardization" in method or method == "z_score":
            return self._zscore_example()
        if "minmax" in method:
            return self._minmax_example()
        if "kmeans" in method:
            return self._kmeans_example()
        if "accuracy" in method:
            return self._accuracy_example()
        if "f1" in method:
            return self._f1_example()
        if "precision" in method:
            return self._precision_example()
        if "recall" in method or method == "sensitivity":
            return self._recall_example()
        if "rmse" in method:
            return self._rmse_example()
        if "mae" in method:
            return self._mae_example()
        if method == "r_squared" or method == "r2":
            return self._r_squared_example()
        if "iqr" in method:
            return self._iqr_example()
        if "zscore_outlier" in method:
            return self._zscore_outlier_example()
        if "tfidf" in method:
            return self._tfidf_example()
        if "psi" in method:
            return self._psi_example()
        return None

    # =====================================================
    # Examples
    # =====================================================
    def _linear_regression_example(self) -> Dict[str, Any]:
        # Simple OLS computation for y = ax + b
        xs = [1, 2, 3, 4, 5]
        ys = [3, 5, 7, 9, 11]  # perfectly linear
        n = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = num / den
        intercept = y_mean - slope * x_mean

        return {
            "title": "Linear Regression dengan 5 data point",
            "input": {"x": xs, "y": ys},
            "computation": [
                f"x̄ (mean x) = ({'+'.join(str(x) for x in xs)}) / {n} = {x_mean}",
                f"ȳ (mean y) = ({'+'.join(str(y) for y in ys)}) / {n} = {y_mean}",
                f"slope w = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)² = {num} / {den} = {slope}",
                f"intercept b = ȳ − w·x̄ = {y_mean} − {slope}·{x_mean} = {intercept}",
            ],
            "model": f"ŷ = {slope}·x + {intercept}",
            "test_prediction": f"x = 6 → ŷ = {slope}·6 + {intercept} = {slope * 6 + intercept}",
        }

    def _logistic_example(self) -> Dict[str, Any]:
        z_values = [-2, -1, 0, 1, 2, 3]
        sigmoids = [1 / (1 + math.exp(-z)) for z in z_values]
        return {
            "title": "Sigmoid σ(z) untuk berbagai z",
            "computation": [
                f"σ({z}) = 1 / (1 + e^(-{z})) = 1 / (1 + {math.exp(-z):.4f}) = {p:.4f}"
                for z, p in zip(z_values, sigmoids)
            ],
            "interpretation": (
                "z = 0 → p = 0.5 (decision boundary). "
                "z > 0 → p > 0.5 → prediksi class 1. "
                "z < 0 → p < 0.5 → prediksi class 0."
            ),
        }

    def _zscore_example(self) -> Dict[str, Any]:
        values = [10, 20, 30, 40, 50]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        zs = [(v - mean) / std for v in values]
        return {
            "title": "Standardization (Z-score) untuk fitur dengan nilai [10, 20, 30, 40, 50]",
            "computation": [
                f"μ = (10+20+30+40+50) / 5 = {mean}",
                f"σ = √((Σ(xᵢ−μ)²) / n) = √({var}) = {std:.4f}",
            ] + [
                f"z({v}) = ({v} − {mean}) / {std:.4f} = {z:.4f}" for v, z in zip(values, zs)
            ],
            "result": "Setelah standardization: nilai berkisar dari -1.41 sampai +1.41",
        }

    def _minmax_example(self) -> Dict[str, Any]:
        values = [10, 20, 30, 40, 50]
        x_min = min(values)
        x_max = max(values)
        scaled = [(v - x_min) / (x_max - x_min) for v in values]
        return {
            "title": "Min-Max Scaling untuk fitur [10, 20, 30, 40, 50]",
            "computation": [
                f"x_min = {x_min}, x_max = {x_max}, range = {x_max - x_min}",
            ] + [
                f"x'({v}) = ({v} − {x_min}) / ({x_max} − {x_min}) = {s:.4f}" for v, s in zip(values, scaled)
            ],
            "result": f"Setelah scaling: {[round(s, 2) for s in scaled]} — range tepat [0, 1]",
        }

    def _kmeans_example(self) -> Dict[str, Any]:
        points = [(1, 1), (1, 2), (2, 1), (8, 8), (8, 9), (9, 8)]
        centroids_init = [(0, 0), (10, 10)]
        # One iteration
        clusters = {0: [], 1: []}
        for p in points:
            d0 = math.sqrt((p[0] - centroids_init[0][0]) ** 2 + (p[1] - centroids_init[0][1]) ** 2)
            d1 = math.sqrt((p[0] - centroids_init[1][0]) ** 2 + (p[1] - centroids_init[1][1]) ** 2)
            clusters[0 if d0 < d1 else 1].append(p)
        new_centroids = {
            i: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
            for i, pts in clusters.items()
        }
        return {
            "title": "K-Means satu iterasi untuk 6 point dengan k=2",
            "input": {
                "points": points,
                "initial_centroids": centroids_init,
            },
            "computation": [
                "Step 1: Hitung jarak Euclidean tiap point ke kedua centroid.",
                "Step 2: Assign point ke centroid terdekat.",
                f"Cluster 0 (closer to (0,0)): {clusters[0]}",
                f"Cluster 1 (closer to (10,10)): {clusters[1]}",
                "Step 3: Update centroid = rata-rata point dalam cluster.",
                f"Centroid 0 baru = {new_centroids[0]}",
                f"Centroid 1 baru = {new_centroids[1]}",
            ],
            "interpretation": "Centroid bergerak ke pusat clusternya. Iterasi berikutnya akan re-assign jika ada perubahan.",
        }

    def _accuracy_example(self) -> Dict[str, Any]:
        return {
            "title": "Accuracy dari 100 prediksi",
            "scenario": {"TP": 70, "TN": 15, "FP": 8, "FN": 7},
            "computation": "Accuracy = (TP + TN) / total = (70 + 15) / (70+15+8+7) = 85 / 100 = 0.85",
            "result": "Accuracy = 85%",
            "warning": (
                "PERHATIAN: jika dari 100 sample 95% adalah class A, "
                "model dummy yang selalu prediksi A = 95% accuracy padahal useless."
            ),
        }

    def _f1_example(self) -> Dict[str, Any]:
        tp, fp, fn = 40, 10, 20
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return {
            "title": "F1 Score dari TP=40, FP=10, FN=20",
            "computation": [
                f"Precision = TP / (TP+FP) = 40 / (40+10) = {precision:.4f}",
                f"Recall = TP / (TP+FN) = 40 / (40+20) = {recall:.4f}",
                f"F1 = 2 × (P × R) / (P + R) = 2 × ({precision:.2f} × {recall:.2f}) / ({precision:.2f} + {recall:.2f}) = {f1:.4f}",
            ],
            "interpretation": "F1 menyeimbangkan precision (kualitas alarm) dan recall (cakupan deteksi).",
        }

    def _precision_example(self) -> Dict[str, Any]:
        return {
            "title": "Precision dari TP=80, FP=20",
            "computation": "Precision = TP / (TP+FP) = 80 / (80+20) = 80 / 100 = 0.80",
            "result": "80% prediksi positive benar; 20% false alarm.",
        }

    def _recall_example(self) -> Dict[str, Any]:
        return {
            "title": "Recall dari TP=80, FN=20",
            "computation": "Recall = TP / (TP+FN) = 80 / (80+20) = 80 / 100 = 0.80",
            "result": "Model menangkap 80% dari semua kasus positive aktual; 20% missed.",
        }

    def _rmse_example(self) -> Dict[str, Any]:
        ys = [100, 200, 300, 400, 500]
        preds = [110, 190, 320, 380, 530]
        n = len(ys)
        squared_errors = [(y - p) ** 2 for y, p in zip(ys, preds)]
        mse = sum(squared_errors) / n
        rmse = math.sqrt(mse)
        return {
            "title": "RMSE untuk 5 prediksi",
            "input": {"actual": ys, "predicted": preds},
            "computation": [
                f"errors² = {squared_errors}",
                f"MSE = sum / n = {sum(squared_errors)} / {n} = {mse}",
                f"RMSE = √MSE = √{mse} = {rmse:.4f}",
            ],
            "interpretation": f"Rata-rata prediksi meleset sekitar {rmse:.0f} unit dari nilai aktual.",
        }

    def _mae_example(self) -> Dict[str, Any]:
        ys = [100, 200, 300, 400, 500]
        preds = [110, 190, 320, 380, 530]
        n = len(ys)
        abs_errors = [abs(y - p) for y, p in zip(ys, preds)]
        mae = sum(abs_errors) / n
        return {
            "title": "MAE untuk 5 prediksi",
            "input": {"actual": ys, "predicted": preds},
            "computation": [
                f"|errors| = {abs_errors}",
                f"MAE = sum / n = {sum(abs_errors)} / {n} = {mae:.4f}",
            ],
            "interpretation": f"Rata-rata prediksi meleset {mae:.0f} unit.",
        }

    def _r_squared_example(self) -> Dict[str, Any]:
        ys = [100, 200, 300, 400, 500]
        preds = [110, 190, 320, 380, 530]
        y_mean = sum(ys) / len(ys)
        ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        r2 = 1 - ss_res / ss_tot
        return {
            "title": "R² untuk 5 prediksi",
            "computation": [
                f"ȳ = {y_mean}",
                f"SS_res = Σ(yᵢ−ŷᵢ)² = {ss_res}",
                f"SS_tot = Σ(yᵢ−ȳ)² = {ss_tot}",
                f"R² = 1 − {ss_res}/{ss_tot} = {r2:.4f}",
            ],
            "interpretation": f"Model menjelaskan {r2*100:.1f}% variance target.",
        }

    def _iqr_example(self) -> Dict[str, Any]:
        data = [10, 20, 25, 30, 35, 40, 45, 50, 100]
        sorted_data = sorted(data)
        n = len(sorted_data)
        q1 = sorted_data[n // 4]
        q3 = sorted_data[3 * n // 4]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = [x for x in data if x < lower or x > upper]
        return {
            "title": "IQR Outlier Detection untuk data [10, 20, 25, 30, 35, 40, 45, 50, 100]",
            "computation": [
                f"Q1 = {q1}, Q3 = {q3}, IQR = {iqr}",
                f"lower bound = Q1 − 1.5·IQR = {q1} − {1.5 * iqr} = {lower}",
                f"upper bound = Q3 + 1.5·IQR = {q3} + {1.5 * iqr} = {upper}",
                f"Outlier (di luar [{lower}, {upper}]): {outliers}",
            ],
        }

    def _zscore_outlier_example(self) -> Dict[str, Any]:
        return {
            "title": "Z-Score Outlier",
            "computation": [
                "Misal: x = 100, μ = 50, σ = 10",
                "z = (100 − 50) / 10 = 5",
                "|z| > 3 → OUTLIER",
            ],
        }

    def _tfidf_example(self) -> Dict[str, Any]:
        return {
            "title": "TF-IDF Computation",
            "input": {
                "doc1": "the cat sat on the mat",
                "doc2": "the dog jumped",
                "doc3": "the bird flew over the cat",
            },
            "computation": [
                "Term 'cat':",
                "  TF di doc1 = 1/6 (cat muncul 1x dari 6 kata)",
                "  IDF = log(3 / 2) = 0.405  (cat di 2 dari 3 doc)",
                "  TF-IDF di doc1 = 1/6 × 0.405 = 0.0676",
                "Term 'the':",
                "  IDF = log(3 / 3) = 0  (di semua doc)",
                "  TF-IDF = 0  (kata umum di-zero-kan)",
            ],
            "interpretation": "Kata umum seperti 'the' di-suppress; kata distinctive seperti 'jumped' diberi weight tinggi.",
        }

    def _psi_example(self) -> Dict[str, Any]:
        return {
            "title": "PSI between training (ref) and production (cur)",
            "computation": [
                "Bin 1: ref=20%, cur=15% → contribution = (0.15-0.20)·ln(0.15/0.20) = 0.014",
                "Bin 2: ref=30%, cur=25% → contribution = (0.25-0.30)·ln(0.25/0.30) = 0.009",
                "Bin 3: ref=30%, cur=35% → contribution = (0.35-0.30)·ln(0.35/0.30) = 0.008",
                "Bin 4: ref=20%, cur=25% → contribution = (0.25-0.20)·ln(0.25/0.20) = 0.011",
                "PSI = 0.014 + 0.009 + 0.008 + 0.011 = 0.042",
            ],
            "interpretation": "PSI = 0.042 < 0.1 → no significant drift, model masih reliable.",
        }
