"""
Miscellaneous explanations: cross-validation, drift, statistical tests, splitting.
"""

MISC_EXPLANATIONS = {

    # ==================================================================
    # K-FOLD CROSS-VALIDATION
    # ==================================================================
    "kfold_cv": {
        "name": "K-Fold Cross-Validation",
        "category": "validation",
        "purpose": "Estimasi performa model lebih reliable dengan rata-rata K split.",
        "how_it_works_simple": (
            "Bagi dataset jadi K bagian (folds). Untuk setiap fold: latih di K-1 fold, test di 1 fold. "
            "Lakukan K kali. Hasil = rata-rata K skor → estimasi performa lebih stable."
        ),
        "math_formulation": {
            "split": "D = D₁ ∪ D₂ ∪ ... ∪ Dₖ  (disjoint folds)",
            "fit_on_train": "Untuk i = 1..K: train pada D \\ Dᵢ, evaluate pada Dᵢ",
            "cv_score": "CV_score = (1/K) Σᵢ score(model_i, Dᵢ)",
            "cv_std": "CV_std = √( (1/K) Σᵢ (scoreᵢ − mean)² )",
        },
        "variables": [
            {"symbol": "K", "name": "n_folds", "description": "Jumlah fold (typical: 5 atau 10)"},
            {"symbol": "Dᵢ", "name": "fold ke-i", "description": "Subset data ke-i"},
        ],
        "interpretation": (
            "CV mean ± CV std memberikan range expected performance. "
            "CV std besar = model tidak stabil → mungkin perlu more data atau regularization. "
            "CV mean dipakai untuk MODEL SELECTION (pilih model dengan CV terbaik)."
        ),
        "limitations": [
            "K tinggi = computation lama (K kali training)",
            "K terlalu kecil = high variance estimate",
            "Standard K-Fold tidak preserve class balance — gunakan StratifiedKFold untuk klasifikasi",
            "Tidak suitable untuk time series — gunakan TimeSeriesSplit",
        ],
    },

    # ==================================================================
    # STRATIFIED K-FOLD
    # ==================================================================
    "stratified_kfold": {
        "name": "Stratified K-Fold Cross-Validation",
        "category": "validation",
        "purpose": "K-Fold yang preserve proporsi class di setiap fold.",
        "how_it_works_simple": (
            "Sama seperti K-Fold, tapi memastikan setiap fold memiliki proporsi class yang SAMA "
            "dengan dataset asli. Wajib untuk imbalanced classification."
        ),
        "interpretation": (
            "Tanpa stratification: fold bisa kebetulan semua satu class → model dilatih tanpa minority class → useless. "
            "Stratified: setiap fold representatif."
        ),
        "limitations": [
            "Tidak applicable untuk regression (bisa pakai binning kontinu kemudian stratify)",
            "Untuk multi-label classification, butuh strategi khusus",
        ],
    },

    # ==================================================================
    # TIME SERIES SPLIT
    # ==================================================================
    "time_series_split": {
        "name": "TimeSeriesSplit",
        "category": "validation",
        "purpose": "CV untuk time series — selalu train pada masa lalu, test pada masa depan.",
        "how_it_works_simple": (
            "Random K-Fold tidak boleh untuk time series — bisa belajar masa depan untuk prediksi masa lalu (data leakage). "
            "TimeSeriesSplit memastikan: fold ke-i = train pada timestamp 1..t, test pada t+1..t+h."
        ),
        "math_formulation": {
            "split_i": "Train: D[1:tᵢ], Test: D[tᵢ+1 : tᵢ+1+h]",
            "expanding_window": "Train growing: tᵢ < tᵢ₊₁",
        },
        "interpretation": "Lebih realistic untuk forecasting evaluation.",
        "limitations": [
            "Less data per fold (especially early folds)",
            "Tidak handle multi-frequency time series langsung",
        ],
    },

    # ==================================================================
    # CONFUSION MATRIX
    # ==================================================================
    "confusion_matrix": {
        "name": "Confusion Matrix",
        "category": "evaluation",
        "purpose": "Tabel breakdown prediksi vs aktual untuk setiap class.",
        "how_it_works_simple": (
            "Baris = aktual, kolom = prediksi (atau sebaliknya). Setiap sel = jumlah sample dengan pasangan tersebut. "
            "Diagonal = benar, off-diagonal = salah."
        ),
        "math_formulation": {
            "binary": "[[TN, FP], [FN, TP]]",
            "multiclass": "M[i,j] = jumlah sample dengan aktual class i diprediksi class j",
        },
        "interpretation": (
            "Mata cepat: diagonal tebal = bagus. "
            "Off-diagonal kuat di kombinasi tertentu = model selalu salah membedakan dua class itu. "
            "Hitung dari sini: precision, recall, F1 per class."
        ),
        "limitations": [
            "Sulit dibaca untuk banyak class (k > 10)",
            "Tidak inform threshold (untuk binary, hanya snapshot di threshold tertentu)",
        ],
    },

    # ==================================================================
    # IQR OUTLIER DETECTION
    # ==================================================================
    "iqr_outlier": {
        "name": "IQR Method for Outlier Detection",
        "category": "outlier_detection",
        "purpose": "Identifikasi outlier menggunakan interquartile range — robust dan distribution-free.",
        "math_formulation": {
            "iqr": "IQR = Q3 − Q1",
            "lower_bound": "lower = Q1 − 1.5 × IQR",
            "upper_bound": "upper = Q3 + 1.5 × IQR",
            "outlier_condition": "x is outlier if x < lower OR x > upper",
        },
        "variables": [
            {"symbol": "Q1", "name": "first quartile", "description": "Nilai di percentile 25"},
            {"symbol": "Q3", "name": "third quartile", "description": "Nilai di percentile 75"},
            {"symbol": "IQR", "name": "interquartile range", "description": "Q3 − Q1, mengukur spread tengah 50% data"},
        ],
        "numerical_example": {
            "description": "Data: [10, 20, 25, 30, 35, 40, 45, 50, 100]",
            "computation": "Q1=22.5, Q3=47.5, IQR=25; lower=22.5-37.5=-15, upper=47.5+37.5=85; outlier: 100",
        },
        "interpretation": (
            "Sangat robust terhadap distribusi (tidak asumsi normal). "
            "Faktor 1.5 standar; bisa diubah ke 3.0 untuk lebih konservatif (extreme outlier saja)."
        ),
        "limitations": [
            "Mungkin terlalu agresif untuk distribusi heavy-tailed valid",
            "Tidak inform 'mengapa' outlier — hanya 'apa'",
        ],
    },

    # ==================================================================
    # Z-SCORE OUTLIER
    # ==================================================================
    "zscore_outlier": {
        "name": "Z-Score Outlier Detection",
        "category": "outlier_detection",
        "purpose": "Identifikasi outlier sebagai point dengan |z-score| > threshold.",
        "math_formulation": {
            "zscore": "z = (x − μ) / σ",
            "outlier_condition": "x is outlier if |z| > 3 (typical threshold)",
        },
        "interpretation": (
            "z > 3 berarti point > 3 standar deviasi dari mean → diharapkan terjadi <0.3% pada distribusi normal. "
            "Untuk distribusi non-normal, gunakan IQR atau Modified Z-score."
        ),
        "limitations": [
            "Asumsi distribusi normal",
            "Sensitive terhadap outlier sendiri (mean dan std bias)",
            "Modified Z-score (gunakan median + MAD) lebih robust",
        ],
    },

    # ==================================================================
    # PSI (POPULATION STABILITY INDEX)
    # ==================================================================
    "psi": {
        "name": "PSI (Population Stability Index)",
        "category": "drift_detection",
        "purpose": "Mengukur seberapa banyak distribusi fitur berubah antara reference (train) dan production.",
        "math_formulation": {
            "binning": "Bin nilai ke 10 quantile dari reference distribution",
            "formula": "PSI = Σᵢ (cur_pctᵢ − ref_pctᵢ) × ln(cur_pctᵢ / ref_pctᵢ)",
        },
        "variables": [
            {"symbol": "ref_pctᵢ", "name": "reference proportion", "description": "Proporsi nilai di bin i untuk training data"},
            {"symbol": "cur_pctᵢ", "name": "current proportion", "description": "Proporsi nilai di bin i untuk production data"},
        ],
        "interpretation": (
            "PSI < 0.1 = no significant shift, model masih reliable. "
            "PSI 0.1-0.25 = moderate shift, monitor closely. "
            "PSI > 0.25 = significant shift, RETRAIN recommended."
        ),
        "limitations": [
            "Sensitive terhadap binning strategy",
            "Tidak detect intra-bin shift",
        ],
    },

    # ==================================================================
    # KOLMOGOROV-SMIRNOV TEST
    # ==================================================================
    "ks_test": {
        "name": "Kolmogorov-Smirnov Test",
        "category": "statistical_test",
        "purpose": "Non-parametric test untuk membandingkan dua distribusi.",
        "math_formulation": {
            "statistic": "D = max|F₁(x) − F₂(x)|  (max difference of CDFs)",
            "p_value": "Menolak H₀ jika D > critical_value",
        },
        "variables": [
            {"symbol": "F₁, F₂", "name": "empirical CDFs", "description": "Cumulative distribution function dari kedua sample"},
            {"symbol": "D", "name": "KS statistic", "description": "Max gap antara dua CDF"},
        ],
        "interpretation": (
            "p-value < 0.05 → distribusi BEDA significant. "
            "Berguna untuk drift detection — apakah feature distribution berubah?"
        ),
        "limitations": [
            "Hanya untuk continuous distribution",
            "Sensitive terhadap perbedaan di tengah, kurang sensitive di ekor",
        ],
    },

    # ==================================================================
    # LOG-RANK TEST
    # ==================================================================
    "log_rank": {
        "name": "Log-Rank Test (Mantel-Cox)",
        "category": "statistical_test",
        "purpose": "Test apakah survival curves dari 2+ groups berbeda significant.",
        "math_formulation": {
            "statistic": "Z² = Σ (Oᵢ − Eᵢ)² / Vᵢ  ~ χ²(g−1)",
            "observed": "Oᵢ = jumlah event observed di group i",
            "expected": "Eᵢ = jumlah event expected di group i (under H₀: same survival)",
        },
        "interpretation": (
            "p-value < 0.05 → survival berbeda significant antar group. "
            "Common: bandingkan treatment arm vs control."
        ),
        "limitations": [
            "Asumsi proportional hazards",
            "Tidak adjust untuk covariates (gunakan Cox regression untuk itu)",
        ],
    },
}
