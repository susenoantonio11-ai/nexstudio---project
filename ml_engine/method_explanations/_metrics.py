"""
Evaluation metric explanations - classification, regression, forecasting metrics.
"""

METRIC_EXPLANATIONS = {

    # ==================================================================
    # ACCURACY
    # ==================================================================
    "accuracy": {
        "name": "Accuracy",
        "category": "classification_metric",
        "purpose": "Proporsi prediksi yang benar.",
        "how_it_works_simple": (
            "Hitung berapa banyak prediksi yang TEPAT, dibagi total prediksi. "
            "Misal: dari 100 sample, 85 diprediksi benar → accuracy = 85%."
        ),
        "math_formulation": {
            "formula": "Accuracy = (TP + TN) / (TP + TN + FP + FN) = jumlah_benar / total",
            "range": "[0, 1] atau [0%, 100%] — semakin tinggi semakin baik",
        },
        "variables": [
            {"symbol": "TP", "name": "True Positive", "description": "Aktual positif, prediksi positif"},
            {"symbol": "TN", "name": "True Negative", "description": "Aktual negatif, prediksi negatif"},
            {"symbol": "FP", "name": "False Positive", "description": "Aktual negatif, prediksi positif (False alarm)"},
            {"symbol": "FN", "name": "False Negative", "description": "Aktual positif, prediksi negatif (Missed)"},
        ],
        "numerical_example": {
            "description": "100 sample: 70 TP, 15 TN, 8 FP, 7 FN",
            "computation": "Accuracy = (70 + 15) / (70 + 15 + 8 + 7) = 85/100 = 0.85",
        },
        "interpretation": (
            "Accuracy = X% berarti X% prediksi benar. "
            "BAHAYA: untuk imbalanced data, accuracy menyesatkan! "
            "Contoh: 95% data adalah class A. Model dummy yang selalu prediksi A = 95% accuracy tapi useless."
        ),
        "limitations": [
            "Misleading untuk imbalanced data — gunakan F1-macro atau balanced accuracy",
            "Tidak membedakan jenis error (FP vs FN)",
            "Tidak memberikan info kalibrasi probability",
        ],
    },

    # ==================================================================
    # PRECISION
    # ==================================================================
    "precision": {
        "name": "Precision",
        "category": "classification_metric",
        "purpose": "Dari semua prediksi POSITIF, berapa yang benar-benar positif.",
        "how_it_works_simple": (
            "Misal model bilang ada 50 pasien high-risk. Ternyata hanya 40 yang benar-benar high-risk. "
            "Precision = 40/50 = 80%. Cocok ketika cost False Positive tinggi (alarm palsu mahal)."
        ),
        "math_formulation": {
            "formula": "Precision = TP / (TP + FP)",
            "range": "[0, 1] — semakin tinggi → semakin sedikit alarm palsu",
        },
        "variables": [
            {"symbol": "TP", "name": "True Positive", "description": "Prediksi positif yang benar"},
            {"symbol": "FP", "name": "False Positive", "description": "Prediksi positif yang salah (alarm palsu)"},
        ],
        "interpretation": (
            "High precision = ketika model bilang positif, biasanya benar. "
            "Penting untuk: spam filter (jangan mark email penting sebagai spam), "
            "fraud detection (jangan block transaksi sah), medical screening."
        ),
        "limitations": [
            "Tidak peduli FN (missed detection)",
            "Bisa tinggi dengan threshold tinggi tapi miss banyak true positive",
        ],
    },

    # ==================================================================
    # RECALL (SENSITIVITY)
    # ==================================================================
    "recall": {
        "name": "Recall (Sensitivity, True Positive Rate)",
        "category": "classification_metric",
        "purpose": "Dari semua kasus AKTUAL positif, berapa yang berhasil ditangkap model.",
        "how_it_works_simple": (
            "Misal ada 100 pasien benar-benar high-risk. Model menangkap 80. "
            "Recall = 80/100 = 80%. Cocok ketika cost False Negative tinggi (missed detection berbahaya)."
        ),
        "math_formulation": {
            "formula": "Recall = TP / (TP + FN)",
            "range": "[0, 1] — semakin tinggi → semakin sedikit missed cases",
        },
        "interpretation": (
            "High recall = model menangkap mayoritas kasus positif. "
            "Penting untuk: cancer screening (jangan miss pasien sakit), "
            "fraud detection (jangan miss transaksi fraud), security alerts."
        ),
        "limitations": [
            "Tidak peduli FP (false alarm)",
            "Bisa tinggi dengan prediksi positif untuk semua → trivially 100%",
        ],
    },

    # ==================================================================
    # F1 SCORE
    # ==================================================================
    "f1_score": {
        "name": "F1 Score",
        "category": "classification_metric",
        "purpose": "Harmonic mean dari precision dan recall — balance keduanya.",
        "how_it_works_simple": (
            "Precision tinggi tapi recall rendah = banyak missed. "
            "Recall tinggi tapi precision rendah = banyak alarm palsu. "
            "F1 mengukur keseimbangan keduanya. Hanya tinggi kalau kedua-duanya tinggi."
        ),
        "math_formulation": {
            "formula": "F1 = 2 · (Precision × Recall) / (Precision + Recall)",
            "alternative": "F1 = 2·TP / (2·TP + FP + FN)",
            "range": "[0, 1] — 1 = perfect, 0 = worst",
        },
        "numerical_example": {
            "description": "Precision = 0.8, Recall = 0.6",
            "computation": "F1 = 2 × (0.8 × 0.6) / (0.8 + 0.6) = 0.96 / 1.4 = 0.686",
        },
        "interpretation": (
            "F1-macro = rata-rata F1 per kelas (treats all classes equally). "
            "F1-weighted = rata-rata F1 di-weight by class size (lebih ramah ke majority class). "
            "Untuk imbalanced data, GUNAKAN F1-macro."
        ),
        "limitations": [
            "Tidak memberikan info absolute prediction count",
            "Treat FP dan FN equally (F-beta untuk weighted version)",
        ],
    },

    # ==================================================================
    # ROC-AUC
    # ==================================================================
    "roc_auc": {
        "name": "ROC-AUC (Area Under ROC Curve)",
        "category": "classification_metric",
        "purpose": "Probability bahwa model rank random positive lebih tinggi dari random negative.",
        "how_it_works_simple": (
            "Bayangkan kamu pilih satu pasien sakit dan satu pasien sehat secara acak. "
            "Probability model memberikan SKOR LEBIH TINGGI ke pasien sakit = ROC-AUC. "
            "0.5 = random; 1.0 = perfect; <0.5 = worse than random."
        ),
        "math_formulation": {
            "formula": "AUC = ∫₀¹ TPR(FPR) d(FPR)",
            "tpr": "TPR = Recall = TP / (TP + FN)",
            "fpr": "FPR = FP / (FP + TN)",
            "interpretation_formula": "AUC = P(score(positive) > score(negative))",
        },
        "variables": [
            {"symbol": "TPR", "name": "True Positive Rate", "description": "Sama dengan Recall"},
            {"symbol": "FPR", "name": "False Positive Rate", "description": "Proporsi negative yang salah diklasifikasi positive"},
        ],
        "interpretation": (
            "AUC > 0.9 = excellent; 0.8-0.9 = good; 0.7-0.8 = fair; <0.7 = poor. "
            "ROC curve memvisualisasi trade-off TPR vs FPR di berbagai threshold. "
            "BAHAYA: ROC-AUC OVER-OPTIMISTIC untuk imbalanced data — gunakan PR-AUC."
        ),
        "limitations": [
            "Misleading untuk severely imbalanced data",
            "Tidak peduli probability magnitude (hanya ranking)",
            "Tidak bisa di-interpret sebagai 'accuracy'",
        ],
    },

    # ==================================================================
    # PR-AUC
    # ==================================================================
    "pr_auc": {
        "name": "PR-AUC (Precision-Recall Area Under Curve)",
        "category": "classification_metric",
        "purpose": "Robust alternative to ROC-AUC untuk imbalanced classification.",
        "how_it_works_simple": (
            "Plot Precision (Y) vs Recall (X) di berbagai threshold. Hitung area di bawah kurva. "
            "Lebih informatif dari ROC-AUC saat positive class langka. "
            "Baseline = proportion of positive (bukan 0.5 seperti ROC)."
        ),
        "math_formulation": {
            "formula": "PR-AUC = ∫₀¹ Precision(Recall) d(Recall)",
            "average_precision": "AP = Σₙ (Rₙ − Rₙ₋₁) · Pₙ",
            "baseline": "PR-AUC random = positive rate (e.g., 0.05 if 5% positive)",
        },
        "interpretation": (
            "PR-AUC tinggi = model bisa achieve tinggi precision DAN recall sekaligus. "
            "Untuk imbalance 1:100, ROC-AUC bisa 0.95 padahal model masih banyak miss. PR-AUC akan jauh lebih rendah."
        ),
        "limitations": [
            "Lebih kompleks untuk dijelaskan ke stakeholder",
            "Tidak ada interpretasi probabilistik seperti ROC-AUC",
        ],
    },

    # ==================================================================
    # MCC
    # ==================================================================
    "mcc": {
        "name": "Matthews Correlation Coefficient (MCC)",
        "category": "classification_metric",
        "purpose": "Single-number metric robust terhadap class imbalance.",
        "how_it_works_simple": (
            "Korelasi antara prediksi dan aktual. -1 = sempurna salah; 0 = random; +1 = sempurna benar. "
            "Tidak bisa dimanipulasi dengan trivial prediction (selalu prediksi majority)."
        ),
        "math_formulation": {
            "formula": "MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
            "range": "[-1, +1] — +1 perfect, 0 random, -1 inverted",
        },
        "interpretation": (
            "MCC = 0.5 sudah dianggap kuat. Robust untuk imbalanced. "
            "Lebih balanced daripada accuracy/F1."
        ),
        "limitations": [
            "Sulit di-extend ke multi-class (ada versi multiclass tapi kurang intuitif)",
            "Kurang dikenal stakeholder",
        ],
    },

    # ==================================================================
    # RMSE
    # ==================================================================
    "rmse": {
        "name": "RMSE (Root Mean Squared Error)",
        "category": "regression_metric",
        "purpose": "Rata-rata error prediksi, dalam unit yang sama dengan target.",
        "how_it_works_simple": (
            "Hitung selisih prediksi vs aktual untuk setiap sample, kuadratkan, rata-rata, lalu akar. "
            "Hasil dalam satuan target. Misal target = harga (Rp), RMSE = Rp 5 juta artinya rata-rata "
            "prediksi meleset Rp 5 juta."
        ),
        "math_formulation": {
            "formula": "RMSE = √( (1/n) Σᵢ₌₁ⁿ (yᵢ − ŷᵢ)² )",
            "range": "[0, ∞) — semakin kecil semakin baik",
        },
        "interpretation": (
            "Karena dikuadratkan, RMSE memberatkan ERROR BESAR. "
            "Sensitive terhadap outlier — beberapa prediksi sangat meleset bisa membuat RMSE besar."
        ),
        "vs_mae": (
            "RMSE: punish outlier, dalam unit asli, lebih populer di ML competition. "
            "MAE: tidak sensitive outlier, interpretasi lebih intuitif (rata-rata absolute error)."
        ),
        "limitations": [
            "Sensitif outlier",
            "Tidak scale-invariant (RMSE 100 untuk target rb beda dengan RMSE 100 untuk target jt)",
        ],
    },

    # ==================================================================
    # MAE
    # ==================================================================
    "mae": {
        "name": "MAE (Mean Absolute Error)",
        "category": "regression_metric",
        "purpose": "Rata-rata absolute error — tidak punish outlier.",
        "math_formulation": {
            "formula": "MAE = (1/n) Σᵢ₌₁ⁿ |yᵢ − ŷᵢ|",
            "range": "[0, ∞)",
        },
        "interpretation": (
            "MAE = 50 berarti RATA-RATA prediksi meleset 50 unit. "
            "Lebih robust terhadap outlier daripada RMSE."
        ),
        "limitations": [
            "Tidak differentiable di 0 (tapi modern optimizer bisa handle)",
            "Tidak punish error besar — bisa miss model yang occasionally sangat salah",
        ],
    },

    # ==================================================================
    # R²
    # ==================================================================
    "r_squared": {
        "name": "R² (Coefficient of Determination)",
        "category": "regression_metric",
        "purpose": "Proporsi variance target yang di-explain oleh model.",
        "how_it_works_simple": (
            "R² = 0 berarti model sama bagusnya dengan PREDIKSI rata-rata target — useless. "
            "R² = 1 berarti perfect fit. R² = 0.7 berarti model menjelaskan 70% variance target."
        ),
        "math_formulation": {
            "formula": "R² = 1 − SS_res / SS_tot",
            "ss_res": "SS_res = Σᵢ (yᵢ − ŷᵢ)²  (residual sum of squares)",
            "ss_tot": "SS_tot = Σᵢ (yᵢ − ȳ)²  (total sum of squares)",
            "range": "(−∞, 1] — bisa negatif jika model lebih buruk dari mean predictor",
        },
        "interpretation": (
            "R² > 0.7 = strong; 0.4-0.7 = moderate; <0.4 = weak. "
            "R² negatif = model SALAH ARAH (lebih buruk dari sekadar prediksi mean)."
        ),
        "limitations": [
            "Bisa naik artificial dengan menambahkan fitur (gunakan Adjusted R²)",
            "Tidak inform tentang validity asumsi linear",
        ],
    },

    # ==================================================================
    # MAPE
    # ==================================================================
    "mape": {
        "name": "MAPE (Mean Absolute Percentage Error)",
        "category": "regression_metric",
        "purpose": "Error sebagai persentase — scale-invariant, intuitif.",
        "math_formulation": {
            "formula": "MAPE = (100/n) Σᵢ |yᵢ − ŷᵢ| / |yᵢ|",
            "range": "[0%, ∞%) — biasanya <50% untuk model decent",
        },
        "interpretation": (
            "MAPE = 10% berarti rata-rata prediksi meleset 10% dari nilai aktual. "
            "Mudah dijelaskan ke business stakeholder."
        ),
        "limitations": [
            "Undefined ketika yᵢ = 0",
            "Asymmetric: lebih punish prediksi over-estimate vs under-estimate",
            "Bisa misleading untuk nilai aktual sangat kecil",
        ],
    },

    # ==================================================================
    # BRIER SCORE
    # ==================================================================
    "brier_score": {
        "name": "Brier Score",
        "category": "calibration_metric",
        "purpose": "Mengukur kalibrasi probability — apakah prediksi 70% benar-benar terjadi 70% kali.",
        "math_formulation": {
            "formula": "Brier = (1/n) Σᵢ (pᵢ − yᵢ)²",
            "range": "[0, 1] — 0 = perfect, 0.25 = random",
        },
        "interpretation": (
            "Brier kecil = probability output reliable. "
            "Critical untuk medical / financial decisions yang menggunakan threshold probability."
        ),
        "limitations": [
            "Hanya untuk binary classification (multi-class butuh extension)",
            "Equal penalty untuk over-confidence dan under-confidence",
        ],
    },
}
