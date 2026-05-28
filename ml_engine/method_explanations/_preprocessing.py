"""
Preprocessing method explanations - scaling, encoding, imputation, feature engineering.
"""

PREPROCESSING_EXPLANATIONS = {

    # ==================================================================
    # STANDARDIZATION (Z-SCORE)
    # ==================================================================
    "standardization": {
        "name": "Standardization (Z-score / StandardScaler)",
        "category": "scaling",
        "purpose": "Mengubah fitur agar memiliki mean = 0 dan std = 1 (zero-mean, unit-variance).",
        "how_it_works_simple": (
            "Centering (kurangi mean) + scaling (bagi std). "
            "Setelah dilakukan: setiap fitur ada di skala yang sama (sekitar -3 sampai +3). "
            "Wajib untuk model yang sensitif terhadap skala: Linear, Logistic, SVM, KNN, Neural Net."
        ),
        "math_formulation": {
            "formula": "z = (x − μ) / σ",
            "mean": "μ = (1/n) Σᵢxᵢ",
            "std": "σ = √( (1/n) Σᵢ(xᵢ − μ)² )",
        },
        "variables": [
            {"symbol": "x", "name": "raw value", "description": "Nilai asli fitur"},
            {"symbol": "μ", "name": "mean", "description": "Rata-rata fitur (di-fit dari TRAIN data)"},
            {"symbol": "σ", "name": "standard deviation", "description": "Standar deviasi fitur (di-fit dari TRAIN data)"},
            {"symbol": "z", "name": "z-score", "description": "Nilai standardized"},
        ],
        "calculation_steps": [
            "1. Pada training: hitung μ dan σ untuk setiap fitur.",
            "2. Transform train: z_train = (x_train − μ) / σ.",
            "3. CRITICAL: simpan μ dan σ.",
            "4. Pada test: z_test = (x_test − μ) / σ  ← gunakan μ, σ DARI TRAIN, bukan dari test (no leakage).",
        ],
        "numerical_example": {
            "description": "Fitur 'umur' dengan μ=35, σ=10",
            "examples": [
                {"raw_value": 35, "z_score": 0.0, "interpretation": "Tepat di rata-rata"},
                {"raw_value": 45, "z_score": 1.0, "interpretation": "1 std di atas rata-rata"},
                {"raw_value": 25, "z_score": -1.0, "interpretation": "1 std di bawah rata-rata"},
                {"raw_value": 65, "z_score": 3.0, "interpretation": "3 std di atas rata-rata (potential outlier)"},
            ],
        },
        "interpretation": (
            "Z-score = berapa std dari mean. |z| > 3 sering dianggap outlier. "
            "Setelah scaling, koefisien linear/logistic regression bisa dibandingkan langsung "
            "untuk inferensi feature importance."
        ),
        "limitations": [
            "Sensitif terhadap outlier (mean dan std dipengaruhi)",
            "Jika ada outlier ekstrem, gunakan RobustScaler",
            "Tidak preserve sparsity (semua nilai jadi non-zero)",
        ],
        "vs_minmax": (
            "StandardScaler: μ=0, σ=1. Lebih cocok untuk data normal-ish, robust ke outlier moderate. "
            "MinMaxScaler: range [0, 1]. Lebih cocok untuk neural network (sigmoid output)."
        ),
    },

    # ==================================================================
    # MIN-MAX SCALING
    # ==================================================================
    "minmax_scaling": {
        "name": "Min-Max Scaling",
        "category": "scaling",
        "purpose": "Scale fitur ke range [0, 1].",
        "math_formulation": {
            "formula": "x' = (x − x_min) / (x_max − x_min)",
            "range_target": "[0, 1] (atau [a, b] dengan transformasi tambahan)",
        },
        "variables": [
            {"symbol": "x_min", "name": "min", "description": "Nilai terkecil di TRAIN data"},
            {"symbol": "x_max", "name": "max", "description": "Nilai terbesar di TRAIN data"},
        ],
        "interpretation": (
            "0 = nilai terkecil di train; 1 = nilai terbesar di train. "
            "Test value > x_max akan menghasilkan x' > 1 (out of range — perlu di-clip)."
        ),
        "limitations": [
            "Sangat sensitif terhadap outlier (1 outlier = x_max ekstrem → semua nilai lain mendekati 0)",
            "Asumsi range training cukup representatif",
        ],
    },

    # ==================================================================
    # ROBUST SCALING
    # ==================================================================
    "robust_scaling": {
        "name": "Robust Scaling",
        "category": "scaling",
        "purpose": "Scale menggunakan median dan IQR — robust terhadap outlier.",
        "math_formulation": {
            "formula": "x' = (x − median) / IQR",
            "iqr": "IQR = Q3 − Q1  (interquartile range)",
        },
        "interpretation": (
            "Mirip standardization tapi tidak terpengaruh outlier ekstrem. "
            "Cocok untuk data dengan banyak outlier (financial, medical lab values)."
        ),
        "limitations": [
            "Tidak menjamin range tertentu",
            "Kurang dikenal stakeholder",
        ],
    },

    # ==================================================================
    # ONE-HOT ENCODING
    # ==================================================================
    "onehot_encoding": {
        "name": "One-Hot Encoding",
        "category": "encoding",
        "purpose": "Convert categorical variable menjadi binary columns (1 untuk satu category, 0 lainnya).",
        "how_it_works_simple": (
            "Misal kolom 'kota' dengan values: Jakarta, Bandung, Surabaya. "
            "Setelah one-hot: 3 kolom baru: is_Jakarta, is_Bandung, is_Surabaya. "
            "Setiap row punya tepat satu '1' (sesuai kotanya), dua '0'."
        ),
        "math_formulation": {
            "transformation": "x ∈ {c₁, c₂, ..., cₖ} → vektor [0,...,1,...,0]ᵀ ∈ {0,1}ᵏ",
            "example": "x = 'Bandung' → [0, 1, 0]ᵀ jika categories = ['Jakarta', 'Bandung', 'Surabaya']",
        },
        "interpretation": (
            "Setiap category jadi independent feature. "
            "Tidak introduce ordering (vs ordinal encoding). "
            "Untuk linear models: drop_first=True untuk hindari multikolinearitas."
        ),
        "limitations": [
            "Curse of dimensionality jika cardinality tinggi (1000 kategori → 1000 kolom)",
            "Sparse matrix (banyak 0)",
            "Tidak handle nilai baru di test (use handle_unknown='ignore')",
        ],
    },

    # ==================================================================
    # ORDINAL ENCODING
    # ==================================================================
    "ordinal_encoding": {
        "name": "Ordinal Encoding",
        "category": "encoding",
        "purpose": "Convert categorical menjadi integer ordering. Hanya untuk kategori dengan urutan natural.",
        "math_formulation": {
            "transformation": "{low, medium, high} → {0, 1, 2}",
            "warning": "JANGAN dipakai untuk kategori tanpa urutan (misal: kota, warna). Akan introduce false ordering.",
        },
        "limitations": [
            "Misleading jika kategori sebenarnya nominal (tanpa order)",
            "Linear models akan memperlakukan 'medium' = 0.5 × 'low' + 0.5 × 'high' (tidak benar)",
        ],
    },

    # ==================================================================
    # MEDIAN IMPUTATION
    # ==================================================================
    "median_imputation": {
        "name": "Median Imputation",
        "category": "imputation",
        "purpose": "Mengisi missing value dengan median fitur.",
        "math_formulation": {
            "formula": "x_imputed = median({xᵢ : xᵢ ≠ NaN})",
        },
        "interpretation": (
            "Robust terhadap outlier (vs mean imputation). "
            "Default sederhana yang aman untuk numerical features."
        ),
        "limitations": [
            "Mengurangi variance dataset (semua missing diisi nilai sama)",
            "Tidak mempertimbangkan korelasi antar fitur",
        ],
        "vs_mean": "Mean: sensitif outlier. Median: robust outlier. Pilih median jika distribution skewed.",
    },

    # ==================================================================
    # KNN IMPUTATION
    # ==================================================================
    "knn_imputation": {
        "name": "KNN Imputation",
        "category": "imputation",
        "purpose": "Isi missing value dengan rata-rata K nearest neighbors.",
        "math_formulation": {
            "formula": "x_missing = (1/k) Σⱼ∈Nₖ(x)  xⱼ",
            "neighbor_distance": "d(xᵢ, xⱼ) = √( Σ over non-missing features (xᵢⱼ − xᵢⱼ)² )",
        },
        "variables": [
            {"symbol": "k", "name": "n_neighbors", "description": "Jumlah neighbor yang dipertimbangkan (typical: 5)"},
            {"symbol": "Nₖ(x)", "name": "k-nearest neighbors", "description": "K sample terdekat ke x"},
        ],
        "interpretation": (
            "Lebih akurat dari median jika ada korelasi antar fitur. "
            "Komputasi lebih mahal — O(n²) untuk small data."
        ),
        "limitations": [
            "Lambat untuk dataset besar",
            "Sensitif terhadap distance metric (gunakan scaling sebelum)",
            "k perlu di-tune",
        ],
    },

    # ==================================================================
    # TF-IDF
    # ==================================================================
    "tfidf": {
        "name": "TF-IDF (Term Frequency-Inverse Document Frequency)",
        "category": "text_vectorization",
        "purpose": "Convert text → numerical vector dengan weight: kata penting di doc tapi jarang di corpus = penting.",
        "how_it_works_simple": (
            "TF (Term Frequency) = berapa sering kata muncul di dokumen ini. "
            "IDF (Inverse Document Frequency) = penalti untuk kata yang muncul di banyak dokumen (kata umum). "
            "TF-IDF = TF × IDF. Hasil: kata distinctive untuk dokumen tertentu mendapat skor tinggi."
        ),
        "math_formulation": {
            "tf": "tf(t, d) = count(t in d) / |d|",
            "idf": "idf(t, D) = log( |D| / |{d ∈ D : t ∈ d}| )",
            "tfidf": "tfidf(t, d, D) = tf(t, d) · idf(t, D)",
        },
        "variables": [
            {"symbol": "t", "name": "term", "description": "Kata/token"},
            {"symbol": "d", "name": "document", "description": "Single dokumen"},
            {"symbol": "D", "name": "corpus", "description": "Kumpulan semua dokumen"},
        ],
        "calculation_steps": [
            "1. Tokenize semua dokumen.",
            "2. Build vocabulary dari semua unique terms.",
            "3. Untuk setiap (term, doc): hitung TF.",
            "4. Untuk setiap term: hitung IDF dari corpus.",
            "5. Multiply: TF-IDF matrix (n_docs × vocab_size, sparse).",
        ],
        "interpretation": (
            "Kata yang muncul di banyak dokumen (misal 'the', 'product') → IDF kecil → TF-IDF kecil. "
            "Kata distinctive di doc tertentu → IDF besar → TF-IDF besar. "
            "Hasil bisa dipakai sebagai feature untuk classification, similarity, clustering text."
        ),
        "limitations": [
            "Bag-of-words: ignore order (tidak tahu 'not good' beda dari 'good')",
            "Tidak handle synonym (good ≠ great untuk model)",
            "Vocabulary fixed di training time",
        ],
    },
}
