"""
ML Model explanations - regression, classification, clustering, anomaly detection.
Uses Unicode math symbols for browser-friendly rendering.
"""

MODEL_EXPLANATIONS = {

    # ==================================================================
    # LINEAR REGRESSION
    # ==================================================================
    "linear_regression": {
        "name": "Linear Regression",
        "category": "regression",
        "purpose": (
            "Memprediksi nilai numerik kontinu dengan mengasumsikan hubungan "
            "LINEAR antara fitur (X) dan target (y)."
        ),
        "why_chosen_template": (
            "Linear Regression dipilih sebagai BASELINE karena: (1) interpretasi koefisien jelas — "
            "setiap unit kenaikan xᵢ menambah ŷ sebesar wᵢ, (2) cepat dan deterministik, "
            "(3) menjadi sanity check sebelum mencoba model kompleks."
        ),
        "why_not_others_template": (
            "Random Forest tidak dipilih karena risk overfitting pada dataset kecil dan "
            "kehilangan interpretabilitas koefisien per fitur. "
            "Polynomial regression tidak dipilih karena meningkatkan kompleksitas tanpa "
            "validasi bahwa hubungan benar-benar non-linear."
        ),
        "how_it_works_simple": (
            "Bayangkan kamu plot semua data point sebagai titik di grafik. "
            "Linear Regression mencari GARIS LURUS yang paling dekat dengan semua titik tersebut. "
            "Garis itu = model. Lalu untuk prediksi: cari posisi x baru di garis, baca nilai y."
        ),
        "math_formulation": {
            "model": "ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b",
            "loss_function": "MSE = (1/n) Σᵢ₌₁ⁿ (yᵢ − ŷᵢ)²",
            "objective": "min(w,b)  (1/n) Σᵢ (yᵢ − (w·xᵢ + b))²",
            "optimization_closed_form": "w* = (XᵀX)⁻¹ Xᵀy",
            "optimization_gradient": "wₜ₊₁ = wₜ − η · ∇MSE(wₜ),   ∇MSE = (2/n) Xᵀ(Xw − y)",
        },
        "variables": [
            {"symbol": "ŷ", "name": "y-hat", "description": "Nilai prediksi model"},
            {"symbol": "y", "name": "y", "description": "Nilai aktual (ground truth)"},
            {"symbol": "xᵢ", "name": "fitur ke-i", "description": "Variabel input ke-i (misal: harga, umur, dll)"},
            {"symbol": "wᵢ", "name": "weight / koefisien", "description": "Bobot pengaruh fitur xᵢ terhadap ŷ"},
            {"symbol": "b", "name": "bias / intercept", "description": "Nilai ŷ ketika semua xᵢ = 0"},
            {"symbol": "n", "name": "n", "description": "Jumlah sampel (rows) dalam dataset"},
            {"symbol": "η", "name": "learning rate", "description": "Ukuran langkah saat update weight (gradient descent)"},
        ],
        "calculation_steps": [
            "1. Setup: kumpulkan dataset (X, y) dengan n baris dan p fitur.",
            "2. Tambahkan kolom konstan 1 untuk intercept: X' = [1 | X].",
            "3. Hitung XᵀX (p+1 × p+1 matrix) dan Xᵀy (p+1 vector).",
            "4. Solve untuk w: w = (XᵀX)⁻¹ Xᵀy. Ini disebut 'normal equation'.",
            "5. Atau gunakan gradient descent: mulai w random, iterasi update wₜ₊₁ = wₜ − η · ∇MSE.",
            "6. Hitung prediksi: ŷ = Xw.",
            "7. Evaluasi: hitung R², RMSE, MAE pada test set.",
        ],
        "numerical_example": {
            "description": "Contoh sederhana: prediksi harga rumah dari luas tanah.",
            "data": [
                {"luas": 50, "harga": 500},
                {"luas": 100, "harga": 950},
                {"luas": 150, "harga": 1500},
            ],
            "computed_w": 9.8,
            "computed_b": 16.7,
            "model_equation": "harga = 9.8 × luas + 16.7",
            "prediction_example": "Untuk luas = 120 m², harga = 9.8 × 120 + 16.7 = Rp 1,192.7 juta",
        },
        "interpretation": (
            "Setiap koefisien wᵢ menjelaskan: 'jika xᵢ naik 1 unit, ŷ naik wᵢ unit (asumsi fitur lain konstan)'. "
            "Tanda positif → korelasi positif; negatif → korelasi negatif. "
            "Magnitude wᵢ menunjukkan seberapa kuat pengaruh fitur tersebut "
            "(asumsi semua fitur sudah di-scale ke skala yang sama)."
        ),
        "limitations": [
            "Hanya menangkap hubungan LINEAR — gagal untuk hubungan kompleks.",
            "Sensitif terhadap outlier karena menggunakan kuadrat error (MSE).",
            "Asumsi residual berdistribusi normal dengan variance konstan (homoscedasticity).",
            "Tidak handle multikolinearitas (fitur yang sangat berkorelasi) dengan baik.",
        ],
        "when_to_use": [
            "Dataset kecil-menengah (<10K rows)",
            "Hubungan diharapkan linear (pengalaman domain)",
            "Butuh interpretabilitas tinggi (koefisien per fitur)",
            "Sebagai baseline sebelum mencoba model kompleks",
        ],
        "when_not_to_use": [
            "Data tabular besar dengan banyak interaksi non-linear → pakai Random Forest / Gradient Boosting",
            "Banyak outlier → pakai Huber Regression atau RANSAC",
            "Multikolinearitas tinggi → pakai Ridge atau Lasso",
        ],
    },

    # ==================================================================
    # RIDGE REGRESSION
    # ==================================================================
    "ridge_regression": {
        "name": "Ridge Regression (L2 regularization)",
        "category": "regression",
        "purpose": (
            "Linear Regression yang ditambah penalti agar koefisien tidak terlalu besar. "
            "Mengatasi overfitting + multikolinearitas."
        ),
        "why_chosen_template": (
            "Ridge dipilih karena dataset memiliki fitur-fitur yang berkorelasi tinggi (multikolinearitas) "
            "atau jumlah fitur dekat dengan jumlah sample. Penalti L2 menstabilkan koefisien."
        ),
        "why_not_others_template": (
            "Linear Regression biasa tidak dipilih karena akan menghasilkan koefisien yang TIDAK STABIL "
            "saat ada multikolinearitas. "
            "Lasso tidak dipilih karena Lasso suka membuat banyak koefisien EXACTLY ZERO yang bisa "
            "menghapus fitur penting; Ridge lebih moderat."
        ),
        "how_it_works_simple": (
            "Sama seperti Linear Regression, tapi memberi 'denda' jika koefisien terlalu besar. "
            "Hasilnya: model tetap akurat tapi koefisien lebih kecil dan stabil. "
            "Cocok ketika fitur-fitur saling terkait (misal: harga & luas tanah)."
        ),
        "math_formulation": {
            "model": "ŷ = w₁x₁ + ... + wₙxₙ + b   (sama seperti Linear)",
            "loss_function": "L_ridge = MSE + α · ||w||²₂ = (1/n) Σᵢ(yᵢ−ŷᵢ)² + α · Σⱼwⱼ²",
            "objective": "min(w,b)  L_ridge",
            "optimization_closed_form": "w* = (XᵀX + αI)⁻¹ Xᵀy",
        },
        "variables": [
            {"symbol": "α", "name": "alpha (regularization strength)", "description": "Seberapa kuat 'denda' untuk koefisien besar. α=0 → Linear biasa; α=∞ → semua w mendekati 0"},
            {"symbol": "||w||²₂", "name": "L2 norm squared", "description": "Σwⱼ² — jumlah kuadrat semua koefisien"},
            {"symbol": "I", "name": "identity matrix", "description": "Matriks identitas (1 di diagonal, 0 lainnya)"},
        ],
        "calculation_steps": [
            "1. Pilih α (typical: 0.1, 1, 10, 100). Tune dengan cross-validation.",
            "2. Solve: w = (XᵀX + αI)⁻¹ Xᵀy.",
            "3. Penambahan αI di diagonal membuat matriks selalu invertible (mengatasi multikolinearitas).",
            "4. Semakin besar α, semakin kecil semua wⱼ — model lebih sederhana.",
        ],
        "interpretation": (
            "Koefisien Ridge selalu LEBIH KECIL dari Linear biasa (shrunk). "
            "Tidak pernah tepat 0 (berbeda dengan Lasso). "
            "Cocok ketika kamu yakin SEMUA fitur relevan tapi mau menstabilkan estimasi."
        ),
        "limitations": [
            "Tidak melakukan feature selection (semua fitur tetap ada)",
            "α perlu di-tune dengan CV",
            "Masih mengasumsikan hubungan linear",
        ],
        "vs_linear": (
            "Linear: w = (XᵀX)⁻¹ Xᵀy. Ridge: w = (XᵀX + αI)⁻¹ Xᵀy. "
            "Penambahan αI membuat solusi STABIL terhadap multikolinearitas dan overfitting."
        ),
    },

    # ==================================================================
    # LASSO REGRESSION
    # ==================================================================
    "lasso_regression": {
        "name": "Lasso Regression (L1 regularization)",
        "category": "regression",
        "purpose": "Linear Regression + L1 penalty — secara otomatis melakukan feature selection.",
        "how_it_works_simple": (
            "Seperti Ridge tapi 'denda'-nya berbeda. Akibatnya: banyak koefisien menjadi TEPAT NOL "
            "→ secara otomatis menghapus fitur yang tidak penting. Cocok ketika kamu punya banyak fitur "
            "tapi tidak yakin mana yang relevan."
        ),
        "math_formulation": {
            "model": "ŷ = w₁x₁ + ... + wₙxₙ + b",
            "loss_function": "L_lasso = MSE + α · ||w||₁ = (1/n) Σᵢ(yᵢ−ŷᵢ)² + α · Σⱼ|wⱼ|",
            "objective": "min(w,b)  L_lasso",
            "optimization": "Coordinate descent atau LARS (no closed-form karena |·| tidak differentiable di 0)",
        },
        "variables": [
            {"symbol": "||w||₁", "name": "L1 norm", "description": "Σ|wⱼ| — jumlah absolute semua koefisien"},
            {"symbol": "α", "name": "alpha", "description": "Regularization strength; α besar → lebih banyak koefisien jadi 0"},
        ],
        "interpretation": (
            "Koefisien yang TEPAT NOL artinya fitur tersebut DIBUANG dari model. "
            "Lasso menghasilkan SPARSE model — hanya beberapa fitur yang berperan."
        ),
        "limitations": [
            "Jika dua fitur sangat berkorelasi, Lasso suka memilih SALAH SATU dan membuat yang lain 0 (arbitrary)",
            "Tidak stabil seperti Ridge ketika n_features > n_samples",
        ],
        "vs_ridge": (
            "Ridge (L2): wⱼ menyusut tapi jarang tepat 0 → smooth shrinkage. "
            "Lasso (L1): banyak wⱼ menjadi tepat 0 → feature selection. "
            "Elastic Net = kombinasi keduanya."
        ),
    },

    # ==================================================================
    # LOGISTIC REGRESSION
    # ==================================================================
    "logistic_regression": {
        "name": "Logistic Regression",
        "category": "classification",
        "purpose": "Memprediksi PROBABILITY suatu kelas (binary atau multi-class).",
        "why_chosen_template": (
            "Logistic Regression dipilih karena (1) memberikan output probability yang sudah terkalibrasi baik "
            "(berbeda dengan SVM), (2) interpretasi koefisien jelas, (3) cepat dan robust."
        ),
        "how_it_works_simple": (
            "Mirip Linear Regression tapi outputnya 'dijepit' antara 0 dan 1 menggunakan fungsi sigmoid, "
            "sehingga bisa diinterpretasi sebagai probability. "
            "Decision boundary: jika prob ≥ 0.5 → class 1, else class 0."
        ),
        "math_formulation": {
            "linear_combination": "z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b",
            "sigmoid": "σ(z) = 1 / (1 + e⁻ᶻ)",
            "model": "p(y=1|x) = σ(z) = 1 / (1 + exp(−(w·x + b)))",
            "loss_function": "Log-Loss (Binary Cross-Entropy):  L = −(1/n) Σᵢ [yᵢ·log(pᵢ) + (1−yᵢ)·log(1−pᵢ)]",
            "objective": "min(w,b)  L  (max likelihood)",
            "optimization": "Gradient descent atau IRLS (Iteratively Reweighted Least Squares)",
        },
        "variables": [
            {"symbol": "z", "name": "logit", "description": "Linear combination dari fitur dan weights"},
            {"symbol": "σ(z)", "name": "sigmoid", "description": "Transformasi z → probability di [0, 1]"},
            {"symbol": "p", "name": "probability", "description": "P(y=1|x) — probability bahwa instance termasuk kelas 1"},
            {"symbol": "yᵢ", "name": "label", "description": "Ground truth label (0 atau 1)"},
        ],
        "calculation_steps": [
            "1. Hitung z = w·x + b untuk setiap sample.",
            "2. Lewatkan z melalui sigmoid: p = 1/(1+e⁻ᶻ) → probability.",
            "3. Loss per sample: −[y·log(p) + (1−y)·log(1−p)].",
            "4. Total loss: rata-rata loss semua sample.",
            "5. Update w dan b dengan gradient descent untuk minimize loss.",
            "6. Prediksi: jika p ≥ 0.5 → class 1; threshold bisa di-tune untuk imbalanced data.",
        ],
        "numerical_example": {
            "description": "Misal seseorang dengan z = 2.3",
            "computation": "p = 1 / (1 + e⁻²·³) = 1 / (1 + 0.1) = 0.91",
            "interpretation": "91% probability termasuk class 1 → prediksi class 1",
        },
        "interpretation": (
            "Setiap exp(wᵢ) = ODDS RATIO. Misal exp(w₁) = 1.5 berarti naik 1 unit x₁ membuat ODDS class 1 "
            "naik 50%. Output p ∈ [0,1] → bisa di-rank atau di-threshold."
        ),
        "limitations": [
            "Decision boundary linear — tidak bisa pisahkan kelas yang non-linear",
            "Sensitif terhadap fitur yang berkorelasi tinggi (multikolinearitas)",
            "Membutuhkan banyak data untuk multi-class",
        ],
    },

    # ==================================================================
    # RANDOM FOREST
    # ==================================================================
    "random_forest": {
        "name": "Random Forest",
        "category": "ensemble",
        "purpose": (
            "Ensemble dari banyak decision tree — voting (klasifikasi) atau averaging (regresi). "
            "Sangat akurat untuk tabular data dengan hubungan non-linear."
        ),
        "why_chosen_template": (
            "Random Forest dipilih karena: (1) handle non-linearity tanpa feature engineering, "
            "(2) robust terhadap outlier dan missing values, (3) provides feature importance, "
            "(4) tidak butuh feature scaling."
        ),
        "how_it_works_simple": (
            "Bayangkan kamu tanya 100 dokter berbeda untuk diagnosis pasien yang sama. "
            "Tiap dokter punya 'pengalaman' berbeda (training subset berbeda + fitur berbeda yang dilihat). "
            "Diagnosis akhir = MAYORITAS suara (klasifikasi) atau RATA-RATA prediksi (regresi). "
            "Ide: banyak tree yang lemah → satu ensemble yang kuat."
        ),
        "math_formulation": {
            "ensemble_classification": "ŷ = mode { Tᵢ(x) }, i=1..M  (majority vote)",
            "ensemble_regression": "ŷ = (1/M) Σᵢ₌₁ᴹ Tᵢ(x)  (averaging)",
            "feature_importance": "Importance(j) = Σ over all trees Σ over splits using j   ΔImpurity",
            "bootstrap": "Setiap Tᵢ dilatih pada sample acak dengan replacement (bagging)",
            "feature_subsampling": "Setiap split mempertimbangkan √p atau log₂(p) fitur acak",
        },
        "variables": [
            {"symbol": "M", "name": "n_estimators", "description": "Jumlah tree dalam forest (typical: 100-500)"},
            {"symbol": "Tᵢ(x)", "name": "tree i", "description": "Decision tree ke-i memberikan prediksi untuk x"},
            {"symbol": "p", "name": "n_features", "description": "Jumlah total fitur"},
            {"symbol": "ΔImpurity", "name": "decrease impurity", "description": "Penurunan Gini (klasifikasi) atau MSE (regresi) saat split menggunakan fitur tersebut"},
        ],
        "calculation_steps": [
            "1. Untuk i = 1 sampai M:",
            "   a. Sample data dengan replacement (bootstrap) → ukuran sama dengan training set.",
            "   b. Bangun decision tree pada sample tersebut.",
            "   c. Setiap split: pilih fitur terbaik dari subset random √p fitur.",
            "   d. Tumbuh tree sampai max_depth atau leaf cukup pure.",
            "2. Untuk prediksi: jalankan x melalui semua M tree.",
            "3. Klasifikasi: majority vote. Regresi: rata-rata.",
        ],
        "interpretation": (
            "Feature importance menunjukkan fitur mana yang paling sering & efektif digunakan untuk split. "
            "Karena ensemble, prediksi lebih stabil daripada single tree. "
            "Probability untuk klasifikasi = proporsi tree yang vote untuk class tersebut."
        ),
        "limitations": [
            "Bias terhadap fitur dengan banyak unique values (misal: ID columns)",
            "Memori besar (menyimpan banyak tree)",
            "Lebih lambat di prediction time vs single model",
            "Probability tidak terkalibrasi sempurna — pakai CalibratedClassifierCV jika butuh probability akurat",
        ],
        "voting_example": {
            "description": "100 tree memprediksi class untuk satu sample:",
            "votes": {"class_A": 73, "class_B": 27},
            "result": "Prediksi: class_A (mayoritas), Probability(A) = 0.73",
        },
    },

    # ==================================================================
    # GRADIENT BOOSTING
    # ==================================================================
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "category": "ensemble",
        "purpose": "Sequential ensemble — setiap tree baru memperbaiki kesalahan tree sebelumnya.",
        "how_it_works_simple": (
            "Berbeda dengan Random Forest yang independen, Gradient Boosting BERGURUTAN: "
            "Tree-1 prediksi → ada error → Tree-2 belajar memprediksi error tersebut → Tree-3 belajar dari error gabungan, dst. "
            "Hasilnya: sangat akurat tapi mudah overfit jika tidak hati-hati."
        ),
        "math_formulation": {
            "iterative_model": "F_m(x) = F_{m-1}(x) + η · h_m(x)",
            "fit_residual": "h_m = argmin_h Σᵢ L(yᵢ, F_{m-1}(xᵢ) + h(xᵢ))",
            "gradient": "h_m fits negative gradient: −∂L/∂F at F_{m-1}",
            "loss_classification": "L = log-loss (sama seperti logistic regression)",
            "loss_regression": "L = MSE atau MAE",
        },
        "variables": [
            {"symbol": "F_m", "name": "ensemble setelah m iterasi", "description": "Total prediksi dari m tree pertama"},
            {"symbol": "h_m", "name": "weak learner ke-m", "description": "Tree kecil baru untuk iterasi ini"},
            {"symbol": "η", "name": "learning rate", "description": "Seberapa besar kontribusi setiap tree (typical: 0.01-0.1)"},
            {"symbol": "M", "name": "n_estimators", "description": "Total iterasi/tree"},
        ],
        "calculation_steps": [
            "1. Inisialisasi: F₀(x) = nilai konstan (rata-rata y untuk regresi).",
            "2. Untuk m = 1..M:",
            "   a. Hitung pseudo-residual: rᵢ = −∂L/∂F at F_{m-1}",
            "   b. Latih tree h_m untuk memprediksi rᵢ.",
            "   c. Update: F_m(x) = F_{m-1}(x) + η · h_m(x)",
            "3. Prediksi akhir: F_M(x).",
        ],
        "interpretation": (
            "Setiap iterasi mengurangi sisa error sedikit demi sedikit. "
            "Learning rate kecil + banyak tree = hasil halus, sedikit overfit. "
            "Learning rate besar + sedikit tree = cepat tapi rentan overshoot."
        ),
        "limitations": [
            "Lebih lambat training daripada Random Forest (sequential, bukan parallel)",
            "Sensitif terhadap noise (akan fit error jika tidak di-regularize)",
            "n_estimators besar tanpa early stopping → overfitting",
        ],
        "vs_random_forest": (
            "RF: parallel, independen, vote/average — RESISTAN terhadap overfitting. "
            "GB: sequential, dependen, fit residual — POTENSI lebih akurat tapi mudah overfit."
        ),
    },

    # ==================================================================
    # K-MEANS
    # ==================================================================
    "kmeans": {
        "name": "K-Means Clustering",
        "category": "clustering",
        "purpose": "Mengelompokkan data ke k cluster berdasarkan kedekatan ke centroid.",
        "how_it_works_simple": (
            "Mirip game: kasih k 'kapten tim' (centroid). Setiap point gabung tim terdekat. "
            "Setelah semua point bergabung, kapten pindah ke pusat anggota timnya. Ulangi sampai kapten tidak bergerak lagi."
        ),
        "math_formulation": {
            "objective": "min Σᵢ₌₁ⁿ Σⱼ₌₁ᵏ wᵢⱼ · ||xᵢ − μⱼ||²",
            "distance": "Euclidean: d(x, μ) = √( Σⱼ(xⱼ − μⱼ)² )",
            "centroid_update": "μⱼ = (1/|Cⱼ|) Σ x∈Cⱼ  x",
            "assignment": "wᵢⱼ = 1 jika xᵢ paling dekat ke μⱼ, else 0",
        },
        "variables": [
            {"symbol": "k", "name": "n_clusters", "description": "Jumlah cluster (harus ditentukan dulu!)"},
            {"symbol": "μⱼ", "name": "centroid cluster j", "description": "Pusat cluster ke-j"},
            {"symbol": "Cⱼ", "name": "cluster j", "description": "Kumpulan points yang termasuk cluster j"},
            {"symbol": "||·||", "name": "Euclidean norm", "description": "Jarak Euclidean"},
        ],
        "calculation_steps": [
            "1. Pilih k (n_clusters) dan inisialisasi μ₁..μₖ random (atau dengan k-means++).",
            "2. Assignment step: tiap xᵢ ditugaskan ke cluster μⱼ terdekat (Euclidean).",
            "3. Update step: hitung centroid baru = rata-rata semua point dalam cluster.",
            "4. Ulangi step 2-3 sampai centroid tidak berubah signifikan (konvergen).",
        ],
        "interpretation": (
            "Setiap cluster diwakili oleh centroid-nya. Inertia (sum of squared distances ke centroid) "
            "menunjukkan seberapa kompak cluster — semakin kecil semakin baik (sampai k tertentu)."
        ),
        "limitations": [
            "Harus tentukan k dulu (gunakan elbow method atau silhouette score)",
            "Asumsi cluster spherical dan ukuran sama — gagal untuk cluster bentuk aneh",
            "Sensitif terhadap inisialisasi → pakai k-means++ atau multiple restarts",
            "Sensitif terhadap outlier (centroid bergeser)",
        ],
    },

    # ==================================================================
    # DBSCAN
    # ==================================================================
    "dbscan": {
        "name": "DBSCAN (Density-Based Spatial Clustering)",
        "category": "clustering",
        "purpose": (
            "Mengelompokkan data berdasarkan KEPADATAN — tidak perlu tentukan k. "
            "Otomatis identifikasi outlier."
        ),
        "how_it_works_simple": (
            "Cari kelompok titik yang BERDEKATAN secara padat. "
            "Titik yang terisolasi (jauh dari kelompok manapun) = outlier. "
            "Bagus untuk shape cluster bebas (tidak harus bulat seperti K-Means)."
        ),
        "math_formulation": {
            "neighborhood": "N_ε(x) = { p ∈ D : dist(x, p) ≤ ε }",
            "core_point": "x adalah core point jika |N_ε(x)| ≥ minPts",
            "density_reachable": "y density-reachable dari x jika ada chain x = x₀, x₁, ..., xₘ = y dimana xᵢ₊₁ ∈ N_ε(xᵢ) dan xᵢ adalah core",
            "cluster": "Maximal set of density-connected points",
        },
        "variables": [
            {"symbol": "ε (eps)", "name": "epsilon", "description": "Radius neighborhood — seberapa dekat dianggap 'tetangga'"},
            {"symbol": "minPts", "name": "min_samples", "description": "Minimum jumlah tetangga untuk jadi core point"},
            {"symbol": "N_ε(x)", "name": "neighborhood", "description": "Semua titik dalam radius ε dari x"},
        ],
        "calculation_steps": [
            "1. Untuk setiap point x, hitung N_ε(x).",
            "2. Klasifikasi: x = CORE jika |N_ε(x)| ≥ minPts; BORDER jika ada di N_ε(core) tapi bukan core; NOISE jika lainnya.",
            "3. Mulai dari core point, expand ke semua density-reachable points → satu cluster.",
            "4. Ulangi untuk core point yang belum ter-cluster.",
            "5. Sisa points = noise / outlier (label -1).",
        ],
        "interpretation": (
            "Output: setiap point dapat label cluster (0, 1, 2, ...) atau -1 (noise). "
            "Jumlah cluster ditemukan secara OTOMATIS. "
            "Outlier tidak dipaksa masuk cluster — ditandai eksplisit."
        ),
        "limitations": [
            "Sensitif terhadap parameter ε dan minPts",
            "Gagal jika cluster memiliki density yang sangat berbeda",
            "Skala fitur penting (gunakan standardization sebelum DBSCAN)",
        ],
        "vs_kmeans": (
            "K-Means: butuh k, hanya cluster bulat, tidak detect outlier. "
            "DBSCAN: tidak butuh k, cluster bentuk apa saja, outlier explicit."
        ),
    },

    # ==================================================================
    # ISOLATION FOREST
    # ==================================================================
    "isolation_forest": {
        "name": "Isolation Forest",
        "category": "anomaly_detection",
        "purpose": "Mendeteksi anomali berdasarkan SEBERAPA MUDAH titik diisolasi.",
        "how_it_works_simple": (
            "Anomali = titik yang 'aneh' = mudah dipisahkan dari yang lain. "
            "Bayangkan kamu acak-acak split data: anomali bisa diisolasi dengan SEDIKIT split, "
            "sedangkan titik normal butuh banyak split. "
            "Isolation Forest membangun banyak random tree dan menghitung 'kedalaman isolasi' rata-rata."
        ),
        "math_formulation": {
            "anomaly_score": "s(x, n) = 2^(−E(h(x)) / c(n))",
            "expected_path_length": "E(h(x)) = rata-rata path length di semua tree",
            "normalization_constant": "c(n) = 2H(n−1) − 2(n−1)/n,   H(i) ≈ ln(i) + 0.5772",
        },
        "variables": [
            {"symbol": "h(x)", "name": "path length", "description": "Kedalaman dari root sampai isolasi titik x"},
            {"symbol": "E(h(x))", "name": "expected h(x)", "description": "Rata-rata path length di semua tree"},
            {"symbol": "c(n)", "name": "normalization", "description": "Konstanta normalisasi untuk dataset ukuran n"},
            {"symbol": "s(x, n)", "name": "anomaly score", "description": "Skor anomali — mendekati 1 = sangat anomali; mendekati 0.5 = normal"},
        ],
        "calculation_steps": [
            "1. Bangun banyak random tree (typical: 100). Setiap tree pakai sample subset.",
            "2. Setiap tree: pilih fitur acak → split di nilai acak dalam range fitur tersebut. Ulangi sampai isolasi.",
            "3. Hitung path length h(x) untuk x di setiap tree.",
            "4. Rata-rata: E(h(x)).",
            "5. Anomaly score: s = 2^(−E(h)/c(n)). Score > 0.5 → anomali.",
        ],
        "interpretation": (
            "Score mendekati 1 → sangat anomali (mudah diisolasi). "
            "Score 0.5 → tidak yakin / normal. "
            "Score < 0.5 → sangat normal."
        ),
        "limitations": [
            "Tidak bisa explain 'mengapa' suatu titik anomali (black-box)",
            "Tidak handle dengan baik anomali dalam cluster padat",
            "Asumsi anomali = 'terisolasi' (tidak selalu benar untuk semua kasus)",
        ],
    },

    # ==================================================================
    # ARIMA
    # ==================================================================
    "arima": {
        "name": "ARIMA (AutoRegressive Integrated Moving Average)",
        "category": "forecasting",
        "purpose": "Forecast time series dengan menggunakan nilai masa lalu dan error masa lalu.",
        "how_it_works_simple": (
            "Bayangkan kamu mau prediksi temperatur besok. ARIMA: "
            "AR — gunakan temperatur beberapa hari terakhir. "
            "I — kalau data tidak stasioner (ada tren), differencing = ambil selisih (membuat stasioner). "
            "MA — koreksi pakai error prediksi sebelumnya. "
            "Gabungkan ketiganya = ARIMA(p,d,q)."
        ),
        "math_formulation": {
            "ar_part": "AR(p): yₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ",
            "ma_part": "MA(q): yₜ = c + εₜ + θ₁εₜ₋₁ + ... + θᵩεₜ₋ᵩ",
            "differencing": "I(d): yₜ' = yₜ − yₜ₋₁  (d kali untuk membuat stasioner)",
            "full_arima": "ARIMA(p,d,q): φ(B)(1−B)ᵈ yₜ = θ(B)εₜ",
        },
        "variables": [
            {"symbol": "p", "name": "AR order", "description": "Berapa banyak lag yₜ₋ᵢ digunakan"},
            {"symbol": "d", "name": "Integration order", "description": "Berapa kali differencing untuk stasioneritas"},
            {"symbol": "q", "name": "MA order", "description": "Berapa banyak lag error εₜ₋ⱼ digunakan"},
            {"symbol": "φᵢ", "name": "AR coefficients", "description": "Bobot untuk lag yᵢ"},
            {"symbol": "θⱼ", "name": "MA coefficients", "description": "Bobot untuk lag error"},
            {"symbol": "εₜ", "name": "white noise", "description": "Error random ~ N(0, σ²)"},
        ],
        "calculation_steps": [
            "1. Cek stasioneritas (ADF test). Jika tidak stasioner, lakukan differencing d kali.",
            "2. Identifikasi p dan q dengan ACF/PACF plots.",
            "3. Estimasi koefisien φ dan θ dengan MLE (maximum likelihood).",
            "4. Forecast: gunakan model untuk prediksi h step ke depan.",
            "5. Validasi: cek residual harus white noise (Ljung-Box test).",
        ],
        "interpretation": (
            "ARIMA(1,1,1) artinya: 1 AR term, 1 differencing, 1 MA term. "
            "Forecast dilengkapi dengan confidence interval — kepastian prediksi turun seiring jarak waktu."
        ),
        "limitations": [
            "Asumsi linearitas dalam time series",
            "Tidak handle seasonality langsung (pakai SARIMA)",
            "Sensitif terhadap outlier",
            "Tidak natively handle external regressors (pakai ARIMAX)",
        ],
    },

    # ==================================================================
    # KAPLAN-MEIER
    # ==================================================================
    "kaplan_meier": {
        "name": "Kaplan-Meier Estimator",
        "category": "survival_analysis",
        "purpose": "Estimasi survival probability over time dengan handling censored observations.",
        "how_it_works_simple": (
            "Untuk setiap waktu kejadian (event), hitung probability bertahan dari titik tersebut. "
            "Total survival probability = produk dari semua probability tersebut. "
            "Kelebihan: handle subjects yang keluar studi sebelum event (censored)."
        ),
        "math_formulation": {
            "estimator": "Ŝ(t) = ∏ᵢ:tᵢ≤t  (1 − dᵢ/nᵢ)",
            "median_survival": "median = min{t : Ŝ(t) ≤ 0.5}",
            "log_rank_test": "Z² = Σ (Oᵢ − Eᵢ)² / Vᵢ  ~ χ²(k−1)",
        },
        "variables": [
            {"symbol": "Ŝ(t)", "name": "estimated survival", "description": "Estimasi probability survive sampai waktu t"},
            {"symbol": "tᵢ", "name": "event time", "description": "Waktu terjadinya event ke-i"},
            {"symbol": "dᵢ", "name": "deaths", "description": "Jumlah event di waktu tᵢ"},
            {"symbol": "nᵢ", "name": "at risk", "description": "Jumlah subject yang masih at risk tepat sebelum tᵢ"},
        ],
        "calculation_steps": [
            "1. Urutkan event times: t₁ < t₂ < ... < tₘ.",
            "2. Pada setiap tᵢ: hitung dᵢ (events) dan nᵢ (at risk).",
            "3. Probability survive lewat tᵢ = 1 − dᵢ/nᵢ.",
            "4. Total survival sampai t = produk dari semua probability tersebut.",
            "5. Plot menghasilkan step function — turun di setiap event.",
        ],
        "interpretation": (
            "Kurva turun di setiap event. Tick mark = censoring. "
            "Median survival = waktu di mana 50% subject masih bertahan."
        ),
        "limitations": [
            "Asumsi censoring non-informative",
            "Tidak adjust untuk covariates (pakai Cox regression untuk itu)",
            "Confidence interval bisa lebar untuk sample kecil",
        ],
    },

    # ==================================================================
    # PLATT SCALING (CALIBRATION)
    # ==================================================================
    "platt_scaling": {
        "name": "Platt Scaling (Sigmoid Calibration)",
        "category": "calibration",
        "purpose": "Mengkalibrasi probability output dari classifier (terutama RF, SVM) agar reliable.",
        "how_it_works_simple": (
            "Random Forest output 'probability' tapi sering tidak akurat — misal: model bilang 80% padahal "
            "actual rate hanya 60%. Platt scaling melatih sigmoid kecil untuk MAP-PING raw output → "
            "true probability menggunakan validation set."
        ),
        "math_formulation": {
            "calibration_function": "P(y=1 | f) = 1 / (1 + exp(A·f + B))",
            "fit": "Estimate A, B dengan MLE pada (f_i, y_i) dari validation set",
        },
        "variables": [
            {"symbol": "f", "name": "raw classifier output", "description": "Output mentah classifier (decision function atau probability)"},
            {"symbol": "A, B", "name": "calibration parameters", "description": "Parameter sigmoid yang di-fit"},
        ],
        "interpretation": (
            "Setelah kalibrasi, prediksi 70% benar-benar berarti 70% kemungkinan. "
            "Critical untuk decision-making (clinical, fraud detection, etc.)."
        ),
        "limitations": [
            "Asumsi sigmoid relationship — tidak fit semua kasus (gunakan isotonic untuk lebih flexible)",
            "Butuh held-out validation set untuk fit kalibrasi",
        ],
    },
}
