"""
PLAIN-LANGUAGE MATHEMATICAL EXPLANATIONS
==========================================
Setiap entry mengikuti spec user:
- real_math_explanation: rumus dijelaskan dalam BAHASA NYATA, bukan sekadar simbol
- variable_meaning: setiap komponen rumus dijelaskan secara konkret
- step_by_step_calculation: contoh angka sederhana, dihitung tahap demi tahap
- result_interpretation: cara membaca hasil
- business_meaning: dampak praktis untuk user

Tujuan: terasa seperti guru data science menjelaskan, bukan textbook simbol.
"""

PLAIN_EXPLANATIONS = {

    # ==================================================================
    # MEAN
    # ==================================================================
    "mean": {
        "real_math_explanation": (
            "Mean adalah nilai RATA-RATA. Cara menghitungnya: jumlahkan semua nilai data, "
            "lalu bagi dengan jumlah data."
        ),
        "variable_meaning": (
            "Σ (sigma) artinya 'jumlahkan semuanya'. n adalah banyaknya data. "
            "Hasilnya adalah satu angka yang mewakili 'nilai khas' dari seluruh data."
        ),
        "step_by_step_calculation": [
            "Contoh data: 10, 20, 30",
            "Langkah 1: Jumlahkan semua data → 10 + 20 + 30 = 60",
            "Langkah 2: Hitung jumlah data → ada 3 data",
            "Langkah 3: Bagi total dengan jumlah data → 60 ÷ 3 = 20",
            "Hasil: Mean = 20",
        ],
        "result_interpretation": (
            "Mean = 20 berarti nilai rata-rata data adalah 20. Setengah data biasanya "
            "berada di bawah 20 dan setengah di atasnya (jika distribusi simetris)."
        ),
        "business_meaning": (
            "Dalam analisis bisnis: kalau Mean revenue = Rp 50jt, ekspektasi revenue per "
            "transaksi adalah Rp 50jt. Tapi waspada — Mean sensitif terhadap outlier "
            "(satu transaksi Rp 1miliar bisa membuat Mean sangat misleading). "
            "Untuk distribusi miring (skewed), gunakan MEDIAN."
        ),
    },

    # ==================================================================
    # MSE / RMSE
    # ==================================================================
    "rmse": {
        "real_math_explanation": (
            "RMSE = Root Mean Squared Error. Artinya: AKAR DARI rata-rata SELISIH KUADRAT antara "
            "nilai aktual dan nilai prediksi.\n\n"
            "Cara menghitungnya: untuk setiap data, hitung SELISIH antara nilai aktual dan "
            "nilai prediksi. Kuadratkan selisih itu (agar negatif jadi positif dan error besar "
            "di-amplifikasi). Jumlahkan semua kuadrat selisih. Bagi dengan jumlah data → ini disebut MSE. "
            "Lalu akar kuadratkan agar hasilnya kembali ke satuan asli → ini RMSE."
        ),
        "variable_meaning": (
            "y = nilai aktual (yang sebenarnya terjadi). "
            "ŷ (y-topi) = nilai prediksi dari model. "
            "(y − ŷ) = error / selisih per data. "
            "n = jumlah data. "
            "Hasil RMSE selalu dalam satuan yang SAMA dengan target — kalau target = harga (Rupiah), "
            "RMSE juga dalam Rupiah."
        ),
        "step_by_step_calculation": [
            "Contoh: 4 prediksi harga rumah",
            "Aktual: [100, 200, 300, 400] (juta)",
            "Prediksi: [110, 190, 320, 380] (juta)",
            "Langkah 1: Hitung selisih per data → (100-110)=-10, (200-190)=10, (300-320)=-20, (400-380)=20",
            "Langkah 2: Kuadratkan setiap selisih → (-10)²=100, (10)²=100, (-20)²=400, (20)²=400",
            "Langkah 3: Jumlahkan semua → 100 + 100 + 400 + 400 = 1000",
            "Langkah 4: Bagi dengan jumlah data → 1000 ÷ 4 = 250 (ini MSE)",
            "Langkah 5: Akar kuadratkan → √250 ≈ 15.81",
            "Hasil: RMSE ≈ 15.81 juta",
        ],
        "result_interpretation": (
            "RMSE = 15.81 juta artinya RATA-RATA prediksi model meleset sekitar 15.81 juta dari "
            "harga aktual. Semakin kecil semakin baik. RMSE 0 = prediksi sempurna."
        ),
        "business_meaning": (
            "Untuk forecasting penjualan: RMSE = Rp 5jt artinya rata-rata error prediksi mingguan "
            "Rp 5jt — bisa diterima jika revenue mingguan Rp 100jt (5% error), tapi mengkhawatirkan "
            "jika revenue mingguan Rp 10jt (50% error). RMSE harus dibandingkan dengan SKALA target "
            "untuk mengambil keputusan deploy/tidak."
        ),
    },

    # ==================================================================
    # MAE
    # ==================================================================
    "mae": {
        "real_math_explanation": (
            "MAE = Mean Absolute Error = rata-rata SELISIH ABSOLUTE (tanpa peduli tanda + atau −) "
            "antara nilai aktual dan prediksi.\n\n"
            "Bedanya dengan RMSE: MAE tidak mengkuadratkan selisih, jadi error besar tidak di-amplifikasi. "
            "Lebih ramah terhadap outlier."
        ),
        "variable_meaning": (
            "|y − ŷ| = absolute value selisih (selalu positif, baik error +10 maupun −10 dihitung 10). "
            "n = jumlah data."
        ),
        "step_by_step_calculation": [
            "Contoh sama: aktual [100, 200, 300, 400], prediksi [110, 190, 320, 380]",
            "Langkah 1: Selisih per data → -10, 10, -20, 20",
            "Langkah 2: Ambil absolute → 10, 10, 20, 20",
            "Langkah 3: Jumlahkan → 60",
            "Langkah 4: Bagi dengan jumlah data → 60 ÷ 4 = 15",
            "Hasil: MAE = 15",
        ],
        "result_interpretation": (
            "MAE = 15 berarti rata-rata prediksi MELESET 15 unit dari nilai aktual. "
            "Lebih intuitif daripada RMSE karena tidak terdistorsi oleh kuadrat."
        ),
        "business_meaning": (
            "Pakai MAE jika kamu ingin tahu 'rata-rata prediksi melenceng berapa'. "
            "Pakai RMSE jika error besar JAUH lebih buruk daripada error kecil "
            "(misal: forecasting stok — kekurangan 100 unit jauh lebih buruk dari 5 kali kekurangan 20 unit)."
        ),
    },

    # ==================================================================
    # R-SQUARED
    # ==================================================================
    "r_squared": {
        "real_math_explanation": (
            "R² (R-squared) mengukur seberapa banyak VARIASI dalam data target yang bisa DIJELASKAN "
            "oleh model.\n\n"
            "Logikanya: anggap kamu punya 'baseline' yang selalu memprediksi RATA-RATA target. "
            "R² bertanya: seberapa BAIK model kamu dibanding baseline ini?\n\n"
            "Rumus pakai dua angka:\n"
            "1. SS_res = jumlah kuadrat selisih antara nilai aktual dan prediksi MODEL (residual sum of squares)\n"
            "2. SS_tot = jumlah kuadrat selisih antara nilai aktual dan RATA-RATA (total sum of squares)\n\n"
            "R² = 1 − (SS_res ÷ SS_tot)"
        ),
        "variable_meaning": (
            "SS_res kecil = model bagus (prediksi dekat dengan aktual). "
            "SS_tot tetap (sifat data, tidak tergantung model). "
            "Jika SS_res = 0 → R² = 1 (perfect). "
            "Jika SS_res = SS_tot (model sama dengan baseline) → R² = 0. "
            "Jika SS_res > SS_tot (model LEBIH BURUK dari baseline) → R² negatif!"
        ),
        "step_by_step_calculation": [
            "Contoh: aktual [100, 200, 300, 400, 500], prediksi [110, 190, 320, 380, 530]",
            "Langkah 1: Hitung rata-rata aktual → (100+200+300+400+500)/5 = 300",
            "Langkah 2: Hitung SS_res = Σ(aktual − prediksi)²",
            "  → (100-110)² + (200-190)² + (300-320)² + (400-380)² + (500-530)² = 100+100+400+400+900 = 1900",
            "Langkah 3: Hitung SS_tot = Σ(aktual − rata_rata)²",
            "  → (100-300)² + (200-300)² + (300-300)² + (400-300)² + (500-300)² = 40000+10000+0+10000+40000 = 100000",
            "Langkah 4: R² = 1 − (1900 ÷ 100000) = 1 − 0.019 = 0.981",
            "Hasil: R² ≈ 0.98 (sangat bagus!)",
        ],
        "result_interpretation": (
            "R² = 0.98 berarti model menjelaskan 98% variasi data — sangat presisi. "
            "Skala interpretasi: >0.7 strong, 0.4-0.7 moderate, <0.4 weak, <0 lebih buruk dari sekadar prediksi rata-rata."
        ),
        "business_meaning": (
            "R² 0.85 untuk model prediksi penjualan = model menangkap 85% pola yang ada di data. "
            "Sisanya 15% adalah noise atau faktor yang tidak ditangkap model. "
            "WARNING: R² tinggi tidak menjamin model bagus untuk DATA BARU — bisa overfit. "
            "Selalu cek R² di TEST SET, bukan training set."
        ),
    },

    # ==================================================================
    # ACCURACY
    # ==================================================================
    "accuracy": {
        "real_math_explanation": (
            "Accuracy = proporsi prediksi yang BENAR dari semua prediksi.\n\n"
            "Cara hitung: HITUNG berapa banyak prediksi benar, lalu BAGI dengan total prediksi."
        ),
        "variable_meaning": (
            "TP (True Positive) = aktual positif, prediksi positif (BENAR). "
            "TN (True Negative) = aktual negatif, prediksi negatif (BENAR). "
            "FP (False Positive) = aktual negatif tapi prediksi positif (SALAH alarm). "
            "FN (False Negative) = aktual positif tapi prediksi negatif (MISSED). "
            "TP + TN = jumlah BENAR; total = TP + TN + FP + FN."
        ),
        "step_by_step_calculation": [
            "Contoh: 100 pasien diuji untuk penyakit",
            "TP (sakit, prediksi sakit) = 70",
            "TN (sehat, prediksi sehat) = 15",
            "FP (sehat, salah diprediksi sakit) = 8",
            "FN (sakit, miss) = 7",
            "Langkah 1: Total benar = TP + TN = 70 + 15 = 85",
            "Langkah 2: Total semua = 70 + 15 + 8 + 7 = 100",
            "Langkah 3: Accuracy = 85 ÷ 100 = 0.85 atau 85%",
        ],
        "result_interpretation": (
            "Accuracy = 85% berarti 85 dari 100 prediksi benar. "
            "BAHAYA: untuk data tidak seimbang ini menyesatkan! "
            "Misal: 95% data adalah class A. Model dummy yang selalu prediksi A → accuracy 95% tapi USELESS untuk class B."
        ),
        "business_meaning": (
            "Accuracy cocok jika kelas seimbang (50/50). Untuk kasus seperti deteksi fraud (1% data fraud), "
            "JANGAN PAKAI accuracy. Pakai PRECISION + RECALL atau F1-score atau PR-AUC. "
            "Accuracy tinggi tapi recall rendah berarti model 'main aman' dengan selalu prediksi mayoritas."
        ),
    },

    # ==================================================================
    # PRECISION
    # ==================================================================
    "precision": {
        "real_math_explanation": (
            "Precision menjawab pertanyaan: 'Dari semua kasus yang model katakan POSITIF, "
            "berapa persen yang BENAR-BENAR positif?'\n\n"
            "Cara hitung: BAGI jumlah True Positive dengan jumlah SEMUA prediksi positif (TP + FP)."
        ),
        "variable_meaning": (
            "TP = prediksi positif yang benar. "
            "FP = prediksi positif yang ternyata salah (false alarm). "
            "TP + FP = total prediksi positif (yang dikatakan model 'positif')."
        ),
        "step_by_step_calculation": [
            "Contoh: model spam filter memberi label 'spam' pada 100 email",
            "TP (benar-benar spam) = 80",
            "FP (email penting yang salah ditandai spam) = 20",
            "Langkah 1: Total prediksi positif = 80 + 20 = 100",
            "Langkah 2: Precision = TP / (TP + FP) = 80 / 100 = 0.80 atau 80%",
        ],
        "result_interpretation": (
            "Precision = 80% berarti dari 100 email yang ditandai spam, 80 benar dan 20 sebenarnya email penting. "
            "Semakin tinggi precision = semakin sedikit false alarm."
        ),
        "business_meaning": (
            "Pakai precision sebagai prioritas jika cost FALSE POSITIVE tinggi. "
            "Contoh: spam filter (jangan mark email penting sebagai spam), fraud detection (jangan block transaksi sah). "
            "Precision tinggi sering trade-off dengan recall rendah — model jadi 'pelit' memberi label positif."
        ),
    },

    # ==================================================================
    # RECALL
    # ==================================================================
    "recall": {
        "real_math_explanation": (
            "Recall (juga disebut Sensitivity) menjawab: 'Dari semua kasus AKTUAL positif, "
            "berapa banyak yang berhasil DITANGKAP model?'\n\n"
            "Cara hitung: BAGI jumlah True Positive dengan TOTAL aktual positif (TP + FN)."
        ),
        "variable_meaning": (
            "TP = prediksi positif yang benar. "
            "FN = aktual positif tapi MISSED (model bilang negatif). "
            "TP + FN = total aktual positif (yang sebenarnya positif di realitas)."
        ),
        "step_by_step_calculation": [
            "Contoh: ada 100 pasien benar-benar sakit kanker",
            "TP (model deteksi sakit) = 80",
            "FN (model bilang sehat padahal sakit) = 20",
            "Langkah 1: Total aktual positif = 80 + 20 = 100",
            "Langkah 2: Recall = TP / (TP + FN) = 80 / 100 = 0.80 atau 80%",
        ],
        "result_interpretation": (
            "Recall = 80% berarti dari 100 pasien sakit, model menangkap 80 dan miss 20. "
            "Semakin tinggi recall = semakin sedikit kasus terlewat."
        ),
        "business_meaning": (
            "Pakai recall sebagai prioritas jika cost FALSE NEGATIVE tinggi. "
            "Contoh: cancer screening (jangan miss pasien sakit), fraud detection (jangan miss transaksi fraud), "
            "security alert (jangan miss serangan cyber). "
            "Recall tinggi sering trade-off dengan precision rendah — model jadi 'paranoid' menandai banyak."
        ),
    },

    # ==================================================================
    # F1 SCORE
    # ==================================================================
    "f1_score": {
        "real_math_explanation": (
            "F1 score adalah HARMONIC MEAN dari Precision dan Recall — menyeimbangkan keduanya.\n\n"
            "Mengapa harmonic mean (bukan rata-rata biasa)? Karena harmonic mean MENGHUKUM ketidakseimbangan. "
            "Kalau precision = 0.99 tapi recall = 0.10, rata-rata biasa = 0.55 (kelihatan ok), "
            "tapi F1 = 0.18 (jelek!).\n\n"
            "Cara hitung: F1 = (2 × Precision × Recall) ÷ (Precision + Recall)"
        ),
        "variable_meaning": (
            "Precision = quality alarm. Recall = coverage detection. "
            "F1 hanya tinggi kalau KEDUA-DUANYA tinggi. F1 = 1 berarti sempurna; F1 = 0 berarti bencana."
        ),
        "step_by_step_calculation": [
            "Contoh: Precision = 0.8, Recall = 0.6",
            "Langkah 1: Pembilang = 2 × Precision × Recall = 2 × 0.8 × 0.6 = 0.96",
            "Langkah 2: Penyebut = Precision + Recall = 0.8 + 0.6 = 1.4",
            "Langkah 3: F1 = 0.96 ÷ 1.4 = 0.686",
            "Hasil: F1 ≈ 0.69",
        ],
        "result_interpretation": (
            "F1 = 0.69 menunjukkan model cukup seimbang antara akurasi alarm dan cakupan deteksi. "
            "F1 > 0.8 = excellent, 0.6-0.8 = good, <0.5 = bermasalah."
        ),
        "business_meaning": (
            "F1 adalah pilihan tepat untuk imbalanced data. F1-MACRO = rata-rata F1 per kelas (semua kelas dianggap sama penting). "
            "F1-WEIGHTED = rata-rata F1 di-weight by class size. "
            "Untuk skripsi atau publikasi: SELALU laporkan F1-macro untuk imbalanced classification, jangan accuracy."
        ),
    },

    # ==================================================================
    # STANDARDIZATION (Z-SCORE)
    # ==================================================================
    "standardization": {
        "real_math_explanation": (
            "Standardization (Z-score) mengubah angka menjadi ukuran 'BERAPA STANDAR DEVIASI dari rata-rata'.\n\n"
            "Cara: untuk setiap nilai, KURANGI rata-rata, lalu BAGI standar deviasi.\n\n"
            "Z-score = (nilai − rata-rata) ÷ standar deviasi\n\n"
            "Hasilnya: setiap fitur jadi punya rata-rata 0 dan standar deviasi 1."
        ),
        "variable_meaning": (
            "x = nilai asli. "
            "μ (mu) = rata-rata seluruh data fitur. "
            "σ (sigma) = standar deviasi (ukuran sebaran) — semakin besar σ, semakin lebar distribusi data. "
            "z = hasil — angka tanpa satuan yang menunjukkan posisi relatif terhadap rata-rata."
        ),
        "step_by_step_calculation": [
            "Contoh data: [10, 20, 30, 40, 50] (misal: tinggi badan dalam cm)",
            "Langkah 1: Hitung rata-rata → (10+20+30+40+50)/5 = 30",
            "Langkah 2: Hitung selisih dari rata-rata → -20, -10, 0, 10, 20",
            "Langkah 3: Kuadratkan & rata-rata → ((-20)²+(-10)²+0²+10²+20²)/5 = 1000/5 = 200 (variance)",
            "Langkah 4: Akar kuadratkan untuk dapat std → √200 ≈ 14.14",
            "Langkah 5: Hitung z untuk setiap nilai:",
            "  z(10) = (10-30)/14.14 = -1.41",
            "  z(20) = (20-30)/14.14 = -0.71",
            "  z(30) = (30-30)/14.14 = 0",
            "  z(40) = (40-30)/14.14 = 0.71",
            "  z(50) = (50-30)/14.14 = 1.41",
        ],
        "result_interpretation": (
            "z = 0 → tepat di rata-rata. z = 1 → 1 std di atas rata-rata. z = -1 → 1 std di bawah. "
            "|z| > 3 sering dianggap OUTLIER (nilai ekstrim, hanya ~0.3% data normal yang punya z >3)."
        ),
        "business_meaning": (
            "Standardization WAJIB sebelum melatih model yang sensitif skala (Linear Regression, Logistic, SVM, KNN, Neural Network). "
            "Tanpa scaling, fitur dengan range besar (misal: gaji dalam Rupiah, range jutaan) akan MENDOMINASI fitur dengan range kecil "
            "(misal: usia, range 20-60). Setelah standardization, semua fitur 'setara' dalam skala."
        ),
    },

    # ==================================================================
    # MIN-MAX SCALING
    # ==================================================================
    "minmax_scaling": {
        "real_math_explanation": (
            "Min-Max Scaling mengubah angka menjadi RANGE [0, 1].\n\n"
            "Cara: kurangi nilai dengan minimum, lalu bagi dengan range (maximum − minimum).\n\n"
            "Hasil: nilai terkecil jadi 0, nilai terbesar jadi 1, sisanya di antara 0 dan 1."
        ),
        "variable_meaning": (
            "x_min = nilai terkecil di data. x_max = nilai terbesar. "
            "(x_max − x_min) = RANGE / sebaran. "
            "Scaled value = posisi relatif x dalam range 0-1."
        ),
        "step_by_step_calculation": [
            "Contoh data: [10, 20, 30, 40, 50]",
            "Langkah 1: x_min = 10, x_max = 50, range = 50 - 10 = 40",
            "Langkah 2: Scale setiap nilai:",
            "  (10-10)/40 = 0/40 = 0.00",
            "  (20-10)/40 = 10/40 = 0.25",
            "  (30-10)/40 = 20/40 = 0.50",
            "  (40-10)/40 = 30/40 = 0.75",
            "  (50-10)/40 = 40/40 = 1.00",
            "Hasil: [0.00, 0.25, 0.50, 0.75, 1.00]",
        ],
        "result_interpretation": (
            "0 = nilai terkecil di training data; 1 = nilai terbesar. "
            "Test data dengan nilai > x_max akan menghasilkan scaled > 1 (out of expected range — perlu di-clip)."
        ),
        "business_meaning": (
            "Min-max cocok untuk neural network (input layer expect 0-1) atau image processing. "
            "TIDAK cocok jika ada outlier — satu outlier ekstrem akan membuat semua nilai lain mendekati 0. "
            "Dalam kasus ada outlier → pakai RobustScaler."
        ),
    },

    # ==================================================================
    # LINEAR REGRESSION
    # ==================================================================
    "linear_regression": {
        "real_math_explanation": (
            "Linear Regression mencari GARIS LURUS terbaik melewati data.\n\n"
            "Bayangkan plot data: setiap data point = titik di grafik. "
            "Linear Regression cari garis lurus yang TOTAL JARAK ke semua titik PALING KECIL.\n\n"
            "Garis = w·x + b\n"
            "  w (slope) = kemiringan garis — seberapa cepat y berubah saat x naik 1 unit.\n"
            "  b (intercept) = nilai y saat x = 0 — titik di mana garis memotong sumbu y.\n\n"
            "Cara cari w dan b: minimize MSE (Mean Squared Error). MSE = rata-rata kuadrat selisih antara nilai aktual dan prediksi."
        ),
        "variable_meaning": (
            "x = fitur input (misal: luas tanah). "
            "y = target output (misal: harga rumah). "
            "ŷ (y-topi) = nilai prediksi model. "
            "w = weight / koefisien per fitur. "
            "b = bias / intercept."
        ),
        "step_by_step_calculation": [
            "Contoh: data luas tanah vs harga rumah",
            "x (luas m²): [50, 100, 150]",
            "y (harga jt): [500, 950, 1500]",
            "Langkah 1: Hitung rata-rata x → (50+100+150)/3 = 100",
            "Langkah 2: Hitung rata-rata y → (500+950+1500)/3 = 983.3",
            "Langkah 3: Hitung slope w (rumus regresi linear sederhana):",
            "  Numerator = Σ(xᵢ−x̄)(yᵢ−ȳ) = (-50)(-483.3) + (0)(-33.3) + (50)(516.7) = 24165 + 0 + 25835 = 50000",
            "  Denominator = Σ(xᵢ−x̄)² = (-50)² + 0² + 50² = 2500 + 0 + 2500 = 5000",
            "  w = 50000 / 5000 = 10",
            "Langkah 4: Hitung intercept b → ȳ − w·x̄ = 983.3 − 10×100 = -16.7",
            "Langkah 5: Model: harga = 10 × luas − 16.7",
            "Test: untuk luas = 120 m² → harga = 10×120 − 16.7 = 1183.3 juta",
        ],
        "result_interpretation": (
            "Slope w = 10 berarti SETIAP m² menambah harga 10 juta. "
            "Intercept b = -16.7 (nilai matematis, mungkin tidak masuk akal kalau luas = 0). "
            "Persamaan ini DIPAKAI untuk prediksi harga rumah baru berdasarkan luas tanah."
        ),
        "business_meaning": (
            "Linear Regression cocok untuk: prediksi revenue, forecast simple (umur biaya), inferensi pengaruh fitur. "
            "Kekuatan utama: INTERPRETABILITAS — kamu bisa bilang 'setiap unit X menambah Y sekian'. "
            "Kelemahan: hanya menangkap pola linear. Untuk pola kompleks pakai Random Forest atau Gradient Boosting."
        ),
    },

    # ==================================================================
    # LOGISTIC REGRESSION
    # ==================================================================
    "logistic_regression": {
        "real_math_explanation": (
            "Logistic Regression seperti Linear Regression, tapi outputnya 'dijepit' antara 0 dan 1 menggunakan FUNGSI SIGMOID.\n\n"
            "Langkah:\n"
            "1. Hitung kombinasi linear z = w·x + b (sama seperti linear regression).\n"
            "2. Lewatkan z melalui SIGMOID: p = 1 ÷ (1 + e⁻ᶻ).\n"
            "Hasilnya p = probability bahwa data termasuk kelas 1 (atau positif).\n\n"
            "Sigmoid mengubah angka apapun menjadi nilai 0-1. Angka besar → mendekati 1. Angka kecil/negatif → mendekati 0. Angka 0 → 0.5."
        ),
        "variable_meaning": (
            "z = logit (kombinasi linear, bisa positif/negatif/besar/kecil). "
            "σ(z) = sigmoid function — mengubah z jadi probability di [0, 1]. "
            "p = P(y=1|x) = probability bahwa instance termasuk kelas 1. "
            "Threshold default 0.5: jika p ≥ 0.5 → prediksi class 1, else class 0."
        ),
        "step_by_step_calculation": [
            "Contoh: prediksi pasien high-risk berdasarkan umur dan tekanan darah",
            "Sudah di-train: w_umur = 0.05, w_bp = 0.02, b = -5",
            "Pasien baru: umur = 60, BP = 140",
            "Langkah 1: Hitung z = 0.05×60 + 0.02×140 + (-5) = 3 + 2.8 - 5 = 0.8",
            "Langkah 2: Hitung sigmoid → σ(0.8) = 1 / (1 + e^(-0.8)) = 1 / (1 + 0.449) = 0.690",
            "Hasil: probability = 69% pasien high-risk",
        ],
        "result_interpretation": (
            "Probability 0.69 → di atas threshold 0.5 → prediksi HIGH-RISK. "
            "Confidence model: cukup yakin (jauh dari 0.5 yang berarti 'tidak yakin'). "
            "Untuk imbalanced data, threshold 0.5 sering tidak optimal — gunakan PR curve untuk pilih threshold yang seimbang precision-recall."
        ),
        "business_meaning": (
            "Logistic Regression IDEAL untuk: probability prediction (clinical risk, churn risk, fraud likelihood). "
            "Output PROBABILITY mempermudah decision making dengan threshold yang sesuai biaya bisnis. "
            "Misal: untuk fraud detection, threshold rendah (0.3) menangkap lebih banyak fraud (recall tinggi) tapi lebih banyak false alarm. "
            "Threshold tinggi (0.7) sebaliknya."
        ),
    },

    # ==================================================================
    # RANDOM FOREST
    # ==================================================================
    "random_forest": {
        "real_math_explanation": (
            "Random Forest adalah ENSEMBLE: kumpulan banyak Decision Tree yang VOTING (klasifikasi) "
            "atau RATA-RATA (regresi).\n\n"
            "Cara membangun:\n"
            "1. Buat 100 (default) Decision Tree.\n"
            "2. Setiap tree dilatih pada SAMPEL ACAK dari data (bootstrap = sampling dengan replacement).\n"
            "3. Setiap split di tree, hanya boleh memilih dari SUBSET ACAK fitur (biasanya √jumlah_fitur).\n\n"
            "Untuk prediksi:\n"
            "• Klasifikasi: setiap tree vote → ambil MAYORITAS suara → hasilnya class\n"
            "• Regresi: setiap tree prediksi → ambil RATA-RATA → hasilnya numerik"
        ),
        "variable_meaning": (
            "M = jumlah tree (typical 100-500). "
            "Setiap tree adalah ahli dengan 'pengalaman' berbeda (data berbeda + fitur berbeda yang dilihat). "
            "Hasil ensemble lebih AKURAT dan LEBIH ROBUST dibanding single tree, karena kesalahan individual tree saling MENGKANCEL."
        ),
        "step_by_step_calculation": [
            "Contoh: 100 tree memprediksi penyakit (sakit/sehat) untuk satu pasien",
            "Tree 1: sakit",
            "Tree 2: sakit",
            "Tree 3: sehat",
            "...",
            "Setelah semua: 73 tree vote 'sakit', 27 tree vote 'sehat'",
            "Langkah 1: Mayoritas → SAKIT",
            "Langkah 2: Probability sakit = 73/100 = 73%",
            "Langkah 3: Output: prediksi=sakit, probability=0.73",
        ],
        "result_interpretation": (
            "Probability 73% berarti 73% tree setuju pasien sakit. "
            "Lebih tinggi probability = lebih kuat consensus model. "
            "Feature importance tersedia: fitur mana yang paling sering & efektif dipakai untuk split."
        ),
        "business_meaning": (
            "Random Forest = workhorse paling reliable untuk tabular data. "
            "Pakai sebagai BASELINE pertama setelah Linear Regression. "
            "Keunggulan: handle missing values, non-linear, feature importance, no scaling needed. "
            "Kelemahan: probability kurang well-calibrated → pakai CalibratedClassifierCV jika butuh probability akurat."
        ),
    },

    # ==================================================================
    # K-MEANS
    # ==================================================================
    "kmeans": {
        "real_math_explanation": (
            "K-Means mengelompokkan data ke K KELOMPOK berdasarkan KEDEKATAN ke pusat (centroid).\n\n"
            "Algoritma:\n"
            "1. Pilih K (jumlah kelompok), misal K=3.\n"
            "2. Random pilih 3 centroid awal.\n"
            "3. Setiap data point gabung ke centroid TERDEKAT (Euclidean distance).\n"
            "4. Setelah semua points bergabung, hitung centroid BARU = rata-rata anggota kelompok.\n"
            "5. Ulangi step 3-4 sampai centroid TIDAK BERGERAK lagi (convergen).\n\n"
            "Jarak Euclidean = √((x₁-x₂)² + (y₁-y₂)² + ...). Untuk 2D: jarak garis lurus standard antara dua titik."
        ),
        "variable_meaning": (
            "K = jumlah cluster (HARUS ditentukan dulu! Pakai elbow method atau silhouette score). "
            "μⱼ = centroid cluster j. "
            "d(x, μ) = jarak Euclidean dari point x ke centroid μ."
        ),
        "step_by_step_calculation": [
            "Contoh: 6 toko di koordinat 2D, K=2 (mau bagi jadi 2 area)",
            "Toko: (1,1), (1,2), (2,1), (8,8), (8,9), (9,8)",
            "Langkah 1: Random init centroid C1=(0,0), C2=(10,10)",
            "Langkah 2: Hitung jarak setiap toko ke kedua centroid:",
            "  (1,1) → ke C1: √(1²+1²)=1.41, ke C2: √(81+81)=12.73 → masuk cluster 1",
            "  (1,2) → ke C1: 2.24, ke C2: 11.40 → cluster 1",
            "  ... (semua toko 1-3 ke cluster 1, toko 4-6 ke cluster 2)",
            "Langkah 3: Hitung centroid baru = rata-rata anggota:",
            "  C1 baru = ((1+1+2)/3, (1+2+1)/3) = (1.33, 1.33)",
            "  C2 baru = ((8+8+9)/3, (8+9+8)/3) = (8.33, 8.33)",
            "Langkah 4: Re-assign points → tidak ada perubahan → CONVERGEN",
            "Hasil: 2 cluster terbentuk, masing-masing 3 toko",
        ],
        "result_interpretation": (
            "Cluster 1 = area pusat kota (centroid 1.33, 1.33). "
            "Cluster 2 = area pinggiran (centroid 8.33, 8.33). "
            "INERTIA (sum of squared distances ke centroid) = ukuran 'compact' cluster — semakin kecil semakin baik."
        ),
        "business_meaning": (
            "K-Means cocok untuk: customer segmentation, market basket grouping, store clustering. "
            "Kelemahan UTAMA: harus tentukan K dulu. Pakai ELBOW METHOD: plot K vs inertia, cari titik 'siku'. "
            "Atau pakai SILHOUETTE SCORE: nilai antara -1 dan 1, semakin tinggi semakin baik clustering."
        ),
    },

    # ==================================================================
    # ROC-AUC
    # ==================================================================
    "roc_auc": {
        "real_math_explanation": (
            "ROC-AUC mengukur kemampuan model membedakan kelas positif dan negatif di SEMUA threshold.\n\n"
            "ROC curve = plot True Positive Rate vs False Positive Rate di berbagai threshold.\n"
            "AUC = AREA UNDER CURVE (luas area di bawah kurva).\n\n"
            "Interpretasi probabilistik: AUC = probability bahwa jika kita pilih SATU positive dan SATU negative secara acak, "
            "model memberikan SKOR LEBIH TINGGI ke yang positive."
        ),
        "variable_meaning": (
            "TPR (True Positive Rate) = sama dengan Recall = TP/(TP+FN) — % positive yang ditangkap. "
            "FPR (False Positive Rate) = FP/(FP+TN) — % negative yang salah di-flag. "
            "AUC = 0.5 berarti random; AUC = 1.0 berarti perfect discrimination."
        ),
        "step_by_step_calculation": [
            "Contoh: model memberi 10 sample dengan probability prediksi:",
            "  Pos: [0.95, 0.85, 0.70, 0.55, 0.40]",
            "  Neg: [0.60, 0.45, 0.30, 0.20, 0.10]",
            "Langkah 1: Untuk setiap pasangan (pos, neg), cek: apakah skor pos > skor neg?",
            "  (0.95, 0.60) → pos lebih tinggi ✓",
            "  (0.95, 0.45) → ✓",
            "  ... dst (5×5 = 25 pasangan)",
            "Langkah 2: Hitung berapa pasangan yang BENAR (pos > neg)",
            "  Misalnya: 23 dari 25 pasangan benar, 2 tied/wrong",
            "Langkah 3: AUC = 23/25 = 0.92",
        ],
        "result_interpretation": (
            "AUC = 0.92 berarti dalam 92% kasus, sample positif diberi skor lebih tinggi dari sample negatif. "
            "Skala umum: >0.9 EXCELLENT, 0.8-0.9 GOOD, 0.7-0.8 FAIR, <0.7 POOR."
        ),
        "business_meaning": (
            "ROC-AUC bagus untuk perbandingan model (independen threshold). "
            "BAHAYA: untuk severely imbalanced data (1% positif), ROC-AUC OVER-OPTIMISTIC! "
            "Misal AUC 0.95 padahal model masih miss banyak fraud. "
            "Untuk imbalanced → pakai PR-AUC (Precision-Recall AUC) yang lebih jujur."
        ),
    },

    # ==================================================================
    # KFOLD CV
    # ==================================================================
    "kfold_cv": {
        "real_math_explanation": (
            "K-Fold Cross-Validation membagi dataset jadi K bagian (lipatan), lalu melatih+testing K kali.\n\n"
            "Cara:\n"
            "1. Bagi data menjadi K bagian sama besar (typical K=5).\n"
            "2. Untuk i = 1 sampai K:\n"
            "   - Test pada bagian i\n"
            "   - Train pada K-1 bagian lainnya\n"
            "   - Catat skor evaluasi\n"
            "3. Hitung RATA-RATA skor → estimasi performa model.\n"
            "4. Hitung STANDAR DEVIASI skor → estimasi stabilitas model."
        ),
        "variable_meaning": (
            "K = jumlah fold/lipatan (standar: 5 atau 10). "
            "CV mean = rata-rata skor, estimasi performa real. "
            "CV std = standar deviasi, semakin kecil semakin stabil model. "
            "Mendapat K skor → bisa hitung confidence interval."
        ),
        "step_by_step_calculation": [
            "Contoh: 100 data, K=5. Setiap fold = 20 data.",
            "Iterasi 1: Test fold 1 (20 data), Train fold 2-5 (80 data) → skor F1 = 0.81",
            "Iterasi 2: Test fold 2, Train sisanya → F1 = 0.85",
            "Iterasi 3: F1 = 0.79",
            "Iterasi 4: F1 = 0.83",
            "Iterasi 5: F1 = 0.82",
            "Langkah akhir:",
            "  CV mean = (0.81 + 0.85 + 0.79 + 0.83 + 0.82) / 5 = 0.82",
            "  CV std = √(varians) ≈ 0.022",
            "Hasil: Estimated F1 = 0.82 ± 0.022",
        ],
        "result_interpretation": (
            "Performa model di-estimasi 0.82 dengan std 0.022 — SANGAT STABIL. "
            "CV std besar (>0.1) = model tidak stabil, butuh more data atau regularization. "
            "Ini estimasi LEBIH RELIABLE dari single train/test split."
        ),
        "business_meaning": (
            "CV adalah GOLD STANDARD untuk model evaluation. "
            "Selalu laporkan CV mean ± std untuk publikasi. "
            "K=5 cukup untuk dataset >1000 rows. K=10 lebih reliable tapi 2x lebih lambat. "
            "Untuk classification IMBALANCED → pakai StratifiedKFold (preserve class proportion)."
        ),
    },

    # ==================================================================
    # IQR OUTLIER
    # ==================================================================
    "iqr_outlier": {
        "real_math_explanation": (
            "IQR (Interquartile Range) Method mendeteksi outlier dengan QUARTILES.\n\n"
            "Quartile = membagi data terurut jadi 4 bagian sama besar:\n"
            "• Q1 = nilai di posisi 25% (data 25% di bawahnya)\n"
            "• Q2 = MEDIAN (50%)\n"
            "• Q3 = nilai di posisi 75%\n"
            "IQR = Q3 − Q1 (range tengah 50% data).\n\n"
            "Aturan outlier:\n"
            "• Lower bound = Q1 − 1.5 × IQR\n"
            "• Upper bound = Q3 + 1.5 × IQR\n"
            "• Nilai di luar [lower, upper] = OUTLIER."
        ),
        "variable_meaning": (
            "IQR mewakili 'spread' tengah data, robust terhadap outlier itu sendiri. "
            "Faktor 1.5 = standar konservatif. "
            "Untuk extreme outlier saja, gunakan faktor 3.0."
        ),
        "step_by_step_calculation": [
            "Contoh data terurut: [10, 20, 25, 30, 35, 40, 45, 50, 100]",
            "Langkah 1: Q1 (25% dari 9 data) = nilai posisi 2-3 → 22.5",
            "Langkah 2: Q3 (75%) = nilai posisi 6-7 → 47.5",
            "Langkah 3: IQR = Q3 − Q1 = 47.5 − 22.5 = 25",
            "Langkah 4: Lower bound = 22.5 − 1.5×25 = 22.5 − 37.5 = -15",
            "Langkah 5: Upper bound = 47.5 + 1.5×25 = 47.5 + 37.5 = 85",
            "Langkah 6: Cek setiap data:",
            "  10, 20, 25, 30, 35, 40, 45, 50 → semua di [-15, 85] → NORMAL",
            "  100 → di atas 85 → OUTLIER!",
        ],
        "result_interpretation": (
            "Nilai 100 ditandai outlier karena melewati upper bound 85. "
            "JANGAN langsung hapus — investigasi dulu: apakah data entry error, atau valid extreme value?"
        ),
        "business_meaning": (
            "IQR robust dan tidak membuat asumsi distribusi. Cocok untuk data finansial, lab medis. "
            "Untuk data yang TERSEBAR (skewed): IQR lebih reliable dari Z-score. "
            "Outlier dalam analisis bisnis sering = peluang/risk: customer pembeli besar, transaksi fraud, anomali sensor."
        ),
    },

    # ==================================================================
    # TF-IDF
    # ==================================================================
    "tfidf": {
        "real_math_explanation": (
            "TF-IDF mengubah text → angka dengan PRINSIP: kata yang sering muncul di dokumen TAPI jarang di corpus = kata KUNCI.\n\n"
            "Dua komponen:\n"
            "1. TF (Term Frequency) = berapa sering kata muncul di dokumen ini.\n"
            "   TF = jumlah kata di doc ÷ total kata di doc\n\n"
            "2. IDF (Inverse Document Frequency) = penalti untuk kata yang muncul di banyak dokumen.\n"
            "   IDF = log(jumlah_dokumen ÷ jumlah_dokumen_yang_mengandung_kata_ini)\n\n"
            "TF-IDF = TF × IDF\n\n"
            "Hasilnya: kata umum (the, and, the) → IDF kecil → TF-IDF kecil. "
            "Kata distinctive untuk dokumen tertentu → TF-IDF besar."
        ),
        "variable_meaning": (
            "Kata seperti 'the' yang ada di SEMUA dokumen → IDF mendekati 0 → TF-IDF kecil. "
            "Kata seperti 'pneumonia' yang hanya di dokumen medis → IDF tinggi → TF-IDF besar di doc tersebut. "
            "Output: matrix sparse berukuran (jumlah_doc × ukuran_vocabulary)."
        ),
        "step_by_step_calculation": [
            "Contoh corpus: 3 dokumen pendek",
            "  Doc1: 'the cat sat on the mat'  (6 kata)",
            "  Doc2: 'the dog jumped'  (3 kata)",
            "  Doc3: 'the bird flew over the cat'  (6 kata)",
            "Hitung untuk kata 'cat':",
            "Langkah 1: TF di Doc1 = 1 muncul di 6 kata = 1/6 ≈ 0.167",
            "Langkah 2: IDF = log(3 / 2) = log(1.5) ≈ 0.405  (cat muncul di 2 dari 3 doc)",
            "Langkah 3: TF-IDF di Doc1 = 0.167 × 0.405 ≈ 0.068",
            "Hitung untuk kata 'the':",
            "Langkah 1: IDF = log(3/3) = log(1) = 0  (the di SEMUA doc)",
            "Langkah 2: TF-IDF = 0  (kata umum di-zero-kan!)",
        ],
        "result_interpretation": (
            "TF-IDF tinggi → kata penting/khas untuk dokumen tersebut. "
            "TF-IDF mendekati 0 → kata umum, tidak informatif. "
            "Vector hasil bisa dipakai untuk: classification text, similarity search, clustering, search ranking."
        ),
        "business_meaning": (
            "TF-IDF adalah dasar pencarian web (Google) — kata distinctive di doc dapat skor tinggi. "
            "Untuk product reviews: 'mantap', 'kecewa' akan jadi kata kunci untuk klasifikasi sentiment. "
            "Limitation: BAG-OF-WORDS — tidak menangkap urutan kata. 'tidak baik' = 'baik tidak' bagi TF-IDF. "
            "Untuk konteks → pakai BERT embeddings."
        ),
    },

    # ==================================================================
    # CONFUSION MATRIX
    # ==================================================================
    "confusion_matrix": {
        "real_math_explanation": (
            "Confusion Matrix adalah TABEL 2D yang membandingkan prediksi vs aktual untuk setiap kelas.\n\n"
            "Untuk binary classification: 2x2 matrix dengan 4 sel:\n"
            "  • TP (True Positive): aktual=1, prediksi=1 (BENAR positif)\n"
            "  • TN (True Negative): aktual=0, prediksi=0 (BENAR negatif)\n"
            "  • FP (False Positive): aktual=0, prediksi=1 (false alarm)\n"
            "  • FN (False Negative): aktual=1, prediksi=0 (missed)\n\n"
            "Diagonal (TP, TN) = prediksi BENAR. Off-diagonal (FP, FN) = prediksi SALAH."
        ),
        "variable_meaning": (
            "Setiap sel = jumlah sample dengan kombinasi tertentu. "
            "Total keempat sel = total test sample. "
            "Dari confusion matrix bisa diturunkan: accuracy, precision, recall, F1, sensitivity, specificity."
        ),
        "step_by_step_calculation": [
            "Contoh: 100 pasien diuji untuk diabetes",
            "Aktual ada diabetes: 30 orang",
            "Aktual sehat: 70 orang",
            "Hasil prediksi model:",
            "  Diprediksi diabetes: 35 orang (28 benar + 7 false alarm)",
            "  Diprediksi sehat: 65 orang (63 benar + 2 missed)",
            "Confusion Matrix:",
            "                     Predicted",
            "                     Sehat   Diabetes",
            "Aktual  Sehat:        63       7        (TN, FP)",
            "Aktual  Diabetes:     2        28       (FN, TP)",
            "Jadi: TP=28, TN=63, FP=7, FN=2",
            "Accuracy = (28+63)/100 = 0.91",
            "Precision = 28/(28+7) = 0.80",
            "Recall (Sensitivity) = 28/(28+2) = 0.93",
            "Specificity = 63/(63+7) = 0.90",
        ],
        "result_interpretation": (
            "Diagonal tebal (TP=28, TN=63) = model bagus dalam kasus ini. "
            "FN=2 = 2 pasien diabetes yang KETERLEWAT (perlu di-investigasi mengapa). "
            "FP=7 = 7 pasien sehat dapat alarm palsu (mungkin perlu test ulang)."
        ),
        "business_meaning": (
            "Confusion matrix memberi gambaran LENGKAP error model. "
            "Untuk medical: FN (missed diagnosis) lebih berbahaya dari FP (false alarm) → optimize Recall. "
            "Untuk fraud: bergantung trade-off cost block transaksi sah (FP) vs miss fraud (FN). "
            "Tampilkan di laporan untuk transparency."
        ),
    },
}
