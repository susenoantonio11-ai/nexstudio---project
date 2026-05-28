"""
Plain-language mathematical explanations untuk metode disaster prediction.
Mengikuti format: METHOD / PURPOSE / WHY USED / REAL MATHEMATICAL EXPLANATION /
VARIABLE MEANING / STEP-BY-STEP CALCULATION / RESULT INTERPRETATION /
BUSINESS MEANING.
"""

DISASTER_PLAIN_EXPLANATIONS = {
    "gutenberg_richter_b_value": {
        "method": "Gutenberg-Richter b-value",
        "purpose": "Mengukur seberapa sering gempa kecil dibanding gempa besar di sebuah daerah.",
        "why_used": (
            "b-value adalah ringkasan tunggal yang menggambarkan distribusi "
            "magnitudo gempa. Nilai mendekati 1 menandakan sebaran energi yang "
            "relatif normal, sedangkan nilai jauh di bawah 1 sering ditemukan "
            "pada area dengan stress kerak tinggi."
        ),
        "real_math": (
            "Hubungan log10 N = a - b*M menyatakan jumlah gempa N yang besarnya "
            "minimal M. Nilai b dihitung dari rata-rata magnitudo lewat estimator "
            "Aki (1965): b = log10(e) dibagi (rata-rata magnitudo dikurangi "
            "magnitude of completeness)."
        ),
        "variables": {
            "N": "jumlah event dengan magnitudo >= M",
            "M": "magnitudo",
            "Mc": "magnitude of completeness (batas terendah katalog yang lengkap)",
            "a, b": "konstanta regional",
        },
        "step_by_step": [
            "Hitung distribusi magnitudo dari katalog historis.",
            "Estimasi Mc dengan metode maximum frequency.",
            "Saring event dengan magnitudo >= Mc.",
            "Hitung rata-rata magnitudo pada subset tersebut.",
            "Substitusi ke formula b = log10(e) / (mean - (Mc - 0.05)).",
        ],
        "result_interpretation": (
            "b ~ 1.0 normal. b < 0.7 menunjukkan crust stress tinggi. b > 1.3 "
            "umumnya menandakan area volkanik atau swarm seismik."
        ),
        "business_meaning": (
            "b-value menjadi indikator awal mitigasi seismik. Nilai rendah dapat "
            "memicu perlu tinjauan ulang risiko bangunan kritikal di daerah rawan."
        ),
        "citations": ["Gutenberg & Richter (1944)", "Aki (1965)"],
    },

    "wells_coppersmith_rupture": {
        "method": "Wells-Coppersmith Surface Rupture Length",
        "purpose": "Memperkirakan panjang patahan yang pecah pada saat gempa besar.",
        "why_used": (
            "Estimasi panjang rupture penting untuk skenario hazard, simulasi "
            "ground motion, dan estimasi luas wilayah terdampak."
        ),
        "real_math": (
            "Untuk patahan strike-slip: log10 SRL = -3.55 + 0.74 * Mw, dengan "
            "SRL dalam kilometer dan Mw magnitudo momen."
        ),
        "variables": {
            "SRL": "Surface Rupture Length (km)",
            "Mw": "Magnitudo momen",
        },
        "step_by_step": [
            "Tentukan magnitudo momen Mw dari katalog atau model.",
            "Hitung 0.74 * Mw - 3.55 untuk mendapatkan log10 SRL.",
            "Hitung 10 pangkat hasil tersebut untuk memperoleh SRL dalam km.",
        ],
        "result_interpretation": (
            "Mw 7.5 menghasilkan SRL ~63 km; Mw 8.0 ~138 km. Semakin besar Mw, "
            "rupture semakin panjang dan dampak permukaan semakin luas."
        ),
        "business_meaning": (
            "Digunakan oleh perencana mitigasi untuk menentukan radius dampak "
            "skenario terburuk dan kebutuhan jalur evakuasi."
        ),
        "citations": ["Wells & Coppersmith (1994)"],
    },

    "rational_method_runoff": {
        "method": "Rational Method untuk Debit Puncak",
        "purpose": "Memperkirakan debit puncak runoff dari hujan di DAS kecil.",
        "why_used": (
            "Sederhana, transparan, dan masih menjadi metode standar untuk "
            "perencanaan drainase perkotaan dan analisis banjir kilat."
        ),
        "real_math": (
            "Q = C * I * A. Q debit puncak, C koefisien runoff (0..1) yang "
            "mencerminkan permeabilitas, I intensitas hujan (m/jam), A luas "
            "DAS (m^2). Konversi satuan dilakukan agar Q dalam m^3/detik."
        ),
        "variables": {
            "Q": "debit puncak (m^3/s)",
            "C": "koefisien runoff (0..1)",
            "I": "intensitas hujan (m/jam)",
            "A": "luas DAS (m^2)",
        },
        "step_by_step": [
            "Tentukan luas DAS A dan durasi hujan.",
            "Hitung intensitas I = curah hujan / durasi.",
            "Estimasi C dari fraksi imperviousness, kelembaban tanah, dan slope.",
            "Hitung Q = C * I * A / 3600 untuk satuan m^3/s.",
        ],
        "result_interpretation": (
            "Q tinggi menandakan potensi banjir signifikan; misal Q > 100 m^3/s "
            "untuk DAS 25 km^2 sudah cukup untuk meluap pada kanal standar."
        ),
        "business_meaning": (
            "Menentukan kapasitas saluran, polder, dan kebutuhan retensi air."
        ),
        "citations": ["Mulvaney (1851)", "Chow, Maidment, Mays (1988)"],
    },

    "infinite_slope_fos": {
        "method": "Faktor Keamanan Lereng Tak Terhingga",
        "purpose": "Mengukur stabilitas lereng dangkal terhadap longsor translasi.",
        "why_used": (
            "Bentuk paling sederhana dari analisis stabilitas lereng yang tetap "
            "informatif untuk longsor dangkal dipicu hujan."
        ),
        "real_math": (
            "FOS = (c' + (gamma - m * gamma_w) * z * cos^2 theta * tan phi') "
            "dibagi (gamma * z * sin theta * cos theta). c' kohesi efektif, "
            "phi' sudut geser dalam, theta sudut lereng, z kedalaman bidang "
            "longsor, m rasio muka air."
        ),
        "variables": {
            "c'": "kohesi efektif (kPa)",
            "phi'": "sudut geser dalam efektif (deg)",
            "theta": "sudut lereng (deg)",
            "z": "kedalaman bidang longsor (m)",
            "m": "rasio kedalaman muka air terhadap z",
            "gamma, gamma_w": "berat jenis tanah dan air",
        },
        "step_by_step": [
            "Tentukan parameter tanah (c', phi') dari uji laboratorium.",
            "Tentukan sudut lereng dan kedalaman bidang longsor.",
            "Estimasi rasio muka air m sesuai kondisi hujan.",
            "Substitusi ke formula FOS.",
        ],
        "result_interpretation": (
            "FOS > 1.5 stabil, 1.0..1.5 marjinal, < 1.0 tidak stabil."
        ),
        "business_meaning": (
            "Membantu prioritas mitigasi: zona FOS rendah perlu drainase, "
            "vegetasi pengikat akar, atau pengurangan beban di mahkota lereng."
        ),
        "citations": ["Skempton & DeLory (1957)", "Terzaghi (1943)"],
    },

    "fwi_van_wagner": {
        "method": "Fire Weather Index (FWI)",
        "purpose": "Indeks risiko kebakaran hutan dari cuaca harian.",
        "why_used": (
            "Standar global yang digunakan banyak negara, termasuk komponen "
            "Indonesia Fire-DRS."
        ),
        "real_math": (
            "FWI dibangun dari subkomponen: FFMC (kelembaban bahan bakar halus), "
            "DMC (duff moisture), DC (drought code), ISI (initial spread), BUI "
            "(buildup index), dan akhirnya FWI = f(ISI, BUI). Input utama: T, "
            "RH, kecepatan angin, dan curah hujan 24 jam."
        ),
        "variables": {
            "T": "suhu (C)",
            "RH": "kelembaban relatif (%)",
            "W": "kecepatan angin (km/jam)",
            "rain": "hujan 24 jam (mm)",
            "FFMC, DMC, DC, ISI, BUI, FWI": "komponen indeks bertingkat",
        },
        "step_by_step": [
            "Hitung FFMC dari T, RH, W, dan hujan.",
            "Hitung DMC dan DC sebagai indikator pengeringan jangka menengah.",
            "Hitung ISI = kombinasi angin dan FFMC.",
            "Hitung BUI = kombinasi DMC dan DC.",
            "Hitung FWI sebagai kombinasi ISI dan BUI.",
        ],
        "result_interpretation": (
            "FWI < 5 rendah, 5..11 sedang, 11..22 tinggi, 22..38 sangat tinggi, "
            "> 38 ekstrem (CFFWIS)."
        ),
        "business_meaning": (
            "Pengaturan patroli, pelarangan pembakaran lahan, dan kesiapan "
            "satgas damkar mengikuti tingkat FWI."
        ),
        "citations": ["Van Wagner (1987)"],
    },

    "spi_mckee": {
        "method": "Standardized Precipitation Index (SPI)",
        "purpose": "Indikator kekeringan multi-skala waktu.",
        "why_used": (
            "Direkomendasikan WMO sebagai indeks kekeringan utama karena hanya "
            "memerlukan data presipitasi panjang dan dapat dihitung pada "
            "skala 1..48 bulan."
        ),
        "real_math": (
            "Akumulasi presipitasi pada window k bulan ditransformasi menjadi "
            "skor terstandardisasi (asumsi distribusi normal setelah transformasi "
            "Gamma). SPI = (X - mean) / sd."
        ),
        "variables": {
            "X": "akumulasi hujan window k bulan",
            "mean, sd": "rata-rata dan simpangan baku historis pada window yang sama",
        },
        "step_by_step": [
            "Pilih time-scale k bulan (1, 3, 6, 12, ...).",
            "Bangun seri akumulasi sliding window dari data historis.",
            "Hitung mean dan sd dari seri tersebut.",
            "SPI terbaru = (X_terbaru - mean) / sd.",
        ],
        "result_interpretation": (
            "SPI >= 2 sangat basah; 1.5..2 basah; -1..1 normal; -1.5..-1 agak "
            "kering; <= -2 sangat kering."
        ),
        "business_meaning": (
            "Memicu peringatan dini bagi pertanian, alokasi air, dan manajemen "
            "bendungan."
        ),
        "citations": ["McKee, Doesken, Kleist (1993)"],
    },

    "gumbel_return_period": {
        "method": "Gumbel Return Level untuk Hujan Ekstrem",
        "purpose": "Memperkirakan curah hujan pada return period T tahun.",
        "why_used": (
            "Distribusi extreme value paling klasik untuk annual maxima."
        ),
        "real_math": (
            "Annual maxima diasumsikan mengikuti distribusi Gumbel dengan "
            "parameter mu dan beta. Return level: x_T = mu - beta * ln(-ln(1 - 1/T))."
        ),
        "variables": {
            "mu": "lokasi (parameter)",
            "beta": "skala (parameter)",
            "T": "return period (tahun)",
            "x_T": "nilai ekstrem yang diharapkan dilampaui sekali per T tahun",
        },
        "step_by_step": [
            "Kumpulkan annual maxima rainfall (>= 5 tahun, idealnya >= 30 tahun).",
            "Method of Moments: beta = sd * sqrt(6) / pi; mu = mean - 0.5772 * beta.",
            "Hitung x_T untuk T = 50 dan T = 100 tahun.",
        ],
        "result_interpretation": (
            "x_50 dan x_100 menjadi acuan desain saluran drainase dan tanggul."
        ),
        "business_meaning": (
            "Standar perencanaan infrastruktur tahan banjir dan asuransi parametrik."
        ),
        "citations": ["Gumbel (1958)", "Coles (2001)"],
    },

    "bayesian_beta_binomial": {
        "method": "Beta-Binomial Bayesian Update",
        "purpose": "Memperbarui keyakinan probabilitas event dengan data baru.",
        "why_used": (
            "Konjugasi Beta-Binomial menghasilkan posterior dalam bentuk tertutup, "
            "cocok untuk update online tanpa retraining."
        ),
        "real_math": (
            "Prior: p ~ Beta(alpha, beta). Setelah observasi k event dari n trial: "
            "Posterior p ~ Beta(alpha + k, beta + n - k). Posterior mean = "
            "(alpha + k) / (alpha + beta + n)."
        ),
        "variables": {
            "alpha, beta": "hyperparameter prior (pseudo-counts)",
            "k": "jumlah event teramati",
            "n": "jumlah trial",
        },
        "step_by_step": [
            "Tentukan prior Beta(alpha, beta) berdasarkan riwayat atau pengetahuan ahli.",
            "Tambahkan jumlah event dan non-event dari data baru ke alpha dan beta.",
            "Hitung posterior mean dan 95% credible interval.",
        ],
        "result_interpretation": (
            "Mean tinggi dengan CI sempit menandakan keyakinan kuat; CI lebar "
            "menandakan ketidakpastian besar."
        ),
        "business_meaning": (
            "Memungkinkan integrasi cepat sinyal multi-source (sensor, citra, "
            "pengamatan ahli) ke estimasi risiko terkini."
        ),
        "citations": ["Bayes (1763)", "Gelman dkk (2013)"],
    },

    "shap_lundberg_lee": {
        "method": "SHAP (SHapley Additive exPlanations)",
        "purpose": "Atribusi kontribusi tiap fitur terhadap prediksi tunggal.",
        "why_used": (
            "SHAP konsisten dengan teori game cooperative dan satu-satunya "
            "metode atribusi yang memenuhi local accuracy + missingness + "
            "consistency."
        ),
        "real_math": (
            "SHAP value phi_i = rata-rata marginal contribution fitur i di seluruh "
            "permutasi subset fitur. Sum semua phi_i + base value = prediksi penuh."
        ),
        "variables": {
            "phi_i": "kontribusi fitur i",
            "base_value": "prediksi rata-rata pada distribusi background",
        },
        "step_by_step": [
            "Tentukan fungsi prediktif f(x) yang ingin dijelaskan.",
            "Pilih background data sebagai referensi.",
            "Hitung phi_i melalui sampling permutasi atau exact algorithm.",
            "Visualisasikan dengan waterfall atau force plot.",
        ],
        "result_interpretation": (
            "Phi positif menambah probabilitas event; phi negatif menguranginya. "
            "Jumlah semua phi + base = prediksi akhir."
        ),
        "business_meaning": (
            "Auditor dan regulator dapat menelusuri alasan tiap peringatan, "
            "memenuhi prinsip explainable AI."
        ),
        "citations": ["Lundberg & Lee (2017)"],
    },

    "ensemble_soft_voting": {
        "method": "Ensemble Soft Voting (LSTM + XGBoost + Bayesian)",
        "purpose": "Menggabungkan prediksi multi-model menjadi probabilitas akhir.",
        "why_used": (
            "Ensemble mengurangi varians (Dietterich 2000) dan mengambil keunggulan "
            "tiap arsitektur: LSTM untuk temporal, XGBoost untuk fitur tabular, "
            "Bayesian untuk integrasi prior."
        ),
        "real_math": (
            "Soft-voting weighted: P_final = sum_i w_i * P_i, dengan sum w_i = 1. "
            "Logit-mean variant: P_final = sigmoid(sum_i w_i * logit(P_i))."
        ),
        "variables": {
            "P_i": "probabilitas model ke-i",
            "w_i": "bobot model ke-i",
        },
        "step_by_step": [
            "Hasilkan probabilitas dari tiap model.",
            "Tentukan bobot (default 0.30 LSTM, 0.45 XGBoost, 0.25 Bayesian).",
            "Hitung weighted mean atau logit-mean.",
            "Estimasi kepercayaan dari kesepakatan antar model.",
        ],
        "result_interpretation": (
            "Probabilitas akhir 0.5 ke atas dapat dikombinasikan dengan threshold "
            "WarningLevelClassifier untuk menentukan kategori peringatan."
        ),
        "business_meaning": (
            "Mengurangi false alarm dan missed detection sehingga rekomendasi "
            "lebih dapat dipercaya untuk pengambilan keputusan."
        ),
        "citations": ["Dietterich (2000)", "Sagi & Rokach (2018)"],
    },
}


def get_disaster_explanation(method_name: str):
    return DISASTER_PLAIN_EXPLANATIONS.get(method_name)


def list_disaster_methods():
    return sorted(DISASTER_PLAIN_EXPLANATIONS.keys())
