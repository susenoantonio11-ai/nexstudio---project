"""
Plain-language explanations untuk method geospatial.
Mengikuti format spec user: real math dalam bahasa nyata, step-by-step dengan angka.
"""

GEOSPATIAL_PLAIN_EXPLANATIONS = {

    # ==================================================================
    # NDWI
    # ==================================================================
    "ndwi": {
        "name": "NDWI (Normalized Difference Water Index)",
        "category": "spectral_index",
        "purpose": "Mendeteksi area air permukaan dari citra satelit Sentinel-2/Landsat.",
        "plain_language": {
            "real_math_explanation": (
                "NDWI dihitung dari perbedaan nilai green band dan near-infrared (NIR) band, "
                "lalu dibagi dengan jumlah keduanya.\n\n"
                "Rumus nyata: NDWI = (Green - NIR) / (Green + NIR)\n\n"
                "Logikanya: air sangat MENYERAP NIR (nilai NIR rendah) tapi MEMANTULKAN sedikit Green "
                "(nilai Green sedang). Jadi pixel air punya Green > NIR → NDWI positif. "
                "Vegetasi dan tanah memantulkan NIR jauh lebih kuat → NDWI negatif untuk pixel non-air."
            ),
            "variable_meaning": (
                "Green = nilai pantulan band hijau (sekitar 530-590 nm). "
                "NIR = nilai pantulan band near-infrared (sekitar 760-900 nm). "
                "Hasil NDWI berkisar -1 sampai +1. "
                "Air biasanya > 0.3, vegetasi negatif, tanah mendekati 0."
            ),
            "step_by_step_calculation": [
                "Contoh pixel dengan nilai band:",
                "  Green = 0.60",
                "  NIR = 0.20",
                "Langkah 1: Kurangkan Green dan NIR → 0.60 - 0.20 = 0.40",
                "Langkah 2: Jumlahkan Green dan NIR → 0.60 + 0.20 = 0.80",
                "Langkah 3: Bagi hasil pengurangan dengan hasil penjumlahan → 0.40 / 0.80 = 0.50",
                "Hasil: NDWI = 0.50",
            ],
            "result_interpretation": (
                "NDWI = 0.50 menunjukkan kemungkinan area air cukup kuat. "
                "Skala: > 0.3 → AIR; 0 sampai 0.3 → mungkin air dangkal/lembap; "
                "< 0 → bukan air (vegetasi/tanah/built-up)."
            ),
            "business_meaning": (
                "Untuk research banjir: pixel dengan NDWI > 0.3 ditandai sebagai potential flood. "
                "PERHATIAN: NDWI sering false-positive di area built-up (gedung). "
                "Untuk akurasi lebih tinggi, gunakan MNDWI (modifikasi dengan SWIR band)."
            ),
        },
        "limitations": [
            "False-positive di area built-up (gedung memantulkan green seperti air)",
            "Tergantung kualitas atmospheric correction citra",
            "Threshold harus disesuaikan dengan kondisi lokal/musim",
        ],
        "reference": "McFeeters (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing, 17(7), 1425-1432.",
    },

    # ==================================================================
    # MNDWI
    # ==================================================================
    "mndwi": {
        "name": "MNDWI (Modified NDWI)",
        "category": "spectral_index",
        "purpose": "Versi superior dari NDWI yang lebih akurat untuk deteksi air, terutama di area urban.",
        "plain_language": {
            "real_math_explanation": (
                "MNDWI = (Green - SWIR) / (Green + SWIR)\n\n"
                "Bedanya dengan NDWI: ganti NIR dengan SWIR (Short-Wave Infrared).\n\n"
                "Mengapa SWIR lebih baik?\n"
                "Air menyerap SWIR JAUH LEBIH KUAT daripada NIR — jadi kontras air vs non-air sangat tajam. "
                "Selain itu, gedung yang sering false-positive di NDWI (memantulkan NIR rendah seperti air) "
                "AKAN memantulkan SWIR tinggi (berbeda dari air) → MNDWI bisa membedakan dengan jelas."
            ),
            "variable_meaning": (
                "Green = band 3 di Sentinel-2 (560 nm) atau band 2 di Landsat-8. "
                "SWIR = band 11 atau 12 di Sentinel-2 (1610 / 2190 nm). "
                "Hasil sama dengan NDWI: -1 sampai +1, > 0 atau > 0.3 = air."
            ),
            "step_by_step_calculation": [
                "Contoh: pixel di area dekat sungai",
                "  Green = 0.55",
                "  SWIR = 0.10  (sangat rendah karena air menyerap SWIR)",
                "Langkah 1: Kurangkan → 0.55 - 0.10 = 0.45",
                "Langkah 2: Jumlahkan → 0.55 + 0.10 = 0.65",
                "Langkah 3: Bagi → 0.45 / 0.65 = 0.692",
                "Hasil: MNDWI = 0.69 → AIR (sangat yakin)",
                "",
                "Bandingkan dengan pixel gedung beton:",
                "  Green = 0.40, SWIR = 0.35",
                "  MNDWI = (0.40-0.35)/(0.40+0.35) = 0.067 → BUKAN air (built-up)",
                "Padahal NDWI gedung tersebut bisa positif → false-positive!",
            ],
            "result_interpretation": (
                "MNDWI > 0.3 → AIR confirmed. "
                "0 < MNDWI < 0.3 → kemungkinan air dangkal atau wet soil. "
                "MNDWI < 0 → bukan air. "
                "MNDWI > 0.69 → air dalam dengan confidence tinggi."
            ),
            "business_meaning": (
                "Untuk flood mapping di area urban (Jakarta, Surabaya, dll), MNDWI WAJIB dipakai "
                "menggantikan NDWI. Akurasi naik signifikan di area mixed land cover. "
                "Standard industri untuk Sentinel-2 flood mapping."
            ),
        },
        "limitations": [
            "Membutuhkan SWIR band (tidak semua sensor punya)",
            "Sentinel-2 punya SWIR; Landsat-8 punya; tapi bukan semua imagery memilikinya",
            "Tetap perlu cloud masking sebelum perhitungan",
        ],
        "reference": "Xu, H. (2006). Modification of normalized difference water index (NDWI) to enhance open water features in remotely sensed imagery. International Journal of Remote Sensing, 27(14), 3025-3033.",
    },

    # ==================================================================
    # NDVI
    # ==================================================================
    "ndvi": {
        "name": "NDVI (Normalized Difference Vegetation Index)",
        "category": "spectral_index",
        "purpose": "Mengukur kepadatan/kesehatan vegetasi.",
        "plain_language": {
            "real_math_explanation": (
                "NDVI = (NIR - Red) / (NIR + Red)\n\n"
                "Logikanya: vegetasi sehat MENYERAP banyak Red (untuk fotosintesis) tapi "
                "MEMANTULKAN banyak NIR (untuk menghindari overheating). "
                "Jadi NIR >> Red → NDVI positif besar. Air kebalikan: memantulkan sedikit NIR → NDVI negatif."
            ),
            "variable_meaning": (
                "NIR = band near-infrared. Red = band merah (visible). "
                "Range hasil: -1 sampai +1."
            ),
            "step_by_step_calculation": [
                "Contoh pixel hutan tropis:",
                "  Red = 0.05 (hijau menyerap merah)",
                "  NIR = 0.50 (hijau memantulkan NIR)",
                "Langkah 1: NIR - Red = 0.50 - 0.05 = 0.45",
                "Langkah 2: NIR + Red = 0.50 + 0.05 = 0.55",
                "Langkah 3: NDVI = 0.45 / 0.55 = 0.818",
                "Hasil: NDVI = 0.82 → vegetasi sangat sehat",
            ],
            "result_interpretation": (
                "NDVI > 0.4 → vegetasi sehat. "
                "0.2 - 0.4 → vegetasi sparse atau stressed. "
                "0 - 0.2 → tanah / batuan / built-up. "
                "< 0 → AIR atau salju."
            ),
            "business_meaning": (
                "Dalam research banjir: NDVI digunakan untuk MEMBANDINGKAN before-after. "
                "Area yang NDVI-nya turun drastis (misal: dari 0.7 → 0.05) menandakan area yang "
                "sebelumnya hijau menjadi tergenang air. Salah satu indikator utama flood damage assessment."
            ),
        },
        "limitations": [
            "Saturasi pada vegetasi sangat lebat (tidak bisa bedakan hutan lebat vs sangat lebat)",
            "Sensitif terhadap atmospheric noise",
        ],
        "reference": "Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. Remote Sensing of Environment, 8(2), 127-150.",
    },

    # ==================================================================
    # SAR THRESHOLD
    # ==================================================================
    "sar_vv_threshold": {
        "name": "SAR VV Backscatter Threshold for Flood Detection",
        "category": "sar_method",
        "purpose": "Deteksi air permukaan menggunakan radar SAR — tembus awan, siang/malam.",
        "plain_language": {
            "real_math_explanation": (
                "SAR (Synthetic Aperture Radar) seperti Sentinel-1 mengirim sinyal radar ke Bumi "
                "lalu mengukur seberapa kuat sinyal yang dipantulkan kembali (backscatter).\n\n"
                "Air permukaan (tenang) bertindak seperti CERMIN — sinyal radar dipantulkan menjauh dari satelit. "
                "Akibatnya: nilai backscatter SANGAT RENDAH (sangat negatif dalam dB).\n\n"
                "Aturan threshold: pixel dengan VV < -17 dB → AIR.\n\n"
                "Skala dB: VV -5 dB (kuat), VV -10 dB (sedang), VV -15 dB (lemah), VV -20 dB (sangat lemah/air)."
            ),
            "variable_meaning": (
                "VV = polarisasi vertikal-vertikal — sinyal radar dikirim dengan polarisasi vertikal "
                "dan diterima dengan polarisasi vertikal. "
                "VH = vertikal dikirim, horizontal diterima — lebih sensitif terhadap volume scattering. "
                "Nilai dalam dB (decibel) — log scale."
            ),
            "step_by_step_calculation": [
                "Contoh: pixel di sungai dengan air tenang",
                "  VV = -19.5 dB",
                "Langkah 1: Bandingkan dengan threshold → -19.5 < -17 → TRUE",
                "Hasil: pixel = AIR (FLOOD)",
                "",
                "Contoh: pixel di hutan",
                "  VV = -8.2 dB",
                "Langkah 1: -8.2 < -17? → FALSE",
                "Hasil: pixel = NON-FLOOD (hutan memantulkan radar dengan kuat)",
            ],
            "result_interpretation": (
                "Threshold -17 dB adalah default validated di literatur. "
                "Bisa di-tune untuk kondisi lokal: dataset urban mungkin butuh -15 dB; "
                "rural mungkin -19 dB. "
                "Pixel di bawah threshold → flagged sebagai air/flood."
            ),
            "business_meaning": (
                "Keunggulan SAR: TIDAK terhalang awan dan bisa beroperasi malam — perfect untuk flood emergency. "
                "Sentinel-1 punya revisit 6-12 hari, gratis di Google Earth Engine. "
                "Limitation: angin yang menggelombangkan air bisa menaikkan backscatter (false-negative); "
                "permukaan halus seperti tarmac bisa false-positive."
            ),
        },
        "limitations": [
            "Wind-roughened water bisa false-negative",
            "Smooth man-made surfaces (asphalt, tarmac) bisa false-positive",
            "Threshold perlu di-tune per kondisi lokal",
        ],
        "reference": "Twele, A., Cao, W., Plank, S., & Martinis, S. (2016). Sentinel-1-based flood mapping: a fully automated processing chain. International Journal of Remote Sensing, 37(13), 2990-3004.",
    },

    # ==================================================================
    # IOU (INTERSECTION OVER UNION)
    # ==================================================================
    "iou": {
        "name": "IoU (Intersection over Union) / Jaccard Index",
        "category": "segmentation_metric",
        "purpose": "Metrik standar untuk evaluasi segmentation/flood mapping accuracy.",
        "plain_language": {
            "real_math_explanation": (
                "IoU = OVERLAP antara prediksi dan ground truth, dibagi dengan TOTAL area gabungan.\n\n"
                "Rumus: IoU = |Intersection| / |Union| = TP / (TP + FP + FN)\n\n"
                "Bayangkan dua himpunan area:\n"
                "• Ground truth (G): area aktual yang banjir\n"
                "• Prediksi (P): area yang model bilang banjir\n\n"
                "Intersection = pixel yang DI KEDUA G dan P (model benar-benar tepat).\n"
                "Union = pixel yang DI G ATAU P ATAU keduanya (semua area yang melibatkan).\n\n"
                "IoU = berapa proporsi 'union' yang berhasil dipotong tepat oleh model."
            ),
            "variable_meaning": (
                "TP = True Positive (pixel flood, prediksi flood) — overlap. "
                "FP = False Positive (pixel non-flood, prediksi flood) — over-prediction. "
                "FN = False Negative (pixel flood, prediksi non-flood) — missed. "
                "Range: [0, 1]. 1 = perfect, 0 = no overlap sama sekali."
            ),
            "step_by_step_calculation": [
                "Contoh: 100 pixel total, dengan kondisi sebenarnya (ground truth):",
                "  60 pixel banjir, 40 pixel kering",
                "Prediksi model:",
                "  TP (flood, prediksi flood) = 50",
                "  FP (kering, prediksi flood) = 8",
                "  FN (flood, prediksi kering) = 10",
                "  TN (kering, prediksi kering) = 32",
                "Langkah 1: Intersection = TP = 50",
                "Langkah 2: Union = TP + FP + FN = 50 + 8 + 10 = 68",
                "Langkah 3: IoU = 50 / 68 = 0.735",
                "Hasil: IoU = 0.74",
            ],
            "result_interpretation": (
                "IoU = 0.74 berarti 74% area yang seharusnya 'flood' atau dianggap 'flood' "
                "berhasil di-segment dengan tepat. "
                "Skala: > 0.75 EXCELLENT (publication quality), 0.5-0.75 GOOD, "
                "0.3-0.5 FAIR (perlu perbaikan), < 0.3 POOR."
            ),
            "business_meaning": (
                "Untuk flood mapping: IoU > 0.7 dianggap reliable untuk operational use "
                "(emergency response, damage assessment). "
                "Untuk publikasi penelitian: IoU > 0.75 paper standar; > 0.85 SOTA. "
                "Lebih ketat dari F1 — bisa F1=0.8 tapi IoU=0.66, karena IoU tidak counts TN."
            ),
        },
        "limitations": [
            "Tidak counts True Negative — semua bias terhadap class minor",
            "Sensitif untuk class kecil (sedikit FN bisa drop IoU drastis)",
            "Untuk multi-class: rata-rata IoU per class (mIoU)",
        ],
        "vs_f1": "IoU lebih ketat dari F1. F1 = 2·TP/(2·TP+FP+FN), IoU = TP/(TP+FP+FN). Untuk binary segmentation, IoU < F1 selalu.",
    },

    # ==================================================================
    # COHEN'S KAPPA
    # ==================================================================
    "cohen_kappa": {
        "name": "Cohen's Kappa Coefficient",
        "category": "classification_metric",
        "purpose": "Mengukur agreement antara prediksi dan aktual, DI ATAS chance — robust terhadap imbalance.",
        "plain_language": {
            "real_math_explanation": (
                "Kappa menjawab: 'Berapa banyak agreement model yang bukan kebetulan?'\n\n"
                "Rumus: κ = (Pₒ - Pₑ) / (1 - Pₑ)\n"
                "  Pₒ = OBSERVED agreement = accuracy aktual = (TP+TN) / total\n"
                "  Pₑ = EXPECTED agreement by chance — agreement yang akan terjadi jika model menebak random\n\n"
                "Logikanya: kalau dataset 90% kelas A, model dummy yang selalu prediksi A = 90% accuracy. "
                "Kappa MENGOREKSI ini — jika model tidak lebih baik dari chance, Kappa = 0."
            ),
            "variable_meaning": (
                "Pₒ = total prediksi benar / total. "
                "Pₑ = probabilitas agreement by chance, dihitung dari distribusi marginal. "
                "Kappa = -1 (worst), 0 (random), +1 (perfect)."
            ),
            "step_by_step_calculation": [
                "Contoh confusion matrix:",
                "                      Pred Flood  Pred Non-Flood",
                "  Aktual Flood:        50          10           (60 total)",
                "  Aktual Non-Flood:    8           32           (40 total)",
                "  Total prediksi:      58          42           (100 total)",
                "Langkah 1: Pₒ = (50 + 32) / 100 = 0.82",
                "Langkah 2: Hitung expected agreement (Pₑ):",
                "  P(flood by chance) = (60/100) × (58/100) = 0.348",
                "  P(non-flood by chance) = (40/100) × (42/100) = 0.168",
                "  Pₑ = 0.348 + 0.168 = 0.516",
                "Langkah 3: Kappa = (Pₒ - Pₑ) / (1 - Pₑ) = (0.82 - 0.516) / (1 - 0.516) = 0.304 / 0.484 = 0.628",
                "Hasil: Kappa = 0.63",
            ],
            "result_interpretation": (
                "Kappa interpretation (Landis & Koch, 1977):\n"
                "  < 0.20 → SLIGHT agreement\n"
                "  0.21-0.40 → FAIR\n"
                "  0.41-0.60 → MODERATE\n"
                "  0.61-0.80 → SUBSTANTIAL\n"
                "  > 0.80 → ALMOST PERFECT\n"
                "Kappa = 0.63 = SUBSTANTIAL agreement → model bagus."
            ),
            "business_meaning": (
                "Kappa adalah STANDARD untuk remote sensing classification accuracy. "
                "Lebih jujur dari accuracy — tidak misleading di imbalanced flood data. "
                "Untuk publikasi penelitian banjir: lapor Kappa wajib selain F1/IoU."
            ),
        },
        "limitations": [
            "Bisa terlalu pesimis untuk certain class distributions",
            "Tidak ada interpretasi tunggal (skala Landis-Koch debatable)",
        ],
        "reference": "Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. Biometrics, 33(1), 159-174.",
    },

    # ==================================================================
    # U-NET
    # ==================================================================
    "unet": {
        "name": "U-Net Architecture (Semantic Segmentation)",
        "category": "deep_learning",
        "purpose": "Mengklasifikasi setiap pixel raster menjadi flood/non-flood menggunakan deep learning.",
        "plain_language": {
            "real_math_explanation": (
                "U-Net adalah convolutional neural network berbentuk huruf 'U'. "
                "Punya dua bagian:\n\n"
                "1. ENCODER (sisi kiri huruf U) — citra di-COMPRESS bertahap dengan convolution + pooling. "
                "Setiap level menangkap feature semakin abstrak: tekstur → bentuk → semantic.\n\n"
                "2. DECODER (sisi kanan huruf U) — fitur abstrak di-EKSPAND kembali ke ukuran asli "
                "menggunakan upsampling.\n\n"
                "RAHASIA U-NET = SKIP CONNECTIONS: setiap level decoder menerima 'salinan' dari level encoder yang sama. "
                "Ini menjaga DETAIL SPATIAL tetap presisi pada output, sehingga prediksi bisa pixel-perfect."
            ),
            "variable_meaning": (
                "Convolution (Conv2d) = filter yang scan citra, deteksi pattern. "
                "BatchNorm = normalisasi aktivasi per batch, mempercepat training. "
                "ReLU = activation function, hanya pertahankan nilai positif. "
                "MaxPool = downsampling 2x2 (ambil max), reduce ukuran. "
                "Upsample = kembalikan ukuran. "
                "Skip connection = concatenate fitur encoder ke decoder yang sama level."
            ),
            "step_by_step_calculation": [
                "Contoh input: citra Sentinel-2 patch ukuran 256×256 dengan 4 band (RGB+NIR)",
                "Shape input: (4, 256, 256)",
                "Langkah 1 — Encoder Block 1: 64 filter conv → output (64, 256, 256). Lalu MaxPool → (64, 128, 128)",
                "Langkah 2 — Encoder Block 2: 128 filter → (128, 128, 128). MaxPool → (128, 64, 64)",
                "Langkah 3 — Encoder Block 3: 256 filter → (256, 64, 64). MaxPool → (256, 32, 32)",
                "Langkah 4 — Encoder Block 4: 512 filter → (512, 32, 32). MaxPool → (512, 16, 16)",
                "Langkah 5 — Bottleneck: 1024 filter → (1024, 16, 16)",
                "Langkah 6 — Decoder Block 4: Upsample 2x → (1024, 32, 32). Concat skip dari Encoder 4 → (1024+512=1536, 32, 32). Conv → (512, 32, 32)",
                "...lanjut decoder block 3, 2, 1...",
                "Langkah final: 1×1 conv → (1, 256, 256). Sigmoid → probability flood per pixel.",
                "Total parameter: ~7.7 juta (default base_filters=64)",
            ],
            "result_interpretation": (
                "Output: tensor (1, H, W) dengan nilai probability di [0, 1]. "
                "Threshold 0.5 default → binary mask: pixel = flood jika probability > 0.5. "
                "Semakin akurat training data, semakin halus dan presisi mask yang dihasilkan."
            ),
            "business_meaning": (
                "U-Net JAUH lebih akurat dari pixel-based classifier (Random Forest pixel) "
                "karena menangkap CONTEXT spasial: gedung yang dikelilingi air → flooded; "
                "gedung di atas bukit kering → non-flooded. "
                "Pre-trained Encoder (transfer learning dari ResNet/EfficientNet) "
                "bisa mengurangi data labeled yang dibutuhkan dari ~10000 → ~1000 patches."
            ),
        },
        "limitations": [
            "Butuh banyak labeled training data (~1000 patches minimal)",
            "Training memakan GPU memory (default 7.7M params)",
            "Sensitif terhadap class imbalance — pakai Dice loss atau Focal loss",
            "Inference time lebih lambat dari pixel-based classifier",
        ],
        "reference": "Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.",
    },

    # ==================================================================
    # DICE LOSS
    # ==================================================================
    "dice_loss": {
        "name": "Dice Loss (untuk Segmentation)",
        "category": "loss_function",
        "purpose": "Loss function untuk semantic segmentation, robust terhadap class imbalance.",
        "plain_language": {
            "real_math_explanation": (
                "Dice Loss = 1 - Dice Coefficient.\n\n"
                "Dice coefficient mengukur OVERLAP antara prediksi dan target:\n"
                "Dice = (2 × overlap) / (jumlah_pixel_pred + jumlah_pixel_target)\n\n"
                "Range Dice: [0, 1]. 0 = no overlap, 1 = perfect overlap. "
                "Loss = 1 - Dice, jadi: 0 = perfect, 1 = worst.\n\n"
                "MENGAPA cocok untuk segmentation imbalanced?\n"
                "Tidak counts True Negative (background pixels). "
                "BCE biasa over-fit ke majority class kalau data imbalanced. "
                "Dice fokus pada region of interest (flooded pixels)."
            ),
            "variable_meaning": (
                "p = predicted probability per pixel (setelah sigmoid). "
                "y = target binary mask (0 atau 1). "
                "intersection = jumlah pixel di mana p×y tinggi (keduanya menyatakan flood). "
                "smooth = +1 di numerator dan denominator untuk hindari divide-by-zero."
            ),
            "step_by_step_calculation": [
                "Contoh: prediksi vs target untuk 6 pixel",
                "  prediksi (p): [0.9, 0.8, 0.1, 0.7, 0.2, 0.1]",
                "  target (y): [1,   1,   0,   1,   0,   0]",
                "Langkah 1: Hitung intersection = Σ(p × y)",
                "  = 0.9×1 + 0.8×1 + 0.1×0 + 0.7×1 + 0.2×0 + 0.1×0 = 2.4",
                "Langkah 2: Hitung jumlah p = 0.9+0.8+0.1+0.7+0.2+0.1 = 2.8",
                "Langkah 3: Hitung jumlah y = 1+1+0+1+0+0 = 3",
                "Langkah 4: Dice = (2 × 2.4 + 1) / (2.8 + 3 + 1) = 5.8 / 6.8 = 0.853",
                "Langkah 5: Dice Loss = 1 - 0.853 = 0.147",
            ],
            "result_interpretation": (
                "Dice Loss = 0.147 (mendekati 0) → prediksi & target overlap kuat → model bagus. "
                "Backpropagation akan minimize loss ini, mendorong overlap mendekati 100%."
            ),
            "business_meaning": (
                "Untuk flood segmentation: pakai BCE+Dice combined (biasanya 50:50). "
                "BCE jaga akurasi pixel-level, Dice jaga overlap region-level. "
                "Bagus terutama saat flood pixels < 20% dari total (sangat imbalanced)."
            ),
        },
        "limitations": [
            "Bisa unstable di awal training (gradient besar)",
            "Tidak detect 'kapan harus berhenti predict positif' seperti BCE",
            "Recommended: combine dengan BCE atau Focal loss",
        ],
        "reference": "Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.",
    },

    # ==================================================================
    # GEE COMPOSITE
    # ==================================================================
    "gee_median_composite": {
        "name": "Google Earth Engine Median Composite",
        "category": "remote_sensing",
        "purpose": "Menghasilkan citra cloud-free dari multiple acquisitions di periode waktu tertentu.",
        "plain_language": {
            "real_math_explanation": (
                "Median composite mengambil NILAI TENGAH per pixel dari beberapa citra "
                "yang diambil pada hari berbeda di periode yang sama.\n\n"
                "Misal, satu lokasi punya 8 citra Sentinel-2 dalam Januari 2024. "
                "Beberapa citra mungkin tertutup awan di area X, beberapa tidak. "
                "Untuk SETIAP pixel, ambil 'median' dari 8 nilai per band → cloud-free image."
            ),
            "variable_meaning": (
                "Per pixel (lat, lon): nilai 8 acquisitions = [v1, v2, ..., v8]. "
                "Median = nilai tengah saat diurutkan. "
                "Berbeda dari mean: tidak bias oleh outlier (cloud yang sangat reflektif)."
            ),
            "step_by_step_calculation": [
                "Contoh untuk satu pixel di Jakarta (band B4 Red), 5 acquisitions:",
                "  2024-01-05: 0.18 (clear)",
                "  2024-01-10: 0.95 (cloud!)",
                "  2024-01-15: 0.21 (clear)",
                "  2024-01-20: 0.19 (clear)",
                "  2024-01-25: 0.85 (cloud!)",
                "Langkah 1: Sort → [0.18, 0.19, 0.21, 0.85, 0.95]",
                "Langkah 2: Median = nilai posisi tengah = 0.21",
                "Hasil: cloud-free pixel value = 0.21 (jauh lebih akurat dari mean=0.476 yang bias)",
            ],
            "result_interpretation": (
                "Median composite menghasilkan citra yang konsisten di seluruh AOI. "
                "Berbeda dari single image yang mungkin tertutup awan, atau mean composite yang bias outlier. "
                "Ini adalah TEKNIK STANDAR di Earth Engine untuk fixed-date analysis."
            ),
            "business_meaning": (
                "Untuk research banjir: pakai PRE-FLOOD median (1 bulan sebelum) sebagai baseline, "
                "POST-FLOOD median (saat banjir) sebagai treatment. "
                "Difference = perubahan akibat banjir. "
                "Menghemat preprocessing manual cloud masking."
            ),
        },
        "limitations": [
            "Tidak handle very dynamic land cover changes within composite period",
            "Gagal kalau SEMUA acquisitions ada awan (median tetap awan)",
            "Untuk SAR Sentinel-1: median juga reduce noise tapi kurang efektif",
        ],
        "reference": "Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone.",
    },

    # ==================================================================
    # TRANSFER LEARNING
    # ==================================================================
    "transfer_learning": {
        "name": "Transfer Learning untuk Flood Segmentation",
        "category": "deep_learning",
        "purpose": "Memanfaatkan model yang sudah dilatih pada dataset besar untuk tugas baru dengan data terbatas.",
        "plain_language": {
            "real_math_explanation": (
                "Transfer Learning bekerja dalam 3 langkah:\n\n"
                "1. PRETRAIN: Model dilatih pada dataset BESAR (misal Sen1Floods11 dengan 4831 patches global). "
                "Network belajar feature general: edge, texture, water signature, dll.\n\n"
                "2. FREEZE: Lapisan ENCODER (yang menangkap feature general) dibekukan. "
                "Berat tidak di-update saat fine-tuning awal. Hanya DECODER yang belajar.\n\n"
                "3. FINE-TUNE: Setelah decoder converge, encoder di-UNFREEZE dengan learning rate 10× lebih kecil. "
                "Semua layer di-update sedikit, men-spesifikasi model untuk domain target."
            ),
            "variable_meaning": (
                "Pretrained weights = parameter (W, b) yang sudah optimal untuk task umum. "
                "Frozen = parameter tidak di-update (gradient = 0). "
                "Learning rate = ukuran langkah update; kecil → perubahan halus, besar → bisa catastrophic forgetting."
            ),
            "step_by_step_calculation": [
                "Contoh praktis: pretrained U-Net Sen1Floods11 (IoU 0.79) → fine-tune untuk Jakarta",
                "Langkah 1: Load pretrained weights dari Zenodo/Github → load_state_dict()",
                "Langkah 2: Sesuaikan input layer kalau channel berbeda (Sen1Floods11 fusion=6 channel, lokal misal 4 channel) → adapt_input_channels()",
                "Langkah 3: Freeze encoder (inc, down1, down2, down3, down4) → 4.5M parameter frozen",
                "Langkah 4: Train decoder hanya, lr = 1e-3, 5-10 epoch",
                "Langkah 5: Validasi: IoU naik dari ~0.55 (initial) → ~0.72",
                "Langkah 6: Unfreeze encoder, lr = 1e-4 (10× lebih kecil), 5 epoch lagi",
                "Langkah 7: Final IoU lokal: ~0.78 (hampir sama dengan pretrained!)",
                "Total training: hanya ~3 jam, vs ~3 hari training from scratch",
            ],
            "result_interpretation": (
                "Pengurangan data labeled yang dibutuhkan: dari 1000+ patches → 100-200 patches. "
                "Pengurangan training time: dari 3 hari → 3 jam. "
                "Akurasi yang dicapai: ~95-98% dari level pretrained. "
                "Risk: jika domain shift terlalu besar (misal Sen1Floods11 fokus tropical, target arctic), "
                "transfer learning kurang efektif."
            ),
            "business_meaning": (
                "Untuk research skripsi: STRONGLY RECOMMENDED — kamu tidak punya 1000 patches manual labeling. "
                "Workflow: Sen1Floods11 pretrained → adapt input → fine-tune dengan 100-200 patches Jakarta → publish. "
                "Hemat waktu, hemat data, hemat compute, hasil tetap state-of-the-art."
            ),
        },
        "limitations": [
            "Domain shift terlalu besar = transfer learning kurang efektif",
            "Pretrained weights mungkin punya bias dari training data",
            "Butuh karefulness: jangan langsung unfreeze semua dengan lr besar",
        ],
        "reference": "Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? NeurIPS.",
    },

    # ==================================================================
    # WMS PROTOCOL
    # ==================================================================
    "wms_protocol": {
        "name": "WMS (Web Map Service) — OGC Standard",
        "category": "geospatial_serving",
        "purpose": "Protocol standar untuk publish geospatial data sebagai web service yang bisa di-request via HTTP.",
        "plain_language": {
            "real_math_explanation": (
                "WMS bekerja sebagai REQUEST-RESPONSE protocol:\n\n"
                "1. CLIENT (browser/Leaflet/QGIS) request peta ke SERVER (GeoServer):\n"
                "   GET https://geoserver/wms?\n"
                "       SERVICE=WMS&REQUEST=GetMap\n"
                "       &LAYERS=jakarta:flood_mask\n"
                "       &BBOX=-6.3,106.7,-6.1,106.9    # area yang diinginkan\n"
                "       &WIDTH=512&HEIGHT=512           # ukuran image\n"
                "       &CRS=EPSG:4326                  # projection\n"
                "       &FORMAT=image/png               # output format\n"
                "       &TRANSPARENT=true               # alpha channel\n\n"
                "2. SERVER bertindak: read GeoTIFF, clip ke bbox, reproject jika perlu, "
                "render dengan SLD style, return PNG.\n\n"
                "3. CLIENT terima PNG → render sebagai layer di map. "
                "Saat user pan/zoom, request baru dengan bbox berbeda."
            ),
            "variable_meaning": (
                "BBOX = area geografis yang diminta (lat/lon atau projected coords). "
                "WIDTH/HEIGHT = pixel dimensions output image. "
                "CRS = Coordinate Reference System (EPSG:4326 = lat/lon WGS84). "
                "LAYERS = nama layer yang publishd di server. "
                "FORMAT = MIME type response (image/png paling umum)."
            ),
            "step_by_step_calculation": [
                "Contoh real: load flood mask dari GeoServer di Leaflet",
                "Langkah 1: Setup GeoServer (Docker): docker run -p 8080:8080 docker.osgeo.org/geoserver:latest",
                "Langkah 2: Upload GeoTIFF flood_mask.tif ke workspace 'jakarta':",
                "  POST /rest/workspaces/jakarta/coveragestores/flood_store/file.geotiff",
                "Langkah 3: GeoServer auto-publish layer 'jakarta:flood_mask'",
                "Langkah 4: Apply SLD style untuk color ramp (optional)",
                "Langkah 5: Di Leaflet:",
                "  L.tileLayer.wms('http://geoserver/jakarta/wms', {",
                "    layers: 'jakarta:flood_mask',",
                "    format: 'image/png',",
                "    transparent: true,",
                "  }).addTo(map);",
                "Hasil: layer otomatis muncul di map, scale dengan zoom, query-able",
            ],
            "result_interpretation": (
                "WMS membuat data geospatial INTEROPERABLE: GIS apapun (QGIS, ArcGIS, Leaflet, Mapbox) "
                "bisa konsumsi tanpa modifikasi. "
                "Cocok untuk publish hasil research ke audience luas (peneliti, pemerintah, masyarakat)."
            ),
            "business_meaning": (
                "Untuk research banjir Indonesia: publish hasil U-Net flood mask ke GeoServer, "
                "share URL ke BNPB, BMKG, atau aplikasi early warning lokal. "
                "Mereka bisa load di sistem mereka tanpa perlu ngedownload GeoTIFF berulang. "
                "WMTS (variant) lebih cepat karena pre-render tile (cached)."
            ),
        },
        "limitations": [
            "Latensi network: setiap pan/zoom = request baru ke server",
            "Server load tinggi jika banyak concurrent users → pakai WMTS untuk caching",
            "Auth/security perlu di-setup terpisah (default GeoServer admin/geoserver)",
        ],
        "reference": "OGC WMS Specification 1.3.0 — Open Geospatial Consortium standard.",
    },

    # ==================================================================
    # TIME-SERIES ANALYSIS
    # ==================================================================
    "time_series_flood_evolution": {
        "name": "Time-Series Flood Evolution Tracking",
        "category": "temporal_analysis",
        "purpose": "Track perubahan extent banjir over time untuk detect peak, durasi, dan recovery rate.",
        "plain_language": {
            "real_math_explanation": (
                "Untuk setiap timestamp tᵢ:\n"
                "1. Fetch satellite image saat tᵢ\n"
                "2. Apply flood detection → flood mask Mᵢ\n"
                "3. Compute statistics:\n"
                "   • flooded_pixels(tᵢ) = sum(Mᵢ)\n"
                "   • flooded_pct(tᵢ) = flooded_pixels(tᵢ) / total_pixels × 100\n"
                "   • area_km²(tᵢ) = flooded_pixels(tᵢ) × pixel_area\n\n"
                "Dari time-series:\n"
                "• Peak time = argmax(flooded_pct(t))\n"
                "• Duration = (last_t with pct > threshold) - (first_t with pct > threshold)\n"
                "• Recovery rate = (peak_pct - final_pct) / time_to_recede\n"
                "• Phase classification: bandingkan pct(tᵢ) vs pct(tᵢ₋₁)\n"
                "  - rising (Δ > +1%)\n"
                "  - receding (Δ < -1%)\n"
                "  - stable (-1% ≤ Δ ≤ +1%)"
            ),
            "variable_meaning": (
                "tᵢ = timestamp ke-i (biasanya beberapa hari interval sesuai satellite revisit). "
                "Mᵢ = binary flood mask di tᵢ. "
                "Δ = perubahan flooded percentage antar timestamps. "
                "Threshold default: 1% perubahan minimum untuk dianggap rising/receding."
            ),
            "step_by_step_calculation": [
                "Contoh: flood Jakarta Januari 2024, monitor 5 timestamps",
                "  T1 (2024-01-01): flooded = 2.1% (baseline)",
                "  T2 (2024-01-08): flooded = 8.5% → rising (Δ = +6.4%)",
                "  T3 (2024-01-15): flooded = 14.0% → rising (Δ = +5.5%) — PEAK",
                "  T4 (2024-01-22): flooded = 9.2% → receding (Δ = -4.8%)",
                "  T5 (2024-01-29): flooded = 3.5% → receding (Δ = -5.7%)",
                "Analisis:",
                "Langkah 1: Peak time = T3 (2024-01-15), pct = 14.0%, area = 26.4 km²",
                "Langkah 2: Duration di atas baseline = T2 → T5 = 21 hari",
                "Langkah 3: Recovery rate = (14% - 3.5%) / 14 hari = 0.75% per hari",
                "Langkah 4: Phase summary: rising (T1→T3), receding (T3→T5)",
            ],
            "result_interpretation": (
                "Peak time + lokasi = INFORMASI EMERGENCY KRITIS untuk evacuation planning. "
                "Recovery rate = indikator drainase area (cepat = drainase baik). "
                "Phase visualization = animasi yang mudah dipahami stakeholder."
            ),
            "business_meaning": (
                "Time-series PALING BERHARGA dibanding single snapshot: \n"
                "1. Emergency response: prediksi peak time + tingkat keparahan\n"
                "2. Insurance assessment: berapa lama property tergenang\n"
                "3. Climate research: pola temporal banjir multi-tahun\n"
                "4. Infrastructure planning: identify recurring flood zones\n"
                "Animasi di frontend Leaflet membuat data ini ENGAGING & ACTIONABLE."
            ),
        },
        "limitations": [
            "Frekuensi tergantung satellite revisit (Sentinel-2 ~5 hari, Sentinel-1 ~6 hari)",
            "Cloud cover bisa miss event puncak",
            "Resolusi temporal kalah dari sensor in-situ",
            "Computation berat untuk area besar × banyak timestamp",
        ],
        "reference": "Lin et al. (2019). Time-series flood mapping with Sentinel-1 SAR. Remote Sensing of Environment, 222, 184-203.",
    },

    # ==================================================================
    # NDBI
    # ==================================================================
    "ndbi": {
        "name": "NDBI (Normalized Difference Built-up Index)",
        "category": "spectral_index",
        "purpose": "Mendeteksi area built-up (urban) dari citra multispektral.",
        "plain_language": {
            "real_math_explanation": (
                "NDBI = (SWIR - NIR) / (SWIR + NIR)\n\n"
                "Logikanya: gedung & infrastruktur (beton, asphalt) memantulkan SWIR LEBIH KUAT "
                "daripada NIR. Vegetasi sebaliknya: NIR > SWIR. Jadi NDBI > 0 → built-up."
            ),
            "variable_meaning": (
                "SWIR = band Short-Wave Infrared. NIR = Near-Infrared. "
                "Range: -1 sampai +1. > 0.1 → likely built-up."
            ),
            "step_by_step_calculation": [
                "Contoh pixel kawasan industri:",
                "  SWIR = 0.40, NIR = 0.25",
                "Langkah 1: SWIR - NIR = 0.40 - 0.25 = 0.15",
                "Langkah 2: SWIR + NIR = 0.40 + 0.25 = 0.65",
                "Langkah 3: NDBI = 0.15 / 0.65 = 0.231",
                "Hasil: NDBI = 0.23 → BUILT-UP confirmed",
            ],
            "result_interpretation": (
                "NDBI > 0.1 → built-up area. "
                "0 sampai 0.1 → mixed (urban-vegetation). "
                "< 0 → natural surface (vegetation/water)."
            ),
            "business_meaning": (
                "Untuk damage assessment pasca banjir di area urban: cross-reference NDBI "
                "(area built-up) dengan flood mask → identifikasi infrastructure yang terdampak."
            ),
        },
        "limitations": [
            "Confused dengan bare soil yang juga memantulkan SWIR tinggi",
            "Menggunakan bersama NDVI untuk membedakan",
        ],
    },
}
