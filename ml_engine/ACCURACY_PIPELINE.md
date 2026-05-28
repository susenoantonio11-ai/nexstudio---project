# AI Accuracy Pipeline - Nexlytics

Production-grade ML pipeline yang prioritaskan **akurasi pada data baru, bukan pada data training** (anti-overfitting). Mengikuti CRISP-DM secara strict dan menerapkan best practices ML modern.

## Arsitektur

```
ml_engine/
├── eda/                          # CRISP-DM Step 2: Data Understanding
│   ├── data_quality_checker.py   #   Wang & Strong (1996) 6-dimension framework
│   ├── missing_analyzer.py       #   MCAR/MAR/MNAR detection + strategy
│   ├── outlier_detector.py       #   IQR + Z-score + Modified Z + consensus
│   ├── imbalance_detector.py     #   Class balance + metric warnings
│   ├── leakage_detector.py       #   Target leakage + time leakage + ID columns
│   └── correlation_analyzer.py   #   Feature-target + multicollinearity
│
├── preprocessing/                # CRISP-DM Step 3: Data Preparation
│   ├── pipeline_builder.py       #   ColumnTransformer (LEAK-SAFE)
│   ├── imputer.py                #   median, KNN, iterative
│   ├── scaler.py                 #   standard, robust, minmax
│   ├── encoder.py                #   onehot, ordinal
│   └── feature_engineer.py       #   datetime decomposition, log transform
│
├── splitting/                    # Leak-safe data splits
│   ├── train_test_splitter.py    #   stratified / random / chronological
│   └── cv_strategy.py            #   StratifiedKFold / KFold / TimeSeriesSplit
│
├── modeling/                     # CRISP-DM Step 4: Modeling
│   ├── baseline_model.py         #   DummyClassifier/Regressor (floor)
│   ├── model_comparator.py       #   Multi-model CV + overfitting check
│   ├── tuner.py                  #   GridSearchCV with leak-safe pipeline
│   └── threshold_tuner.py        #   Classification threshold optimization
│
├── evaluation/                   # CRISP-DM Step 5: Evaluation
│   ├── metric_selector.py        #   Right metric per task + balance
│   ├── classification_eval.py    #   F1, ROC-AUC, PR-AUC, MCC, balanced acc
│   ├── regression_eval.py        #   R², RMSE, MAE, MAPE
│   └── overfitting_detector.py   #   Train-test gap + CV-holdout drift
│
├── research_lab/                 # Reproducibility
│   ├── experiment_tracker.py     #   Persist experiments with seeds + env hash
│   ├── experiment_comparator.py  #   Side-by-side ranking
│   └── reproducibility.py        #   Seed management
│
├── monitoring/                   # CRISP-DM Step 6: Deployment Monitoring
│   ├── drift_detector.py         #   PSI + KS test for data drift
│   ├── concept_drift.py          #   Performance degradation
│   └── retraining_trigger.py     #   Decide when to retrain
│
└── accuracy_pipeline.py          # TOP-LEVEL ORCHESTRATOR
```

## Anti-Overfitting Garansi

**Mengapa pipeline ini tidak overfit:**

1. **Train/test split DULU**, lalu preprocessing (sklearn ColumnTransformer fitted on train only)
2. **Cross-validation di training set saja** (test set tidak pernah disentuh untuk tuning)
3. **Multi-model comparison** dengan CV scores ± std (bukan single training accuracy)
4. **Overfitting detection** automatic via train-test gap analysis
5. **Adjusted score** = CV_test - 0.5 × max(0, train_test_gap) → menghukum model yang overfit
6. **Final test pada held-out set** yang tidak pernah dipakai untuk apapun (true generalization estimate)
7. **Leakage detector** flag kolom yang berisiko (high correlation, name patterns, identifiers)

## Metric Selection Logic

| Task | Balance | Primary Metric | Avoided |
|------|---------|----------------|---------|
| Regression | - | R² + RMSE + MAE + MAPE | - |
| Classification balanced binary | balanced | F1 + ROC-AUC | - |
| Classification balanced multiclass | balanced | F1-weighted + accuracy | - |
| Classification mild/moderate imbalance | imbalanced | F1-macro + PR-AUC + balanced_acc | accuracy |
| Classification severe imbalance | severe | PR-AUC + MCC + balanced_acc | accuracy, ROC-AUC |
| Forecasting | - | MAPE + RMSE + MAE | - |
| Anomaly Detection | - | PR-AUC + Precision + Recall | accuracy, ROC-AUC |

**Mengapa accuracy sering misleading**: pada imbalanced data (misal 95:5), model yang selalu prediksi majority class scores 95% accuracy tapi useless untuk minority class. F1-macro dan PR-AUC robust terhadap imbalance. Reference: He & Garcia (2009) - Learning from imbalanced data.

## Cara Pakai

```python
from ml_engine.accuracy_pipeline import AccuracyPipeline
import pandas as pd

df = pd.read_csv("your_data.csv")

pipeline = AccuracyPipeline(
    goal="Predict customer churn with maximum generalization",
    random_seed=42,
    track_experiments=True,
)

result = pipeline.run(
    df=df,
    target_column="churn",
    task_type="auto",          # auto-detect classification/regression
    datetime_column="signup_date",  # optional: triggers chronological split
    test_size=0.2,
    n_cv_splits=5,
    tune_hyperparameters=True,
    experiment_name="churn_v1",
)

# Akses hasil
print(result['eda']['data_quality']['overall_score'])     # data quality 0-100
print(result['model_comparison']['best_model'])           # winning model
print(result['final_evaluation']['test_set_metrics'])     # holdout metrics
print(result['overfitting_diagnosis']['overall_assessment'])  # generalization verdict
print(result['method_monitor']['steps'])                  # full reasoning chain
```

## Method Monitor Output

Setiap eksekusi menghasilkan trace lengkap untuk Explainable AI:

```json
{
  "step": "model_comparison",
  "selected_method": "gradient_boosting_regressor",
  "why_chosen": "Best CV score (0.917) with manageable overfitting risk",
  "why_not_alternatives": [
    {"alternative": "random_forest_regressor", "reason_rejected": "Lower CV score (0.894)"},
    {"alternative": "ridge_regression", "reason_rejected": "Lower CV score (0.674); linear assumption inadequate"},
    {"alternative": "linear_regression", "reason_rejected": "No regularization, prone to issues with multicollinear features"}
  ],
  "benefits": ["Sequential boosting often top performer", "Captures non-linearities"],
  "limitations": ["Slower than RF", "Sensitive to hyperparameters"]
}
```

## Hasil Test pada Sample Dataset

Dataset: 1500 e-commerce transactions, target = `total_amount`.

| Metric | Value |
|--------|-------|
| Data Quality | 98.7/100 |
| Leakage High Risk | 0 columns |
| Best Model | gradient_boosting_regressor |
| CV R² | 0.917 ± 0.051 |
| Holdout R² | 0.843 |
| CV→Holdout drift | 7% (acceptable) |
| Verdict | ACCEPTABLE - safe to deploy with monitoring |
| Total runtime | 3.0 seconds |

**Kunci**: holdout R² (0.843) < CV R² (0.917) by 7% — ini wajar dan menunjukkan TIDAK ADA leakage. Jika holdout R² > CV R² atau sama dengan training R² (0.999), itu red flag overfitting/leakage.

## Referensi Akademik

- Adadi & Berrada (2018) - Explainable AI Survey, IEEE Access
- Chandola, Banerjee & Kumar (2009) - Anomaly Detection: A Survey, ACM
- Hastie, Tibshirani & Friedman (2009) - The Elements of Statistical Learning
- He & Garcia (2009) - Learning from Imbalanced Data, IEEE TKDE
- Hyndman & Athanasopoulos (2018) - Forecasting: Principles and Practice
- Iglewicz & Hoaglin (1993) - How to Detect and Handle Outliers
- Kaufman et al. (2012) - Leakage in Data Mining, ACM TKDD
- Little & Rubin (2019) - Statistical Analysis with Missing Data
- Tsymbal (2004) - The Problem of Concept Drift
- Wang & Strong (1996) - Beyond Accuracy: Data Quality, JMIS
- Wirth & Hipp (2000) - CRISP-DM
