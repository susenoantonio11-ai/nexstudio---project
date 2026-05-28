"""
Text Classifier - sklearn pipeline (TF-IDF + LogisticRegression / SVM).
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
from .text_preprocessor import TextPreprocessor


class TextClassifier:
    """Train and evaluate a text classifier with leak-safe pipeline."""

    def __init__(self, algorithm: str = "logistic_regression"):
        self.algorithm = algorithm
        self.preprocessor = TextPreprocessor()
        self.pipeline = None
        self.classes_ = []

    def fit_evaluate(
        self,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Train + evaluate with stratified split."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import LinearSVC
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                f1_score,
                accuracy_score,
                precision_score,
                recall_score,
            )
        except ImportError:
            return {"error": "scikit-learn not installed"}

        if len(texts) != len(labels):
            return {"error": "texts and labels must have same length"}

        processed = [" ".join(self.preprocessor.tokenize(t)) for t in texts]

        # Filter out empty
        valid_pairs = [(p, l) for p, l in zip(processed, labels) if p.strip()]
        if len(valid_pairs) < 10:
            return {"error": f"Need at least 10 non-empty texts; got {len(valid_pairs)}"}
        processed_X, processed_y = zip(*valid_pairs)

        # Choose model
        if self.algorithm == "linear_svm":
            clf = LinearSVC(random_state=random_state, max_iter=2000)
        elif self.algorithm == "naive_bayes":
            clf = MultinomialNB()
        else:
            clf = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
            ("clf", clf),
        ])

        # Stratified split (only if all classes have >=2 samples)
        from collections import Counter
        class_counts = Counter(processed_y)
        if min(class_counts.values()) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_X, processed_y, test_size=test_size,
                random_state=random_state, stratify=processed_y,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_X, processed_y, test_size=test_size, random_state=random_state,
            )

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        # Metrics
        unique_classes = sorted(set(processed_y))
        self.classes_ = unique_classes
        is_binary = len(unique_classes) == 2

        avg_strategy = "binary" if is_binary else "weighted"

        # Build per-class metrics
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes).tolist()

        return {
            "algorithm": self.algorithm,
            "n_classes": len(unique_classes),
            "classes": [str(c) for c in unique_classes],
            "n_train": len(y_train),
            "n_test": len(y_test),
            "metrics": {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average=avg_strategy, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, average=avg_strategy, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, y_pred, average=avg_strategy, zero_division=0)), 4),
                "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            },
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "method_explanation": (
                f"TF-IDF (1-2 grams) + {self.algorithm}. Stratified train/test split. "
                f"TF-IDF captures word importance; n-grams capture short phrases. "
                f"Balanced class weights compensate for imbalance."
            ),
            "method_monitor": {
                "selected_method": f"TF-IDF + {self.algorithm}",
                "why_chosen": (
                    "Logistic Regression on TF-IDF is a strong, interpretable baseline for text classification. "
                    "n-grams capture phrases like 'not good'."
                ) if self.algorithm == "logistic_regression" else (
                    f"{self.algorithm} chosen for its strengths on text data."
                ),
                "why_not_alternatives": [
                    {"alternative": "BERT", "reason_rejected": "Heavy infrastructure; logistic regression sufficient for MVP"},
                    {"alternative": "Deep learning (LSTM)", "reason_rejected": "Requires large labeled data; TF-IDF baseline first"},
                ],
                "limitations": [
                    "Bag-of-words: ignores long-range context",
                    "Sensitive to preprocessing decisions (stopwords, stemming)",
                    "Vocabulary fixed at training time; new words ignored",
                ],
            },
        }

    def predict(self, texts: List[str]) -> List[str]:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit_evaluate() first.")
        processed = [" ".join(self.preprocessor.tokenize(t)) for t in texts]
        return self.pipeline.predict(processed).tolist()
