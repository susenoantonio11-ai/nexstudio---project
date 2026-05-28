"""
Keyword Extractor - TF-IDF based + word frequency.
"""
from __future__ import annotations
from typing import Dict, Any, List
from collections import Counter
from .text_preprocessor import TextPreprocessor


class KeywordExtractor:
    """Extract top keywords from a corpus of texts."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def extract(
        self,
        texts: List[str],
        method: str = "tfidf",
        top_k: int = 30,
    ) -> Dict[str, Any]:
        """Extract top keywords using TF-IDF or raw frequency."""
        if not texts:
            return {"keywords": [], "n_texts": 0}

        if method == "frequency":
            return self._frequency(texts, top_k)
        return self._tfidf(texts, top_k)

    def _frequency(self, texts: List[str], top_k: int) -> Dict[str, Any]:
        all_tokens = []
        for t in texts:
            all_tokens.extend(self.preprocessor.tokenize(t))
        counter = Counter(all_tokens)
        top = counter.most_common(top_k)
        max_count = top[0][1] if top else 1

        return {
            "method": "word_frequency",
            "n_texts": len(texts),
            "n_unique_tokens": len(counter),
            "total_tokens": sum(counter.values()),
            "keywords": [
                {
                    "word": w,
                    "score": float(c),
                    "normalized_score": round(c / max_count, 4),
                    "count": int(c),
                }
                for w, c in top
            ],
            "method_explanation": (
                "Raw word frequency after preprocessing (lowercase, stopword removal). "
                "Simple and fast; suitable for word clouds and quick exploration."
            ),
        }

    def _tfidf(self, texts: List[str], top_k: int) -> Dict[str, Any]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return self._frequency(texts, top_k)

        processed = [" ".join(self.preprocessor.tokenize(t)) for t in texts]
        processed = [p for p in processed if p.strip()]
        if not processed:
            return {"keywords": [], "n_texts": 0}

        vectorizer = TfidfVectorizer(max_features=2000, min_df=2)
        try:
            X = vectorizer.fit_transform(processed)
        except ValueError:
            return self._frequency(texts, top_k)

        # Sum TF-IDF scores across all docs
        scores = X.sum(axis=0).A1
        feature_names = vectorizer.get_feature_names_out()

        # Sort
        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = ranked[0][1] if ranked else 1

        return {
            "method": "tfidf",
            "n_texts": len(processed),
            "n_features": X.shape[1],
            "keywords": [
                {
                    "word": str(w),
                    "score": round(float(s), 4),
                    "normalized_score": round(float(s) / max_score, 4),
                }
                for w, s in ranked
            ],
            "method_explanation": (
                "TF-IDF (Term Frequency-Inverse Document Frequency) ranks terms by importance "
                "across the corpus. Common words are down-weighted; distinctive terms surface. "
                "Better than raw frequency for finding meaningful keywords. "
                "Reference: Salton & Buckley (1988)."
            ),
            "method_monitor": {
                "selected_method": "TF-IDF",
                "why_chosen": "Penalizes common words across documents, surfaces distinctive vocabulary per corpus.",
                "why_not_alternatives": [
                    {"alternative": "Raw frequency", "reason_rejected": "Common words like 'product' dominate"},
                    {"alternative": "RAKE", "reason_rejected": "Better for short single docs; TF-IDF works better on corpora"},
                    {"alternative": "TextRank", "reason_rejected": "More complex graph-based; minimal benefit for keywords vs key phrases"},
                ],
                "limitations": [
                    "Ignores semantic similarity (synonyms scored separately)",
                    "Bag-of-words: no phrase context",
                ],
            },
        }
