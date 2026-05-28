"""
Topic Modeler - LDA / NMF for unsupervised topic discovery.
"""
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from .text_preprocessor import TextPreprocessor


class TopicModeler:
    """Latent Dirichlet Allocation / Non-negative Matrix Factorization."""

    def __init__(self, method: str = "lda"):
        self.method = method
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        self.feature_names = None

    def fit(
        self,
        texts: List[str],
        n_topics: int = 5,
        n_top_words: int = 10,
        max_features: int = 1000,
    ) -> Dict[str, Any]:
        """Train topic model. Returns topics with top keywords + reasoning."""
        try:
            from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation, NMF
        except ImportError:
            return {"error": "scikit-learn not installed"}

        # Preprocess texts
        processed = [" ".join(self.preprocessor.tokenize(t)) for t in texts]
        processed = [p for p in processed if p.strip()]
        if len(processed) < n_topics:
            return {
                "error": f"Need at least {n_topics} valid texts; got {len(processed)}",
                "n_topics": 0,
            }

        # Vectorize
        if self.method == "nmf":
            self.vectorizer = TfidfVectorizer(max_features=max_features, min_df=2)
        else:
            self.vectorizer = CountVectorizer(max_features=max_features, min_df=2)
        try:
            X = self.vectorizer.fit_transform(processed)
        except ValueError:
            return {"error": "Insufficient vocabulary after preprocessing"}
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Fit model
        if self.method == "nmf":
            self.model = NMF(n_components=n_topics, random_state=42, max_iter=200)
        else:
            self.model = LatentDirichletAllocation(
                n_components=n_topics, random_state=42,
                learning_method="online", max_iter=20,
            )
        topic_word_dist = self.model.fit_transform(X)

        # Extract top words per topic
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[: -n_top_words - 1 : -1]
            top_words = [
                {"word": self.feature_names[i], "weight": round(float(topic[i]), 4)}
                for i in top_indices
            ]
            topics.append({
                "topic_id": topic_idx,
                "label": f"Topic {topic_idx}",
                "top_words": top_words,
                "auto_label": ", ".join(w["word"] for w in top_words[:3]),
            })

        # Document-topic assignments
        doc_topic = topic_word_dist
        if self.method == "lda":
            # Normalize so rows sum to 1
            row_sums = doc_topic.sum(axis=1, keepdims=True)
            doc_topic = doc_topic / np.where(row_sums == 0, 1, row_sums)

        dominant_topics = doc_topic.argmax(axis=1).tolist()

        return {
            "method": self.method.upper(),
            "n_topics": n_topics,
            "n_documents": len(processed),
            "n_features": X.shape[1],
            "topics": topics,
            "dominant_topic_per_doc": dominant_topics,
            "topic_distribution": {
                str(t): int(dominant_topics.count(t)) for t in range(n_topics)
            },
            "method_explanation": (
                f"{self.method.upper()} with {n_topics} topics and {n_top_words} top words each. "
                + ("LDA uses generative probabilistic modeling. " if self.method == "lda" else "NMF factorizes term-document matrix. ")
                + f"Vocabulary: {X.shape[1]} terms after preprocessing."
            ),
            "method_monitor": {
                "selected_method": self.method.upper(),
                "why_chosen": (
                    "LDA is the gold standard for probabilistic topic modeling, robust to noise. "
                    if self.method == "lda" else
                    "NMF produces more coherent topics than LDA on shorter texts and gives sparse representations. "
                ) + "Pre-specified n_topics requires domain validation.",
                "why_not_alternatives": [
                    {"alternative": "BERTopic", "reason_rejected": "Requires transformer embeddings; heavy infrastructure for MVP"},
                    {"alternative": "K-Means on TF-IDF", "reason_rejected": "Hard clustering; can't capture mixed-topic documents"},
                    {"alternative": "LDA" if self.method == "nmf" else "NMF",
                     "reason_rejected": "Different probabilistic assumptions; we chose this method based on document length"},
                ],
                "limitations": [
                    "n_topics must be pre-specified",
                    "Bag-of-words ignores word order and context",
                    "Topic coherence requires domain validation",
                ],
            },
        }
