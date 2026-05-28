"""
Sentiment Analyzer
==================
Lexicon + rule-based sentiment scoring (English + Indonesian).
For production: swap with VADER, BERT, or RoBERTa fine-tuned.

Returns:
- polarity score in [-1, 1]
- label: positive / negative / neutral
- confidence (proportion of sentiment words)
"""
from __future__ import annotations
from typing import Dict, Any, List
import re
from .text_preprocessor import TextPreprocessor


# Compact lexicon (extensible)
POSITIVE_WORDS = {
    # English
    "good", "great", "excellent", "amazing", "awesome", "best", "love", "perfect",
    "wonderful", "fantastic", "beautiful", "happy", "satisfied", "recommend",
    "fast", "easy", "useful", "helpful", "clean", "fresh", "delicious", "comfortable",
    "outstanding", "superb", "brilliant", "impressive", "reliable", "smooth",
    # Indonesian
    "bagus", "baik", "hebat", "mantap", "keren", "luar", "biasa", "sangat",
    "puas", "memuaskan", "menyenangkan", "indah", "cantik", "tampan", "ramah",
    "cepat", "mudah", "berguna", "bermanfaat", "bersih", "segar", "lezat", "enak",
    "nyaman", "rekomendasi", "rekomen", "suka", "senang", "gembira", "terbaik",
}

NEGATIVE_WORDS = {
    # English
    "bad", "awful", "terrible", "horrible", "hate", "worst", "poor", "disappointing",
    "slow", "broken", "useless", "annoying", "rude", "dirty", "ugly", "boring",
    "unreliable", "frustrating", "uncomfortable", "expensive", "overpriced",
    "scam", "fake", "stupid", "ridiculous", "garbage", "trash",
    # Indonesian
    "buruk", "jelek", "parah", "kecewa", "mengecewakan", "lambat", "rusak",
    "tidak", "berguna", "menjengkelkan", "kasar", "kotor", "membosankan",
    "mahal", "sampah", "bohong", "palsu", "bodoh", "menyebalkan",
    "menjijikkan", "kacau", "berantakan", "gagal",
}

NEGATION_WORDS = {"not", "no", "never", "tidak", "bukan", "jangan"}

INTENSIFIERS = {"very": 1.5, "really": 1.4, "extremely": 2.0, "super": 1.6,
                "sangat": 1.5, "amat": 1.4, "betul": 1.3, "banget": 1.6}


class SentimentAnalyzer:
    """Lexicon-based sentiment analyzer."""

    def __init__(self, custom_positive=None, custom_negative=None):
        self.preprocessor = TextPreprocessor()
        self.positive = POSITIVE_WORDS | set(custom_positive or [])
        self.negative = NEGATIVE_WORDS | set(custom_negative or [])

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze a single text. Returns polarity, label, breakdown."""
        if not isinstance(text, str) or not text.strip():
            return {"label": "neutral", "polarity": 0.0, "confidence": 0.0,
                    "n_positive": 0, "n_negative": 0, "n_total_tokens": 0}

        clean = self.preprocessor.clean(text)
        tokens = clean.split()

        n_positive = 0
        n_negative = 0
        score = 0.0
        prev_token = None

        for i, tok in enumerate(tokens):
            multiplier = 1.0
            # Look back for intensifier
            if i > 0 and tokens[i - 1] in INTENSIFIERS:
                multiplier = INTENSIFIERS[tokens[i - 1]]
            # Negation flips polarity
            if i > 0 and tokens[i - 1] in NEGATION_WORDS:
                multiplier = -1.0

            if tok in self.positive:
                score += 1.0 * multiplier
                if multiplier > 0:
                    n_positive += 1
                else:
                    n_negative += 1
            elif tok in self.negative:
                score -= 1.0 * multiplier
                if multiplier > 0:
                    n_negative += 1
                else:
                    n_positive += 1

        n_sentiment = n_positive + n_negative
        n_total = max(1, len(tokens))

        # Normalize polarity to [-1, 1]
        polarity = max(-1.0, min(1.0, score / max(1, n_sentiment)))
        confidence = n_sentiment / n_total

        if polarity > 0.15:
            label = "positive"
        elif polarity < -0.15:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "polarity": round(polarity, 4),
            "confidence": round(confidence, 4),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_total_tokens": n_total,
        }

    def analyze_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Aggregate sentiment across many texts."""
        results = [self.analyze(t) for t in texts]
        labels = [r["label"] for r in results]
        polarities = [r["polarity"] for r in results]
        n = len(results)

        return {
            "n_texts": n,
            "individual_results": results,
            "label_distribution": {
                "positive": labels.count("positive"),
                "negative": labels.count("negative"),
                "neutral": labels.count("neutral"),
            },
            "label_proportions": {
                "positive": round(labels.count("positive") / max(1, n), 4),
                "negative": round(labels.count("negative") / max(1, n), 4),
                "neutral": round(labels.count("neutral") / max(1, n), 4),
            },
            "average_polarity": round(sum(polarities) / max(1, n), 4),
            "method_explanation": (
                "Lexicon-based sentiment scoring with negation and intensifier handling. "
                "Polarity in [-1, 1], threshold ±0.15 for label assignment. "
                "Suitable for English and Indonesian. For production with domain-specific "
                "vocabulary, fine-tune a transformer model (BERT/RoBERTa)."
            ),
            "method_monitor": {
                "selected_method": "Lexicon + negation + intensifier",
                "why_chosen": (
                    "Fast, interpretable, no training required. Sufficient as MVP baseline "
                    "for general-purpose sentiment with bilingual coverage."
                ),
                "why_not_alternatives": [
                    {"alternative": "VADER", "reason_rejected": "English-only; we need Indonesian support"},
                    {"alternative": "BERT/transformer", "reason_rejected": "Heavy infrastructure; overkill for MVP without labeled training data"},
                    {"alternative": "TextBlob", "reason_rejected": "English-only; limited accuracy"},
                ],
                "limitations": [
                    "Misses sarcasm and complex context",
                    "Fixed vocabulary; new slang requires lexicon updates",
                    "Domain-agnostic; specialized domains (medical, legal) need custom lexicons",
                ],
            },
        }
