"""
Aspect-Based Sentiment Analyzer
================================
Identifies sentiment for specific aspects (price, quality, delivery, etc.)
within a single review text. Uses windowed sentiment around aspect keywords.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import re
from .text_preprocessor import TextPreprocessor
from .sentiment_analyzer import SentimentAnalyzer


# Default aspect dictionary (extensible per domain)
DEFAULT_ASPECTS = {
    "price": ["price", "cost", "expensive", "cheap", "affordable", "value",
              "harga", "biaya", "mahal", "murah", "terjangkau"],
    "quality": ["quality", "build", "material", "durable", "sturdy", "fragile",
                "kualitas", "bahan", "tahan", "awet", "rusak"],
    "delivery": ["delivery", "shipping", "arrive", "package", "courier", "ship",
                 "pengiriman", "ongkir", "kurir", "sampai", "datang"],
    "service": ["service", "support", "staff", "helpful", "rude", "responsive",
                "pelayanan", "layanan", "ramah", "kasar", "responsif"],
    "design": ["design", "style", "look", "appearance", "aesthetic",
               "desain", "tampilan", "model", "warna"],
    "performance": ["performance", "speed", "fast", "slow", "lag", "smooth",
                    "kinerja", "kecepatan", "cepat", "lambat", "lemot"],
}


class AspectSentimentAnalyzer:
    """Aspect-based sentiment analysis."""

    def __init__(self, aspects: Optional[Dict[str, List[str]]] = None):
        self.aspects = aspects or DEFAULT_ASPECTS
        self.preprocessor = TextPreprocessor()
        self.sentiment = SentimentAnalyzer()
        self.window_size = 5  # words before/after aspect keyword

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment per aspect in a single text."""
        if not isinstance(text, str) or not text.strip():
            return {"aspects": {}, "n_aspects_found": 0}

        cleaned = self.preprocessor.clean(text)
        tokens = cleaned.split()
        n_tokens = len(tokens)

        aspect_findings: Dict[str, Dict[str, Any]] = {}

        for aspect_name, keywords in self.aspects.items():
            mentions = []
            for i, tok in enumerate(tokens):
                if tok in keywords:
                    # Extract window around aspect mention
                    start = max(0, i - self.window_size)
                    end = min(n_tokens, i + self.window_size + 1)
                    window = " ".join(tokens[start:end])
                    sent_result = self.sentiment.analyze(window)
                    mentions.append({
                        "keyword": tok,
                        "context": window,
                        "polarity": sent_result["polarity"],
                        "label": sent_result["label"],
                    })

            if mentions:
                # Aggregate mentions into single aspect score
                avg_pol = sum(m["polarity"] for m in mentions) / len(mentions)
                if avg_pol > 0.15:
                    aspect_label = "positive"
                elif avg_pol < -0.15:
                    aspect_label = "negative"
                else:
                    aspect_label = "neutral"

                aspect_findings[aspect_name] = {
                    "n_mentions": len(mentions),
                    "average_polarity": round(avg_pol, 4),
                    "label": aspect_label,
                    "mentions": mentions[:5],  # cap for size
                }

        return {
            "n_aspects_found": len(aspect_findings),
            "aspects": aspect_findings,
        }

    def analyze_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Aggregate aspect sentiment across many texts."""
        per_aspect_polarities: Dict[str, List[float]] = {}
        per_aspect_mentions: Dict[str, int] = {}
        per_text_results = []

        for t in texts:
            result = self.analyze(t)
            per_text_results.append(result)
            for aspect_name, data in result["aspects"].items():
                per_aspect_polarities.setdefault(aspect_name, []).append(data["average_polarity"])
                per_aspect_mentions[aspect_name] = (
                    per_aspect_mentions.get(aspect_name, 0) + data["n_mentions"]
                )

        # Aggregate
        aspect_summary = []
        for aspect_name, polarities in per_aspect_polarities.items():
            avg = sum(polarities) / len(polarities) if polarities else 0
            n_pos = sum(1 for p in polarities if p > 0.15)
            n_neg = sum(1 for p in polarities if p < -0.15)
            n_neu = sum(1 for p in polarities if -0.15 <= p <= 0.15)

            if avg > 0.15:
                overall = "positive"
            elif avg < -0.15:
                overall = "negative"
            else:
                overall = "neutral"

            aspect_summary.append({
                "aspect": aspect_name,
                "n_documents_mentioning": len(polarities),
                "n_total_mentions": per_aspect_mentions.get(aspect_name, 0),
                "average_polarity": round(float(avg), 4),
                "overall_label": overall,
                "label_breakdown": {
                    "positive": n_pos,
                    "neutral": n_neu,
                    "negative": n_neg,
                },
            })

        # Sort by mention frequency
        aspect_summary.sort(key=lambda a: a["n_total_mentions"], reverse=True)

        return {
            "n_texts_analyzed": len(texts),
            "n_aspects_with_data": len(aspect_summary),
            "aspect_summary": aspect_summary,
            "method_explanation": (
                "Aspect-Based Sentiment Analysis (ABSA) using windowed lexicon scoring. "
                "Each aspect keyword found triggers sentiment scoring on ±5-word window. "
                "Aggregated across documents to reveal which product/service aspects "
                "drive positive vs negative sentiment. Reference: Pontiki et al. (2014)."
            ),
            "method_monitor": {
                "selected_method": "Windowed lexicon ABSA",
                "why_chosen": (
                    "ABSA reveals NUANCED feedback: a product with positive overall sentiment "
                    "may have negative 'price' aspect. Window approach is fast, no training needed."
                ),
                "why_not_alternatives": [
                    {"alternative": "BERT-ABSA", "reason_rejected": "Heavy infrastructure; lexicon baseline first"},
                    {"alternative": "Aspect extraction (autodiscovery)", "reason_rejected": "Predefined aspects more interpretable for business"},
                ],
                "limitations": [
                    "Predefined aspect dictionary; new aspects require manual addition",
                    "Window size limits long-range dependency (aspect at start, sentiment at end)",
                    "Co-mentioned aspects may share sentiment incorrectly",
                ],
            },
        }
