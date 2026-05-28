"""
Text Preprocessor
=================
CRISP-DM Step 3 (Data Preparation) for text:
- Lowercasing
- Tokenization
- Punctuation/digit removal
- Stopword removal
- Bilingual support (English + Indonesian basics)
"""
from __future__ import annotations
from typing import List, Set, Optional
import re


# Compact stopword lists (no NLTK dependency required for MVP)
STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "whom", "whose", "as", "am",
}

STOPWORDS_ID = {
    "yang", "dan", "di", "dari", "ke", "untuk", "dengan", "ada", "ini", "itu",
    "atau", "akan", "tidak", "juga", "dalam", "saja", "saya", "anda", "kami",
    "kita", "dia", "mereka", "adalah", "tersebut", "telah", "sudah", "belum",
    "bisa", "dapat", "harus", "lagi", "lebih", "kurang", "sama", "seperti",
    "yaitu", "yakni", "agar", "supaya", "karena", "sebab", "namun", "tetapi",
    "tapi", "sehingga", "maka", "kalau", "jika", "apabila", "saat", "pada",
    "oleh", "sebagai", "menjadi", "membuat", "membawa", "saat",
}


class TextPreprocessor:
    """Text cleaning and tokenization."""

    def __init__(
        self,
        languages: List[str] = None,
        custom_stopwords: Optional[Set[str]] = None,
        min_token_length: int = 2,
    ):
        self.languages = languages or ["en", "id"]
        self.stopwords = self._build_stopwords(custom_stopwords)
        self.min_token_length = min_token_length

    def clean(self, text: str) -> str:
        """Lowercase, remove punctuation/digits, normalize whitespace."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)  # urls
        text = re.sub(r"@\w+", "", text)             # mentions
        text = re.sub(r"[^\w\s]", " ", text)          # punctuation
        text = re.sub(r"\d+", "", text)               # digits
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean(text)
        tokens = cleaned.split()
        return [
            t for t in tokens
            if t not in self.stopwords and len(t) >= self.min_token_length
        ]

    def preprocess_corpus(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenize(t) for t in texts]

    def _build_stopwords(self, custom: Optional[Set[str]]) -> Set[str]:
        sw: Set[str] = set()
        if "en" in self.languages:
            sw |= STOPWORDS_EN
        if "id" in self.languages:
            sw |= STOPWORDS_ID
        if custom:
            sw |= custom
        return sw
