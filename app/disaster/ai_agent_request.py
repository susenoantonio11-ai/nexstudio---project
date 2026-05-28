"""
AI AGENT REQUEST SCHEMA dengan parameter language.

Setiap permintaan ke AI Agent (insight generator, method monitor explainer,
recommendation engine, dll) harus menyertakan kode bahasa user yang aktif.
AI Agent akan merespons dalam bahasa tersebut.

Bahasa yang didukung (sinkron dengan src/i18n/language.config.ts):
    en, id, ja, zh, fr, de, ru, ar, es, pt, ko, hi, tr, it
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


SUPPORTED_LANGUAGES = [
    "en", "id", "ja", "zh", "fr", "de", "ru",
    "ar", "es", "pt", "ko", "hi", "tr", "it",
]
DEFAULT_LANGUAGE = "en"


def normalize_language(code: Optional[str]) -> str:
    """Validasi & normalisasi kode bahasa. Fallback ke English."""
    if not code:
        return DEFAULT_LANGUAGE
    short = str(code).lower().split("-")[0]
    return short if short in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


@dataclass
class AIAgentRequest:
    """
    Request universal untuk AI Agent.

    Field 'language' wajib agar response AI mengikuti preferensi user.
    Field 'task' menentukan jenis pekerjaan: generate_insight,
    explain_method, recommend_action, summarize_dataset, dst.
    """
    task:        str
    data:        Dict[str, Any]
    language:    str = DEFAULT_LANGUAGE
    user_id:     Optional[int] = None
    project_id:  Optional[int] = None
    context:     Dict[str, Any] = field(default_factory=dict)
    max_tokens:  int = 1024
    temperature: float = 0.3

    def __post_init__(self):
        self.language = normalize_language(self.language)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_system_prompt_prefix(self) -> str:
        """Prefix yang ditempel ke system prompt untuk meminta output bahasa user."""
        lang_name = {
            "en": "English", "id": "Bahasa Indonesia", "ja": "Japanese",
            "zh": "Chinese (Simplified)", "fr": "French", "de": "German",
            "ru": "Russian", "ar": "Arabic", "es": "Spanish",
            "pt": "Portuguese", "ko": "Korean", "hi": "Hindi",
            "tr": "Turkish", "it": "Italian",
        }.get(self.language, "English")
        return (
            f"You MUST respond in {lang_name}. All explanations, insights, "
            f"recommendations, and summaries must be written in {lang_name}. "
            f"Numerical values, units, and proper nouns may remain in their "
            f"original form."
        )


@dataclass
class AIAgentResponse:
    task:     str
    language: str
    output:   str                                 # plain-text or markdown
    structured: Dict[str, Any] = field(default_factory=dict)
    citations:  List[Dict[str, Any]] = field(default_factory=list)
    model:    str = "nexa-agent-v1"
    tokens:   int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
