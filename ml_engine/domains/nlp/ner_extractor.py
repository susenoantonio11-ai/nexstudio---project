"""
Named Entity Recognition (NER) - regex + dictionary based.
For production: integrate spaCy with proper NER models.
"""
from __future__ import annotations
from typing import Dict, Any, List
import re


class NamedEntityExtractor:
    """Extract entities (PERSON, ORG, LOCATION, MONEY, DATE) using rules."""

    # Indonesian common cities
    INDONESIAN_CITIES = {
        "jakarta", "surabaya", "bandung", "medan", "makassar", "denpasar",
        "yogyakarta", "semarang", "palembang", "bogor", "bekasi", "tangerang",
        "depok", "padang", "malang", "pekanbaru", "pontianak", "balikpapan",
        "samarinda", "manado", "banjarmasin",
    }
    # Common English organizations / countries (small subset)
    COUNTRIES = {
        "indonesia", "singapore", "malaysia", "thailand", "vietnam",
        "philippines", "japan", "china", "korea", "australia",
        "united states", "us", "usa", "uk", "britain", "germany", "france",
    }

    MONEY_PATTERNS = [
        re.compile(r"(?:Rp\s?[\d.,]+(?:\s?(?:juta|miliar|ribu))?)", re.IGNORECASE),
        re.compile(r"(?:USD?\s?\$?\s?\d[\d.,]*\s?(?:million|billion|k)?)", re.IGNORECASE),
        re.compile(r"\$\s?\d[\d.,]+(?:\s?(?:million|billion|k|m|b))?", re.IGNORECASE),
    ]
    DATE_PATTERNS = [
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
        re.compile(r"\b\d{1,2}\s+(jan|feb|mar|apr|mei|jun|jul|agu|sep|okt|nov|des|january|february|march|april|may|june|july|august|september|october|november|december)\w*\s+\d{2,4}\b", re.IGNORECASE),
    ]
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-]{7,}\d")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%")

    def extract(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            return {"entities": [], "n_entities": 0}

        entities: List[Dict[str, Any]] = []
        text_lower = text.lower()

        # MONEY
        for pat in self.MONEY_PATTERNS:
            for m in pat.finditer(text):
                entities.append({"text": m.group(), "type": "MONEY", "start": m.start(), "end": m.end()})

        # DATE
        for pat in self.DATE_PATTERNS:
            for m in pat.finditer(text):
                entities.append({"text": m.group(), "type": "DATE", "start": m.start(), "end": m.end()})

        # EMAIL / URL / PHONE / PERCENT
        for m in self.EMAIL_PATTERN.finditer(text):
            entities.append({"text": m.group(), "type": "EMAIL", "start": m.start(), "end": m.end()})
        for m in self.URL_PATTERN.finditer(text):
            entities.append({"text": m.group(), "type": "URL", "start": m.start(), "end": m.end()})
        for m in self.PHONE_PATTERN.finditer(text):
            entities.append({"text": m.group(), "type": "PHONE", "start": m.start(), "end": m.end()})
        for m in self.PERCENT_PATTERN.finditer(text):
            entities.append({"text": m.group(), "type": "PERCENT", "start": m.start(), "end": m.end()})

        # LOCATION (city/country dictionary lookup)
        for city in self.INDONESIAN_CITIES:
            for m in re.finditer(r"\b" + re.escape(city) + r"\b", text_lower):
                entities.append({
                    "text": text[m.start():m.end()],
                    "type": "LOCATION",
                    "start": m.start(), "end": m.end(),
                })
        for country in self.COUNTRIES:
            for m in re.finditer(r"\b" + re.escape(country) + r"\b", text_lower):
                entities.append({
                    "text": text[m.start():m.end()],
                    "type": "LOCATION",
                    "start": m.start(), "end": m.end(),
                })

        # PERSON heuristic: capitalized 2-word sequences
        person_pat = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
        for m in person_pat.finditer(text):
            entities.append({
                "text": m.group(),
                "type": "PERSON",
                "start": m.start(), "end": m.end(),
                "confidence": 0.5,
            })

        # Dedupe (same text + type)
        seen = set()
        unique = []
        for e in entities:
            key = (e["text"].lower(), e["type"])
            if key not in seen:
                seen.add(key)
                unique.append(e)

        type_counts: Dict[str, int] = {}
        for e in unique:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1

        return {
            "entities": unique,
            "n_entities": len(unique),
            "type_counts": type_counts,
            "method": "regex + dictionary lookup",
            "method_monitor": {
                "selected_method": "Rule-based NER",
                "why_chosen": (
                    "Fast, deterministic, no model required. Reliable for structured entities "
                    "(MONEY, DATE, EMAIL, URL) and known locations."
                ),
                "why_not_alternatives": [
                    {"alternative": "spaCy NER", "reason_rejected": "Adds 50MB+ dependency; not all entity types needed for MVP"},
                    {"alternative": "BERT NER", "reason_rejected": "Heavy infrastructure; rule-based covers business cases"},
                ],
                "limitations": [
                    "PERSON detection is heuristic (capitalized words); can have false positives",
                    "Limited to dictionary entries; new locations need dictionary updates",
                    "No contextual disambiguation (e.g., Apple = company vs fruit)",
                ],
            },
        }

    def extract_batch(self, texts: List[str]) -> Dict[str, Any]:
        all_entities: List[Dict[str, Any]] = []
        type_counts: Dict[str, int] = {}
        for i, t in enumerate(texts):
            r = self.extract(t)
            for e in r["entities"]:
                e["doc_id"] = i
                all_entities.append(e)
            for k, v in r["type_counts"].items():
                type_counts[k] = type_counts.get(k, 0) + v

        # Top entities per type
        from collections import Counter
        top_per_type: Dict[str, List[Dict[str, Any]]] = {}
        for ent_type in type_counts.keys():
            counter = Counter(e["text"].lower() for e in all_entities if e["type"] == ent_type)
            top_per_type[ent_type] = [
                {"text": w, "count": int(c)} for w, c in counter.most_common(10)
            ]

        return {
            "n_documents": len(texts),
            "n_entities_total": len(all_entities),
            "type_counts": type_counts,
            "top_entities_per_type": top_per_type,
            "all_entities": all_entities[:200],  # cap for response size
        }
