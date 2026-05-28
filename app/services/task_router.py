"""
NXLYTICS · Multi-Model Task Router
====================================
Routing engine yang pilih AI provider plus model optimal per task type.
Implementasi pattern "ML system orchestration" yang dijelaskan di
Sculley et al. (2015) NeurIPS "Hidden Technical Debt in ML Systems".

Logic:
    1. User submit request dengan task_type plus content
    2. Router cek task_type → ROUTING_TABLE
    3. Untuk task tersebut, iterasi preferred providers berurutan
    4. Provider yang connected (key tersedia) dipilih first
    5. Fallback ke provider berikutnya kalau primary fail
    6. Return result plus info routing decision

Task taxonomy (7 kategori utama):
    - summarize          : merangkum paper plus dokumen panjang
    - reason             : analisis logis, statistical interpretation
    - code               : generate Python code, debugging
    - translate          : terjemahan bahasa
    - fact_check         : verifikasi klaim, cek sitasi
    - classify           : klasifikasi text (sentiment, topic)
    - draft_academic     : drafting paragraf akademik
    - chat_general       : chat umum, fallback default

Routing strategy per task didasarkan pada:
    - Quality requirement (akademik > umum)
    - Cost sensitivity (heavy use plus iteration → prefer lokal)
    - Latency requirement (interactive > batch)
    - Reasoning depth (complex → DeepSeek R1 atau Claude Opus)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services import ai_client, secrets_store


# ============================================================================
# ROUTING TABLE
# ============================================================================
# Format: task_type → list of (provider, model_override) tuples in priority order
# First connected provider in list yang dipakai
# OLLAMA-FIRST routing · user preference: pakai DeepSeek R1 lokal sebagai
# default untuk hemat biaya plus jaga privacy data research. Cloud providers
# (Claude, OpenAI, Gemini) ada di fallback chain hanya kalau user sengaja
# configure key-nya plus Ollama tidak available.
ROUTING_TABLE: Dict[str, List[Dict[str, Optional[str]]]] = {
    "summarize": [
        {"provider": "ollama",    "model": "deepseek-r1:8b"},
        {"provider": "ollama",    "model": "deepseek-r1:latest"},
        {"provider": "deepseek",  "model": "deepseek-chat"},
        {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    ],
    "reason": [
        # DeepSeek R1 punya explicit reasoning chain · lokal jadi prioritas
        {"provider": "ollama",    "model": "deepseek-r1:8b"},
        {"provider": "ollama",    "model": "deepseek-r1:latest"},
        {"provider": "deepseek",  "model": "deepseek-reasoner"},
        {"provider": "anthropic", "model": "claude-opus-4-6"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        {"provider": "openai",    "model": "gpt-4o"},
    ],
    "code": [
        # Ollama coder dulu, fallback ke DeepSeek API
        {"provider": "ollama",    "model": "deepseek-coder-v2:16b"},
        {"provider": "ollama",    "model": "deepseek-r1:8b"},
        {"provider": "ollama",    "model": "codellama:7b"},
        {"provider": "deepseek",  "model": "deepseek-coder"},
        {"provider": "openai",    "model": "gpt-4o"},
    ],
    "translate": [
        # Qwen lokal bagus untuk Bahasa Indonesia
        {"provider": "ollama",    "model": "qwen2.5:7b"},
        {"provider": "ollama",    "model": "deepseek-r1:8b"},
        {"provider": "gemini",    "model": "gemini-1.5-pro"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ],
    "fact_check": [
        # Ollama R1 dulu, kalau butuh extra akurasi escalate ke Claude
        {"provider": "ollama",    "model": "deepseek-r1:8b"},
        {"provider": "ollama",    "model": "deepseek-r1:latest"},
        {"provider": "anthropic", "model": "claude-opus-4-6"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ],
    "classify": [
        {"provider": "ollama",    "model": "deepseek-r1:8b"},
        {"provider": "ollama",    "model": "deepseek-r1:latest"},
        {"provider": "deepseek",  "model": "deepseek-chat"},
    ],
    "draft_academic": [
        # Academic drafting · Ollama R1 reasoning chain bagus untuk struktur
        {"provider": "ollama",    "model": "deepseek-r1:8b"},
        {"provider": "ollama",    "model": "deepseek-r1:latest"},
        {"provider": "deepseek",  "model": "deepseek-reasoner"},
        {"provider": "anthropic", "model": "claude-opus-4-6"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ],
    "chat_general": [
        # General chat · Ollama always first
        {"provider": "ollama",    "model": "deepseek-r1:8b"},
        {"provider": "ollama",    "model": "deepseek-r1:latest"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        {"provider": "deepseek",  "model": "deepseek-chat"},
        {"provider": "gemini",    "model": "gemini-2.0-flash"},
    ],
    "structured_fast": [
        # Structured JSON output yang butuh KECEPATAN, BUKAN reasoning chain.
        # Dipakai oleh generate-scope, auto-map-sources, dan endpoint lain
        # yang return JSON terstruktur (bukan analitis). Prioritas remote API
        # cepat dulu (latency predictable), Ollama qwen2.5 sebagai backup
        # lokal yang lebih cepat dari R1 (tidak ada thinking chain).
        # Skip R1 yang lambat karena extensive reasoning token output.
        {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},  # ~5-15s remote
        {"provider": "deepseek",  "model": "deepseek-chat"},               # ~10-20s remote
        {"provider": "gemini",    "model": "gemini-2.0-flash"},            # ~5-15s remote
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},           # ~10-25s remote
        {"provider": "ollama",    "model": "qwen2.5:7b"},                  # ~15-30s local non-reasoning
        {"provider": "ollama",    "model": "llama3.2"},                    # ~10-20s local small
        {"provider": "ollama",    "model": "deepseek-r1:8b"},              # last resort
    ],
}


def is_provider_ready(provider: str) -> bool:
    """Cek apakah provider sudah configured (key tersedia atau Ollama lokal jalan)."""
    if provider == "ollama":
        # Ollama tidak butuh key, asumsikan ready (akan fail di chat call kalau tidak jalan)
        return True
    key = secrets_store.get_provider_key(provider)
    return key is not None and len(key) > 0


def route(task_type: str, messages: List[Dict[str, str]], max_tokens: int = 2048,
          temperature: float = 0.7, override_provider: Optional[str] = None,
          override_model: Optional[str] = None) -> Dict[str, Any]:
    """Main routing entry point.

    Args:
        task_type: salah satu dari ROUTING_TABLE keys atau 'chat_general' sebagai fallback
        messages: chat history dalam OpenAI format
        max_tokens: token output max
        temperature: sampling temperature
        override_provider: kalau ingin force ke provider tertentu (skip routing)
        override_model: kalau ingin force ke model spesifik

    Returns:
        dict dengan key 'ok', 'text', 'routing' (info provider yang dipilih plus kenapa)
    """
    task_type = (task_type or "chat_general").lower()
    if task_type not in ROUTING_TABLE:
        task_type = "chat_general"

    # Override path · skip routing, langsung pakai provider yang ditentukan
    if override_provider:
        # Set model kalau ada override
        if override_model:
            secrets_store.update_provider_model(override_provider, override_model)
        result = ai_client.chat(
            provider=override_provider,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result["routing"] = {
            "task_type": task_type,
            "selected_provider": override_provider,
            "selected_model": override_model or secrets_store.get_provider_model(override_provider),
            "decision": "override",
            "fallback_chain": [],
        }
        return result

    # Normal routing · iterasi candidates dari priority list
    candidates = ROUTING_TABLE[task_type]
    fallback_attempts: List[Dict[str, str]] = []

    for cand in candidates:
        provider = cand["provider"]
        preferred_model = cand.get("model")

        if not is_provider_ready(provider):
            fallback_attempts.append({"provider": provider, "reason": "not_configured"})
            continue

        # Set model preferensi untuk task ini (temporary override)
        original_model = secrets_store.get_provider_model(provider)
        if preferred_model and preferred_model != original_model:
            ok = secrets_store.update_provider_model(provider, preferred_model)
            if not ok:
                # Model tidak ada di catalog, skip
                fallback_attempts.append({"provider": provider, "reason": f"model_{preferred_model}_not_in_catalog"})
                continue

        # Coba chat
        result = ai_client.chat(
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if result.get("ok"):
            result["routing"] = {
                "task_type": task_type,
                "selected_provider": provider,
                "selected_model": preferred_model or original_model,
                "decision": "primary" if not fallback_attempts else f"fallback_after_{len(fallback_attempts)}_skipped",
                "fallback_chain": fallback_attempts,
            }
            return result

        # Kalau fail, restore model lama plus coba candidate berikutnya
        if preferred_model and original_model:
            secrets_store.update_provider_model(provider, original_model)
        fallback_attempts.append({
            "provider": provider,
            "reason": result.get("error", "unknown_error")[:200],
        })

    # Semua candidate fail
    return {
        "ok": False,
        "error": f"All {len(candidates)} candidates failed for task '{task_type}'. "
                 f"Check provider configurations atau model availability.",
        "routing": {
            "task_type": task_type,
            "selected_provider": None,
            "selected_model": None,
            "decision": "all_failed",
            "fallback_chain": fallback_attempts,
        },
    }


def route_stream(task_type: str, messages: List[Dict[str, str]], max_tokens: int = 2048,
                 temperature: float = 0.7, override_provider: Optional[str] = None,
                 override_model: Optional[str] = None):
    """Streaming variant dari route(). Yield text chunks plus dict event di
    akhir dengan routing info.

    Yields:
        - dict {'type': 'routing', 'provider': ..., 'model': ...} di awal
        - str text chunks selama AI generate
        - dict {'type': 'done'} di akhir (atau {'type': 'error', 'error': ...} jika fail)
    """
    task_type = (task_type or "chat_general").lower()
    if task_type not in ROUTING_TABLE:
        task_type = "chat_general"

    candidates = [{"provider": override_provider, "model": override_model}] if override_provider else ROUTING_TABLE[task_type]
    fallback_attempts: List[Dict[str, str]] = []

    for cand in candidates:
        provider = cand["provider"]
        preferred_model = cand.get("model")

        if not is_provider_ready(provider):
            fallback_attempts.append({"provider": provider, "reason": "not_configured"})
            continue

        original_model = secrets_store.get_provider_model(provider)
        if preferred_model and preferred_model != original_model:
            ok = secrets_store.update_provider_model(provider, preferred_model)
            if not ok:
                fallback_attempts.append({"provider": provider, "reason": f"model_{preferred_model}_not_in_catalog"})
                continue

        client = ai_client.get_client(provider)
        if client is None:
            fallback_attempts.append({"provider": provider, "reason": "client_factory_returned_none"})
            continue

        try:
            yield {"type": "routing", "provider": provider, "model": preferred_model or original_model, "task_type": task_type}
            for chunk in client.chat_stream(messages=messages, max_tokens=max_tokens, temperature=temperature):
                if chunk:
                    yield chunk
            yield {"type": "done", "provider": provider, "model": preferred_model or original_model}
            return
        except Exception as exc:
            # Fail, try next candidate
            if preferred_model and original_model:
                secrets_store.update_provider_model(provider, original_model)
            fallback_attempts.append({"provider": provider, "reason": str(exc)[:200]})
            continue

    # Semua candidate fail
    yield {"type": "error", "error": f"All {len(candidates)} candidates failed for task '{task_type}'", "fallback_chain": fallback_attempts}


def list_routing_rules() -> Dict[str, Any]:
    """Return routing table untuk UI display."""
    rules = {}
    for task, candidates in ROUTING_TABLE.items():
        rules[task] = {
            "candidates": candidates,
            "primary": candidates[0] if candidates else None,
        }
    return {"tasks": list(ROUTING_TABLE.keys()), "rules": rules}


def get_provider_readiness() -> Dict[str, bool]:
    """Snapshot status connection setiap provider · dipakai UI dashboard."""
    providers = ["anthropic", "openai", "gemini", "deepseek", "kimi", "ollama"]
    return {p: is_provider_ready(p) for p in providers}
