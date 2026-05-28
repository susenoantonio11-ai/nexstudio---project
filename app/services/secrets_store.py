"""
NXLYTICS · Encrypted Secrets Store
====================================
File-backed secrets storage untuk API keys multi-provider AI.

Encryption: Fernet (AES-128-CBC + HMAC-SHA256) dari library cryptography.
Master key: derived dari machine-bound seed (hostname + MAC) dengan PBKDF2.
Storage: backend/storage/.nxlytics-secrets.json dengan filesystem permission 600.

WARNING: master key derived dari machine info, jadi file secrets tidak portable
antar device. Kalau pindah laptop, user harus re-enter keys. Ini intentional
untuk mencegah accidental git commit atau file leak.

Format file:
{
    "providers": {
        "anthropic": {
            "ciphertext": "gAAAAAB...",
            "last_4": "ABcd",
            "updated_at": "2026-05-20T07:00:00",
            "model": "claude-opus-4-6"
        },
        ...
    }
}

Public API:
    get_provider_status(name) → dict tanpa ciphertext (UI-safe)
    list_providers() → list of statuses
    set_provider_key(name, key, model) → bool
    get_provider_key(name) → str (decrypted, internal use)
    delete_provider(name) → bool
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _CRYPTO_AVAILABLE = True
except ImportError:
    # Gracefully degrade · backend tetap jalan tanpa secrets endpoints
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore
    hashes = None  # type: ignore
    PBKDF2HMAC = None  # type: ignore
    _CRYPTO_AVAILABLE = False

# ============================================================================
# Provider catalog · definisi metadata untuk UI plus validasi
# ============================================================================
PROVIDER_CATALOG: Dict[str, Dict[str, Any]] = {
    "anthropic": {
        "label": "Anthropic Claude",
        "description": "Claude Opus 4, Sonnet 4, Haiku 4.5",
        "default_model": "claude-opus-4-6",
        "available_models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "key_prefix": "sk-ant-",
        "key_min_length": 50,
        "icon": "ti-brain",
    },
    "openai": {
        "label": "OpenAI",
        "description": "GPT-4o, GPT-4 Turbo, GPT-3.5",
        "default_model": "gpt-4o",
        "available_models": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "key_prefix": "sk-",
        "key_min_length": 40,
        "icon": "ti-message-chatbot",
    },
    "gemini": {
        "label": "Google Gemini",
        "description": "Gemini 2.0 Flash, Gemini 1.5 Pro",
        "default_model": "gemini-2.0-flash",
        "available_models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "key_prefix": "AIza",
        "key_min_length": 30,
        "icon": "ti-sparkles",
    },
    "deepseek": {
        "label": "DeepSeek",
        "description": "DeepSeek Chat, Coder, Reasoner (R1)",
        "default_model": "deepseek-chat",
        "available_models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "key_prefix": "sk-",
        "key_min_length": 30,
        "icon": "ti-search",
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "description": "Moonshot v1 8k, 32k, 128k context",
        "default_model": "moonshot-v1-32k",
        "available_models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "key_prefix": "sk-",
        "key_min_length": 30,
        "icon": "ti-moon",
    },
    "ollama": {
        "label": "Ollama (lokal)",
        "description": "Self-hosted models · no API key needed",
        "default_model": "llama3.2:latest",
        "available_models": [
            # Llama family
            "llama3.2:latest", "llama3.2:1b", "llama3.2:3b",
            "llama3.1:8b", "llama3.1:70b",
            # Mistral
            "mistral:latest", "mistral:7b",
            # Qwen
            "qwen2.5:latest", "qwen2.5:7b", "qwen2.5:14b",
            # DeepSeek R1 · semua variant ukuran
            "deepseek-r1:latest", "deepseek-r1:1.5b", "deepseek-r1:7b",
            "deepseek-r1:8b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b",
            # DeepSeek coder
            "deepseek-coder-v2:latest", "deepseek-coder-v2:16b",
            # Code specialized
            "codellama:latest", "codellama:7b",
            # Phi
            "phi3:latest", "phi3:mini"
        ],
        "key_prefix": "",
        "key_min_length": 0,
        "icon": "ti-server",
    },
    "semantic_scholar": {
        "label": "Semantic Scholar API",
        "description": "Academic search engine · 200M+ papers · citation graph",
        "default_model": "graph-v1",
        "available_models": ["graph-v1"],
        "key_prefix": "",
        "key_min_length": 20,
        "icon": "ti-book",
        "category": "academic_search",
        "key_url": "https://www.semanticscholar.org/product/api",
    },
}


# ============================================================================
# File location plus master key derivation
# ============================================================================
def _storage_path() -> Path:
    p = Path(__file__).resolve().parents[2] / "storage" / ".nxlytics-secrets.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _machine_seed() -> bytes:
    """Derive machine-bound seed dari hostname plus MAC. Tidak rotateable tapi
    cukup untuk mencegah file dipindahkan ke laptop lain dapat di-decrypt."""
    try:
        host = socket.gethostname()
    except Exception:
        host = "unknown-host"
    try:
        mac = uuid.getnode()
    except Exception:
        mac = 0
    seed = f"nxlytics-v1:{host}:{mac}".encode("utf-8")
    return hashlib.sha256(seed).digest()


def _derive_master_key() -> bytes:
    """PBKDF2-HMAC-SHA256 dengan 600.000 iterations sesuai OWASP 2023."""
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library belum terinstall · pip install cryptography==43.0.3")
    seed = _machine_seed()
    salt = b"nxlytics-secrets-v1"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600_000,
    )
    key = kdf.derive(seed)
    return base64.urlsafe_b64encode(key)


_FERNET = None  # type: ignore


def _fernet():
    global _FERNET
    if _FERNET is None:
        if not _CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library belum terinstall · pip install cryptography==43.0.3")
        _FERNET = Fernet(_derive_master_key())
    return _FERNET


def is_available() -> bool:
    """Cek apakah cryptography library tersedia · dipakai UI guard."""
    return _CRYPTO_AVAILABLE


# ============================================================================
# File I/O helpers
# ============================================================================
def _load_data() -> Dict[str, Any]:
    p = _storage_path()
    if not p.exists():
        return {"providers": {}}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"providers": {}}


def _save_data(data: Dict[str, Any]) -> None:
    p = _storage_path()
    p.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass


# ============================================================================
# PUBLIC API
# ============================================================================
def list_providers() -> List[Dict[str, Any]]:
    """List semua provider yang available beserta status connected/disconnected.
    UI-safe · tidak mengembalikan ciphertext atau plaintext key.
    Kalau crypto belum terinstall, tetap return daftar provider tapi semua
    disconnected · UI bisa tampilkan instruksi install."""
    data = _load_data() if _CRYPTO_AVAILABLE else {"providers": {}}
    saved = data.get("providers", {})
    out = []
    for name, meta in PROVIDER_CATALOG.items():
        entry = saved.get(name, {})
        out.append({
            "name": name,
            "label": meta["label"],
            "description": meta["description"],
            "icon": meta.get("icon", "ti-bolt"),
            "default_model": meta["default_model"],
            "available_models": meta["available_models"],
            "needs_key": meta["key_min_length"] > 0,
            "connected": bool(entry.get("ciphertext")) or name == "ollama",
            "last_4": entry.get("last_4", ""),
            "model": entry.get("model", meta["default_model"]),
            "updated_at": entry.get("updated_at", ""),
            "category": meta.get("category", "ai_llm"),
            "key_url": meta.get("key_url"),
        })
    return out


def get_provider_status(name: str) -> Optional[Dict[str, Any]]:
    name = name.lower()
    if name not in PROVIDER_CATALOG:
        return None
    data = _load_data()
    entry = data.get("providers", {}).get(name, {})
    meta = PROVIDER_CATALOG[name]
    return {
        "name": name,
        "label": meta["label"],
        "connected": bool(entry.get("ciphertext")) or name == "ollama",
        "last_4": entry.get("last_4", ""),
        "model": entry.get("model", meta["default_model"]),
        "updated_at": entry.get("updated_at", ""),
    }


def set_provider_key(name: str, key: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Simpan API key terenkripsi. Validasi prefix plus length sebelum save."""
    if not _CRYPTO_AVAILABLE:
        return {"ok": False, "message": "cryptography library belum terinstall di backend. Install dulu: pip install cryptography==43.0.3 httpx==0.27.2"}
    name = name.lower()
    if name not in PROVIDER_CATALOG:
        return {"ok": False, "message": f"Unknown provider: {name}"}
    meta = PROVIDER_CATALOG[name]
    key = (key or "").strip()
    if meta["key_min_length"] > 0:
        if len(key) < meta["key_min_length"]:
            return {"ok": False, "message": f"Key terlalu pendek. Minimal {meta['key_min_length']} karakter."}
        if meta["key_prefix"] and not key.startswith(meta["key_prefix"]):
            return {"ok": False, "message": f"Format key tidak valid. Harus mulai dengan '{meta['key_prefix']}'."}

    ciphertext = ""
    if key:
        ciphertext = _fernet().encrypt(key.encode("utf-8")).decode("utf-8")

    data = _load_data()
    data.setdefault("providers", {})
    data["providers"][name] = {
        "ciphertext": ciphertext,
        "last_4": key[-4:] if len(key) >= 4 else "",
        "updated_at": datetime.utcnow().isoformat(),
        "model": model or meta["default_model"],
    }
    _save_data(data)
    return {"ok": True, "name": name, "last_4": key[-4:] if len(key) >= 4 else "", "connected": True}


def get_provider_key(name: str) -> Optional[str]:
    """Internal use only · return decrypted API key. Never expose lewat HTTP."""
    name = name.lower()
    data = _load_data()
    entry = data.get("providers", {}).get(name)
    if not entry or not entry.get("ciphertext"):
        return None
    try:
        plain = _fernet().decrypt(entry["ciphertext"].encode("utf-8"))
        return plain.decode("utf-8")
    except InvalidToken:
        return None
    except Exception:
        return None


def get_provider_model(name: str) -> Optional[str]:
    name = name.lower()
    data = _load_data()
    entry = data.get("providers", {}).get(name, {})
    if entry.get("model"):
        return entry["model"]
    meta = PROVIDER_CATALOG.get(name)
    return meta["default_model"] if meta else None


def delete_provider(name: str) -> bool:
    name = name.lower()
    data = _load_data()
    providers = data.get("providers", {})
    if name in providers:
        del providers[name]
        _save_data(data)
        return True
    return False


def update_provider_model(name: str, model: str) -> bool:
    name = name.lower()
    if name not in PROVIDER_CATALOG:
        return False
    if model not in PROVIDER_CATALOG[name]["available_models"]:
        return False
    data = _load_data()
    data.setdefault("providers", {}).setdefault(name, {})
    data["providers"][name]["model"] = model
    data["providers"][name]["updated_at"] = datetime.utcnow().isoformat()
    _save_data(data)
    return True
