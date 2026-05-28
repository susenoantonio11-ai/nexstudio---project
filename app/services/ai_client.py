"""
NXLYTICS · Unified AI Client (multi-provider)
==============================================
Abstraction layer untuk 6 AI provider yang didukung NXLYTICS:

    1. Anthropic Claude · native Messages API
    2. OpenAI GPT       · /v1/chat/completions
    3. Google Gemini    · /v1beta/models/{model}:generateContent
    4. DeepSeek         · OpenAI-compatible di https://api.deepseek.com/v1
    5. Kimi (Moonshot)  · OpenAI-compatible di https://api.moonshot.cn/v1
    6. Ollama (lokal)   · /api/chat di localhost:11434

Semua client expose interface yang sama:
    client.chat(messages, max_tokens, temperature) -> str

Routing dilakukan oleh dispatcher chat() yang membaca provider name plus
mengambil key dari secrets_store.

Test plus connection-check tersedia lewat test_provider(name) yang kirim
single-message ping untuk validasi key plus reachability.

DeepSeek dan Kimi technically OpenAI-compatible jadi share OpenAIClient
class dengan base URL yang di-override.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False

from app.services import secrets_store


# ============================================================================
# Endpoint configuration per provider
# ============================================================================
PROVIDER_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai":    "https://api.openai.com/v1/chat/completions",
    "gemini":    "https://generativelanguage.googleapis.com/v1beta/models",
    "deepseek":  "https://api.deepseek.com/v1/chat/completions",
    "kimi":      "https://api.moonshot.cn/v1/chat/completions",
    "ollama":    "http://localhost:11434/api/chat",
}


# ============================================================================
# Base class plus per-provider implementations
# ============================================================================
class AIClient:
    """Base class untuk semua AI provider. Subclass implement chat() dan
    optionally chat_stream() untuk streaming response."""
    name: str = "base"

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7) -> str:
        raise NotImplementedError

    def chat_stream(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7):
        """Generator yang yield text chunks satu per satu. Default fallback
        ke non-streaming, yield seluruh response sekaligus di akhir. Subclass
        bisa override untuk true streaming."""
        text = self.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
        yield text


class AnthropicClient(AIClient):
    """Anthropic Claude · pakai native Messages API."""
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-opus-4-6"):
        self.api_key = api_key
        self.model = model

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7) -> str:
        # Separate system message dari list messages
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})

        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_msg:
            body["system"] = system_msg

        with httpx.Client(timeout=60.0) as cli:
            r = cli.post(
                PROVIDER_ENDPOINTS["anthropic"],
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
            data = r.json()
            return data["content"][0]["text"]

    def chat_stream(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7):
        """Streaming via Anthropic Messages API server-sent events.
        Event types: message_start, content_block_start, content_block_delta
        (yang punya text), content_block_stop, message_delta, message_stop.
        Yield hanya text deltas supaya consumer dapet pure text stream."""
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})

        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
            "stream": True,
        }
        if system_msg:
            body["system"] = system_msg

        with httpx.Client(timeout=120.0) as cli:
            with cli.stream(
                "POST",
                PROVIDER_ENDPOINTS["anthropic"],
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            ) as r:
                r.raise_for_status()
                # Anthropic SSE format: lines berupa "event: ..." dan "data: {...}"
                # Hanya data lines yang relevant. Parse JSON dan extract delta text.
                import json as _json
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]  # strip "data: "
                    if payload == "[DONE]":
                        break
                    try:
                        obj = _json.loads(payload)
                        ev_type = obj.get("type", "")
                        if ev_type == "content_block_delta":
                            delta = obj.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_chunk = delta.get("text", "")
                                if text_chunk:
                                    yield text_chunk
                        # Ignore other event types untuk simplicity
                    except Exception:
                        continue


class OpenAICompatibleClient(AIClient):
    """OpenAI plus DeepSeek plus Kimi pakai pattern yang sama."""
    def __init__(self, api_key: str, model: str, endpoint: str, name: str):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.name = name

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7) -> str:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        with httpx.Client(timeout=60.0) as cli:
            r = cli.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

    def chat_stream(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7):
        """OpenAI-compatible streaming via SSE 'data: {chunk}' format. Setiap
        chunk berisi choices[0].delta.content yang merupakan text delta.
        Last chunk adalah 'data: [DONE]'. Yield pure text chunks."""
        import json as _json
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
        }
        with httpx.Client(timeout=120.0) as cli:
            with cli.stream(
                "POST",
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        obj = _json.loads(payload)
                        choices = obj.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            text_chunk = delta.get("content", "")
                            if text_chunk:
                                yield text_chunk
                    except Exception:
                        continue


class GeminiClient(AIClient):
    """Google Gemini · REST API."""
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7) -> str:
        # Convert OpenAI format ke Gemini contents format
        contents = []
        system_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        url = f"{PROVIDER_ENDPOINTS['gemini']}/{self.model}:generateContent?key={self.api_key}"
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        with httpx.Client(timeout=60.0) as cli:
            r = cli.post(url, json=body)
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]


class OllamaClient(AIClient):
    """Ollama · self-hosted local model. No API key needed."""
    name = "ollama"

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7) -> str:
        with httpx.Client(timeout=120.0) as cli:
            r = cli.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]

    def chat_stream(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7):
        """Ollama streaming via NDJSON · setiap baris adalah JSON object dengan
        key 'message.content' (text chunk) dan 'done' flag. Yield text chunks."""
        import json as _json
        with httpx.Client(timeout=180.0) as cli:
            with cli.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                        chunk = (obj.get("message") or {}).get("content", "")
                        if chunk:
                            yield chunk
                        if obj.get("done"):
                            break
                    except Exception:
                        continue


# ============================================================================
# Factory · dispatch ke client yang sesuai
# ============================================================================
def get_client(provider: str) -> Optional[AIClient]:
    """Build instance client untuk provider tertentu. Return None kalau key
    tidak tersedia (kecuali Ollama yang no-key)."""
    provider = provider.lower()
    model = secrets_store.get_provider_model(provider)
    key = secrets_store.get_provider_key(provider)

    if provider == "anthropic":
        if not key:
            return None
        return AnthropicClient(api_key=key, model=model or "claude-opus-4-6")

    if provider == "openai":
        if not key:
            return None
        return OpenAICompatibleClient(api_key=key, model=model or "gpt-4o",
                                       endpoint=PROVIDER_ENDPOINTS["openai"], name="openai")

    if provider == "gemini":
        if not key:
            return None
        return GeminiClient(api_key=key, model=model or "gemini-2.0-flash")

    if provider == "deepseek":
        if not key:
            return None
        return OpenAICompatibleClient(api_key=key, model=model or "deepseek-chat",
                                       endpoint=PROVIDER_ENDPOINTS["deepseek"], name="deepseek")

    if provider == "kimi":
        if not key:
            return None
        return OpenAICompatibleClient(api_key=key, model=model or "moonshot-v1-32k",
                                       endpoint=PROVIDER_ENDPOINTS["kimi"], name="kimi")

    if provider == "ollama":
        return OllamaClient(model=model or "llama3.2")

    return None


def chat(provider: str, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7) -> Dict[str, Any]:
    """Top-level entry point yang dipakai oleh /api/ai/chat. Return dict
    dengan key 'ok', 'text' atau 'error'."""
    if not _HTTPX_AVAILABLE:
        return {"ok": False, "error": "httpx belum terinstall · jalankan pip install httpx==0.27.2"}
    cli = get_client(provider)
    if cli is None:
        return {"ok": False, "error": f"Provider '{provider}' belum di-configure. Set API key di Settings."}
    try:
        text = cli.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
        return {"ok": True, "provider": provider, "text": text}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "provider": provider, "error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except httpx.RequestError as e:
        return {"ok": False, "provider": provider, "error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"ok": False, "provider": provider, "error": f"Unexpected: {str(e)}"}


def test_provider(provider: str) -> Dict[str, Any]:
    """Kirim ping minimal untuk validasi key plus reachability.

    Untuk provider chat (AI), kirim 1 message pendek. Untuk provider non-chat
    seperti Semantic Scholar, lakukan GET test ke endpoint search.
    """
    provider = provider.lower()

    # Special handler · Semantic Scholar adalah academic search bukan chat AI
    if provider == "semantic_scholar":
        key = secrets_store.get_provider_key(provider)
        if not key:
            return {"ok": False, "error": "No API key configured."}
        try:
            import httpx
            with httpx.Client(timeout=10.0) as client:
                r = client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": "test", "limit": 1, "fields": "title"},
                    headers={"x-api-key": key, "User-Agent": "NXLYTICS/1.0"},
                )
                if r.status_code == 200:
                    data = r.json()
                    total = data.get("total", 0)
                    return {"ok": True, "reply": f"API key valid. Test search returned total {total} papers.", "provider": provider}
                if r.status_code == 401 or r.status_code == 403:
                    return {"ok": False, "error": f"HTTP {r.status_code}: API key tidak valid atau ditolak."}
                if r.status_code == 429:
                    return {"ok": False, "error": "HTTP 429: rate limited. Try again in 1 minute."}
                return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"ok": False, "error": f"Network error: {str(e)[:200]}", "provider": provider}

    # Default · chat AI provider
    cli = get_client(provider)
    if cli is None:
        return {"ok": False, "error": "No client available. Set key dulu."}
    try:
        reply = cli.chat(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
            temperature=0.0,
        )
        return {"ok": True, "reply": reply[:50], "provider": provider}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300], "provider": provider}
