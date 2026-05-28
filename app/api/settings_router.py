"""
NXLYTICS · Settings Router (Multi-Provider AI Keys)
=====================================================
FastAPI endpoints untuk manage AI provider API keys plus model selection.

Endpoints:
    GET    /api/settings/providers              · list semua provider + status
    GET    /api/settings/providers/{name}       · status single provider
    POST   /api/settings/providers/{name}       · save key + model
    DELETE /api/settings/providers/{name}       · revoke key
    POST   /api/settings/providers/{name}/test  · validate key dengan ping API
    POST   /api/settings/providers/{name}/model · update default model
    POST   /api/ai/chat                          · proxy chat dengan provider routing

SECURITY: API key tidak pernah di-return ke frontend. Hanya 4 karakter terakhir.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import ai_client, secrets_store

# Optional · task_router plus academic_search butuh ai_client plus httpx
try:
    from app.services import task_router as _task_router
    _TASK_ROUTER_AVAILABLE = True
except Exception as _e:
    _task_router = None
    _TASK_ROUTER_AVAILABLE = False

try:
    from app.services import academic_search as _academic_search
    _ACADEMIC_SEARCH_AVAILABLE = True
except Exception as _e:
    _academic_search = None
    _ACADEMIC_SEARCH_AVAILABLE = False

router = APIRouter(prefix="/api", tags=["settings"])


# ============================================================================
# Request schemas
# ============================================================================
class SaveProviderRequest(BaseModel):
    key: str = Field(..., min_length=0, description="API key plaintext, akan di-encrypt di backend")
    model: Optional[str] = Field(None, description="Optional model override")


class UpdateModelRequest(BaseModel):
    model: str


class ChatMessage(BaseModel):
    role: str = Field(..., description="system / user / assistant")
    content: str


class ChatRequest(BaseModel):
    provider: str = Field(..., description="anthropic / openai / gemini / deepseek / kimi / ollama")
    messages: List[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7


# ============================================================================
# Provider list plus status
# ============================================================================
@router.get("/settings/providers")
def list_providers() -> Dict[str, Any]:
    """Daftar semua provider yang available dengan status connection."""
    return {
        "status": "success",
        "providers": secrets_store.list_providers(),
    }


@router.get("/settings/providers/{name}")
def get_provider(name: str) -> Dict[str, Any]:
    status = secrets_store.get_provider_status(name)
    if not status:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {name}")
    return {"status": "success", "provider": status}


# ============================================================================
# Save plus delete plus test
# ============================================================================
@router.post("/settings/providers/{name}")
def save_provider(name: str, req: SaveProviderRequest) -> Dict[str, Any]:
    """Simpan API key terenkripsi. Frontend kirim key plaintext via HTTPS body."""
    result = secrets_store.set_provider_key(name, req.key, req.model)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("message", "Save failed"))
    return {"status": "success", **result}


@router.delete("/settings/providers/{name}")
def delete_provider(name: str) -> Dict[str, Any]:
    ok = secrets_store.delete_provider(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Provider not configured: {name}")
    return {"status": "success", "deleted": name}


@router.post("/settings/providers/{name}/test")
def test_provider(name: str) -> Dict[str, Any]:
    """Validasi API key dengan kirim ping kecil ke provider."""
    result = ai_client.test_provider(name)
    return {"status": "success" if result.get("ok") else "error", **result}


@router.post("/settings/providers/{name}/model")
def update_model(name: str, req: UpdateModelRequest) -> Dict[str, Any]:
    ok = secrets_store.update_provider_model(name, req.model)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Invalid model '{req.model}' for provider '{name}'")
    return {"status": "success", "name": name, "model": req.model}


# ============================================================================
# AI chat proxy · entry point untuk semua AI feature di NXLYTICS
# ============================================================================
@router.post("/ai/chat")
def ai_chat(req: ChatRequest) -> Dict[str, Any]:
    """Proxy chat request ke provider yang dipilih. Frontend tidak pernah
    perlu API key, semua handled di backend."""
    messages = [m.model_dump() for m in req.messages]
    result = ai_client.chat(
        provider=req.provider,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    if not result.get("ok"):
        return {"status": "error", **result}
    return {"status": "success", **result}


# ============================================================================
# Multi-Model Task Orchestration (Arah 3)
# ============================================================================
class OrchestrateRequest(BaseModel):
    task_type: str = Field(..., description="summarize/reason/code/translate/fact_check/classify/draft_academic/chat_general")
    messages: List[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    override_provider: Optional[str] = None
    override_model: Optional[str] = None


@router.post("/ai/orchestrate")
def ai_orchestrate(req: OrchestrateRequest) -> Dict[str, Any]:
    """Task-aware AI routing. Otomatis pilih provider+model optimal per task type.
    Misal task='summarize' → DeepSeek R1 lokal (hemat). task='reason' → Claude Opus.
    Lihat task_router.ROUTING_TABLE untuk full mapping."""
    if not _TASK_ROUTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Task router not available · check ai_client plus httpx install")
    messages = [m.model_dump() for m in req.messages]
    result = _task_router.route(
        task_type=req.task_type,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        override_provider=req.override_provider,
        override_model=req.override_model,
    )
    return {"status": "success" if result.get("ok") else "error", **result}


@router.get("/ai/routing-rules")
def ai_routing_rules() -> Dict[str, Any]:
    """Return routing table untuk UI display · user lihat task mana route ke provider mana."""
    if not _TASK_ROUTER_AVAILABLE:
        return {"status": "error", "message": "Task router not available"}
    return {"status": "success", **_task_router.list_routing_rules()}


@router.get("/ai/providers-status")
def ai_providers_status() -> Dict[str, Any]:
    """Status connection setiap provider · dipakai dashboard plus routing decision."""
    if not _TASK_ROUTER_AVAILABLE:
        return {"status": "error", "message": "Task router not available"}
    return {"status": "success", "providers": _task_router.get_provider_readiness()}


# ============================================================================
# Academic Search Aggregator (multi-source)
# ============================================================================
class AcademicSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    max_per_source: int = Field(default=15, ge=1, le=50)
    year_from: Optional[int] = None
    sources: Optional[List[str]] = Field(default=None, description="openalex, crossref, semantic_scholar, arxiv")
    use_cache: bool = Field(default=True, description="Set false untuk force fresh result (skip in-memory cache)")


@router.post("/academic/search")
async def academic_search(req: AcademicSearchRequest) -> Dict[str, Any]:
    """Search paper akademik di multiple database paralel.

    Default sources: OpenAlex (Scopus equivalent), Crossref (DOI registry),
    Semantic Scholar (citation graph), arXiv (preprints).

    Returns: list paper unified format dengan dedup via DOI plus title-year.
    Relevance scored berdasarkan keyword overlap + citation count + recency."""
    if not _ACADEMIC_SEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Academic search not available · install httpx")
    result = await _academic_search.search_all(
        query=req.query,
        max_per_source=req.max_per_source,
        year_from=req.year_from,
        include_sources=req.sources,
        use_cache=req.use_cache,
    )
    return result


@router.get("/academic/diagnose")
async def academic_diagnose(query: str = "flood prediction") -> Dict[str, Any]:
    """Test setiap source academic search secara terpisah dengan output verbose.

    Untuk troubleshoot kalau ada source yang return 0. Tiap source diuji
    independent dengan kueri pendek dan timeout sendiri, output mencakup
    hitungan paper, durasi, plus pesan error eksplisit.
    """
    if not _ACADEMIC_SEARCH_AVAILABLE:
        return {"status": "error", "message": "Academic search not available · install httpx"}

    import time
    try:
        import httpx
    except ImportError:
        return {"status": "error", "message": "httpx not installed"}

    diagnostics = []
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for label, fn in [
            ("openalex", _academic_search.search_openalex),
            ("crossref", _academic_search.search_crossref),
            ("semantic_scholar", _academic_search.search_semantic_scholar),
            ("arxiv", _academic_search.search_arxiv),
        ]:
            t0 = time.time()
            try:
                if label == "arxiv":
                    res = await fn(client, query, 5)
                else:
                    res = await fn(client, query, 5, None)
                if isinstance(res, dict):
                    papers_n = len(res.get("papers", []))
                    err = res.get("error")
                else:
                    papers_n = len(res) if isinstance(res, list) else 0
                    err = None
                # Tiga state: ok (sukses ada papers), zero_results (API OK tapi
                # 0 hasil), error (API call gagal atau timeout). Bedakan supaya
                # user tidak menganggap kueri yang return 0 sebagai bug.
                if err:
                    status_label = "error"
                elif papers_n == 0:
                    status_label = "zero_results"
                else:
                    status_label = "ok"
                diagnostics.append({
                    "source": label,
                    "ok": status_label == "ok",
                    "status": status_label,
                    "papers": papers_n,
                    "error": err,
                    "duration_ms": int((time.time() - t0) * 1000),
                })
            except Exception as e:
                diagnostics.append({
                    "source": label,
                    "ok": False,
                    "papers": 0,
                    "error": f"{type(e).__name__}: {e}",
                    "duration_ms": int((time.time() - t0) * 1000),
                })

    return {
        "status": "success",
        "query": query,
        "diagnostics": diagnostics,
        "summary": {
            "total_sources": len(diagnostics),
            "ok_sources": sum(1 for d in diagnostics if d.get("status") == "ok"),
            "zero_results_sources": sum(1 for d in diagnostics if d.get("status") == "zero_results"),
            "error_sources": sum(1 for d in diagnostics if d.get("status") == "error"),
        },
        "hint": "Sources with status zero_results returned HTTP 200 but no papers match the query. This is NOT an error. Try different keywords or use the full title.",
    }


@router.post("/academic/cache/clear")
def academic_cache_clear() -> Dict[str, Any]:
    """Clear in-memory cache academic search. Dipakai setelah user pasang
    API key baru supaya hasil stale (dari sebelum key aktif) bisa di-refresh
    tanpa restart backend.
    """
    if not _ACADEMIC_SEARCH_AVAILABLE:
        return {"status": "error", "message": "Academic search not available"}
    result = _academic_search.clear_cache()
    return {"status": "success", **result}


@router.get("/academic/sources")
def academic_sources() -> Dict[str, Any]:
    """List sources yang available untuk academic search."""
    return {
        "status": "success",
        "sources": [
            {"id": "openalex", "label": "OpenAlex", "free": True, "coverage": "200M+ works, Scopus equivalent"},
            {"id": "crossref", "label": "Crossref", "free": True, "coverage": "130M+ DOI registry"},
            {"id": "semantic_scholar", "label": "Semantic Scholar", "free": True, "coverage": "200M+ papers, citation graph"},
            {"id": "arxiv", "label": "arXiv", "free": True, "coverage": "preprints physics, CS, math, stat"},
        ],
        "future_sources": [
            {"id": "sinta", "label": "SINTA (Indonesia)", "status": "planned · butuh scraper"},
            {"id": "google_scholar", "label": "Google Scholar", "status": "planned · rate limited"},
            {"id": "scopus", "label": "Scopus", "status": "needs institutional API subscription"},
        ],
    }
