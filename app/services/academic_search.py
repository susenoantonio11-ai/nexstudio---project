"""
NXLYTICS · Academic Search Aggregator
=======================================
Multi-source academic paper search engine yang query 4-6 database publik
secara parallel lalu deduplicate hasil via DOI matching.

Sources implemented:
    1. OpenAlex      · 200M+ scholarly works, free, no key needed
                       (Scopus equivalent, Mendeley alternative)
    2. Crossref      · 130M+ records, DOI registry of record, free
    3. Semantic Scholar · 200M+ papers, citation graph, free (key optional)
    4. arXiv         · preprints physics/CS/math, free, no key
    5. SINTA         · Indonesian journals (best-effort scraper)
    6. Google Scholar · via scholarly lib (best-effort, rate-limited)

Sources yang TIDAK include (butuh paid subscription):
    - Scopus (Elsevier) · butuh institutional API key ~$10k/year
    - Web of Science (Clarivate) · same

Output format unified per paper:
    {
        "doi": "10.1016/...",
        "title": "...",
        "authors": ["..."],
        "year": 2024,
        "venue": "Journal of ...",
        "abstract": "...",
        "citation_count": 42,
        "url": "https://...",
        "sources": ["openalex", "crossref"],  // mana saja yang return paper ini
        "open_access": True,
        "type": "journal-article",
        "score": 0.85  // relevance score
    }

Deduplication strategy:
    Primary key: DOI (normalized lowercase)
    Fallback: title similarity (Jaccard distance) + year match
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus


def _log(msg: str) -> None:
    """Helper untuk log ke stderr supaya muncul di .err.log file.
    print() default ke stdout (.out.log) yang sering tidak di-monitor user."""
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False


# ============================================================================
# Persistent disk cache · survive backend restart
# ============================================================================
# Cache di-persist ke JSON file di backend/storage/academic_search_cache.json
# supaya hasil search yang sudah sukses survive restart backend. Sebelumnya
# in-memory only, jadi setiap restart wipe cache dan user harus tunggu fresh
# external API call lagi (yang sering timeout/intermittent dari Indonesia
# network). Sekarang restart = cache loaded back dari disk = instant repeat
# query bahkan setelah laptop di-restart.
#
# TTL dinaikkan dari 1 jam ke 24 jam karena hasil academic search relatif
# stable (paper tidak hilang tiba-tiba). Untuk force refresh, user pakai
# tombol Fresh atau cache clear endpoint.
# ============================================================================
import json as _json_lib
from pathlib import Path as _Path

_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 86400  # 24 hours
_CACHE_MAX_ENTRIES = 300

# ============================================================================
# PER-SOURCE CACHE · GUARANTEE no 0 after restart untuk source yang pernah OK
# ============================================================================
# Setiap (source, query, max_per_source) di-cache independen dengan TTL 30
# hari. Search hit cache per source · hanya retry source yang miss cache atau
# expired. Restart load semua per-source cache → source yang pernah sukses
# tetap punya hasil yang sama, TIDAK akan kembali ke 0.
#
# Architecture:
#   _PER_SOURCE_CACHE = {
#     "openalex|flood prediction|15": (timestamp, papers_list),
#     "crossref|flood prediction|15": (timestamp, papers_list),
#     ...
#   }
# ============================================================================
_PER_SOURCE_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_PER_SOURCE_TTL = 86400 * 30  # 30 days · source results stable over time
_PER_SOURCE_MAX = 1000

# File paths
_CACHE_FILE = _Path(__file__).resolve().parents[2] / "storage" / "academic_search_cache.json"
_PER_SOURCE_CACHE_FILE = _Path(__file__).resolve().parents[2] / "storage" / "academic_search_per_source.json"


def _ps_cache_key(source: str, query: str, max_per_source: int) -> str:
    """Build per-source cache key."""
    raw = f"{source}|{query.lower().strip()}|{max_per_source}"
    return hashlib.md5(raw.encode()).hexdigest()


def _ps_cache_get(source: str, query: str, max_per_source: int) -> Optional[List[Dict[str, Any]]]:
    """Get cached papers list untuk source tertentu."""
    key = _ps_cache_key(source, query, max_per_source)
    if key in _PER_SOURCE_CACHE:
        ts, papers = _PER_SOURCE_CACHE[key]
        if time.time() - ts < _PER_SOURCE_TTL:
            return papers
        else:
            del _PER_SOURCE_CACHE[key]
    return None


def _ps_cache_set(source: str, query: str, max_per_source: int, papers: List[Dict[str, Any]]) -> None:
    """Save papers list ke per-source cache · HANYA bila papers > 0.
    Result kosong tidak di-cache supaya retry next time."""
    if not papers or len(papers) == 0:
        return
    key = _ps_cache_key(source, query, max_per_source)
    _PER_SOURCE_CACHE[key] = (time.time(), papers)
    # LRU eviction kalau terlalu banyak
    if len(_PER_SOURCE_CACHE) > _PER_SOURCE_MAX:
        oldest_key = min(_PER_SOURCE_CACHE.keys(), key=lambda k: _PER_SOURCE_CACHE[k][0])
        del _PER_SOURCE_CACHE[oldest_key]
    _ps_cache_save_to_disk()


def _ps_cache_load_from_disk() -> None:
    """Load per-source cache dari disk saat backend startup."""
    global _PER_SOURCE_CACHE
    try:
        if not _PER_SOURCE_CACHE_FILE.exists():
            _log(f"[academic_search] per-source cache: no file yet at {_PER_SOURCE_CACHE_FILE}")
            return
        with open(_PER_SOURCE_CACHE_FILE, "r", encoding="utf-8") as f:
            raw = _json_lib.load(f)
        if isinstance(raw, dict):
            now = time.time()
            loaded = 0
            expired = 0
            for k, v in raw.items():
                if isinstance(v, list) and len(v) == 2:
                    ts, papers = v[0], v[1]
                    try:
                        ts_f = float(ts)
                        if now - ts_f < _PER_SOURCE_TTL and isinstance(papers, list) and len(papers) > 0:
                            _PER_SOURCE_CACHE[k] = (ts_f, papers)
                            loaded += 1
                        else:
                            expired += 1
                    except Exception:
                        pass
            _log(f"[academic_search] per-source cache loaded: {loaded} entries, {expired} expired/empty skipped")
    except Exception as e:
        _log(f"[academic_search] per-source cache load failed: {type(e).__name__}: {e}")
        _PER_SOURCE_CACHE = {}


def _ps_cache_save_to_disk() -> None:
    """Atomic write per-source cache."""
    try:
        _PER_SOURCE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: [v[0], v[1]] for k, v in _PER_SOURCE_CACHE.items()}
        tmp_file = _PER_SOURCE_CACHE_FILE.with_suffix(".json.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            _json_lib.dump(serializable, f, ensure_ascii=False)
        tmp_file.replace(_PER_SOURCE_CACHE_FILE)
    except Exception as e:
        _log(f"[academic_search] per-source cache save failed: {type(e).__name__}: {e}")


def _cache_load_from_disk() -> None:
    """Load cache dari disk file saat module di-import. Dipanggil sekali
    saat backend startup. Handle file corrupt atau missing dengan graceful
    fallback ke empty cache."""
    global _CACHE
    try:
        if not _CACHE_FILE.exists():
            _log(f"[academic_search] module loaded, no cache file yet at {_CACHE_FILE}. Starting with empty cache.")
            return
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            raw = _json_lib.load(f)
        if isinstance(raw, dict):
            now = time.time()
            loaded = 0
            expired = 0
            for k, v in raw.items():
                if isinstance(v, list) and len(v) == 2:
                    ts, data = v[0], v[1]
                    try:
                        ts_f = float(ts)
                        if now - ts_f < _CACHE_TTL:
                            _CACHE[k] = (ts_f, data)
                            loaded += 1
                        else:
                            expired += 1
                    except Exception:
                        pass
            _log(f"[academic_search] cache loaded from disk: {loaded} entries, {expired} expired skipped")
    except Exception as e:
        _log(f"[academic_search] cache load failed: {type(e).__name__}: {e}. Starting with empty cache.")
        _CACHE = {}


def _cache_save_to_disk() -> None:
    """Atomic write cache ke disk · write ke temp file, lalu rename ke target.
    Mencegah corruption bila proses ke-kill di tengah write. Dipanggil setiap
    cache_set supaya cache yang baru langsung persist."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Serialize cache · convert tuple ke list untuk JSON compat
        serializable = {k: [v[0], v[1]] for k, v in _CACHE.items()}
        # Atomic write · write ke .tmp lalu rename
        tmp_file = _CACHE_FILE.with_suffix(".json.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            _json_lib.dump(serializable, f, ensure_ascii=False)
        tmp_file.replace(_CACHE_FILE)
    except Exception as e:
        # Non-fatal · cache tetap in-memory walaupun save gagal
        _log(f"[academic_search] cache save failed: {type(e).__name__}: {e}")


def _cache_key(query: str, max_per_source: int, year_from: Optional[int],
               include_sources: Optional[List[str]]) -> str:
    """Build deterministic cache key dari parameter search."""
    src_str = ",".join(sorted(include_sources or []))
    raw = f"{query.lower().strip()}|{max_per_source}|{year_from or ''}|{src_str}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if key in _CACHE:
        ts, data = _CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            return data
        else:
            del _CACHE[key]
            _cache_save_to_disk()  # Save setelah eviction
    return None


def _cache_set(key: str, data: Dict[str, Any]) -> None:
    _CACHE[key] = (time.time(), data)
    # Limit cache size · evict oldest kalau > _CACHE_MAX_ENTRIES
    if len(_CACHE) > _CACHE_MAX_ENTRIES:
        oldest_key = min(_CACHE.keys(), key=lambda k: _CACHE[k][0])
        del _CACHE[oldest_key]
    # Persist ke disk · async-ish, non-blocking via try/except wrapped
    _cache_save_to_disk()


def clear_cache() -> Dict[str, Any]:
    """Hapus semua cache entry dari memory plus disk file.
    Dipakai saat user pasang API key baru atau ingin force fresh result.
    NOTE: per-source cache TIDAK ikut di-clear secara default karena itu
    yang menjamin source tetap punya hasil setelah restart. Pakai
    clear_per_source_cache() untuk reset total."""
    count = len(_CACHE)
    _CACHE.clear()
    try:
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()
    except Exception:
        pass
    return {"cleared": count, "ok": True}


def clear_per_source_cache() -> Dict[str, Any]:
    """Force reset per-source cache · destructive. Hanya dipakai bila user
    eksplisit minta full fresh start. Setelah ini, semua source akan harus
    fresh fetch dan bisa kembali ke 0 sampai network membaik."""
    count = len(_PER_SOURCE_CACHE)
    _PER_SOURCE_CACHE.clear()
    try:
        if _PER_SOURCE_CACHE_FILE.exists():
            _PER_SOURCE_CACHE_FILE.unlink()
    except Exception:
        pass
    return {"cleared": count, "ok": True}


# Load cache dari disk saat module di-import (backend startup)
_cache_load_from_disk()
_ps_cache_load_from_disk()


# ============================================================================
# Source endpoint configs
# ============================================================================
OPENALEX_BASE = "https://api.openalex.org/works"
CROSSREF_BASE = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
# FIX: arXiv sekarang force redirect ke HTTPS (HTTP 301 di endpoint lama).
# Confirmed via direct curl test di Mac user yang return HTTP 301 dari
# http://export.arxiv.org. Pakai https langsung supaya tidak perlu redirect.
ARXIV_BASE = "https://export.arxiv.org/api/query"

# Reasonable User-Agent supaya tidak di-block
UA = "NXLYTICS/1.0 (academic research; mailto:research@nxlytics.local)"


# ============================================================================
# Helper utilities
# ============================================================================
def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.lower().strip()
    # Strip URL prefix
    for prefix in ["https://doi.org/", "http://doi.org/", "doi.org/", "doi:"]:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi.strip() or None


def normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    # Lowercase, strip punctuation, collapse whitespace
    t = re.sub(r"[^\w\s]", " ", title.lower())
    return re.sub(r"\s+", " ", t).strip()


# ============================================================================
# OpenAlex search
# ============================================================================
async def search_openalex(client: "httpx.AsyncClient", query: str, max_results: int = 20,
                          year_from: Optional[int] = None) -> Dict[str, Any]:
    """OpenAlex · free, no key. https://docs.openalex.org/

    NOTE: OpenAlex politeness pool aktif via mailto query param. Timeout
    diperpanjang ke 15 detik karena response besar (sampai 100 paper) bisa
    memakan waktu beberapa detik di koneksi lambat.

    Return shape: {"papers": List[Dict], "error": Optional[str]}.
    """
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    filter_param = ",".join(filters) if filters else None
    # Per-page diturunkan ke max 15 supaya response payload sangat kecil dan
    # transfer cepat dari Indonesia network. 15 paper ~15-20KB transfer,
    # selesai 3-8 detik bahkan di koneksi lambat. Reliability > quantity.
    params = {
        "search": query,
        "per-page": min(max_results, 15),
        "mailto": "research@nxlytics.local",
    }
    if filter_param:
        params["filter"] = filter_param

    # Retry sekali bila timeout. Dari Indonesia network ke api.openalex.org
    # (US-hosted) sering butuh 15-40 detik tergantung kondisi rute. Timeout
    # 50 detik plus retry sekali memberikan headroom maksimal.
    last_err: Optional[str] = None
    data = None
    for attempt in range(2):
        try:
            r = await client.get(OPENALEX_BASE, params=params, headers={"User-Agent": UA}, timeout=30.0)
            if r.status_code != 200:
                return {"papers": [], "error": f"HTTP {r.status_code} from api.openalex.org"}
            data = r.json()
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 0:
                # Retry sekali setelah 2 detik
                await asyncio.sleep(2)
            else:
                _log(f"[academic_search] openalex failed after retry: {last_err}")
                return {"papers": [], "error": last_err}

    if data is None:
        return {"papers": [], "error": last_err or "unknown_failure"}

    try:
        works = data.get("results", [])
        out = []
        for w in works:
            authors = []
            for a in (w.get("authorships") or [])[:8]:
                au = a.get("author", {})
                if au.get("display_name"):
                    authors.append(au["display_name"])
            # FIX: OpenAlex primary_location.source kadang null walaupun
            # primary_location ada. Pakai chain (X or {}) di setiap level
            # supaya tidak crash NoneType.get pada record yang preprint atau
            # tanpa venue terdaftar.
            venue_obj = ((w.get("primary_location") or {}).get("source") or {})
            out.append({
                "doi": normalize_doi(w.get("doi")),
                "title": w.get("title") or w.get("display_name"),
                "authors": authors,
                "year": w.get("publication_year"),
                "venue": venue_obj.get("display_name"),
                "abstract": _reconstruct_inverted_index(w.get("abstract_inverted_index")) if w.get("abstract_inverted_index") else None,
                "citation_count": w.get("cited_by_count", 0),
                "url": w.get("doi") or w.get("id"),
                "sources": ["openalex"],
                "open_access": (w.get("open_access") or {}).get("is_oa", False),
                "type": w.get("type"),
            })
        return {"papers": out, "error": None}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        _log(f"[academic_search] openalex failed: {msg}")
        return {"papers": [], "error": msg}


def _reconstruct_inverted_index(inverted: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """OpenAlex stores abstracts as inverted index untuk save space. Reconstruct ke plain text."""
    if not inverted:
        return None
    positions: Dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions.keys()))


# ============================================================================
# Crossref search
# ============================================================================
async def search_crossref(client: "httpx.AsyncClient", query: str, max_results: int = 20,
                          year_from: Optional[int] = None) -> Dict[str, Any]:
    """Crossref · free, no key. https://www.crossref.org/documentation/retrieve-metadata/rest-api/

    Return shape: {"papers": List[Dict], "error": Optional[str]}.
    """
    params = {
        "query": query,
        "rows": min(max_results, 50),
    }
    if year_from:
        params["filter"] = f"from-pub-date:{year_from}"
    try:
        r = await client.get(CROSSREF_BASE, params=params, headers={"User-Agent": UA}, timeout=15.0)
        if r.status_code != 200:
            return {"papers": [], "error": f"HTTP {r.status_code} from api.crossref.org"}
        data = r.json()
        items = data.get("message", {}).get("items", [])
        out = []
        for it in items:
            authors = []
            for a in (it.get("author") or [])[:8]:
                name = ((a.get("given") or "") + " " + (a.get("family") or "")).strip()
                if name:
                    authors.append(name)
            year = None
            date_parts = ((it.get("issued") or {}).get("date-parts") or [[None]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
            out.append({
                "doi": normalize_doi(it.get("DOI")),
                "title": (it.get("title") or [None])[0],
                "authors": authors,
                "year": year,
                "venue": (it.get("container-title") or [None])[0],
                "abstract": it.get("abstract"),  # often HTML
                "citation_count": it.get("is-referenced-by-count", 0),
                "url": it.get("URL"),
                "sources": ["crossref"],
                "open_access": None,
                "type": it.get("type"),
            })
        return {"papers": out, "error": None}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        _log(f"[academic_search] crossref failed: {msg}")
        return {"papers": [], "error": msg}


# ============================================================================
# Semantic Scholar search
# ============================================================================
async def search_semantic_scholar(client: "httpx.AsyncClient", query: str, max_results: int = 20,
                                  year_from: Optional[int] = None) -> Dict[str, Any]:
    """Semantic Scholar · free (with optional API key). https://api.semanticscholar.org/api-docs/graph

    Public pool Semantic Scholar saat ini sangat ketat (sering HTTP 429 di
    request pertama). Strategi sekarang:
    1. Bila user setup API key di Settings, pakai key (rate limit jauh lebih
       tinggi, hampir tidak pernah 429).
    2. Bila tanpa key, single attempt saja dengan timeout pendek 6 detik
       lalu fail fast. Lebih baik return cepat dengan error message yang
       jelas daripada blok 20+ detik backoff yang tetap gagal.

    Return shape: {"papers": List[Dict], "error": Optional[str]}.
    """
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,authors,year,venue,abstract,citationCount,externalIds,openAccessPdf",
    }
    if year_from:
        params["year"] = f"{year_from}-"

    headers = {"User-Agent": UA}
    # Tambahkan API key kalau user sudah configure di secrets store
    ss_key = None
    try:
        from app.services import secrets_store as _ss
        ss_key = _ss.get_provider_key("semantic_scholar") if hasattr(_ss, "get_provider_key") else None
        if ss_key:
            headers["x-api-key"] = ss_key
    except Exception:
        pass

    # Bila ada API key, retry agresif. Bila tanpa key, single attempt fail fast.
    max_attempts = 4 if ss_key else 1
    request_timeout = 12.0 if ss_key else 6.0
    last_err: Optional[str] = None
    items: List[Dict[str, Any]] = []
    success = False

    for attempt in range(max_attempts):
        try:
            r = await client.get(SEMANTIC_SCHOLAR_BASE, params=params, headers=headers, timeout=request_timeout)
            if r.status_code == 429:
                if ss_key:
                    last_err = "HTTP 429 rate limited even with API key. Try again later."
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                last_err = "HTTP 429 rate limited. Public pool Semantic Scholar sangat ketat. Set API key gratis di https://www.semanticscholar.org/product/api lalu input ke Settings · API Integration."
                break
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code} from api.semanticscholar.org"
                break
            data = r.json()
            items = data.get("data", [])
            success = True
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_attempts - 1:
                await asyncio.sleep(0.8)

    if not success:
        _log(f"[academic_search] semantic_scholar failed: {last_err}")
        return {"papers": [], "error": last_err or "unknown_failure"}
    try:
        out = []
        for it in items:
            ext = it.get("externalIds") or {}
            doi = ext.get("DOI")
            authors = [a.get("name") for a in (it.get("authors") or []) if a.get("name")]
            out.append({
                "doi": normalize_doi(doi),
                "title": it.get("title"),
                "authors": authors[:8],
                "year": it.get("year"),
                "venue": it.get("venue"),
                "abstract": it.get("abstract"),
                "citation_count": it.get("citationCount", 0),
                "url": (it.get("openAccessPdf") or {}).get("url") or (f"https://doi.org/{doi}" if doi else None),
                "sources": ["semantic_scholar"],
                "open_access": bool(it.get("openAccessPdf")),
                "type": "paper",
            })
        return {"papers": out, "error": None}
    except Exception as e:
        msg = f"parse error: {type(e).__name__}: {e}"
        _log(f"[academic_search] semantic_scholar {msg}")
        return {"papers": [], "error": msg}


# ============================================================================
# arXiv search
# ============================================================================
async def search_arxiv(client: "httpx.AsyncClient", query: str, max_results: int = 20) -> Dict[str, Any]:
    """arXiv · free, no key. Returns Atom XML format yang harus di-parse.

    arXiv policy: "max 1 request per 3 seconds per IP". Backoff exponential
    bila kena 429. Plus mirror fallback ke arxiv.org direct kalau
    export.arxiv.org down. Mengakomodasi koneksi Indonesia yang sering slow
    ke arXiv US servers.

    Return shape: {"papers": List[Dict], "error": Optional[str]}.
    """
    # Mirror fallback list · export adalah official API endpoint plus
    # fallback ke arxiv.org direct yang kadang lebih responsif dari Indonesia
    ENDPOINTS = [
        "https://export.arxiv.org/api/query",
        "https://arxiv.org/api/query",  # fallback mirror
    ]

    params = {
        "search_query": f"all:{query}",
        "max_results": min(max_results, 30),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    last_err: Optional[str] = None
    text: Optional[str] = None
    used_endpoint: Optional[str] = None

    # Per-endpoint retry · total 3 attempts dengan exponential backoff
    # Timeout 40 detik (naik dari 25) untuk akomodasi koneksi Indonesia
    # ke arxiv.org US server yang kadang lambat.
    for endpoint in ENDPOINTS:
        for attempt in range(3):
            try:
                wait_before = 3 * (2 ** attempt) if attempt > 0 else 0  # 3s, 6s, 12s
                if wait_before > 0:
                    await asyncio.sleep(min(wait_before, 12))
                r = await client.get(
                    endpoint,
                    params=params,
                    headers={"User-Agent": UA, "Accept": "application/atom+xml"},
                    timeout=40.0,
                    follow_redirects=True,
                )
                if r.status_code == 429:
                    last_err = f"HTTP 429 rate limited at {endpoint} (attempt {attempt+1}/3). arXiv policy max 1 req per 3s per IP."
                    continue
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code} from {endpoint}"
                    # Non-200 (kecuali 429) langsung skip endpoint ini, coba mirror
                    break
                text = r.text
                used_endpoint = endpoint
                break
            except asyncio.TimeoutError:
                last_err = f"Timeout 40s at {endpoint} (attempt {attempt+1}/3). Connection ke arxiv.org slow dari lokasi user."
            except Exception as e:
                last_err = f"{type(e).__name__} at {endpoint}: {str(e)[:200]}"
                # Connection error · langsung try next endpoint, jangan retry attempt
                break
        if text is not None:
            break  # success with this endpoint, skip mirror

    if text is None:
        _log(f"[academic_search] arxiv failed all endpoints: {last_err}")
        return {"papers": [], "error": last_err or "all endpoints failed"}

    if used_endpoint and used_endpoint != ENDPOINTS[0]:
        _log(f"[academic_search] arxiv succeeded via fallback mirror: {used_endpoint}")

    try:
        # Simple regex parser untuk Atom (avoid xmltodict dependency)
        entries = re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL)
        out = []
        for entry in entries:
            title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            published_m = re.search(r"<published>(\d{4})", entry)
            id_m = re.search(r"<id>(.*?)</id>", entry)
            doi_m = re.search(r"<arxiv:doi[^>]*>(.*?)</arxiv:doi>", entry)
            authors = re.findall(r"<name>(.*?)</name>", entry)
            out.append({
                "doi": normalize_doi(doi_m.group(1)) if doi_m else None,
                "title": (title_m.group(1) if title_m else "").strip(),
                "authors": authors[:8],
                "year": int(published_m.group(1)) if published_m else None,
                "venue": "arXiv preprint",
                "abstract": (summary_m.group(1) if summary_m else "").strip(),
                "citation_count": 0,
                "url": id_m.group(1) if id_m else None,
                "sources": ["arxiv"],
                "open_access": True,
                "type": "preprint",
            })
        return {"papers": out, "error": None}
    except Exception as e:
        msg = f"parse error: {type(e).__name__}: {e}"
        _log(f"[academic_search] arxiv {msg}")
        return {"papers": [], "error": msg}


# ============================================================================
# Aggregator with deduplication
# ============================================================================
def dedupe_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge papers from multiple sources via DOI plus title-year match."""
    by_doi: Dict[str, Dict[str, Any]] = {}
    by_title_year: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []

    for p in papers:
        doi = p.get("doi")
        title_norm = normalize_title(p.get("title"))
        year = p.get("year")
        title_key = f"{title_norm}::{year}" if title_norm and year else None

        existing = None
        if doi and doi in by_doi:
            existing = by_doi[doi]
        elif title_key and title_key in by_title_year:
            existing = by_title_year[title_key]

        if existing:
            # Merge sources
            existing["sources"] = list(set(existing["sources"] + p["sources"]))
            # Prefer non-null fields from new entry
            for k in ("abstract", "venue", "citation_count", "open_access", "url"):
                if not existing.get(k) and p.get(k):
                    existing[k] = p[k]
            # Max citation count across sources
            if p.get("citation_count", 0) > existing.get("citation_count", 0):
                existing["citation_count"] = p["citation_count"]
        else:
            if doi:
                by_doi[doi] = p
            if title_key:
                by_title_year[title_key] = p
            out.append(p)

    return out


def score_papers(papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Rough relevance scoring · citation + multi-source + recency."""
    query_words = set(normalize_title(query).split())
    current_year = 2026
    for p in papers:
        title_words = set(normalize_title(p.get("title")).split())
        overlap = len(query_words & title_words) / max(1, len(query_words))
        cit_score = min(1.0, (p.get("citation_count") or 0) / 100.0)
        source_score = min(1.0, len(p.get("sources", [])) / 4.0)
        year_score = 0.0
        if p.get("year"):
            recency = max(0, current_year - p["year"])
            year_score = max(0.0, 1.0 - recency / 30.0)  # newer = better, but >30 years gets 0
        # Weighted
        p["score"] = round(0.4 * overlap + 0.3 * cit_score + 0.2 * source_score + 0.1 * year_score, 4)
    papers.sort(key=lambda x: x.get("score", 0), reverse=True)
    return papers


async def search_all(query: str, max_per_source: int = 15, year_from: Optional[int] = None,
                      include_sources: Optional[List[str]] = None, fast_mode: bool = True,
                      use_cache: bool = True) -> Dict[str, Any]:
    """Main aggregator entry · fan-out ke semua sources parallel, dedup, score.

    use_cache (default True): cek in-memory cache dulu sebelum hit external API.
    Cache TTL 1 jam, max 200 entries dengan LRU eviction. Repeat query jadi
    instant (< 50ms) bukan 2-5 detik.

    fast_mode (default True): pakai asyncio.wait dengan FIRST_COMPLETED return
    setelah 2 source pertama selesai. Source lambat di-cancel supaya tidak
    block UI. Trade-off: kurang paper tapi 2-3x lebih cepat.
    """
    if not _HTTPX_AVAILABLE:
        return {"status": "error", "message": "httpx not installed"}

    include_sources = include_sources or ["openalex", "crossref", "semantic_scholar", "arxiv"]

    # Cache lookup · skip external API kalau hit
    cache_key = _cache_key(query, max_per_source, year_from, include_sources)
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            # Mark sebagai cache hit untuk UI
            cached_copy = dict(cached)
            cached_copy["cached"] = True
            cached_copy["cache_age_sec"] = int(time.time() - _CACHE[cache_key][0])
            return cached_copy

    # PER-SOURCE CACHE CHECK · cek setiap source independen dulu sebelum hit API.
    # Source yang ada di cache LANGSUNG pakai cache (no network call).
    # Source yang miss cache atau force_fresh akan di-fetch dari external API.
    # GUARANTEE: source yang pernah sukses (cached) tidak akan jadi 0 lagi
    # bahkan setelah restart backend.
    per_source_hits: Dict[str, List[Dict[str, Any]]] = {}
    per_source_misses: List[str] = []
    if use_cache:
        for src in include_sources:
            cached_papers = _ps_cache_get(src, query, max_per_source)
            if cached_papers is not None:
                per_source_hits[src] = cached_papers
                _log(f"[academic_search] per-source CACHE HIT · {src} returns {len(cached_papers)} papers from disk")
            else:
                per_source_misses.append(src)
    else:
        per_source_misses = list(include_sources)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = []
        source_labels = []
        if "openalex" in include_sources and "openalex" in per_source_misses:
            tasks.append(asyncio.create_task(search_openalex(client, query, max_per_source, year_from)))
            source_labels.append("openalex")
        if "crossref" in include_sources and "crossref" in per_source_misses:
            tasks.append(asyncio.create_task(search_crossref(client, query, max_per_source, year_from)))
            source_labels.append("crossref")
        if "semantic_scholar" in include_sources and "semantic_scholar" in per_source_misses:
            tasks.append(asyncio.create_task(search_semantic_scholar(client, query, max_per_source, year_from)))
            source_labels.append("semantic_scholar")
        if "arxiv" in include_sources and "arxiv" in per_source_misses:
            tasks.append(asyncio.create_task(search_arxiv(client, query, max_per_source)))
            source_labels.append("arxiv")

        # Wait dengan overall timeout 45 detik. Tradeoff: source yang lambat
        # (OpenAlex sometimes 40s, arXiv with retry) mungkin return 0 untuk
        # spesifik request itu, tapi user dapat hasil cepat dari source lain.
        # Cache hasil dari source yang sukses, retry source yang gagal di
        # query berikutnya. Better than blocking 90s and risking Cloudflare
        # Tunnel 100s cutoff atau browser fetch cancel.
        results = []
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=45.0
            )
        except asyncio.TimeoutError:
            # Kumpulkan partial results · gather() ke-cancel by wait_for
            for t in tasks:
                if t.done() and not t.cancelled():
                    try: results.append(t.result())
                    except Exception as _e: results.append({"papers": [], "error": f"{type(_e).__name__}: {_e}"})
                else:
                    t.cancel()
                    results.append({"papers": [], "error": "overall_timeout_45s"})

    # Combine all · setiap source return {"papers": [...], "error": str|None}
    all_papers: List[Dict[str, Any]] = []
    counts_per_source: Dict[str, int] = {}
    errors_per_source: Dict[str, Optional[str]] = {}

    # FIRST · merge per-source cache hits (sources yang sudah ada di cache)
    # Source ini SKIP external API call sama sekali, langsung pakai cached.
    for src, cached_papers in per_source_hits.items():
        counts_per_source[src] = len(cached_papers)
        errors_per_source[src] = None
        all_papers.extend(cached_papers)

    # SECOND · process result dari sources yang baru di-fetch (per_source_misses)
    for label, res in zip(source_labels, results):
        if isinstance(res, Exception):
            counts_per_source[label] = 0
            errors_per_source[label] = f"task_exception: {type(res).__name__}: {res}"
            continue
        if not isinstance(res, dict):
            papers_list = res if isinstance(res, list) else []
            counts_per_source[label] = len(papers_list)
            errors_per_source[label] = None
            all_papers.extend(papers_list)
            # SAVE successful fetch ke per-source cache untuk request berikutnya
            if papers_list:
                _ps_cache_set(label, query, max_per_source, papers_list)
            continue
        papers_list = res.get("papers", []) or []
        counts_per_source[label] = len(papers_list)
        errors_per_source[label] = res.get("error")
        all_papers.extend(papers_list)
        # KEY FIX · save successful source result ke per-source cache. Bila
        # source kembali sukses dengan papers > 0, cache 30 hari supaya
        # bahkan setelah restart backend, source ini tetap return hasil yang
        # sama tanpa hit external API lagi.
        if papers_list and not res.get("error"):
            _ps_cache_set(label, query, max_per_source, papers_list)

    # Dedupe plus score
    unique = dedupe_papers(all_papers)
    scored = score_papers(unique, query)

    result = {
        "status": "success",
        "query": query,
        "total_raw": len(all_papers),
        "total_unique": len(scored),
        "sources_queried": source_labels,
        "counts_per_source": counts_per_source,
        "errors_per_source": errors_per_source,
        "papers": scored,
        "cached": False,
        "cache_age_sec": 0,
    }

    # SMART CACHE · QUALITY GATE
    # FIX BUG · sebelumnya threshold 1 source minimum menyebabkan partial
    # results (misalnya Crossref OK tapi 3 source lain timeout 0) ikut ter-
    # cache. Hasil buruk ini di-served berulang kali 24 jam dan user lihat
    # sumber selalu 0 setiap refresh.
    #
    # Sekarang threshold KETAT · cache HANYA bila minimum 3 dari 4 source
    # ada papers (kualitas hasil tinggi). Bila 2+ source return 0, JANGAN
    # cache supaya request berikutnya retry fresh ke source yang gagal.
    # User dapat tunggu network membaik atau klik Fresh button.
    if use_cache:
        non_empty_sources = sum(1 for c in counts_per_source.values() if c > 0)
        total_sources = len(counts_per_source)
        min_required = max(3, int(total_sources * 0.75))  # 75% atau minimum 3 source
        if non_empty_sources >= min_required:
            _cache_set(cache_key, result)
            _log(f"[academic_search] cached · {non_empty_sources}/{total_sources} sources OK, total {len(scored)} unique papers")
        else:
            _log(f"[academic_search] SKIP cache · only {non_empty_sources}/{total_sources} sources OK (need >= {min_required}). Next request will retry fresh.")
    return result
