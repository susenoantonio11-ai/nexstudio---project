"""
NEXLYTICS Backend - FastAPI Application Entry Point
====================================================
Run: uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import sys, pathlib
# Add project root (parent of backend/) so ml_engine can be imported.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from contextlib import asynccontextmanager
from pathlib import Path

from app.core.config import settings
from app.database.session import init_db
from app.api import auth_router, dataset_router, experiment_router, realtime_router, geo_router
from app.api.disaster_router import router as disaster_router
from app.api.quality_router import router as quality_router
from app.api.registry_router import router as registry_router
from app.api.complex_router import router as complex_router
from app.api.research_router import router as research_router
from app.api.advanced_router import router as advanced_router

# Settings router optional · butuh cryptography plus httpx library.
# Kalau library belum terinstall, backend tetap jalan tanpa fitur multi-provider.
try:
    from app.api.settings_router import router as settings_router
    _SETTINGS_AVAILABLE = True
except Exception as _e:
    settings_router = None
    _SETTINGS_AVAILABLE = False
    print(f"[WARN] settings_router not loaded: {_e}. Install: pip install cryptography==43.0.3 httpx==0.27.2")
from app.database import geo_models  # ensure tables registered


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Nexlytics Realtime Intelligence Engine. "
        "Multi-model AI analytics platform with Method Monitor (Explainable AI). "
        "Built following CRISP-DM methodology."
    ),
    lifespan=lifespan,
)

# CORS for frontend — production-aware
import os, re
_extra = os.environ.get("CORS_EXTRA_ORIGINS", "").split(",")
_extra = [o.strip() for o in _extra if o.strip()]
_origins = list(settings.CORS_ORIGINS) + _extra
# Allow Cloud Run wildcard *.run.app via regex when deployed
_origin_regex = os.environ.get("CORS_ORIGIN_REGEX", r"https://.*\.run\.app$")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression · reduces JS/CSS transfer by 70-80%, huge speedup for frontend
# minimum_size=500 means small responses (under 500 bytes) skip compression
app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)


# Cache headers middleware · cache static files for 1 hour in browser
@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    # Cache static frontend assets aggressively (CSS, JS, fonts, images)
    if path.startswith("/nxlytics/") or path.startswith("/static/"):
        if path.endswith((".js", ".css", ".woff", ".woff2", ".ttf", ".png", ".jpg", ".jpeg", ".svg", ".ico")):
            response.headers["Cache-Control"] = "public, max-age=3600, immutable"
        elif path.endswith(".html"):
            # HTML always re-fetch but allow conditional
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

# Routers
app.include_router(auth_router.router)
app.include_router(dataset_router.router)
app.include_router(experiment_router.router)
app.include_router(realtime_router.router)
app.include_router(geo_router.router)
app.include_router(disaster_router)
app.include_router(quality_router)
app.include_router(registry_router)
app.include_router(complex_router)
app.include_router(research_router)
app.include_router(advanced_router)
if _SETTINGS_AVAILABLE and settings_router is not None:
    app.include_router(settings_router)


@app.get("/")
def root():
    """Root path · redirect ke frontend UI biar user langsung dapat halaman NXLYTICS.
    JSON API metadata sekarang di /api-info."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/nxlytics/index.html", status_code=302)


@app.get("/api-info")
def api_info():
    """API metadata endpoint · sebelumnya di / sebelum diganti redirect."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
        "tagline": "See Everything. Decide Faster.",
        "components": {
            "auth": "/api/auth",
            "datasets": "/api/datasets",
            "experiments": "/api/experiments",
            "realtime_websocket": "/api/realtime/ws",
            "geospatial": "/api/geo",
            "research": "/api/research",
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# Serve static frontend · single-server mode untuk MacBook hemat resource
# Mount nxlytics folder di /nxlytics, project root di /static untuk legacy assets.
# Akses: http://localhost:8000/nxlytics/index.html#re-workspace
project_root = Path(__file__).resolve().parents[2]
nxlytics_dir = project_root / "nxlytics"
if nxlytics_dir.exists():
    app.mount(
        "/nxlytics",
        StaticFiles(directory=str(nxlytics_dir), html=True),
        name="nxlytics",
    )
if (project_root / "index.html").exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(project_root)),
        name="static",
    )
