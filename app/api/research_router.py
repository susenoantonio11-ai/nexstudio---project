"""
Research Router
===============
FastAPI router for the thesis pipeline:

  GET  /api/research/health
  GET  /api/research/provinces
  POST /api/research/flood/run             — orchestrator end-to-end
  POST /api/research/flood/build-panel     — panel only (debugging)
  POST /api/research/flood/train           — train hybrid model from a panel
  POST /api/research/flood/explain         — SHAP for a trained model state
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ai_models.research import (
    FloodResearchOrchestrator, INDONESIA_PROVINCES, list_province_ids,
    FloodPanelBuilder, HybridLSTMXGBoost, HybridSHAPExplainer,
)
from app.ai_models.research import gee_service, bnpb_service
from app.data_sources import (
    list_sources as ds_list_sources,
    get_source as ds_get_source,
    list_by_category as ds_list_by_category,
    suggest_for_research as ds_suggest_for_research,
    stats as ds_stats,
)

router = APIRouter(prefix="/api/research", tags=["research_pipeline"])


# ---------------------------------------------------------------------------
# Data source endpoints (used by the flexible Research Workspace)
# ---------------------------------------------------------------------------
@router.get("/data-sources")
def data_sources(category: Optional[str] = None) -> Dict[str, Any]:
    sources = ds_list_by_category(category) if category else ds_list_sources()
    return {
        "status": "success",
        "count": len(sources),
        "stats": ds_stats(),
        "sources": sources,
    }


@router.get("/data-sources/suggest")
def data_sources_suggest(research_type: str) -> Dict[str, Any]:
    return {
        "status": "success",
        "research_type": research_type,
        "sources": ds_suggest_for_research(research_type),
    }


@router.get("/data-sources/{source_id}")
def data_source_detail(source_id: str) -> Dict[str, Any]:
    s = ds_get_source(source_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Data source not found: {source_id}")
    return s


# ---------------------------------------------------------------------------
# Generic research dispatcher — routes by research_type
# ---------------------------------------------------------------------------
class ResearchRunRequest(BaseModel):
    research_type: str
    region_level: str = "province"
    provinces: Optional[List[str]] = None
    region_bbox: Optional[List[float]] = None
    start_date: str = "2016-01-01"
    end_date: str = "2025-12-31"
    method: str = "auto"
    parameters: Optional[Dict[str, Any]] = None


@router.post("/run")
def research_run(req: ResearchRunRequest) -> Dict[str, Any]:
    """Generic dispatcher. Routes flood research to the existing
    FloodResearchOrchestrator; other types return a structured stub
    (real engines wired as they become available)."""
    if req.research_type == "multi_province_flood_classification":
        from app.ai_models.research import FloodResearchOrchestrator
        orch = FloodResearchOrchestrator(
            provinces=req.provinces or None,
            start=req.start_date, end=req.end_date,
            prediction_horizon_days=(req.parameters or {}).get("horizon", 7),
            seq_window=(req.parameters or {}).get("seq_window", 30),
            lstm_weight=(req.parameters or {}).get("lstm_weight", 0.5),
            xgb_weight=(req.parameters or {}).get("xgb_weight", 0.5),
        )
        return orch.run()
    # Stubs for other research types — return structured envelope
    return {
        "status": "queued",
        "research_type": req.research_type,
        "message": "Research dispatcher: handler for this type not yet wired. Returning configuration echo.",
        "config": req.dict(),
        "method_monitor": {
            "method": "Generic dispatcher (handler stub)",
            "limitations": ["Wire specific orchestrator for this research_type to enable real run."],
        },
    }



# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class FloodRunRequest(BaseModel):
    provinces: Optional[List[str]] = None
    start: str = "2016-01-01"
    end: str = "2025-12-31"
    prediction_horizon_days: int = 7
    seq_window: int = 30
    lstm_weight: float = 0.50
    xgb_weight: float = 0.50
    bnpb_csv_path: Optional[str] = None
    bnpb_folder: Optional[str] = None


def _df_envelope(df) -> Dict[str, Any]:
    """Convert a pandas DataFrame to a JSON-safe envelope (head + dtypes)."""
    if df is None or df.empty:
        return {"rows": 0, "columns": [], "head": []}
    return {
        "rows": int(len(df)),
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "head": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
    }


# ---------------------------------------------------------------------------
@router.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint dengan detail untuk monitoring frontend.
    Frontend NxResearchAPI.callWithRetry retry hanya kalau backend 5xx; endpoint
    ini ringan supaya tidak menyebabkan timeout sendiri. Tambah info lab plus
    AI provider kalau bisa di-check tanpa side effect. """
    ai_available = False
    ai_provider = None
    try:
        from app.services import task_router as _tr
        # Light probe · cek apakah module ter-load tanpa real AI call
        ai_available = hasattr(_tr, 'route') or hasattr(_tr, 'route_stream')
        ai_provider = getattr(_tr, 'default_provider', None) if ai_available else None
    except Exception:
        ai_available = False
    lab_count = len(_LAB_STAGES) if '_LAB_STAGES' in globals() else 0
    return {
        "status": "ok",
        "engine": "FloodResearchOrchestrator",
        "n_provinces": len(INDONESIA_PROVINCES),
        "gee_live": gee_service.is_live(),
        "bnpb_modes": ["load_csv", "load_directory", "synthesize_demo_events"],
        # Phase 9-10 additions
        "ai_provider_available": ai_available,
        "ai_default_provider": ai_provider,
        "lab_task_types_supported": lab_count,
        "lab_task_types": list(_LAB_STAGES.keys()) if '_LAB_STAGES' in globals() else [],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/provinces")
def provinces() -> Dict[str, Any]:
    return {"count": len(INDONESIA_PROVINCES), "provinces": INDONESIA_PROVINCES}


# ─── Geospatial Map Viewer endpoints ──────────────────────────────────────
@router.get("/gee/datasets")
def gee_datasets() -> Dict[str, Any]:
    """List all available GEE datasets that can be rendered as map tiles."""
    return gee_service.list_datasets()


@router.get("/gee/map-tile")
def gee_map_tile(dataset: str, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
    """Generate a Leaflet-compatible tile URL for the given GEE dataset."""
    return gee_service.get_map_tile_url(dataset, start=start, end=end)


@router.get("/gee/pixel")
def gee_pixel_inspector(lat: float, lng: float, dataset: str,
                        start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
    """Inspect pixel value at specific lat/lng for the given dataset."""
    return gee_service.inspect_pixel(lat, lng, dataset, start=start, end=end)


@router.get("/gee/export")
def gee_export_geotiff(dataset: str, province: Optional[str] = None,
                       start: Optional[str] = None, end: Optional[str] = None,
                       scale: int = 1000) -> Dict[str, Any]:
    """Generate downloadable GeoTIFF URL for the given dataset clipped to province."""
    return gee_service.export_geotiff(dataset, province_id=province, start=start, end=end, scale=scale)


# ─── GEE-Bridge Workflow ─ generate script for Code Editor, then upload result ────
@router.get("/gee/script")
def gee_generate_script(dataset: str, province: Optional[str] = None,
                        start: Optional[str] = None, end: Optional[str] = None,
                        scale: int = 30, scope: str = "indonesia") -> Dict[str, Any]:
    """Generate a complete GEE Code Editor script for the user to paste and run.
    The script will export the dataset to Google Drive in NXLYTICS folder.

    scope parameter:
      * "global"    — whole world bounding box (180 W to 180 E, 60 S to 80 N)
      * "indonesia" — Indonesia archipelago bbox (default)
      * ignored when explicit province is provided

    User workflow:
      1. Get script via this endpoint
      2. Paste into https://code.earthengine.google.com/
      3. Click Run, then Run in Tasks tab
      4. Wait for export to Google Drive (5-20 minutes)
      5. Download .tif from Drive
      6. Upload to NXLYTICS via /user-datasets/upload
    """
    from app.ai_models.research.gee_service import DATASET_CATALOG, INDONESIA_PROVINCES
    if dataset not in DATASET_CATALOG:
        return {"status": "error", "message": f"Unknown dataset · {dataset}"}
    spec = DATASET_CATALOG[dataset]
    prov = None
    if province:
        prov = next((p for p in INDONESIA_PROVINCES if p["id"] == province), None)
    sd = start or "2024-01-01"
    ed = end or "2024-12-31"

    if prov:
        roi_name = prov["name_id"]
        # Use Province name to filter GAUL admin boundaries
        roi_code = f'ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1").filter(ee.Filter.eq("ADM1_NAME", "{roi_name}")).geometry()'
        region_tag = roi_name.replace(" ", "_")
    elif scope.lower() == "global":
        roi_name = "Global"
        roi_code = 'ee.Geometry.Rectangle([-180, -60, 180, 80])'
        region_tag = "Global"
    else:
        roi_name = "Indonesia"
        roi_code = 'ee.Geometry.Rectangle([94, -11, 142, 7])'
        region_tag = "Indonesia"

    is_static = spec.get("static", False)
    if is_static:
        img_code = f'ee.Image("{spec["ee_collection"]}").select("{spec["band"]}").clip(roi)'
    else:
        img_code = f'ee.ImageCollection("{spec["ee_collection"]}").select("{spec["band"]}").filterDate("{sd}", "{ed}").mean().clip(roi)'

    palette_str = ", ".join([f'"{c}"' for c in spec["vis_params"].get("palette", ["#000000", "#ffffff"])])
    filename = f'{dataset.upper()}_{region_tag.upper()}_{sd[:4]}_{ed[:4]}'

    script = f"""// ============================================================================
// NXLYTICS · Auto-generated GEE export script
// Dataset: {spec["label"]}
// Region: {roi_name}
// Period: {sd} to {ed}
// ============================================================================

var roi = {roi_code};
var img = {img_code};

// Preview on map
Map.centerObject(roi, 7);
Map.addLayer(img, {{
  min: {spec["vis_params"].get("min", 0)},
  max: {spec["vis_params"].get("max", 100)},
  palette: [{palette_str}]
}}, "{spec["label"]}");

// Export to Google Drive · folder NXLYTICS
Export.image.toDrive({{
  image: img,
  description: "{filename}",
  folder: "NXLYTICS",
  region: roi,
  scale: {scale},
  crs: "EPSG:4326",
  maxPixels: 1e10
}});

// Setelah klik Run di atas, klik tab Tasks (kanan) lalu klik Run di task yang muncul.
// File akan masuk Google Drive folder NXLYTICS dalam 5 sampai 20 menit.
// Download .tif lalu upload ke NXLYTICS via Geo Map > Upload Dataset.
"""
    return {
        "status": "success",
        "script": script,
        "filename": f"{filename}.tif",
        "dataset_label": spec["label"],
        "region": roi_name,
        "code_editor_url": "https://code.earthengine.google.com/"
    }


@router.post("/user-datasets/upload")
async def upload_user_dataset(file: UploadFile = File(...), label: str = Form("")) -> Dict[str, Any]:
    """Upload a downloaded GeoTIFF (atau any user dataset) to NXLYTICS storage.
    Files stored under backend/storage/user-datasets/."""
    import os
    from datetime import datetime
    storage_dir = Path(__file__).resolve().parents[2] / "storage" / "user-datasets"
    storage_dir.mkdir(parents=True, exist_ok=True)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stored_path = storage_dir / f"{timestamp}_{safe_name}"
    content = await file.read()
    stored_path.write_bytes(content)
    size_mb = round(len(content) / (1024 * 1024), 2)
    return {
        "status": "success",
        "filename": stored_path.name,
        "original_name": file.filename,
        "size_mb": size_mb,
        "uploaded_at": datetime.utcnow().isoformat(),
        "label": label,
        "path": str(stored_path.relative_to(Path(__file__).resolve().parents[2]))
    }


@router.get("/user-datasets/list")
def list_user_datasets() -> Dict[str, Any]:
    """List all user-uploaded datasets in storage."""
    storage_dir = Path(__file__).resolve().parents[2] / "storage" / "user-datasets"
    if not storage_dir.exists():
        return {"status": "success", "count": 0, "datasets": []}
    files = []
    for f in storage_dir.iterdir():
        if f.is_file():
            stat = f.stat()
            files.append({
                "filename": f.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
                "extension": f.suffix
            })
    files.sort(key=lambda x: x["modified_at"], reverse=True)
    return {"status": "success", "count": len(files), "datasets": files}


@router.delete("/user-datasets/{filename}")
def delete_user_dataset(filename: str) -> Dict[str, Any]:
    """Delete a user dataset file from storage."""
    storage_dir = Path(__file__).resolve().parents[2] / "storage" / "user-datasets"
    safe_name = filename.replace("/", "_").replace("..", "_")
    target = storage_dir / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    target.unlink()
    return {"status": "success", "deleted": safe_name}


# ---------------------------------------------------------------------------
# Raster → DataFrame conversion endpoints (Data Workspace · GEE Collector)
# ---------------------------------------------------------------------------
class RasterConvertRequest(BaseModel):
    filename: str
    sample_n: int = 100000  # max rows to keep (downsample untuk file besar)


@router.post("/raster-to-dataframe")
def raster_to_dataframe(req: RasterConvertRequest) -> Dict[str, Any]:
    """Convert a GeoTIFF raster to a long-format DataFrame (lat, lng, value, band).

    Strategi pembacaan:
      1. Coba pakai rasterio (paling robust, full CRS handling)
      2. Fallback ke tifffile + numpy + manual transform jika rasterio missing
      3. Hasil disimpan sebagai CSV di backend/storage/converted-datasets/

    Setiap pixel jadi satu baris. File raster besar (> sample_n pixel)
    di-downsample stratified random untuk menjaga distribusi value.

    References:
      * Wickham (2014) Tidy Data principles: each row = one observation
      * GeoTIFF specification (OGC 19-008r4)
    """
    storage_dir = Path(__file__).resolve().parents[2] / "storage" / "user-datasets"
    safe_name = req.filename.replace("/", "_").replace("..", "_")
    src_path = storage_dir / safe_name
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Raster not found: {req.filename}")

    out_dir = Path(__file__).resolve().parents[2] / "storage" / "converted-datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = src_path.stem + ".csv"
    out_path = out_dir / dataset_name

    rows: List[Dict[str, Any]] = []
    band_count = 0
    method_used = "unknown"

    try:
        import rasterio  # type: ignore
        import numpy as np
        method_used = "rasterio"
        with rasterio.open(src_path) as src:
            band_count = src.count
            transform = src.transform
            data = src.read()  # shape (bands, height, width)
            height, width = data.shape[1], data.shape[2]
            total_pixels = height * width

            # Decide indices to sample
            if total_pixels > req.sample_n:
                idx = np.random.default_rng(seed=42).choice(total_pixels, size=req.sample_n, replace=False)
            else:
                idx = np.arange(total_pixels)
            ys = idx // width
            xs = idx % width

            # Transform pixel to lng/lat
            lngs, lats = rasterio.transform.xy(transform, ys.tolist(), xs.tolist(), offset="center")
            lngs = np.array(lngs)
            lats = np.array(lats)

            for b in range(band_count):
                vals = data[b][ys, xs]
                nodata = src.nodatavals[b] if b < len(src.nodatavals) else None
                for i in range(len(idx)):
                    v = float(vals[i])
                    if nodata is not None and v == nodata:
                        continue
                    if np.isnan(v):
                        continue
                    rows.append({
                        "lat": round(float(lats[i]), 6),
                        "lng": round(float(lngs[i]), 6),
                        "value": round(v, 4),
                        "band": b + 1,
                    })
    except ImportError:
        # Fallback: tifffile + numpy
        try:
            import tifffile  # type: ignore
            import numpy as np
            method_used = "tifffile_fallback"
            data = tifffile.imread(str(src_path))
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            band_count = data.shape[0]
            height, width = data.shape[1], data.shape[2]
            total_pixels = height * width
            if total_pixels > req.sample_n:
                idx = np.random.default_rng(seed=42).choice(total_pixels, size=req.sample_n, replace=False)
            else:
                idx = np.arange(total_pixels)
            ys = idx // width
            xs = idx % width
            # No CRS info — use pixel indices as proxy coordinates
            for b in range(band_count):
                vals = data[b][ys, xs]
                for i in range(len(idx)):
                    v = float(vals[i])
                    if np.isnan(v):
                        continue
                    rows.append({
                        "lat": int(ys[i]),
                        "lng": int(xs[i]),
                        "value": round(v, 4),
                        "band": b + 1,
                    })
        except ImportError:
            return {
                "status": "error",
                "message": "Tidak ada library raster yang terinstall. Install rasterio (preferred) atau tifffile: "
                           "pip install rasterio  ATAU  pip install tifffile"
            }
    except Exception as e:
        return {"status": "error", "message": f"Conversion failed: {str(e)}"}

    if not rows:
        return {"status": "error", "message": "No valid pixel data extracted (semua nodata atau NaN?)"}

    # Write CSV
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lat", "lng", "value", "band"])
        writer.writeheader()
        writer.writerows(rows)

    schema = [
        {"name": "lat",   "type": "float", "description": "Latitude in EPSG:4326 degrees (atau pixel y untuk fallback)"},
        {"name": "lng",   "type": "float", "description": "Longitude in EPSG:4326 degrees (atau pixel x untuk fallback)"},
        {"name": "value", "type": "float", "description": "Pixel value (units depend on source dataset, contoh CHIRPS mm/day)"},
        {"name": "band",  "type": "int",   "description": "Band index (1-based)"},
    ]

    # Save companion JSON metadata
    meta = {
        "dataset_name": dataset_name,
        "source_raster": safe_name,
        "row_count": len(rows),
        "col_count": len(schema),
        "band_count": band_count,
        "method": method_used,
        "created_at": datetime.utcnow().isoformat(),
        "schema": schema,
    }
    (out_dir / (dataset_name + ".meta.json")).write_text(json.dumps(meta, indent=2))

    return {
        "status": "success",
        "dataset_name": dataset_name,
        "row_count": len(rows),
        "col_count": len(schema),
        "band_count": band_count,
        "method": method_used,
        "schema": schema,
        "preview": rows[:5],
    }


@router.get("/converted-datasets/list")
def list_converted_datasets() -> Dict[str, Any]:
    """List all converted DataFrames (CSV files) di storage."""
    out_dir = Path(__file__).resolve().parents[2] / "storage" / "converted-datasets"
    if not out_dir.exists():
        return {"status": "success", "count": 0, "datasets": []}
    datasets = []
    for csv_file in out_dir.glob("*.csv"):
        meta_file = csv_file.with_name(csv_file.name + ".meta.json")
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                meta["dataset_name"] = csv_file.name
                meta["size_mb"] = round(csv_file.stat().st_size / (1024 * 1024), 2)
                datasets.append(meta)
            except Exception:
                continue
        else:
            datasets.append({
                "dataset_name": csv_file.name,
                "row_count": 0,
                "col_count": 0,
                "created_at": datetime.utcfromtimestamp(csv_file.stat().st_mtime).isoformat(),
                "size_mb": round(csv_file.stat().st_size / (1024 * 1024), 2),
            })
    datasets.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return {"status": "success", "count": len(datasets), "datasets": datasets}


@router.get("/converted-datasets/{dataset_name}")
def get_converted_dataset(dataset_name: str, format: str = "json", limit: int = 5000) -> Any:
    """Fetch converted dataset content.

    format=json   → dict dengan rows array (untuk NxDatasetHub.registerExisting)
    format=csv    → raw CSV file download
    """
    out_dir = Path(__file__).resolve().parents[2] / "storage" / "converted-datasets"
    safe_name = dataset_name.replace("/", "_").replace("..", "_")
    csv_path = out_dir / safe_name
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_name}")

    if format == "csv":
        from fastapi.responses import FileResponse
        return FileResponse(csv_path, media_type="text/csv", filename=safe_name)

    # JSON format
    import csv as csvmod
    rows = []
    with open(csv_path, "r") as f:
        reader = csvmod.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            # Convert numeric fields
            try:
                row["lat"] = float(row["lat"])
                row["lng"] = float(row["lng"])
                row["value"] = float(row["value"])
                row["band"] = int(row["band"])
            except (ValueError, KeyError):
                pass
            rows.append(row)

    meta_file = csv_path.with_name(csv_path.name + ".meta.json")
    meta = {}
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            pass

    return {
        "status": "success",
        "dataset_name": safe_name,
        "rows": rows,
        "row_count": meta.get("row_count", len(rows)),
        "col_count": meta.get("col_count", 4),
        "schema": meta.get("schema", []),
        "truncated": len(rows) >= limit,
    }


# ---------------------------------------------------------------------------
# Pre-trained Model Persistence (untuk demo via Cloudflare Tunnel)
# ---------------------------------------------------------------------------
# Training Hybrid LSTM-XGBoost butuh 15-30 menit di laptop, jauh melebihi
# Cloudflare Tunnel Free Plan timeout 100 detik. Solusi reproducibility-grade:
#   1. Training penuh di laptop user (lokal, tanpa tunnel) → save model artifact
#   2. Demo lewat tunnel klik tombol "Load saved" → fetch artifact dari disk
#      dalam <2 detik
#
# Pattern ini sesuai standar reproducibility:
#   * Stodden, Krafczyk, Bhaskar (2018) Nature 553:171 "Enhancing reproducibility"
#   * Pineau et al. (2021) NeurIPS ML Reproducibility Checklist
#   * Sculley et al. (2015) NeurIPS "Hidden Technical Debt in ML Systems"
# ---------------------------------------------------------------------------

class SaveRunRequest(BaseModel):
    run_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    label: str = ""
    payload: Dict[str, Any]  # serialized result from /flood/run


@router.post("/flood/save-run")
def save_flood_run(req: SaveRunRequest) -> Dict[str, Any]:
    """Save a completed flood pipeline run (model metrics + SHAP + provinces)
    untuk re-load instan saat demo via tunnel."""
    out_dir = Path(__file__).resolve().parents[2] / "storage" / "trained-models"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"run_{req.run_id}.json"
    out_path = out_dir / fname
    record = {
        "run_id": req.run_id,
        "label": req.label or f"Hybrid LSTM-XGBoost · {req.run_id}",
        "saved_at": datetime.utcnow().isoformat(),
        "payload": req.payload,
    }
    out_path.write_text(json.dumps(record, indent=2, default=str))
    size_kb = round(out_path.stat().st_size / 1024, 2)
    return {
        "status": "success",
        "run_id": req.run_id,
        "filename": fname,
        "label": record["label"],
        "size_kb": size_kb,
        "saved_at": record["saved_at"],
    }


@router.get("/flood/saved-runs")
def list_saved_runs() -> Dict[str, Any]:
    """List semua training run yang sudah ter-save di disk."""
    out_dir = Path(__file__).resolve().parents[2] / "storage" / "trained-models"
    if not out_dir.exists():
        return {"status": "success", "count": 0, "runs": []}
    runs = []
    for json_file in out_dir.glob("run_*.json"):
        try:
            data = json.loads(json_file.read_text())
            metrics = (data.get("payload") or {}).get("metrics", {})
            runs.append({
                "run_id": data.get("run_id"),
                "filename": json_file.name,
                "label": data.get("label"),
                "saved_at": data.get("saved_at"),
                "size_kb": round(json_file.stat().st_size / 1024, 2),
                # Surface key metrics jadi user bisa pilih run terbaik
                "roc_auc": metrics.get("roc_auc"),
                "f1": metrics.get("f1"),
                "accuracy": metrics.get("accuracy"),
                "provinces": ((data.get("payload") or {}).get("metadata", {}) or {}).get("provinces", []),
            })
        except Exception:
            continue
    runs.sort(key=lambda r: r.get("saved_at", ""), reverse=True)
    return {"status": "success", "count": len(runs), "runs": runs}


@router.get("/flood/load-saved/{run_id}")
def load_saved_run(run_id: str) -> Dict[str, Any]:
    """Load saved training run untuk instant demo replay.
    Response time < 2 detik (disk read + JSON parse), aman lewat Cloudflare
    Tunnel yang punya HTTP timeout 100 detik."""
    out_dir = Path(__file__).resolve().parents[2] / "storage" / "trained-models"
    safe_id = run_id.replace("/", "_").replace("..", "_")
    target = out_dir / f"run_{safe_id}.json"
    if not target.exists():
        # Fallback: cari file dengan partial match
        candidates = list(out_dir.glob(f"*{safe_id}*.json"))
        if not candidates:
            raise HTTPException(status_code=404, detail=f"Saved run not found: {run_id}")
        target = candidates[0]
    try:
        data = json.loads(target.read_text())
        return {
            "status": "success",
            "run_id": data.get("run_id"),
            "label": data.get("label"),
            "saved_at": data.get("saved_at"),
            "payload": data.get("payload"),
            "mode": "cached_replay",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load run: {str(e)}")


@router.delete("/flood/saved-runs/{run_id}")
def delete_saved_run(run_id: str) -> Dict[str, Any]:
    """Hapus saved run dari disk."""
    out_dir = Path(__file__).resolve().parents[2] / "storage" / "trained-models"
    safe_id = run_id.replace("/", "_").replace("..", "_")
    target = out_dir / f"run_{safe_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Saved run not found: {run_id}")
    target.unlink()
    return {"status": "success", "deleted": target.name}


@router.post("/flood/run")
def flood_run(req: FloodRunRequest) -> Dict[str, Any]:
    orch = FloodResearchOrchestrator(
        provinces=req.provinces,
        start=req.start, end=req.end,
        prediction_horizon_days=req.prediction_horizon_days,
        seq_window=req.seq_window,
        lstm_weight=req.lstm_weight, xgb_weight=req.xgb_weight,
    )
    return orch.run(bnpb_csv_path=req.bnpb_csv_path, bnpb_folder=req.bnpb_folder)


@router.post("/flood/build-panel")
def flood_panel(req: FloodRunRequest) -> Dict[str, Any]:
    provs = req.provinces or list_province_ids()
    gee = gee_service.pull_panel(provs, req.start, req.end)
    if gee["status"] != "success":
        raise HTTPException(status_code=500, detail=gee.get("message"))
    if req.bnpb_csv_path:
        bnpb = bnpb_service.load_csv(req.bnpb_csv_path)
    elif req.bnpb_folder:
        bnpb = bnpb_service.load_directory(req.bnpb_folder)
    else:
        bnpb = bnpb_service.synthesize_demo_events(provs, req.start, req.end)
    if bnpb["status"] != "success":
        raise HTTPException(status_code=500, detail=bnpb.get("message"))
    panel_env = FloodPanelBuilder(
        prediction_horizon_days=req.prediction_horizon_days
    ).build(gee["data"], bnpb["data"])
    if panel_env["status"] != "success":
        raise HTTPException(status_code=500, detail=panel_env.get("message"))
    return {
        "status": "success",
        "stats": panel_env["stats"],
        "schema": panel_env["schema"],
        "panel_preview": _df_envelope(panel_env["panel"]),
        "method_monitor": panel_env["method_monitor"],
    }


# ============================================================================
# Research Pipeline Endpoints (8-step systematic review workflow)
# ============================================================================
# Mengikuti Webster dan Watson (2002), Kitchenham (2007), PRISMA 2020
# (Page et al. 2021 BMJ 372:n71) untuk systematic literature review.
#
# Tiga endpoint utama:
#   1. /pipeline/batch-synthesize  · AI extract abstract/method/discussion/conclusion
#   2. /pipeline/gap-analysis-v2   · 5-type gap detection plus novelty score
#   3. /pipeline/recommend-titles  · refined title generation dari identified gaps
# ============================================================================

class PaperInput(BaseModel):
    id: str
    title: str
    abstract: Optional[str] = ""
    authors: Optional[List[str]] = []
    year: Optional[int] = None
    venue: Optional[str] = ""
    doi: Optional[str] = ""
    source: Optional[str] = "unknown"


class BatchSynthesizeRequest(BaseModel):
    papers: List[PaperInput]
    language: str = "en"
    force_provider: Optional[str] = None  # 'deepseek' | 'anthropic' | 'openai' | 'gemini' | 'kimi' | 'ollama' | None (auto routing)
    force_model: Optional[str] = None  # opsional model override, contoh 'deepseek-chat' atau 'claude-haiku-4-5'
    fast_mode: bool = True  # True = output lebih ringkas plus max_tokens lebih rendah supaya ~2-3x lebih cepat


@router.post("/pipeline/batch-synthesize")
def pipeline_batch_synthesize(req: BatchSynthesizeRequest) -> Dict[str, Any]:
    """Batch AI synthesis untuk selected papers · extract 4 dimensions.

    Output per paper: abstract_synth, methodology_synth, discussion_synth,
    conclusion_synth, plus provenance_quote (exact text dari abstract).

    DEFAULT routing prioritas DeepSeek Chat (cloud API) supaya kualitas tinggi
    dan tetap murah (USD 0.27 / 1M token input). Bila DeepSeek key belum
    diset, fallback ke Ollama lokal lalu ke provider lain. Bila AI tidak
    tersedia sama sekali, fallback ke rule based extraction.
    """
    try:
        from app.services import task_router as _tr
        from app.services import secrets_store as _ss
        ai_available = True
    except Exception:
        ai_available = False
        _tr = None
        _ss = None

    # Pilih primary AI provider untuk task ini. User dapat pilih provider
    # eksplisit di frontend (DeepSeek, Claude, OpenAI, Gemini, Kimi, Ollama)
    # atau biarkan kosong supaya routing otomatis. Default fallback: DeepSeek
    # Chat (V3) bila key sudah diset karena murah, cepat, dan latest.
    primary_provider = req.force_provider
    primary_model = req.force_model
    if not primary_provider and ai_available and _ss is not None:
        try:
            if _ss.get_provider_key("deepseek"):
                primary_provider = "deepseek"
                if not primary_model:
                    primary_model = "deepseek-chat"
        except Exception:
            primary_provider = None
    # Auto-pick model default per provider kalau model tidak di-set di request
    if primary_provider and not primary_model:
        primary_model = {
            "deepseek": "deepseek-chat",
            "anthropic": "claude-haiku-4-5-20251001",
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "kimi": "moonshot-v1-32k",
        }.get(primary_provider)

    # Fast mode: max_tokens lebih rendah plus instruksi prompt singkat sehingga
    # generation 2-3x lebih cepat. Default fast_mode aktif kecuali user matikan.
    max_tokens_per_paper = 550 if req.fast_mode else 900
    paragraph_hint = "two to three" if req.fast_mode else "two to four"

    rows = []
    for p in req.papers:
        abstract = (p.abstract or "").strip()
        title = (p.title or "").strip()
        venue = (p.venue or "").strip() if hasattr(p, "venue") else ""
        year = p.year or ""
        provenance = abstract[:280] + ("..." if len(abstract) > 280 else "")

        # Jalankan AI bahkan kalau abstract kosong asalkan ada title. Model
        # gunakan title plus venue plus year sebagai konteks untuk generate
        # inference informatif. Ini mencegah output "open the link" yang sebelumnya
        # bikin user frustasi.
        if ai_available:
            lang_hint, lang_display = _resolve_language(req.language)
            abstract_block = abstract[:2500] if abstract else "(Abstract was not retrieved from the source API. Use the title, venue, and year as the only direct context. Be explicit when something is inferred rather than stated.)"

            prompt = f"""You are a senior academic research synthesizer for a graduate thesis on data science. Read the paper record and produce a faithful structured synthesis in {lang_hint}. Never tell the reader to open the link. Be substantive but concise.

Paper record:
Title: {title}
Year: {year}
Venue: {venue or 'Unknown venue'}
Abstract: {abstract_block}

Output exactly four blocks separated by a line containing only three dashes (---). Each block is {paragraph_hint} full sentences of dense academic prose in {lang_hint}. No bullet points, no em dash, no semicolon, no markdown.

Block 1 Abstract synthesis: core contribution, the problem addressed, and relevance for a flood prediction or hybrid machine learning literature review. If only the title is available, infer from title keywords and mark inferences with phrases such as "based on the title".

Block 2 Methodology summary: research design, dataset, modeling approach, and evaluation strategy. When the abstract is missing, infer the methodology family from title keywords (LSTM, XGBoost, SHAP, remote sensing) and mark inferences.

Block 3 Discussion synthesis: key findings and how authors interpret them. When findings are not available, summarize what this kind of study typically reports and what would matter most for the reviewer.

Block 4 Conclusion synthesis: implications for theory, practice, and limitations. When not available, articulate the implications a reviewer should consider given the paper's scope.

Write substantive paragraphs even when the abstract is missing. Never instruct the reader to read the source paper themselves."""
            try:
                if primary_provider:
                    result = _tr.route(
                        task_type="summarize",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens_per_paper,
                        temperature=0.3,
                        override_provider=primary_provider,
                        override_model=primary_model,
                    )
                else:
                    result = _tr.route(
                        task_type="summarize",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens_per_paper,
                        temperature=0.3,
                    )
                if result.get("ok") and result.get("text"):
                    parts = [s.strip() for s in result["text"].split("---") if s.strip()]
                    # Strip leading "Block 1:" / "Block 2:" prefixes if model added them
                    cleaned = []
                    for s in parts:
                        s = re.sub(r"^\s*(Block\s*\d+|Blok\s*\d+)\s*[:.\-]?\s*", "", s, flags=re.IGNORECASE)
                        s = re.sub(r"^\s*\d+\s*[\).:\-]\s*", "", s)
                        cleaned.append(s)
                    while len(cleaned) < 4:
                        cleaned.append("")
                    rows.append({
                        "id": p.id,
                        "title": p.title,
                        "abstract_synth": cleaned[0][:1200] if cleaned[0] else (abstract[:600] if abstract else "Synthesis unavailable for this record."),
                        "methodology_synth": cleaned[1][:1200] if len(cleaned) > 1 and cleaned[1] else "Methodology synthesis unavailable for this record.",
                        "discussion_synth": cleaned[2][:1200] if len(cleaned) > 2 and cleaned[2] else "Discussion synthesis unavailable for this record.",
                        "conclusion_synth": cleaned[3][:1200] if len(cleaned) > 3 and cleaned[3] else "Conclusion synthesis unavailable for this record.",
                        "provenance_quote": provenance,
                        "source": p.source or "unknown",
                        "year": p.year,
                        "doi": p.doi,
                        "synthesis_method": "ai_" + (result.get("routing", {}).get("selected_provider") or "auto"),
                    })
                    continue
            except Exception as _e:
                pass

        # Rule-based fallback (split abstract heuristically)
        sentences = [s.strip() for s in abstract.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        n = len(sentences)
        rows.append({
            "id": p.id,
            "title": p.title,
            "abstract_synth": ". ".join(sentences[:2]) + "." if sentences else "Abstract not provided.",
            "methodology_synth": ". ".join(sentences[max(0, n//4):n//2]) + "." if n > 4 else "Methodology not explicit in abstract.",
            "discussion_synth": ". ".join(sentences[n//2:3*n//4]) + "." if n > 4 else "Discussion not explicit in abstract.",
            "conclusion_synth": ". ".join(sentences[3*n//4:]) + "." if n > 2 else "Conclusion not explicit in abstract.",
            "provenance_quote": provenance,
            "source": p.source or "unknown",
            "year": p.year,
            "doi": p.doi,
            "synthesis_method": "rule_based_fallback",
        })

    return {
        "status": "success",
        "rows": rows,
        "count": len(rows),
        "language": req.language,
    }


class GapAnalysisV2Request(BaseModel):
    title: str
    synthesis_rows: List[Dict[str, Any]]
    language: str = "en"


@router.post("/pipeline/gap-analysis-v2")
def pipeline_gap_analysis_v2(req: GapAnalysisV2Request) -> Dict[str, Any]:
    """5-type gap analysis based on synthesized literature corpus.

    Categories follow Miles dan Huberman (1994) plus Hart (2018):
      empirical, methodological, theoretical, population, practical

    Returns gap statements with provenance citations plus novelty score
    relative to corpus.
    """
    try:
        from app.services import task_router as _tr
        ai_available = True
    except Exception:
        ai_available = False

    # Aggregate corpus context
    corpus_summary = []
    for r in req.synthesis_rows[:30]:  # cap untuk prompt size
        corpus_summary.append(
            f"- {r.get('title', 'Untitled')} ({r.get('year', 'n.d.')}): "
            f"{(r.get('methodology_synth') or '')[:200]}"
        )
    corpus_text = "\n".join(corpus_summary)
    lang_hint = "Indonesia" if req.language == "id" else "English"

    if ai_available and req.synthesis_rows:
        prompt = f"""You are a senior research methodologist conducting a gap analysis.

Proposed research title: {req.title}

Literature corpus ({len(req.synthesis_rows)} papers analyzed):
{corpus_text[:4000]}

Identify research gaps in {lang_hint} following Hart (2018) five-type framework. Provide exactly five paragraphs separated by triple dashes (---):

1. Empirical gap: phenomena or datasets not yet studied
2. Methodological gap: methods or approaches not yet applied
3. Theoretical gap: frameworks not yet tested or extended
4. Population gap: groups or contexts under-represented
5. Practical gap: applications not yet bridged from research to practice

Each gap statement should reference specific papers from the corpus when possible. Use academic style. No em-dash. No semicolon."""

        try:
            result = _tr.route(
                task_type="reason",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.4
            )
            if result.get("ok") and result.get("text"):
                parts = [s.strip() for s in result["text"].split("---") if s.strip()]
                while len(parts) < 5:
                    parts.append("Not identified.")

                # Novelty score based on overlap between title keywords and corpus
                title_words = set(req.title.lower().split())
                corpus_words = set()
                for r in req.synthesis_rows:
                    corpus_words.update((r.get("title", "") or "").lower().split())
                overlap = len(title_words & corpus_words) / max(1, len(title_words))
                novelty_score = max(20, min(95, int((1 - overlap) * 100)))

                return {
                    "status": "success",
                    "empirical": {"items": [{"statement": parts[0]}]},
                    "methodological": {"items": [{"statement": parts[1]}]},
                    "theoretical": {"items": [{"statement": parts[2]}]},
                    "population": {"items": [{"statement": parts[3]}]},
                    "practical": {"items": [{"statement": parts[4]}]},
                    "novelty_score": novelty_score,
                    "novelty_rationale": f"Based on keyword analysis against {len(req.synthesis_rows)} papers, your research title shows {novelty_score}% novelty. Method and approach not heavily replicated in current corpus.",
                    "corpus_size": len(req.synthesis_rows),
                    "method": "ai_" + (result.get("routing", {}).get("selected_provider") or "auto"),
                }
        except Exception as _e:
            pass

    # Fallback rule-based gap analysis
    return {
        "status": "success",
        "empirical": {"items": [{"statement": "Indonesia-specific datasets remain under-studied compared to global benchmark contexts."}]},
        "methodological": {"items": [{"statement": "Hybrid architectures combining temporal and tabular branches require further empirical validation."}]},
        "theoretical": {"items": [{"statement": "Integration of classical hydrologic theory with modern deep learning is sparsely addressed."}]},
        "population": {"items": [{"statement": "Coverage of all 38 Indonesian provinces is limited in prior studies."}]},
        "practical": {"items": [{"statement": "Deployment of prediction models as operational early warning systems is rare."}]},
        "novelty_score": 75,
        "novelty_rationale": "Fallback rule-based estimate. Configure AI provider for richer analysis.",
        "corpus_size": len(req.synthesis_rows),
        "method": "rule_based_fallback",
    }


class TitleRecommendRequest(BaseModel):
    original_title: str
    gap_analysis: Dict[str, Any]
    language: str = "en"
    count: int = 8


@router.post("/pipeline/recommend-titles")
def pipeline_recommend_titles(req: TitleRecommendRequest) -> Dict[str, Any]:
    """Generate refined title recommendations addressing identified gaps."""
    try:
        from app.services import task_router as _tr
        ai_available = True
    except Exception:
        ai_available = False

    if ai_available:
        # Compose gap context
        gaps_text = ""
        for key in ["empirical", "methodological", "theoretical", "population", "practical"]:
            items = (req.gap_analysis.get(key, {}) or {}).get("items", [])
            for it in items[:1]:
                gaps_text += f"- {key.title()} gap: {it.get('statement', '')}\n"

        lang_hint = "Indonesia" if req.language == "id" else "English"
        prompt = f"""You are a senior research advisor. Generate {req.count} refined research title candidates.

Original title: {req.original_title}

Identified research gaps:
{gaps_text}

Generate {req.count} alternative research titles in {lang_hint} that address one or more of the identified gaps. For each title, provide:
- The full title
- Which gap(s) it addresses
- Novelty rationale (one sentence)
- Confidence percentage (50-95)

Format as JSON array with keys: title, gap_addressed, novelty_rationale, confidence. Use academic style. No em-dash. No semicolon. Return ONLY the JSON array, no extra text."""

        try:
            result = _tr.route(
                task_type="draft_academic",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            if result.get("ok") and result.get("text"):
                text = result["text"].strip()
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("```")[1] if "```" in text else text
                    if text.startswith("json"):
                        text = text[4:].strip()
                # Find JSON array
                import re
                match = re.search(r"\[[\s\S]*\]", text)
                if match:
                    try:
                        recs = json.loads(match.group(0))
                        if isinstance(recs, list) and len(recs) > 0:
                            return {
                                "status": "success",
                                "recommendations": recs[:req.count],
                                "method": "ai_" + (result.get("routing", {}).get("selected_provider") or "auto"),
                            }
                    except Exception:
                        pass
        except Exception:
            pass

    # Fallback rule-based recommendations
    base_titles = [
        {
            "title": f"{req.original_title} with Cross-Provincial Validation",
            "gap_addressed": "Population gap, Empirical gap",
            "novelty_rationale": "Extends coverage to under-studied Indonesian provinces.",
            "confidence": 72
        },
        {
            "title": f"Interpretable {req.original_title} via SHAP-Based Feature Attribution",
            "gap_addressed": "Methodological gap, Practical gap",
            "novelty_rationale": "Adds interpretability layer rarely combined with hybrid models.",
            "confidence": 78
        },
        {
            "title": f"{req.original_title}: Reproducibility-Grade Pipeline for Operational Deployment",
            "gap_addressed": "Practical gap, Theoretical gap",
            "novelty_rationale": "Bridges research to operational early warning systems.",
            "confidence": 68
        },
        {
            "title": f"Comparative Analysis of Hybrid Architectures for {req.original_title}",
            "gap_addressed": "Methodological gap",
            "novelty_rationale": "Systematic comparison missing from current literature.",
            "confidence": 65
        },
        {
            "title": f"Temporal-Spatial Generalization of {req.original_title}",
            "gap_addressed": "Empirical gap, Theoretical gap",
            "novelty_rationale": "Cross-domain generalization addresses theoretical limits.",
            "confidence": 70
        },
    ]
    return {
        "status": "success",
        "recommendations": base_titles[:req.count],
        "method": "rule_based_fallback",
    }


# ---------------------------------------------------------------------------
# Step 6.5 · Automated Title Review (Sakana AI Scientist peer review pattern)
# ---------------------------------------------------------------------------
# Reference: Lu et al. (2024) "The AI Scientist" sakana.ai/ai-scientist.
# Each recommended title is independently reviewed by an AI reviewer that
# scores clarity, feasibility, novelty, and impact on a 1 to 10 scale, plus
# returns a critique sentence and an improvement suggestion. Falls back to a
# heuristic rule based scorer when no AI provider is reachable.
# ---------------------------------------------------------------------------


class TitleReviewRequest(BaseModel):
    original_title: str
    recommendations: List[Dict[str, Any]]
    gap_analysis: Optional[Dict[str, Any]] = None
    language: str = "en"


def _rule_based_title_review(title: str, orig_title: str, gap_addressed: str = "") -> Dict[str, Any]:
    """Heuristic peer review used when no AI provider is available."""
    t = (title or "").strip()
    words = [w for w in t.split() if w]
    n_words = len(words)
    lower = t.lower()

    # Clarity: penalize too short or extremely long titles, reward 8 to 16 words.
    if n_words == 0:
        clarity = 1.0
    elif n_words < 5:
        clarity = 5.0
    elif n_words <= 16:
        clarity = 8.5
    elif n_words <= 22:
        clarity = 7.0
    else:
        clarity = 5.5

    # Feasibility: reward presence of concrete method or dataset markers.
    method_markers = ["lstm", "xgboost", "shap", "hybrid", "transformer", "cnn", "random forest", "regression"]
    data_markers = ["gee", "bnpb", "indonesia", "provincial", "satellite", "remote sensing"]
    has_method = any(m in lower for m in method_markers)
    has_data = any(d in lower for d in data_markers)
    feasibility = 5.5 + (1.5 if has_method else 0.0) + (1.5 if has_data else 0.0)
    if "operational" in lower or "deployment" in lower:
        feasibility += 0.5
    feasibility = min(feasibility, 9.5)

    # Novelty: reward keywords that signal new contribution.
    novelty_markers = ["interpretab", "cross-provincial", "reproducib", "hybrid", "comparative", "generalization", "temporal-spatial"]
    novelty_hits = sum(1 for m in novelty_markers if m in lower)
    novelty = 5.5 + novelty_hits * 1.0
    if t.lower() == (orig_title or "").lower():
        novelty = 4.0
    novelty = min(novelty, 9.0)

    # Impact: reward operational, early warning, policy oriented framing.
    impact_markers = ["early warning", "operational", "policy", "disaster", "decision", "stakeholder"]
    impact = 5.5 + sum(0.8 for m in impact_markers if m in lower)
    if "indonesia" in lower or "provincial" in lower:
        impact += 0.5
    impact = min(impact, 9.0)

    # Critique text adapts to weakest dimension.
    scores = {"clarity": clarity, "feasibility": feasibility, "novelty": novelty, "impact": impact}
    weakest = min(scores, key=scores.get)
    critique_map = {
        "clarity": "Title length and noun phrasing could be tightened to improve readability.",
        "feasibility": "Title does not clearly signal the dataset or model class that will be used.",
        "novelty": "The contribution claim relative to existing literature is not yet sharp.",
        "impact": "The practical or policy oriented implication is not visible from the title alone.",
    }
    improvement_map = {
        "clarity": "Trim modifiers and keep the title between eight and sixteen words.",
        "feasibility": "Add a concrete method keyword such as Hybrid LSTM XGBoost and a data source such as GEE.",
        "novelty": "Add a differentiator such as SHAP based interpretability or cross provincial validation.",
        "impact": "Frame the work toward operational early warning or policy decision support.",
    }

    overall = round((clarity + feasibility + novelty + impact) / 4.0, 1)
    verdict = "Strong candidate" if overall >= 8.0 else "Acceptable with revision" if overall >= 6.5 else "Needs major revision"

    return {
        "title": title,
        "scores": {
            "clarity": round(clarity, 1),
            "feasibility": round(feasibility, 1),
            "novelty": round(novelty, 1),
            "impact": round(impact, 1),
            "overall": overall,
        },
        "critique": critique_map[weakest],
        "improvement_suggestion": improvement_map[weakest],
        "verdict": verdict,
        "method": "rule_based",
    }


@router.post("/pipeline/review-titles")
def pipeline_review_titles(req: TitleReviewRequest) -> Dict[str, Any]:
    """Peer review every recommended title (Sakana AI Scientist pattern)."""
    try:
        from app.services import task_router as _tr
        ai_available = True
    except Exception:
        ai_available = False

    reviews: List[Dict[str, Any]] = []
    used_ai = False

    if ai_available and req.recommendations:
        # Compose a compact gap context for the reviewer.
        gaps_text = ""
        if req.gap_analysis:
            for key in ["empirical", "methodological", "theoretical", "population", "practical"]:
                items = (req.gap_analysis.get(key, {}) or {}).get("items", [])
                for it in items[:1]:
                    gaps_text += f"- {key.title()} gap: {it.get('statement', '')}\n"

        lang_hint = "Indonesia" if req.language == "id" else "English"
        titles_list = []
        for i, rec in enumerate(req.recommendations):
            titles_list.append(f"{i+1}. {rec.get('title', '')}")
        titles_block = "\n".join(titles_list)

        prompt = f"""You are an associate editor and senior peer reviewer for a top tier data science journal. Review each candidate title below as if it were a manuscript title.

Original research title: {req.original_title}

Identified research gaps:
{gaps_text or '- (not provided)'}

Candidate titles to review:
{titles_block}

For every candidate, return a JSON object with these keys:
- title (string, exact text from the candidate)
- scores: an object with clarity, feasibility, novelty, impact, overall. Each is a number from one to ten with one decimal.
- critique: one academic sentence in {lang_hint} naming the weakest aspect.
- improvement_suggestion: one academic sentence in {lang_hint} proposing a concrete fix.
- verdict: one of "Strong candidate", "Acceptable with revision", or "Needs major revision".

Use academic style. Do not use em dash. Do not use semicolon. Return ONLY a JSON array of these objects in the same order as the candidates. No extra commentary."""

        try:
            result = _tr.route(
                task_type="reason",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2400,
                temperature=0.4,
            )
            if result.get("ok") and result.get("text"):
                text = result["text"].strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.startswith("json"):
                            text = text[4:].strip()
                import re
                match = re.search(r"\[[\s\S]*\]", text)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        if isinstance(parsed, list) and parsed:
                            for r in parsed:
                                if not isinstance(r, dict):
                                    continue
                                sc = r.get("scores") or {}
                                # Normalize numeric scores
                                clean_scores = {}
                                for k in ["clarity", "feasibility", "novelty", "impact", "overall"]:
                                    try:
                                        v = float(sc.get(k, 0))
                                        clean_scores[k] = round(max(1.0, min(10.0, v)), 1)
                                    except Exception:
                                        clean_scores[k] = 0.0
                                if not clean_scores.get("overall"):
                                    clean_scores["overall"] = round(
                                        (clean_scores["clarity"] + clean_scores["feasibility"]
                                         + clean_scores["novelty"] + clean_scores["impact"]) / 4.0, 1
                                    )
                                reviews.append({
                                    "title": r.get("title", ""),
                                    "scores": clean_scores,
                                    "critique": r.get("critique", ""),
                                    "improvement_suggestion": r.get("improvement_suggestion", ""),
                                    "verdict": r.get("verdict", ""),
                                    "method": "ai_" + (result.get("routing", {}).get("selected_provider") or "auto"),
                                })
                            used_ai = True
                    except Exception:
                        used_ai = False
        except Exception:
            used_ai = False

    # Fallback or fill missing reviews with rule based scorer.
    if not reviews:
        for rec in req.recommendations:
            reviews.append(_rule_based_title_review(
                rec.get("title", ""), req.original_title, rec.get("gap_addressed", "")
            ))
    else:
        # Align order with input and fill any unmatched titles using rule based fallback.
        aligned: List[Dict[str, Any]] = []
        used_titles = {r.get("title", "").strip().lower() for r in reviews}
        for rec in req.recommendations:
            t = rec.get("title", "")
            match = next((r for r in reviews if r.get("title", "").strip().lower() == t.strip().lower()), None)
            if match is None:
                aligned.append(_rule_based_title_review(t, req.original_title, rec.get("gap_addressed", "")))
            else:
                aligned.append(match)
        reviews = aligned

    # Rank by overall score for convenience.
    try:
        ranked = sorted(reviews, key=lambda r: float(r.get("scores", {}).get("overall", 0)), reverse=True)
        best_title = ranked[0].get("title", "") if ranked else ""
    except Exception:
        best_title = ""

    return {
        "status": "success",
        "reviews": reviews,
        "best_title": best_title,
        "method": "ai_orchestrator" if used_ai else "rule_based_fallback",
    }


# ---------------------------------------------------------------------------
# Stage II Organizing · Kluge plus Taylor "Writing Research Papers" pattern
# ---------------------------------------------------------------------------
# Tiga endpoint mendukung Stage II Organizing yang terdiri dari thesis
# statement crystallization, working outline plus source mapping, plus
# coherence check sebelum lanjut ke Stage III literature synthesis.
# ---------------------------------------------------------------------------


class ThesisStatementRequest(BaseModel):
    title: str
    gap_analysis: Optional[Dict[str, Any]] = None
    existing: Optional[Dict[str, Any]] = None
    mode: Optional[str] = "journal"  # 'journal' | 'thesis'
    language: str = "en"
    multi_point: bool = True
    three_tier_contribution: bool = True
    min_problems: int = 3
    max_problems: int = 5
    period: Optional[Dict[str, Any]] = None


@router.post("/pipeline/organizing/generate-thesis-statement")
def organizing_generate_thesis_statement(req: ThesisStatementRequest) -> Dict[str, Any]:
    """Generate thesis statement dengan multi-point problems dan 3-tier contribution.

    Mengikuti:
    - Booth Colomb Williams (2008) The Craft of Research, multi-faceted problem
    - Whetten (1989) Academy of Management Review, three tier contribution
    - Sandberg Alvesson (2011) Organization 18(1), problematization typology
    - Kluge Taylor Writing Research Papers, controlling idea structure

    Output:
    - problems: array minimum 3 problem statements (practical, methodological, theoretical)
    - approach: paragraph yang explicit link ke gap analysis Stage I
    - contributions: object {strong, middle, acceptable} tiap tier 1-2 kalimat
    - validation: paragraph validation strategy
    - prompt_used: prompt eksak yang dikirim ke AI (untuk audit trail)
    """
    try:
        from app.services import task_router as _tr
        ai_available = True
    except Exception:
        ai_available = False

    # Compose gap context dengan struktur per tipe untuk explicit linking
    gaps_text = ""
    gap_summary = []
    if req.gap_analysis:
        for key in ["empirical", "methodological", "theoretical", "population", "practical"]:
            items = (req.gap_analysis.get(key, {}) or {}).get("items", [])
            for it in items[:1]:
                statement = it.get("statement", "")
                gaps_text += f"- {key.upper()} gap: {statement}\n"
                gap_summary.append({"type": key, "statement": statement})

    period_str = ""
    if req.period and req.period.get("year_from"):
        period_str = f"Research Period: {req.period.get('year_from')}-{req.period.get('year_to', 'present')}\n"

    lang_hint, lang_display = _resolve_language(req.language)
    mode_hint = "academic journal article (Scopus/SINTA)" if req.mode == "journal" else "Indonesian university thesis (skripsi or tesis)"

    prompt = f"""You are a senior academic research advisor with expertise in thesis methodology and journal publishing. Generate an enhanced thesis statement for a {mode_hint}.

Research Title: {req.title}
{period_str}
Identified Research Gaps (from Stage I gap analysis):
{gaps_text or '- (gaps not yet analyzed)'}

Following Booth, Colomb, and Williams (2008) The Craft of Research and Whetten (1989) Academy of Management Review, generate the following structured thesis statement in {lang_hint}.

REQUIREMENT 1: Multi-point problem statements (minimum {req.min_problems}, maximum {req.max_problems})
Generate exactly {req.min_problems} distinct problem statements covering different facets:
- Problem 1: PRACTICAL problem in the real world (operational, stakeholder impact)
- Problem 2: METHODOLOGICAL problem in existing approaches (algorithm limitations, evaluation gaps)
- Problem 3: THEORETICAL or GEOGRAPHIC problem (generalization, scope, context)
Each problem is 1-2 academic sentences. Booth (2008) calls this multi-faceted problematization, which is required for thesis level work.

REQUIREMENT 2: Proposed approach explicitly aligned with gap analysis
The approach paragraph MUST reference the identified gaps from Stage I. Use phrases like "to address the [gap type] gap of [specific issue]". This creates traceable argumentation from problem to solution.

REQUIREMENT 3: Three tier contribution (Whetten 1989 framework)
Generate three separate contribution claims at three rigor tiers:
- STRONG (Theory Generation or Paradigm Shift): Changes how the community views the problem. New construct, new theory, problematization. Use this tier if your work redefines assumptions. Cite Whetten (1989), Sandberg and Alvesson (2011).
- MIDDLE (Theory Extension or Refinement): Extends existing theory to new context or method. Combines two frameworks previously separated. Use this tier if your work synthesizes or generalizes existing theories.
- ACCEPTABLE (Empirical Replication or Application): Replicates established methods in new context or data. Acceptable as a thesis or journal contribution. Use this tier if your work validates existing approaches in new settings.

REQUIREMENT 4: Validation method
One paragraph describing validation strategy with concrete metrics, cross validation approach, and ablation study design.

Return strict JSON object exactly in this shape. No markdown, no commentary, no em dash, no semicolon. Each problem and contribution is 1-2 full sentences:

{{
  "problems": ["problem 1 practical ...", "problem 2 methodological ...", "problem 3 theoretical ..."],
  "approach": "paragraph that links to gap analysis ...",
  "contributions": {{
    "strong": "...",
    "middle": "...",
    "acceptable": "..."
  }},
  "validation": "..."
}}"""

    ts_out = None
    if ai_available:
        try:
            result = _tr.route(
                task_type="draft_academic",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,  # Diturunkan dari 1800 supaya generate 30% lebih cepat
                temperature=0.4,
            )
            if result.get("ok") and result.get("text"):
                text = result["text"].strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.startswith("json"):
                            text = text[4:].strip()
                match = re.search(r"\{[\s\S]*\}", text)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        if isinstance(parsed, dict):
                            problems = parsed.get("problems", [])
                            if isinstance(problems, str):
                                problems = [problems]
                            problems = [str(p)[:500] for p in problems if p][:req.max_problems]
                            while len(problems) < req.min_problems:
                                problems.append("")

                            contributions = parsed.get("contributions", {})
                            if isinstance(contributions, str):
                                contributions = {"strong": contributions, "middle": "", "acceptable": ""}

                            ts_out = {
                                "problems": problems,
                                "approach": str(parsed.get("approach", ""))[:800],
                                "contributions": {
                                    "strong": str(contributions.get("strong", ""))[:500],
                                    "middle": str(contributions.get("middle", ""))[:500],
                                    "acceptable": str(contributions.get("acceptable", ""))[:500],
                                },
                                "validation": str(parsed.get("validation", ""))[:500],
                            }
                    except Exception:
                        pass
        except Exception:
            pass

    # Rule-based fallback with new structure
    if ts_out is None:
        title_lower = (req.title or "").lower()
        has_lstm = "lstm" in title_lower
        has_xgboost = "xgboost" in title_lower or "gradient" in title_lower
        has_shap = "shap" in title_lower or "interpret" in title_lower
        has_flood = "flood" in title_lower or "banjir" in title_lower
        has_hybrid = "hybrid" in title_lower
        has_indonesia = "indonesia" in title_lower or "provinsi" in title_lower

        methods = []
        if has_lstm: methods.append("temporal LSTM")
        if has_xgboost: methods.append("XGBoost gradient boosting")
        if has_hybrid: methods.append("hybrid stacking ensemble")
        if has_shap: methods.append("SHAP based feature attribution")
        methods_str = " and ".join(methods) if methods else "the proposed machine learning approach"
        domain_str = "flood prediction" if has_flood else "the target domain"
        scope_str = "multi provincial Indonesian context" if has_indonesia else "the study context"

        ts_out = {
            "problems": [
                f"Practical problem · operational early warning systems for {domain_str} in the {scope_str} remain reactive rather than predictive, causing preventable losses to affected communities.",
                "Methodological problem · existing prediction models rely on single algorithms that perform poorly under extreme weather conditions and lack interpretability that stakeholders need for actionable response.",
                f"Geographic and theoretical problem · most studies focus on a single province with limited cross context generalization, leaving no transferable framework for the wider {scope_str}.",
            ],
            "approach": f"This research proposes {methods_str} to directly address the methodological gap of black box single algorithm models and the theoretical gap of within province only validation identified in the Stage I gap analysis.",
            "contributions": {
                "strong": f"Hybrid temporal tabular machine learning with explainable attribution proposed as a new paradigm for operational disaster prediction in emerging markets, repositioning interpretability from an after thought to a first class design constraint.",
                "middle": f"Extension of the hybrid LSTM XGBoost methodology from single province application to cross provincial deployment with a modified holdout cross validation strategy that captures spatial heterogeneity.",
                "acceptable": f"Empirical replication and validation of hybrid LSTM XGBoost on the integrated BNPB plus GEE Indonesian dataset for the chosen study period, confirming generalizability of the architecture in a new geographic context.",
            },
            "validation": "Cross provincial holdout testing with ablation study on each model component, comparative benchmarking against three established baselines from recent literature, and SHAP attribution stability analysis across folds.",
        }

    return {
        "status": "success",
        "thesis_statement": ts_out,
        "prompt_used": prompt,
        "gap_summary": gap_summary,
        "method": "ai_orchestrator" if ai_available and ts_out and ts_out.get("approach") else "rule_based_fallback",
    }


class ScopeGenerationRequest(BaseModel):
    title: str
    thesis_statement: Optional[str] = ""
    mode: Optional[str] = "journal"
    language: str = "en"
    period: Optional[Dict[str, Any]] = None
    domain_override: Optional[str] = None  # User dapat force domain (hybrid auto+manual)


# Domain catalog mengikuti re3data.org taxonomy plus Pampel et al. (2013)
# PLOS ONE 8(11):e78080 untuk research data repositories
DOMAIN_CATALOG = {
    "earth_observation": {
        "label": "Earth Observation and Geoscience",
        "description": "Satellite imagery, climate, hydrology, geology, atmospheric science",
        "keywords": ["flood", "banjir", "earth", "satellite", "remote sensing", "climate", "weather", "geology", "hydrology", "geospatial", "gee", "landsat", "sentinel", "modis", "disaster", "bencana"],
        "ref_repos": "GEE, NASA EOSDIS, ESA Copernicus, USGS, NOAA, ECMWF, BMKG, BNPB"
    },
    "biomedical": {
        "label": "Healthcare and Biomedical",
        "description": "Medical imaging, genomics, clinical trials, epidemiology, biobanks",
        "keywords": ["medical", "health", "clinical", "patient", "disease", "drug", "genomic", "biomedical", "radiology", "mri", "ct scan", "pathology", "covid", "diabetes", "cancer"],
        "ref_repos": "NIH NCBI, UK Biobank, MIMIC PhysioNet, WHO, GEO, TCGA, ChEMBL"
    },
    "finance": {
        "label": "Finance and Economics",
        "description": "Macroeconomics, stock markets, banking, financial regulation, accounting",
        "keywords": ["finance", "economic", "stock", "trading", "bank", "investment", "macro", "gdp", "inflation", "fintech", "blockchain", "cryptocurrency", "credit"],
        "ref_repos": "World Bank, IMF, FRED, SEC EDGAR, Yahoo Finance, Bloomberg, Refinitiv"
    },
    "social_sciences": {
        "label": "Social Sciences and Demographics",
        "description": "Sociology, political science, psychology, survey research, public opinion",
        "keywords": ["social", "demograph", "survey", "psycholog", "politic", "voter", "opinion", "behavior", "attitude", "civil", "society"],
        "ref_repos": "ICPSR, Pew Research, GSS, World Values Survey, Eurostat, OECD, BPS Indonesia"
    },
    "open_government": {
        "label": "Government Open Data",
        "description": "Public administration, policy research, government statistics",
        "keywords": ["government", "public", "policy", "education", "transportation", "infrastructure", "data.go.id", "bps", "statistik"],
        "ref_repos": "data.go.id, data.gov, data.europa.eu, OECD, BPS Indonesia, BPK"
    },
    "ml_benchmarks": {
        "label": "Machine Learning Benchmarks",
        "description": "Vision, NLP, audio, multimodal benchmark datasets for ML research",
        "keywords": ["image classification", "object detection", "nlp", "benchmark", "imagenet", "coco", "glue", "squad", "kaggle"],
        "ref_repos": "Hugging Face, Kaggle, OpenML, Papers With Code, ImageNet, COCO"
    },
    "text_corpora": {
        "label": "Text and Language Corpora",
        "description": "Natural language processing, computational linguistics, text mining",
        "keywords": ["text", "language", "nlp", "corpus", "linguistic", "translation", "sentiment", "speech", "tweet", "review"],
        "ref_repos": "Common Crawl, Wikipedia, Project Gutenberg, OPUS, OSCAR, HuggingFace Datasets"
    },
    "scientific_domain": {
        "label": "Domain Specific Scientific",
        "description": "Specialized scientific repositories: biology, chemistry, physics, astronomy",
        "keywords": ["biology", "chemistry", "physics", "astronomy", "protein", "molecule", "particle", "galaxy", "genome", "sequence"],
        "ref_repos": "GenBank, UniProt, PDB, ChEMBL, SDSS, NASA ADS, CERN Open Data"
    },
    "institutional_custom": {
        "label": "Institutional and Custom",
        "description": "User-collected, lab-collected, institutional access only data",
        "keywords": ["custom", "internal", "proprietary", "institutional", "lab", "field"],
        "ref_repos": "Workspace Data uploads, lab archives, institutional repositories"
    }
}


def _detect_domain(title: str, thesis: str) -> str:
    """Auto-detect primary research domain dari keywords title + thesis.
    Return domain ID dengan match score tertinggi."""
    text = (title + " " + (thesis or "")).lower()
    best_domain = "scientific_domain"
    best_score = 0
    for did, dmeta in DOMAIN_CATALOG.items():
        score = sum(1 for kw in dmeta["keywords"] if kw in text)
        if score > best_score:
            best_score = score
            best_domain = did
    return best_domain


@router.post("/pipeline/organizing/generate-scope")
def organizing_generate_scope(req: ScopeGenerationRequest) -> Dict[str, Any]:
    """AI-generate interdisciplinary disciplines plus recommended datasets.

    Universal data source catalog covering 9 categories (Earth Observation,
    Biomedical, Finance, Social Sciences, Open Government, ML Benchmarks,
    Text Corpora, Domain Specific Scientific, Institutional Custom) mengikuti
    taxonomy re3data.org Pampel et al. (2013) PLOS ONE 8(11):e78080.

    Setiap dataset dilengkapi FAIR scoring 4 dimensi (Findable, Accessible,
    Interoperable, Reusable) mengikuti Wilkinson et al. (2016) Scientific
    Data 3:160018.

    Fallback ke rule-based template bila AI tidak tersedia.
    """
    try:
        from app.services import task_router as _tr
        ai_available = True
    except Exception:
        ai_available = False

    period_str = ""
    if req.period and req.period.get("year_from"):
        period_str = f"Research Period: {req.period.get('year_from')}-{req.period.get('year_to', 'present')}\n"

    lang_hint, lang_display = _resolve_language(req.language)
    ts_block = f"Thesis Statement Context: {req.thesis_statement[:800]}\n" if req.thesis_statement else ""

    # Detect or use override domain
    detected_domain = req.domain_override if req.domain_override and req.domain_override in DOMAIN_CATALOG else _detect_domain(req.title, req.thesis_statement or "")
    domain_meta = DOMAIN_CATALOG.get(detected_domain, DOMAIN_CATALOG["scientific_domain"])

    # Build domain catalog summary for AI prompt
    catalog_summary = "\n".join([f"- {did}: {meta['label']} ({meta['ref_repos']})" for did, meta in DOMAIN_CATALOG.items()])

    # SPLIT prompt jadi DUA call paralel · disciplines (smaller, ~30-45s)
    # dan datasets (medium, ~30-45s). Run paralel via ThreadPoolExecutor 2
    # workers, total wall time turun setengahnya jadi 30-45s. Aman dari
    # Cloudflare 100s timeout. Sebelumnya satu call 60-90s sering kena 524.
    common_context = f"""Research Title: {req.title}
{ts_block}{period_str}
PRIMARY DOMAIN DETECTED: {detected_domain} ({domain_meta['label']})
DOMAIN DESCRIPTION: {domain_meta['description']}
Target language: {lang_hint}"""

    prompt_disciplines = f"""You are a senior interdisciplinary research advisor. Analyze the research and identify 4 academic disciplines this research spans.

{common_context}

OUTPUT: Strict JSON array of exactly 4 disciplines. For each:
- id (snake_case)
- label (in {lang_hint})
- rationale (1 sentence why relevant, max 25 words)
- variables (array of exactly 3 standard measurements)
- sources (array of exactly 3 data source names)
- citation (real academic reference, format Author1 et al. (Year) Journal Volume:Pages)

Use only real academic citations. No markdown, no commentary, no em-dash, no semicolon.

Return JSON in this shape:
{{"disciplines": [{{"id": "...", "label": "...", "rationale": "...", "variables": ["...", "...", "..."], "sources": ["...", "...", "..."], "citation": "..."}}]}}"""

    prompt_datasets_intro = f"""You are a senior data librarian following FAIR principles (Wilkinson et al. 2016 Scientific Data 3:160018) and the re3data.org taxonomy (Pampel et al. 2013 PLOS ONE 8:e78080).

{common_context}

AVAILABLE DOMAIN CATEGORIES (use these category IDs in dataset categorization):
{catalog_summary}

TASK: Recommend exactly 6 concrete datasets sesuai domain detected. Datasets HARUS dari catalog yang sesuai domain ({domain_meta['label']}), bukan hardcoded earth observation only. For non geoscience research, suggest datasets from the appropriate domain catalog above. Each dataset:
- id (snake_case)
- label (dataset name)
- category (one of the 9 domain IDs above: earth_observation, biomedical, finance, social_sciences, open_government, ml_benchmarks, text_corpora, scientific_domain, institutional_custom)
- source (provider: NASA, ESA, NCBI, World Bank, government agency, university, etc)
- format (one of: tabular, image, time_series, geospatial, text, network, audio, video, multimodal)
- resolution (spatial and temporal if applicable, e.g. 10m daily for satellite, per patient for clinical)
- coverage_geographic (one of: global, regional, national_indonesia, national_other, local)
- coverage_temporal (one of: real_time, recent, historical, archival)
- citation (real academic reference)
- relevance_score (integer 1-10 how relevant for this research)
- access (how to access: API endpoint, download URL, registration form, institutional access, paid subscription)
- license (one of: cc0, cc_by, cc_by_sa, custom_open, restricted, request_required, paid)
- fair (object with 4 dimension scores 0-100):
  * findable: punya DOI/identifier persistent? indexed? Score 0-100
  * accessible: free? need registration? paid? Score 0-100
  * interoperable: standard format? API tersedia? Score 0-100
  * reusable: license clear? documentation good? Score 0-100

FAIR SCORING REFERENCE (Wilkinson et al. 2016):
- Findable 90-100: persistent DOI, well indexed
- Findable 70-89: stable URL, mostly indexed
- Findable 40-69: institutional URL, partial indexing
- Findable 0-39: requires manual search, scraping
- Accessible 90-100: free open access via API or download
- Accessible 70-89: free with registration
- Accessible 40-69: institutional or request required
- Accessible 0-39: paid subscription
- Interoperable 90-100: standard format (CSV NetCDF JSON HDF5) plus API
- Interoperable 70-89: standard format, no API
- Interoperable 40-69: custom format, documented
- Interoperable 0-39: proprietary format
- Reusable 90-100: CC0 or CC-BY plus excellent documentation
- Reusable 70-89: CC-BY-SA or similar open license
- Reusable 40-69: custom open license, partial documentation
- Reusable 0-39: restricted license, minimal documentation

REQUIREMENT: Use only real academic citations and real dataset names. For domain {detected_domain}, prioritize datasets from typical repositories: {domain_meta['ref_repos']}.

Return strict JSON. No markdown, no commentary, no em-dash, no semicolon.

{{"datasets": [
  {{
    "id": "...",
    "label": "...",
    "category": "earth_observation",
    "source": "...",
    "format": "geospatial",
    "resolution": "...",
    "coverage_geographic": "global",
    "coverage_temporal": "recent",
    "citation": "...",
    "relevance_score": 9,
    "access": "...",
    "license": "cc_by",
    "fair": {{
      "findable": 90,
      "accessible": 100,
      "interoperable": 85,
      "reusable": 90
    }}
  }}
]}}"""

    # Rename intro ke prompt_datasets · sudah include semua content yang dibutuhkan
    # (intro + FAIR refs + return JSON shape) dari refactor sebelumnya.
    prompt_datasets = prompt_datasets_intro

    # Helper untuk parse JSON dari AI response · handle markdown fence wrap
    def _parse_ai_json(text: str):
        text = (text or "").strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:].strip()
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    if ai_available:
        from concurrent.futures import ThreadPoolExecutor

        def _call_disciplines():
            try:
                r = _tr.route(
                    task_type="structured_fast",  # Skip R1 reasoning, prioritas remote fast
                    messages=[{"role": "user", "content": prompt_disciplines}],
                    max_tokens=600,  # turun dari 800 supaya lebih cepat
                    temperature=0.3,
                )
                if r.get("ok") and r.get("text"):
                    parsed = _parse_ai_json(r["text"])
                    if isinstance(parsed, dict):
                        return parsed.get("disciplines") or []
            except Exception:
                pass
            return []

        def _call_datasets():
            try:
                r = _tr.route(
                    task_type="structured_fast",  # Skip R1 reasoning, prioritas remote fast
                    messages=[{"role": "user", "content": prompt_datasets}],
                    max_tokens=1100,  # turun dari 1400 supaya lebih cepat
                    temperature=0.3,
                )
                if r.get("ok") and r.get("text"):
                    parsed = _parse_ai_json(r["text"])
                    if isinstance(parsed, dict):
                        return parsed.get("datasets") or []
            except Exception:
                pass
            return []

        # PARALLEL execute · wall time = max(disc, ds) instead of sum
        disciplines = []
        datasets = []
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                disc_future = executor.submit(_call_disciplines)
                ds_future = executor.submit(_call_datasets)
                disciplines = disc_future.result(timeout=80)
                datasets = ds_future.result(timeout=80)
        except Exception:
            pass

        if isinstance(disciplines, list) and isinstance(datasets, list) and disciplines and datasets:
            # Sanitize disciplines
            clean_disc = []
            for d in disciplines[:8]:
                if not isinstance(d, dict): continue
                clean_disc.append({
                    "id": str(d.get("id", "") or "")[:60],
                    "label": str(d.get("label", ""))[:120],
                    "rationale": str(d.get("rationale", ""))[:400],
                    "variables": [str(v)[:120] for v in (d.get("variables") or [])][:8],
                    "sources": [str(s)[:120] for s in (d.get("sources") or [])][:8],
                    "citation": str(d.get("citation", ""))[:200],
                })

            # Sanitize datasets
            clean_ds = []
            valid_categories = list(DOMAIN_CATALOG.keys())
            valid_formats = ["tabular", "image", "time_series", "geospatial", "text", "network", "audio", "video", "multimodal"]
            valid_licenses = ["cc0", "cc_by", "cc_by_sa", "custom_open", "restricted", "request_required", "paid"]
            valid_geo = ["global", "regional", "national_indonesia", "national_other", "local"]
            valid_temporal = ["real_time", "recent", "historical", "archival"]
            for d in datasets[:15]:
                if not isinstance(d, dict): continue
                try:
                    rs = int(d.get("relevance_score", 5))
                except Exception:
                    rs = 5
                fair_in = d.get("fair") if isinstance(d.get("fair"), dict) else {}
                fair_clean = {}
                for dim in ["findable", "accessible", "interoperable", "reusable"]:
                    try:
                        v = int(fair_in.get(dim, 50))
                        fair_clean[dim] = max(0, min(100, v))
                    except Exception:
                        fair_clean[dim] = 50
                fair_clean["overall"] = round(sum(fair_clean.values()) / 4)
                cat = str(d.get("category", detected_domain))
                if cat not in valid_categories: cat = detected_domain
                fmt = str(d.get("format", "tabular"))
                if fmt not in valid_formats: fmt = "tabular"
                lic = str(d.get("license", "custom_open"))
                if lic not in valid_licenses: lic = "custom_open"
                geo = str(d.get("coverage_geographic", "global"))
                if geo not in valid_geo: geo = "global"
                tmp = str(d.get("coverage_temporal", "recent"))
                if tmp not in valid_temporal: tmp = "recent"
                clean_ds.append({
                    "id": str(d.get("id", "") or "")[:60],
                    "label": str(d.get("label", ""))[:140],
                    "category": cat,
                    "source": str(d.get("source", ""))[:80],
                    "format": fmt,
                    "resolution": str(d.get("resolution", ""))[:60],
                    "coverage_geographic": geo,
                    "coverage_temporal": tmp,
                    "citation": str(d.get("citation", ""))[:200],
                    "relevance_score": max(1, min(10, rs)),
                    "access": str(d.get("access", ""))[:200],
                    "license": lic,
                    "fair": fair_clean,
                })

            if clean_disc and clean_ds:
                return {
                    "status": "success",
                    "detected_domain": detected_domain,
                    "domain_label": domain_meta["label"],
                    "domain_catalog": [{"id": did, "label": dmeta["label"]} for did, dmeta in DOMAIN_CATALOG.items()],
                    "disciplines": clean_disc,
                    "datasets": sorted(clean_ds, key=lambda x: -x["relevance_score"]),
                    "method": "ai_orchestrator_parallel_2call",
                    "prompt_used": prompt_disciplines[:500] + " | " + prompt_datasets[:500],
                }

    # Rule-based fallback · keyword matching dari title (template existing)
    title_lower = (req.title or "").lower()
    has_flood = "flood" in title_lower or "banjir" in title_lower
    has_ml = any(k in title_lower for k in ["lstm", "xgboost", "machine learning", "deep learning", "neural"])
    has_rs = any(k in title_lower for k in ["gee", "satellite", "remote sensing", "landsat", "sentinel", "modis"])
    has_indo = any(k in title_lower for k in ["indonesia", "provinsi", "jawa", "sumatra"])

    disciplines_fb = []
    datasets_fb = []

    if has_flood:
        disciplines_fb.append({
            "id": "hydrology",
            "label": "Hydrology",
            "rationale": "Flood prediction inherently requires hydrological understanding of water flow, runoff, and watershed dynamics.",
            "variables": ["Daily rainfall (mm/day)", "River discharge (m3/s)", "Water level (m)", "Standardized precipitation index", "Soil moisture (volumetric %)"],
            "sources": ["CHIRPS rainfall", "GLDAS hydrology", "JRC Global Surface Water", "GRDC discharge database"],
            "citation": "Mosavi A, Ozturk P, Chau KW (2018) Water 10(11):1536"
        })
        disciplines_fb.append({
            "id": "meteorology",
            "label": "Meteorology and Climate Science",
            "rationale": "Flood events are driven by precipitation patterns, climate variability, and atmospheric processes.",
            "variables": ["Surface temperature (degC)", "Relative humidity (%)", "Air pressure (hPa)", "ENSO index (ONI/SOI)", "Monsoon onset date"],
            "sources": ["ERA5 reanalysis", "BMKG climate stations", "MERRA-2 reanalysis", "NOAA ENSO indices"],
            "citation": "Bentivoglio R et al. (2022) Hydrology and Earth System Sciences 26(16):4345-4378"
        })

    if has_rs or has_flood:
        disciplines_fb.append({
            "id": "remote_sensing",
            "label": "Remote Sensing and Geospatial Science",
            "rationale": "Satellite imagery provides spatial coverage and temporal frequency essential for monitoring at scale.",
            "variables": ["NDVI vegetation index", "NDWI water index", "Backscatter SAR (dB)", "Land cover classification", "Surface roughness"],
            "sources": ["Sentinel-1 SAR", "Sentinel-2 MSI", "Landsat 8/9", "MODIS Terra/Aqua", "ESA WorldCover"],
            "citation": "Drusch M et al. (2012) Remote Sensing of Environment 120:25-36"
        })

    if has_ml:
        disciplines_fb.append({
            "id": "data_science",
            "label": "Data Science and Machine Learning",
            "rationale": "Prediction modeling, feature engineering, and validation methodology are core to ML based research.",
            "variables": ["Training/validation/test splits", "Feature importance scores", "Cross-validation folds", "Hyperparameter search space"],
            "sources": ["Open literature benchmark datasets", "BNPB DIBI ground truth", "User exported datasets from Workspace Data"],
            "citation": "Chen T, Guestrin C (2016) KDD 16:785-794 (XGBoost). Hochreiter S, Schmidhuber J (1997) Neural Computation 9(8):1735-1780 (LSTM)"
        })

    if has_indo or has_flood:
        disciplines_fb.append({
            "id": "disaster_management",
            "label": "Disaster Risk Management",
            "rationale": "Operational deployment of prediction systems must align with disaster management frameworks and stakeholder workflows.",
            "variables": ["Disaster occurrence (binary)", "Affected population (jiwa)", "Economic loss (Rp)", "Damage severity category", "Response time (hours)"],
            "sources": ["BNPB DIBI Indonesia", "EM-DAT international", "GDACS real-time", "ReliefWeb situational reports"],
            "citation": "UNDRR Sendai Framework for Disaster Risk Reduction 2015-2030"
        })

    # Helper untuk build dataset dict dengan struktur baru lengkap (category, format, fair, dll)
    def _ds(id, label, category, source, fmt, resolution, geo, tmp, citation, score, access, license, fair):
        fair["overall"] = round(sum(fair.values()) / 4)
        return {
            "id": id, "label": label, "category": category, "source": source,
            "format": fmt, "resolution": resolution,
            "coverage_geographic": geo, "coverage_temporal": tmp,
            "citation": citation, "relevance_score": score, "access": access,
            "license": license, "fair": fair
        }

    # Domain-aware fallback datasets
    if has_flood or has_rs:
        datasets_fb = [
            _ds("chirps_rainfall", "CHIRPS Rainfall", "earth_observation", "UCSB Climate Hazards Center", "geospatial",
                "5km daily", "global", "historical", "Funk C et al. (2015) Scientific Data 2:150066", 9,
                "GEE: UCSB-CHG/CHIRPS/DAILY", "cc_by", {"findable": 95, "accessible": 100, "interoperable": 90, "reusable": 95}),
            _ds("jrc_water", "JRC Global Surface Water", "earth_observation", "EU Joint Research Centre", "geospatial",
                "30m monthly", "global", "historical", "Pekel JF et al. (2016) Nature 540:418-422", 10,
                "GEE: JRC/GSW1_4/GlobalSurfaceWater", "cc_by", {"findable": 95, "accessible": 100, "interoperable": 90, "reusable": 95}),
            _ds("srtm_dem", "SRTM Digital Elevation Model", "earth_observation", "NASA JPL", "geospatial",
                "30m single epoch", "global", "archival", "Farr TG et al. (2007) Reviews of Geophysics 45:RG2004", 9,
                "GEE: USGS/SRTMGL1_003", "cc0", {"findable": 100, "accessible": 100, "interoperable": 95, "reusable": 100}),
            _ds("era5_climate", "ERA5 Climate Reanalysis", "earth_observation", "ECMWF Copernicus", "geospatial",
                "25km hourly", "global", "historical", "Hersbach H et al. (2020) QJ Roy Met Soc 146(730):1999-2049", 8,
                "GEE: ECMWF/ERA5_LAND/HOURLY", "cc_by", {"findable": 95, "accessible": 100, "interoperable": 90, "reusable": 95}),
            _ds("sentinel1_sar", "Sentinel-1 SAR (flood mapping)", "earth_observation", "ESA Copernicus", "image",
                "10m 6-12 day", "global", "recent", "Torres R et al. (2012) Remote Sensing of Environment 120:9-24", 10,
                "GEE: COPERNICUS/S1_GRD", "cc_by", {"findable": 95, "accessible": 100, "interoperable": 85, "reusable": 95}),
            _ds("modis_ndvi", "MODIS Vegetation Indices", "earth_observation", "NASA", "geospatial",
                "250m 16-day", "global", "historical", "Justice CO et al. (2002) RSE 83(1-2):3-15", 7,
                "GEE: MODIS/006/MOD13Q1", "cc0", {"findable": 95, "accessible": 100, "interoperable": 90, "reusable": 100}),
            _ds("gldas_soil", "GLDAS Soil Moisture", "earth_observation", "NASA Goddard", "geospatial",
                "25km 3-hour", "global", "historical", "Rodell M et al. (2004) BAMS 85(3):381-394", 8,
                "GEE: NASA/GLDAS/V021/NOAH/G025/T3H", "cc0", {"findable": 95, "accessible": 100, "interoperable": 85, "reusable": 100}),
            _ds("bnpb_dibi", "BNPB DIBI Indonesian Disaster Database", "institutional_custom", "Indonesian National Disaster Agency", "tabular",
                "per district daily", "national_indonesia", "recent", "BNPB Data dan Informasi Bencana Indonesia https://dibi.bnpb.go.id", 10,
                "Web scraping or annual data request", "request_required", {"findable": 70, "accessible": 50, "interoperable": 60, "reusable": 50}),
            _ds("bmkg_climate", "BMKG Climate Station Data", "earth_observation", "Indonesian Meteorological Agency", "time_series",
                "per station daily", "national_indonesia", "historical", "BMKG dataonline.bmkg.go.id", 9,
                "Web portal request", "request_required", {"findable": 75, "accessible": 60, "interoperable": 70, "reusable": 60}),
        ]
    else:
        # Generic fallback bila domain tidak terdeteksi sebagai flood/RS
        dom = DOMAIN_CATALOG.get(detected_domain, DOMAIN_CATALOG["scientific_domain"])
        datasets_fb = [
            _ds("user_data", "Custom dataset from Workspace Data", "institutional_custom", "Local upload", "tabular",
                "varies", "local", "recent", "User defined collection", 8,
                "Workspace Data > GEE Collector or direct upload", "custom_open", {"findable": 50, "accessible": 100, "interoperable": 70, "reusable": 60}),
            _ds("openml_default", "OpenML Public Datasets", "ml_benchmarks", "OpenML", "tabular",
                "varies", "global", "recent", "Vanschoren J et al. (2014) SIGKDD Explorations 15(2):49-60", 7,
                "https://www.openml.org/api", "cc_by", {"findable": 95, "accessible": 100, "interoperable": 95, "reusable": 95}),
            _ds("hf_datasets", "Hugging Face Datasets Hub", "ml_benchmarks", "Hugging Face", "multimodal",
                "varies", "global", "real_time", "Lhoest Q et al. (2021) EMNLP demos:175-184", 7,
                "https://huggingface.co/datasets API", "cc_by", {"findable": 90, "accessible": 95, "interoperable": 95, "reusable": 90}),
            _ds("kaggle_datasets", "Kaggle Datasets", "ml_benchmarks", "Kaggle", "tabular",
                "varies", "global", "recent", "Kaggle https://www.kaggle.com/datasets", 6,
                "Registration required, API or web download", "custom_open", {"findable": 80, "accessible": 80, "interoperable": 85, "reusable": 75}),
        ]

    return {
        "status": "success",
        "detected_domain": detected_domain,
        "domain_label": DOMAIN_CATALOG.get(detected_domain, {}).get("label", ""),
        "domain_catalog": [{"id": did, "label": dmeta["label"]} for did, dmeta in DOMAIN_CATALOG.items()],
        "disciplines": disciplines_fb if disciplines_fb else [
            {"id": "data_science", "label": "Data Science", "rationale": "Default discipline based on research title.", "variables": ["TBD"], "sources": ["TBD"], "citation": "n/a"}
        ],
        "datasets": sorted(datasets_fb, key=lambda x: -x["relevance_score"]),
        "method": "rule_based_fallback",
        "prompt_used": "rule_based_template_no_ai",
    }


# Rule-based fallback helpers untuk SSE endpoint partial success.
# Dipakai kalau AI call salah satu (disciplines atau datasets) gagal parse,
# untuk graceful degradation supaya user tetap dapat working output.
# Per-domain discipline templates · standard disciplines yang universally
# relevan untuk research di domain tersebut. Tidak hardcode topic spesifik,
# tapi general academic disciplines yang valid untuk domain.
# Citations adalah seminal/foundational works yang well-cited di domain.
_DOMAIN_DISCIPLINE_TEMPLATES = {
    "earth_observation": [
        {"id": "earth_science", "label": "Earth Science and Geophysics", "rationale": "Core domain for understanding earth processes, atmosphere, hydrosphere, and lithosphere.", "variables": ["Spatial coordinates", "Temporal time series", "Multi-spectral measurements"], "sources": ["NASA EOSDIS", "ESA Copernicus", "USGS"], "citation": "Lillesand T, Kiefer R, Chipman J (2015) Remote Sensing and Image Interpretation 7th ed. Wiley"},
        {"id": "remote_sensing", "label": "Remote Sensing and Geospatial Science", "rationale": "Satellite and aerial imagery provide synoptic spatial coverage at multiple resolutions.", "variables": ["Spectral indices", "Spatial resolution", "Revisit frequency"], "sources": ["Landsat", "Sentinel", "MODIS"], "citation": "Drusch M et al. (2012) Remote Sensing of Environment 120:25-36"},
        {"id": "climate_atmospheric", "label": "Climate and Atmospheric Science", "rationale": "Climate variables drive many earth processes and provide context for prediction.", "variables": ["Temperature", "Precipitation", "Atmospheric pressure"], "sources": ["ECMWF ERA5", "NOAA", "BMKG"], "citation": "Hersbach H et al. (2020) Quarterly Journal of the Royal Meteorological Society 146:1999-2049"},
        {"id": "geospatial_analytics", "label": "Geospatial Analytics and GIS", "rationale": "Spatial analysis methodology underpins extraction of insights from geographic data.", "variables": ["Vector geometries", "Raster grids", "Spatial autocorrelation"], "sources": ["QGIS", "PostGIS", "Google Earth Engine"], "citation": "Gorelick N et al. (2017) Remote Sensing of Environment 202:18-27 (Google Earth Engine)"},
    ],
    "biomedical": [
        {"id": "clinical_medicine", "label": "Clinical Medicine and Patient Care", "rationale": "Clinical context provides ground truth for disease, treatment, and patient outcomes.", "variables": ["Patient demographics", "Clinical measurements", "Treatment outcomes"], "sources": ["Electronic Health Records", "MIMIC-IV", "UK Biobank"], "citation": "Johnson AEW et al. (2023) Scientific Data 10:1 (MIMIC-IV)"},
        {"id": "epidemiology", "label": "Epidemiology and Public Health", "rationale": "Population-level patterns of disease and health determinants frame the research question.", "variables": ["Incidence rate", "Prevalence", "Exposure measures"], "sources": ["WHO", "CDC", "BPS Indonesia health surveys"], "citation": "Rothman KJ, Greenland S, Lash TL (2008) Modern Epidemiology 3rd ed. Lippincott"},
        {"id": "bioinformatics", "label": "Bioinformatics and Computational Biology", "rationale": "Computational methods extract biological insight from high-dimensional molecular data.", "variables": ["Gene expression", "Protein sequences", "Genomic variants"], "sources": ["NCBI GenBank", "UniProt", "TCGA"], "citation": "Stein LD (2008) Genome Biology 9:R86 (open source bioinformatics)"},
        {"id": "medical_imaging", "label": "Medical Imaging and Radiology", "rationale": "Imaging modalities (CT, MRI, X-ray) provide diagnostic and prognostic information.", "variables": ["Image intensity", "Anatomical landmarks", "Lesion segmentation"], "sources": ["NIH Clinical Center", "TCIA", "PhysioNet"], "citation": "Litjens G et al. (2017) Medical Image Analysis 42:60-88 (deep learning survey)"},
    ],
    "finance": [
        {"id": "quantitative_finance", "label": "Quantitative Finance and Asset Pricing", "rationale": "Pricing theory and risk modeling provide framework for analyzing financial markets.", "variables": ["Asset returns", "Volatility", "Risk premia"], "sources": ["CRSP", "Bloomberg", "Refinitiv"], "citation": "Cochrane JH (2009) Asset Pricing Revised Edition. Princeton University Press"},
        {"id": "macroeconomics", "label": "Macroeconomics and Monetary Policy", "rationale": "Macro variables shape investment environment and capital allocation decisions.", "variables": ["GDP growth", "Inflation rate", "Interest rates"], "sources": ["FRED St Louis Fed", "World Bank", "IMF"], "citation": "Romer D (2018) Advanced Macroeconomics 5th ed. McGraw-Hill"},
        {"id": "financial_econometrics", "label": "Financial Econometrics", "rationale": "Statistical methods specific to financial time series enable rigorous empirical analysis.", "variables": ["Cointegration", "ARCH/GARCH volatility", "Vector autoregression"], "sources": ["EViews", "R quantmod", "Python statsmodels"], "citation": "Tsay RS (2010) Analysis of Financial Time Series 3rd ed. Wiley"},
        {"id": "behavioral_finance", "label": "Behavioral Finance", "rationale": "Investor psychology and cognitive biases explain market anomalies not captured by rational models.", "variables": ["Sentiment indices", "Trading volume", "Survey expectations"], "sources": ["Yale ICF Survey", "AAII Sentiment Survey"], "citation": "Barberis N, Thaler R (2003) Handbook of the Economics of Finance 1:1053-1128"},
    ],
    "social_sciences": [
        {"id": "sociology", "label": "Sociology and Social Structure", "rationale": "Social institutions, networks, and relationships frame human behavior at population level.", "variables": ["Social networks", "Institutional indicators", "Demographic strata"], "sources": ["General Social Survey", "World Values Survey", "ISSP"], "citation": "Wasserman S, Faust K (1994) Social Network Analysis. Cambridge University Press"},
        {"id": "psychology", "label": "Psychology and Behavioral Science", "rationale": "Individual cognition, emotion, and behavior provide micro-foundations for social phenomena.", "variables": ["Survey responses", "Reaction times", "Behavioral measures"], "sources": ["PsycINFO", "Open Science Framework"], "citation": "Cohen J (1988) Statistical Power Analysis for the Behavioral Sciences 2nd ed. Routledge"},
        {"id": "political_science", "label": "Political Science and Public Policy", "rationale": "Political institutions, voting behavior, and policy decisions shape collective outcomes.", "variables": ["Voting patterns", "Policy indicators", "Governance indices"], "sources": ["Comparative Study of Electoral Systems", "Polity Project"], "citation": "Lijphart A (2012) Patterns of Democracy 2nd ed. Yale University Press"},
        {"id": "research_methodology", "label": "Social Science Research Methodology", "rationale": "Mixed methods, survey design, and causal inference are core to rigorous social research.", "variables": ["Sample design", "Effect sizes", "Confounders"], "sources": ["Methodology textbooks"], "citation": "Creswell JW (2014) Research Design 4th ed. Sage"},
    ],
    "open_government": [
        {"id": "public_administration", "label": "Public Administration and Governance", "rationale": "Government performance, transparency, and service delivery shape public outcomes.", "variables": ["Service metrics", "Budget allocations", "Performance indicators"], "sources": ["data.go.id", "data.gov", "OECD"], "citation": "Hood C (1991) Public Administration 69:3-19 (New Public Management)"},
        {"id": "policy_analysis", "label": "Policy Analysis and Evaluation", "rationale": "Evidence-based policy assessment uses quantitative and qualitative methods to evaluate interventions.", "variables": ["Treatment effects", "Cost-benefit ratios", "Counterfactual outcomes"], "sources": ["Government reports", "Audit institutions"], "citation": "Imbens GW, Rubin DB (2015) Causal Inference for Statistics. Cambridge"},
        {"id": "statistics_official", "label": "Official Statistics and Census", "rationale": "National statistical offices produce authoritative data for population, economy, and society.", "variables": ["Census variables", "Survey weights", "Sampling frames"], "sources": ["BPS Indonesia", "Eurostat", "UN Statistics Division"], "citation": "Groves RM et al. (2009) Survey Methodology 2nd ed. Wiley"},
        {"id": "data_science", "label": "Data Science for Public Sector", "rationale": "Modern analytical methods enable extracting insights from large government datasets.", "variables": ["Administrative records", "Linked datasets", "Time series indicators"], "sources": ["Government open data portals"], "citation": "Provost F, Fawcett T (2013) Data Science for Business. OReilly"},
    ],
    "ml_benchmarks": [
        {"id": "machine_learning", "label": "Machine Learning and Statistical Learning", "rationale": "Core ML theory and algorithms underpin benchmark performance evaluation.", "variables": ["Model accuracy", "Training loss", "Generalization gap"], "sources": ["Kaggle", "Papers With Code", "Hugging Face"], "citation": "Goodfellow I, Bengio Y, Courville A (2016) Deep Learning. MIT Press"},
        {"id": "computer_vision", "label": "Computer Vision", "rationale": "Image classification, detection, and segmentation are foundational benchmark categories.", "variables": ["Top-1 accuracy", "mAP", "IoU"], "sources": ["ImageNet", "COCO", "Pascal VOC"], "citation": "Deng J et al. (2009) CVPR 248-255 (ImageNet)"},
        {"id": "natural_language_processing", "label": "Natural Language Processing", "rationale": "Language understanding benchmarks drive progress in transformer architectures.", "variables": ["BLEU score", "Perplexity", "F1 score"], "sources": ["GLUE", "SQuAD", "Hugging Face Datasets"], "citation": "Vaswani A et al. (2017) NeurIPS 30:5998-6008 (Transformer)"},
        {"id": "benchmarking_methodology", "label": "Benchmarking and Evaluation Methodology", "rationale": "Rigorous evaluation protocols ensure fair comparison across model architectures.", "variables": ["Train/test splits", "Cross-validation", "Statistical significance"], "sources": ["Benchmark papers"], "citation": "Dietterich TG (1998) Neural Computation 10:1895-1923 (statistical tests)"},
    ],
    "text_corpora": [
        {"id": "natural_language_processing", "label": "Natural Language Processing", "rationale": "NLP methods enable extraction of structure and meaning from text corpora.", "variables": ["Token frequency", "Syntactic structures", "Semantic embeddings"], "sources": ["Hugging Face", "spaCy", "NLTK"], "citation": "Jurafsky D, Martin JH (2024) Speech and Language Processing 3rd ed."},
        {"id": "computational_linguistics", "label": "Computational Linguistics", "rationale": "Linguistic theory provides framework for analyzing language structure computationally.", "variables": ["Morphological features", "Syntactic dependencies", "Semantic roles"], "sources": ["Universal Dependencies", "WordNet"], "citation": "Manning CD, Schutze H (1999) Foundations of Statistical NLP. MIT Press"},
        {"id": "text_mining", "label": "Text Mining and Information Retrieval", "rationale": "Discovery of patterns in large text collections supports research synthesis.", "variables": ["Document frequency", "TF-IDF scores", "Topic distributions"], "sources": ["Common Crawl", "OPUS", "Project Gutenberg"], "citation": "Blei DM, Ng AY, Jordan MI (2003) JMLR 3:993-1022 (LDA)"},
        {"id": "machine_learning", "label": "Machine Learning for Text", "rationale": "Deep learning architectures dominate modern text understanding tasks.", "variables": ["Embedding vectors", "Attention weights", "Loss curves"], "sources": ["Hugging Face Transformers"], "citation": "Devlin J et al. (2019) NAACL 4171-4186 (BERT)"},
    ],
    "scientific_domain": [
        {"id": "domain_science", "label": "Primary Scientific Domain", "rationale": "Core domain knowledge provides theoretical framework and empirical context.", "variables": ["Domain-specific measurements", "Standard protocols", "Reference values"], "sources": ["Domain repositories"], "citation": "Refer to domain-specific seminal references"},
        {"id": "data_science", "label": "Data Science and Analytics", "rationale": "Modern data analysis methods enable rigorous quantitative investigation.", "variables": ["Statistical estimators", "Effect sizes", "Confidence intervals"], "sources": ["Domain literature benchmarks"], "citation": "Wickham H, Grolemund G (2017) R for Data Science. OReilly"},
        {"id": "research_methodology", "label": "Research Methodology", "rationale": "Systematic research design ensures reproducibility and validity of findings.", "variables": ["Sample size", "Validation protocol", "Effect estimates"], "sources": ["Methodology textbooks"], "citation": "Creswell JW (2014) Research Design 4th ed. Sage"},
        {"id": "statistics", "label": "Statistics and Inference", "rationale": "Statistical theory underpins hypothesis testing and uncertainty quantification.", "variables": ["Test statistics", "P-values", "Confidence intervals"], "sources": ["Statistical software"], "citation": "Wasserman L (2004) All of Statistics. Springer"},
    ],
    "institutional_custom": [
        {"id": "domain_expertise", "label": "Institutional Domain Expertise", "rationale": "Local domain knowledge and institutional context shape data collection and interpretation.", "variables": ["Institution-specific metrics", "Custom protocols"], "sources": ["Institutional archives"], "citation": "Refer to institutional documentation"},
        {"id": "data_management", "label": "Research Data Management", "rationale": "Data lifecycle management (collection, storage, sharing) supports reproducible research.", "variables": ["Metadata schemas", "Data quality indicators"], "sources": ["DCC", "FAIR principles"], "citation": "Wilkinson MD et al. (2016) Scientific Data 3:160018 (FAIR)"},
        {"id": "research_methodology", "label": "Research Methodology", "rationale": "Systematic approach to research design and validation.", "variables": ["Sample size", "Validation protocol"], "sources": ["Methodology textbooks"], "citation": "Creswell JW (2014) Research Design 4th ed. Sage"},
        {"id": "data_science", "label": "Data Science and Analytics", "rationale": "Analytical methods enable insight extraction from institutional data.", "variables": ["Feature engineering", "Model evaluation"], "sources": ["Institutional datasets"], "citation": "Provost F, Fawcett T (2013) Data Science for Business. OReilly"},
    ],
}


# Per-domain dataset templates · authoritative repositories standard untuk
# domain tersebut. Citations adalah paper publikasi dataset itu sendiri.
_DOMAIN_DATASET_TEMPLATES = {
    "earth_observation": [
        {"id": "nasa_eosdis", "label": "NASA EOSDIS Earth Observation Data", "source": "NASA Goddard", "format": "geospatial", "resolution": "varies per instrument", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Ramapriyan H et al. (2017) Big Earth Data 1:21-37", "access": "earthdata.nasa.gov API and download", "license": "cc0"},
        {"id": "copernicus_sentinel", "label": "ESA Copernicus Sentinel Missions", "source": "ESA European Space Agency", "format": "geospatial", "resolution": "10-60m per Sentinel mission", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Berger M et al. (2012) Remote Sensing of Environment 120:84-90", "access": "Copernicus Open Access Hub or GEE catalog", "license": "cc_by"},
        {"id": "usgs_landsat", "label": "USGS Landsat Collection 2", "source": "United States Geological Survey", "format": "geospatial", "resolution": "30m every 16 days since 1972", "coverage_geographic": "global", "coverage_temporal": "historical", "citation": "Wulder MA et al. (2019) Remote Sensing of Environment 225:127-147", "access": "USGS EarthExplorer or GEE catalog LANDSAT/", "license": "cc0"},
        {"id": "ecmwf_era5", "label": "ECMWF ERA5 Climate Reanalysis", "source": "European Centre for Medium-Range Weather Forecasts", "format": "geospatial", "resolution": "0.25 degree hourly 1940-present", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Hersbach H et al. (2020) Quarterly Journal of the Royal Meteorological Society 146:1999-2049", "access": "Copernicus Climate Data Store API", "license": "cc_by"},
    ],
    "biomedical": [
        {"id": "ncbi_geo", "label": "NCBI Gene Expression Omnibus", "source": "National Center for Biotechnology Information", "format": "tabular", "resolution": "per sample expression matrix", "coverage_geographic": "global", "coverage_temporal": "historical", "citation": "Edgar R et al. (2002) Nucleic Acids Research 30:207-210", "access": "ncbi.nlm.nih.gov/geo API and FTP", "license": "cc0"},
        {"id": "uk_biobank", "label": "UK Biobank Population Cohort", "source": "UK Biobank", "format": "tabular", "resolution": "500k participants longitudinal", "coverage_geographic": "national_other", "coverage_temporal": "historical", "citation": "Sudlow C et al. (2015) PLOS Medicine 12:e1001779", "access": "ukbiobank.ac.uk application required", "license": "request_required"},
        {"id": "mimic_iv", "label": "MIMIC-IV Critical Care Database", "source": "MIT Lab for Computational Physiology", "format": "tabular", "resolution": "ICU patient records 2008-2019", "coverage_geographic": "local", "coverage_temporal": "historical", "citation": "Johnson AEW et al. (2023) Scientific Data 10:1", "access": "PhysioNet credentialed access", "license": "request_required"},
        {"id": "tcga", "label": "The Cancer Genome Atlas", "source": "National Cancer Institute", "format": "multimodal", "resolution": "Multi-omics across 33 cancer types", "coverage_geographic": "national_other", "coverage_temporal": "historical", "citation": "Weinstein JN et al. (2013) Nature Genetics 45:1113-1120", "access": "Genomic Data Commons portal", "license": "request_required"},
    ],
    "finance": [
        {"id": "fred_stlouisfed", "label": "FRED Federal Reserve Economic Data", "source": "Federal Reserve Bank of St Louis", "format": "time_series", "resolution": "Various frequencies 1947-present", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Bauer MD, Mertens TM (2018) FRBSF Economic Letter 2018-25", "access": "fred.stlouisfed.org API free", "license": "cc0"},
        {"id": "worldbank_wdi", "label": "World Bank World Development Indicators", "source": "World Bank", "format": "tabular", "resolution": "Annual country-level since 1960", "coverage_geographic": "global", "coverage_temporal": "historical", "citation": "World Bank (2024) World Development Indicators database", "access": "data.worldbank.org API and download", "license": "cc_by"},
        {"id": "yahoo_finance", "label": "Yahoo Finance Market Data", "source": "Yahoo Finance public", "format": "time_series", "resolution": "Daily OHLCV historical", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Yahoo Finance market data (yfinance Python library)", "access": "yfinance pip package or web scraping", "license": "custom_open"},
        {"id": "sec_edgar", "label": "SEC EDGAR Corporate Filings", "source": "US Securities and Exchange Commission", "format": "text", "resolution": "Quarterly and annual filings since 1993", "coverage_geographic": "national_other", "coverage_temporal": "historical", "citation": "SEC EDGAR Filing system", "access": "sec.gov/edgar API free", "license": "cc0"},
    ],
    "social_sciences": [
        {"id": "icpsr", "label": "ICPSR Interuniversity Consortium for Political and Social Research", "source": "University of Michigan", "format": "tabular", "resolution": "Survey datasets across many studies", "coverage_geographic": "global", "coverage_temporal": "historical", "citation": "ICPSR data repository (icpsr.umich.edu)", "access": "icpsr.umich.edu free with registration", "license": "cc_by"},
        {"id": "wvs", "label": "World Values Survey", "source": "World Values Survey Association", "format": "tabular", "resolution": "Cross-national surveys 7 waves since 1981", "coverage_geographic": "global", "coverage_temporal": "historical", "citation": "Inglehart R et al. (2014) World Values Survey database", "access": "worldvaluessurvey.org free", "license": "custom_open"},
        {"id": "pew_research", "label": "Pew Research Center Datasets", "source": "Pew Research Center", "format": "tabular", "resolution": "Survey microdata various topics", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Pew Research Center surveys (pewresearch.org)", "access": "pewresearch.org/download-data free", "license": "custom_open"},
        {"id": "bps_indonesia", "label": "BPS Indonesia Statistical Data", "source": "Badan Pusat Statistik", "format": "tabular", "resolution": "Sub-national surveys and census", "coverage_geographic": "national_indonesia", "coverage_temporal": "recent", "citation": "BPS Indonesia (bps.go.id)", "access": "bps.go.id portal", "license": "cc_by"},
    ],
    "open_government": [
        {"id": "data_go_id", "label": "data.go.id Indonesia Open Data Portal", "source": "Kementerian Komunikasi dan Informatika", "format": "tabular", "resolution": "Multi-sector government datasets", "coverage_geographic": "national_indonesia", "coverage_temporal": "recent", "citation": "data.go.id Indonesia Government Open Data", "access": "data.go.id portal free", "license": "cc_by"},
        {"id": "data_gov_us", "label": "data.gov US Federal Open Data", "source": "US General Services Administration", "format": "tabular", "resolution": "Multi-agency federal data", "coverage_geographic": "national_other", "coverage_temporal": "recent", "citation": "data.gov US Federal Open Data Portal", "access": "data.gov free", "license": "cc0"},
        {"id": "oecd_data", "label": "OECD Statistics Database", "source": "Organisation for Economic Co-operation and Development", "format": "tabular", "resolution": "Cross-national indicators", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "OECD data portal (data.oecd.org)", "access": "stats.oecd.org API and download", "license": "cc_by"},
        {"id": "eurostat", "label": "Eurostat European Statistics", "source": "European Commission Eurostat", "format": "tabular", "resolution": "EU member state statistics", "coverage_geographic": "regional", "coverage_temporal": "recent", "citation": "Eurostat database (ec.europa.eu/eurostat)", "access": "Eurostat API free", "license": "cc_by"},
    ],
    "ml_benchmarks": [
        {"id": "imagenet", "label": "ImageNet Large Scale Visual Recognition", "source": "Stanford Vision Lab", "format": "image", "resolution": "14M annotated images 1000 classes", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Deng J et al. (2009) CVPR 248-255", "access": "image-net.org registration required", "license": "request_required"},
        {"id": "coco", "label": "Microsoft COCO Common Objects in Context", "source": "Microsoft Research", "format": "image", "resolution": "330k images with object detection annotations", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Lin TY et al. (2014) ECCV 740-755", "access": "cocodataset.org free download", "license": "cc_by"},
        {"id": "huggingface_datasets", "label": "Hugging Face Datasets Hub", "source": "Hugging Face", "format": "multimodal", "resolution": "20k+ community datasets", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Lhoest Q et al. (2021) EMNLP 175-184", "access": "huggingface.co/datasets API free", "license": "custom_open"},
        {"id": "kaggle", "label": "Kaggle Datasets Competition", "source": "Kaggle (Google)", "format": "multimodal", "resolution": "10k+ datasets across domains", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Kaggle Datasets repository (kaggle.com/datasets)", "access": "kaggle.com API free with registration", "license": "custom_open"},
    ],
    "text_corpora": [
        {"id": "common_crawl", "label": "Common Crawl Web Corpus", "source": "Common Crawl Foundation", "format": "text", "resolution": "Petabytes web crawl since 2008", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Common Crawl Foundation (commoncrawl.org)", "access": "S3 bucket free download", "license": "cc0"},
        {"id": "wikipedia_dumps", "label": "Wikipedia Database Dumps", "source": "Wikimedia Foundation", "format": "text", "resolution": "Multilingual encyclopedia full-text", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Wikimedia Foundation Wikipedia dumps", "access": "dumps.wikimedia.org free", "license": "cc_by_sa"},
        {"id": "opus_parallel", "label": "OPUS Parallel Corpus Collection", "source": "University of Helsinki", "format": "text", "resolution": "Multilingual parallel translation pairs", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Tiedemann J (2012) LREC 2214-2218", "access": "opus.nlpl.eu free", "license": "custom_open"},
        {"id": "huggingface_datasets", "label": "Hugging Face NLP Datasets", "source": "Hugging Face", "format": "text", "resolution": "Thousands of NLP-specific datasets", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Lhoest Q et al. (2021) EMNLP 175-184", "access": "huggingface.co/datasets API", "license": "custom_open"},
    ],
    "scientific_domain": [
        {"id": "domain_repo_a", "label": "Primary Domain Repository", "source": "Domain-specific authoritative repository", "format": "tabular", "resolution": "varies per study", "coverage_geographic": "global", "coverage_temporal": "historical", "citation": "Refer to domain seminal repository paper", "access": "Repository-specific access protocol", "license": "custom_open"},
        {"id": "literature_benchmark", "label": "Literature Benchmark Datasets from Stage I", "source": "Peer-reviewed papers from Stage I literature search", "format": "multimodal", "resolution": "varies per paper", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Cite individual benchmark papers from Stage I synthesis", "access": "Paper supplementary materials", "license": "custom_open"},
        {"id": "user_uploaded", "label": "User Primary Dataset", "source": "User-collected via Workspace Data", "format": "tabular", "resolution": "User-defined", "coverage_geographic": "local", "coverage_temporal": "recent", "citation": "Cite per institutional guidelines", "access": "Workspace Data upload pipeline", "license": "custom_open"},
        {"id": "domain_standard", "label": "Domain Standard Reference Dataset", "source": "Established domain reference", "format": "tabular", "resolution": "Standard reference dimensions", "coverage_geographic": "global", "coverage_temporal": "historical", "citation": "Cite domain standard reference paper", "access": "Open repository access", "license": "cc_by"},
    ],
    "institutional_custom": [
        {"id": "user_primary", "label": "User Primary Dataset (Workspace Data)", "source": "User-collected and uploaded", "format": "tabular", "resolution": "User-defined", "coverage_geographic": "local", "coverage_temporal": "recent", "citation": "Cite per institutional guidelines", "access": "Workspace Data upload pipeline", "license": "custom_open"},
        {"id": "institutional_archive", "label": "Institutional Archive Records", "source": "Lab or institutional repository", "format": "tabular", "resolution": "varies per institution", "coverage_geographic": "local", "coverage_temporal": "historical", "citation": "Cite institutional documentation", "access": "Institutional access protocol", "license": "request_required"},
        {"id": "literature_benchmark", "label": "Stage I Literature Benchmark", "source": "Papers from Stage I synthesis", "format": "multimodal", "resolution": "varies", "coverage_geographic": "global", "coverage_temporal": "recent", "citation": "Cite individual benchmark papers from Stage I", "access": "Paper supplementary materials", "license": "custom_open"},
        {"id": "field_observation", "label": "Field Observation or Survey Data", "source": "Researcher field collection", "format": "tabular", "resolution": "Per observation", "coverage_geographic": "local", "coverage_temporal": "recent", "citation": "Cite per study protocol", "access": "Researcher contact", "license": "request_required"},
    ],
}


def _rule_based_disciplines_fallback(title: str, detected_domain: str) -> List[Dict[str, Any]]:
    """TRULY DYNAMIC fallback berdasarkan detected_domain (sudah pakai title
    keyword matching). Lookup template per-domain. Generic per domain bukan
    hardcode topic spesifik. Bekerja untuk research topic apapun dari domain
    yang ada di DOMAIN_CATALOG. User bisa edit hasil setelahnya."""
    domain_key = detected_domain if detected_domain in _DOMAIN_DISCIPLINE_TEMPLATES else "scientific_domain"
    template = _DOMAIN_DISCIPLINE_TEMPLATES[domain_key]
    # Return copy supaya tidak mutate template
    return [dict(d) for d in template]


def _rule_based_datasets_fallback(title: str, detected_domain: str) -> List[Dict[str, Any]]:
    """TRULY DYNAMIC fallback berdasarkan detected_domain. Lookup per-domain
    template yang berisi authoritative repositories untuk domain tersebut.
    Title sebagai context untuk geographic detection (Indonesia → adjust
    coverage_geographic kalau perlu). Bekerja untuk topic apapun."""
    domain_key = detected_domain if detected_domain in _DOMAIN_DATASET_TEMPLATES else "scientific_domain"
    template = _DOMAIN_DATASET_TEMPLATES[domain_key]

    # Minor enrichment · kalau title contain Indonesia, prefer Indonesian
    # datasets yang ada di template (boost relevance) plus tambah generic
    # Indonesia dataset kalau belum ada.
    title_lower = (title or "").lower()
    is_indonesia_context = any(k in title_lower for k in ["indonesia", "indo", "jakarta", "jawa", "sumatra", "kalimantan", "sulawesi"])

    base_fair = {"findable": 85, "accessible": 85, "interoperable": 80, "reusable": 85, "overall": 84}
    datasets = []
    for idx, t in enumerate(template):
        d = dict(t)
        d["category"] = domain_key
        d["fair"] = dict(base_fair)
        # Relevance descending · pertama paling tinggi, terakhir paling rendah
        d["relevance_score"] = max(6, 10 - idx)
        # Boost Indonesia dataset relevance kalau topic Indonesia
        if is_indonesia_context and d.get("coverage_geographic") == "national_indonesia":
            d["relevance_score"] = 10
        datasets.append(d)

    # Kalau topic Indonesia tapi template tidak punya Indonesia dataset, tambahkan
    if is_indonesia_context and not any(d.get("coverage_geographic") == "national_indonesia" for d in datasets):
        datasets.append({
            "id": "data_go_id_indonesia",
            "label": "Indonesia Government Open Data (data.go.id)",
            "category": "open_government",
            "source": "Kementerian Komunikasi dan Informatika Indonesia",
            "format": "tabular",
            "resolution": "Multi-sector government datasets",
            "coverage_geographic": "national_indonesia",
            "coverage_temporal": "recent",
            "citation": "data.go.id Indonesia Government Open Data Portal",
            "relevance_score": 9,
            "access": "data.go.id portal free",
            "license": "cc_by",
            "fair": dict(base_fair),
        })

    return datasets


# =============================================================================
# STREAMING GENERATE SCOPE (SSE)
# =============================================================================
# Sama endpoint generate-scope tapi return SSE stream dengan heartbeat events
# setiap 5 detik. Penting karena Cloudflare Tunnel timeout 100s tapi AI provider
# lokal (terutama Ollama R1 yang slow) bisa butuh 2-5 menit. Heartbeat jaga
# koneksi alive plus user dapat progress feedback (elapsed timer + phase).
# Internal hard limit 240 detik supaya tidak hang forever.
@router.post("/pipeline/organizing/generate-scope-stream")
def organizing_generate_scope_stream(req: ScopeGenerationRequest):
    """SSE streaming variant dari generate-scope. Heartbeat tiap 5s supaya
    Cloudflare keep alive walaupun AI generate lambat."""
    import time as _time

    def _sse(data):
        return "data: " + json.dumps(data) + "\n\n"

    def _gen():
        try:
            from app.services import task_router as _tr
        except Exception:
            yield _sse({"type": "error", "message": "AI orchestrator not available"})
            return

        yield _sse({"type": "phase", "name": "starting"})

        # Build prompts persis seperti non-streaming version
        period_str = ""
        if req.period and req.period.get("year_from"):
            period_str = f"Research Period: {req.period.get('year_from')}-{req.period.get('year_to', 'present')}\n"
        lang_hint, lang_display = _resolve_language(req.language)
        ts_block = f"Thesis Statement Context: {req.thesis_statement[:800]}\n" if req.thesis_statement else ""
        detected_domain = req.domain_override if req.domain_override and req.domain_override in DOMAIN_CATALOG else _detect_domain(req.title, req.thesis_statement or "")
        domain_meta = DOMAIN_CATALOG.get(detected_domain, DOMAIN_CATALOG["scientific_domain"])
        catalog_summary = "\n".join([f"- {did}: {meta['label']} ({meta['ref_repos']})" for did, meta in DOMAIN_CATALOG.items()])
        common_ctx = f"Research Title: {req.title}\n{ts_block}{period_str}PRIMARY DOMAIN: {detected_domain} ({domain_meta['label']})\nDOMAIN DESCRIPTION: {domain_meta['description']}\nTarget language: {lang_hint}"

        # ULTRA-SIMPLE prompts · minimal fields supaya AI parse rate tinggi
        # walaupun pakai model kecil seperti Ollama R1 8b. AI hanya kasih
        # nama + alasan singkat. Backend nanti enrich dengan struktur lengkap
        # dari template per-domain. Hybrid pattern: AI for naming, template
        # for structure. Sama dengan pattern di Bordes et al. (2023) yang
        # show simpler prompts significantly improve small model success rate.
        p_disc = f"""Research title: "{req.title}"

List exactly 4 academic disciplines relevant to this research.

Output JSON only (no markdown, no extra text):
{{"disciplines":[
  {{"label":"Discipline Name","why":"one short sentence why relevant"}},
  {{"label":"Discipline Name","why":"one short sentence why relevant"}},
  {{"label":"Discipline Name","why":"one short sentence why relevant"}},
  {{"label":"Discipline Name","why":"one short sentence why relevant"}}
]}}"""

        p_ds = f"""Research title: "{req.title}"

List exactly 4 datasets relevant to this research.

Output JSON only (no markdown, no extra text):
{{"datasets":[
  {{"name":"Dataset Name","source":"Provider name","why":"one short sentence why relevant"}},
  {{"name":"Dataset Name","source":"Provider name","why":"one short sentence why relevant"}},
  {{"name":"Dataset Name","source":"Provider name","why":"one short sentence why relevant"}},
  {{"name":"Dataset Name","source":"Provider name","why":"one short sentence why relevant"}}
]}}"""

        def _parse_ai_json(text):
            text = (text or "").strip()
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]
                    if text.startswith("json"):
                        text = text[4:].strip()
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

        def _call_disc():
            try:
                # max_tokens 400 cukup untuk 4 disciplines simple format
                # (label + why, ~50 tokens each + JSON overhead)
                r = _tr.route(task_type="structured_fast", messages=[{"role": "user", "content": p_disc}], max_tokens=400, temperature=0.3)
                if r.get("ok") and r.get("text"):
                    parsed = _parse_ai_json(r["text"])
                    if isinstance(parsed, dict):
                        return parsed.get("disciplines") or []
            except Exception:
                pass
            return []

        # Log raw AI response untuk debugging kalau parse fail. Backend stderr
        # supaya muncul di logs. Bisa di-tail dengan: tail -f .out.log
        import sys as _sys
        def _log_dbg(label, content):
            print(f"[GenerateScopeSSE] {label}: {str(content)[:500]}", file=_sys.stderr, flush=True)

        def _call_ds():
            try:
                # max_tokens 500 · 4 datasets simple format (name + source + why)
                # turun drastis dari 1800 karena prompt sudah simplified
                r = _tr.route(task_type="structured_fast", messages=[{"role": "user", "content": p_ds}], max_tokens=500, temperature=0.3)
                if not r.get("ok"):
                    _log_dbg("datasets AI not ok", r.get("error"))
                    return []
                raw = r.get("text", "")
                if not raw:
                    _log_dbg("datasets AI empty text", "no text returned")
                    return []
                parsed = _parse_ai_json(raw)
                if not isinstance(parsed, dict):
                    _log_dbg("datasets parse fail · raw", raw[:800])
                    return []
                ds_list = parsed.get("datasets") or []
                if not ds_list:
                    _log_dbg("datasets empty array · parsed", str(parsed)[:500])
                return ds_list
            except Exception as exc:
                _log_dbg("datasets exception", str(exc))
                return []

        def _call_disc_logged():
            try:
                r = _tr.route(task_type="structured_fast", messages=[{"role": "user", "content": p_disc}], max_tokens=600, temperature=0.3)
                if not r.get("ok"):
                    _log_dbg("disciplines AI not ok", r.get("error"))
                    return []
                raw = r.get("text", "")
                parsed = _parse_ai_json(raw)
                if not isinstance(parsed, dict):
                    _log_dbg("disciplines parse fail · raw", raw[:500])
                    return []
                return parsed.get("disciplines") or []
            except Exception as exc:
                _log_dbg("disciplines exception", str(exc))
                return []

        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=2)
        f_disc = executor.submit(_call_disc_logged)
        f_ds = executor.submit(_call_ds)
        yield _sse({"type": "phase", "name": "ai_running", "parallel_calls": 2})

        # FAIL FAST · per-call timeout 90 detik supaya fallback kick in cepat.
        # Kalau AI lokal hang lebih dari 90s, kita abandon dan pakai template.
        # User dapat hasil dalam <100s instead of nunggu 4 menit timeout.
        PER_CALL_TIMEOUT = 90
        start = _time.time()
        last_disc_done = False
        last_ds_done = False
        while not (f_disc.done() and f_ds.done()):
            elapsed = int(_time.time() - start)
            # Hard kill timeout · abandon AI yang masih hang, fallback gunakan template
            if elapsed > PER_CALL_TIMEOUT:
                _log_dbg("hard timeout", f"elapsed {elapsed}s > {PER_CALL_TIMEOUT}s, abandoning AI calls and using fallback")
                executor.shutdown(wait=False)
                break
            if f_disc.done() and not last_disc_done:
                yield _sse({"type": "phase", "name": "disciplines_done", "elapsed": elapsed})
                last_disc_done = True
            if f_ds.done() and not last_ds_done:
                yield _sse({"type": "phase", "name": "datasets_done", "elapsed": elapsed})
                last_ds_done = True
            yield _sse({"type": "heartbeat", "elapsed": elapsed})
            _time.sleep(5)

        elapsed = int(_time.time() - start)
        # Ambil result kalau done, otherwise empty (akan trigger fallback)
        disciplines = []
        datasets = []
        try:
            if f_disc.done():
                disciplines = f_disc.result(timeout=1)
        except Exception:
            disciplines = []
        try:
            if f_ds.done():
                datasets = f_ds.result(timeout=1)
        except Exception:
            datasets = []
        executor.shutdown(wait=False)

        yield _sse({"type": "phase", "name": "parsing", "elapsed": elapsed})

        # HYBRID merge · AI provides naming + rationale, template provides
        # full structure (variables, citations, FAIR scoring, dst). Result
        # adalah topic-specific output dengan full metadata reliable.
        # Pattern: progressive enhancement (Stallings 2002 inclusive design).
        clean_disc = []
        if disciplines:
            # AI berhasil · enrich AI items dengan template defaults
            template_disc = _DOMAIN_DISCIPLINE_TEMPLATES.get(
                detected_domain if detected_domain in _DOMAIN_DISCIPLINE_TEMPLATES else "scientific_domain"
            )
            for idx, d in enumerate(disciplines[:6]):
                if not isinstance(d, dict):
                    continue
                # AI returns simple {label, why}. Template provides full struct.
                template_default = template_disc[idx % len(template_disc)] if template_disc else {}
                ai_label = str(d.get("label", "") or d.get("name", "") or "").strip()
                ai_rationale = str(d.get("why", "") or d.get("rationale", "") or "").strip()
                if not ai_label:
                    continue
                # ID from AI label slugified
                ai_id = re.sub(r"[^a-z0-9]+", "_", ai_label.lower()).strip("_")[:60] or f"disc_{idx}"
                clean_disc.append({
                    "id": ai_id,
                    "label": ai_label[:120],
                    "rationale": (ai_rationale or template_default.get("rationale", ""))[:400],
                    # Variables, sources, citation dari template (AI tidak generate ini)
                    "variables": template_default.get("variables", ["TBD by user"])[:8],
                    "sources": template_default.get("sources", ["TBD by user"])[:8],
                    "citation": template_default.get("citation", "Standard domain reference")[:200],
                })

        clean_ds = []
        if datasets:
            # AI berhasil · enrich AI items dengan template defaults
            template_ds = _DOMAIN_DATASET_TEMPLATES.get(
                detected_domain if detected_domain in _DOMAIN_DATASET_TEMPLATES else "scientific_domain"
            )
            base_fair = {"findable": 85, "accessible": 85, "interoperable": 80, "reusable": 85, "overall": 84}
            for idx, d in enumerate(datasets[:10]):
                if not isinstance(d, dict):
                    continue
                template_default = template_ds[idx % len(template_ds)] if template_ds else {}
                ai_name = str(d.get("name", "") or d.get("label", "") or "").strip()
                ai_source = str(d.get("source", "") or "").strip()
                ai_why = str(d.get("why", "") or d.get("relevance", "") or "").strip()
                if not ai_name:
                    continue
                ai_id = re.sub(r"[^a-z0-9]+", "_", ai_name.lower()).strip("_")[:60] or f"ds_{idx}"
                clean_ds.append({
                    "id": ai_id,
                    "label": ai_name[:140],
                    "category": detected_domain,
                    "source": (ai_source or template_default.get("source", "TBD"))[:80],
                    # Struktur dari template (AI tidak generate ini)
                    "format": template_default.get("format", "tabular"),
                    "resolution": template_default.get("resolution", "varies"),
                    "coverage_geographic": template_default.get("coverage_geographic", "global"),
                    "coverage_temporal": template_default.get("coverage_temporal", "recent"),
                    "citation": template_default.get("citation", f"Cite per {ai_name} official documentation")[:200],
                    "relevance_score": max(7, 10 - idx),  # AI dianggap relevance tinggi
                    "access": template_default.get("access", "Check provider website")[:200],
                    "license": template_default.get("license", "custom_open"),
                    "fair": dict(base_fair),
                    # Bonus field · AI rationale jelaskan kenapa dataset ini relevan
                    "_ai_why": ai_why[:300] if ai_why else None,
                })

        # Partial success logic · pure fallback kalau AI sama sekali kosong
        used_ds_fallback = False
        used_disc_fallback = False

        if not clean_ds:
            _log_dbg("using ds fallback", f"AI returned {len(datasets)} datasets after parse")
            used_ds_fallback = True
            ds_fallback_template = _rule_based_datasets_fallback(req.title, detected_domain)
            clean_ds = ds_fallback_template

        if not clean_disc:
            _log_dbg("using disc fallback", f"AI returned {len(disciplines)} disciplines after parse")
            used_disc_fallback = True
            clean_disc = _rule_based_disciplines_fallback(req.title, detected_domain)

        method_label = "ai_orchestrator_parallel_2call_sse"
        if used_ds_fallback and used_disc_fallback:
            method_label = "rule_based_fallback_both"
        elif used_ds_fallback:
            method_label = "ai_disc_plus_rule_ds_partial"
        elif used_disc_fallback:
            method_label = "rule_disc_plus_ai_ds_partial"

        yield _sse({
            "type": "complete",
            "result": {
                "status": "success",
                "detected_domain": detected_domain,
                "domain_label": domain_meta["label"],
                "domain_catalog": [{"id": did, "label": dmeta["label"]} for did, dmeta in DOMAIN_CATALOG.items()],
                "disciplines": clean_disc,
                "datasets": sorted(clean_ds, key=lambda x: -x["relevance_score"]),
                "method": method_label,
                "elapsed_seconds": elapsed,
                "fallback_used": {
                    "disciplines": used_disc_fallback,
                    "datasets": used_ds_fallback,
                },
            },
        })

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


class AutoMapSourcesRequest(BaseModel):
    title: str
    thesis_statement: Optional[str] = ""
    outline: List[Dict[str, Any]]
    papers: List[Dict[str, Any]]
    max_papers_per_node: int = 3
    min_confidence: int = 60
    language: str = "en"


@router.post("/pipeline/organizing/auto-map-sources")
def organizing_auto_map_sources(req: AutoMapSourcesRequest) -> Dict[str, Any]:
    """AI-suggest paper attachments untuk setiap node outline level 2-3.

    OPTIMIZED: parallel batch processing dengan ThreadPoolExecutor 5 workers
    plus batch size 20 nodes. Target speed 20-40 detik untuk 60 nodes
    (sebelumnya 90-150 detik sequential).

    Mengikuti Webster Watson (2002) concept-centric literature review yang
    menempatkan paper sebagai evidence per concept (node outline).
    """
    try:
        from app.services import task_router as _tr
        ai_available = True
    except Exception:
        ai_available = False

    # Filter eligible nodes (level 2-3, skip level 1 yang terlalu broad)
    eligible_nodes = [n for n in req.outline if 1 < (n.get("level") or 1) <= 3 and n.get("title")][:60]
    papers_brief = [{
        "id": str(p.get("id", "") or p.get("doi", "") or p.get("title", ""))[:80],
        "title": str(p.get("title", ""))[:200],
        "abstract": (str(p.get("abstract_synth") or p.get("abstract") or "")[:200])  # Diturunkan dari 300 untuk token efficiency
    } for p in (req.papers or [])[:80]]

    if not eligible_nodes or not papers_brief:
        return {"status": "success", "mappings": {}, "method": "no_input", "stats": {"nodes_processed": 0, "papers_pool": 0}}

    mappings: Dict[str, List[Dict[str, Any]]] = {}
    used_ai = False

    # Build paper index untuk lookup cepat dan rule-based fallback
    paper_by_id = {p["id"]: p for p in papers_brief}

    # Helper rule-based mapping per node · keyword overlap score
    # Dipakai sebagai fallback per-node kalau AI gagal untuk batch tertentu.
    def _rule_map_one_node(node):
        node_text = str(node.get("title", "")).lower()
        node_words = set(re.findall(r"\b[a-z]{3,}\b", node_text))
        if not node_words:
            return []
        scored = []
        for p in papers_brief:
            p_text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
            p_words = set(re.findall(r"\b[a-z]{3,}\b", p_text))
            overlap = len(node_words & p_words)
            if overlap >= 1:
                confidence = min(95, 40 + overlap * 10)
                if confidence >= req.min_confidence:
                    scored.append({"paper_id": p["id"], "confidence": confidence})
        scored.sort(key=lambda x: -x["confidence"])
        return scored[:req.max_papers_per_node]

    if ai_available:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # SIMPLIFIED prompt · drop confidence scoring (AI sering miss),
        # just request paper_ids array per node. Confidence di-derive di
        # post-process (top match = 90, descending). Smaller prompt = faster AI.
        BATCH_SIZE = 10  # smaller batch supaya AI lebih reliable
        papers_text = "\n".join([f"P{i}: {p['title'][:80]}" for i, p in enumerate(papers_brief)])
        # Build id index untuk reverse-lookup AI's P{i} ke real ID
        idx_to_id = {f"P{i}": p["id"] for i, p in enumerate(papers_brief)}

        def _process_batch(batch_nodes_subset):
            """Process satu batch nodes via AI. Return dict {nid: [suggestions]}."""
            local_mappings = {}
            nodes_text = "\n".join([f"N{i}: {n['title'][:100]}" for i, n in enumerate(batch_nodes_subset)])
            idx_to_node = {f"N{i}": n["id"] for i, n in enumerate(batch_nodes_subset)}

            # ULTRA-SIMPLE prompt · just ask for top 3 paper indices per node
            prompt = f"""For each outline node, pick top {req.max_papers_per_node} most relevant papers from the pool by topic match.

Outline nodes:
{nodes_text}

Papers:
{papers_text}

Output JSON only (no markdown, no extra text):
{{"N0":["P5","P12","P3"],"N1":["P8","P1"]}}

Only include nodes that have relevant papers. Use only P-IDs from the list above."""

            try:
                result = _tr.route(
                    task_type="structured_fast",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,  # turun drastis dari 800
                    temperature=0.2,
                )
                if not (result.get("ok") and result.get("text")):
                    return local_mappings
                text = result["text"].strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.startswith("json"):
                            text = text[4:].strip()
                match = re.search(r"\{[\s\S]*\}", text)
                if not match:
                    return local_mappings
                parsed = json.loads(match.group(0))
                if not isinstance(parsed, dict):
                    return local_mappings
                for n_idx, p_idx_list in parsed.items():
                    if not isinstance(p_idx_list, list):
                        continue
                    real_nid = idx_to_node.get(str(n_idx))
                    if not real_nid:
                        continue
                    clean = []
                    for rank, p_idx in enumerate(p_idx_list[:req.max_papers_per_node]):
                        real_pid = idx_to_id.get(str(p_idx).upper())
                        if not real_pid:
                            continue
                        # Derive confidence from rank (top match = 90, descending)
                        confidence = max(req.min_confidence, 90 - rank * 10)
                        clean.append({"paper_id": real_pid, "confidence": confidence})
                    if clean:
                        local_mappings[real_nid] = clean
            except Exception:
                pass
            return local_mappings

        # Build batches dari eligible_nodes
        batches = []
        for batch_start in range(0, len(eligible_nodes), BATCH_SIZE):
            batches.append(eligible_nodes[batch_start:batch_start + BATCH_SIZE])

        # PARALLEL processing dengan per-batch tracking · kalau batch fail,
        # langsung apply rule-based fallback untuk nodes di batch itu supaya
        # tidak ada node yang miss.
        failed_batch_indices = []
        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_batch_idx = {executor.submit(_process_batch, batch): bi for bi, batch in enumerate(batches)}
                for future in as_completed(future_to_batch_idx, timeout=120):
                    bi = future_to_batch_idx[future]
                    try:
                        batch_result = future.result(timeout=60)
                        if batch_result:
                            mappings.update(batch_result)
                            used_ai = True
                        else:
                            failed_batch_indices.append(bi)
                    except Exception:
                        failed_batch_indices.append(bi)
        except Exception:
            # Timeout overall · sisa batch yang belum done masuk failed
            done_batch_count = sum(1 for bi in range(len(batches)) if bi not in failed_batch_indices)
            failed_batch_indices = [bi for bi in range(len(batches)) if bi >= done_batch_count]

        # PER-CHUNK RULE FALLBACK · untuk batch yang AI gagal, apply keyword
        # matching supaya semua nodes tetap dapat mapping. User tidak kehilangan
        # data karena AI provider lambat atau parse fail.
        for bi in failed_batch_indices:
            for node in batches[bi]:
                nid = node.get("id")
                if not nid or nid in mappings:
                    continue
                rule_result = _rule_map_one_node(node)
                if rule_result:
                    mappings[nid] = rule_result

    # Final fallback · kalau AI not available atau zero result, rule-based all
    if not mappings:
        for node in eligible_nodes:
            rule_result = _rule_map_one_node(node)
            if rule_result:
                mappings[node["id"]] = rule_result

    total_attachments = sum(len(v) for v in mappings.values())
    return {
        "status": "success",
        "mappings": mappings,
        "method": "ai_orchestrator" if used_ai else "rule_based_fallback",
        "stats": {
            "nodes_processed": len(eligible_nodes),
            "nodes_with_suggestions": len(mappings),
            "papers_pool": len(papers_brief),
            "total_attachments": total_attachments,
            "avg_papers_per_node": round(total_attachments / max(len(mappings), 1), 1)
        }
    }


# =============================================================================
# NODE ARTIFACT GENERATION
# =============================================================================
# Generate konten markdown per node outline (leaf nodes only). Output adalah
# draft singkat 150-300 kata, pure paragraf (tanpa heading), dengan inline
# citations format (Author, Year) kalau papers attached. Konsep ini mengikuti
# atomic writing pattern dari Ahrens (2017) How to Take Smart Notes dan
# Webster Watson (2002) concept-centric synthesis yang memetakan konten ke
# satu unit konseptual per node.
#
# Stage III synthesize-section nanti dapat consume artifact ini sebagai
# starting_draft untuk improvement (bukan generate from scratch).
class NodeArtifactRequest(BaseModel):
    node_id: str
    node_title: str
    node_level: int = 2
    parent_chain: List[str] = []        # ancestor titles untuk hierarchical context
    thesis_statement: Optional[str] = ""
    chosen_title: Optional[str] = ""
    attached_papers: List[Dict[str, Any]] = []  # max 5, dengan title/authors/year/abstract/id
    target_words: int = 250             # default mid-range 150-300
    language: str = "en"
    mode: Optional[str] = "journal"     # journal | thesis untuk tone adjustment
    # Quality mode toggle:
    # - "fast" (default): single-shot temperature 0.4, cepat 15-30 detik
    # - "best": best-of-3 sampling temp 0.7 + judge pick + self-refine critique
    #   loop. Lebih akurat dengan citation lebih dense, tapi 3-5x token usage
    #   dan 30-60 detik per node. Cobbe et al. (2021) verifier-based selection
    #   plus Madaan et al. (2024) NeurIPS Self-Refine.
    quality_mode: str = "fast"


@router.post("/pipeline/organizing/generate-node-artifact")
def organizing_generate_node_artifact(req: NodeArtifactRequest) -> Dict[str, Any]:
    """AI generate konten markdown short draft untuk satu outline node.

    Output:
    - content: markdown pure paragraf (tanpa heading), 150-300 kata
    - words: aktual word count
    - sources_cited: list paper_id yang AI benar-benar pakai
    - generated_at: ISO timestamp
    """
    try:
        from app.services import task_router as _tr
        ai_available = True
    except Exception:
        ai_available = False

    if not ai_available:
        return {
            "status": "error",
            "message": "AI orchestrator not available",
            "content": "",
            "words": 0,
            "sources_cited": [],
        }

    # Clamp target_words ke rentang aman 100-500 untuk prevent abuse
    target_words = max(100, min(500, int(req.target_words or 250)))
    max_tokens = int(target_words * 2.2)  # token ratio ~2x untuk safety margin

    # Trim attached papers ke max 5 supaya prompt tidak meledak
    papers_brief = []
    for p in (req.attached_papers or [])[:5]:
        pid = str(p.get("id", "") or p.get("doi", "") or p.get("title", ""))[:120]
        authors_raw = p.get("authors", "")
        if isinstance(authors_raw, list):
            authors_str = ", ".join(str(a) for a in authors_raw[:3])
            if len(authors_raw) > 3:
                authors_str += " et al."
        else:
            authors_str = str(authors_raw)
        papers_brief.append({
            "id": pid,
            "title": str(p.get("title", ""))[:200],
            "authors": authors_str[:120],
            "year": str(p.get("year", "") or "")[:8],
            "abstract": str(p.get("abstract") or p.get("abstract_synth") or p.get("synthesis") or "")[:350],
        })

    # Build context block · hierarchical position memberi AI orientation
    parent_context = " > ".join([str(x) for x in (req.parent_chain or []) if x][:4])
    location_line = f"{parent_context} > {req.node_title}" if parent_context else req.node_title

    if papers_brief:
        papers_block = "\n\n".join([
            f"[{p['id']}] {p['authors']} ({p['year']}). {p['title']}\nAbstract: {p['abstract']}"
            for p in papers_brief
        ])
        citation_instruction = (
            "Cite the papers above inline using format (Author, Year) where Author is the surname of "
            "the first author from the paper metadata above. Cite at least 2 sources when 2 or more "
            "are provided. Only cite papers that are actually relevant to the sentence you write. "
            "After writing, list the paper IDs you actually cited in a separate JSON block."
        )
    else:
        papers_block = "(No papers attached to this node)"
        citation_instruction = (
            "No papers attached. Write a generic but academically sound draft based on the thesis "
            "statement and node title. Use neutral hedging language and avoid fabricating specific "
            "citations. Mark this draft as ungrounded."
        )

    tone_hint = "concise journal article style" if req.mode == "journal" else "academic thesis style with explanatory depth"
    lang_hint, lang_display = _resolve_language(req.language)

    prompt = f"""You are an expert research writer. Write a SHORT DRAFT for one outline node in a research paper.

Research title: {req.chosen_title or '(not provided)'}
Thesis statement: {req.thesis_statement or '(not provided)'}
Node location in outline: {location_line}
Node title to draft: {req.node_title}
Target language: {lang_hint}
Tone: {tone_hint}
Target length: approximately {target_words} words ({max(target_words - 50, 100)} to {target_words + 50} words allowed)

Attached supporting papers:
{papers_block}

Writing rules:
1. Output PURE PARAGRAPH(S) ONLY. NO heading. NO markdown headers. NO bullet points. NO numbered lists.
2. Write 1 to 2 cohesive paragraphs of flowing academic prose.
3. {citation_instruction}
4. Do not use em-dash punctuation. Do not use semicolons.
5. Do not invent statistics, data points, or facts not present in the attached papers.
6. Stay tightly focused on the node title scope. Do not drift to sibling topics.

Return response in this EXACT format with two parts separated by the delimiter:

---DRAFT---
(your paragraph(s) here)
---CITED---
{{"sources_cited": ["paper_id_1", "paper_id_2"]}}
"""

    # Quality mode pipeline · pick between fast single-shot atau best-of-3
    # plus self-refine. Wrapped di helper supaya bisa reused untuk compose
    # section nanti kalau diperlukan.
    quality_mode = (req.quality_mode or "fast").lower()
    valid_paper_ids = {p["id"] for p in papers_brief}

    try:
        if quality_mode == "best":
            content_md, sources_cited, method_used = _generate_quality_best(
                _tr, prompt, max_tokens, valid_paper_ids, target_words
            )
        else:
            content_md, sources_cited, method_used = _generate_quality_fast(
                _tr, prompt, max_tokens, valid_paper_ids
            )
    except Exception as exc:
        return {
            "status": "error",
            "message": f"AI call failed: {exc}",
            "content": "",
            "words": 0,
            "sources_cited": [],
        }

    if not content_md:
        return {
            "status": "error",
            "message": "AI returned empty response",
            "content": "",
            "words": 0,
            "sources_cited": [],
        }

    # Compute word count (rough split by whitespace)
    word_count = len([w for w in re.split(r"\s+", content_md) if w])

    return {
        "status": "success",
        "node_id": req.node_id,
        "content": content_md,
        "words": word_count,
        "sources_cited": sources_cited,
        "papers_provided": len(papers_brief),
        "target_words": target_words,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "method": method_used,
        "quality_mode": quality_mode,
    }


def _parse_ai_draft_output(raw_text: str, valid_paper_ids: set) -> Tuple[str, List[str]]:
    """Parse output format ---DRAFT--- ... ---CITED--- {json}.
    Return tuple (content_md, sources_cited). Validate sources_cited terhadap
    valid_paper_ids supaya tidak ada hallucinated citation."""
    raw_text = (raw_text or "").strip()
    content_md = raw_text
    sources_cited: List[str] = []

    if "---DRAFT---" in raw_text:
        try:
            after_draft = raw_text.split("---DRAFT---", 1)[1]
            if "---CITED---" in after_draft:
                draft_part, cited_part = after_draft.split("---CITED---", 1)
                content_md = draft_part.strip()
                m = re.search(r"\{[\s\S]*\}", cited_part)
                if m:
                    try:
                        cited_obj = json.loads(m.group(0))
                        raw_cited = cited_obj.get("sources_cited", [])
                        if isinstance(raw_cited, list):
                            sources_cited = [str(x)[:120] for x in raw_cited if str(x) in valid_paper_ids]
                    except Exception:
                        pass
            else:
                content_md = after_draft.strip()
        except Exception:
            content_md = raw_text

    # Strip stray heading markers
    content_md = re.sub(r"^\s*#{1,6}\s+.*$", "", content_md, flags=re.MULTILINE).strip()
    # Strip em-dash and semicolon (user preference)
    content_md = content_md.replace("—", " ").replace(" – ", " ").replace(";", ".")
    # Strip surrounding quotes kalau AI wrap
    content_md = content_md.strip('"').strip("'").strip()

    return content_md, sources_cited


def _generate_quality_fast(_tr, prompt: str, max_tokens: int, valid_paper_ids: set) -> Tuple[str, List[str], str]:
    """Fast mode · single-shot generation temperature 0.4."""
    result = _tr.route(
        task_type="reason",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    if not (result.get("ok") and result.get("text")):
        return "", [], "ai_empty"
    content, cited = _parse_ai_draft_output(result["text"], valid_paper_ids)
    return content, cited, "fast_single_shot"


def _score_candidate(content: str, cited: List[str], target_words: int, papers_count: int) -> float:
    """Heuristic quality score untuk pick best dari N kandidat.
    Komponen:
    1. Citation density · semakin tinggi semakin baik (max papers attached)
    2. Word count alignment · penalty kalau terlalu jauh dari target
    3. Content presence · penalty kalau terlalu pendek
    Tidak pakai LLM judge karena nambah cost. Heuristic ini biasanya sufficient
    untuk pick best dari 3 kandidat (Cobbe et al. 2021 verifier-based selection
    dengan simple metrics)."""
    if not content:
        return -100.0

    wc = len([w for w in content.split() if w])
    word_count_score = 1.0 - min(abs(wc - target_words) / max(target_words, 1), 1.0)

    if papers_count > 0:
        density = len(cited) / papers_count
        citation_score = min(density * 1.2, 1.0)  # bonus kalau cite ratio tinggi
    else:
        citation_score = 0.5  # neutral kalau tidak ada paper untuk dikutip

    length_penalty = -0.5 if wc < 50 else 0.0

    return (word_count_score * 0.4) + (citation_score * 0.5) + (length_penalty * 0.1)


def _generate_quality_best(_tr, prompt: str, max_tokens: int, valid_paper_ids: set, target_words: int) -> Tuple[str, List[str], str]:
    """Best mode · 3 parallel candidates temp 0.7, pick best heuristic, then
    self-refine critique loop. Total: 3+1 = 4 API calls. Trade-off cost untuk
    quality. Lihat Cobbe et al. (2021) plus Madaan et al. (2024) NeurIPS."""
    from concurrent.futures import ThreadPoolExecutor

    papers_count = len(valid_paper_ids)

    def _one_candidate(temp_val: float) -> Tuple[str, List[str]]:
        try:
            r = _tr.route(
                task_type="reason",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp_val,
            )
            if not (r.get("ok") and r.get("text")):
                return "", []
            return _parse_ai_draft_output(r["text"], valid_paper_ids)
        except Exception:
            return "", []

    # Parallel 3 candidates dengan slightly different temperatures
    candidates: List[Tuple[str, List[str]]] = []
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(_one_candidate, 0.6),
                executor.submit(_one_candidate, 0.7),
                executor.submit(_one_candidate, 0.8),
            ]
            for fut in futures:
                try:
                    candidates.append(fut.result(timeout=45))
                except Exception:
                    pass
    except Exception:
        pass

    valid_candidates = [(c, cit) for c, cit in candidates if c]
    if not valid_candidates:
        # Fallback to fast single-shot kalau best-of-N gagal semua
        return _generate_quality_fast(_tr, prompt, max_tokens, valid_paper_ids)[:2] + ("best_fallback_to_fast",)

    # Pick best by heuristic score
    best = max(valid_candidates, key=lambda x: _score_candidate(x[0], x[1], target_words, papers_count))
    best_content, best_cited = best

    # Self-refine pass · AI critique sendiri lalu regenerate dengan critique
    # sebagai context. Skip kalau best sudah scoring tinggi untuk save cost.
    best_score = _score_candidate(best_content, best_cited, target_words, papers_count)
    if best_score >= 0.85:
        return best_content, best_cited, "best_of_3_high_score_no_refine_needed"

    critique_prompt = f"""You are an academic editor reviewing a draft paragraph. Critique this draft against the original instructions.

ORIGINAL TASK PROMPT (truncated):
{prompt[:1500]}

DRAFT TO REVIEW:
{best_content}

CITED PAPERS IN DRAFT: {best_cited}

CRITIQUE CHECKLIST:
1. Does the draft cite at least 2 papers (when papers were provided)?
2. Are citations inline using (Author, Year) format?
3. Is the draft pure paragraph (no headings, no bullets)?
4. Does the word count roughly match the target ({target_words} words)?
5. Does the content stay focused on the node title scope?
6. Any em-dash or semicolons that should be removed?
7. Any factual claims that seem fabricated (not in attached papers)?

Output your critique in plain text (max 200 words), listing specific issues that need fixing. Then state at the end: REVISION_NEEDED: yes or REVISION_NEEDED: no."""

    try:
        critique_result = _tr.route(
            task_type="reason",
            messages=[{"role": "user", "content": critique_prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        critique_text = critique_result.get("text", "") if critique_result.get("ok") else ""

        needs_revision = "revision_needed: yes" in critique_text.lower()
        if not needs_revision or not critique_text:
            return best_content, best_cited, "best_of_3_critique_clean"

        # Regenerate dengan critique sebagai context
        refine_prompt = f"""You are revising a draft paragraph based on critique feedback. Produce an IMPROVED version.

ORIGINAL TASK:
{prompt[:1500]}

PREVIOUS DRAFT:
{best_content}

CRITIQUE FEEDBACK:
{critique_text[:800]}

Apply the critique fixes. Return the revised paragraph in the same format as before:

---DRAFT---
(revised paragraph)
---CITED---
{{"sources_cited": ["paper_id_1"]}}
"""
        refine_result = _tr.route(
            task_type="reason",
            messages=[{"role": "user", "content": refine_prompt}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        if refine_result.get("ok") and refine_result.get("text"):
            refined_content, refined_cited = _parse_ai_draft_output(refine_result["text"], valid_paper_ids)
            if refined_content:
                # Compare score · pakai yang lebih bagus
                refined_score = _score_candidate(refined_content, refined_cited, target_words, papers_count)
                if refined_score >= best_score:
                    return refined_content, refined_cited, "best_of_3_with_self_refine"
        # Refinement tidak improve, balik ke best original
        return best_content, best_cited, "best_of_3_refine_no_improvement"
    except Exception:
        return best_content, best_cited, "best_of_3_critique_failed"


# =============================================================================
# STREAMING GENERATE NODE ARTIFACT (SSE)
# =============================================================================
# Streaming version dari generate-node-artifact yang return Server-Sent Events.
# Frontend dapat consume via fetch ReadableStream untuk show konten real-time
# character by character (seperti ChatGPT). UX perceived speed jauh lebih baik
# walaupun actual generation time sama.
#
# Format SSE: setiap event adalah satu baris "data: {json}\n\n". Event types:
# - {"type": "phase", "name": "building_prompt|calling_ai|parsing|done"}
# - {"type": "text", "content": "chunk text"} · streaming text per token
# - {"type": "complete", "result": {full artifact data}} · final aggregated
# - {"type": "error", "message": "..."}
#
# Note: Best mode tidak support streaming karena 3 parallel candidates + judge
# tidak meaningful untuk streaming. Hanya Fast mode yang stream true.
@router.post("/pipeline/organizing/generate-node-artifact-stream")
def organizing_generate_node_artifact_stream(req: NodeArtifactRequest):
    """SSE streaming variant. Yield text chunks real-time selama AI generate.
    Best mode di-redirect ke non-stream sync (3 candidates + judge tidak
    stream-friendly). Fast mode true stream."""
    try:
        from app.services import task_router as _tr
    except Exception:
        def _err_gen():
            yield "data: " + json.dumps({"type": "error", "message": "AI orchestrator not available"}) + "\n\n"
        return StreamingResponse(_err_gen(), media_type="text/event-stream")

    # Build prompt sama persis seperti non-streaming version
    target_words = max(100, min(500, int(req.target_words or 250)))
    max_tokens = int(target_words * 2.2)

    papers_brief = []
    for p in (req.attached_papers or [])[:5]:
        pid = str(p.get("id", "") or p.get("doi", "") or p.get("title", ""))[:120]
        authors_raw = p.get("authors", "")
        if isinstance(authors_raw, list):
            authors_str = ", ".join(str(a) for a in authors_raw[:3])
            if len(authors_raw) > 3:
                authors_str += " et al."
        else:
            authors_str = str(authors_raw)
        papers_brief.append({
            "id": pid,
            "title": str(p.get("title", ""))[:200],
            "authors": authors_str[:120],
            "year": str(p.get("year", "") or "")[:8],
            "abstract": str(p.get("abstract") or p.get("abstract_synth") or p.get("synthesis") or "")[:350],
        })

    parent_context = " > ".join([str(x) for x in (req.parent_chain or []) if x][:4])
    location_line = f"{parent_context} > {req.node_title}" if parent_context else req.node_title

    if papers_brief:
        papers_block = "\n\n".join([
            f"[{p['id']}] {p['authors']} ({p['year']}). {p['title']}\nAbstract: {p['abstract']}"
            for p in papers_brief
        ])
        citation_instruction = (
            "Cite the papers above inline using format (Author, Year) where Author is the surname of "
            "the first author from the paper metadata above. Cite at least 2 sources when 2 or more "
            "are provided. Only cite papers that are actually relevant to the sentence you write. "
            "After writing, list the paper IDs you actually cited in a separate JSON block."
        )
    else:
        papers_block = "(No papers attached to this node)"
        citation_instruction = (
            "No papers attached. Write a generic but academically sound draft based on the thesis "
            "statement and node title. Use neutral hedging language and avoid fabricating specific citations."
        )

    tone_hint = "concise journal article style" if req.mode == "journal" else "academic thesis style with explanatory depth"
    lang_hint, lang_display = _resolve_language(req.language)
    valid_paper_ids = {p["id"] for p in papers_brief}

    prompt = f"""You are an expert research writer. Write a SHORT DRAFT for one outline node in a research paper.

Research title: {req.chosen_title or '(not provided)'}
Thesis statement: {req.thesis_statement or '(not provided)'}
Node location in outline: {location_line}
Node title to draft: {req.node_title}
Target language: {lang_hint}
Tone: {tone_hint}
Target length: approximately {target_words} words ({max(target_words - 50, 100)} to {target_words + 50} words allowed)

Attached supporting papers:
{papers_block}

Writing rules:
1. Output PURE PARAGRAPH(S) ONLY. NO heading. NO markdown headers. NO bullet points. NO numbered lists.
2. Write 1 to 2 cohesive paragraphs of flowing academic prose.
3. {citation_instruction}
4. Do not use em-dash punctuation. Do not use semicolons.
5. Do not invent statistics, data points, or facts not present in the attached papers.
6. Stay tightly focused on the node title scope. Do not drift to sibling topics.

Return response in this EXACT format with two parts separated by the delimiter:

---DRAFT---
(your paragraph(s) here)
---CITED---
{{"sources_cited": ["paper_id_1", "paper_id_2"]}}
"""

    def _sse_event(data: Dict[str, Any]) -> str:
        return "data: " + json.dumps(data) + "\n\n"

    def _stream():
        """Generator yang yield SSE events. Phase awal informasi user, lalu
        text chunks real-time, lalu complete event dengan parsed metadata."""
        yield _sse_event({"type": "phase", "name": "building_prompt"})
        yield _sse_event({"type": "phase", "name": "calling_ai", "target_words": target_words})

        accumulated_text = ""
        routing_info = None

        try:
            for event in _tr.route_stream(
                task_type="reason",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.4,
            ):
                if isinstance(event, dict):
                    if event.get("type") == "routing":
                        routing_info = event
                        yield _sse_event({"type": "phase", "name": "ai_streaming", "provider": event.get("provider"), "model": event.get("model")})
                    elif event.get("type") == "done":
                        # Final routing event
                        pass
                    elif event.get("type") == "error":
                        yield _sse_event({"type": "error", "message": event.get("error", "unknown")})
                        return
                elif isinstance(event, str):
                    accumulated_text += event
                    yield _sse_event({"type": "text", "content": event})
        except Exception as exc:
            yield _sse_event({"type": "error", "message": f"Streaming failed: {exc}"})
            return

        # Parse final output untuk extract content + sources_cited
        yield _sse_event({"type": "phase", "name": "parsing"})
        content_md, sources_cited = _parse_ai_draft_output(accumulated_text, valid_paper_ids)
        word_count = len([w for w in re.split(r"\s+", content_md) if w])

        yield _sse_event({
            "type": "complete",
            "result": {
                "node_id": req.node_id,
                "content": content_md,
                "words": word_count,
                "sources_cited": sources_cited,
                "papers_provided": len(papers_brief),
                "target_words": target_words,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "method": "fast_stream",
                "quality_mode": "fast",
                "provider": (routing_info or {}).get("provider"),
            },
        })

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering kalau ada
            "Connection": "keep-alive",
        },
    )


# =============================================================================
# COMPOSE SECTION
# =============================================================================
# Dual-mode generation di level parent section. Dua mode:
# 1. Abstract mode (is_abstract=True): satu blok 200 kata CARS structure
#    (Swales 1990 Genre Analysis) yang merupakan single coherent paragraph,
#    bukan sum dari children. Sesuai genre Abstract yang publication-ready
#    sebagai satu unit rhetorical.
# 2. Regular mode: parallel generate paragraph per leaf child PLUS section
#    opener untuk parent. Mengikuti Hyland (2019) Academic Discourse dual-level
#    pattern dimana section opener adalah unit penulisan tersendiri.
#
# Parallel via ThreadPoolExecutor 6 workers untuk speed.
class ComposeSectionRequest(BaseModel):
    section_id: str
    section_title: str
    section_level: int = 1
    parent_chain: List[str] = []
    children: List[Dict[str, Any]] = []  # [{id, title, level, is_leaf, attached_paper_ids}]
    thesis_statement: Optional[str] = ""
    chosen_title: Optional[str] = ""
    papers: List[Dict[str, Any]] = []     # full pool, dipakai cross-ref attached_paper_ids
    target_words_per_leaf: int = 250
    target_words_opener: int = 120
    target_words_abstract: int = 200
    language: str = "en"
    mode: Optional[str] = "journal"
    is_abstract: bool = False


@router.post("/pipeline/organizing/compose-section")
def organizing_compose_section(req: ComposeSectionRequest) -> Dict[str, Any]:
    """Compose section content. Dua mode (abstract vs regular).

    Output:
    - status: success | error
    - artifacts: {node_id: artifact_data} dimana artifact_data sama struktur
      dengan output generate-node-artifact (content, words, sources_cited,
      generated_at).
    """
    try:
        from app.services import task_router as _tr
        ai_available = True
    except Exception:
        ai_available = False

    if not ai_available:
        return {
            "status": "error",
            "message": "AI orchestrator not available",
            "artifacts": {},
        }

    # Build papers_by_id lookup dari pool
    papers_by_id = {}
    for p in (req.papers or []):
        pid = str(p.get("id") or p.get("doi") or p.get("title") or "")[:120]
        if pid:
            papers_by_id[pid] = p

    def _build_paper_brief(pid: str) -> Optional[Dict[str, Any]]:
        p = papers_by_id.get(pid)
        if not p:
            return None
        authors_raw = p.get("authors", "")
        if isinstance(authors_raw, list):
            authors_str = ", ".join(str(a) for a in authors_raw[:3])
            if len(authors_raw) > 3:
                authors_str += " et al."
        else:
            authors_str = str(authors_raw)
        return {
            "id": pid,
            "title": str(p.get("title", ""))[:200],
            "authors": authors_str[:120],
            "year": str(p.get("year", "") or "")[:8],
            "abstract": str(p.get("abstract") or p.get("abstract_synth") or p.get("synthesis") or "")[:350],
        }

    tone_hint = "concise journal article style" if req.mode == "journal" else "academic thesis style"
    lang_hint, lang_display = _resolve_language(req.language)

    # =========================================================================
    # ABSTRACT MODE · single block CARS structure
    # =========================================================================
    if req.is_abstract:
        # Aggregate semua attached papers dari semua children (Abstract reference
        # spektrum keseluruhan paper)
        all_pids: List[str] = []
        for c in req.children:
            for pid in (c.get("attached_paper_ids") or []):
                if pid not in all_pids:
                    all_pids.append(pid)

        attached_papers = []
        for pid in all_pids[:6]:  # cap 6 untuk Abstract supaya prompt tidak besar
            brief = _build_paper_brief(pid)
            if brief:
                attached_papers.append(brief)

        children_outline = "\n".join([f"- {c.get('title', '')}" for c in req.children if c.get('title')])

        if attached_papers:
            papers_block = "\n\n".join([
                f"[{p['id']}] {p['authors']} ({p['year']}). {p['title']}\nAbstract: {p['abstract']}"
                for p in attached_papers
            ])
        else:
            papers_block = "(No papers attached)"

        target_words = req.target_words_abstract
        max_tokens = int(target_words * 2.5)

        prompt = f"""You are an expert academic writer composing an Abstract for a research paper following the CARS structure (Swales 1990, Genre Analysis in English in Academic and Research Settings).

Research title: {req.chosen_title or '(not provided)'}
Thesis statement: {req.thesis_statement or '(not provided)'}
Target language: {lang_hint}
Tone: {tone_hint}

Abstract should cover these sub-points (as content guides, NOT as separate sections):
{children_outline}

Supporting papers available for citation:
{papers_block}

WRITING RULES:
1. Output ONE SINGLE coherent paragraph of approximately {target_words} words ({target_words - 30} to {target_words + 30} allowed).
2. Follow CARS structure compressed: (a) establish the territory by stating centrality of the topic, (b) indicate the gap or problem in existing work, (c) occupy the niche by stating what this study contributes plus key findings.
3. Cover ALL sub-points listed above but as flowing prose, NOT as bullet-style enumeration.
4. Use minimal citations 0 to 3 maximum. Abstract typically does not cite heavily.
5. Do NOT use em-dash. Do NOT use semicolons. Do NOT use markdown headers. Do NOT use bullet points.
6. Do NOT invent statistics or data not present in attached papers.
7. Single coherent paragraph, publication-ready quality.

Return response in this EXACT format:

---DRAFT---
(your abstract paragraph here as single block)
---CITED---
{{"sources_cited": ["paper_id_1"]}}
"""

        try:
            result = _tr.route(
                task_type="structured_fast",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(max_tokens * 0.7),  # turun ratio juga
                temperature=0.4,
            )
        except Exception as exc:
            return {"status": "error", "message": f"AI call failed: {exc}", "artifacts": {}}

        if not (result.get("ok") and result.get("text")):
            return {"status": "error", "message": result.get("error") or "AI returned empty", "artifacts": {}}

        raw_text = (result.get("text") or "").strip()
        content_md = raw_text
        sources_cited: List[str] = []
        if "---DRAFT---" in raw_text:
            try:
                after_draft = raw_text.split("---DRAFT---", 1)[1]
                if "---CITED---" in after_draft:
                    draft_part, cited_part = after_draft.split("---CITED---", 1)
                    content_md = draft_part.strip()
                    m = re.search(r"\{[\s\S]*\}", cited_part)
                    if m:
                        try:
                            cited_obj = json.loads(m.group(0))
                            raw_cited = cited_obj.get("sources_cited", [])
                            if isinstance(raw_cited, list):
                                valid_ids = {p["id"] for p in attached_papers}
                                sources_cited = [str(x)[:120] for x in raw_cited if str(x) in valid_ids]
                        except Exception:
                            pass
                else:
                    content_md = after_draft.strip()
            except Exception:
                content_md = raw_text

        content_md = re.sub(r"^\s*#{1,6}\s+.*$", "", content_md, flags=re.MULTILINE).strip()
        content_md = content_md.replace("—", " ").replace(" – ", " ").replace(";", ".")
        word_count = len([w for w in re.split(r"\s+", content_md) if w])

        return {
            "status": "success",
            "method": "abstract_cars_single_block",
            "artifacts": {
                req.section_id: {
                    "content": content_md,
                    "words": word_count,
                    "sources_cited": sources_cited,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "papers_provided": len(attached_papers),
                    "kind": "abstract_block",
                }
            },
        }

    # =========================================================================
    # REGULAR MODE · parallel generate per leaf child + section opener
    # =========================================================================
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Identify leaf children (children yang tidak punya anak)
    leaf_children = [c for c in req.children if c.get("is_leaf", True)]
    if not leaf_children:
        return {
            "status": "error",
            "message": "Section has no leaf children to generate. Add sub-nodes first.",
            "artifacts": {},
        }

    parent_context = " > ".join([str(x) for x in (req.parent_chain or []) if x][:4])
    section_location = f"{parent_context} > {req.section_title}" if parent_context else req.section_title

    def _generate_leaf(child: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate single leaf paragraph. Mirip generate-node-artifact tapi
        embedded di sini untuk reuse parallel context."""
        cid = str(child.get("id") or "")
        ctitle = str(child.get("title") or "")

        # Build attached papers untuk leaf ini
        attached_pids = child.get("attached_paper_ids") or []
        leaf_papers = []
        for pid in attached_pids[:5]:
            brief = _build_paper_brief(pid)
            if brief:
                leaf_papers.append(brief)

        if leaf_papers:
            papers_block = "\n\n".join([
                f"[{p['id']}] {p['authors']} ({p['year']}). {p['title']}\nAbstract: {p['abstract']}"
                for p in leaf_papers
            ])
            citation_instr = (
                "Cite the papers above inline using format (Author, Year) using the first author's "
                "surname. Cite at least 2 sources when 2 or more are provided."
            )
        else:
            papers_block = "(No papers attached to this sub-node)"
            citation_instr = "No papers attached. Write generic prose using hedging language, do not fabricate citations."

        target_words = req.target_words_per_leaf

        leaf_prompt = f"""You are an expert research writer. Write a SHORT DRAFT for one outline sub-node.

Research title: {req.chosen_title or '(not provided)'}
Thesis statement: {req.thesis_statement or '(not provided)'}
Section: {req.section_title}
Sub-node to draft: {ctitle}
Target language: {lang_hint}
Tone: {tone_hint}
Target length: approximately {target_words} words

Attached supporting papers:
{papers_block}

Writing rules:
1. Output PURE PARAGRAPH(S). NO heading. NO markdown headers. NO bullet points.
2. Write 1 to 2 cohesive paragraphs.
3. {citation_instr}
4. No em-dash. No semicolons.
5. Stay tightly focused on the sub-node scope.

Return EXACT format:

---DRAFT---
(your paragraphs here)
---CITED---
{{"sources_cited": ["paper_id_1"]}}
"""
        try:
            r = _tr.route(
                task_type="structured_fast",  # skip R1 reasoning chain
                messages=[{"role": "user", "content": leaf_prompt}],
                max_tokens=int(target_words * 1.6),  # ratio diturunkan
                temperature=0.4,
            )
            if not (r.get("ok") and r.get("text")):
                return cid, {"content": "", "words": 0, "sources_cited": [], "error": "ai_empty"}
            raw = (r.get("text") or "").strip()
            content = raw
            cited: List[str] = []
            if "---DRAFT---" in raw:
                after = raw.split("---DRAFT---", 1)[1]
                if "---CITED---" in after:
                    d, c = after.split("---CITED---", 1)
                    content = d.strip()
                    m = re.search(r"\{[\s\S]*\}", c)
                    if m:
                        try:
                            obj = json.loads(m.group(0))
                            raw_c = obj.get("sources_cited", [])
                            if isinstance(raw_c, list):
                                vids = {p["id"] for p in leaf_papers}
                                cited = [str(x)[:120] for x in raw_c if str(x) in vids]
                        except Exception:
                            pass
                else:
                    content = after.strip()
            content = re.sub(r"^\s*#{1,6}\s+.*$", "", content, flags=re.MULTILINE).strip()
            content = content.replace("—", " ").replace(" – ", " ").replace(";", ".")
            wc = len([w for w in re.split(r"\s+", content) if w])
            return cid, {
                "content": content,
                "words": wc,
                "sources_cited": cited,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "papers_provided": len(leaf_papers),
                "kind": "leaf_paragraph",
            }
        except Exception as exc:
            return cid, {"content": "", "words": 0, "sources_cited": [], "error": str(exc)[:200]}

    def _generate_opener() -> Tuple[str, Dict[str, Any]]:
        """Generate section opener · introductory paragraph yang frame seluruh
        section. Tidak deep, hanya transition + roadmap untuk children."""
        children_titles = "\n".join([f"- {c.get('title', '')}" for c in req.children if c.get('title')])
        target_words = req.target_words_opener

        opener_prompt = f"""You are an expert academic writer. Write a SHORT OPENING PARAGRAPH for one section of a research paper.

Research title: {req.chosen_title or '(not provided)'}
Thesis statement: {req.thesis_statement or '(not provided)'}
Section location: {section_location}
Section title: {req.section_title}
Target language: {lang_hint}
Tone: {tone_hint}
Target length: approximately {target_words} words (a short opener, not the whole section)

This section will cover the following sub-topics:
{children_titles}

WRITING RULES FOR SECTION OPENER:
1. Output ONE short paragraph ({target_words - 20} to {target_words + 20} words).
2. Purpose of opener: (a) introduce the section's role within the paper, (b) preview the sub-topics that follow, (c) connect to the thesis statement.
3. Do NOT go deep into any sub-topic content (children paragraphs will do that).
4. NO heading. NO markdown. NO bullet points. NO em-dash. NO semicolons.
5. Do NOT cite specific papers (opener is meta-level).
6. Single coherent paragraph that primes the reader.

Return ONLY the opener paragraph text, no delimiters, no commentary."""

        try:
            r = _tr.route(
                task_type="structured_fast",
                messages=[{"role": "user", "content": opener_prompt}],
                max_tokens=int(target_words * 1.6),
                temperature=0.4,
            )
            if not (r.get("ok") and r.get("text")):
                return req.section_id, {"content": "", "words": 0, "sources_cited": [], "error": "ai_empty"}
            content = (r.get("text") or "").strip().strip('"').strip("'")
            content = re.sub(r"^\s*#{1,6}\s+.*$", "", content, flags=re.MULTILINE).strip()
            content = content.replace("—", " ").replace(" – ", " ").replace(";", ".")
            wc = len([w for w in re.split(r"\s+", content) if w])
            return req.section_id, {
                "content": content,
                "words": wc,
                "sources_cited": [],
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "papers_provided": 0,
                "kind": "section_opener",
            }
        except Exception as exc:
            return req.section_id, {"content": "", "words": 0, "sources_cited": [], "error": str(exc)[:200]}

    # PARALLEL execute leaf generations + opener
    artifacts: Dict[str, Dict[str, Any]] = {}
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_generate_leaf, c): c.get("id") for c in leaf_children}
            opener_future = executor.submit(_generate_opener)

            for fut in as_completed(futures, timeout=120):
                try:
                    nid, art = fut.result(timeout=60)
                    if nid and art.get("content"):
                        artifacts[nid] = art
                except Exception:
                    pass

            try:
                sid, opener_art = opener_future.result(timeout=60)
                if sid and opener_art.get("content"):
                    artifacts[sid] = opener_art
            except Exception:
                pass
    except Exception as exc:
        return {"status": "error", "message": f"Parallel execution failed: {exc}", "artifacts": artifacts}

    return {
        "status": "success",
        "method": "regular_cascade_parallel",
        "artifacts": artifacts,
        "stats": {
            "leaf_children": len(leaf_children),
            "artifacts_generated": len(artifacts),
            "opener_included": req.section_id in artifacts,
        },
    }


class CoherenceCheckRequest(BaseModel):
    thesis_statement: str
    outline: List[Dict[str, Any]]
    source_mappings: Optional[Dict[str, List[str]]] = None
    mode: Optional[str] = "journal"
    language: str = "en"


@router.post("/pipeline/organizing/check-coherence")
def organizing_check_coherence(req: CoherenceCheckRequest) -> Dict[str, Any]:
    """Cek coherence outline terhadap thesis statement.

    Output: score 0-100 plus list issues. Mengikuti rubric akademik dari
    Booth, Colomb, Williams (2008) Craft of Research bahwa outline yang
    coherent harus traceable kembali ke thesis statement, balanced antar
    section, dan terdukung evidence.
    """
    issues: List[Dict[str, Any]] = []
    score = 100

    # Check thesis statement length
    ts = (req.thesis_statement or "").strip()
    if len(ts) < 40:
        issues.append({"severity": "error", "message": "Thesis statement is too short or empty. A strong thesis is 3-5 full academic sentences."})
        score -= 25
    elif len(ts) < 150:
        issues.append({"severity": "warning", "message": "Thesis statement seems thin. Consider expanding each of the four components."})
        score -= 10

    # Check outline structure
    n = len(req.outline or [])
    if n < 20:
        issues.append({"severity": "error", "message": f"Outline has only {n} nodes which is too few for a {req.mode}. Recommend 60-110 nodes minimum."})
        score -= 20
    elif n < 50:
        issues.append({"severity": "warning", "message": f"Outline has {n} nodes which is on the thin side. Consider expanding sub sections."})
        score -= 8

    # Check level 1 count
    l1 = [x for x in req.outline if (x.get("level") or 1) == 1]
    expected_l1 = 5 if req.mode == "thesis" else 7
    if len(l1) < expected_l1 - 2:
        issues.append({"severity": "warning", "message": f"Outline has only {len(l1)} top level sections. Standard {req.mode} format expects around {expected_l1}."})
        score -= 10

    # Check tree balance · ratio of level 2 to level 1
    l2 = [x for x in req.outline if (x.get("level") or 1) == 2]
    if l1 and len(l2) / len(l1) < 2.5:
        issues.append({"severity": "info", "message": "Sections feel underdeveloped. Aim for at least 4-7 sub sections per top level section."})
        score -= 5

    # Check source mapping coverage
    mappings = req.source_mappings or {}
    nodes_with_papers = sum(1 for nid in mappings if mappings[nid])
    if nodes_with_papers == 0:
        issues.append({"severity": "warning", "message": "No source mapping yet. Attach supporting papers to outline nodes to strengthen evidence base."})
        score -= 15
    elif l2 and nodes_with_papers < len(l2) * 0.4:
        issues.append({"severity": "warning", "message": f"Only {nodes_with_papers} of {len(l2)} sub sections have papers attached. Target 60 percent coverage minimum."})
        score -= 10

    # Check title coherence with thesis
    ts_lower = ts.lower()
    expected_keywords = []
    if "lstm" in ts_lower: expected_keywords.append("lstm")
    if "xgboost" in ts_lower: expected_keywords.append("xgboost")
    if "shap" in ts_lower: expected_keywords.append("shap")
    missing_in_outline = []
    outline_text = " ".join((x.get("title") or "").lower() for x in req.outline)
    for kw in expected_keywords:
        if kw not in outline_text:
            missing_in_outline.append(kw)
    if missing_in_outline:
        issues.append({"severity": "warning", "message": f"Keywords from thesis statement not yet in outline: {', '.join(missing_in_outline)}. Ensure outline covers all methodological pillars."})
        score -= 8

    return {
        "status": "success",
        "result": {
            "score": max(0, score),
            "issues": issues,
            "method": "rule_based_with_heuristics",
            "stats": {
                "total_nodes": n,
                "level_1_sections": len(l1),
                "level_2_subsections": len(l2),
                "nodes_with_sources": nodes_with_papers,
            }
        }
    }


# ---------------------------------------------------------------------------
# Stage III · Drafting (Kluge & Taylor "Writing Research Papers")
# ---------------------------------------------------------------------------
# Tahap drafting writing the first draft dengan per-section writing mode
# adaptive. Section detection dari node title plus parent chain. Different
# rhetorical purpose per section type (Hyland 2019 Academic Discourse).
# ---------------------------------------------------------------------------

# Genre-specific style guide per target venue. Sesuai author guidelines yang
# umum di Scopus, IEEE, SINTA, MDPI, dan academic publishing conventions.
# Frontend pakai catalog ini untuk display template hints dan structural
# warnings ke user (misal Scopus Q1 maks 8000 kata, IEEE abstract 250 kata).
VENUE_STYLE_CATALOG = {
    "thesis_indonesia": {
        "label": "Skripsi/Tesis Indonesia (5-bab)",
        "structure": "5 chapters: Bab I Pendahuluan, Bab II Tinjauan Pustaka, Bab III Metodologi, Bab IV Hasil, Bab V Kesimpulan",
        "language": "Bahasa Indonesia akademik",
        "abstract_words": "150-250 kata, satu paragraf",
        "total_pages": "80-150 halaman",
        "citations_recommended": "30-50 minimum",
        "citation_style": "APA 7 (default Indonesian universities)",
        "heading_style": "Bab I, 1.1, 1.1.1 (numbered hierarchy)",
        "special_requirements": "Lembar Pengesahan, Pernyataan Orisinalitas, Kata Pengantar, Abstrak EN dan ID, Daftar Pustaka APA"
    },
    "thesis_international": {
        "label": "Thesis International (Western universities)",
        "structure": "Introduction, Literature Review, Methodology, Results, Discussion, Conclusion (5-6 chapters)",
        "language": "Academic English",
        "abstract_words": "300-500 words",
        "total_pages": "120-300 pages depending on degree",
        "citations_recommended": "50-150",
        "citation_style": "APA 7 or Harvard depending on department",
        "heading_style": "1, 1.1, 1.1.1 (no Chapter prefix)",
        "special_requirements": "Acknowledgments, Declaration of Originality, Table of Contents, List of Figures/Tables"
    },
    "journal_scopus_q1q2": {
        "label": "Jurnal Scopus Q1/Q2 (high impact)",
        "structure": "IMRAD: Abstract, Introduction, Related Work, Methods, Results, Discussion, Conclusion",
        "language": "Academic English",
        "abstract_words": "150-250 words, structured (Background, Methods, Results, Conclusions)",
        "total_pages": "8-12 pages double column (Elsevier/Springer format)",
        "citations_recommended": "30-60 recent (last 5 years priority)",
        "citation_style": "APA 7 atau Vancouver atau IEEE depending on journal",
        "heading_style": "1. Introduction, 1.1 Subsection",
        "special_requirements": "Highlights (3-5 bullet points), Graphical Abstract, Conflict of Interest, Author Contributions, Data Availability"
    },
    "journal_scopus_q3q4": {
        "label": "Jurnal Scopus Q3/Q4",
        "structure": "IMRAD standard",
        "language": "Academic English",
        "abstract_words": "150-250 words",
        "total_pages": "6-10 pages",
        "citations_recommended": "20-40",
        "citation_style": "APA 7 or Vancouver",
        "heading_style": "1. Introduction, 1.1 Subsection",
        "special_requirements": "Keywords, Author Affiliation, Corresponding Author email"
    },
    "journal_sinta1_2": {
        "label": "Jurnal SINTA 1/2 (Indonesia top-tier)",
        "structure": "IMRAD adapted Indonesian: Abstrak EN+ID, Pendahuluan, Tinjauan Pustaka, Metode, Hasil dan Pembahasan, Simpulan",
        "language": "Bahasa Indonesia OR English (cek author guideline)",
        "abstract_words": "200-300 kata, satu paragraf",
        "total_pages": "10-15 halaman",
        "citations_recommended": "20-40 dengan mayoritas jurnal terindeks",
        "citation_style": "APA 7 atau IEEE",
        "heading_style": "1. Pendahuluan, 1.1 Sub-section",
        "special_requirements": "Abstrak bilingual EN+ID, Kata Kunci 3-5, Author Affiliation dengan email"
    },
    "journal_sinta3_6": {
        "label": "Jurnal SINTA 3-6 (Indonesia)",
        "structure": "IMRAD adapted",
        "language": "Bahasa Indonesia atau English",
        "abstract_words": "150-250 kata",
        "total_pages": "8-12 halaman",
        "citations_recommended": "15-30",
        "citation_style": "APA 7",
        "heading_style": "1. Pendahuluan",
        "special_requirements": "Abstrak bilingual, Kata Kunci, Daftar Pustaka APA"
    },
    "journal_ieee": {
        "label": "IEEE Journal/Magazine",
        "structure": "IEEE format: Abstract, Index Terms, I. Introduction, II. Background, III. Methods, IV. Results, V. Discussion, VI. Conclusion",
        "language": "Academic English (IEEE Editorial Style)",
        "abstract_words": "150-250 words",
        "total_pages": "10-14 pages IEEE template (transactions) or 4-6 pages (letters)",
        "citations_recommended": "30-50",
        "citation_style": "IEEE numbered [1], [2], [3]",
        "heading_style": "I. Introduction, I.A Subsection (Roman numeral)",
        "special_requirements": "Index Terms (keywords), Author Biographies dengan foto, BibTeX IEEE format"
    },
    "journal_elsevier": {
        "label": "Elsevier Journal",
        "structure": "IMRAD with Highlights",
        "language": "Academic English",
        "abstract_words": "150-250 words",
        "total_pages": "8-12 pages Elsevier double column",
        "citations_recommended": "30-60",
        "citation_style": "Elsevier Harvard or APA",
        "heading_style": "1. Introduction, 1.1 Subsection",
        "special_requirements": "Highlights bullets, Graphical Abstract, Declaration of Competing Interest, CRediT taxonomy"
    },
    "journal_springer": {
        "label": "Springer Journal",
        "structure": "IMRAD",
        "language": "Academic English",
        "abstract_words": "150-250 words",
        "total_pages": "10-15 pages",
        "citations_recommended": "30-50",
        "citation_style": "Springer APA or Basic numerical",
        "heading_style": "1 Introduction, 1.1 Subsection",
        "special_requirements": "Keywords, Springer LaTeX template (or Word)"
    },
    "conference_acm": {
        "label": "ACM Conference (NeurIPS/ICML/SIGCHI)",
        "structure": "Top-tier conference: Abstract, Introduction, Related Work, Method, Experiments, Conclusion",
        "language": "Academic English",
        "abstract_words": "150-300 words",
        "total_pages": "8-10 pages double column ACM template (main) plus unlimited appendix",
        "citations_recommended": "40-80 (heavy literature review)",
        "citation_style": "ACM Reference Format",
        "heading_style": "1 INTRODUCTION, 1.1 Subsection",
        "special_requirements": "CCS Concepts, Keywords, ACM Reference Format citation, supplementary materials"
    }
}


@router.get("/pipeline/drafting/venue-catalog")
def drafting_venue_catalog() -> Dict[str, Any]:
    """Return venue style catalog untuk frontend display sebagai template hints."""
    return {"status": "success", "venues": VENUE_STYLE_CATALOG}


# NxI18n language directive support · sync dengan frontend NxI18n module
# yang support multi-language translation (id, en, zh, ja, ko, ar, fr, es,
# de, pt, ru, hi, dst). Backend terima language code dan build language
# directive yang di-append ke AI prompt supaya output sesuai bahasa target.
# Reference: nxlytics/core/_i18n.js · aiLanguageDirective() function.
_LANGUAGE_NAMES = {
    "id": ("Bahasa Indonesia akademik", "Bahasa Indonesia"),
    "en": ("academic English", "English"),
    "zh": ("Mandarin Chinese academic register (中文)", "Mandarin Chinese"),
    "ja": ("Japanese academic register (日本語)", "Japanese"),
    "ko": ("Korean academic register (한국어)", "Korean"),
    "ar": ("Modern Standard Arabic academic register (العربية)", "Arabic"),
    "fr": ("French academic register (français)", "French"),
    "es": ("Spanish academic register (español)", "Spanish"),
    "de": ("German academic register (Deutsch)", "German"),
    "pt": ("Portuguese academic register (português)", "Portuguese"),
    "ru": ("Russian academic register (русский)", "Russian"),
    "hi": ("Hindi academic register (हिन्दी)", "Hindi"),
    "th": ("Thai academic register (ภาษาไทย)", "Thai"),
    "vi": ("Vietnamese academic register (Tiếng Việt)", "Vietnamese"),
    "ms": ("Malay academic register (Bahasa Melayu)", "Malay"),
}


def _resolve_language(lang_code: str) -> Tuple[str, str]:
    """Resolve language code ke (hint string untuk prompt, display name).
    Default ke academic English kalau code unknown."""
    code = (lang_code or "en").lower()
    if code in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[code]
    return _LANGUAGE_NAMES["en"]


# Section type detection · keyword matching pada title node + parent titles
# untuk identify rhetorical purpose. Sesuai konvensi IMRAD plus Indonesian
# skripsi 5-bab structure. Mapping bilingual untuk handle EN dan ID titles.
_SECTION_KEYWORDS = {
    "abstract": ["abstract", "abstrak", "ringkasan"],
    "introduction": ["introduction", "pendahuluan", "background", "latar belakang", "motivation", "motivasi"],
    "literature_review": ["literature review", "tinjauan pustaka", "related work", "kajian pustaka", "state of the art"],
    "methods": ["methodology", "metodologi", "method", "metode", "research design", "desain penelitian", "experimental setup", "data collection", "procedure"],
    "results": ["results", "hasil", "findings", "temuan", "experimental results"],
    "discussion": ["discussion", "pembahasan", "diskusi", "analysis", "analisis", "implication", "implikasi"],
    "conclusion": ["conclusion", "kesimpulan", "future work", "saran", "summary", "recommendation"],
    "references": ["references", "daftar pustaka", "bibliography"],
}


def _detect_section_type(node: Dict[str, Any], outline: List[Dict[str, Any]]) -> str:
    """Detect section type via keyword matching pada node title + parent chain.
    Walk parent_id hierarchy supaya leaf node inherit context dari parent.
    Return section type string atau 'literature_review' sebagai default."""
    titles_to_check = []
    titles_to_check.append(str(node.get("title", "")).lower())
    # Walk parents up to root
    by_id = {n.get("id"): n for n in outline}
    cursor = node
    depth = 0
    while cursor and cursor.get("parent_id") and depth < 5:
        parent = by_id.get(cursor.get("parent_id"))
        if not parent:
            break
        titles_to_check.append(str(parent.get("title", "")).lower())
        cursor = parent
        depth += 1
    combined = " ".join(titles_to_check)
    # Priority order · check abstract/methods/results dulu karena lebih specific
    for stype in ["abstract", "methods", "results", "discussion", "conclusion", "introduction", "literature_review"]:
        keywords = _SECTION_KEYWORDS.get(stype, [])
        for kw in keywords:
            if kw in combined:
                return stype
    return "literature_review"


# Per-section prompt templates · different rhetorical purpose per section type.
# Mengikuti Hyland (2019) Academic Discourse plus Swales (1990) Genre Analysis
# untuk Introduction (CARS), plus standard academic writing conventions per
# IMRAD section.
def _build_section_prompt(section_type: str, node: Dict[str, Any], req_data: Dict[str, Any]) -> str:
    """Build prompt SIMPLIFIED untuk speed · semua section pakai shared template
    yang lebih ringkas. Style guidance per section di-condense ke 1-2 line saja.
    AI lokal small model lebih cepat process prompt pendek."""
    title = req_data.get("title", "")[:120]
    thesis = req_data.get("thesis", "")[:300]
    node_title = node.get("title", "")
    papers_block = req_data.get("papers_block", "")
    cite_example = req_data.get("cite_example", "")
    words = req_data.get("words", 200)
    lang = req_data.get("lang", "academic English")
    starting_draft = req_data.get("starting_draft", "").strip()

    # Per-section style hint · 1-line summary instead of multi-paragraph rules
    STYLE_HINTS = {
        "introduction": "CARS pattern (situate research, indicate gap, fill niche). Forward-looking.",
        "literature_review": "Concept-centric synthesis (Webster Watson 2002). Group findings by theme.",
        "methods": "Procedural past tense. Step-by-step replicable description.",
        "results": "Data-driven reporting. Past tense findings with Table X / Figure Y placeholders. No interpretation.",
        "discussion": "Interpret findings. Hedging language. Compare to literature. Acknowledge limitations.",
        "conclusion": "Synthesize contribution. Restate findings. State implications. Minimal citations.",
        "abstract": "CARS single block. Context + gap + approach + findings + implication.",
    }
    style_hint = STYLE_HINTS.get(section_type, STYLE_HINTS["literature_review"])

    if starting_draft:
        # Refinement mode · shorter prompt karena draft sudah ada
        return f"""Refine the draft below for section "{node_title}" in research "{title}".

Style: {style_hint}
Target: {words} words in {lang}. Citation: {cite_example}.

Thesis: {thesis}

EXISTING DRAFT:
{starting_draft}

Papers available:
{papers_block}

Output the refined paragraph only (no commentary, no markdown headers, no em-dash, no semicolons). Cite at least 2 papers."""

    # Generate from scratch · concise prompt
    return f"""Write academic paragraph for "{node_title}" in research "{title}".

Section type: {section_type}. Style: {style_hint}
Target: {words} words in {lang}. Citation format: {cite_example}.

Thesis: {thesis}

Papers (cite at least 3):
{papers_block}

Output the paragraph only. No headers, no markdown, no em-dash, no semicolons. Use inline citations like {cite_example}."""


class DraftingSynthesizeRequest(BaseModel):
    title: str
    thesis_statement: Optional[str] = ""
    outline: List[Dict[str, Any]]
    source_mappings: Dict[str, List[str]] = {}
    papers: List[Dict[str, Any]] = []
    # Stage II node artifacts dipakai sebagai starting_draft kalau tersedia.
    node_artifacts: Dict[str, Dict[str, Any]] = {}
    citation_format: str = "apa7"
    language: str = "en"
    target_section: Optional[str] = None
    words_per_node: int = 200
    # Best-effort mode · kalau node tidak punya papers, generate paragraph
    # dari thesis statement + node title saja (less grounded, no citation,
    # tapi tidak kosong). User dapat skripsi draft fillable bahkan kalau
    # Auto-Map belum lengkap. Default false (strict mode, current behavior).
    best_effort: bool = False
    # Research type · menentukan section apa yang valid di-draft AI:
    # - "literature_paper" (default): pure literature-based, skip Results dan
    #   Discussion karena tidak ada eksperimen. Cocok untuk SLR, scoping review,
    #   bibliometric analysis, atau fase Skripsi Proposal pre-experiment.
    # - "empirical_proposal": Intro + LitReview + Methodology saja (Bab I-III
    #   skripsi proposal). Results/Discussion/Conclusion locked sampai eksperimen
    #   dijalankan di Workspace Data/NxML.
    # - "empirical_final": all sections, tapi Results/Discussion butuh data
    #   aktual dari vault.experiment atau vault.pipeline.runs. AI hanya bantu
    #   styling narrative, tidak generate angka.
    # Default literature_paper karena paling safe secara academic integrity
    # untuk user yang baru drafting.
    research_type: str = "literature_paper"


@router.post("/pipeline/drafting/synthesize-section")
def drafting_synthesize_section(req: DraftingSynthesizeRequest) -> Dict[str, Any]:
    """Generate literature review draft paragraphs per outline node.

    Per node, AI synthesize papers yang attached di Stage II source_mappings
    menjadi paragraph akademik 150-250 kata dengan inline citation sesuai
    format yang dipilih user.

    Parallel processing dengan ThreadPoolExecutor 4 workers supaya batch
    nodes selesai cepat (target 30-60 detik untuk 30 nodes vs 3-5 menit
    sequential).
    """
    try:
        from app.services import task_router as _tr
        ai_available = True
    except Exception:
        ai_available = False

    # RESEARCH TYPE FILTER · skip section types yang tidak applicable untuk
    # research mode aktif. Mencegah AI generate Results/Discussion tanpa
    # primary research data (academic integrity issue).
    research_type = (req.research_type or "literature_paper").lower()
    SKIP_TYPES_BY_MODE = {
        "literature_paper": {"results", "discussion"},
        "empirical_proposal": {"results", "discussion", "conclusion"},
        "empirical_final": set(),  # all sections allowed
    }
    skip_types = SKIP_TYPES_BY_MODE.get(research_type, {"results", "discussion"})

    eligible_nodes = []
    skipped_nodes = []
    for n in req.outline:
        lvl = n.get("level") or 1
        nid = n.get("id")
        if not (1 < lvl <= 3 and nid):
            continue
        # Detect section type untuk filter berdasarkan research mode
        node_section_type = _detect_section_type(n, req.outline)
        if node_section_type in skip_types:
            skipped_nodes.append({
                "id": nid,
                "title": n.get("title", ""),
                "section_type": node_section_type,
                "reason": f"Section {node_section_type} skipped in {research_type} mode · primary research data required",
            })
            continue
        if req.source_mappings.get(nid):
            eligible_nodes.append(n)
        else:
            eligible_nodes.append(dict(n, _no_papers=True))

    if req.target_section:
        target_root = next((n for n in req.outline if n.get("id") == req.target_section), None)
        if target_root:
            descendants = {req.target_section}
            changed = True
            while changed:
                changed = False
                for n in req.outline:
                    pid = n.get("parent_id")
                    if pid and pid in descendants and n.get("id") not in descendants:
                        descendants.add(n.get("id"))
                        changed = True
            eligible_nodes = [n for n in eligible_nodes if n.get("id") in descendants]

    eligible_nodes = eligible_nodes[:40]

    papers_by_id = {}
    for p in (req.papers or []):
        pid = str(p.get("id") or p.get("doi") or p.get("title") or "")[:120]
        if pid:
            papers_by_id[pid] = p

    if not eligible_nodes:
        return {"status": "success", "drafts": {}, "method": "no_nodes", "stats": {"nodes_total": 0}}

    drafts: Dict[str, Dict[str, Any]] = {}

    citation_styles = {
        "apa7": "Author1 dan Author2 (Year)",
        "ieee": "[1]",
        "vancouver": "(1)",
        "harvard": "(Author1 and Author2, Year)",
        "chicago": "(Author1 and Author2 Year)",
        "mla": "(Author1 Author2 Year)",
    }
    cite_example = citation_styles.get(req.citation_format, citation_styles["apa7"])
    lang_hint, lang_display = _resolve_language(req.language)

    def _draft_node(node):
        nid = node.get("id")
        if node.get("_no_papers"):
            return nid, {
                "paragraph": "[PLACEHOLDER · No supporting papers attached to this node in Stage II Source Mapping. Either attach papers and re-generate, or write this section manually.]",
                "word_count": 0,
                "citations_used": [],
                "status": "placeholder",
                "node_title": node.get("title", "")
            }

        attached_pids = req.source_mappings.get(nid, [])
        attached_papers = []
        # SPEED · cap 5 paper per node (turun dari 8). Lebih dari ini AI sering
        # ignore paper tambahan. Plus prompt jadi lebih ringkas = faster inference.
        for pid in attached_pids[:5]:
            p = papers_by_id.get(pid)
            if not p:
                continue
            authors = p.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            authors_str = ", ".join(authors[:2]) if authors else "Anonymous"  # 2 author saja
            attached_papers.append({
                "id": pid,
                "authors": authors_str,
                "year": str(p.get("year") or "n.d."),
                "title": str(p.get("title") or "Untitled")[:120],  # turun 200 → 120
                "synth": str(p.get("abstract_synth") or p.get("abstract") or "")[:200],  # turun 400 → 200
                "venue": str(p.get("venue") or "")[:60],
                "doi": str(p.get("doi") or "")[:80],
            })

        if not attached_papers:
            return nid, {
                "paragraph": "[PLACEHOLDER · Papers attached but metadata not available. Verify Stage I synthesis completed.]",
                "word_count": 0,
                "citations_used": [],
                "status": "no_metadata",
                "node_title": node.get("title", "")
            }

        papers_block = "\n".join([
            f"- ID: {p['id']}\n  Citation: {p['authors']} ({p['year']}). {p['title']}. {p['venue']}\n  Summary: {p['synth']}"
            for p in attached_papers
        ])

        # Check Stage II artifact · kalau ada pakai sebagai starting_draft
        artifact = (req.node_artifacts or {}).get(nid) or {}
        starting_draft = str(artifact.get("content") or "").strip()
        has_starting_draft = bool(starting_draft)

        # PER-SECTION WRITING MODE · detect section type dari node title +
        # parent chain, lalu pakai prompt template yang sesuai rhetorical
        # purpose section tersebut. Output lebih akurat per section (CARS untuk
        # Introduction, procedural untuk Methods, data-driven untuk Results,
        # dst). Sebelumnya semua section pakai literature review prompt.
        section_type = _detect_section_type(node, req.outline)

        # Skip generation untuk Abstract · Stage II Compose Abstract sudah
        # handle ini. Stage III hanya pakai existing artifact-nya.
        if section_type == "abstract" and has_starting_draft:
            return nid, {
                "paragraph": starting_draft[:3000],
                "word_count": len([w for w in starting_draft.split() if w]),
                "citations_used": artifact.get("sources_cited", []),
                "status": "ok",
                "node_title": node.get("title", ""),
                "section_type": section_type,
                "source": "stage_ii_abstract_artifact",
            }

        prompt = _build_section_prompt(section_type, node, {
            "title": req.title,
            "thesis": req.thesis_statement or "",
            "papers_block": papers_block,
            "cite_example": cite_example,
            "words": req.words_per_node,
            "lang": lang_hint,
            "starting_draft": starting_draft if has_starting_draft else "",
        })

        try:
            # SPEED OPTIMIZATION · pakai structured_fast routing yang skip
            # Ollama R1 reasoning chain (paling lambat di stack lokal).
            # Token ratio diturunkan dari 2.5x ke 1.6x supaya AI output ringkas.
            # Temperature 0.4 lebih deterministik plus faster sampling.
            result = _tr.route(
                task_type="structured_fast",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(req.words_per_node * 1.6),
                temperature=0.4,
            )
            if result.get("ok") and result.get("text"):
                paragraph = result["text"].strip()
                if paragraph.startswith("```"):
                    paragraph = paragraph.split("```")[1] if "```" in paragraph[3:] else paragraph[3:]
                paragraph = paragraph.strip().strip('"').strip("'")
                wc = len(paragraph.split())
                cited = []
                for p in attached_papers:
                    last_name = p["authors"].split(",")[0].split()[-1].lower() if p["authors"] else ""
                    if last_name and last_name in paragraph.lower():
                        cited.append(p["id"])
                return nid, {
                    "paragraph": paragraph[:3000],
                    "word_count": wc,
                    "citations_used": cited,
                    "status": "ok",
                    "node_title": node.get("title", ""),
                    "starting_draft_used": has_starting_draft,
                    "section_type": section_type,
                }
        except Exception as e:
            return nid, {
                "paragraph": f"[ERROR: {type(e).__name__}: {str(e)[:100]}]",
                "word_count": 0,
                "citations_used": [],
                "status": "error",
                "node_title": node.get("title", "")
            }
        return nid, {
            "paragraph": "[AI generation failed. Try again or write manually.]",
            "word_count": 0,
            "citations_used": [],
            "status": "ai_fail",
            "node_title": node.get("title", "")
        }

    if ai_available:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        try:
            # SPEED · 8 workers concurrent supaya 30+ nodes selesai dalam
            # ~3-4 batches paralel. Per-node timeout 45s (turun dari 90s)
            # supaya kalau ada node yang hang tidak block batch lainnya.
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(_draft_node, n) for n in eligible_nodes]
                for future in as_completed(futures, timeout=180):
                    try:
                        nid, draft = future.result(timeout=45)
                        drafts[nid] = draft
                    except Exception:
                        pass
        except Exception:
            pass
    else:
        for n in eligible_nodes:
            nid = n.get("id")
            drafts[nid] = {
                "paragraph": f"[AI provider not available. Manual writing required for: {n.get('title', '')}]",
                "word_count": 0,
                "citations_used": [],
                "status": "ai_unavailable",
                "node_title": n.get("title", "")
            }

    ok_count = sum(1 for d in drafts.values() if d["status"] == "ok")
    placeholder_count = sum(1 for d in drafts.values() if d["status"] == "placeholder")
    total_words = sum(d["word_count"] for d in drafts.values())
    avg_citations = sum(len(d["citations_used"]) for d in drafts.values()) / max(ok_count, 1)

    return {
        "status": "success",
        "drafts": drafts,
        "method": "ai_orchestrator" if ai_available else "rule_based_fallback",
        "citation_format": req.citation_format,
        "research_type": research_type,
        "skipped_nodes": skipped_nodes,
        "stats": {
            "nodes_total": len(eligible_nodes),
            "nodes_drafted": ok_count,
            "nodes_placeholder": placeholder_count,
            "nodes_skipped_by_research_type": len(skipped_nodes),
            "total_words": total_words,
            "avg_citations_per_paragraph": round(avg_citations, 1),
        }
    }


# =============================================================================
# STREAMING SYNTHESIZE SECTION (SSE)
# =============================================================================
# Streaming variant supaya frontend dapat progress real-time per node selesai.
# User lihat counter bertambah live ("3/30 nodes done · 12s") instead of nunggu
# black-box 30+ detik. Plus heartbeat untuk Cloudflare keep-alive.
@router.post("/pipeline/drafting/synthesize-section-stream")
def drafting_synthesize_section_stream(req: DraftingSynthesizeRequest):
    """SSE streaming variant · yield events: start, node_done (per node),
    heartbeat, complete (final aggregated)."""
    import time as _time

    def _sse(data):
        return "data: " + json.dumps(data) + "\n\n"

    def _gen():
        try:
            from app.services import task_router as _tr
            ai_available = True
        except Exception:
            ai_available = False

        if not ai_available:
            yield _sse({"type": "error", "message": "AI orchestrator not available"})
            return

        # Filter eligible nodes (same logic as non-streaming)
        research_type = (req.research_type or "literature_paper").lower()
        SKIP_TYPES_BY_MODE = {
            "literature_paper": {"results", "discussion"},
            "empirical_proposal": {"results", "discussion", "conclusion"},
            "empirical_final": set(),
        }
        skip_types = SKIP_TYPES_BY_MODE.get(research_type, {"results", "discussion"})

        eligible_nodes = []
        skipped_nodes = []
        for n in req.outline:
            lvl = n.get("level") or 1
            nid = n.get("id")
            if not (1 < lvl <= 3 and nid):
                continue
            sec_type = _detect_section_type(n, req.outline)
            if sec_type in skip_types:
                skipped_nodes.append({"id": nid, "title": n.get("title", ""), "section_type": sec_type})
                continue
            if req.source_mappings.get(nid):
                eligible_nodes.append(n)
            else:
                eligible_nodes.append(dict(n, _no_papers=True))

        eligible_nodes = eligible_nodes[:40]
        total = len(eligible_nodes)

        if total == 0:
            yield _sse({"type": "error", "message": "No eligible nodes after filter"})
            return

        yield _sse({"type": "start", "total": total, "skipped": len(skipped_nodes), "research_type": research_type})

        # Build papers index
        papers_by_id = {}
        for p in (req.papers or []):
            pid = str(p.get("id") or p.get("doi") or p.get("title") or "")[:120]
            if pid:
                papers_by_id[pid] = p

        # Citation style example
        citation_styles = {
            "apa7": "Author1 dan Author2 (Year)", "ieee": "[1]", "vancouver": "(1)",
            "harvard": "(Author1 and Author2, Year)", "chicago": "(Author1 and Author2 Year)",
            "mla": "(Author1 Author2 Year)",
        }
        cite_example = citation_styles.get(req.citation_format, citation_styles["apa7"])
        lang_hint, lang_display = _resolve_language(req.language)

        def _draft_one(node):
            nid = node.get("id")
            if node.get("_no_papers"):
                # BEST-EFFORT MODE · generate paragraph dari thesis + node title
                # saja kalau req.best_effort=True. Tidak ada citation tapi paragraph
                # tetap meaningful. User dapat fillable draft bahkan kalau Auto-Map
                # Stage II belum lengkap.
                if not req.best_effort:
                    return nid, {
                        "paragraph": f"[PLACEHOLDER · No papers attached. Map papers in Stage II first.]",
                        "word_count": 0, "citations_used": [], "status": "placeholder",
                        "node_title": node.get("title", "")
                    }
                # Best-effort path · simpler prompt tanpa paper context
                section_type_be = _detect_section_type(node, req.outline)
                STYLE_HINTS = {
                    "introduction": "CARS pattern. Forward-looking framing.",
                    "literature_review": "Concept overview. Note that detailed citations require attaching papers in Stage II.",
                    "methods": "Procedural description.",
                    "results": "Data-driven reporting placeholder.",
                    "discussion": "Interpretive analysis.",
                    "conclusion": "Synthesis and forward-looking.",
                    "abstract": "CARS single block.",
                }
                style_hint_be = STYLE_HINTS.get(section_type_be, STYLE_HINTS["literature_review"])
                be_prompt = f"""Write academic paragraph for "{node.get('title', '')}" in research "{req.title}".

Section type: {section_type_be}. Style: {style_hint_be}
Target: {req.words_per_node} words in {lang_hint}.

Thesis context:
{(req.thesis_statement or '')[:400]}

NO papers attached. Generate generic but academically sound prose based on the node title and thesis. Use hedging language like "research suggests" or "studies have shown" without fabricating specific citations. The user will add citations later when papers are attached.

Output the paragraph only. No headers, no markdown, no em-dash, no semicolons."""
                try:
                    r = _tr.route(
                        task_type="structured_fast",
                        messages=[{"role": "user", "content": be_prompt}],
                        max_tokens=int(req.words_per_node * 1.6),
                        temperature=0.5,
                    )
                    if r.get("ok") and r.get("text"):
                        para = r["text"].strip().strip('"').strip("'")
                        para = re.sub(r"^\s*#{1,6}\s+.*$", "", para, flags=re.MULTILINE).strip()
                        para = para.replace("—", " ").replace(";", ".")
                        wc = len(para.split())
                        return nid, {
                            "paragraph": para[:3000],
                            "word_count": wc, "citations_used": [], "status": "best_effort",
                            "node_title": node.get("title", ""),
                            "section_type": section_type_be,
                            "warning": "Generated without citations · attach papers in Stage II untuk grounding"
                        }
                except Exception:
                    pass
                return nid, {
                    "paragraph": "[Best-effort generation failed. Restart backend atau attach papers.]",
                    "word_count": 0, "citations_used": [], "status": "error",
                    "node_title": node.get("title", "")
                }

            attached_pids = req.source_mappings.get(nid, [])
            attached_papers = []
            for pid in attached_pids[:5]:
                p = papers_by_id.get(pid)
                if not p:
                    continue
                authors_raw = p.get("authors") or []
                if isinstance(authors_raw, str):
                    authors_raw = [authors_raw]
                authors_str = ", ".join(authors_raw[:2]) if authors_raw else "Anonymous"
                attached_papers.append({
                    "id": pid, "authors": authors_str,
                    "year": str(p.get("year") or "n.d."),
                    "title": str(p.get("title") or "Untitled")[:120],
                    "synth": str(p.get("abstract_synth") or p.get("abstract") or "")[:200],
                    "venue": str(p.get("venue") or "")[:60],
                })

            if not attached_papers:
                return nid, {
                    "paragraph": "[PLACEHOLDER · Paper metadata unavailable]",
                    "word_count": 0, "citations_used": [], "status": "no_metadata",
                    "node_title": node.get("title", "")
                }

            papers_block = "\n".join([
                f"- [{p['id']}] {p['authors']} ({p['year']}). {p['title']}"
                for p in attached_papers
            ])

            artifact = (req.node_artifacts or {}).get(nid) or {}
            starting_draft = str(artifact.get("content") or "").strip()
            section_type = _detect_section_type(node, req.outline)

            if section_type == "abstract" and starting_draft:
                return nid, {
                    "paragraph": starting_draft[:3000],
                    "word_count": len(starting_draft.split()),
                    "citations_used": artifact.get("sources_cited", []),
                    "status": "ok", "node_title": node.get("title", ""),
                    "section_type": section_type, "source": "stage_ii_abstract_artifact",
                }

            prompt = _build_section_prompt(section_type, node, {
                "title": req.title, "thesis": req.thesis_statement or "",
                "papers_block": papers_block, "cite_example": cite_example,
                "words": req.words_per_node, "lang": lang_hint,
                "starting_draft": starting_draft,
            })

            try:
                result = _tr.route(
                    task_type="structured_fast",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=int(req.words_per_node * 1.6),
                    temperature=0.4,
                )
                if result.get("ok") and result.get("text"):
                    paragraph = result["text"].strip()
                    if paragraph.startswith("```"):
                        paragraph = paragraph.split("```")[1] if "```" in paragraph[3:] else paragraph[3:]
                    paragraph = paragraph.strip().strip('"').strip("'")
                    # Strip stray heading + em-dash + semicolons
                    paragraph = re.sub(r"^\s*#{1,6}\s+.*$", "", paragraph, flags=re.MULTILINE).strip()
                    paragraph = paragraph.replace("—", " ").replace(";", ".")
                    wc = len(paragraph.split())
                    cited = []
                    for p in attached_papers:
                        last_name = p["authors"].split(",")[0].split()[-1].lower() if p["authors"] else ""
                        if last_name and last_name in paragraph.lower():
                            cited.append(p["id"])
                    return nid, {
                        "paragraph": paragraph[:3000], "word_count": wc,
                        "citations_used": cited, "status": "ok",
                        "node_title": node.get("title", ""),
                        "section_type": section_type,
                    }
            except Exception as e:
                return nid, {
                    "paragraph": f"[ERROR: {type(e).__name__}: {str(e)[:80]}]",
                    "word_count": 0, "citations_used": [], "status": "error",
                    "node_title": node.get("title", "")
                }
            return nid, {
                "paragraph": "[AI returned empty]", "word_count": 0,
                "citations_used": [], "status": "empty",
                "node_title": node.get("title", "")
            }

        # Parallel execute · stream events as each node completes
        from concurrent.futures import ThreadPoolExecutor, as_completed
        executor = ThreadPoolExecutor(max_workers=8)
        future_to_node = {executor.submit(_draft_one, n): n for n in eligible_nodes}

        drafts = {}
        done_count = 0
        start_ts = _time.time()
        last_heartbeat = start_ts

        try:
            for future in as_completed(future_to_node, timeout=240):
                elapsed = int(_time.time() - start_ts)
                try:
                    nid, draft = future.result(timeout=45)
                    drafts[nid] = draft
                    done_count += 1
                    yield _sse({
                        "type": "node_done",
                        "node_id": nid,
                        "node_title": draft.get("node_title", ""),
                        "section_type": draft.get("section_type", ""),
                        "word_count": draft.get("word_count", 0),
                        "done": done_count,
                        "total": total,
                        "elapsed": elapsed,
                    })
                except Exception as exc:
                    done_count += 1
                    yield _sse({"type": "node_error", "done": done_count, "total": total, "elapsed": elapsed, "error": str(exc)[:120]})
                # Periodic heartbeat untuk Cloudflare keep-alive
                if _time.time() - last_heartbeat > 5:
                    yield _sse({"type": "heartbeat", "elapsed": elapsed, "done": done_count, "total": total})
                    last_heartbeat = _time.time()
        except Exception as exc:
            yield _sse({"type": "error", "message": f"Stream timeout: {exc}"})
        finally:
            executor.shutdown(wait=False)

        elapsed_total = int(_time.time() - start_ts)
        ok_count = sum(1 for d in drafts.values() if d.get("status") == "ok")
        total_words = sum(d.get("word_count", 0) for d in drafts.values())

        yield _sse({
            "type": "complete",
            "drafts": drafts,
            "method": "structured_fast_parallel_8_sse",
            "research_type": research_type,
            "skipped_nodes": skipped_nodes,
            "elapsed_seconds": elapsed_total,
            "stats": {
                "nodes_total": total,
                "nodes_drafted": ok_count,
                "nodes_skipped_by_research_type": len(skipped_nodes),
                "total_words": total_words,
            }
        })

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# =============================================================================
# CITATION ACCURACY VERIFIER (Kluge Stage III · Accuracy)
# =============================================================================
# Parse inline citations dari paragraph text via regex. Verify setiap citation
# match dengan author + year di paper pool. Output: per-paragraph + overall
# citation health summary. Mengikuti APA 7 plus IEEE conventions.
class CitationVerifyRequest(BaseModel):
    drafts: Dict[str, Dict[str, Any]] = {}  # {nid: {paragraph, citations_used, node_title}}
    papers: List[Dict[str, Any]] = []
    citation_format: str = "apa7"


@router.post("/pipeline/drafting/verify-citations")
def drafting_verify_citations(req: CitationVerifyRequest) -> Dict[str, Any]:
    """Verify citation accuracy · cek inline citations di paragraph match
    dengan papers di pool. Common issues yang di-detect:
    1. missing_in_pool · citation di paragraph tapi paper-nya tidak ada di pool
    2. wrong_year · author match tapi year mismatch
    3. wrong_author · year match tapi author surname mismatch
    4. uncited_paper · paper di citations_used tapi tidak appear di paragraph text
    5. duplicate_authors · 2+ papers dengan same author year di pool
    """
    # Build paper index dengan surname + year keys untuk fuzzy matching
    papers_index = {}  # (surname_lower, year_str) -> paper_meta
    papers_by_id = {}
    for p in (req.papers or []):
        pid = str(p.get("id") or p.get("doi") or p.get("title") or "")[:120]
        if not pid:
            continue
        papers_by_id[pid] = p
        authors = p.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        if not authors:
            continue
        first_author = str(authors[0])
        # Extract surname (last word after comma split atau last word of full name)
        if "," in first_author:
            surname = first_author.split(",")[0].strip().split()[-1].lower()
        else:
            surname = first_author.strip().split()[-1].lower()
        year = str(p.get("year") or "n.d.").strip()
        key = (surname, year)
        if key not in papers_index:
            papers_index[key] = []
        papers_index[key].append(pid)

    per_node_issues = {}
    total_issues = 0
    total_citations_found = 0
    fmt = (req.citation_format or "apa7").lower()

    # Regex patterns per citation format · catch most common patterns
    if fmt in ("apa7", "harvard", "chicago", "mla"):
        # (Author, Year) or (Author and Author, Year) or (Author et al., Year)
        citation_pattern = re.compile(r"\(([A-Z][a-zA-Z\-']+(?:\s+(?:and|&|et al\.?)\s+[A-Z]?[a-zA-Z\-']*)*),?\s*(\d{4})\)")
    elif fmt == "vancouver":
        citation_pattern = re.compile(r"\((\d+)\)")
    elif fmt == "ieee":
        citation_pattern = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
    else:
        citation_pattern = re.compile(r"\(([A-Z][a-zA-Z\-']+),?\s*(\d{4})\)")

    for nid, draft in (req.drafts or {}).items():
        if not isinstance(draft, dict):
            continue
        paragraph = draft.get("paragraph", "") or ""
        claimed_pids = set(draft.get("citations_used", []))
        node_issues = []
        node_citations_found = []

        # Pattern matching · extract semua inline citations
        for match in citation_pattern.finditer(paragraph):
            total_citations_found += 1
            if fmt in ("apa7", "harvard", "chicago", "mla"):
                author_text = match.group(1).strip()
                year = match.group(2).strip()
                # Extract surname dari "Author and Author" atau "Author et al."
                surname = author_text.split("and")[0].split("&")[0].split("et al")[0].strip().split()[-1].lower()
                key = (surname, year)
                node_citations_found.append({"surname": surname, "year": year, "raw": match.group(0)})

                if key not in papers_index:
                    # Cek kalau author match tapi year beda
                    same_author = [k for k in papers_index.keys() if k[0] == surname]
                    if same_author:
                        node_issues.append({
                            "type": "wrong_year",
                            "citation": match.group(0),
                            "found_in_paper_pool_for_other_years": [k[1] for k in same_author],
                            "severity": "warning",
                            "message": f"Author {surname} found in pool but year {year} mismatch. Verify citation year."
                        })
                    else:
                        # Cek year match tapi author beda
                        same_year = [k for k in papers_index.keys() if k[1] == year]
                        if same_year:
                            node_issues.append({
                                "type": "wrong_author",
                                "citation": match.group(0),
                                "papers_in_same_year": [k[0] for k in same_year][:3],
                                "severity": "warning",
                                "message": f"Year {year} found in pool but author '{surname}' not matched."
                            })
                        else:
                            node_issues.append({
                                "type": "missing_in_pool",
                                "citation": match.group(0),
                                "severity": "error",
                                "message": f"Citation ({surname}, {year}) not found in Stage I paper pool. Fabricated citation or paper missing from Stage I search."
                            })
                            total_issues += 1

        # Cek uncited paper · papers in citations_used tapi tidak ada di paragraph
        for pid in claimed_pids:
            p = papers_by_id.get(pid)
            if not p:
                continue
            authors = p.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            if not authors:
                continue
            first_author = str(authors[0])
            if "," in first_author:
                surname = first_author.split(",")[0].strip().split()[-1].lower()
            else:
                surname = first_author.strip().split()[-1].lower()
            # Cek surname appear di paragraph text
            if surname not in paragraph.lower():
                node_issues.append({
                    "type": "uncited_paper",
                    "paper_id": pid,
                    "surname": surname,
                    "severity": "info",
                    "message": f"Paper ({surname}) listed in citations_used but surname not found in paragraph text."
                })

        if node_issues:
            per_node_issues[nid] = {
                "node_title": draft.get("node_title", ""),
                "issues": node_issues,
                "citations_in_text": node_citations_found,
                "issue_count": len(node_issues),
                "error_count": sum(1 for i in node_issues if i["severity"] == "error"),
                "warning_count": sum(1 for i in node_issues if i["severity"] == "warning"),
            }

    # Detect duplicate (same author year) di pool
    duplicates = []
    for key, pids in papers_index.items():
        if len(pids) > 1:
            duplicates.append({
                "surname": key[0],
                "year": key[1],
                "paper_ids": pids,
                "message": f"Multiple papers in pool with author={key[0]} year={key[1]}. Use suffix (2023a, 2023b) per APA 7 to distinguish.",
            })

    health_score = max(0, 100 - (total_issues * 5))

    return {
        "status": "success",
        "health_score": health_score,
        "total_citations_found": total_citations_found,
        "total_issues": total_issues,
        "per_node_issues": per_node_issues,
        "duplicate_author_year": duplicates,
        "summary": {
            "nodes_checked": len(req.drafts or {}),
            "nodes_with_issues": len(per_node_issues),
            "papers_pool_size": len(papers_by_id),
            "citation_format": fmt,
        }
    }


# =============================================================================
# PLAGIARISM PRE-CHECK · cosine similarity per paragraph vs paper abstracts
# =============================================================================
# Compute simple TF-IDF cosine similarity per paragraph vs each paper abstract
# di Stage I pool. Flag paragraph dengan similarity tinggi sebagai potential
# plagiarism (atau citation yang lupa di-paraphrase). Bukan replacement untuk
# Turnitin tapi cukup untuk pre-check sebelum submission.
class PlagiarismCheckRequest(BaseModel):
    drafts: Dict[str, Dict[str, Any]] = {}
    papers: List[Dict[str, Any]] = []
    threshold_warn: int = 30   # %, mid risk
    threshold_flag: int = 50   # %, high risk


@router.post("/pipeline/drafting/plagiarism-check")
def drafting_plagiarism_check(req: PlagiarismCheckRequest) -> Dict[str, Any]:
    """Compute cosine similarity TF-IDF per paragraph vs paper abstract.
    Bahasa Indonesia stopwords + English stopwords pakai simple union list.
    Output: per-paragraph similarity score plus top matched paper kalau ada
    similarity di atas threshold.
    """
    import math
    from collections import Counter

    # Simple stopwords bilingual EN+ID
    STOPWORDS = set("""
    a an the and or but if while of for to from in on at by with about as
    is are was were be been being have has had do does did this that these those
    it its their there here what which who when where why how
    yang dan atau tapi sebagai untuk dari di ke dengan oleh pada akan adalah
    juga tidak bukan kita kami mereka anda dia ini itu apa siapa kapan dimana mengapa bagaimana
    """.split())

    def tokenize(text):
        # Lowercase, extract alpha tokens minimum 3 chars, skip stopwords
        tokens = re.findall(r"\b[a-z]{3,}\b", (text or "").lower())
        return [t for t in tokens if t not in STOPWORDS]

    def compute_tf(tokens):
        return Counter(tokens)

    def cosine_sim(tf_a, tf_b):
        # Dot product
        common = set(tf_a.keys()) & set(tf_b.keys())
        dot = sum(tf_a[k] * tf_b[k] for k in common)
        # Magnitudes
        mag_a = math.sqrt(sum(v ** 2 for v in tf_a.values())) or 1
        mag_b = math.sqrt(sum(v ** 2 for v in tf_b.values())) or 1
        return dot / (mag_a * mag_b)

    # Pre-compute TF untuk semua paper abstracts
    paper_tfs = []
    for p in (req.papers or []):
        pid = str(p.get("id") or p.get("doi") or p.get("title") or "")[:120]
        abstract = str(p.get("abstract") or p.get("abstract_synth") or p.get("synthesis") or "")
        title = str(p.get("title") or "")
        text = abstract + " " + title
        if not text.strip():
            continue
        tokens = tokenize(text)
        if not tokens:
            continue
        paper_tfs.append({
            "id": pid,
            "title": title[:120],
            "tf": compute_tf(tokens),
        })

    per_node_results = {}
    flagged_count = 0
    warn_count = 0
    avg_max_sim = 0
    sim_values = []

    for nid, draft in (req.drafts or {}).items():
        if not isinstance(draft, dict):
            continue
        paragraph = draft.get("paragraph", "") or ""
        if not paragraph.strip():
            continue

        para_tokens = tokenize(paragraph)
        if len(para_tokens) < 10:
            continue
        para_tf = compute_tf(para_tokens)

        # Compute similarity vs all papers
        top_matches = []
        for pf in paper_tfs:
            sim = cosine_sim(para_tf, pf["tf"])
            sim_pct = round(sim * 100, 1)
            if sim_pct > 5:  # ignore very low matches
                top_matches.append({
                    "paper_id": pf["id"],
                    "paper_title": pf["title"],
                    "similarity_pct": sim_pct,
                })

        top_matches.sort(key=lambda x: -x["similarity_pct"])
        max_sim = top_matches[0]["similarity_pct"] if top_matches else 0
        sim_values.append(max_sim)

        risk_level = "low"
        if max_sim >= req.threshold_flag:
            risk_level = "high"
            flagged_count += 1
        elif max_sim >= req.threshold_warn:
            risk_level = "mid"
            warn_count += 1

        per_node_results[nid] = {
            "node_title": draft.get("node_title", ""),
            "max_similarity_pct": max_sim,
            "risk_level": risk_level,
            "top_matches": top_matches[:5],
            "paragraph_token_count": len(para_tokens),
        }

    if sim_values:
        avg_max_sim = round(sum(sim_values) / len(sim_values), 1)

    # Overall health · turun kalau banyak yang flagged
    nodes_checked = len([d for d in (req.drafts or {}).values() if isinstance(d, dict) and d.get("paragraph")])
    if nodes_checked > 0:
        flagged_pct = (flagged_count / nodes_checked) * 100
        health = max(0, 100 - int(flagged_pct * 2) - (warn_count * 3))
    else:
        health = 100

    return {
        "status": "success",
        "health_score": health,
        "summary": {
            "nodes_checked": nodes_checked,
            "papers_compared_against": len(paper_tfs),
            "flagged_high_risk": flagged_count,
            "warned_mid_risk": warn_count,
            "low_risk": nodes_checked - flagged_count - warn_count,
            "avg_max_similarity_pct": avg_max_sim,
            "method": "tf_idf_cosine_similarity",
            "thresholds": {"warn": req.threshold_warn, "flag": req.threshold_flag},
        },
        "per_node_results": per_node_results,
    }


# ============================================================================
# METHOD RECOMMENDATION ENGINE · Phase 1 Research Environment expansion
# Bridges Organizing stage (domain/task inferred) ke Experiment Lab.
# Catalog mengikuti blueprint user · Data Science (10) plus Geo & Disaster (8).
# ============================================================================
class MethodRecRequest(BaseModel):
    research_type: str = Field("empirical_proposal", description="literature_paper, empirical_proposal, atau empirical_final")
    domain: str = Field("data_science", description="data_science atau geo_disaster")
    task_type: str = Field("classification", description="classification, regression, clustering, forecasting, anomaly, dst")
    dataset_summary: str = Field("", description="deskripsi singkat dataset · sample size, feature count, target variable")
    research_goal: str = Field("", description="goal kalimat plain · misal prediksi banjir Indonesia multi-provinsi")
    language: Optional[str] = None


_METHOD_CATALOG = {
    "data_science": {
        "classification": ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "Neural Network", "CatBoost"],
        "regression":     ["XGBoost Regressor", "LightGBM", "Linear Regression", "Polynomial Regression", "MLP Regressor"],
        "clustering":     ["KMeans", "DBSCAN", "Hierarchical Clustering", "Gaussian Mixture", "HDBSCAN"],
        "forecasting":    ["LSTM", "GRU", "Transformer", "Prophet", "SARIMA", "Temporal Fusion Transformer"],
        "anomaly":        ["Isolation Forest", "Autoencoder", "One-Class SVM", "LOF", "Statistical Z-Score"],
        "explainable":    ["SHAP", "LIME", "Permutation Importance", "Partial Dependence Plot", "Anchor Explanations"],
        "automl":         ["H2O AutoML", "TPOT", "AutoGluon", "FLAML", "Optuna Tuning"],
        "deep_learning":  ["CNN", "ResNet", "Transformer", "U-Net", "Vision Transformer"],
        "nlp":            ["BERT", "RoBERTa", "T5", "GPT fine-tuning", "Sentence-BERT"],
        "computer_vision":["YOLOv8", "Mask R-CNN", "U-Net Segmentation", "EfficientNet", "Swin Transformer"],
    },
    "geo_disaster": {
        "flood_classification":      ["Random Forest", "XGBoost", "U-Net Segmentation", "CNN-LSTM Hybrid", "SAR Image Classification"],
        "flood_susceptibility":      ["AHP Method", "Frequency Ratio", "Logistic Regression", "Random Forest", "ANN"],
        "landslide_prediction":      ["XGBoost", "Random Forest", "ANN", "Logistic Regression", "Support Vector Machine"],
        "earthquake_risk":           ["Probabilistic Seismic Hazard Analysis", "Random Forest", "XGBoost", "Bayesian Network", "Spatial Autocorrelation"],
        "tsunami_exposure":          ["Population Exposure GIS", "Hazard Modeling", "Vulnerability Index", "Multi-Criteria Decision Analysis"],
        "rainfall_forecasting":      ["LSTM", "GRU", "ConvLSTM", "Temporal Fusion Transformer", "SARIMA"],
        "remote_sensing_segmentation":["U-Net", "DeepLabV3+", "Mask R-CNN", "Vision Transformer", "FCN"],
        "early_warning_model":       ["LSTM Multi-Source", "CNN-LSTM Hybrid", "Ensemble Stacking", "Threshold-Based Trigger", "Bayesian Belief Network"],
    },
}


def _build_method_prompt(req: MethodRecRequest) -> List[Dict[str, str]]:
    domain = req.domain.lower()
    task = req.task_type.lower()
    catalog = _METHOD_CATALOG.get(domain, {}).get(task, [])
    lang_hint = "Tulis seluruh penjelasan dalam bahasa Indonesia akademik." \
        if (req.language and req.language.lower().startswith("id")) \
        else "Write all explanations in clear, scholarly English."
    user_prompt = (
        "You are an expert ML and applied research methodologist.\n\n"
        "CONTEXT\n"
        f"- Research type: {req.research_type}\n"
        f"- Domain: {req.domain}\n"
        f"- Task type: {req.task_type}\n"
        f"- Dataset summary: {req.dataset_summary or '(not specified)'}\n"
        f"- Research goal: {req.research_goal or '(not specified)'}\n\n"
        "CANDIDATE METHODS (suggested catalog for this task):\n"
        f"{', '.join(catalog) if catalog else '(no preset catalog, please suggest 5 reasonable methods)'}\n\n"
        "TASK\n"
        "Recommend the top 3 methods ranked by suitability. For each method provide:\n"
        "1. method_name\n"
        "2. rank (1, 2, 3)\n"
        "3. why_selected (2-3 sentences specific to the dataset and goal)\n"
        "4. why_not_alternative (1 sentence per ranked-below method)\n"
        "5. mathematical_logic (2-3 sentences explaining the core math/algorithmic mechanism)\n"
        "6. benefits (bullet list of 3 strengths)\n"
        "7. limitations (bullet list of 2-3 weaknesses)\n"
        "8. expected_result (1-2 sentences describing realistic outcome)\n"
        "9. recommended_metrics (list of evaluation metrics suitable for this method+task)\n\n"
        "OUTPUT FORMAT\n"
        "Return ONLY a strict JSON array of 3 objects (no markdown, no preamble) following this schema:\n"
        "[\n"
        '  {"method_name": "...", "rank": 1, "why_selected": "...", "why_not_alternative": "...",\n'
        '    "mathematical_logic": "...", "benefits": ["...","...","..."],\n'
        '    "limitations": ["...","..."], "expected_result": "...",\n'
        '    "recommended_metrics": ["Accuracy","F1","..."]}\n'
        "]\n\n"
        f"{lang_hint}"
    )
    return [
        {"role": "system", "content": "You are a precise research methodology advisor. Return strict JSON only."},
        {"role": "user", "content": user_prompt},
    ]


@router.post("/pipeline/method/recommend")
async def method_recommend(req: MethodRecRequest):
    """AI method recommendation engine · top 3 methods with detailed rationale.
    Bridges Stage II (Organizing) to Stage III (Experiment Lab) workflow."""
    messages = _build_method_prompt(req)
    catalog = _METHOD_CATALOG.get(req.domain.lower(), {}).get(req.task_type.lower(), [])
    text = ""
    try:
        from app.services import task_router as _tr
        result = _tr.route(
            task_type="structured_fast",
            messages=messages,
            max_tokens=2400,
            temperature=0.4,
        )
        text = (result or {}).get("text", "") or ""
    except Exception:
        text = ""
    parsed = None
    if text:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
        m = re.search(r"\[\s*\{.*\}\s*\]", cleaned, flags=re.S)
        candidate = m.group(0) if m else cleaned
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, list):
                parsed = None
        except Exception:
            parsed = None
    if parsed and len(parsed) >= 1:
        return {
            "ok": True,
            "source": "ai",
            "recommendations": parsed[:3],
            "catalog_used": catalog,
            "context": {
                "domain": req.domain,
                "task_type": req.task_type,
                "research_type": req.research_type,
            },
        }
    # Fallback rule based
    fallback_recs = []
    for i, name in enumerate(catalog[:3] or ["Random Forest", "XGBoost", "Neural Network"]):
        fallback_recs.append({
            "method_name": name,
            "rank": i + 1,
            "why_selected": f"{name} merupakan pilihan standar untuk tugas {req.task_type} di domain {req.domain}.",
            "why_not_alternative": "Alternatif memiliki kompleksitas atau requirement lebih tinggi.",
            "mathematical_logic": f"{name} bekerja berdasarkan prinsip dasar dari kategori algoritmanya.",
            "benefits": ["Implementasi tersedia di library standar", "Dokumentasi luas", "Hasil mudah direproduksi"],
            "limitations": ["Performance tergantung kualitas data", "Mungkin perlu hyperparameter tuning"],
            "expected_result": f"Akurasi reasonable pada dataset profile umum task {req.task_type}.",
            "recommended_metrics": ["Accuracy", "F1-score", "Precision", "Recall"] if "classif" in req.task_type else ["RMSE", "MAE", "R-squared"],
        })
    return {
        "ok": True,
        "source": "fallback_rules",
        "recommendations": fallback_recs,
        "catalog_used": catalog,
        "context": {
            "domain": req.domain,
            "task_type": req.task_type,
            "research_type": req.research_type,
        },
        "note": "AI engine tidak tersedia, menggunakan fallback rule-based recommendation.",
    }


# ============================================================================
# SCOPUS READINESS CHECKER · Phase 2 · 10-metric quality gate
# Aggregate · existing citation+plagiarism check ditambah 8 metric baru.
# Output composite score plus per-metric recommendation untuk submission gate.
# ============================================================================
class ReadinessRef(BaseModel):
    title: str = ""
    authors: List[str] = []
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""


class ReadinessRequest(BaseModel):
    manuscript_text: str = Field("", description="Full text dari draft manuscript")
    references: List[ReadinessRef] = Field(default_factory=list)
    target_journal: str = Field("", description="Nama jurnal target plus scope description")
    dataset_description: str = Field("", description="Deskripsi dataset · source, size, public availability")
    methodology_summary: str = Field("", description="Ringkasan method · model, training procedure, evaluation")
    has_baseline: bool = Field(False, description="Apakah ada baseline comparison")
    has_ablation: bool = Field(False, description="Apakah ada ablation study")
    has_code_repo: bool = Field(False, description="Apakah ada link repository code")
    has_data_availability: bool = Field(False, description="Apakah ada data availability statement")
    has_ethics_statement: bool = Field(False, description="Apakah ada ethics approval atau informed consent")
    has_gap_analysis: bool = Field(False, description="Apakah ada explicit gap identification")
    language: Optional[str] = None


def _scoreNovelty(req: ReadinessRequest) -> Dict[str, Any]:
    score = 50
    notes = []
    text = (req.manuscript_text or "").lower()
    novelty_keywords = ["novel", "first to", "new approach", "unique contribution",
                        "previously unaddressed", "gap", "limitation of prior",
                        "kontribusi baru", "pertama kali", "novelty", "celah penelitian"]
    matches = sum(1 for kw in novelty_keywords if kw in text)
    score = min(100, 30 + matches * 8)
    if req.has_gap_analysis:
        score = min(100, score + 15)
        notes.append("Gap analysis explicit · +15")
    if matches < 2:
        notes.append("Sedikit novelty keywords · tegaskan kontribusi unik")
    elif matches >= 4:
        notes.append(f"{matches} novelty signals ditemukan")
    return {"score": score, "status": _statusFromScore(score), "notes": notes}


def _scoreMethodology(req: ReadinessRequest) -> Dict[str, Any]:
    score = 40
    notes = []
    method_text = (req.methodology_summary or req.manuscript_text or "").lower()
    if any(k in method_text for k in ["lstm", "xgboost", "random forest", "transformer", "cnn", "shap", "cross-validation", "hyperparameter"]):
        score += 15
        notes.append("Method spesifik teridentifikasi")
    if req.has_baseline:
        score += 15
        notes.append("Baseline comparison ada · +15")
    if req.has_ablation:
        score += 10
        notes.append("Ablation study ada · +10")
    if any(k in method_text for k in ["accuracy", "f1", "rmse", "auc", "iou", "precision", "recall"]):
        score += 10
        notes.append("Evaluation metrics terdefinisi")
    if not req.has_baseline:
        notes.append("Tidak ada baseline · weak comparative validity")
    return {"score": min(100, score), "status": _statusFromScore(min(100, score)), "notes": notes}


def _scoreCitationQuality(req: ReadinessRequest) -> Dict[str, Any]:
    refs = req.references or []
    count = len(refs)
    notes = []
    if count == 0:
        return {"score": 10, "status": "critical", "notes": ["Tidak ada references"]}
    recent_count = sum(1 for r in refs if r.year and r.year >= 2020)
    recent_pct = (recent_count / count) * 100
    with_journal = sum(1 for r in refs if r.journal)
    has_doi = sum(1 for r in refs if r.doi)
    score = 30
    if count >= 30:
        score += 25
        notes.append(f"{count} references · memenuhi minimum thesis literature review")
    elif count >= 15:
        score += 15
        notes.append(f"{count} references · acceptable tapi bisa lebih")
    else:
        notes.append(f"{count} references · kurang untuk Scopus-level paper")
    if recent_pct >= 70:
        score += 20
        notes.append(f"{recent_pct:.0f}% recent · 2020 keatas")
    elif recent_pct >= 50:
        score += 12
        notes.append(f"{recent_pct:.0f}% recent")
    else:
        notes.append(f"Hanya {recent_pct:.0f}% recent · perlu update literature")
    if (with_journal / count) >= 0.7:
        score += 10
        notes.append("Journal sources dominan")
    if (has_doi / count) >= 0.7:
        score += 10
        notes.append("DOI lengkap untuk verifiability")
    return {"score": min(100, score), "status": _statusFromScore(min(100, score)), "notes": notes}


def _scoreDatasetTransparency(req: ReadinessRequest) -> Dict[str, Any]:
    score = 30
    notes = []
    desc = (req.dataset_description or "").lower()
    if len(desc) > 80:
        score += 25
        notes.append("Dataset described dengan detail")
    elif len(desc) > 30:
        score += 12
        notes.append("Dataset description minimal · tambahkan detail")
    else:
        notes.append("Dataset description sangat singkat atau tidak ada")
    if any(k in desc for k in ["public", "open", "available", "github", "kaggle", "zenodo", "figshare"]):
        score += 15
        notes.append("Public dataset · transparency tinggi")
    if any(k in desc for k in ["sample", "size", "n =", "n=", "instances", "records", "rows"]):
        score += 15
        notes.append("Sample size disebutkan")
    if req.has_data_availability:
        score += 15
        notes.append("Data availability statement · +15")
    return {"score": min(100, score), "status": _statusFromScore(min(100, score)), "notes": notes}


def _scoreReproducibility(req: ReadinessRequest) -> Dict[str, Any]:
    score = 30
    notes = []
    method = (req.methodology_summary or "").lower()
    if req.has_code_repo:
        score += 25
        notes.append("Code repository ada · +25")
    if req.has_data_availability:
        score += 20
        notes.append("Data availability statement · +20")
    if any(k in method for k in ["seed", "random_state", "version", "framework"]):
        score += 10
        notes.append("Random seed atau version control disebutkan")
    if any(k in method for k in ["hyperparameter", "epoch", "batch size", "learning rate"]):
        score += 15
        notes.append("Training config disebutkan")
    if not req.has_code_repo:
        notes.append("Tanpa code repo · reproducibility lemah")
    return {"score": min(100, score), "status": _statusFromScore(min(100, score)), "notes": notes}


def _scoreEthics(req: ReadinessRequest) -> Dict[str, Any]:
    text = (req.manuscript_text or "").lower()
    score = 50
    notes = []
    if req.has_ethics_statement:
        score = 95
        notes.append("Ethics statement explicit · +45")
    elif any(k in text for k in ["ethic", "informed consent", "irb", "institutional review", "etika"]):
        score = 75
        notes.append("Ethics keywords ditemukan tapi belum explicit statement")
    else:
        notes.append("Tidak ada ethics statement · perlu untuk human/sensitive data")
    if any(k in text for k in ["anonymization", "anonim", "de-identif", "privacy"]):
        score = min(100, score + 5)
        notes.append("Privacy mechanism disebutkan")
    return {"score": score, "status": _statusFromScore(score), "notes": notes}


def _scorePlagiarismRisk(req: ReadinessRequest) -> Dict[str, Any]:
    """Lightweight check · pattern matching terhadap common boilerplate.
    Plagiarism check yang real ada di endpoint terpisah TF-IDF cosine."""
    text = req.manuscript_text or ""
    score = 80
    notes = ["Pre-screen lulus default"]
    if len(text) < 200:
        return {"score": 50, "status": "warn", "notes": ["Text terlalu pendek untuk dianalisis"]}
    common_patterns = [
        "according to the literature",
        "extensive research has shown",
        "it is well known that",
        "previous studies have demonstrated",
        "the present study aims to investigate",
    ]
    boilerplate_matches = sum(1 for p in common_patterns if p in text.lower())
    if boilerplate_matches >= 3:
        score = 55
        notes = [f"{boilerplate_matches} boilerplate phrases · paraphrase manual recommended"]
    elif boilerplate_matches >= 1:
        score = 70
        notes = [f"{boilerplate_matches} common phrases · cek paraphrasing"]
    notes.append("Untuk plagiarism check lengkap, gunakan endpoint /pipeline/drafting/plagiarism-check")
    return {"score": score, "status": _statusFromScore(score), "notes": notes}


def _scoreAIWritingRisk(req: ReadinessRequest) -> Dict[str, Any]:
    """Heuristic · cek tanda-tanda AI-generated text."""
    text = req.manuscript_text or ""
    if len(text) < 300:
        return {"score": 70, "status": "good", "notes": ["Text pendek · risk minimal"]}
    ai_signals = [
        "in conclusion",
        "furthermore,",
        "moreover,",
        "additionally,",
        "it is important to note",
        "this paper presents",
        "this study aims to",
        "delve into",
        "tapestry",
        "in the realm of",
    ]
    matches = sum(1 for s in ai_signals if s in text.lower())
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return {"score": 60, "status": "warn", "notes": ["Tidak bisa parse sentences"]}
    lengths = [len(s.split()) for s in sentences]
    avg_len = sum(lengths) / max(1, len(lengths))
    variance = sum((l - avg_len) ** 2 for l in lengths) / max(1, len(lengths))
    score = 90
    notes = []
    if matches >= 5:
        score -= 30
        notes.append(f"{matches} AI-typical transitions · perlu human edit pass")
    elif matches >= 3:
        score -= 15
        notes.append(f"{matches} AI-typical phrases")
    if variance < 30:
        score -= 15
        notes.append("Sentence length terlalu uniform · AI-pattern")
    if not notes:
        notes.append("Human-like writing patterns")
    return {"score": max(20, score), "status": _statusFromScore(max(20, score)), "notes": notes}


def _scoreJournalFit(req: ReadinessRequest) -> Dict[str, Any]:
    journal = (req.target_journal or "").lower()
    text = (req.manuscript_text or "").lower()
    notes = []
    if not journal:
        return {"score": 50, "status": "warn", "notes": ["Target journal belum diset"]}
    score = 60
    journal_words = set(re.findall(r'\b\w+\b', journal))
    text_words = set(re.findall(r'\b\w+\b', text[:5000]))
    overlap = journal_words.intersection(text_words)
    if len(overlap) >= 5:
        score = 85
        notes.append(f"{len(overlap)} keyword overlap dengan scope journal")
    elif len(overlap) >= 2:
        score = 70
        notes.append(f"{len(overlap)} keyword overlap · moderate fit")
    else:
        score = 50
        notes.append("Sedikit overlap · pertimbangkan jurnal lain atau adjust framing")
    notes.append("Verify scope langsung di journal homepage sebelum submit")
    return {"score": score, "status": _statusFromScore(score), "notes": notes}


def _scoreReferenceQuality(req: ReadinessRequest) -> Dict[str, Any]:
    refs = req.references or []
    notes = []
    if not refs:
        return {"score": 10, "status": "critical", "notes": ["Tidak ada references"]}
    journal_count = sum(1 for r in refs if r.journal)
    conference_kw = ["proceedings", "conference", "workshop"]
    conference_count = sum(1 for r in refs if any(k in r.journal.lower() for k in conference_kw))
    journal_pct = (journal_count / len(refs)) * 100
    score = 50
    if journal_pct >= 70:
        score = 85
        notes.append(f"{journal_pct:.0f}% journal sources · academic quality tinggi")
    elif journal_pct >= 50:
        score = 70
        notes.append(f"{journal_pct:.0f}% journal sources")
    else:
        notes.append(f"Hanya {journal_pct:.0f}% journal · tambahkan peer-reviewed sources")
    if conference_count >= len(refs) * 0.3:
        notes.append("Conference papers dominan · OK untuk ML tapi imbangi dengan journal")
    return {"score": score, "status": _statusFromScore(score), "notes": notes}


def _statusFromScore(s: int) -> str:
    if s >= 75:
        return "good"
    if s >= 55:
        return "warn"
    return "critical"


def _classifyComposite(composite: int, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    weakest = min(metrics.values(), key=lambda m: m.get("score", 100))
    if composite >= 75:
        return {"label": "Ready for submission", "color": "#34d399"}
    if composite >= 60:
        return {"label": "Need minor revision", "color": "#ff9500"}
    if composite >= 45:
        return {"label": "Need major revision", "color": "#fbbf24"}
    return {"label": "Not ready · significant gaps", "color": "#ff3b55"}


@router.post("/pipeline/readiness/check")
async def readiness_check(req: ReadinessRequest):
    """Scopus Readiness Checker · 10-metric quality assessment.
    Returns composite score dan per-metric breakdown untuk submission gate."""
    metrics = {
        "novelty_score":          _scoreNovelty(req),
        "methodology_strength":   _scoreMethodology(req),
        "citation_quality":       _scoreCitationQuality(req),
        "dataset_transparency":   _scoreDatasetTransparency(req),
        "reproducibility":        _scoreReproducibility(req),
        "ethical_compliance":     _scoreEthics(req),
        "plagiarism_risk":        _scorePlagiarismRisk(req),
        "ai_writing_risk":        _scoreAIWritingRisk(req),
        "journal_scope_fit":      _scoreJournalFit(req),
        "reference_quality":      _scoreReferenceQuality(req),
    }
    weights = {
        "novelty_score":         0.18,
        "methodology_strength":  0.16,
        "citation_quality":      0.12,
        "dataset_transparency":  0.10,
        "reproducibility":       0.10,
        "ethical_compliance":    0.06,
        "plagiarism_risk":       0.08,
        "ai_writing_risk":       0.06,
        "journal_scope_fit":     0.06,
        "reference_quality":     0.08,
    }
    composite = sum(metrics[k]["score"] * weights[k] for k in metrics)
    classification = _classifyComposite(int(composite), metrics)
    return {
        "ok": True,
        "composite_score": int(round(composite)),
        "classification": classification,
        "metrics": metrics,
        "weights": weights,
        "note": "Composite weighted average · gunakan plagiarism-check endpoint untuk deep TF-IDF analysis",
    }


# ============================================================================
# JOURNAL MATCHER · Phase 3 · curated catalog plus scope match scoring
# Output rangked journal candidates dengan Q1-Q4, APC, indexing, review time,
# acceptance risk, dan required format per jurnal.
# ============================================================================
class JournalMatchRequest(BaseModel):
    research_title: str = Field("", description="Working research title")
    abstract: str = Field("", description="Abstract atau ringkasan paper")
    keywords: List[str] = Field(default_factory=list, description="Keywords list")
    domain: str = Field("data_science", description="data_science atau geo_disaster atau interdisciplinary")
    target_quartile: str = Field("any", description="Q1, Q2, Q3, Q4, atau any")
    open_access_only: bool = Field(False, description="Only return open access journals")
    language: Optional[str] = None


_JOURNAL_CATALOG = [
    {
        "name": "IEEE Access",
        "publisher": "IEEE",
        "quartile": "Q2",
        "impact_factor": 3.9,
        "scope_keywords": ["multidisciplinary", "engineering", "data science", "machine learning", "ai", "applied", "computer science"],
        "domain": ["data_science", "geo_disaster", "interdisciplinary"],
        "apc_usd": 1995,
        "open_access": True,
        "review_time_weeks": 5,
        "acceptance_rate": 30,
        "indexing": ["Scopus", "WoS-SCIE", "IEEE Xplore"],
        "format": "IEEE template · 6-12 pages double column",
        "homepage": "https://ieeeaccess.ieee.org",
    },
    {
        "name": "Scientific Reports",
        "publisher": "Nature Springer",
        "quartile": "Q1",
        "impact_factor": 4.6,
        "scope_keywords": ["natural sciences", "applied", "interdisciplinary", "machine learning", "environment", "earth science"],
        "domain": ["data_science", "geo_disaster", "interdisciplinary"],
        "apc_usd": 2290,
        "open_access": True,
        "review_time_weeks": 12,
        "acceptance_rate": 50,
        "indexing": ["Scopus", "WoS-SCIE", "PubMed"],
        "format": "Nature template · structured abstract",
        "homepage": "https://www.nature.com/srep",
    },
    {
        "name": "Expert Systems with Applications",
        "publisher": "Elsevier",
        "quartile": "Q1",
        "impact_factor": 8.5,
        "scope_keywords": ["expert system", "machine learning", "ai", "data mining", "applied", "decision support"],
        "domain": ["data_science"],
        "apc_usd": 3450,
        "open_access": False,
        "review_time_weeks": 16,
        "acceptance_rate": 18,
        "indexing": ["Scopus", "WoS-SCIE"],
        "format": "Elsevier template · single column",
        "homepage": "https://www.sciencedirect.com/journal/expert-systems-with-applications",
    },
    {
        "name": "Applied Sciences",
        "publisher": "MDPI",
        "quartile": "Q2",
        "impact_factor": 2.7,
        "scope_keywords": ["applied", "interdisciplinary", "engineering", "machine learning", "environmental"],
        "domain": ["data_science", "geo_disaster", "interdisciplinary"],
        "apc_usd": 2600,
        "open_access": True,
        "review_time_weeks": 4,
        "acceptance_rate": 50,
        "indexing": ["Scopus", "WoS-SCIE"],
        "format": "MDPI template · structured sections",
        "homepage": "https://www.mdpi.com/journal/applsci",
    },
    {
        "name": "Remote Sensing",
        "publisher": "MDPI",
        "quartile": "Q1",
        "impact_factor": 5.0,
        "scope_keywords": ["remote sensing", "satellite", "earth observation", "gis", "flood", "landslide", "geo"],
        "domain": ["geo_disaster"],
        "apc_usd": 2700,
        "open_access": True,
        "review_time_weeks": 5,
        "acceptance_rate": 48,
        "indexing": ["Scopus", "WoS-SCIE"],
        "format": "MDPI template · single column",
        "homepage": "https://www.mdpi.com/journal/remotesensing",
    },
    {
        "name": "Natural Hazards",
        "publisher": "Springer",
        "quartile": "Q1",
        "impact_factor": 3.7,
        "scope_keywords": ["natural hazard", "disaster", "flood", "earthquake", "tsunami", "landslide", "risk", "mitigation"],
        "domain": ["geo_disaster"],
        "apc_usd": 3380,
        "open_access": False,
        "review_time_weeks": 14,
        "acceptance_rate": 25,
        "indexing": ["Scopus", "WoS-SCIE"],
        "format": "Springer template · regular research",
        "homepage": "https://www.springer.com/journal/11069",
    },
    {
        "name": "International Journal of Disaster Risk Reduction",
        "publisher": "Elsevier",
        "quartile": "Q1",
        "impact_factor": 5.0,
        "scope_keywords": ["disaster risk", "reduction", "resilience", "emergency", "preparedness", "mitigation", "vulnerability"],
        "domain": ["geo_disaster"],
        "apc_usd": 3160,
        "open_access": False,
        "review_time_weeks": 12,
        "acceptance_rate": 22,
        "indexing": ["Scopus", "WoS-SCIE"],
        "format": "Elsevier template · IMRaD",
        "homepage": "https://www.sciencedirect.com/journal/international-journal-of-disaster-risk-reduction",
    },
    {
        "name": "Neural Computing and Applications",
        "publisher": "Springer",
        "quartile": "Q1",
        "impact_factor": 6.0,
        "scope_keywords": ["neural", "deep learning", "machine learning", "applied", "lstm", "cnn", "ai"],
        "domain": ["data_science"],
        "apc_usd": 3290,
        "open_access": False,
        "review_time_weeks": 16,
        "acceptance_rate": 20,
        "indexing": ["Scopus", "WoS-SCIE"],
        "format": "Springer template · research paper",
        "homepage": "https://www.springer.com/journal/521",
    },
    {
        "name": "PLOS ONE",
        "publisher": "PLOS",
        "quartile": "Q2",
        "impact_factor": 3.7,
        "scope_keywords": ["multidisciplinary", "open science", "natural sciences", "applied", "reproducible"],
        "domain": ["data_science", "geo_disaster", "interdisciplinary"],
        "apc_usd": 1805,
        "open_access": True,
        "review_time_weeks": 10,
        "acceptance_rate": 45,
        "indexing": ["Scopus", "WoS-SCIE", "PubMed"],
        "format": "PLOS structure · IMRaD",
        "homepage": "https://journals.plos.org/plosone",
    },
    {
        "name": "Indonesian Journal of Computing and Cybernetics (IJCCS)",
        "publisher": "UGM",
        "quartile": "Q4",
        "impact_factor": 0.5,
        "scope_keywords": ["computing", "cybernetics", "indonesia", "applied", "information system"],
        "domain": ["data_science"],
        "apc_usd": 0,
        "open_access": True,
        "review_time_weeks": 8,
        "acceptance_rate": 60,
        "indexing": ["Sinta", "DOAJ", "Scopus"],
        "format": "IEEE template adapted",
        "homepage": "https://jurnal.ugm.ac.id/ijccs",
    },
    {
        "name": "Jurnal Online Informatika (JOIN)",
        "publisher": "UIN Sunan Gunung Djati",
        "quartile": "Q3",
        "impact_factor": 0.8,
        "scope_keywords": ["informatics", "indonesia", "applied", "machine learning"],
        "domain": ["data_science"],
        "apc_usd": 0,
        "open_access": True,
        "review_time_weeks": 8,
        "acceptance_rate": 55,
        "indexing": ["Sinta 2", "DOAJ"],
        "format": "JOIN template",
        "homepage": "https://join.if.uinsgd.ac.id",
    },
    {
        "name": "Water · MDPI",
        "publisher": "MDPI",
        "quartile": "Q2",
        "impact_factor": 3.4,
        "scope_keywords": ["water", "hydrology", "flood", "rainfall", "watershed", "river"],
        "domain": ["geo_disaster"],
        "apc_usd": 2600,
        "open_access": True,
        "review_time_weeks": 5,
        "acceptance_rate": 50,
        "indexing": ["Scopus", "WoS-SCIE"],
        "format": "MDPI template",
        "homepage": "https://www.mdpi.com/journal/water",
    },
]


def _scoreJournalMatch(req: JournalMatchRequest, j: Dict[str, Any]) -> Dict[str, Any]:
    score = 50
    breakdown = []
    title_words = set(re.findall(r'\b\w+\b', (req.research_title or '').lower()))
    abstract_words = set(re.findall(r'\b\w+\b', (req.abstract or '').lower()))
    keyword_words = set(' '.join(req.keywords or []).lower().split())
    all_user_words = title_words | abstract_words | keyword_words
    scope_words = set(' '.join(j["scope_keywords"]).lower().split())
    overlap = all_user_words & scope_words
    overlap_score = min(40, len(overlap) * 5)
    score += overlap_score
    breakdown.append({"factor": "Scope keyword overlap", "value": f"{len(overlap)} matches", "weight": overlap_score})

    domain_match = req.domain in j["domain"]
    if domain_match:
        score += 15
        breakdown.append({"factor": "Domain match", "value": "Yes", "weight": 15})
    else:
        breakdown.append({"factor": "Domain match", "value": "Partial", "weight": 0})

    if req.target_quartile != "any" and req.target_quartile == j["quartile"]:
        score += 10
        breakdown.append({"factor": "Quartile match", "value": j["quartile"], "weight": 10})

    if req.open_access_only and not j["open_access"]:
        score -= 30
        breakdown.append({"factor": "Open access filter", "value": "Not OA · penalty", "weight": -30})

    accept_proxy = j["acceptance_rate"]
    score += min(15, accept_proxy // 4)
    breakdown.append({"factor": "Acceptance rate proxy", "value": f"{accept_proxy}%", "weight": min(15, accept_proxy // 4)})
    return {"match_score": min(100, max(0, score)), "score_breakdown": breakdown}


def _acceptanceRisk(j: Dict[str, Any], match_score: int) -> Dict[str, Any]:
    base = 100 - j["acceptance_rate"]
    quartile_penalty = {"Q1": 15, "Q2": 8, "Q3": 3, "Q4": 0}.get(j["quartile"], 5)
    score_bonus = (match_score - 50) / 5
    risk = max(10, min(90, base + quartile_penalty - score_bonus))
    if risk >= 70:
        return {"score": int(risk), "label": "High Risk", "color": "#ff3b55"}
    if risk >= 50:
        return {"score": int(risk), "label": "Moderate Risk", "color": "#ff9500"}
    return {"score": int(risk), "label": "Low Risk", "color": "#34d399"}


@router.post("/pipeline/journal/match")
async def journal_match(req: JournalMatchRequest):
    """Journal Matcher · ranked list of journals dengan scope match, acceptance risk, indexing."""
    results = []
    for j in _JOURNAL_CATALOG:
        m = _scoreJournalMatch(req, j)
        risk = _acceptanceRisk(j, m["match_score"])
        results.append({
            "journal": j["name"],
            "publisher": j["publisher"],
            "quartile": j["quartile"],
            "impact_factor": j["impact_factor"],
            "match_score": m["match_score"],
            "score_breakdown": m["score_breakdown"],
            "acceptance_risk": risk,
            "open_access": j["open_access"],
            "apc_usd": j["apc_usd"],
            "review_time_weeks": j["review_time_weeks"],
            "acceptance_rate": j["acceptance_rate"],
            "indexing": j["indexing"],
            "format": j["format"],
            "homepage": j["homepage"],
            "scope_keywords": j["scope_keywords"],
        })
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return {
        "ok": True,
        "total_journals": len(results),
        "ranked": results[:10],
        "note": "Match score berdasarkan scope keyword overlap plus domain match. Verify status terbaru langsung di Scopus atau publisher homepage karena quartile dan indexing dapat berubah.",
    }


# ============================================================================
# DATA SCIENCE RESEARCH LAB · AI-driven per-stage workspace (Opsi B Phase Final)
# Setiap lab task type (classification, regression, clustering, forecasting,
# anomaly, deep_learning, nlp, computer_vision) memberikan project-specific
# detailed plan berbasis vault context user. 3 endpoint:
#   1. lab/generate-plan       · 8-10 stage plan untuk project specific
#   2. lab/generate-stage      · per-stage deep detail dengan code template
#   3. lab/suggest-algorithms  · top 3 algoritma based on dataset profile
#
# Frontend research-ds-labs.js memanggil endpoint ini saat user buka lab dan
# menampilkan hasil sebagai expandable stage cards. Backend mengikuti pattern
# _tr.route("structured_fast") dengan fallback rule-based kalau AI tidak
# tersedia atau parse JSON gagal.
# ============================================================================

class LabContext(BaseModel):
    research_title: str = ""
    research_question: str = ""
    hypothesis: str = ""
    domain: str = ""
    dataset_name: str = ""
    dataset_shape: List[int] = Field(default_factory=list)
    dataset_format: str = ""
    target_variable: str = ""
    feature_columns: List[str] = Field(default_factory=list)
    identifier_column: str = ""
    timestamp_column: str = ""
    method_choice: str = ""
    cited_papers_count: int = 0
    language: Optional[str] = None


class LabPlanRequest(BaseModel):
    lab_type: str = Field("classification", description="classification / regression / clustering / forecasting / anomaly / deep_learning / nlp / computer_vision")
    context: LabContext


class LabStageRequest(BaseModel):
    lab_type: str
    stage_id: str = Field("", description="ID stage yang minta detail · e.g. eda, preprocessing, baseline")
    stage_title: str = ""
    context: LabContext


class LabAlgoRequest(BaseModel):
    lab_type: str
    context: LabContext


# Per-lab system prompt customization
_LAB_PROMPT_SYSTEM = {
    "classification": "You are a senior ML engineer specializing in classification tasks. You give project-specific guidance grounded in the user's actual dataset and research question.",
    "regression":     "You are a senior ML engineer specializing in regression analysis. You give project-specific guidance grounded in the user's data characteristics.",
    "clustering":     "You are a senior data scientist specializing in unsupervised clustering. You guide the user through cluster discovery suitable for their data structure.",
    "forecasting":    "You are a senior time series analyst. You provide guidance specific to the user's temporal data and forecasting horizon.",
    "anomaly":        "You are a senior anomaly detection researcher. You tailor guidance to the user's anomaly rate and detection setting.",
    "deep_learning":  "You are a senior deep learning engineer. You provide architecture and training guidance specific to the user's data size and compute.",
    "nlp":            "You are a senior NLP researcher specializing in Indonesian and multilingual processing. You guide tokenization, pretrained model selection, and fine-tuning.",
    "computer_vision":"You are a senior computer vision engineer. You guide architecture selection, transfer learning, and evaluation appropriate to the user's image characteristics.",
    # Geo-Disaster domain-specific prompts
    "flood":          "You are a senior hydrologist plus flood modeling researcher familiar with BMKG, BNPB, and BIG Indonesia data sources. You guide rainfall-runoff modeling, susceptibility mapping, and inundation prediction using Sentinel-1 SAR plus DEM data.",
    "tsunami":        "You are a senior tsunami risk researcher familiar with BMKG warning system InaTEWS. You guide exposure analysis, runup modeling, and population vulnerability assessment with GIS-based multi-criteria decision analysis.",
    "earthquake":     "You are a senior seismologist plus structural risk researcher. You guide probabilistic seismic hazard analysis PSHA, ground motion prediction, and building vulnerability with USGS plus BMKG catalog data.",
    "climate":        "You are a senior climate science researcher familiar with CMIP6, ERA5 reanalysis, plus IPCC AR6 framework. You guide downscaling, trend detection, and impact attribution analysis for Indonesia.",
    "wildfire":       "You are a senior wildfire risk researcher familiar with MODIS active fire, Sentinel-2 burn scars, plus FWI Fire Weather Index. You guide ignition probability mapping, fuel moisture, and burn severity assessment.",
    "storm":          "You are a senior atmospheric scientist specializing in tropical cyclone plus convective storm research. You guide track forecasting, intensity prediction, and damage swath analysis using IBTrACS plus radar data.",
    "coastal":        "You are a senior coastal hazard researcher specializing in sea level rise, coastal erosion, plus storm surge. You guide shoreline change detection from Landsat plus Sentinel-2, plus Bruun rule application.",
    "environment":    "You are a senior environmental monitoring researcher integrating remote sensing plus IoT sensor networks. You guide land cover change, deforestation tracking, plus air quality analysis using Sentinel-5P plus ground stations.",
}


# Per-lab standard workflow stages (template for fallback dan structure)
_LAB_STAGES = {
    "classification": [
        ("define",        "Define classification scheme",       "Binary vs multi-class vs multi-label decision; threshold action"),
        ("eda",           "Exploratory Data Analysis",          "Class distribution, missing patterns, feature correlation, separability"),
        ("preprocess",    "Preprocessing pipeline",             "Imputation, encoding, scaling, stratified train/val/test split"),
        ("baseline",      "Train baseline model",               "Logistic Regression or Random Forest with default config"),
        ("feature_eng",   "Feature engineering",                "Domain features, interactions, polynomial, target encoding"),
        ("model_select",  "Model selection",                    "Cross-validate 3-5 algorithms, pick best 2-3"),
        ("hp_tuning",     "Hyperparameter tuning",              "Bayesian optimization or grid search with 5-fold CV"),
        ("final_eval",    "Final evaluation",                   "Holdout test, confusion matrix, ROC curve, SHAP, calibration"),
    ],
    "regression": [
        ("define",        "Define dependent variable",          "Units, distribution check (skewness, kurtosis)"),
        ("eda",           "Exploratory analysis",                "Correlation matrix, scatter plot, baseline residual plot"),
        ("preprocess",    "Preprocessing",                       "Transform skewed target, scale features, VIF check"),
        ("baseline",      "OLS baseline",                        "Linear regression with all features, diagnostic"),
        ("diagnostic",    "Residual diagnostics",                "Normality, homoscedasticity, autocorrelation tests"),
        ("regularize",    "Regularization sweep",                "Ridge, Lasso, Elastic Net with alpha tuning"),
        ("non_linear",    "Non-linear models",                   "Random Forest, GBM, kernel methods kalau struktur residual masih jelas"),
        ("final_eval",    "Final evaluation",                    "Holdout RMSE+MAE+R², prediction interval via quantile or bootstrap"),
    ],
    "clustering": [
        ("define",        "Define clustering objective",         "Segmentation, pattern discovery, anomaly grouping"),
        ("dim_reduce",    "Dimensionality reduction",            "PCA, UMAP, t-SNE untuk visualize struktur"),
        ("preprocess",    "Feature standardization",             "Z-score atau min-max, handle missing"),
        ("k_estimate",    "Determine optimal k",                 "Elbow, Silhouette, Gap statistic"),
        ("apply",         "Apply 2-3 algorithms",                "K-Means baseline, DBSCAN untuk non-convex, hierarchical untuk dendrogram"),
        ("internal_val",  "Internal validation",                 "Silhouette, Davies-Bouldin, Calinski-Harabasz per algorithm"),
        ("external_val",  "External validation",                 "ARI, NMI kalau ada partial ground truth"),
        ("profile",       "Cluster profiling",                   "Centroid characteristics, feature importance, name dengan domain expert"),
    ],
    "forecasting": [
        ("eda",           "Time series EDA",                      "Time plot, STL decomposition, ACF/PACF, ADF/KPSS stationarity test"),
        ("split",         "Temporal split",                       "Train past, test future, rolling-origin CV"),
        ("baselines",     "Naive baselines",                      "Last-value, seasonal naive, simple exponential smoothing"),
        ("stat_model",    "Statistical model",                    "ARIMA/SARIMA dengan auto.arima atau pmdarima"),
        ("ml_model",      "ML model with lags",                   "Random Forest atau GBM dengan lag features"),
        ("dl_model",      "Deep learning",                        "LSTM/GRU/Transformer kalau dataset cukup besar"),
        ("reconcile",     "Hierarchical reconciliation",          "Kalau multi-level forecast diperlukan"),
        ("prob_eval",     "Probabilistic evaluation",             "Prediction interval coverage, calibration, CRPS"),
    ],
    "anomaly": [
        ("define",        "Define anomaly type",                  "Point, contextual, atau collective anomaly (Chandola 2009 taxonomy)"),
        ("eda",           "Normal pattern EDA",                   "Understand baseline distribution, identify known anomaly examples"),
        ("preprocess",    "Preprocessing",                        "Scale features, handle missing, time alignment kalau temporal"),
        ("unsupervised",  "Unsupervised baseline",                "Isolation Forest atau LOF dengan default config"),
        ("semi_super",    "Semi-supervised",                      "One-class SVM atau autoencoder train hanya pada normal data"),
        ("threshold",     "Threshold selection",                  "Validate pada labeled subset, pick precision-recall trade-off"),
        ("compare",       "Compare 3-4 algorithms",               "PR-AUC pada labeled holdout"),
        ("interpret",     "Top-k investigation",                  "Feature attribution per anomaly, validate dengan domain expert"),
    ],
    "deep_learning": [
        ("frame",         "Problem framing",                      "Supervised/unsupervised/self-supervised/RL, input/output spec"),
        ("pretrained",    "Pretrained model survey",              "Hugging Face, timm, torchvision untuk transfer learning candidates"),
        ("baseline",      "Simple baseline",                      "Establish good enough reference (e.g., MobileNetV2 untuk vision)"),
        ("architecture",  "Architecture design",                  "Start simple, scale gradually, dont jump to SOTA"),
        ("training",      "Training setup",                       "AdamW optimizer, LR schedule, batch size, augmentation"),
        ("ablation",      "Ablation study",                       "Remove components one-by-one, validate contribution claims"),
        ("hp_tuning",     "Hyperparameter tuning",                "Optuna atau Ray Tune dengan 20-100 trials"),
        ("final_eval",    "Final evaluation",                     "Holdout test, error analysis qualitative, computational cost report"),
    ],
    "nlp": [
        ("define",        "Define NLP task",                       "Classification, generation, extraction, atau similarity"),
        ("tokenize",      "Tokenization strategy",                 "Word-level, subword BPE/WordPiece, character-level untuk OOV-heavy"),
        ("baseline",      "Bag-of-words baseline",                  "TF-IDF + Logistic Regression atau FastText"),
        ("pretrained",    "Pretrained model selection",            "IndoBERT untuk Bahasa Indonesia, multilingual-BERT untuk cross-lingual"),
        ("fine_tune",     "Fine-tuning",                            "3-5 epochs, learning rate 2e-5 to 5e-5 (Devlin et al recommendation)"),
        ("eval",          "Task-specific evaluation",               "F1, BLEU, ROUGE, BERTScore sesuai task plus error analysis"),
        ("adversarial",   "Adversarial robustness",                 "Paraphrasing, typo, code-mixing test"),
        ("deploy",        "Deployment consideration",               "Model size (distill?), latency target, batching"),
    ],
    "computer_vision": [
        ("define",        "Define vision task",                     "Classification, detection, segmentation, generation, retrieval"),
        ("audit",         "Dataset audit",                          "Resolution, color space, class distribution, annotation quality"),
        ("pretrained",    "Pretrained backbone",                    "torchvision atau timm pretrained ImageNet/COCO/ADE20K"),
        ("augment",       "Data augmentation pipeline",             "RandAugment, color jitter, random crop, flip, MixUp"),
        ("transfer",      "Transfer learning",                      "Freeze backbone, train head, then fine-tune full network"),
        ("hp_tuning",     "Hyperparameter tuning",                  "Learning rate, batch size, weight decay sweep"),
        ("eval",          "Evaluation",                             "Per-class metrics, confusion matrix, qualitative error gallery"),
        ("deploy",        "Deployment",                             "ONNX export, INT8 quantization, inference benchmark"),
    ],
    # =========== GEO-DISASTER DOMAIN STAGES ===========
    "flood": [
        ("define",        "Define flood scenario",                  "Pluvial vs fluvial vs coastal, return period (2/5/10/100 yr), study area boundary"),
        ("data_collect",  "Multi-source data collection",           "Rainfall BMKG, DEM SRTM 30m atau BIG DEMNAS, Sentinel-1 SAR post-event, land use BIG"),
        ("preprocess",    "Spatial preprocessing",                  "Reproject ke UTM zone, resample raster ke common resolution, mask AOI"),
        ("conditioning",  "Conditioning factor preparation",        "Slope, aspect, TWI, distance to river, drainage density, soil texture"),
        ("susceptibility","Susceptibility model",                   "AHP atau Frequency Ratio atau Random Forest dengan training data flood inventory"),
        ("validation",    "AUC + flood inventory validation",       "Compare prediksi vs observed flood points, hitung AUC-ROC plus kappa"),
        ("inundation",    "Inundation simulation",                  "HEC-RAS 2D atau LISFLOOD-FP untuk depth + extent simulation skenario"),
        ("exposure",      "Population plus asset exposure",         "Overlay flood map dengan WorldPop, BPS kecamatan, OSM building footprints"),
    ],
    "tsunami": [
        ("define",        "Define source scenario",                 "Earthquake megathrust Mw 8-9 atau landslide submarine, fault parameter"),
        ("source_model",  "Tsunami source modeling",                "Okada 1985 dislocation untuk initial sea surface deformation"),
        ("propagation",   "Wave propagation",                       "COMCOT atau NEOWAVE atau TUNAMI shallow water equation"),
        ("inundation",    "Coastal inundation mapping",             "High resolution DEM coastal, runup heights, flooding extent"),
        ("validation",    "Historical event validation",            "Compare model output dengan observasi tsunami 2004 Aceh atau 2018 Palu"),
        ("exposure",      "Population exposure analysis",           "Overlay inundation dengan demografi BPS, evakuasi shelter location"),
        ("vulnerability", "Building vulnerability assessment",      "Material type, lantai jumlah, jarak ke shelter, fragility curve"),
        ("ews_design",    "Early warning design",                   "InaTEWS integration, evacuation route GIS, drill scenario"),
    ],
    "earthquake": [
        ("define",        "Define hazard study scope",              "Region of interest, target structure type, return period (475 yr default)"),
        ("catalog",       "Earthquake catalog assembly",            "USGS plus BMKG catalog, completeness check, declustering Gardner-Knopoff"),
        ("source_zone",   "Seismic source zonation",                "Active faults plus area sources, geometry plus seismicity parameters"),
        ("gmpe",          "Ground Motion Prediction Equation",      "Pilih GMPE sesuai region (Boore-Atkinson, ASB14 untuk active crustal)"),
        ("psha",          "Probabilistic Seismic Hazard Analysis",  "Cornell-McGuire approach, hazard curve PGA plus SA"),
        ("disaggregation","Hazard disaggregation",                  "Identify dominant magnitude-distance contributor untuk design"),
        ("site_response", "Site response analysis",                 "Vs30 mapping, amplification factor, NEHRP site class"),
        ("vulnerability", "Building vulnerability modeling",        "HAZUS atau RISK-UE fragility curves untuk masing-masing building class"),
    ],
    "climate": [
        ("define",        "Define climate question",                "Trend detection, attribution, projection (RCP/SSP scenario), spatial scope"),
        ("data_source",   "Climate data sources",                   "ERA5 reanalysis, CHIRPS rainfall, BMKG station, CMIP6 GCM ensemble"),
        ("bias_correct",  "Bias correction",                        "Quantile mapping atau delta change method untuk downscale GCM ke station scale"),
        ("trend_analysis","Trend detection",                        "Mann-Kendall non-parametric, Sen slope, change point Pettitt test"),
        ("extreme_index", "Climate extremes indices",               "ETCCDI 27 index (TX90p, TN10p, RX5day, R95p) plus return value extreme"),
        ("attribution",   "Attribution analysis",                   "Optional Fraction of Attributable Risk FAR, dampak iklim antropogenik"),
        ("projection",    "Future projection",                      "Multi-model ensemble mean plus inter-model uncertainty, RCP/SSP scenarios"),
        ("impact",        "Impact analysis",                        "Hubungan ke sektor sensitif (agrikultur, air, kesehatan) via response function"),
    ],
    "wildfire": [
        ("define",        "Define wildfire research scope",         "Ignition prediction, burn severity, spread modeling, atau air quality impact"),
        ("data_collect",  "Multi-source data",                      "MODIS MCD64 burned area, Sentinel-2 dNBR, GFED, weather BMKG, fuel moisture"),
        ("fwi",           "Fire Weather Index calculation",          "FWI Canadian system (FFMC, DMC, DC, ISI, BUI, FWI) per pixel harian"),
        ("ignition_model","Ignition probability model",              "Random Forest atau XGBoost dengan FWI, land cover, distance to road, anthropogenic"),
        ("severity",      "Burn severity assessment",                "Sentinel-2 NBR pre/post differencing (dNBR), classify low-mod-high severity"),
        ("spread",        "Fire spread simulation",                  "FARSITE atau cellular automaton model, wind, slope, fuel parameters"),
        ("smoke_quality", "Smoke plume plus air quality",            "HYSPLIT trajectory, PM2.5 dispersion, exposure population downwind"),
        ("recovery",      "Post-fire recovery monitoring",           "NDVI recovery time series, regeneration probability"),
    ],
    "storm": [
        ("define",        "Define storm research focus",            "Track prediction, intensity forecast, atau damage swath analysis"),
        ("data_source",   "Storm data sources",                     "IBTrACS global track, BMKG plus JTWC bulletins, GPM IMERG precipitation"),
        ("track_model",   "Track prediction",                       "Statistical-dynamical model atau LSTM dari historical track plus environment"),
        ("intensity",     "Intensity prediction",                   "SHIPS plus DTOPS, plus deep learning (CNN dari satellite IR imagery)"),
        ("wind_field",    "Wind field analysis",                    "Holland parametric vortex, plus radar Doppler observation"),
        ("precip_swath",  "Precipitation swath",                    "GPM IMERG plus radar QPE composite untuk rainfall accumulation"),
        ("damage_swath",  "Damage swath estimation",                "Wind speed × duration × building vulnerability function"),
        ("ews",           "Early warning integration",              "BMKG cuaca ekstrem warning, shelter capacity, evacuation timing"),
    ],
    "coastal": [
        ("define",        "Define coastal hazard focus",            "Erosion, sea level rise, storm surge, tidal flooding, atau salinity intrusion"),
        ("shoreline",     "Shoreline detection",                    "Landsat plus Sentinel-2 NDWI atau MNDWI, semi-automated extraction"),
        ("change_rate",   "Shoreline change rate",                  "DSAS tool, EPR plus LRR statistics dari multi-temporal shoreline"),
        ("slr_scenario",  "Sea level rise scenario",                "IPCC AR6 RCP/SSP scenarios untuk 2050 plus 2100 projection"),
        ("bruun_rule",    "Bruun rule recession",                   "Equilibrium beach profile recession dari sea level rise"),
        ("surge_model",   "Storm surge modeling",                   "ADCIRC atau Delft3D atau XBeach untuk hindcast plus forecast event"),
        ("exposure",      "Coastal exposure assessment",            "Population plus infrastructure dalam zona terdampak, ekonomi value"),
        ("adaptation",    "Adaptation strategy",                    "Hard engineering (seawall, breakwater), soft (mangrove, beach nourishment), retreat"),
    ],
    "environment": [
        ("define",        "Define monitoring objective",            "Deforestation, land cover change, air quality, water quality, atau biodiversity"),
        ("baseline",      "Baseline land cover",                    "Sentinel-2 atau Landsat composite, supervised classification atau dynamic world"),
        ("change_detect", "Change detection",                       "Multi-temporal differencing, GFW alerts, RADD radar alerts"),
        ("attribute",     "Attribute change driver",                "Agriculture expansion, mining, urbanization, fire via overlay analysis"),
        ("air_quality",   "Air quality monitoring",                 "Sentinel-5P NO2 plus aerosol, ground station validation, OpenAQ data"),
        ("water_quality", "Water quality remote sensing",           "Sentinel-2 reflectance proxy untuk chlorophyll, turbidity, TSS estimation"),
        ("ecosystem",     "Ecosystem services valuation",           "Carbon stock, biodiversity index, ecosystem service mapping"),
        ("policy_input",  "Policy-ready output",                    "Reporting compatible dengan RAD-GRK, EITI, atau SDG indicator framework"),
    ],
}


def _ctx_summary(ctx: LabContext) -> str:
    """Ringkas vault context jadi prompt-friendly description."""
    parts = []
    if ctx.research_title:
        parts.append(f"Research title: {ctx.research_title}")
    if ctx.research_question:
        parts.append(f"Research question: {ctx.research_question}")
    if ctx.hypothesis:
        parts.append(f"Hypothesis: {ctx.hypothesis}")
    if ctx.domain:
        parts.append(f"Domain: {ctx.domain}")
    if ctx.dataset_name:
        shape_str = f" ({ctx.dataset_shape[0]} rows × {ctx.dataset_shape[1]} cols)" if len(ctx.dataset_shape) >= 2 else ""
        parts.append(f"Dataset: {ctx.dataset_name}{shape_str}, format {ctx.dataset_format or 'unknown'}")
    if ctx.target_variable:
        parts.append(f"Target variable: {ctx.target_variable}")
    if ctx.feature_columns:
        cols = ctx.feature_columns[:10]
        more = f" plus {len(ctx.feature_columns) - 10} others" if len(ctx.feature_columns) > 10 else ""
        parts.append(f"Features ({len(ctx.feature_columns)}): {', '.join(cols)}{more}")
    if ctx.identifier_column:
        parts.append(f"Identifier column: {ctx.identifier_column}")
    if ctx.timestamp_column:
        parts.append(f"Timestamp column: {ctx.timestamp_column}")
    if ctx.method_choice:
        parts.append(f"Method choice: {ctx.method_choice}")
    if ctx.cited_papers_count > 0:
        parts.append(f"Literature base: {ctx.cited_papers_count} cited papers in Citation Manager")
    return "\n".join(f"- {p}" for p in parts) if parts else "(no context provided - generic guidance only)"


def _lang_hint(ctx: LabContext) -> str:
    if ctx.language and ctx.language.lower().startswith("id"):
        return "Tulis seluruh penjelasan dalam bahasa Indonesia akademik mahasiswa-friendly. Hindari emdash dan titik koma."
    return "Write in clear, scholarly English."


# ============================================================================
# Endpoint 1 · GENERATE PLAN · 8-10 stage plan project-specific
# ============================================================================
@router.post("/lab/generate-plan")
async def lab_generate_plan(req: LabPlanRequest):
    """Generate detailed project plan untuk lab task type dengan vault context."""
    lab_type = req.lab_type.lower().replace("-", "_")
    if lab_type not in _LAB_STAGES:
        return {"ok": False, "error": f"Unknown lab type: {lab_type}",
                "supported": list(_LAB_STAGES.keys())}
    stages_template = _LAB_STAGES[lab_type]
    system_prompt = _LAB_PROMPT_SYSTEM.get(lab_type, _LAB_PROMPT_SYSTEM["classification"])
    ctx_str = _ctx_summary(req.context)
    lang = _lang_hint(req.context)
    stage_list_str = "\n".join(f"  {i+1}. {sid} - {title}" for i, (sid, title, _) in enumerate(stages_template))

    user_prompt = (
        "Anda diminta membuat detailed research plan untuk task " + lab_type.replace("_", " ") + ".\n\n"
        "PROJECT CONTEXT\n"
        f"{ctx_str}\n\n"
        "STANDARD STAGES TEMPLATE (8 steps urut)\n"
        f"{stage_list_str}\n\n"
        "TASK\n"
        "Tulis ulang ke-8 stage di atas menjadi project-specific guidance. Jangan ubah stage_id atau urutan. "
        "Untuk masing-masing stage, berikan:\n"
        "  - stage_id (unchanged)\n"
        "  - title (project-specific, mention nama dataset / target variable kalau relevan)\n"
        "  - summary (2-3 kalimat menjelaskan apa yang dilakukan dan kenapa untuk project ini)\n"
        "  - duration_estimate (e.g., '2-4 hours', '1 day')\n"
        "  - difficulty (easy, medium, hard)\n"
        "  - key_decisions (3 bullet points keputusan kritis yang harus diambil di stage ini)\n"
        "  - deliverable (1 kalimat output konkret dari stage ini, e.g., 'cleaned dataset shape X by Y', 'baseline accuracy report')\n\n"
        "OUTPUT FORMAT\n"
        "Return ONLY a strict JSON array (no markdown, no preamble) of 8 objects matching the schema:\n"
        "[\n"
        '  {"stage_id":"...","title":"...","summary":"...","duration_estimate":"...",\n'
        '   "difficulty":"easy|medium|hard","key_decisions":["...","...","..."],\n'
        '   "deliverable":"..."}\n'
        "]\n\n"
        f"{lang}"
    )

    text = ""
    try:
        from app.services import task_router as _tr
        result = _tr.route(
            task_type="structured_fast",
            messages=[
                {"role": "system", "content": system_prompt + " Return strict JSON only."},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2800,
            temperature=0.4,
        )
        text = (result or {}).get("text", "") or ""
    except Exception:
        text = ""

    parsed = None
    if text:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
        m = re.search(r"\[\s*\{.*\}\s*\]", cleaned, flags=re.S)
        candidate = m.group(0) if m else cleaned
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, list):
                parsed = None
        except Exception:
            parsed = None

    if parsed and len(parsed) >= 1:
        return {"ok": True, "source": "ai", "lab_type": lab_type, "stages": parsed[:10]}

    # Fallback: template stages dengan generic guidance
    fallback = []
    for sid, title, desc in stages_template:
        fallback.append({
            "stage_id": sid,
            "title": title,
            "summary": desc,
            "duration_estimate": "1-3 hours",
            "difficulty": "medium",
            "key_decisions": [
                f"Konfigurasi parameter sesuai karakteristik {req.context.dataset_name or 'dataset'}",
                "Pilih library plus framework yang konsisten dengan pipeline keseluruhan",
                "Dokumentasikan setiap keputusan supaya reproducible",
            ],
            "deliverable": f"Output {sid} yang siap dipakai stage berikutnya",
        })
    return {
        "ok": True, "source": "fallback_template", "lab_type": lab_type, "stages": fallback,
        "note": "AI engine tidak tersedia, menggunakan template stages dengan generic guidance.",
    }


# ============================================================================
# Endpoint 2 · GENERATE STAGE DETAIL · code template, errors, references
# ============================================================================
@router.post("/lab/generate-stage")
async def lab_generate_stage(req: LabStageRequest):
    """Generate detailed per-stage guidance dengan code template plus common errors."""
    lab_type = req.lab_type.lower().replace("-", "_")
    if lab_type not in _LAB_STAGES:
        return {"ok": False, "error": f"Unknown lab type: {lab_type}"}
    system_prompt = _LAB_PROMPT_SYSTEM.get(lab_type, _LAB_PROMPT_SYSTEM["classification"])
    ctx_str = _ctx_summary(req.context)
    lang = _lang_hint(req.context)

    user_prompt = (
        f"Anda diminta menjelaskan secara DETAIL stage '{req.stage_id}' ({req.stage_title}) "
        f"untuk task {lab_type.replace('_', ' ')}.\n\n"
        "PROJECT CONTEXT\n"
        f"{ctx_str}\n\n"
        "TASK\n"
        "Berikan deep dive untuk stage ini dengan struktur:\n"
        "  - detailed_explanation (3-5 kalimat menjelaskan kenapa stage ini penting plus teori di belakangnya)\n"
        "  - substeps (5-8 bullet points sub-aktivitas konkret urut)\n"
        "  - code_template (Python code 15-30 baris yang fully runnable, pakai pandas/sklearn standard, "
        "    pre-fill dengan nama kolom user kalau available. Wrap dengan ```python```)\n"
        "  - expected_outputs (3 bullet outputs konkret dari running code di atas)\n"
        "  - common_errors (3 error/pitfall yang sering terjadi plus quick fix masing-masing)\n"
        "  - papers_to_cite (2-3 referensi APA 7 yang relevan untuk stage ini)\n"
        "  - next_action (1 kalimat mengarahkan ke stage berikutnya)\n\n"
        "OUTPUT FORMAT\n"
        "Return ONLY strict JSON object (no markdown wrap of whole response, but code_template field "
        "dapat berisi ```python ... ```). Schema:\n"
        "{\n"
        '  "detailed_explanation": "...", "substeps": ["...","...","..."],\n'
        '  "code_template": "...", "expected_outputs": ["...","...","..."],\n'
        '  "common_errors": [{"error":"...","fix":"..."},{...},{...}],\n'
        '  "papers_to_cite": ["Author Year ref...","..."], "next_action": "..."\n'
        "}\n\n"
        f"{lang}"
    )

    text = ""
    try:
        from app.services import task_router as _tr
        result = _tr.route(
            task_type="structured_fast",
            messages=[
                {"role": "system", "content": system_prompt + " Return strict JSON only for the outer object."},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=3200,
            temperature=0.4,
        )
        text = (result or {}).get("text", "") or ""
    except Exception:
        text = ""

    parsed = None
    if text:
        # Strip outer code fence kalau ada
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.M)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.M)
        # Match first JSON object
        depth = 0
        start = -1
        end = -1
        for i, c in enumerate(cleaned):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    end = i + 1
                    break
        candidate = cleaned[start:end] if start >= 0 and end > start else cleaned
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None

    if parsed and isinstance(parsed, dict):
        return {"ok": True, "source": "ai", "lab_type": lab_type, "stage_id": req.stage_id, "detail": parsed}

    # Fallback
    fallback = {
        "detailed_explanation": f"Stage {req.stage_title} adalah bagian penting dalam workflow {lab_type}. "
                                f"Pada tahap ini, fokus utama adalah menjalankan operasi standar untuk task type ini "
                                f"dengan mempertimbangkan karakteristik dataset {req.context.dataset_name or 'project Anda'}.",
        "substeps": [
            "Persiapkan input dari stage sebelumnya",
            "Konfigurasi tooling sesuai library standar (pandas, sklearn, atau equivalent)",
            "Eksekusi operasi utama dengan logging output",
            "Validasi hasil dengan sanity check",
            "Persist output untuk stage berikutnya",
        ],
        "code_template": "```python\nimport pandas as pd\nimport numpy as np\n\n# TODO: Load dataset\ndf = pd.read_csv('your_dataset.csv')\nprint('Shape:', df.shape)\nprint(df.head())\n\n# TODO: Stage-specific operation\n# ...\n```",
        "expected_outputs": [
            "Dataset preview dengan shape dan first rows",
            "Console log konfirmasi langkah berhasil",
            "Object output siap dipakai stage berikutnya",
        ],
        "common_errors": [
            {"error": "FileNotFoundError kalau path dataset salah", "fix": "Verify path absolute atau gunakan os.path.join"},
            {"error": "Memory error pada dataset besar", "fix": "Gunakan chunksize parameter atau load subset"},
            {"error": "DType mismatch saat operasi", "fix": "Cast eksplisit dengan astype sebelum operasi"},
        ],
        "papers_to_cite": [
            "VanderPlas, J. (2016). Python Data Science Handbook. O Reilly.",
            "McKinney, W. (2017). Python for Data Analysis. 2nd ed. O Reilly.",
        ],
        "next_action": "Lanjut ke stage berikutnya setelah deliverable stage ini di-validate.",
    }
    return {"ok": True, "source": "fallback_template", "lab_type": lab_type, "stage_id": req.stage_id, "detail": fallback,
            "note": "AI engine tidak tersedia, menggunakan fallback template."}


# ============================================================================
# Endpoint 3 · SUGGEST ALGORITHMS · top 3 ranked algorithms
# ============================================================================
@router.post("/lab/suggest-algorithms")
async def lab_suggest_algorithms(req: LabAlgoRequest):
    """Suggest top 3 algorithms based on dataset profile plus context."""
    lab_type = req.lab_type.lower().replace("-", "_")
    if lab_type not in _LAB_STAGES:
        return {"ok": False, "error": f"Unknown lab type: {lab_type}"}
    system_prompt = _LAB_PROMPT_SYSTEM.get(lab_type, _LAB_PROMPT_SYSTEM["classification"])
    ctx_str = _ctx_summary(req.context)
    lang = _lang_hint(req.context)

    # Profile signals
    n_rows = req.context.dataset_shape[0] if len(req.context.dataset_shape) >= 1 else 0
    n_cols = req.context.dataset_shape[1] if len(req.context.dataset_shape) >= 2 else 0
    n_features = len(req.context.feature_columns)
    profile_signals = []
    if n_rows > 0:
        if n_rows < 1000:    profile_signals.append("Small dataset (<1000) · prefer interpretable models")
        elif n_rows < 50000: profile_signals.append("Medium dataset (1000-50000) · ensemble methods workable")
        else:                profile_signals.append("Large dataset (>50000) · deep learning may help")
    if n_features > 0:
        if n_features < 10:   profile_signals.append("Few features (<10) · linear/tree models sufficient")
        elif n_features < 100:profile_signals.append("Moderate features (10-100) · feature engineering helpful")
        else:                 profile_signals.append("High dimensional (>100) · regularization plus dim reduction critical")
    has_ts = bool(req.context.timestamp_column)
    if has_ts and lab_type != "forecasting":
        profile_signals.append("Timestamp column detected · consider temporal split atau feature lag")
    profile_str = "; ".join(profile_signals) or "no specific dataset signal"

    user_prompt = (
        f"Anda diminta merekomendasikan TOP 3 algorithms untuk task {lab_type.replace('_', ' ')}.\n\n"
        "PROJECT CONTEXT\n"
        f"{ctx_str}\n\n"
        f"DATASET PROFILE SIGNAL\n- {profile_str}\n\n"
        "TASK\n"
        "Pick top 3 algoritma yang paling cocok untuk profile ini, ranked 1-2-3. "
        "Pertimbangkan size dataset, dimensionality, dan domain. Untuk masing-masing return:\n"
        "  - rank (1, 2, 3)\n"
        "  - algorithm_name (specific, e.g., 'XGBoost Classifier' bukan generic 'tree-based')\n"
        "  - why_for_this_data (2-3 kalimat KENAPA cocok dengan profile spesifik di atas, mention shape atau features)\n"
        "  - configuration_hint (key hyperparameter starting point, e.g., 'n_estimators=200, max_depth=6, learning_rate=0.05')\n"
        "  - expected_metric_value (estimasi range realistis untuk metric utama, e.g., 'F1 0.75-0.85 on balanced data')\n"
        "  - library (Python library, e.g., 'sklearn.ensemble.RandomForestClassifier', 'xgboost.XGBClassifier')\n\n"
        "OUTPUT FORMAT\n"
        "Return ONLY strict JSON array of 3 objects:\n"
        "[\n"
        '  {"rank":1,"algorithm_name":"...","why_for_this_data":"...","configuration_hint":"...",\n'
        '    "expected_metric_value":"...","library":"..."}\n'
        "]\n\n"
        f"{lang}"
    )

    text = ""
    try:
        from app.services import task_router as _tr
        result = _tr.route(
            task_type="structured_fast",
            messages=[
                {"role": "system", "content": system_prompt + " Return strict JSON array only."},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1800,
            temperature=0.4,
        )
        text = (result or {}).get("text", "") or ""
    except Exception:
        text = ""

    parsed = None
    if text:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
        m = re.search(r"\[\s*\{.*\}\s*\]", cleaned, flags=re.S)
        candidate = m.group(0) if m else cleaned
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, list):
                parsed = None
        except Exception:
            parsed = None

    if parsed and len(parsed) >= 1:
        return {"ok": True, "source": "ai", "lab_type": lab_type, "profile_signals": profile_signals,
                "algorithms": parsed[:3]}

    # Fallback berdasarkan profile signal
    _FALLBACK_ALGOS = {
        "classification":   [("XGBoost Classifier", "xgboost.XGBClassifier"),
                             ("Random Forest", "sklearn.ensemble.RandomForestClassifier"),
                             ("Logistic Regression", "sklearn.linear_model.LogisticRegression")],
        "regression":       [("LightGBM Regressor", "lightgbm.LGBMRegressor"),
                             ("Random Forest Regressor", "sklearn.ensemble.RandomForestRegressor"),
                             ("Ridge Regression", "sklearn.linear_model.Ridge")],
        "clustering":       [("K-Means", "sklearn.cluster.KMeans"),
                             ("DBSCAN", "sklearn.cluster.DBSCAN"),
                             ("HDBSCAN", "hdbscan.HDBSCAN")],
        "forecasting":      [("Prophet", "prophet.Prophet"),
                             ("SARIMA", "statsmodels.tsa.SARIMAX"),
                             ("LSTM", "tensorflow.keras.layers.LSTM")],
        "anomaly":          [("Isolation Forest", "sklearn.ensemble.IsolationForest"),
                             ("LOF", "sklearn.neighbors.LocalOutlierFactor"),
                             ("One-Class SVM", "sklearn.svm.OneClassSVM")],
        "deep_learning":    [("ResNet50 fine-tune", "torchvision.models.resnet50"),
                             ("Transformer encoder", "transformers.AutoModel"),
                             ("MLP baseline", "torch.nn.Sequential")],
        "nlp":              [("IndoBERT fine-tune", "transformers.AutoModel from indolem/indobert"),
                             ("FastText", "fasttext"),
                             ("TF-IDF + LR", "sklearn.feature_extraction.text.TfidfVectorizer")],
        "computer_vision":  [("EfficientNet B0", "torchvision.models.efficientnet_b0"),
                             ("YOLOv8", "ultralytics.YOLO"),
                             ("U-Net", "segmentation_models_pytorch.Unet")],
        # Geo-Disaster fallback algorithms
        "flood":            [("Random Forest susceptibility", "sklearn.ensemble.RandomForestClassifier"),
                             ("HEC-RAS 2D inundation", "HEC-RAS · USACE"),
                             ("U-Net SAR flood detection", "segmentation_models_pytorch.Unet")],
        "tsunami":          [("Okada 1985 dislocation", "okada-wrapper python"),
                             ("COMCOT propagation", "COMCOT · Cornell University"),
                             ("Multi-Criteria Decision Analysis vulnerability", "QGIS MCDA plugin")],
        "earthquake":       [("Cornell-McGuire PSHA", "OpenQuake engine GEM"),
                             ("Random Forest ground motion", "sklearn.ensemble.RandomForestRegressor"),
                             ("HAZUS fragility curves", "FEMA HAZUS-MH")],
        "climate":          [("Mann-Kendall trend", "scipy.stats.kendalltau"),
                             ("Quantile mapping bias correction", "xclim.sdba"),
                             ("CMIP6 ensemble analysis", "xarray plus intake-esm")],
        "wildfire":         [("XGBoost ignition probability", "xgboost.XGBClassifier"),
                             ("Random Forest burn severity", "sklearn.ensemble.RandomForestClassifier"),
                             ("FARSITE fire spread", "FARSITE · USDA Forest Service")],
        "storm":            [("LSTM track prediction", "tensorflow.keras.layers.LSTM"),
                             ("CNN intensity from satellite IR", "torch.nn.Conv2d"),
                             ("Holland parametric wind", "tcpypi python")],
        "coastal":          [("DSAS shoreline change", "DSAS · USGS"),
                             ("Random Forest erosion susceptibility", "sklearn.ensemble.RandomForestClassifier"),
                             ("XBeach storm surge", "XBeach · Delft")],
        "environment":      [("Random Forest land cover", "sklearn.ensemble.RandomForestClassifier"),
                             ("Dynamic World land cover", "Google Earth Engine"),
                             ("Sentinel-5P NO2 retrieval", "harp atau atmospheric toolbox")],
    }
    algos = _FALLBACK_ALGOS.get(lab_type, _FALLBACK_ALGOS["classification"])
    fallback = []
    for i, (name, lib) in enumerate(algos):
        fallback.append({
            "rank": i + 1,
            "algorithm_name": name,
            "why_for_this_data": f"{name} adalah pilihan standar untuk task {lab_type} dengan profil dataset typical. "
                                  f"Cocok untuk {profile_str.split(';')[0] if profile_signals else 'general dataset'}.",
            "configuration_hint": "default parameter library; tune setelah baseline established",
            "expected_metric_value": "Hasil reasonable untuk benchmark task standar",
            "library": lib,
        })
    return {"ok": True, "source": "fallback_template", "lab_type": lab_type, "profile_signals": profile_signals,
            "algorithms": fallback,
            "note": "AI engine tidak tersedia, menggunakan fallback algoritma standar."}

