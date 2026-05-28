"""
BNPB Service — Disaster Information Database adapter
====================================================
Ingests flood event records from BNPB's national disaster databases.

ACCESS PATHS (verified May 2026 — no programmatic API exists):

  1. Portal Satu Data Bencana Indonesia
     URL: https://data.bnpb.go.id/dataset/data-bencana-indonesia
     Format: CSV / XLSX downloadable (annual compilation files).

  2. DIBI interactive search (Desinventar/Desconsultar engine)
     URL: https://dibi.bnpb.go.id/
     Format: HTML query results — supports CSV export per query.

  3. GIS Geoportal
     URL: https://gis.bnpb.go.id/
     Format: ArcGIS REST FeatureServer endpoints (when authenticated).

This service expects the user to download the CSV/XLSX and place it under
  storage/uploads/bnpb_dibi_<year>.csv
The loader then normalizes columns to Nexlytics' canonical schema:
  province_id, date, flood_count, victims, damage_idr

CITATIONS (Method Monitor)
  * Marulanda, M.C., Cardona, O.D., Mora, M.G., Barbat, A.H. (2014).
    Disaster databases in disaster risk modelling. Nat. Hazards 73(2).
  * UNDRR (2020). Sendai Framework Monitor — disaster loss data
    standardization (Desinventar lineage).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .indonesia_provinces import INDONESIA_PROVINCES


# Canonical column mapping — DIBI exports vary by year, this dict captures
# the field names commonly seen in the public CSV downloads.
COLUMN_MAPS = {
    "tanggal_kejadian":   "date",
    "tgl_kejadian":       "date",
    "tanggal":            "date",
    "kejadian_date":      "date",
    "provinsi":           "province_name",
    "nama_provinsi":      "province_name",
    "province":           "province_name",
    "jenis_bencana":      "disaster_type",
    "jenis":              "disaster_type",
    "korban_jiwa":        "victims",
    "korban_meninggal":   "deaths",
    "luka_luka":          "injured",
    "menderita":          "affected",
    "mengungsi":          "displaced",
    "rumah_rusak_berat":  "houses_destroyed",
    "rumah_rusak_sedang": "houses_damaged_med",
    "rumah_rusak_ringan": "houses_damaged_light",
    "kerugian_rupiah":    "damage_idr",
    "estimasi_kerugian":  "damage_idr",
}


def _normalize_province(name: str) -> Optional[str]:
    if not name: return None
    n = name.strip().lower().replace("provinsi ", "").replace("prov.", "").replace("prov ", "")
    n = n.replace("d.i.", "di").replace("d.i ", "di ").replace("daerah istimewa ", "di ")
    n = n.replace("d.k.i.", "dki").replace("d.k.i ", "dki ").replace("daerah khusus ibukota ", "dki ")
    for p in INDONESIA_PROVINCES:
        if p["name_id"].lower() in n or n in p["name_id"].lower():
            return p["id"]
    # Some BNPB exports use English-leaning names
    for p in INDONESIA_PROVINCES:
        if p["name_en"].lower() in n or n in p["name_en"].lower():
            return p["id"]
    return None


def load_csv(path: str, disaster_filter: str = "banjir") -> Dict[str, Any]:
    """Read a single BNPB CSV/XLSX and normalize to Nexlytics schema."""
    if not HAS_PANDAS:
        return {"status": "error", "message": "pandas unavailable"}
    if not os.path.exists(path):
        return {"status": "error", "message": f"File not found: {path}"}
    t0 = time.perf_counter()
    try:
        df = (pd.read_excel(path) if path.lower().endswith(("xlsx", "xls"))
              else pd.read_csv(path, low_memory=False, encoding="utf-8", on_bad_lines="skip"))
    except Exception as e:
        return {"status": "error", "message": f"Could not read file: {e}"}

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_MAPS.items() if k in df.columns})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date"])

    if "disaster_type" in df.columns:
        if disaster_filter:
            df = df[df["disaster_type"].astype(str).str.lower().str.contains(disaster_filter, na=False)]

    if "province_name" in df.columns:
        df["province_id"] = df["province_name"].apply(_normalize_province)
        df = df.dropna(subset=["province_id"])

    df["flood_count"] = 1
    if "victims" not in df.columns:
        # Compose victims from sub-fields when present
        for c in ("affected", "displaced", "deaths", "injured"):
            if c in df.columns:
                df["victims"] = df.get("victims", 0).fillna(0) + df[c].fillna(0)
        df["victims"] = df.get("victims", 0)
    if "damage_idr" not in df.columns:
        df["damage_idr"] = 0

    out_cols = ["province_id", "date", "flood_count", "victims", "damage_idr"]
    out_cols += [c for c in ("disaster_type", "deaths", "displaced", "houses_destroyed") if c in df.columns]
    df = df[[c for c in out_cols if c in df.columns]].copy()

    return {
        "status": "success",
        "rows": int(len(df)),
        "data": df,
        "duration_ms": int((time.perf_counter() - t0) * 1000),
        "method_monitor": {
            "method": "BNPB CSV/XLSX ingestion + province normalization + flood-only filter",
            "limitations": [
                "Schema varies year-to-year; new columns may need to be added to COLUMN_MAPS.",
                "Province name fuzzy-matching can mis-route ambiguous names. Manual review recommended."
            ],
            "citations": [
                "BNPB DIBI — Data Informasi Bencana Indonesia.",
                "Marulanda et al. (2014) Nat. Hazards 73(2) — Disaster databases.",
            ],
        },
    }


def load_directory(folder: str, disaster_filter: str = "banjir") -> Dict[str, Any]:
    """Walk a folder of BNPB CSV/XLSX files and concatenate."""
    if not HAS_PANDAS:
        return {"status": "error", "message": "pandas unavailable"}
    p = Path(folder)
    if not p.exists():
        return {"status": "error", "message": f"Folder not found: {folder}"}
    frames = []
    files_loaded: List[str] = []
    for f in sorted(p.rglob("*")):
        if f.suffix.lower() not in (".csv", ".xlsx", ".xls"): continue
        r = load_csv(str(f), disaster_filter=disaster_filter)
        if r["status"] == "success":
            frames.append(r["data"])
            files_loaded.append(str(f.name))
    if not frames:
        return {"status": "error", "message": "No CSV/XLSX could be parsed in folder."}
    df = pd.concat(frames, ignore_index=True).sort_values(["province_id", "date"])
    return {
        "status": "success",
        "rows": int(len(df)),
        "files_loaded": files_loaded,
        "data": df,
        "method_monitor": {"method": "Aggregate of N annual BNPB CSV/XLSX files."},
    }


def synthesize_demo_events(province_ids: List[str], start: str, end: str,
                           lambda_per_year: float = 12.0,
                           seed: int = 42) -> Dict[str, Any]:
    """Generate plausible synthetic flood events per province for demos
    (when real BNPB CSV is not yet downloaded). Poisson process, peak in
    DJF monsoon. Seeded for reproducibility."""
    if not HAS_PANDAS:
        return {"status": "error", "message": "pandas unavailable"}
    rng = np.random.default_rng(seed)
    sd = pd.Timestamp(start); ed = pd.Timestamp(end)
    rows = []
    for pid in province_ids:
        years = ed.year - sd.year + 1
        n_events = int(rng.poisson(lam=lambda_per_year * years))
        for _ in range(n_events):
            doy = int(rng.integers(1, 366))
            year = int(rng.integers(sd.year, ed.year + 1))
            try: d = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
            except Exception: continue
            if d < sd or d > ed: continue
            # Bias toward DJF
            if d.month not in (12, 1, 2):
                if rng.random() < 0.65: continue
            rows.append({
                "province_id": pid, "date": d,
                "flood_count": 1,
                "victims": int(max(0, rng.gamma(2.0, 80))),
                "damage_idr": int(max(0, rng.gamma(2.0, 5e8))),
                "disaster_type": "banjir",
            })
    df = pd.DataFrame(rows)
    return {"status": "success", "mode": "synthetic_demo",
            "rows": len(df), "data": df,
            "method_monitor": {"method": "Synthetic Poisson flood events with DJF bias (seed=42)"}}
