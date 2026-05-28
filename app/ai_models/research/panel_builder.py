"""
FloodPanelBuilder
=================
Build the canonical panel dataset for the thesis pipeline:

  panel:  province × time-step → [features..., label]

INPUT
  gee_features:  pandas.DataFrame with columns
                 [province_id, date, chirps_precip, modis_ndwi, gldas_soilmoi,
                  srtm_elevation, srtm_slope, srtm_twi, lc_built_up, lc_water,
                  jrc_water_occurrence, ...]   (one row per province per day)

  bnpb_events:   pandas.DataFrame with columns
                 [province_id, date, flood_count, victims, damage_idr]
                 (sparse — many province-days have no event)

ENGINEERED FEATURES
  Lag windows (Tehrany et al., 2014):
    rainfall_lag_{1,3,7,14,30}     ← shift CHIRPS
    rainfall_acc_{7,14,30}d        ← rolling sum
    rainfall_max_{7,30}d           ← rolling max
    soilmoi_lag_{1,7}              ← shift GLDAS
    ndwi_lag_{1,7}                 ← shift MODIS
  Antecedent Precipitation Index (Kohler & Linsley, 1951):
    API_t = k·API_{t-1} + P_t    (default k = 0.85)
  Calendar:
    month, dayofyear, monsoon_phase  (DJF / MAM / JJA / SON)

LABEL
  flood_in_next_h_days = 1 if any BNPB event in (t, t+h]   default h = 7
  Severity (regression): victims_in_next_7_days

TEMPORAL SPLIT
  Train  : 2016-01-01 ..  2022-12-31
  Val    : 2023-01-01 ..  2023-12-31
  Test   : 2024-01-01 ..  2025-12-31
  This is a STRICT temporal hold-out — no random shuffling, prevents data
  leakage in time-series settings (Bergmeir & Benítez, 2012, Inf. Sci. 191).

CITATIONS
  * Tehrany, M. S., Pradhan, B., Jebur, M. N. (2014). Flood susceptibility
    mapping using a novel ensemble weights-of-evidence and SVM models in GIS.
    J. Hydrol. 512, 332-343.
  * Kohler, M. A., Linsley, R. K. (1951). Predicting the runoff from storm
    rainfall. U.S. Weather Bureau Research Paper No. 34.
  * Bergmeir, C., Benítez, J. M. (2012). On the use of cross-validation for
    time series predictor evaluation. Information Sciences 191, 192-213.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _monsoon_phase(month: int) -> int:
    """Indonesian monsoon: 0=DJF (wet), 1=MAM (transition), 2=JJA (dry), 3=SON (transition)."""
    if month in (12, 1, 2):  return 0
    if month in (3, 4, 5):   return 1
    if month in (6, 7, 8):   return 2
    return 3


class FloodPanelBuilder:
    """Builds province × time panel data with lag features and labels."""

    name = "FloodPanelBuilder"
    domain = "research_pipeline"

    def __init__(
        self,
        prediction_horizon_days: int = 7,
        api_decay: float = 0.85,
        train_end: str = "2022-12-31",
        val_end: str = "2023-12-31",
        test_end: str = "2025-12-31",
    ) -> None:
        self.prediction_horizon_days = int(prediction_horizon_days)
        self.api_decay = float(api_decay)
        self.train_end = train_end
        self.val_end = val_end
        self.test_end = test_end

    # ------------------------------------------------------------------
    def build(self, gee_features, bnpb_events) -> Dict[str, Any]:
        if not HAS_PANDAS:
            return {"status": "error", "message": "pandas/numpy unavailable"}
        t0 = time.perf_counter()

        if not isinstance(gee_features, pd.DataFrame):
            gee_features = pd.DataFrame(gee_features)
        if not isinstance(bnpb_events, pd.DataFrame):
            bnpb_events = pd.DataFrame(bnpb_events)

        # Normalize date dtype
        for df in (gee_features, bnpb_events):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        df = gee_features.copy().sort_values(["province_id", "date"]).reset_index(drop=True)

        # 1. Lag features per province
        df = df.groupby("province_id", group_keys=False).apply(self._add_lag_features)

        # 2. Calendar features
        df["month"] = df["date"].dt.month
        df["dayofyear"] = df["date"].dt.dayofyear
        df["monsoon_phase"] = df["month"].apply(_monsoon_phase)

        # 3. Label: flood_in_next_h_days
        df = self._add_label(df, bnpb_events)

        # 4. Drop rows with any NaN feature (leading lag rows have NaNs)
        before = len(df)
        feature_cols = [c for c in df.columns if c not in {"province_id", "date", "label_flood", "label_victims"}]
        df = df.dropna(subset=feature_cols + ["label_flood"]).reset_index(drop=True)
        dropped_for_lag = before - len(df)

        # 5. Temporal split
        train_mask = df["date"] <= pd.Timestamp(self.train_end)
        val_mask = (df["date"] > pd.Timestamp(self.train_end)) & (df["date"] <= pd.Timestamp(self.val_end))
        test_mask = (df["date"] > pd.Timestamp(self.val_end)) & (df["date"] <= pd.Timestamp(self.test_end))
        train, val, test = df[train_mask], df[val_mask], df[test_mask]

        positive_rate = lambda d: float(d["label_flood"].mean()) if len(d) else 0.0
        duration_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "status": "success",
            "model_name": self.name,
            "panel": df,
            "splits": {"train": train, "val": val, "test": test},
            "schema": {
                "feature_columns": feature_cols,
                "label_columns": ["label_flood", "label_victims"],
                "n_features": len(feature_cols),
            },
            "stats": {
                "rows_total": int(len(df)),
                "rows_dropped_for_lag": int(dropped_for_lag),
                "n_provinces": int(df["province_id"].nunique()),
                "date_min": str(df["date"].min().date()) if len(df) else None,
                "date_max": str(df["date"].max().date()) if len(df) else None,
                "train_rows": int(len(train)),
                "val_rows": int(len(val)),
                "test_rows": int(len(test)),
                "train_positive_rate": round(positive_rate(train), 4),
                "val_positive_rate": round(positive_rate(val), 4),
                "test_positive_rate": round(positive_rate(test), 4),
            },
            "duration_ms": duration_ms,
            "method_monitor": {
                "method": "Province-time panel with lag windows + Antecedent Precipitation Index + monsoon phase + temporal split",
                "why_used": "Standard panel structure for spatio-temporal flood ML (Tehrany et al., 2014).",
                "formulas": [
                    f"flood_label_t = 1[∃ BNPB event in (t, t+{self.prediction_horizon_days}d]]",
                    f"API_t = {self.api_decay}·API_{{t-1}} + rainfall_t  (Kohler & Linsley, 1951)",
                    "rainfall_acc_h = Σ_{i=0}^{h-1} rainfall_{t-i}",
                ],
                "limitations": [
                    "Strict temporal split prevents data leakage but may underfit if test era has unprecedented events.",
                    "Province-level aggregation hides intra-province variability — kabupaten-level is a follow-up.",
                ],
                "citations": [
                    "Tehrany et al. (2014) J. Hydrol. 512:332-343.",
                    "Kohler & Linsley (1951) U.S. Weather Bureau RP 34.",
                    "Bergmeir & Benítez (2012) Inf. Sci. 191:192-213.",
                ],
            },
        }

    # ------------------------------------------------------------------
    def _add_lag_features(self, group):
        g = group.copy()
        if "chirps_precip" in g.columns:
            for lag in (1, 3, 7, 14, 30):
                g[f"rainfall_lag_{lag}"] = g["chirps_precip"].shift(lag)
            for win in (7, 14, 30):
                g[f"rainfall_acc_{win}d"] = g["chirps_precip"].rolling(win, min_periods=win).sum()
            for win in (7, 30):
                g[f"rainfall_max_{win}d"] = g["chirps_precip"].rolling(win, min_periods=win).max()
            # Antecedent Precipitation Index
            api = []
            prev = 0.0
            for v in g["chirps_precip"].fillna(0).values:
                prev = self.api_decay * prev + float(v)
                api.append(prev)
            g["api_kohler_linsley"] = api
        if "gldas_soilmoi" in g.columns:
            for lag in (1, 7):
                g[f"soilmoi_lag_{lag}"] = g["gldas_soilmoi"].shift(lag)
        if "modis_ndwi" in g.columns:
            for lag in (1, 7):
                g[f"ndwi_lag_{lag}"] = g["modis_ndwi"].shift(lag)
        return g

    # ------------------------------------------------------------------
    def _add_label(self, df, events):
        if events.empty:
            df["label_flood"] = 0
            df["label_victims"] = 0.0
            return df
        h = self.prediction_horizon_days
        # Index events by (province_id, date)
        events = events.sort_values(["province_id", "date"]).copy()
        events["flood_count"] = events.get("flood_count", 1)
        events["victims"] = events.get("victims", 0).fillna(0)
        ev = events.groupby(["province_id", "date"]).agg({
            "flood_count": "sum",
            "victims": "sum",
        }).reset_index()

        # For each row in df, look ahead up to h days within province
        labels_flood = []
        labels_victims = []
        # Index for fast lookup
        ev_idx = ev.set_index(["province_id", "date"])
        ev_dates_per_prov = ev.groupby("province_id")["date"].apply(list).to_dict()
        for prov, d in zip(df["province_id"].values, df["date"].values):
            d = pd.Timestamp(d)
            future_dates = ev_dates_per_prov.get(prov, [])
            sub = [ed for ed in future_dates if d < ed <= d + pd.Timedelta(days=h)]
            if sub:
                tot_count = int(sum(ev_idx.loc[(prov, ed), "flood_count"] for ed in sub))
                tot_victims = float(sum(ev_idx.loc[(prov, ed), "victims"] for ed in sub))
                labels_flood.append(1)
                labels_victims.append(tot_victims)
            else:
                labels_flood.append(0)
                labels_victims.append(0.0)
        df["label_flood"] = labels_flood
        df["label_victims"] = labels_victims
        return df
