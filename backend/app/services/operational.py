"""Operational GFS + ECMWF rainfall blend (Phase 5), reusing the validated Phase 1–4 components.

For the latest published 00 UTC cycle:
  * fetch precipitation with the Phase 1 providers (cache, validation, normalisation);
  * form IMD-aligned 24 h windows (03 UTC -> 03 UTC, lead days 1-3) with the Phase 2 code;
  * assign the regime with the Phase 4 regime model (forecast-only features, fitted 2023-24);
  * blend with the Phase 4 regional regime weights (Experiment A, validated on 2025);
  * handle missing models with ``production.blend_available`` (never fabricates a forecast);
  * flag heavy rain >= 64.5 mm per model and for the blend.

Validation caveat carried into every product: on the 2025 test season the blend had the
lowest MAE, but ECMWF alone had higher heavy-rain CSI, so ECMWF exceedances are shown too.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from app.config import BACKEND_ROOT, Settings
from app.services.data.ecmwf import ECMWFProvider
from app.services.data.gfs import GFSProvider
from app.services.pipeline import run_forecast_provider, save_netcdf
from app.services.verification.accumulation import forecast_to_24h_accumulation, required_lead_hours, verification_windows
from app.services.verification.adaptive import weight_field
from app.services.verification.blending import apply_blend
from app.services.verification.production import NoForecastAvailable, blend_available
from app.services.verification.regimes import RegimeModel, regime_features
from app.services.verification.regions import REGIONS, region_map

log = logging.getLogger(__name__)

HEAVY_MM = 64.5
LEAD_DAYS = (1, 2, 3)
MODELS = ("GFS", "ECMWF")
PHASE4 = BACKEND_ROOT / "results" / "phase4"
REGIME_MODEL = PHASE4 / "regime_analysis" / "expA_regime_model_rain.json"
REGIME_WEIGHTS = PHASE4 / "weights" / "expA_adaptive_weights.csv"


def store_dir(settings: Settings) -> Path:
    return settings.processed_dir / "operational"


def land_mask(settings: Settings) -> xr.DataArray:
    """IMD land mask on the target grid (static geography), cached after first use."""
    path = store_dir(settings) / "imd_land_mask.nc"
    if path.is_file():
        with xr.open_dataarray(path) as da:
            return da.load().astype(bool)
    aligned = PHASE4 / "datasets" / "expA_aligned_2025.nc"
    if aligned.is_file():
        with xr.open_dataset(aligned) as ds:
            mask = ds["observation"].isel(sample=slice(0, 5)).notnull().any("sample").load()
    else:  # fall back to one IMD archive day through the Phase 1 provider
        from app.services.data.observations import IMDRainfallProvider

        s = replace(settings, imd_source="archive", imd_start_date=datetime(2025, 7, 1).date(),
                    imd_end_date=datetime(2025, 7, 1).date())
        obs, _ = IMDRainfallProvider(s).get_observations()
        mask = obs["precipitation"].isel(time=0).notnull()
    mask = mask.drop_vars([c for c in mask.coords if c not in ("latitude", "longitude")]).rename("land")
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.astype("int8").to_netcdf(path)
    return mask.astype(bool)


def latest_cycle(providers, lead_hours, now: datetime | None = None, days_back: int = 3) -> datetime | None:
    """Most recent 00 UTC cycle (the validated cycle) published by ALL providers; if none within
    ``days_back`` days, the most recent cycle published by at least one (single-model fallback)."""
    now = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    cands = [(now - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0) for d in range(days_back + 1)]
    published = {c: [p.is_available(c, lead_hours) for p in providers] for c in cands}
    for rule in (all, any):
        for c in cands:
            if rule(published[c]):
                return c
    return None


def blend_fields(fields: dict[str, xr.DataArray | None], regime: int, lead_day: int, table: pd.DataFrame,
                 regions: xr.DataArray) -> tuple[xr.DataArray, dict[str, Any]]:
    """Regional regime-weighted blend with missing-model fallback and explanation metadata."""
    rows = table[(table.regime == regime) & (table.lead_day == lead_day)]
    india_w = {m: float(rows[rows.model == m].weight.mean()) for m in MODELS}
    res = blend_available(fields, india_w, context={"regime": regime, "lead_day": lead_day})
    meta = res.metadata
    avail = meta["available_models"]
    if len(avail) >= 2:  # regional weights, renormalised per point over available models
        w = weight_field(table, "regime", regime, lead_day, avail, regions)
        stack = xr.concat([fields[m] for m in avail], dim=xr.DataArray(avail, dims="model", name="model"))
        out = apply_blend(stack, w)
    else:
        out = res.forecast
    meta["regional_weights"] = {r: {m: round(float(rows[(rows.region == r) & (rows.model == m)].weight.iloc[0]), 4)
                                    for m in MODELS} for r in REGIONS}
    meta["training_error_mae_mm"] = {m: round(float(rows[rows.model == m].training_error.mean()), 3) for m in MODELS}
    meta["sample_count"] = int(rows.sample_count.iloc[0]) if len(rows) else 0
    return out, meta


def alerts_for(field: xr.DataArray, land: xr.DataArray, regions: xr.DataArray) -> dict[str, Any]:
    heavy = (field >= HEAVY_MM) & land
    n = int(heavy.sum())
    out = {"cells": n, "max_mm": round(float(field.where(land).max()), 1) if bool(field.where(land).notnull().any()) else None}
    out["regions"] = {r: int((heavy & (regions == r)).sum()) for r in REGIONS if int((heavy & (regions == r)).sum())}
    return out


def build_operational_forecast(settings: Settings, init: datetime | None = None, http=None) -> dict[str, Any]:
    """Run (or reuse cached inputs for) one operational cycle; saves NetCDF + JSON summary."""
    table = pd.read_csv(REGIME_WEIGHTS)
    table = table[(table.feature_set == "rain") & (table.method == "regime")]
    regime_model = RegimeModel.load(REGIME_MODEL)
    s = replace(settings, variables=("precipitation",), forecast_init_time=None)
    probe = [cls(s, http=http) for cls in (GFSProvider, ECMWFProvider)]
    leads = required_lead_hours(verification_windows(datetime(2000, 1, 1), LEAD_DAYS))
    init = init or latest_cycle(probe, leads)
    if init is None:
        raise NoForecastAvailable("no 00 UTC cycle published by GFS or ECMWF in the last 3 days")
    s = replace(s, lead_hours=tuple(leads))
    status, datasets = {}, {}
    for p in [cls(s, http=http) for cls in (GFSProvider, ECMWFProvider)]:
        r = run_forecast_provider(p, s, init)
        status[p.name] = {"status": r.status.value, "message": r.message, "downloads": r.downloads, "cache_hits": r.cache_hits}
        if r.ok:
            datasets[p.name] = r.dataset
    land = land_mask(settings)
    regions = region_map(land.latitude.values, land.longitude.values)
    products, days = {}, []
    for w in verification_windows(init, LEAD_DAYS):
        fields: dict[str, xr.DataArray | None] = {m: None for m in MODELS}
        for m, ds in datasets.items():
            pr = ds["precipitation"].sel(model=m).isel(time=0, drop=True)
            fields[m] = forecast_to_24h_accumulation(pr, w.start_lead_h, w.end_lead_h).drop_vars("model", errors="ignore")
        avail = [m for m in MODELS if fields[m] is not None]
        if not avail:
            days.append({"lead_day": w.lead_day, "error": "no model available"})
            continue
        stack = xr.concat([fields[m] for m in avail], dim=xr.DataArray(avail, dims="model", name="model"))
        feats = regime_features(stack.expand_dims(sample=[0]).transpose("model", "sample", "latitude", "longitude"),
                                land, regions)
        regime = int(regime_model.predict(feats)[0])
        blend, meta = blend_fields(fields, regime, w.lead_day, table, regions)
        for name, f in [("blend", blend)] + [(m, fields[m]) for m in avail]:
            products[f"{name}_day{w.lead_day}"] = f.where(land).astype("float32")
        days.append({
            "lead_day": w.lead_day, "window_start_utc": w.start.isoformat(), "window_end_utc": w.end.isoformat(),
            "imd_valid_date": w.end.date().isoformat(), "regime": regime,
            "regime_name": json.loads(REGIME_MODEL.read_text())["regime_names"][str(regime)], **meta,
            "alerts": {"blend": alerts_for(blend, land, regions),
                       **{m: alerts_for(fields[m], land, regions) for m in avail}},
            "india_mean_mm": {k: round(float(products[f"{k}_day{w.lead_day}"].mean()), 2) for k in ["blend", *avail]},
        })
    summary = {
        "init_time": init.isoformat(), "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cycle_note": "00 UTC cycle; 24 h windows end 03 UTC (IMD day), the configuration validated in Phase 4",
        "model_status": status, "lead_days": days, "heavy_threshold_mm": HEAVY_MM,
        "weights_source": str(REGIME_WEIGHTS.relative_to(BACKEND_ROOT)),
        "caveat": "Validated on JJAS 2025: blend lowest MAE; ECMWF alone higher heavy-rain CSI (0.088 vs 0.078). "
                  "Treat ECMWF >= 64.5 mm exceedances as the primary heavy-rain signal.",
    }
    out = store_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    if products:
        save_netcdf(xr.Dataset(products), out / f"{init:%Y%m%d%H}.nc")
    (out / f"{init:%Y%m%d%H}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def list_runs(settings: Settings) -> list[str]:
    return sorted((p.stem for p in store_dir(settings).glob("*.json")), reverse=True)


def load_summary(settings: Settings, run: str) -> dict[str, Any]:
    return json.loads((store_dir(settings) / f"{run}.json").read_text(encoding="utf-8"))


def load_field(settings: Settings, run: str, product: str, lead_day: int, stride: int = 1) -> dict[str, Any]:
    with xr.open_dataset(store_dir(settings) / f"{run}.nc") as ds:
        da = ds[f"{product}_day{lead_day}"].load()
    da = da.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
    vals = np.where(np.isfinite(da.values), np.round(da.values, 1), np.nan)
    return {"product": product, "lead_day": lead_day, "lat": da.latitude.values.tolist(), "lon": da.longitude.values.tolist(),
            "values": [[None if np.isnan(v) else float(v) for v in row] for row in vals], "units": "mm/24h"}
