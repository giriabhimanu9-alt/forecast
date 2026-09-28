"""Regime-conditioned adaptive weights: w(model | regime, lead_day, region).

regime       w = (1/(MAE+eps)) / sum(1/(MAE+eps)), MAE from the fit-period pairs of
             that (regime, lead day, region). Thin cells fall back hierarchically:
             (regime, lead_day, region) -> (regime, lead_day) all India -> static
             (lead_day) weights. The level used is recorded for every row.
extreme_aware  identical, except in regimes whose fit-period heavy-rain frequency
             exceeds the overall rate: there w ∝ each model's fit-period CSI at
             64.5 mm for (regime, lead_day), all India (heavy events are too sparse
             per region). If CSI is undefined or zero for every model, the MAE
             weights are kept. Nothing is forced; the numbers decide.

Weights are learned from observations of the fit period only; applying a
blend needs forecasts, regime labels and the table — never observations.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from app.services.verification.blending import apply_blend, inverse_error_weights
from app.services.verification.metrics import categorical_scores
from app.services.verification.regions import REGIONS

LEVEL_REGION = "regime+lead+region"
LEVEL_INDIA = "regime+lead (all India)"
LEVEL_STATIC = "lead only (static fallback)"
LEVEL_CSI = "regime+lead CSI@64.5 (all India)"


def _common_mask(fc: xr.DataArray, obs: xr.DataArray, models: list[str]) -> xr.DataArray:
    mask = obs.notnull()
    for m in models:
        mask = mask & fc.sel(model=m).notnull()
    return mask


def _mae(fc: xr.DataArray, obs: xr.DataArray, mask: xr.DataArray) -> float:
    n = int(mask.sum())
    return float(np.abs(fc - obs).where(mask).sum()) / n if n else float("nan")


def _available(fc: xr.DataArray) -> list[str]:
    return [m for m in fc.model.values.tolist() if bool(fc.sel(model=m).notnull().any())]


def learn_regime_weights(
    fit: xr.Dataset,
    labels: np.ndarray,
    regions: xr.DataArray,
    epsilon: float,
    min_samples: int = 3,
    min_points: int = 500,
    regime_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Weight table for every (regime, lead_day, region, model) using the fit-period samples."""
    models = fit.model.values.tolist()
    start = pd.Timestamp(fit.observation_period_start.values.min()).isoformat()
    end = pd.Timestamp(fit.observation_period_end.values.max()).isoformat()
    rows: list[dict] = []
    regimes = sorted(set(labels.tolist()))
    for lead_day in sorted(set(fit.lead_day.values.tolist())):
        lead_idx = np.flatnonzero(fit.lead_day.values == lead_day)
        lead = fit.isel(sample=lead_idx)
        avail_lead = _available(lead["forecast"])
        lead_mask = _common_mask(lead["forecast"], lead["observation"], avail_lead)
        static_err = {m: _mae(lead["forecast"].sel(model=m), lead["observation"], lead_mask) for m in avail_lead}
        static_w = inverse_error_weights(static_err, epsilon)
        for regime in regimes:
            idx = np.flatnonzero((fit.lead_day.values == lead_day) & (labels == regime))
            sub = fit.isel(sample=idx)
            avail = _available(sub["forecast"]) if idx.size else []
            mask = _common_mask(sub["forecast"], sub["observation"], avail) if idx.size else None
            india_err = {m: _mae(sub["forecast"].sel(model=m), sub["observation"], mask) for m in avail} if idx.size else {}
            india_pts = int(mask.sum()) if mask is not None else 0
            india_ok = idx.size >= min_samples and india_pts >= min_points and bool(india_err)
            for region in REGIONS:
                if india_ok:
                    rmask = mask & (regions == region)
                    npts = int(rmask.sum())
                    if npts >= min_points:
                        err = {m: _mae(sub["forecast"].sel(model=m), sub["observation"], rmask) for m in avail}
                        w, level = inverse_error_weights(err, epsilon), LEVEL_REGION
                    else:
                        err, npts = india_err, india_pts
                        w, level = inverse_error_weights(india_err, epsilon), LEVEL_INDIA
                else:
                    err, npts = static_err, int(lead_mask.sum())
                    w, level = static_w, LEVEL_STATIC
                for m in models:
                    rows.append({
                        "method": "regime", "regime": int(regime),
                        "regime_name": (regime_names or {}).get(int(regime), f"REGIME_{regime}"),
                        "lead_day": int(lead_day), "region": region, "model": m,
                        "weight": w.get(m, 0.0), "training_error": err.get(m, float("nan")),
                        "error_metric": "mae", "sample_count": int(idx.size), "n_points": npts, "level": level,
                        "training_start": start, "training_end": end,
                    })
    table = pd.DataFrame(rows)
    check_weights(table)
    return table


def learn_extreme_aware_weights(
    fit: xr.Dataset,
    labels: np.ndarray,
    regime_table: pd.DataFrame,
    heavy_regimes: Iterable[int],
    threshold: float = 64.5,
    min_samples: int = 3,
) -> pd.DataFrame:
    """Copy of the regime table where heavy regimes use CSI-proportional weights (data-driven)."""
    heavy_regimes = set(int(r) for r in heavy_regimes)
    table = regime_table.copy()
    table["method"] = "extreme_aware"
    table["error_metric"] = "mae"
    for lead_day in sorted(set(fit.lead_day.values.tolist())):
        for regime in heavy_regimes:
            idx = np.flatnonzero((fit.lead_day.values == lead_day) & (labels == regime))
            if idx.size < min_samples:
                continue
            sub = fit.isel(sample=idx)
            avail = _available(sub["forecast"])
            mask = _common_mask(sub["forecast"], sub["observation"], avail)
            csi = {}
            for m in avail:
                f = sub["forecast"].sel(model=m).where(mask).values.ravel()
                o = sub["observation"].where(mask).values.ravel()
                ok = np.isfinite(f) & np.isfinite(o)
                csi[m] = categorical_scores(f[ok], o[ok], threshold)["csi"]
            finite = {m: c for m, c in csi.items() if np.isfinite(c)}
            if not finite or sum(finite.values()) <= 0:
                continue  # CSI undefined/zero: keep MAE-based regime weights
            total = sum(finite.values())
            w = {m: c / total for m, c in finite.items()}
            sel = (table.lead_day == lead_day) & (table.regime == regime)
            table.loc[sel, "weight"] = table.loc[sel, "model"].map(lambda m: w.get(m, 0.0))
            table.loc[sel, "training_error"] = table.loc[sel, "model"].map(lambda m: 1.0 - csi.get(m, np.nan))
            table.loc[sel, "error_metric"] = f"1 - CSI@{threshold:g}mm"
            table.loc[sel, "level"] = LEVEL_CSI
            table.loc[sel, "n_points"] = int(mask.sum())
    check_weights(table)
    return table


def check_weights(table: pd.DataFrame) -> None:
    if ((table.weight < -1e-12) | (table.weight > 1 + 1e-12)).any():
        raise ValueError("weights outside [0, 1]")
    sums = table.groupby(["method", "regime", "lead_day", "region"]).weight.sum()
    bad = sums[(sums - 1).abs() > 1e-9]
    if not bad.empty:
        raise ValueError(f"weights do not sum to 1: {bad.head().to_dict()}")


def weight_field(table: pd.DataFrame, method: str, regime: int, lead_day: int, models: list[str],
                 regions: xr.DataArray) -> xr.DataArray:
    sub = table[(table.method == method) & (table.regime == regime) & (table.lead_day == lead_day)]
    if sub.empty:
        raise KeyError(f"no weights for {method} regime={regime} lead_day={lead_day}")
    out = xr.DataArray(np.zeros((len(models),) + regions.shape), dims=("model", "latitude", "longitude"),
                       coords={"model": models, "latitude": regions.latitude, "longitude": regions.longitude})
    for i, m in enumerate(models):
        lookup = dict(zip(sub[sub.model == m].region, sub[sub.model == m].weight))
        out[i] = xr.apply_ufunc(np.vectorize(lambda r: lookup.get(r, 0.0), otypes=[float]), regions)
    return out


def apply_regime_blend(samples: xr.Dataset, labels: np.ndarray, table: pd.DataFrame, method: str,
                       regions: xr.DataArray) -> xr.DataArray:
    """Blend every sample with the weights of its (regime, lead_day). Uses forecasts only."""
    models = samples.model.values.tolist()
    fc = samples["forecast"]
    cache: dict[tuple[int, int], xr.DataArray] = {}
    parts = []
    for s in range(samples.sizes["sample"]):
        key = (int(labels[s]), int(samples.lead_day.values[s]))
        if key not in cache:
            cache[key] = weight_field(table, method, key[0], key[1], models, regions)
        parts.append(apply_blend(fc.isel(sample=[s]), cache[key]))
    return xr.concat(parts, dim="sample")


def static_weight_table(fit: xr.Dataset, epsilon: float, method: str = "static") -> pd.DataFrame:
    """Lead-day static inverse-error weights (Phase 2 global method) in the Phase 3 table layout."""
    rows = []
    for lead_day in sorted(set(fit.lead_day.values.tolist())):
        lead = fit.isel(sample=np.flatnonzero(fit.lead_day.values == lead_day))
        avail = _available(lead["forecast"])
        mask = _common_mask(lead["forecast"], lead["observation"], avail)
        err = {m: _mae(lead["forecast"].sel(model=m), lead["observation"], mask) for m in avail}
        w = inverse_error_weights(err, epsilon)
        for m in fit.model.values.tolist():
            rows.append({"method": method, "lead_day": int(lead_day), "model": m, "weight": w.get(m, 0.0),
                         "training_error": err.get(m, float("nan")), "n_points": int(mask.sum())})
    return pd.DataFrame(rows)


def weight_differences(table: pd.DataFrame, static: pd.DataFrame, near_identical: float = 0.02) -> pd.DataFrame:
    """How much do weights change between regimes, and versus static weights?"""
    rows = []
    for (method, lead_day, region, model), g in table.groupby(["method", "lead_day", "region", "model"]):
        s = static[(static.lead_day == lead_day) & (static.model == model)].weight
        sw = float(s.iloc[0]) if len(s) else float("nan")
        row = {"method": method, "lead_day": lead_day, "region": region, "model": model, "static_weight": sw}
        for r in g.itertuples():
            row[f"weight_regime_{r.regime}"] = r.weight
            row[f"level_regime_{r.regime}"] = r.level
        row["spread_across_regimes"] = float(g.weight.max() - g.weight.min())
        row["max_abs_diff_from_static"] = float((g.weight - sw).abs().max())
        row["regime_specific_rows"] = int((g.level != LEVEL_STATIC).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["nearly_identical_across_regimes"] = out.spread_across_regimes < near_identical
    return out
