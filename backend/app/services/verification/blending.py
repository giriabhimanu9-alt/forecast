"""Weight learning (training period only) and blend application.

Methods
-------
equal           w_i = 1 / N
inverse_error   w_i = (1 / (e_i + eps)) / sum_j (1 / (e_j + eps)),  e_i = training MAE
                over all of India, per lead day
regional        same formula, e_i computed separately inside each regional block

Errors are computed on the COMMON training pairs (grid points and samples
where every available model and the observation are finite), so a model is
not rewarded for missing a difficult day.

Applying a blend needs only forecasts and the weight table — never
observations — so test-period observations cannot influence weights.
Missing models are removed and the remaining weights renormalised per grid
point (sum = 1).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from app.services.verification.regions import REGIONS

METHODS = ("equal", "inverse_error", "regional")
ALL_REGIONS = "ALL"


def inverse_error_weights(errors: dict[str, float], epsilon: float) -> dict[str, float]:
    """Normalised inverse-error weights; models with missing (NaN) error are dropped."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    usable = {m: e for m, e in errors.items() if e is not None and np.isfinite(e)}
    if any(e < 0 for e in usable.values()):
        raise ValueError("errors must be >= 0")
    if not usable:
        return {}
    inv = {m: 1.0 / (e + epsilon) for m, e in usable.items()}
    total = sum(inv.values())
    return {m: v / total for m, v in inv.items()}


def equal_weights(models: Iterable[str]) -> dict[str, float]:
    models = list(models)
    return {m: 1.0 / len(models) for m in models} if models else {}


def _common_mask(forecast: xr.DataArray, observation: xr.DataArray, models: list[str]) -> xr.DataArray:
    mask = observation.notnull()
    for m in models:
        mask = mask & forecast.sel(model=m).notnull()
    return mask


def _mae(forecast: xr.DataArray, observation: xr.DataArray, mask: xr.DataArray) -> float:
    err = np.abs(forecast - observation).where(mask)
    n = int(mask.sum())
    return float(err.sum()) / n if n else float("nan")


def learn_weights(
    train: xr.Dataset,
    regions: xr.DataArray,
    epsilon: float,
    variable: str = "precipitation",
    metric: str = "mae",
) -> pd.DataFrame:
    """Weight table learned from ``train`` (dims model, sample, latitude, longitude)."""
    if metric != "mae":
        raise ValueError("only metric='mae' is implemented")
    start = pd.Timestamp(train.observation_period_start.values.min()).isoformat()
    end = pd.Timestamp(train.observation_period_end.values.max()).isoformat()
    rows: list[dict] = []
    for lead_day in sorted(set(train.lead_day.values.tolist())):
        sub = train.isel(sample=np.flatnonzero(train.lead_day.values == lead_day))
        fc, obs = sub["forecast"], sub["observation"]
        available = [m for m in fc.model.values.tolist() if bool(fc.sel(model=m).notnull().any())]
        mask = _common_mask(fc, obs, available)
        window = f"{int(sub.start_lead_h.values[0])}-{int(sub.end_lead_h.values[0])}h"

        def add(method: str, region: str, weights: dict[str, float], errors: dict[str, float], npts: int) -> None:
            for m in fc.model.values.tolist():
                rows.append(
                    {
                        "method": method,
                        "model": m,
                        "region": region,
                        "variable": variable,
                        "lead_day": int(lead_day),
                        "lead_window": window,
                        "metric": metric,
                        "training_start": start,
                        "training_end": end,
                        "error": errors.get(m, float("nan")),
                        "weight": weights.get(m, 0.0),
                        "n_train_points": npts,
                        "available": m in weights,
                    }
                )

        global_err = {m: _mae(fc.sel(model=m), obs, mask) for m in available}
        add("equal", ALL_REGIONS, equal_weights(available), global_err, int(mask.sum()))
        add("inverse_error", ALL_REGIONS, inverse_error_weights(global_err, epsilon), global_err, int(mask.sum()))
        for region in REGIONS:
            rmask = mask & (regions == region)
            npts = int(rmask.sum())
            errs = {m: _mae(fc.sel(model=m), obs, rmask) for m in available}
            # A region without training data falls back to the all-India inverse-error weights.
            weights = inverse_error_weights(errs, epsilon) if npts else inverse_error_weights(global_err, epsilon)
            add("regional", region, weights, errs, npts)
    table = pd.DataFrame(rows)
    check_weight_table(table)
    return table


def check_weight_table(table: pd.DataFrame) -> None:
    if ((table.weight < 0) | (table.weight > 1)).any():
        raise ValueError("weights must be within [0, 1]")
    sums = table.groupby(["method", "region", "lead_day"]).weight.sum()
    bad = sums[(sums - 1.0).abs() > 1e-9]
    if not bad.empty:
        raise ValueError(f"weights do not sum to 1: {bad.to_dict()}")


def weight_field(table: pd.DataFrame, method: str, lead_day: int, models: list[str], regions: xr.DataArray) -> xr.DataArray:
    """Weights as (model, latitude, longitude) for one method and lead day."""
    sub = table[(table.method == method) & (table.lead_day == lead_day)]
    if sub.empty:
        raise KeyError(f"no weights for method={method} lead_day={lead_day}")
    out = xr.DataArray(
        np.zeros((len(models),) + regions.shape), dims=("model", "latitude", "longitude"),
        coords={"model": models, "latitude": regions.latitude, "longitude": regions.longitude},
    )
    for i, m in enumerate(models):
        rows = sub[sub.model == m]
        if method == "regional":
            lookup = dict(zip(rows.region, rows.weight))
            out[i] = xr.apply_ufunc(np.vectorize(lambda r: lookup.get(r, 0.0), otypes=[float]), regions)
        else:
            out[i] = float(rows.weight.iloc[0]) if len(rows) else 0.0
    return out


def apply_blend(forecast: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """Weighted blend over ``model``; missing members are dropped and weights renormalised per point."""
    available = forecast.notnull()
    w = weights.broadcast_like(forecast).where(available, 0.0)
    total = w.sum("model")
    blend = (forecast.fillna(0.0) * w).sum("model") / total.where(total > 0)
    blend.attrs = {"units": forecast.attrs.get("units", "mm")}
    return blend
