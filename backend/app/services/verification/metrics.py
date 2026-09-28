"""Verification metrics on paired forecast/observation grids.

Only grid points where BOTH forecast and observation are finite are scored
(IMD is NaN over the ocean). All metrics pool every (sample, grid point) pair.

Continuous:  MAE, RMSE, bias = mean(forecast - observation)       [mm / 24 h]
Categorical, per threshold t (event = value >= t mm / 24 h):
    hits H = f>=t & o>=t,  false alarms F = f>=t & o<t,  misses M = f<t & o>=t
    precision = H/(H+F)   recall (POD) = H/(H+M)
    F1 = 2PR/(P+R)        CSI = H/(H+M+F)
    POD = recall        FAR (false-alarm ratio) = F/(H+F) = 1 - precision
Undefined ratios (zero denominator) are reported as NaN, never as 0 or 1.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import xarray as xr

DEFAULT_THRESHOLDS_MM = (2.5, 15.6, 64.5)  # IMD light / moderate / heavy rainfall limits


def _paired(forecast: xr.DataArray, observation: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    f, o = xr.broadcast(forecast, observation)
    f, o = np.asarray(f.values, dtype="float64").ravel(), np.asarray(o.values, dtype="float64").ravel()
    ok = np.isfinite(f) & np.isfinite(o)
    return f[ok], o[ok]


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def continuous_scores(f: np.ndarray, o: np.ndarray) -> dict[str, float]:
    if f.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    err = f - o
    return {
        "n": int(f.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def categorical_scores(f: np.ndarray, o: np.ndarray, threshold: float) -> dict[str, float]:
    fe, oe = f >= threshold, o >= threshold
    hits = int(np.sum(fe & oe))
    false_alarms = int(np.sum(fe & ~oe))
    misses = int(np.sum(~fe & oe))
    precision = _ratio(hits, hits + false_alarms)
    recall = _ratio(hits, hits + misses)
    f1 = _ratio(2 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else float("nan")
    return {
        "threshold_mm": threshold,
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "observed_events": int(np.sum(oe)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "csi": _ratio(hits, hits + misses + false_alarms),
        "pod": recall,  # probability of detection = recall
        "far": _ratio(false_alarms, hits + false_alarms),  # false-alarm RATIO = 1 - precision
    }


def score(
    forecast: xr.DataArray, observation: xr.DataArray, thresholds: Iterable[float] = DEFAULT_THRESHOLDS_MM
) -> dict[str, float]:
    """Flat dict of continuous scores plus ``<metric>@<t>mm`` categorical scores."""
    f, o = _paired(forecast, observation)
    out: dict[str, float] = continuous_scores(f, o)
    for t in thresholds:
        cat = categorical_scores(f, o, t)
        for key in ("precision", "recall", "f1", "csi", "observed_events"):
            out[f"{key}@{t:g}mm"] = cat[key]
    return out


def correlation(f: np.ndarray, o: np.ndarray) -> float:
    """Pearson correlation of paired values; NaN when undefined (fewer than 3 pairs or zero variance)."""
    if f.size < 3 or np.std(f) == 0 or np.std(o) == 0:
        return float("nan")
    return float(np.corrcoef(f, o)[0, 1])


def extended_score(
    forecast: xr.DataArray, observation: xr.DataArray, thresholds: Iterable[float] = DEFAULT_THRESHOLDS_MM,
    heavy_threshold: float = 64.5,
) -> dict[str, float]:
    """Continuous scores + correlation + CSI/POD/FAR per threshold + errors on observed-heavy points."""
    f, o = _paired(forecast, observation)
    out: dict[str, float] = continuous_scores(f, o)
    out["corr"] = correlation(f, o)
    for t in thresholds:
        cat = categorical_scores(f, o, t)
        for key in ("csi", "pod", "far", "hits", "false_alarms", "misses", "observed_events"):
            out[f"{key}@{t:g}mm"] = cat[key]
    heavy = o >= heavy_threshold
    sub = continuous_scores(f[heavy], o[heavy])
    out["n_obs_heavy"] = sub["n"]
    out["mae_obs_heavy"] = sub["mae"]
    out["bias_obs_heavy"] = sub["bias"]
    return out
