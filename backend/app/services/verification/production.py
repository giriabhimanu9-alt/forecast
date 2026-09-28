"""Production blending with missing-model fallback and explainable metadata (Phase 4, Steps 16-17).

Rules:
  * a model is unavailable if its forecast is None or entirely NaN;
  * unavailable models are removed and the remaining learned weights renormalised to sum 1;
  * if no learned weight remains positive, available models get equal weights (recorded);
  * one available model -> that model is the forecast;
  * no available model -> NoForecastAvailable (a forecast is never fabricated).
Point-wise gaps (a model NaN at some grid points) are handled by ``apply_blend``,
which renormalises weights per grid point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import xarray as xr

from app.models.schema import DataPipelineError, ProviderStatus
from app.services.verification.blending import apply_blend


class NoForecastAvailable(DataPipelineError):
    status = ProviderStatus.DOWNLOAD_FAILED


@dataclass
class BlendResult:
    forecast: xr.DataArray
    metadata: dict[str, Any] = field(default_factory=dict)


def blend_available(
    forecasts: Mapping[str, xr.DataArray | None],
    weights: Mapping[str, float],
    context: Mapping[str, Any] | None = None,
) -> BlendResult:
    """Blend whichever models are available; ``weights`` are the learned (possibly un-normalised) weights."""
    names = list(dict.fromkeys([*weights, *forecasts]))
    available = [m for m in names if forecasts.get(m) is not None and bool(forecasts[m].notnull().any())]
    missing = [m for m in names if m not in available]
    if not available:
        raise NoForecastAvailable(f"no forecast model available (missing: {missing}); refusing to fabricate a forecast")
    raw = {m: float(weights.get(m, 0.0)) for m in available}
    if any(w < 0 for w in raw.values()):
        raise ValueError("weights must be >= 0")
    total = sum(raw.values())
    reasons = []
    if missing:
        reasons.append(f"removed unavailable models {missing} and renormalised")
    if total <= 0:
        w = {m: 1.0 / len(available) for m in available}
        reasons.append("no positive learned weight among available models: equal weights")
    else:
        w = {m: v / total for m, v in raw.items()}
    if len(available) == 1:
        reasons.append(f"single model available: {available[0]}")
        out = forecasts[available[0]].copy()
    else:
        stack = xr.concat([forecasts[m] for m in available], dim=xr.DataArray(available, dims="model", name="model"))
        out = apply_blend(stack, xr.DataArray([w[m] for m in available], dims="model", coords={"model": available}))
    meta = {
        **(dict(context) if context else {}),
        "available_models": available,
        "missing_models": missing,
        "weights": {m: round(w.get(m, 0.0), 6) for m in names},
        "learned_weights": {m: float(weights.get(m, np.nan)) for m in names},
        "fallback_used": bool(reasons),
        "fallback_reason": "; ".join(reasons),
    }
    return BlendResult(out, meta)
