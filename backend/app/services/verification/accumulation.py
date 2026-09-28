"""Forecast verification windows and 24-hour accumulations.

GFS ``APCP 0-N`` and ECMWF ``tp`` are accumulated since initialisation
(Phase 1 stores them as ``precipitation`` in mm, dims ``..., lead_time, ...``).
A 24 h total over the window (init + L0, init + L0 + 24 h] is therefore

    rain_24h = acc(L0 + 24 h) - acc(L0)

with ``acc(0) = 0`` by definition (nothing has accumulated at the analysis
time), so a window that starts at the initialisation needs no step-0 field.

IMD daily windows run 03 UTC -> 03 UTC (08:30 IST). For a cycle at hour H the
first window that starts at or after initialisation begins at lead
``L0 = (3 - H) mod 24`` hours (00Z -> 3 h, 12Z -> 15 h, 03Z -> 0 h); lead-day k
uses ``L0 + 24 (k-1)`` .. ``L0 + 24 k``. Never is a 48/72 h cumulative total
compared with a 24 h observation: every window is an explicit difference of
two accumulations exactly 24 h apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import xarray as xr

from app.models.schema import InvalidDataError

WINDOW_HOURS = 24
# Same tolerance as Phase 1 validation (check_accumulation_monotonic): GRIB packing noise can make an
# accumulated field decrease by a few hundredths of a mm; anything larger is invalid data.
PACKING_NOISE_TOLERANCE_MM = 0.1
IMD_WINDOW_END_HOUR_UTC = 3  # 08:30 IST


@dataclass(frozen=True)
class ForecastWindow:
    init_time: datetime
    lead_day: int
    start_lead_h: int
    end_lead_h: int

    @property
    def start(self) -> datetime:
        return self.init_time + timedelta(hours=self.start_lead_h)

    @property
    def end(self) -> datetime:
        return self.init_time + timedelta(hours=self.end_lead_h)


def verification_windows(
    init_time: datetime, lead_days: Iterable[int], window_end_hour_utc: int = IMD_WINDOW_END_HOUR_UTC
) -> list[ForecastWindow]:
    """24 h windows aligned to the observation day boundary, starting at/after ``init_time``."""
    first = (window_end_hour_utc - init_time.hour) % 24
    out = []
    for k in lead_days:
        if k < 1:
            raise ValueError(f"lead_day must be >= 1, got {k}")
        start = first + WINDOW_HOURS * (k - 1)
        out.append(ForecastWindow(init_time, k, start, start + WINDOW_HOURS))
    return out


def required_lead_hours(windows: Iterable[ForecastWindow]) -> list[int]:
    """Accumulation leads that must be downloaded (lead 0 is implicit zero)."""
    return sorted({h for w in windows for h in (w.start_lead_h, w.end_lead_h) if h > 0})


def forecast_to_24h_accumulation(
    precip: xr.DataArray, start_lead_h: int, end_lead_h: int,
    clip_tolerance: float = PACKING_NOISE_TOLERANCE_MM,
) -> xr.DataArray:
    """24 h (or any window) total from an accumulated-since-init field.

    ``precip`` must have a ``lead_time`` dimension and ``units == "mm"`` and
    ``accumulation`` metadata saying it is accumulated since initialisation.
    Differences slightly below zero (GRIB packing noise, >= -clip_tolerance)
    are clipped to 0; anything more negative raises.
    """
    if end_lead_h - start_lead_h != WINDOW_HOURS:
        raise InvalidDataError(
            f"Window {start_lead_h}-{end_lead_h} h is not {WINDOW_HOURS} h; refusing to compare "
            "a non-24 h accumulation with daily observations"
        )
    if precip.attrs.get("units") != "mm":
        raise InvalidDataError(f"precipitation units {precip.attrs.get('units')!r}, expected 'mm'")
    if "since forecast initialisation" not in str(precip.attrs.get("accumulation", "")):
        raise InvalidDataError("precipitation is not marked as accumulated since initialisation")

    def at(h: int) -> xr.DataArray:
        if h == 0:
            return xr.zeros_like(precip.isel(lead_time=0, drop=True))
        target = np.timedelta64(h, "h")
        if target not in precip.lead_time.values:
            raise InvalidDataError(
                f"lead {h} h not present (have {list(precip.lead_time.values / np.timedelta64(1, 'h'))})"
            )
        return precip.sel(lead_time=target, drop=True)

    diff = at(end_lead_h) - at(start_lead_h)
    most_negative = float(diff.min(skipna=True)) if diff.notnull().any() else 0.0
    if most_negative < -clip_tolerance:
        raise InvalidDataError(
            f"accumulation decreases by {-most_negative:.3f} mm between {start_lead_h} and {end_lead_h} h"
        )
    n_clipped = int((diff < 0).sum())
    diff = diff.clip(min=0.0)
    diff.attrs = {
        "units": "mm",
        "long_name": "24 h precipitation total",
        "accumulation": f"acc({end_lead_h} h) - acc({start_lead_h} h)",
        "negative_differences_clipped": n_clipped,
    }
    return diff

