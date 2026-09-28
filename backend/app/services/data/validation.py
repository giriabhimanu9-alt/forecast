"""Data validation. Every check raises :class:`InvalidDataError` on failure.

The pipeline never continues silently with invalid data: a provider whose
data fails any check is reported as ``INVALID DATA`` and is not saved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from app.models.schema import (
    CANONICAL_VARIABLES,
    InvalidDataError,
    UnitMismatchError,
    validate_forecast_schema,
    validate_observation_schema,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- files


def check_file_present(path: Path) -> None:
    if not path.is_file():
        raise InvalidDataError(f"Missing file: {path}")
    if path.stat().st_size == 0:
        raise InvalidDataError(f"Empty file: {path}")


def check_grib_file(path: Path) -> int:
    """Walk every GRIB message in ``path`` and verify framing; return the message count.

    Detects truncated downloads, HTML error pages saved as data, and
    trailing garbage: each message must start with ``GRIB``, declare a
    length that fits in the file, and end with ``7777``.
    """
    check_file_present(path)
    data = path.read_bytes()
    size = len(data)
    offset = 0
    count = 0
    while offset < size:
        if data[offset : offset + 4] != b"GRIB":
            raise InvalidDataError(
                f"Corrupt GRIB file {path.name}: no 'GRIB' marker at byte {offset}"
            )
        edition = data[offset + 7] if offset + 8 <= size else None
        if edition == 2:
            if offset + 16 > size:
                raise InvalidDataError(f"Corrupt GRIB file {path.name}: truncated header")
            length = int.from_bytes(data[offset + 8 : offset + 16], "big")
        elif edition == 1:
            length = int.from_bytes(data[offset + 4 : offset + 7], "big")
        else:
            raise InvalidDataError(f"Corrupt GRIB file {path.name}: unknown edition {edition}")
        end = offset + length
        if length < 16 or end > size:
            raise InvalidDataError(
                f"Truncated GRIB file {path.name}: message at byte {offset} declares "
                f"{length} bytes but only {size - offset} remain"
            )
        if data[end - 4 : end] != b"7777":
            raise InvalidDataError(
                f"Corrupt GRIB file {path.name}: message at byte {offset} lacks '7777' end marker"
            )
        offset = end
        count += 1
    if count == 0:
        raise InvalidDataError(f"GRIB file {path.name} contains no messages")
    return count


def check_binary_size(path: Path, expected_bytes: int) -> None:
    check_file_present(path)
    size = path.stat().st_size
    if size != expected_bytes:
        raise InvalidDataError(
            f"Corrupt/incomplete binary file {path.name}: {size} bytes, expected {expected_bytes}"
        )


# --------------------------------------------------------------------------- structure


def check_coordinates(ds: xr.Dataset | xr.DataArray) -> None:
    for coord, lo, hi in (("latitude", -90.0, 90.0), ("longitude", -180.0, 360.0)):
        if coord not in ds.coords:
            raise InvalidDataError(f"Missing coordinate {coord!r}")
        values = np.asarray(ds[coord].values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise InvalidDataError(f"Coordinate {coord!r} must be a non-empty 1-D axis")
        if not np.all(np.isfinite(values)):
            raise InvalidDataError(f"Coordinate {coord!r} contains NaN/Inf")
        if values.min() < lo or values.max() > hi:
            raise InvalidDataError(
                f"Coordinate {coord!r} out of range [{lo}, {hi}]: "
                f"{values.min()}..{values.max()}"
            )
        if np.unique(values).size != values.size:
            raise InvalidDataError(f"Coordinate {coord!r} has duplicate values")


def check_no_duplicate_times(ds: xr.Dataset | xr.DataArray, dims: Iterable[str]) -> None:
    for dim in dims:
        if dim not in ds.coords:
            continue
        values = ds[dim].values
        if np.unique(values).size != values.size:
            raise InvalidDataError(f"Duplicate {dim} values: {values}")
        if values.size > 1 and not np.all(values[1:] > values[:-1]):
            raise InvalidDataError(f"{dim} is not strictly increasing")


def check_required_variables(ds: xr.Dataset, required: Iterable[str]) -> None:
    missing = [v for v in required if v not in ds.data_vars]
    if missing:
        raise InvalidDataError(f"Missing forecast variables: {missing}")


def check_units(ds: xr.Dataset) -> None:
    for name, da in ds.data_vars.items():
        spec = CANONICAL_VARIABLES.get(name)
        if spec is None:
            continue
        units = da.attrs.get("units")
        if units != spec.units:
            raise UnitMismatchError(f"{name} has units {units!r}, expected {spec.units!r}")


# --------------------------------------------------------------------------- values


def check_finite(ds: xr.Dataset, allow_nan: bool = False) -> dict[str, int]:
    """Reject Inf always; reject NaN unless ``allow_nan``. Returns NaN counts."""
    nan_counts: dict[str, int] = {}
    for name, da in ds.data_vars.items():
        values = np.asarray(da.values)
        if np.isinf(values).any():
            raise InvalidDataError(f"{name} contains {int(np.isinf(values).sum())} Inf values")
        n_nan = int(np.isnan(values).sum())
        if n_nan and not allow_nan:
            raise InvalidDataError(f"{name} contains {n_nan} NaN values")
        if n_nan == values.size:
            raise InvalidDataError(f"{name} is entirely NaN")
        nan_counts[name] = n_nan
    return nan_counts


def check_physical_ranges(
    ds: xr.Dataset, accumulation_hours: xr.DataArray | float | None = None
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Reject physically impossible values.

    Precipitation below 0 by less than ``clip_tolerance`` (GRIB packing
    noise, e.g. -1e-7 m in ECMWF ``tp``) is clipped to 0 and the count is
    reported; anything more negative, or above the per-accumulation-period
    maximum, is invalid. ``accumulation_hours`` broadcasts against the
    precipitation array (lead_time for forecasts, 24 for daily obs).
    """
    report: dict[str, Any] = {}
    out = ds.copy()
    for name, da in ds.data_vars.items():
        spec = CANONICAL_VARIABLES.get(name)
        if spec is None:
            continue
        vmin = float(da.min(skipna=True))
        if name == "precipitation":
            hours = 24.0 if accumulation_hours is None else accumulation_hours
            too_high = da > spec.valid_max * np.maximum(hours / 24.0, 1.0)
        else:
            too_high = da > spec.valid_max
        n_high = int(too_high.sum())
        if n_high:
            raise InvalidDataError(
                f"{name}: {n_high} impossible values above physical maximum "
                f"(max found {float(da.max(skipna=True)):.2f} {spec.units})"
            )
        if vmin < spec.valid_min - spec.clip_tolerance:
            raise InvalidDataError(
                f"{name}: impossible value {vmin:.4f} {spec.units} below {spec.valid_min}"
            )
        below = da < spec.valid_min
        n_clipped = int(below.sum())
        if n_clipped:
            log.info("%s: clipped %d tiny negative values (>= -%s) to %s", name, n_clipped,
                     spec.clip_tolerance, spec.valid_min)
            out[name] = da.where(~below, spec.valid_min)
            out[name].attrs = da.attrs
        report[name] = {
            "min": round(float(out[name].min(skipna=True)), 4),
            "max": round(float(out[name].max(skipna=True)), 4),
            "mean": round(float(out[name].mean(skipna=True)), 4),
            "clipped_small_negatives": n_clipped,
        }
    return out, report


def check_accumulation_monotonic(da: xr.DataArray, dim: str = "lead_time", tolerance: float = 0.1) -> None:
    """Accumulated-since-init precipitation must never decrease with lead time."""
    if da.sizes.get(dim, 0) < 2:
        return
    decrease = -da.diff(dim).min(skipna=True)
    if float(decrease) > tolerance:
        raise InvalidDataError(
            f"Accumulated precipitation decreases by {float(decrease):.3f} mm along {dim}; "
            "the field is not an accumulation since initialisation"
        )


def check_observation_coverage(ds: xr.Dataset, var: str = "precipitation", min_fraction: float = 0.9) -> dict[str, int]:
    """Every day must have (nearly) the same number of valid grid points.

    IMD grids are NaN outside the Indian land mask, so NaN itself is
    expected; a day with far fewer valid points than the others signals a
    broken or partial file.
    """
    valid = ds[var].notnull().sum(dim=("latitude", "longitude"))
    max_valid = int(valid.max())
    if max_valid == 0:
        raise InvalidDataError(f"{var}: every day is entirely missing")
    bad = valid.where(valid < min_fraction * max_valid, drop=True)
    if bad.size:
        days = [str(t)[:10] for t in bad.time.values]
        raise InvalidDataError(
            f"{var}: days {days} have fewer than {min_fraction:.0%} of the expected "
            f"{max_valid} valid grid points"
        )
    return {"valid_points_per_day": max_valid}


# --------------------------------------------------------------------------- composite


def validate_forecast_dataset(ds: xr.Dataset, variables: Iterable[str]) -> tuple[xr.Dataset, dict[str, Any]]:
    """Run every forecast check; return the (possibly clipped) dataset and a stats report."""
    check_coordinates(ds)
    check_no_duplicate_times(ds, ["time", "lead_time"])
    check_required_variables(ds, variables)
    check_units(ds)
    check_finite(ds, allow_nan=False)
    if "precipitation" in ds:
        check_accumulation_monotonic(ds["precipitation"])
    lead_hours = ds.lead_time / np.timedelta64(1, "h")
    ds, stats = check_physical_ranges(ds, accumulation_hours=lead_hours)
    validate_forecast_schema(ds)
    return ds, stats


def validate_observation_dataset(ds: xr.Dataset) -> tuple[xr.Dataset, dict[str, Any]]:
    check_coordinates(ds)
    check_no_duplicate_times(ds, ["time"])
    check_required_variables(ds, ["precipitation"])
    check_units(ds)
    nan_counts = check_finite(ds, allow_nan=True)
    coverage = check_observation_coverage(ds)
    ds, stats = check_physical_ranges(ds, accumulation_hours=24.0)
    validate_observation_schema(ds)
    stats["precipitation"].update(coverage)
    stats["precipitation"]["nan_cells_outside_imd_land_mask"] = nan_counts["precipitation"]
    return ds, stats
