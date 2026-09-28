"""Normalisation of provider data onto the common data contract.

Steps applied to every source:
  * coordinate names  -> ``latitude`` / ``longitude``
  * longitude         -> [-180, 180), ascending; latitude ascending
  * timestamps        -> naive UTC ``datetime64[ns]``; lead times ``timedelta64[ns]``
  * missing values    -> source fill values (e.g. IMD -999) become NaN
  * units             -> canonical units (mm, degC, m s-1) via an explicit table;
                         unknown units raise :class:`UnitMismatchError`
  * grid              -> cropped and put on the common target grid
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from app.config import TargetGrid
from app.models.schema import CANONICAL_VARIABLES, FORECAST_DIMS, InvalidDataError, UnitMismatchError

log = logging.getLogger(__name__)

_COORD_ALIASES = {
    "lat": "latitude",
    "lats": "latitude",
    "y": "latitude",
    "lon": "longitude",
    "lons": "longitude",
    "long": "longitude",
    "x": "longitude",
}

# canonical quantity -> {source unit string: (scale, offset)};  canonical = raw * scale + offset
UNIT_CONVERSIONS: dict[str, dict[str, tuple[float, float]]] = {
    "precipitation": {
        "mm": (1.0, 0.0),
        "kg m**-2": (1.0, 0.0),  # 1 kg of water per m^2 == 1 mm depth
        "kg m-2": (1.0, 0.0),
        "kg/m^2": (1.0, 0.0),
        "m": (1000.0, 0.0),
    },
    "temperature_2m": {
        "K": (1.0, -273.15),
        "degC": (1.0, 0.0),
        "C": (1.0, 0.0),
        "°C": (1.0, 0.0),
    },
    "mslp": {
        "Pa": (0.01, 0.0),
        "hPa": (1.0, 0.0),
    },
    "precipitable_water": {
        "kg m**-2": (1.0, 0.0),  # 1 kg of water per m^2 == 1 mm depth
        "kg m-2": (1.0, 0.0),
        "mm": (1.0, 0.0),
    },
    "wind": {
        "m s-1": (1.0, 0.0),
        "m s**-1": (1.0, 0.0),
        "m/s": (1.0, 0.0),
        "km h-1": (1.0 / 3.6, 0.0),
        "knots": (0.514444, 0.0),
        "kt": (0.514444, 0.0),
    },
}


# --------------------------------------------------------------------------- coordinates


def standardize_coords(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Rename lat/lon aliases, wrap longitude to [-180, 180) and sort both axes ascending."""
    renames = {k: v for k, v in _COORD_ALIASES.items() if k in obj.dims or k in obj.coords}
    if renames:
        obj = obj.rename(renames)
    for coord in ("latitude", "longitude"):
        if coord not in obj.coords:
            raise InvalidDataError(f"Dataset has no {coord} coordinate (coords: {list(obj.coords)})")
    lon = obj["longitude"].astype("float64")
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    obj = obj.assign_coords(longitude=np.round(wrapped, 6), latitude=np.round(obj["latitude"].astype("float64"), 6))
    if np.unique(obj.longitude.values).size != obj.longitude.size:
        raise InvalidDataError("Duplicate longitudes after wrapping to [-180, 180)")
    if np.unique(obj.latitude.values).size != obj.latitude.size:
        raise InvalidDataError("Duplicate latitudes in source data")
    return obj.sortby("latitude").sortby("longitude")


def crop_to_grid(obj: xr.Dataset | xr.DataArray, grid: TargetGrid, margin: float = 1.0) -> xr.Dataset | xr.DataArray:
    """Crop to the target region plus ``margin`` degrees (for interpolation at the edges)."""
    cropped = obj.sel(
        latitude=slice(grid.lat_min - margin, grid.lat_max + margin),
        longitude=slice(grid.lon_min - margin, grid.lon_max + margin),
    )
    if cropped.sizes["latitude"] == 0 or cropped.sizes["longitude"] == 0:
        raise InvalidDataError(
            f"Source data does not overlap the target region {grid.as_dict()}"
        )
    return cropped


def _axis_contains(source: np.ndarray, target: np.ndarray, tol: float = 1e-6) -> bool:
    """True if every target point inside the source's extent is a source grid point."""
    target = target[(target >= source.min() - tol) & (target <= source.max() + tol)]
    if target.size == 0:
        return False
    idx = np.searchsorted(source, target)
    idx = np.clip(idx, 0, source.size - 1)
    near = np.minimum(np.abs(source[idx] - target), np.abs(source[np.maximum(idx - 1, 0)] - target))
    return bool(np.all(near < tol))


def regrid_to_target(obj: xr.Dataset | xr.DataArray, grid: TargetGrid) -> xr.Dataset | xr.DataArray:
    """Put data on the target grid.

    If every target point inside the source domain is also a source grid
    point (e.g. 0.25° sources on a 0.25° target) values are copied exactly
    and no interpolation happens — this also avoids bilinear interpolation
    smearing NaN (e.g. IMD's ocean mask) onto neighbouring valid cells.
    Otherwise bilinear interpolation is used. Target points outside the
    source domain become NaN in both cases (never extrapolated).
    """
    lats, lons = grid.latitudes, grid.longitudes
    src_lat = obj.latitude.values
    src_lon = obj.longitude.values
    if _axis_contains(src_lat, lats) and _axis_contains(src_lon, lons):
        out = obj.reindex(latitude=lats, longitude=lons, method="nearest", tolerance=1e-6)
        method = "exact-subset"
    else:
        out = obj.interp(latitude=lats, longitude=lons, method="linear")
        method = "bilinear"
    out = out.assign_coords(latitude=lats, longitude=lons)
    out.latitude.attrs.update(standard_name="latitude", units="degrees_north")
    out.longitude.attrs.update(standard_name="longitude", units="degrees_east")
    out.attrs["regrid_method"] = method
    return out


# --------------------------------------------------------------------------- time


def to_utc_naive(value: datetime | np.datetime64 | str | pd.Timestamp) -> np.datetime64:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.to_datetime64(), "ns")


def lead_hours_to_timedelta(hours: Iterable[int]) -> np.ndarray:
    return np.array([np.timedelta64(int(h), "h") for h in hours], dtype="timedelta64[ns]")


# --------------------------------------------------------------------------- values


def mask_fill_values(da: xr.DataArray, fill_values: Iterable[float]) -> xr.DataArray:
    mask = xr.zeros_like(da, dtype=bool)
    for fv in fill_values:
        mask = mask | np.isclose(da, fv)
    return da.where(~mask)


def convert_units(da: xr.DataArray, quantity: str, target_units: str) -> xr.DataArray:
    """Convert ``da`` (with a ``units`` or ``GRIB_units`` attr) to ``target_units``."""
    units = da.attrs.get("units") or da.attrs.get("GRIB_units")
    if units is None:
        raise UnitMismatchError(f"{da.name or quantity}: source data has no units attribute")
    table = UNIT_CONVERSIONS[quantity]
    if units not in table:
        raise UnitMismatchError(
            f"{da.name or quantity}: unsupported source units {units!r} "
            f"(known: {sorted(table)})"
        )
    scale, offset = table[units]
    out = da.astype("float32") * np.float32(scale) + np.float32(offset)
    out.attrs = {"units": target_units, "source_units": units}
    return out


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    u_ms = convert_units(u, "wind", "m s-1")
    v_ms = convert_units(v, "wind", "m s-1")
    speed = np.hypot(u_ms, v_ms).astype("float32")
    speed.attrs = {"units": "m s-1", "source_units": u_ms.attrs["source_units"], "derived_from": "u10, v10"}
    return speed


# --------------------------------------------------------------------------- datasets


def normalize_forecast(
    fields: Mapping[str, xr.DataArray],
    model: str,
    init_time: datetime | np.datetime64,
    grid: TargetGrid,
    variables: Iterable[str],
    source_attrs: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Build a contract-compliant forecast dataset from provider fields.

    ``fields`` keys: ``precipitation`` (accumulated since initialisation),
    ``temperature_2m``, ``u_10m``, ``v_10m``; each DataArray has dims
    (lead_time, latitude, longitude) in the provider's native units/grid.
    """
    out: dict[str, xr.DataArray] = {}
    method = "none"
    for var in variables:
        if var == "precipitation":
            da = convert_units(_field(fields, "precipitation"), "precipitation", "mm")
            da.attrs["accumulation"] = "since forecast initialisation (time) up to valid_time"
            da.attrs["cell_methods"] = "lead_time: sum"
        elif var == "temperature_2m":
            da = convert_units(_field(fields, "temperature_2m"), "temperature_2m", "degC")
        elif var == "wind_speed_10m":
            da = wind_speed(_field(fields, "u_10m"), _field(fields, "v_10m"))
        elif var == "mslp":
            da = convert_units(_field(fields, "mslp"), "mslp", "hPa")
        elif var == "u_850hpa":
            da = convert_units(_field(fields, "u_850hpa"), "wind", "m s-1")
        elif var == "precipitable_water":
            da = convert_units(_field(fields, "precipitable_water"), "precipitable_water", "mm")
        else:
            raise InvalidDataError(f"Unsupported variable {var!r}")
        da = regrid_to_target(crop_to_grid(standardize_coords(da), grid), grid)
        method = da.attrs.pop("regrid_method")
        da.attrs["long_name"] = CANONICAL_VARIABLES[var].long_name
        out[var] = da

    ds = xr.Dataset(out)
    init = to_utc_naive(init_time)
    ds = ds.expand_dims(model=[model], time=[init]).transpose(*FORECAST_DIMS)
    ds = ds.assign_coords(valid_time=ds.time + ds.lead_time)
    ds = ds.drop_vars([c for c in ds.coords if c not in (*FORECAST_DIMS, "valid_time")])
    ds.attrs.update(
        {
            "title": f"{model} forecast normalised to the common {grid.name} grid",
            "Conventions": "CF-1.8",
            "grid_resolution_deg": grid.resolution,
            "regrid_method": method,
            "time_convention": "time = initialisation (UTC); valid_time = time + lead_time",
            **{k: str(v) for k, v in (source_attrs or {}).items()},
        }
    )
    return ds


def _field(fields: Mapping[str, xr.DataArray], key: str) -> xr.DataArray:
    if key not in fields:
        raise InvalidDataError(f"Provider did not supply required field {key!r}")
    return fields[key]


def normalize_observations(
    rain: xr.DataArray,
    grid: TargetGrid,
    source: str,
    period_end_offset: pd.Timedelta,
    fill_values: Iterable[float] = (),
    source_attrs: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Normalise a daily rainfall DataArray (time, lat, lon) to the observation contract.

    Sources label days differently, so the canonical ``time`` is the END of
    each 24 h accumulation window (UTC): ``time = day_label + period_end_offset``.
    The source's own label is kept in the ``source_date`` coordinate.
    """
    da = standardize_coords(rain)
    da = mask_fill_values(da, fill_values)
    da = convert_units(da, "precipitation", "mm")
    da = regrid_to_target(crop_to_grid(da, grid), grid)
    method = da.attrs.pop("regrid_method")
    da.attrs.update(
        long_name=CANONICAL_VARIABLES["precipitation"].long_name,
        accumulation="24 h over (period_start, time]",
    )
    labels = pd.DatetimeIndex(da.time.values).normalize()
    end = labels + period_end_offset
    da = da.assign_coords(time=np.asarray(end, dtype="datetime64[ns]"))
    ds = xr.Dataset({"precipitation": da.transpose("time", "latitude", "longitude")})
    ds = ds.assign_coords(
        period_start=("time", np.asarray(end - pd.Timedelta(hours=24), dtype="datetime64[ns]")),
        source_date=("time", np.asarray(labels.strftime("%Y-%m-%d"), dtype=str)),
    )
    keep = ("time", "latitude", "longitude", "period_start", "source_date")
    ds = ds.drop_vars([c for c in ds.coords if c not in keep])
    ds.attrs.update(
        {
            "title": f"{source} observed rainfall normalised to the common {grid.name} grid",
            "Conventions": "CF-1.8",
            "source": source,
            "grid_resolution_deg": grid.resolution,
            "regrid_method": method,
            "time_convention": "time = end of the 24 h accumulation window (UTC)",
            **{k: str(v) for k, v in (source_attrs or {}).items()},
        }
    )
    return ds
