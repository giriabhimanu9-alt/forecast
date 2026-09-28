"""Common data contract shared by every provider and later pipeline phases.

Normalised forecast ``xarray.Dataset``
--------------------------------------
dims       : model, time, lead_time, latitude, longitude
coords     : model (str), time (datetime64[ns], forecast initialisation, UTC, naive),
             lead_time (timedelta64[ns]), latitude (ascending, degrees_north),
             longitude (ascending, -180..180, degrees_east),
             valid_time (time, lead_time) = time + lead_time
data vars  : one per canonical variable (see ``CANONICAL_VARIABLES``)

``to_variable_array`` stacks the data variables into a single DataArray with
an explicit ``variable`` dimension for code that wants that layout.

Normalised observation ``xarray.Dataset``
-----------------------------------------
dims       : time, latitude, longitude
coords     : time (datetime64[ns]) = END of the 24 h accumulation window (UTC),
             period_start (time) = start of that window,
             source_date (time) = the day label used by the source (e.g. IMD date)
data vars  : precipitation (mm accumulated over (period_start, time])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import xarray as xr

FORECAST_DIMS = ("model", "time", "lead_time", "latitude", "longitude")
OBSERVATION_DIMS = ("time", "latitude", "longitude")


@dataclass(frozen=True)
class VariableSpec:
    name: str
    units: str
    long_name: str
    valid_min: float
    valid_max: float
    # Values slightly below valid_min (e.g. GRIB packing noise) within this
    # tolerance are clipped and counted; anything beyond it is invalid.
    clip_tolerance: float = 0.0


# precipitation valid_max is per 24 h of accumulation; see validation.py.
CANONICAL_VARIABLES: dict[str, VariableSpec] = {
    "precipitation": VariableSpec(
        name="precipitation",
        units="mm",
        long_name="Total precipitation",
        valid_min=0.0,
        valid_max=1900.0,  # ~world-record 24 h rainfall; anything above is impossible
        clip_tolerance=0.05,
    ),
    "temperature_2m": VariableSpec(
        name="temperature_2m",
        units="degC",
        long_name="Air temperature at 2 m",
        valid_min=-90.0,
        valid_max=65.0,
    ),
    "wind_speed_10m": VariableSpec(
        name="wind_speed_10m",
        units="m s-1",
        long_name="Wind speed at 10 m",
        valid_min=0.0,
        valid_max=120.0,
    ),
    # Phase 4: atmospheric fields used only as regime features
    "mslp": VariableSpec(
        name="mslp", units="hPa", long_name="Mean sea level pressure", valid_min=850.0, valid_max=1100.0,
    ),
    "u_850hpa": VariableSpec(
        name="u_850hpa", units="m s-1", long_name="Zonal wind at 850 hPa", valid_min=-150.0, valid_max=150.0,
    ),
    "precipitable_water": VariableSpec(
        name="precipitable_water", units="mm", long_name="Total column precipitable water",
        valid_min=0.0, valid_max=150.0,
    ),
}


class ProviderStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    NOT_CONFIGURED = "NOT CONFIGURED"
    DOWNLOAD_FAILED = "DOWNLOAD FAILED"
    INVALID_DATA = "INVALID DATA"
    PLUGIN_ONLY = "PLUG-IN ONLY"


class DataPipelineError(Exception):
    """Base class for errors that map to a provider status."""

    status: ProviderStatus = ProviderStatus.INVALID_DATA


class NotConfiguredError(DataPipelineError):
    status = ProviderStatus.NOT_CONFIGURED


class DownloadError(DataPipelineError):
    status = ProviderStatus.DOWNLOAD_FAILED


class InvalidDataError(DataPipelineError):
    status = ProviderStatus.INVALID_DATA


class UnitMismatchError(InvalidDataError):
    pass


@dataclass
class ProviderResult:
    """Outcome of one provider run; serialised into the pipeline report."""

    name: str
    kind: str  # "forecast" | "observation" | "ai_forecast"
    status: ProviderStatus
    message: str = ""
    dataset: xr.Dataset | None = field(default=None, repr=False)
    raw_files: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    cache_hits: int = 0
    downloads: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ProviderStatus.AVAILABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status.value,
            "message": self.message,
            "raw_files": self.raw_files,
            "output_files": self.output_files,
            "cache_hits": self.cache_hits,
            "downloads": self.downloads,
            **self.details,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidDataError(f"Schema violation: {message}")


def _check_spatial(ds: xr.Dataset) -> None:
    for coord in ("latitude", "longitude"):
        _require(coord in ds.coords, f"missing coordinate {coord!r}")
        values = ds[coord].values
        _require(values.ndim == 1, f"{coord} must be 1-D")
        _require(bool(np.all(np.diff(values) > 0)), f"{coord} must be strictly ascending")
    _require(
        bool(np.all((ds.longitude >= -180) & (ds.longitude < 180))),
        "longitude must be in [-180, 180)",
    )
    _require(bool(np.all(np.abs(ds.latitude) <= 90)), "latitude must be in [-90, 90]")


def _check_variables(ds: xr.Dataset, dims: tuple[str, ...]) -> None:
    _require(len(ds.data_vars) > 0, "dataset has no data variables")
    for name, da in ds.data_vars.items():
        _require(name in CANONICAL_VARIABLES, f"unexpected variable {name!r}")
        _require(da.dims == dims, f"{name} dims {da.dims} != {dims}")
        expected = CANONICAL_VARIABLES[name].units
        _require(
            da.attrs.get("units") == expected,
            f"{name} units {da.attrs.get('units')!r} != {expected!r}",
        )


def validate_forecast_schema(ds: xr.Dataset) -> None:
    """Raise :class:`InvalidDataError` unless ``ds`` follows the forecast contract."""
    _check_spatial(ds)
    _require("model" in ds.coords and ds.model.dtype.kind in "OU", "model coord must be strings")
    _require(np.issubdtype(ds.time.dtype, np.datetime64), "time must be datetime64")
    _require(np.issubdtype(ds.lead_time.dtype, np.timedelta64), "lead_time must be timedelta64")
    _require("valid_time" in ds.coords, "missing valid_time coordinate")
    _require(
        ds.valid_time.dims == ("time", "lead_time"), "valid_time must have dims (time, lead_time)"
    )
    _check_variables(ds, FORECAST_DIMS)


def validate_observation_schema(ds: xr.Dataset) -> None:
    """Raise :class:`InvalidDataError` unless ``ds`` follows the observation contract."""
    _check_spatial(ds)
    _require(np.issubdtype(ds.time.dtype, np.datetime64), "time must be datetime64")
    for coord in ("period_start", "source_date"):
        _require(coord in ds.coords, f"missing coordinate {coord!r}")
    _check_variables(ds, OBSERVATION_DIMS)


def to_variable_array(ds: xr.Dataset) -> xr.DataArray:
    """Stack data variables into one DataArray with a ``variable`` dimension."""
    return ds.to_dataarray(dim="variable")


def to_standard_forecast(ds: xr.Dataset) -> xr.DataArray:
    """Model-agnostic view of a normalised forecast for blending engines and future AI providers.

    dims: model_name, initialization_time, lead_time, latitude, longitude, variable; name "forecast".
    Any ForecastProvider (NWP or AI) whose output passes :func:`validate_forecast_schema`
    can be viewed this way, so the blending code never depends on the source.
    """
    validate_forecast_schema(ds)
    da = ds.to_dataarray(dim="variable").rename(model="model_name", time="initialization_time")
    da.name = "forecast"
    return da.transpose("model_name", "initialization_time", "lead_time", "latitude", "longitude", "variable")
