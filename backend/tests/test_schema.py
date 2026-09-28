from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from app.config import DEFAULT_VARIABLES, TargetGrid
from app.models.schema import (
    CANONICAL_VARIABLES,
    FORECAST_DIMS,
    InvalidDataError,
    ProviderStatus,
    to_variable_array,
    validate_forecast_schema,
)
from app.services.data.normalization import normalize_forecast

GRID = TargetGrid(lat_min=10, lat_max=12, lon_min=70, lon_max=72, resolution=0.5)
INIT = datetime(2026, 9, 27, 12)


def _fields():
    lats = np.arange(14.0, 7.99, -0.5)  # descending, like GRIB
    lons = np.arange(66.0, 76.01, 0.5)
    lead = np.array([np.timedelta64(6, "h"), np.timedelta64(12, "h")], dtype="timedelta64[ns]")

    def field(value, units):
        return xr.DataArray(
            np.full((2, lats.size, lons.size), value, dtype="float32") * np.array([1, 2]).reshape(2, 1, 1),
            dims=("lead_time", "latitude", "longitude"),
            coords={"lead_time": lead, "latitude": lats, "longitude": lons},
            attrs={"units": units},
        )

    return {
        "precipitation": field(0.002, "m"),
        "temperature_2m": field(150.0, "K"),
        "u_10m": field(3.0, "m s**-1"),
        "v_10m": field(4.0, "m s**-1"),
    }


def test_normalize_forecast_follows_contract():
    ds = normalize_forecast(_fields(), "TEST", INIT, GRID, list(DEFAULT_VARIABLES))
    validate_forecast_schema(ds)
    for name in DEFAULT_VARIABLES:
        assert ds[name].dims == FORECAST_DIMS
        assert ds[name].attrs["units"] == CANONICAL_VARIABLES[name].units
    assert ds.sizes["latitude"] == 5 and ds.sizes["longitude"] == 5
    assert ds.model.values.tolist() == ["TEST"]
    assert ds.valid_time.values[0, 1] == np.datetime64("2026-09-28T00:00", "ns")
    assert float(ds.precipitation.isel(lead_time=1).max()) == pytest.approx(4.0)  # 0.004 m -> 4 mm
    assert float(ds.temperature_2m.isel(lead_time=0).max()) == pytest.approx(150 - 273.15, abs=1e-3)
    assert float(ds.wind_speed_10m.isel(lead_time=0).max()) == pytest.approx(5.0)
    assert ds.attrs["regrid_method"] == "exact-subset"


def test_to_variable_array_has_variable_dim():
    ds = normalize_forecast(_fields(), "TEST", INIT, GRID, ["precipitation", "wind_speed_10m"])
    da = to_variable_array(ds)
    assert da.dims[0] == "variable"
    assert set(FORECAST_DIMS) <= set(da.dims)
    assert list(da["variable"].values) == ["precipitation", "wind_speed_10m"]


def test_models_concatenate_along_model_dim():
    a = normalize_forecast(_fields(), "A", INIT, GRID, ["precipitation"])
    b = normalize_forecast(_fields(), "B", INIT, GRID, ["precipitation"])
    both = xr.concat([a, b], dim="model")
    validate_forecast_schema(both)
    assert both.sizes["model"] == 2


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda ds: ds.transpose("time", "model", "lead_time", "latitude", "longitude"), "dims"),
        (lambda ds: ds.assign(precipitation=ds.precipitation.assign_attrs(units="m")), "units"),
        (lambda ds: ds.drop_vars("valid_time"), "valid_time"),
        (lambda ds: ds.isel(latitude=slice(None, None, -1)), "ascending"),
        (lambda ds: ds.rename(precipitation="rain"), "unexpected variable"),
    ],
)
def test_schema_violations_detected(mutate, message):
    ds = normalize_forecast(_fields(), "TEST", INIT, GRID, ["precipitation"])
    with pytest.raises(InvalidDataError, match=message):
        validate_forecast_schema(mutate(ds))


def test_status_labels():
    assert [s.value for s in ProviderStatus] == [
        "AVAILABLE", "NOT CONFIGURED", "DOWNLOAD FAILED", "INVALID DATA", "PLUG-IN ONLY",
    ]
