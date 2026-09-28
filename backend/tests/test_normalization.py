from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from app.config import TargetGrid
from app.models.schema import InvalidDataError, UnitMismatchError
from app.services.data import normalization as norm

GRID = TargetGrid(lat_min=10, lat_max=12, lon_min=70, lon_max=72, resolution=0.5)


def _da(lats, lons, values=None, units="mm", names=("latitude", "longitude")):
    values = np.arange(len(lats) * len(lons), dtype="float32").reshape(len(lats), len(lons)) if values is None else values
    return xr.DataArray(values, dims=names, coords={names[0]: lats, names[1]: lons}, attrs={"units": units})


# --------------------------------------------------------------------------- coordinates

def test_standardize_renames_aliases_and_sorts_latitude():
    da = _da([12.0, 11.0, 10.0], [70.0, 71.0], names=("lat", "lon"))
    out = norm.standardize_coords(da)
    assert out.dims == ("latitude", "longitude")
    assert list(out.latitude.values) == [10.0, 11.0, 12.0]
    # the value that was at lat=12 must still be at lat=12
    assert float(out.sel(latitude=12, longitude=70)) == float(da.sel(lat=12, lon=70))


def test_standardize_wraps_0_360_longitudes():
    da = _da([10.0], [0.0, 90.0, 180.0, 270.0, 359.75])
    out = norm.standardize_coords(da)
    assert list(out.longitude.values) == [-180.0, -90.0, -0.25, 0.0, 90.0]
    assert float(out.sel(latitude=10, longitude=-90)) == float(da.sel(latitude=10, longitude=270))


def test_standardize_rejects_duplicate_longitudes_after_wrap():
    with pytest.raises(InvalidDataError, match="Duplicate longitudes"):
        norm.standardize_coords(_da([10.0], [-90.0, 270.0]))


def test_standardize_requires_spatial_coords():
    with pytest.raises(InvalidDataError, match="latitude"):
        norm.standardize_coords(xr.DataArray(np.zeros((2, 2)), dims=("a", "b")))


def test_crop_raises_when_no_overlap():
    da = norm.standardize_coords(_da([50.0, 51.0], [0.0, 1.0]))
    with pytest.raises(InvalidDataError, match="does not overlap"):
        norm.crop_to_grid(da, GRID, margin=0)


def test_regrid_exact_subset_copies_values():
    lats = np.arange(8.0, 14.01, 0.25)
    lons = np.arange(68.0, 74.01, 0.25)
    da = _da(lats, lons)
    out = norm.regrid_to_target(da, GRID)
    assert out.attrs["regrid_method"] == "exact-subset"
    assert out.shape == GRID.shape
    assert float(out.sel(latitude=11.5, longitude=71.0)) == float(da.sel(latitude=11.5, longitude=71.0))


def test_regrid_bilinear_on_coarser_source():
    lats = np.array([9.0, 13.0])
    lons = np.array([69.0, 73.0])
    da = _da(lats, lons, values=np.array([[0.0, 4.0], [8.0, 12.0]], dtype="float32"))
    out = norm.regrid_to_target(da, GRID)
    assert out.attrs["regrid_method"] == "bilinear"
    # centre of the 4 points -> mean
    assert float(out.sel(latitude=11.0, longitude=71.0)) == pytest.approx(6.0)


def test_regrid_aligned_partial_domain_does_not_smear_nan():
    # Source (like IMD) starts north of the target's southern edge and has a NaN mask.
    lats = np.arange(11.0, 12.01, 0.5)
    lons = np.arange(70.0, 72.01, 0.5)
    values = np.ones((lats.size, lons.size), dtype="float32")
    values[0, 0] = np.nan
    out = norm.regrid_to_target(_da(lats, lons, values), GRID)
    assert out.attrs["regrid_method"] == "exact-subset"
    assert np.isnan(out.sel(latitude=10.0)).all()  # outside source domain: NaN, not extrapolated
    assert np.isnan(float(out.sel(latitude=11.0, longitude=70.0)))
    assert float(out.sel(latitude=11.0, longitude=70.5)) == 1.0  # neighbour of NaN stays valid
    assert int(out.notnull().sum()) == values.size - 1


# --------------------------------------------------------------------------- units

@pytest.mark.parametrize(
    "quantity, units, raw, target_units, expected",
    [
        ("precipitation", "m", 0.012, "mm", 12.0),
        ("precipitation", "kg m**-2", 7.5, "mm", 7.5),
        ("precipitation", "mm", 3.0, "mm", 3.0),
        ("temperature_2m", "K", 300.15, "degC", 27.0),
        ("temperature_2m", "degC", 20.0, "degC", 20.0),
        ("wind", "knots", 10.0, "m s-1", 5.14444),
        ("wind", "km h-1", 36.0, "m s-1", 10.0),
    ],
)
def test_convert_units(quantity, units, raw, target_units, expected):
    da = xr.DataArray(np.array([raw], dtype="float64"), dims="x", attrs={"units": units})
    out = norm.convert_units(da, quantity, target_units)
    assert float(out[0]) == pytest.approx(expected, rel=1e-5)
    assert out.attrs == {"units": target_units, "source_units": units}
    assert out.dtype == np.float32


def test_convert_units_uses_grib_units_attr():
    da = xr.DataArray([1.0], dims="x", attrs={"GRIB_units": "m"})
    assert float(norm.convert_units(da, "precipitation", "mm")[0]) == pytest.approx(1000.0)


def test_unknown_or_missing_units_raise():
    with pytest.raises(UnitMismatchError, match="unsupported source units 'inch'"):
        norm.convert_units(xr.DataArray([1.0], dims="x", attrs={"units": "inch"}), "precipitation", "mm")
    with pytest.raises(UnitMismatchError, match="no units"):
        norm.convert_units(xr.DataArray([1.0], dims="x"), "precipitation", "mm")
    # a rate is not an accumulation
    with pytest.raises(UnitMismatchError):
        norm.convert_units(xr.DataArray([1.0], dims="x", attrs={"units": "mm/day"}), "precipitation", "mm")


def test_wind_speed_from_components():
    u = xr.DataArray([3.0, -6.0], dims="x", attrs={"units": "m s**-1"})
    v = xr.DataArray([4.0, 8.0], dims="x", attrs={"units": "m s**-1"})
    speed = norm.wind_speed(u, v)
    assert list(speed.values) == [5.0, 10.0]
    assert speed.attrs["units"] == "m s-1"


def test_mask_fill_values():
    da = xr.DataArray([1.0, -999.0, 2.0], dims="x")
    out = norm.mask_fill_values(da, [-999.0])
    assert np.isnan(out[1]) and float(out[0]) == 1.0


# --------------------------------------------------------------------------- time

def test_to_utc_naive_converts_timezones():
    ist = datetime(2026, 9, 27, 17, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert norm.to_utc_naive(ist) == np.datetime64("2026-09-27T12:00:00", "ns")
    assert norm.to_utc_naive("2026-09-27T12:00") == np.datetime64("2026-09-27T12:00:00", "ns")


def test_lead_hours_to_timedelta():
    out = norm.lead_hours_to_timedelta([6, 12])
    assert out.dtype == np.dtype("timedelta64[ns]")
    assert list(out / np.timedelta64(1, "h")) == [6, 12]


def test_normalize_observations_time_is_window_end():
    days = pd.date_range("2026-09-19", periods=2)
    lats = np.arange(10.0, 12.01, 0.5)
    lons = np.arange(70.0, 72.01, 0.5)
    rain = xr.DataArray(
        np.full((2, lats.size, lons.size), 5.0, dtype="float32"),
        dims=("time", "lat", "lon"), coords={"time": days, "lat": lats, "lon": lons}, attrs={"units": "mm"},
    )
    ds = norm.normalize_observations(rain, GRID, "TEST", period_end_offset=pd.Timedelta(hours=3), fill_values=(-999.0,))
    assert list(ds.time.values) == [np.datetime64("2026-09-19T03:00", "ns"), np.datetime64("2026-09-20T03:00", "ns")]
    assert ds.period_start.values[0] == np.datetime64("2026-09-18T03:00", "ns")
    assert list(ds.source_date.values) == ["2026-09-19", "2026-09-20"]
    assert ds.precipitation.dims == ("time", "latitude", "longitude")
