from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from app.models.schema import InvalidDataError, UnitMismatchError
from app.services.data import validation as v
from tests.synthetic import grib_message, synthetic_field

INIT = datetime(2026, 9, 27, 12)


def _forecast(precip=None, lead_hours=(6, 12), **extra_vars):
    lats, lons = np.array([10.0, 10.5]), np.array([70.0, 70.5])
    if precip is None:
        precip = np.stack([np.full((2, 2), 1.0 * h) for h in lead_hours])
    coords = {
        "model": ["TEST"],
        "time": [np.datetime64(INIT, "ns")],
        "lead_time": np.array([np.timedelta64(h, "h") for h in lead_hours], dtype="timedelta64[ns]"),
        "latitude": lats,
        "longitude": lons,
    }
    dims = ("model", "time", "lead_time", "latitude", "longitude")
    data = {"precipitation": (dims, np.asarray(precip, dtype="float32")[None, None], {"units": "mm"})}
    for name, (values, units) in extra_vars.items():
        data[name] = (dims, np.asarray(values, dtype="float32")[None, None], {"units": units})
    ds = xr.Dataset(data, coords=coords)
    return ds.assign_coords(valid_time=ds.time + ds.lead_time)


# --------------------------------------------------------------------------- files

def test_valid_grib_file_passes(tmp_path):
    path = tmp_path / "ok.grib2"
    path.write_bytes(grib_message("kwbc", "2t", INIT, 6, synthetic_field("temp_k", 6)) * 2)
    assert v.check_grib_file(path) == 2


def test_missing_and_empty_files(tmp_path):
    with pytest.raises(InvalidDataError, match="Missing file"):
        v.check_grib_file(tmp_path / "nope.grib2")
    (tmp_path / "empty.grib2").write_bytes(b"")
    with pytest.raises(InvalidDataError, match="Empty file"):
        v.check_grib_file(tmp_path / "empty.grib2")


def test_truncated_grib_detected(tmp_path):
    msg = grib_message("kwbc", "2t", INIT, 6, synthetic_field("temp_k", 6))
    path = tmp_path / "trunc.grib2"
    path.write_bytes(msg[:-100])
    with pytest.raises(InvalidDataError, match="Truncated"):
        v.check_grib_file(path)


def test_html_error_page_saved_as_grib_detected(tmp_path):
    path = tmp_path / "html.grib2"
    path.write_bytes(b"<html><body>503 Service Unavailable</body></html>")
    with pytest.raises(InvalidDataError, match="no 'GRIB' marker"):
        v.check_grib_file(path)


def test_trailing_garbage_detected(tmp_path):
    path = tmp_path / "junk.grib2"
    path.write_bytes(grib_message("kwbc", "2t", INIT, 6, synthetic_field("temp_k", 6)) + b"garbage")
    with pytest.raises(InvalidDataError):
        v.check_grib_file(path)


def test_missing_end_marker_detected(tmp_path):
    msg = bytearray(grib_message("kwbc", "2t", INIT, 6, synthetic_field("temp_k", 6)))
    msg[-4:] = b"XXXX"
    path = tmp_path / "bad.grib2"
    path.write_bytes(bytes(msg))
    with pytest.raises(InvalidDataError, match="7777"):
        v.check_grib_file(path)


def test_binary_size_check(tmp_path):
    path = tmp_path / "x.grd"
    path.write_bytes(b"\0" * 10)
    v.check_binary_size(path, 10)
    with pytest.raises(InvalidDataError, match="expected 12"):
        v.check_binary_size(path, 12)


# --------------------------------------------------------------------------- structure

def test_invalid_coordinates():
    ds = _forecast()
    with pytest.raises(InvalidDataError, match="out of range"):
        v.check_coordinates(ds.assign_coords(latitude=[10.0, 95.0]))
    with pytest.raises(InvalidDataError, match="duplicate"):
        v.check_coordinates(ds.assign_coords(longitude=[70.0, 70.0]))
    with pytest.raises(InvalidDataError, match="NaN"):
        v.check_coordinates(ds.assign_coords(latitude=[10.0, np.nan]))
    with pytest.raises(InvalidDataError, match="Missing coordinate"):
        v.check_coordinates(ds.rename(latitude="lat"))


def test_duplicate_timestamps():
    ds = _forecast(lead_hours=(6, 6))
    with pytest.raises(InvalidDataError, match="Duplicate lead_time"):
        v.check_no_duplicate_times(ds, ["lead_time"])
    ds = _forecast(lead_hours=(12, 6), precip=np.stack([np.full((2, 2), 2.0), np.full((2, 2), 1.0)]))
    with pytest.raises(InvalidDataError, match="not strictly increasing"):
        v.check_no_duplicate_times(ds, ["lead_time"])


def test_missing_forecast_variables():
    with pytest.raises(InvalidDataError, match="temperature_2m"):
        v.check_required_variables(_forecast(), ["precipitation", "temperature_2m"])


def test_unit_mismatch():
    ds = _forecast()
    ds["precipitation"].attrs["units"] = "m"
    with pytest.raises(UnitMismatchError):
        v.check_units(ds)


# --------------------------------------------------------------------------- values

def test_nan_and_inf_rejected():
    ds = _forecast()
    ds["precipitation"][0, 0, 0, 0, 0] = np.nan
    with pytest.raises(InvalidDataError, match="NaN"):
        v.check_finite(ds)
    assert v.check_finite(ds, allow_nan=True)["precipitation"] == 1
    ds["precipitation"][0, 0, 0, 0, 1] = np.inf
    with pytest.raises(InvalidDataError, match="Inf"):
        v.check_finite(ds, allow_nan=True)


def test_all_nan_rejected_even_when_nan_allowed():
    ds = _forecast()
    ds["precipitation"][:] = np.nan
    with pytest.raises(InvalidDataError, match="entirely NaN"):
        v.check_finite(ds, allow_nan=True)


def test_negative_rainfall_rejected():
    precip = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 2.0)])
    precip[1, 0, 0] = -5.0
    with pytest.raises(InvalidDataError, match="impossible value"):
        v.check_physical_ranges(_forecast(precip))


def test_tiny_negative_rainfall_clipped_and_counted():
    precip = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 2.0)])
    precip[0, 0, 0] = -0.001
    out, stats = v.check_physical_ranges(_forecast(precip))
    assert float(out.precipitation.min()) == 0.0
    assert stats["precipitation"]["clipped_small_negatives"] == 1


def test_impossible_rainfall_rejected_with_lead_scaled_limit():
    ds = _forecast(precip=np.stack([np.full((2, 2), 2500.0), np.full((2, 2), 2600.0)]), lead_hours=(6, 72))
    hours = ds.lead_time / np.timedelta64(1, "h")
    # 2500 mm in 6 h is impossible; 2600 mm in 72 h is below 3 * 1900 mm
    with pytest.raises(InvalidDataError, match="1 impossible|4 impossible"):
        v.check_physical_ranges(ds, accumulation_hours=hours)


def test_temperature_in_kelvin_labelled_celsius_is_caught():
    t = np.full((2, 2, 2), 300.0)
    ds = _forecast(temperature_2m=(t, "degC"))
    with pytest.raises(InvalidDataError, match="temperature_2m"):
        v.check_physical_ranges(ds)


def test_accumulation_must_not_decrease():
    ds = _forecast(precip=np.stack([np.full((2, 2), 10.0), np.full((2, 2), 4.0)]))
    with pytest.raises(InvalidDataError, match="decreases"):
        v.check_accumulation_monotonic(ds.precipitation)
    v.check_accumulation_monotonic(_forecast().precipitation)


def test_observation_coverage_detects_partial_day():
    data = np.ones((3, 4, 4), dtype="float32")
    data[2, :3, :] = np.nan  # day 3 lost most of its cells
    ds = xr.Dataset(
        {"precipitation": (("time", "latitude", "longitude"), data, {"units": "mm"})},
        coords={"time": np.array(["2026-09-01", "2026-09-02", "2026-09-03"], dtype="datetime64[ns]"),
                "latitude": [1.0, 2.0, 3.0, 4.0], "longitude": [1.0, 2.0, 3.0, 4.0]},
    )
    with pytest.raises(InvalidDataError, match="2026-09-03"):
        v.check_observation_coverage(ds)


def test_validate_forecast_dataset_full_pass():
    ds = _forecast(temperature_2m=(np.full((2, 2, 2), 25.0), "degC"))
    out, stats = v.validate_forecast_dataset(ds, ["precipitation", "temperature_2m"])
    assert stats["precipitation"]["max"] == 12.0
