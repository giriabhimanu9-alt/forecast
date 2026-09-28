"""Phase 2 unit tests: windows, 24 h accumulation, alignment, IMD convention, metrics,
split/leakage, weights, blending. All arrays are small hand-built test inputs."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from app.models.schema import InvalidDataError
from app.services.data.observations import PERIOD_END_OFFSET
from app.services.verification import metrics
from app.services.verification.accumulation import (
    ForecastWindow,
    forecast_to_24h_accumulation,
    required_lead_hours,
    verification_windows,
)
from app.services.verification.alignment import AlignmentError, align_forecast_to_observation, build_aligned_dataset
from app.services.verification.blending import (
    apply_blend,
    check_weight_table,
    equal_weights,
    inverse_error_weights,
    learn_weights,
    weight_field,
)
from app.services.verification.experiment import ExperimentConfig, observation_period_start, plan_init_times
from app.services.verification.regions import REGIONS, region_map
from app.services.verification.split import LeakageError, assert_no_leakage, chronological_split

LATS = np.array([10.0, 20.0, 30.0])
LONS = np.array([75.0, 80.0, 92.0])
INIT = datetime(2025, 7, 10, 0)


def accumulated(leads_h, values_per_lead):
    """Accumulated-since-init field shaped like Phase 1 output for one model/time."""
    data = np.stack([np.full((LATS.size, LONS.size), v, dtype="float32") for v in values_per_lead])
    da = xr.DataArray(
        data, dims=("lead_time", "latitude", "longitude"),
        coords={"lead_time": np.array([np.timedelta64(h, "h") for h in leads_h], dtype="timedelta64[ns]"),
                "latitude": LATS, "longitude": LONS},
        attrs={"units": "mm", "accumulation": "since forecast initialisation (time) up to valid_time"},
    )
    return da


def observations(labels, route="archive", values=None):
    """Observation dataset in the Phase 1 contract for IMD day labels."""
    labels = pd.DatetimeIndex(labels)
    end = labels + PERIOD_END_OFFSET[route]
    vals = values if values is not None else np.stack([np.full((LATS.size, LONS.size), float(i + 1)) for i in range(len(labels))])
    ds = xr.Dataset(
        {"precipitation": (("time", "latitude", "longitude"), vals.astype("float32"), {"units": "mm"})},
        coords={"time": end.values.astype("datetime64[ns]"), "latitude": LATS, "longitude": LONS},
    )
    return ds.assign_coords(
        period_start=("time", (end - pd.Timedelta(hours=24)).values.astype("datetime64[ns]")),
        source_date=("time", np.array(labels.strftime("%Y-%m-%d"))),
    )


# --------------------------------------------------------------------------- windows / lead handling

def test_windows_for_00z_12z_and_03z_cycles():
    w00 = verification_windows(datetime(2025, 7, 10, 0), [1, 2, 3])
    assert [(w.start_lead_h, w.end_lead_h) for w in w00] == [(3, 27), (27, 51), (51, 75)]
    assert w00[0].start == datetime(2025, 7, 10, 3) and w00[0].end == datetime(2025, 7, 11, 3)
    w12 = verification_windows(datetime(2025, 7, 10, 12), [1])
    assert (w12[0].start_lead_h, w12[0].end_lead_h) == (15, 39)
    w03 = verification_windows(datetime(2025, 7, 10, 3), [1])
    assert (w03[0].start_lead_h, w03[0].end_lead_h) == (0, 24)  # window starts at step 0
    with pytest.raises(ValueError):
        verification_windows(INIT, [0])


def test_required_leads_exclude_step_zero():
    assert required_lead_hours(verification_windows(datetime(2025, 7, 10, 3), [1, 2])) == [24, 48]
    assert required_lead_hours(verification_windows(INIT, [1, 2, 3])) == [3, 27, 51, 75]


# --------------------------------------------------------------------------- 24 h accumulation

def test_24h_accumulation_is_difference_not_cumulative():
    acc = accumulated([3, 27, 51, 75], [1.0, 11.0, 31.0, 36.0])
    assert float(forecast_to_24h_accumulation(acc, 3, 27).max()) == pytest.approx(10.0)
    assert float(forecast_to_24h_accumulation(acc, 27, 51).max()) == pytest.approx(20.0)  # not 31 (0-51 h)
    assert float(forecast_to_24h_accumulation(acc, 51, 75).max()) == pytest.approx(5.0)   # not 36 (0-75 h)


def test_step_zero_accumulation_is_zero():
    acc = accumulated([24, 48], [7.0, 9.0])
    assert float(forecast_to_24h_accumulation(acc, 0, 24).max()) == pytest.approx(7.0)


def test_non_24h_windows_rejected():
    acc = accumulated([3, 27, 51], [1.0, 2.0, 3.0])
    with pytest.raises(InvalidDataError, match="not 24 h"):
        forecast_to_24h_accumulation(acc, 3, 51)  # 48 h cumulative vs daily obs
    with pytest.raises(InvalidDataError, match="not 24 h"):
        forecast_to_24h_accumulation(acc, 0, 51)


def test_accumulation_input_checks():
    acc = accumulated([3, 27], [1.0, 2.0])
    with pytest.raises(InvalidDataError, match="lead 51 h not present"):
        forecast_to_24h_accumulation(accumulated([3, 27], [1.0, 2.0]), 27, 51)
    with pytest.raises(InvalidDataError, match="units"):
        forecast_to_24h_accumulation(acc.assign_attrs(units="m"), 3, 27)
    with pytest.raises(InvalidDataError, match="not marked as accumulated"):
        forecast_to_24h_accumulation(acc.assign_attrs(accumulation="6 h bucket"), 3, 27)


def test_accumulation_negative_handling():
    assert float(forecast_to_24h_accumulation(accumulated([3, 27], [5.0, 4.98]), 3, 27).min()) == 0.0
    with pytest.raises(InvalidDataError, match="decreases"):
        forecast_to_24h_accumulation(accumulated([3, 27], [5.0, 3.0]), 3, 27)


# --------------------------------------------------------------------------- IMD period convention

def test_observation_period_convention():
    # Both IMD routes: label D = 24 h ending 08:30 IST (03 UTC) on D (verified with July 2025 data).
    for route in ("archive", "realtime"):
        assert observation_period_start(date(2025, 7, 11), route) == datetime(2025, 7, 10, 3)
    obs = observations(["2025-07-11"])
    assert pd.Timestamp(obs.period_start.values[0]) == pd.Timestamp("2025-07-10 03:00")
    assert pd.Timestamp(obs.time.values[0]) == pd.Timestamp("2025-07-11 03:00")
    assert obs.source_date.values[0] == "2025-07-11"


def test_planned_inits_verify_exactly_the_target_periods():
    cfg = ExperimentConfig("t", date(2025, 7, 1), date(2025, 7, 5), output_dir=None, cycles=(0, 12))
    inits = plan_init_times(cfg, "archive")
    targets = {observation_period_start(d.date(), "archive") for d in pd.date_range("2025-07-01", "2025-07-05")}
    hits = {w.start for i in inits for w in verification_windows(i, cfg.lead_days) if w.start in targets}
    assert hits == targets
    assert all(i <= min(targets) + timedelta(days=5) for i in inits)
    assert all(i.hour in (0, 12) for i in inits)


# --------------------------------------------------------------------------- alignment

def _fc24(value=1.0):
    return xr.DataArray(np.full((LATS.size, LONS.size), value, dtype="float32"), dims=("latitude", "longitude"),
                        coords={"latitude": LATS, "longitude": LONS}, attrs={"units": "mm"})


def test_alignment_matches_exact_period():
    # Forecast initialised 2025-07-10 00Z, lead day 1 = 10 Jul 03Z -> 11 Jul 03Z = IMD label 11 Jul
    obs = observations(["2025-07-10", "2025-07-11", "2025-07-12"])
    s = align_forecast_to_observation(_fc24(), "GFS", verification_windows(INIT, [1])[0], obs)
    assert s.valid_period.valid_date == "2025-07-11"
    assert s.valid_period.start == pd.Timestamp("2025-07-10 03:00")
    assert s.valid_period.end == pd.Timestamp("2025-07-11 03:00")
    assert float(s.observation_value.max()) == 2.0  # second label
    assert s.init_time == pd.Timestamp(INIT) and s.lead_window_h == (3, 27)


def test_one_day_offset_cannot_occur_silently():
    window = verification_windows(INIT, [1])[0]
    # Observations exist for the day before and after, but NOT for the forecast window (label 11 Jul).
    obs = observations(["2025-07-10", "2025-07-12"])
    with pytest.raises(AlignmentError, match="no unique observation period"):
        align_forecast_to_observation(_fc24(), "GFS", window, obs)
    # Label 10 Jul under the verified convention ends a day early -> mismatch, never a silent match.
    with pytest.raises(AlignmentError):
        align_forecast_to_observation(_fc24(), "GFS", window, observations(["2025-07-10"]))
    # Relabelling it with the old (disproved) +1 day convention is what WOULD match that window, which is
    # exactly the error the per-run convention check guards against.
    shifted = observations(["2025-07-10"])
    shifted = shifted.assign_coords(time=shifted.time + np.timedelta64(1, "D"),
                                    period_start=shifted.period_start + np.timedelta64(1, "D"))
    assert align_forecast_to_observation(_fc24(), "GFS", window, shifted).valid_period.valid_date == "2025-07-10"
    # An observation ending at the right time but starting a day earlier (48 h period) is rejected too.
    bad = observations(["2025-07-11"]).assign_coords(period_start=("time", [np.datetime64("2025-07-09T03:00", "ns")]))
    with pytest.raises(AlignmentError, match="starts"):
        align_forecast_to_observation(_fc24(), "GFS", window, bad)


def test_alignment_rejects_forecast_issued_after_period_start():
    late = ForecastWindow(datetime(2025, 7, 10, 6), 1, -3, 21)
    with pytest.raises(AlignmentError, match="after the verified period start"):
        align_forecast_to_observation(_fc24(), "GFS", late, observations(["2025-07-11"]))


def test_alignment_rejects_grid_mismatch():
    fc = _fc24().assign_coords(latitude=LATS + 0.25)
    with pytest.raises(AlignmentError, match="grids differ"):
        align_forecast_to_observation(fc, "GFS", verification_windows(INIT, [1])[0], observations(["2025-07-11"]))


def _forecast_ds(model, init, values=(1.0, 11.0, 31.0, 36.0)):
    acc = accumulated([3, 27, 51, 75], values)
    return acc.expand_dims(model=[model], time=[np.datetime64(init, "ns")]).to_dataset(name="precipitation")


def test_build_aligned_dataset_with_missing_model_and_observation():
    inits = [datetime(2025, 7, 10), datetime(2025, 7, 11)]
    forecasts = {"GFS": {i: _forecast_ds("GFS", i) for i in inits}, "ECMWF": {inits[0]: _forecast_ds("ECMWF", inits[0])}}
    obs = observations(["2025-07-11", "2025-07-12", "2025-07-13"])  # no obs for label 14 Jul
    ds, log = build_aligned_dataset(forecasts, obs, inits, [1, 2, 3], ["GFS", "ECMWF"])
    # init 10: days 1-3 -> labels 11,12,13 ; init 11: days 1,2 -> 12,13 ; its day 3 (label 14 Jul) has no obs
    assert ds.sizes["sample"] == 5
    assert len(log.missing_observation) == 1 and "2025-07-14 03:00" in log.missing_observation[0]
    assert any("ECMWF 2025-07-11" in m for m in log.missing_forecast)
    ec = ds["forecast"].sel(model="ECMWF")
    missing = ds.init_time.values == np.datetime64("2025-07-11", "ns")
    assert np.isnan(ec.isel(sample=np.flatnonzero(missing))).all()
    day2 = ds.isel(sample=np.flatnonzero(ds.lead_day.values == 2))
    assert float(day2["forecast"].sel(model="GFS").max()) == pytest.approx(20.0)  # 31 - 11
    for s in range(ds.sizes["sample"]):
        assert ds.observation_period_end.values[s] - ds.observation_period_start.values[s] == np.timedelta64(24, "h")
        assert ds.init_time.values[s] <= ds.observation_period_start.values[s]
    only = {datetime(2025, 7, 12, 3)}
    ds2, _ = build_aligned_dataset(forecasts, obs, inits, [1, 2, 3], ["GFS", "ECMWF"], window_starts=only)
    assert set(ds2.valid_date.values.tolist()) == {"2025-07-13"}


# --------------------------------------------------------------------------- metrics

def test_continuous_metrics_by_hand():
    f = xr.DataArray([2.0, 4.0, np.nan, 10.0])
    o = xr.DataArray([1.0, 6.0, 3.0, np.nan])
    s = metrics.score(f, o, thresholds=[])
    assert s["n"] == 2
    assert s["mae"] == pytest.approx(1.5)
    assert s["rmse"] == pytest.approx(np.sqrt((1 + 4) / 2))
    assert s["bias"] == pytest.approx(-0.5)


def test_categorical_metrics_by_hand():
    f = np.array([20.0, 20.0, 0.0, 0.0, 30.0])
    o = np.array([20.0, 0.0, 20.0, 0.0, 16.0])
    c = metrics.categorical_scores(f, o, 15.6)
    assert (c["hits"], c["false_alarms"], c["misses"]) == (2, 1, 1)
    assert c["precision"] == pytest.approx(2 / 3)
    assert c["recall"] == pytest.approx(2 / 3)
    assert c["f1"] == pytest.approx(2 / 3)
    assert c["csi"] == pytest.approx(0.5)


def test_categorical_undefined_is_nan_not_zero():
    c = metrics.categorical_scores(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 64.5)
    assert np.isnan(c["precision"]) and np.isnan(c["recall"]) and np.isnan(c["csi"]) and np.isnan(c["f1"])


# --------------------------------------------------------------------------- weights

def test_inverse_error_weights():
    w = inverse_error_weights({"A": 1.0, "B": 3.0}, epsilon=0.0 + 1e-9)
    assert w["A"] == pytest.approx(0.75, rel=1e-6) and w["B"] == pytest.approx(0.25, rel=1e-6)
    w = inverse_error_weights({"A": 0.0, "B": 0.0}, epsilon=0.1)
    assert w == {"A": 0.5, "B": 0.5}
    w = inverse_error_weights({"A": 2.0, "B": float("nan")}, epsilon=0.1)
    assert w == {"A": 1.0}  # missing model removed, rest renormalised
    assert sum(inverse_error_weights({"A": 1.3, "B": 2.1, "C": 0.4}, 0.1).values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        inverse_error_weights({"A": 1.0}, epsilon=0)
    assert equal_weights(["A", "B", "C", "D"]) == {m: 0.25 for m in "ABCD"}


def _aligned(n_days=8, gfs_err=1.0, ec_err=3.0, lats=None, lons=None, ec_missing_day=None):
    """Aligned dataset where GFS and ECMWF have known constant errors."""
    lats = np.arange(8.0, 36.0, 4.0) if lats is None else lats
    lons = np.arange(70.0, 97.0, 4.0) if lons is None else lons
    labels = pd.date_range("2025-07-01", periods=n_days)
    obs = np.full((n_days, lats.size, lons.size), 10.0)
    fc = np.stack([obs + gfs_err, obs - ec_err])
    if ec_missing_day is not None:
        fc[1, ec_missing_day] = np.nan
    start = labels + pd.Timedelta(hours=3)
    return xr.Dataset(
        {"forecast": (("model", "sample", "latitude", "longitude"), fc),
         "observation": (("sample", "latitude", "longitude"), obs)},
        coords={"model": ["GFS", "ECMWF"], "sample": np.arange(n_days), "latitude": lats, "longitude": lons,
                "lead_day": ("sample", np.ones(n_days, dtype=int)),
                "start_lead_h": ("sample", np.full(n_days, 3)), "end_lead_h": ("sample", np.full(n_days, 27)),
                "init_time": ("sample", labels.values),
                "observation_period_start": ("sample", start.values),
                "observation_period_end": ("sample", (start + pd.Timedelta(hours=24)).values),
                "valid_date": ("sample", np.array(labels.strftime("%Y-%m-%d")))},
    )


def test_learn_weights_global_and_table_invariants():
    ds = _aligned()
    regions = region_map(ds.latitude.values, ds.longitude.values)
    t = learn_weights(ds, regions, epsilon=1e-9)
    check_weight_table(t)
    g = t[(t.method == "inverse_error")].set_index("model")
    assert g.loc["GFS", "error"] == pytest.approx(1.0) and g.loc["ECMWF", "error"] == pytest.approx(3.0)
    assert g.loc["GFS", "weight"] == pytest.approx(0.75, rel=1e-6)
    e = t[t.method == "equal"].set_index("model")
    assert e.weight.tolist() == [0.5, 0.5]
    assert set(t.columns) >= {"model", "region", "variable", "lead_day", "metric", "training_start",
                              "training_end", "error", "weight"}
    assert ((t.weight >= 0) & (t.weight <= 1)).all()
    assert set(t[t.method == "regional"].region) == set(REGIONS)


def test_regional_weights_follow_regional_errors():
    ds = _aligned()
    regions = region_map(ds.latitude.values, ds.longitude.values)
    south = (regions == "South").values
    # Make ECMWF perfect in the South only.
    fc = ds["forecast"].values.copy()
    fc[1][:, south] = ds["observation"].values[:, south]
    ds["forecast"] = (("model", "sample", "latitude", "longitude"), fc)
    t = learn_weights(ds, regions, epsilon=0.1)
    reg = t[t.method == "regional"].set_index(["region", "model"]).weight
    assert reg[("South", "ECMWF")] > 0.9
    assert reg[("North", "GFS")] == pytest.approx((1 / 1.1) / (1 / 1.1 + 1 / 3.1))
    w = weight_field(t, "regional", 1, ["GFS", "ECMWF"], regions)
    assert np.allclose(w.sum("model"), 1.0)
    assert float(w.sel(model="ECMWF").where(regions == "South").max()) > 0.9


def test_missing_model_fallback():
    ds = _aligned()
    ds["forecast"].loc[dict(model="ECMWF")] = np.nan  # ECMWF entirely unavailable in training
    t = learn_weights(ds, region_map(ds.latitude.values, ds.longitude.values), epsilon=0.1)
    for method in ("equal", "inverse_error"):
        w = t[t.method == method].set_index("model")
        assert w.loc["GFS", "weight"] == 1.0 and w.loc["ECMWF", "weight"] == 0.0
        assert not w.loc["ECMWF", "available"]
    # Blend-time fallback: a member missing on one sample is dropped and weights renormalised.
    fc = _aligned(ec_missing_day=2)["forecast"]
    weights = xr.DataArray([0.4, 0.6], dims="model", coords={"model": ["GFS", "ECMWF"]})
    blend = apply_blend(fc, weights)
    assert float(blend.isel(sample=2).max()) == pytest.approx(11.0)  # GFS only
    assert float(blend.isel(sample=0).max()) == pytest.approx(0.4 * 11 + 0.6 * 7)
    none = apply_blend(fc * np.nan, weights)
    assert bool(none.isnull().all())


# --------------------------------------------------------------------------- split & leakage

def test_chronological_split():
    s = chronological_split(_aligned(n_days=10), 0.75)
    assert s.train_dates == [f"2025-07-{d:02d}" for d in range(1, 9)]
    assert s.test_dates == ["2025-07-09", "2025-07-10"]
    assert pd.Timestamp(s.train.observation_period_end.values.max()) <= pd.Timestamp(s.test.observation_period_start.values.min())
    with pytest.raises(ValueError):
        chronological_split(_aligned(n_days=1), 0.75)
    with pytest.raises(ValueError):
        chronological_split(_aligned(), 1.0)


def test_split_keeps_all_lead_days_of_a_valid_date_together():
    ds = xr.concat([_aligned(n_days=6), _aligned(n_days=6).assign_coords(lead_day=("sample", np.full(6, 2)))], dim="sample")
    s = chronological_split(ds, 0.5)
    assert not set(s.train.valid_date.values.tolist()) & set(s.test.valid_date.values.tolist())


def test_leakage_detected():
    ds = _aligned(n_days=6)
    with pytest.raises(LeakageError):
        assert_no_leakage(ds.isel(sample=[0, 1, 4]), ds.isel(sample=[2, 3]))


def test_weights_do_not_depend_on_test_observations():
    from app.services.verification.experiment import blended_forecasts

    ds = _aligned(n_days=10)
    regions = region_map(ds.latitude.values, ds.longitude.values)
    split = chronological_split(ds, 0.7)
    w1 = learn_weights(split.train, regions, 0.1)
    b1 = blended_forecasts(split.test, w1, regions, ["GFS", "ECMWF"])

    tampered = ds.copy(deep=True)
    tampered["observation"][7:] = 999.0  # change ONLY test-period observations
    split2 = chronological_split(tampered, 0.7)
    w2 = learn_weights(split2.train, regions, 0.1)
    b2 = blended_forecasts(split2.test, w2, regions, ["GFS", "ECMWF"])
    pd.testing.assert_frame_equal(w1, w2)
    for m in b1:
        xr.testing.assert_equal(b1[m], b2[m])


def test_region_map_labels():
    r = region_map(np.array([12.0, 24.0, 26.0, 32.0]), np.array([74.0, 80.0, 86.0, 92.0]))
    assert r.sel(latitude=12.0, longitude=80.0) == "South"
    assert r.sel(latitude=24.0, longitude=74.0) == "West"
    assert r.sel(latitude=24.0, longitude=80.0) == "Central"
    assert r.sel(latitude=24.0, longitude=86.0) == "East"
    assert r.sel(latitude=26.0, longitude=92.0) == "Northeast"
    assert r.sel(latitude=32.0, longitude=80.0) == "North"
