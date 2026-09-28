"""Phase 4 tests: AI provider interface, AI precipitation conversion, regridding, temporal alignment,
multi-season handling, missing-model fallback, 3-model weights, leakage, regimes, extreme metrics.

All arrays are small hand-built inputs; nothing here is presented as weather data.
"""

from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from app.config import TargetGrid
from app.models.schema import to_standard_forecast
from app.services.data.ai_model import AIModelProvider, create_ai_provider, registered_ai_providers
from app.services.data.base import ForecastProvider
from app.services.data.ecmwf import ECMWFProvider
from app.services.data.gfs import GFSAtmosProvider, GFSProvider
from app.services.data.normalization import convert_units, normalize_forecast, normalize_observations
from app.services.data.wb2 import GraphCastWB2Provider, HRESWB2Provider
from app.services.verification.accumulation import verification_windows
from app.services.verification.adaptive import check_weights, learn_regime_weights
from app.services.verification.alignment import AlignmentError, build_aligned_dataset
from app.services.verification.experiment import plan_init_times, provider_class
from app.services.verification.phase4 import (
    BlendConfig,
    ExperimentAConfig,
    _by_season,
    atmos_features,
    coarse_land_mask,
    concat_seasons,
    evaluate,
)
from app.services.verification.production import NoForecastAvailable, blend_available
from app.services.verification.regions import region_map
from app.services.verification.split import LeakageError, assert_no_leakage

WB2_GRID = TargetGrid(6.0, 37.5, 67.5, 97.5, 1.5, name="test 1.5")


# --------------------------------------------------------------------------- provider interface

def test_ai_provider_implements_common_interface(make_settings):
    s = make_settings()
    g = GraphCastWB2Provider(s)
    assert isinstance(g, AIModelProvider) and isinstance(g, ForecastProvider)
    assert g.kind == "ai_forecast" and g.model_name == "GraphCast"
    assert "graphcast-wb2" in registered_ai_providers()
    assert isinstance(create_ai_provider(make_settings(AI_MODEL_PROVIDER="graphcast-wb2")), GraphCastWB2Provider)
    assert provider_class("graphcast-wb2") is GraphCastWB2Provider
    h = HRESWB2Provider(s)
    assert h.kind == "forecast" and not isinstance(h, AIModelProvider)


def _write_wb2_cache(provider, init, var, values_by_lead):
    lats = np.arange(3.0, 40.6, 1.5)
    lons = np.arange(64.5, 100.6, 1.5)
    leads = np.array([np.timedelta64(h, "h") for h in values_by_lead], dtype="timedelta64[ns]")
    data = np.stack([np.full((lons.size, lats.size), v, dtype="float32") for v in values_by_lead.values()])
    ds = xr.Dataset({var: (("prediction_timedelta", "longitude", "latitude"), data)},
                    coords={"prediction_timedelta": leads, "longitude": lons, "latitude": lats,
                            "time": np.datetime64(init, "ns")})
    path = provider.cache_path(init)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_graphcast_6h_buckets_to_accumulation_with_negative_clipping(make_settings):
    s = replace(make_settings(), grid=WB2_GRID)
    g = GraphCastWB2Provider(s)
    init = datetime(2020, 7, 1)
    buckets = {h: 0.001 for h in range(6, 73, 6)}  # 1 mm per 6 h
    buckets[12] = -0.00005  # artefact: slightly negative bucket
    _write_wb2_cache(g, init, "total_precipitation_6hr", buckets)
    ds, stats = g.get_forecast(init, [24, 48, 72], ["precipitation"])
    pr = ds.precipitation.isel(model=0, time=0)
    assert pr.attrs["units"] == "mm"
    np.testing.assert_allclose(pr.sel(lead_time=np.timedelta64(24, "h")).values, 3.0, atol=1e-4)  # 1+0+1+1
    np.testing.assert_allclose(pr.sel(lead_time=np.timedelta64(72, "h")).values, 11.0, atol=1e-4)
    assert ds.attrs["regrid_method"] == "exact-subset" and ds.sizes["latitude"] == 22
    assert stats.downloads == 0 and stats.cache_hits == 1  # read from cache, no network
    view = to_standard_forecast(ds)
    assert view.dims[0] == "model_name" and view.model_name.values.tolist() == ["GraphCast"]


def test_hres_since_init_conversion(make_settings):
    s = replace(make_settings(), grid=WB2_GRID)
    h = HRESWB2Provider(s)
    init = datetime(2020, 7, 1)
    _write_wb2_cache(h, init, "total_precipitation", {24: 0.004, 48: 0.010, 72: 0.012})
    ds, _ = h.get_forecast(init, [24, 48, 72], ["precipitation"])
    pr = ds.precipitation.isel(model=0, time=0)
    np.testing.assert_allclose(pr.sel(lead_time=np.timedelta64(48, "h")).values, 10.0, rtol=1e-5)


# --------------------------------------------------------------------------- regridding

def test_bilinear_regrid_from_ecmwf_0p4_grid():
    grid = TargetGrid(10.0, 12.0, 70.0, 72.0, 0.25)
    lats = np.round(np.arange(8.0, 14.01, 0.4), 6)
    lons = np.round(np.arange(68.0, 74.01, 0.4), 6)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    field = xr.DataArray((2 * lat2d + lon2d)[None] / 1000.0, dims=("lead_time", "latitude", "longitude"),
                         coords={"lead_time": [np.timedelta64(24, "h")], "latitude": lats, "longitude": lons},
                         attrs={"units": "m"})
    ds = normalize_forecast({"precipitation": field}, "ECMWF", datetime(2023, 7, 1), grid, ["precipitation"])
    assert ds.attrs["regrid_method"] == "bilinear"
    got = ds.precipitation.isel(model=0, time=0, lead_time=0)
    np.testing.assert_allclose(got.sel(latitude=11.25, longitude=70.75), 2 * 11.25 + 70.75, rtol=1e-5)  # linear field exact


def test_coarse_land_mask():
    fine = TargetGrid(5.0, 38.0, 68.0, 98.0, 0.25)
    land = xr.DataArray(np.zeros(fine.shape, bool), dims=("latitude", "longitude"),
                        coords={"latitude": fine.latitudes, "longitude": fine.longitudes})
    land.loc[dict(latitude=slice(20, 30), longitude=slice(75, 85))] = True
    m = coarse_land_mask(land, WB2_GRID, 0.5)
    assert bool(m.sel(latitude=25.5, longitude=79.5)) and not bool(m.sel(latitude=10.5, longitude=70.5))


# --------------------------------------------------------------------------- temporal alignment

def test_six_hourly_models_can_never_match_imd_windows():
    for cycle in (0, 6, 12, 18):
        w = verification_windows(datetime(2024, 7, 1, cycle), [1, 2, 3])
        assert all(x.start_lead_h % 6 != 0 and x.end_lead_h % 6 != 0 for x in w), cycle
    # ERA5 benchmark windows (end 00 UTC) are 6-hourly compatible from 00Z
    w = verification_windows(datetime(2020, 7, 1, 0), [1, 2, 3], window_end_hour_utc=0)
    assert [(x.start_lead_h, x.end_lead_h) for x in w] == [(0, 24), (24, 48), (48, 72)]


def _acc_dataset(model, init, values):
    lats, lons = np.array([10.0, 11.5]), np.array([70.5, 72.0])
    leads = np.array([np.timedelta64(h, "h") for h in values], dtype="timedelta64[ns]")
    data = np.stack([np.full((2, 2), v, "float32") for v in values.values()])
    da = xr.DataArray(data, dims=("lead_time", "latitude", "longitude"),
                      coords={"lead_time": leads, "latitude": lats, "longitude": lons},
                      attrs={"units": "mm", "accumulation": "since forecast initialisation (time) up to valid_time"})
    return da.expand_dims(model=[model], time=[np.datetime64(init, "ns")]).to_dataset(name="precipitation")


def _era5_like(ends):
    lats, lons = np.array([10.0, 11.5]), np.array([70.5, 72.0])
    rain = xr.DataArray(np.stack([np.full((2, 2), 0.001 * (i + 1)) for i in range(len(ends))]),
                        dims=("time", "latitude", "longitude"),
                        coords={"time": pd.DatetimeIndex(ends), "latitude": lats, "longitude": lons}, attrs={"units": "m"})
    grid = TargetGrid(10.0, 11.5, 70.5, 72.0, 1.5)
    return normalize_observations(rain, grid, "ERA5", period_end_offset=pd.Timedelta(hours=0))


def test_era5_windows_align_at_00utc_and_reject_imd_windows():
    init = datetime(2020, 7, 1)
    fc = {"GraphCast": {init: _acc_dataset("GraphCast", init, {24: 5.0, 48: 9.0, 72: 12.0})}}
    obs = _era5_like(["2020-07-02", "2020-07-03", "2020-07-04"])
    assert pd.Timestamp(obs.period_start.values[0]) == pd.Timestamp("2020-07-01 00:00")
    ds, log = build_aligned_dataset(fc, obs, [init], [1, 2, 3], ["GraphCast"], window_end_hour_utc=0)
    assert ds.sizes["sample"] == 3
    np.testing.assert_allclose(ds["forecast"].sel(model="GraphCast").max(("latitude", "longitude")).values, [5, 4, 3])
    np.testing.assert_allclose(ds["observation"].max(("latitude", "longitude")).values, [1, 2, 3])
    # With IMD (03 UTC) windows the same 6-hourly forecast has no 3/27 h leads -> hard error, never a silent match
    with pytest.raises(Exception):
        build_aligned_dataset(fc, obs, [init], [1], ["GraphCast"])


def test_unit_conversions_for_new_fields():
    pa = xr.DataArray([100800.0], dims="x", attrs={"units": "Pa"})
    assert float(convert_units(pa, "mslp", "hPa")[0]) == pytest.approx(1008.0)
    pw = xr.DataArray([55.0], dims="x", attrs={"units": "kg m**-2"})
    assert float(convert_units(pw, "precipitable_water", "mm")[0]) == pytest.approx(55.0)


# --------------------------------------------------------------------------- providers / multi-season

def test_ecmwf_product_path_by_date_and_gfs_1deg(make_settings):
    e = ECMWFProvider(make_settings())
    assert "/0p4-beta/oper/" in e.grib_url(datetime(2023, 7, 10), 27) and e.resolution_deg(datetime(2023, 7, 10)) == 0.4
    assert "/0p25/oper/" in e.grib_url(datetime(2024, 2, 10), 27) and "/ifs/" not in e.grib_url(datetime(2024, 2, 10), 27)
    assert "/ifs/0p25/oper/" in e.grib_url(datetime(2025, 7, 10), 27)
    g = GFSAtmosProvider(make_settings())
    assert g.grib_url(datetime(2023, 7, 10), 27).endswith("pgrb2.1p00.f027")
    assert GFSProvider(make_settings()).grib_url(datetime(2023, 7, 10), 27).endswith("pgrb2.0p25.f027")
    entries = g.parse_index(
        "1:0:d=2023071000:PRMSL:mean sea level:27 hour fcst:\n2:10:d=2023071000:UGRD:850 mb:27 hour fcst:\n"
        "3:20:d=2023071000:PWAT:entire atmosphere (considered as a single layer):27 hour fcst:\n4:30:d=x:TMP:2 m:x:\n",
        27, ["mslp", "u_850hpa", "precipitable_water"])
    assert (entries["u_850hpa"].byte_range.start, entries["u_850hpa"].byte_range.end) == (10, 19)


def test_season_config_and_planning():
    cfg = ExperimentAConfig()
    s = cfg.season_config(2024)
    assert (s.valid_start, s.valid_end) == (date(2024, 6, 1), date(2024, 9, 30))
    inits = plan_init_times(s, "archive")
    assert inits[0] == datetime(2024, 5, 29) and inits[-1] == datetime(2024, 9, 29)
    assert len(inits) == len(set(inits)) and all(i.hour == 0 for i in inits)


def _season_samples(seasons=(2021, 2022, 2023), days=6, n_models=3, seed=0):
    rng = np.random.default_rng(seed)
    lats, lons = np.arange(8.0, 36.0, 2.0), np.arange(68.0, 98.0, 2.0)
    models = ["GFS", "ECMWF", "GraphCast"][:n_models]
    parts = []
    for season in seasons:
        rows, fcs, obs = [], [], []
        for d in range(days):
            for k in (1, 2):
                base = rng.gamma(1.0, 4.0, (lats.size, lons.size)) + (60.0 if d % 3 == 0 else 0.0)
                obs.append(base)
                fcs.append([base + 1.0, base - 2.0 + rng.normal(0, 1, base.shape), base * 0.8][:n_models])
                label = pd.Timestamp(f"{season}-07-{d + 1:02d}")
                start = label - pd.Timedelta(hours=21)
                rows.append({"valid_date": label.strftime("%Y-%m-%d"), "lead_day": k, "init_time": start - pd.Timedelta(hours=3 + 24 * (k - 1)),
                             "observation_period_start": start, "observation_period_end": start + pd.Timedelta(hours=24),
                             "start_lead_h": 3 + 24 * (k - 1), "end_lead_h": 27 + 24 * (k - 1)})
        m = pd.DataFrame(rows)
        ds = xr.Dataset({"forecast": (("model", "sample", "latitude", "longitude"), np.clip(np.moveaxis(np.array(fcs), 1, 0), 0, None)),
                         "observation": (("sample", "latitude", "longitude"), np.array(obs))},
                        coords={"model": models, "sample": np.arange(len(m)), "latitude": lats, "longitude": lons,
                                **{c: ("sample", m[c].to_numpy()) for c in m.columns}})
        parts.append(ds.assign_coords(season=("sample", np.full(len(m), season))))
    return concat_seasons(parts)


def test_multi_season_concat_and_split():
    ds = _season_samples()
    assert ds.sizes["sample"] == 36 and list(ds.sample.values) == list(range(36))
    assert set(_by_season(ds, [2022]).season.values) == {2022}
    assert_no_leakage(_by_season(ds, [2021]), _by_season(ds, [2022]))
    with pytest.raises(LeakageError):
        assert_no_leakage(_by_season(ds, [2022, 2023]), _by_season(ds, [2021]))


def test_three_model_weights_normalise():
    ds = _season_samples(seasons=(2021,), days=9)
    regions = region_map(ds.latitude.values, ds.longitude.values)
    labels = np.array([0 if d % 3 == 0 else 1 for d in range(9) for _ in (1, 2)])
    t = learn_regime_weights(ds, labels, regions, 0.1, min_samples=2, min_points=10)
    check_weights(t)
    assert set(t.model) == {"GFS", "ECMWF", "GraphCast"}
    assert ((t.weight >= 0) & (t.weight <= 1)).all()


# --------------------------------------------------------------------------- production fallback

def _f(v):
    return xr.DataArray(np.full((2, 2), v, float), dims=("latitude", "longitude"))


def test_missing_ai_model_fallback():
    r = blend_available({"GFS": _f(1.0), "ECMWF": _f(3.0), "GraphCast": None}, {"GFS": 0.2, "ECMWF": 0.3, "GraphCast": 0.5},
                        context={"regime": 1, "lead_day": 2})
    assert r.metadata["missing_models"] == ["GraphCast"] and r.metadata["fallback_used"]
    assert r.metadata["weights"] == {"GFS": 0.4, "ECMWF": 0.6, "GraphCast": 0.0}
    assert float(r.forecast.mean()) == pytest.approx(0.4 * 1 + 0.6 * 3)
    assert r.metadata["regime"] == 1 and r.metadata["lead_day"] == 2


def test_missing_nwp_and_multiple_models():
    r = blend_available({"GFS": _f(np.nan), "ECMWF": _f(3.0), "GraphCast": _f(5.0)}, {"GFS": 0.5, "ECMWF": 0.25, "GraphCast": 0.25})
    assert r.metadata["available_models"] == ["ECMWF", "GraphCast"] and r.metadata["weights"]["ECMWF"] == 0.5
    one = blend_available({"GFS": None, "ECMWF": None, "GraphCast": _f(5.0)}, {"GFS": 0.5, "ECMWF": 0.5})
    assert "single model" in one.metadata["fallback_reason"] and float(one.forecast.mean()) == 5.0
    assert one.metadata["weights"]["GraphCast"] == 1.0  # no learned weight -> equal among available
    with pytest.raises(NoForecastAvailable):
        blend_available({"GFS": None, "ECMWF": _f(np.nan)}, {"GFS": 1.0})
    none_missing = blend_available({"GFS": _f(1.0), "ECMWF": _f(3.0)}, {"GFS": 0.5, "ECMWF": 0.5})
    assert not none_missing.metadata["fallback_used"]


# --------------------------------------------------------------------------- regimes & leakage (engine)

def test_atmos_features_from_window_leads():
    ds = _season_samples(seasons=(2021,), days=2, n_models=2)
    lats, lons = ds.latitude.values, ds.longitude.values
    land = xr.DataArray(np.ones((lats.size, lons.size), bool), dims=("latitude", "longitude"),
                        coords={"latitude": lats, "longitude": lons})
    atmos = {}
    for init in pd.DatetimeIndex(ds.init_time.values).unique():
        leads = np.array([3, 27, 51], dtype="timedelta64[h]").astype("timedelta64[ns]")
        shape = (1, 1, leads.size, lats.size, lons.size)
        mk = lambda v: (("model", "time", "lead_time", "latitude", "longitude"), np.full(shape, v, "float32"))
        atmos[init.to_pydatetime()] = xr.Dataset({"mslp": mk(1000.0), "u_850hpa": mk(12.0), "precipitable_water": mk(50.0)},
                                                 coords={"model": ["GFS_ATMOS_1P00"], "time": [init], "lead_time": leads,
                                                         "latitude": lats, "longitude": lons})
    f = atmos_features(ds, atmos, land)
    assert list(f.columns) == ["mslp_monsoon_trough", "u850_arabian_sea_jet", "pwat_india_land"]
    assert f.mslp_monsoon_trough.eq(1000.0).all() and f.pwat_india_land.eq(50.0).all()
    with pytest.raises(ValueError, match="missing"):
        atmos_features(ds, {}, land)


def test_evaluate_no_test_leakage(tmp_path):
    ds = _season_samples(days=9)
    lats, lons = ds.latitude.values, ds.longitude.values
    land = xr.DataArray(np.ones((lats.size, lons.size), bool), dims=("latitude", "longitude"),
                        coords={"latitude": lats, "longitude": lons})
    regions = region_map(lats, lons)
    from app.services.verification.regimes import regime_features

    feats = regime_features(ds["forecast"], land, regions)
    bcfg = BlendConfig(min_samples=2, min_points=10, bootstrap_samples=50, ks=(2, 3))

    def run(dataset, sub):
        out = tmp_path / sub
        for d in ("model_scores", "regime_analysis", "weights", "verification", "figures"):
            (out / d).mkdir(parents=True, exist_ok=True)
        return evaluate(dataset, {"rain": feats}, (2021,), (2022,), (2023,), bcfg, land, regions, out, "t")

    a = run(ds, "a")
    tampered = ds.copy(deep=True)
    test_idx = np.flatnonzero(ds.season.values == 2023)
    tampered["observation"][test_idx] = 999.0  # change ONLY test-season observations
    b = run(tampered, "b")
    pd.testing.assert_frame_equal(a["weights"], b["weights"])
    pd.testing.assert_frame_equal(a["static_weights"], b["static_weights"])
    assert a["chosen_feature_set"] == b["chosen_feature_set"]
    assert (tmp_path / "a" / "verification" / "t_blend_explanations.jsonl").read_text().count("\n") == test_idx.size
    fin = a["final"]
    row = fin[(fin.method == "GraphCast") & (fin.lead_day == "all") & (fin.regime.astype(str) == "all")].iloc[0]
    assert {"csi@64.5mm", "pod@64.5mm", "far@64.5mm", "n_obs_heavy"} <= set(fin.columns) and row.n > 0


def test_practical_significance_assessment():
    from app.services.verification.phase4 import assess

    assert assess("better (95% CI < 0)", -0.02, 1.0, True).startswith("negligible")  # 0.02 % of MAE
    assert assess("better (95% CI < 0)", -6.3, 1.0, True) == "meaningful improvement"
    assert assess("worse (95% CI > 0)", 2.0, 1.0, True) == "meaningful degradation"
    assert assess("not distinguishable", -5.0, 1.0, True) == "not distinguishable"
    assert assess("better (95% CI > 0)", 0.012, 0.005, False) == "meaningful improvement"
    assert assess("better (95% CI > 0)", 0.0015, 0.005, False).startswith("negligible")
    assert assess("undefined (no events)", float("nan"), 0.005, False).startswith("undefined")
