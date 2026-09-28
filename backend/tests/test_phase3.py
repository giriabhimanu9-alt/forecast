"""Phase 3 unit tests: regime features, clustering, persistence, leakage, adaptive weights, blends, metrics.

All arrays are small hand-built test inputs (never presented as weather data).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from app.services.verification import metrics
from app.services.verification.adaptive import (
    LEVEL_CSI,
    LEVEL_INDIA,
    LEVEL_REGION,
    LEVEL_STATIC,
    apply_regime_blend,
    check_weights,
    learn_extreme_aware_weights,
    learn_regime_weights,
    static_weight_table,
    weight_differences,
)
from app.services.verification.phase3 import Phase3Config, bootstrap_csi_difference, daily_counts, fit_stage
from app.services.verification.regimes import (
    FEATURE_NAMES,
    RegimeModel,
    ocean_box_masks,
    regime_features,
    regime_statistics,
    select_k,
)
from app.services.verification.regions import REGIONS, region_map

LATS = np.arange(8.0, 36.0, 2.0)
LONS = np.arange(68.0, 98.0, 2.0)


def make_samples(n_days=12, lead_days=(1, 2), seed=0, wet_days=None, gfs_err=1.0, ec_err=2.0):
    """Aligned-samples dataset: 'wet' days have heavy rain; ECMWF catches heavy rain better, GFS is
    closer in light rain (so weights must differ by regime)."""
    rng = np.random.default_rng(seed)
    wet_days = set(range(0, n_days, 3)) if wet_days is None else set(wet_days)
    labels = pd.date_range("2025-07-01", periods=n_days)
    obs, gfs, ec, meta = [], [], [], []
    for d in range(n_days):
        for k in lead_days:
            base = rng.gamma(1.0, 3.0, size=(LATS.size, LONS.size))
            if d in wet_days:
                base = base + 80.0
                g, e = base - 30.0 + rng.normal(0, 1, base.shape), base + rng.normal(0, 1, base.shape)
            else:
                g, e = base + gfs_err, base + ec_err
            obs.append(base); gfs.append(np.clip(g, 0, None)); ec.append(np.clip(e, 0, None))
            start = labels[d] - pd.Timedelta(hours=21)
            meta.append({"valid_date": labels[d].strftime("%Y-%m-%d"), "lead_day": k,
                         "init_time": start - pd.Timedelta(hours=3 + 24 * (k - 1)),
                         "observation_period_start": start, "observation_period_end": start + pd.Timedelta(hours=24),
                         "start_lead_h": 3 + 24 * (k - 1), "end_lead_h": 27 + 24 * (k - 1)})
    m = pd.DataFrame(meta)
    ds = xr.Dataset(
        {"forecast": (("model", "sample", "latitude", "longitude"), np.stack([gfs, ec])),
         "observation": (("sample", "latitude", "longitude"), np.array(obs))},
        coords={"model": ["GFS", "ECMWF"], "sample": np.arange(len(m)), "latitude": LATS, "longitude": LONS,
                **{c: ("sample", m[c].to_numpy()) for c in m.columns}},
    )
    return ds


@pytest.fixture
def geo():
    land = xr.DataArray(np.ones((LATS.size, LONS.size), bool), dims=("latitude", "longitude"),
                        coords={"latitude": LATS, "longitude": LONS})
    land[:, :2] = False  # an 'ocean' strip for the wind boxes
    return land, region_map(LATS, LONS)


def feats_for(ds, geo):
    land, regions = geo
    wind = {"wind_arabian_sea": np.linspace(5, 9, ds.sizes["sample"]), "wind_bay_of_bengal": np.full(ds.sizes["sample"], 6.0)}
    return regime_features(ds["forecast"], land, regions, wind)


# --------------------------------------------------------------------------- features & clustering

def test_features_are_forecast_only(geo):
    ds = make_samples()
    f1 = feats_for(ds, geo)
    assert list(f1.columns) == FEATURE_NAMES
    tampered = ds.copy(deep=True)
    tampered["observation"][:] = 0.0  # observations must not influence regime features
    pd.testing.assert_frame_equal(f1, feats_for(tampered, geo))


def test_ocean_boxes_exclude_land(geo):
    land, _ = geo
    boxes = ocean_box_masks(LATS, LONS, land)
    assert not bool((boxes["wind_arabian_sea"] & land).any())


def test_clustering_is_reproducible_and_json_roundtrip(tmp_path, geo):
    f = feats_for(make_samples(), geo)
    a, b = RegimeModel.fit(f, 2, seed=7), RegimeModel.fit(f, 2, seed=7)
    np.testing.assert_allclose(a.centers, b.centers)
    np.testing.assert_array_equal(a.predict(f), b.predict(f))
    a.save(tmp_path / "m.json")
    loaded = RegimeModel.load(tmp_path / "m.json")
    np.testing.assert_array_equal(loaded.predict(f), a.predict(f))
    # numpy predict matches sklearn's own KMeans on the same PCA space
    from sklearn.cluster import KMeans
    km = KMeans(2, n_init=50, random_state=7).fit(a.transform(f))
    assert abs(pd.crosstab(km.labels_, a.predict(f)).values).max(axis=1).sum() == len(f)


def test_regime_ids_ordered_by_rain_coverage(geo):
    ds = make_samples()
    f = feats_for(ds, geo)
    labels = RegimeModel.fit(f, 2, seed=0).predict(f)
    cov = f["frac_rain_ge_2.5mm"]
    assert cov[labels == 0].mean() <= cov[labels == 1].mean()


def test_wet_days_form_their_own_regime(geo):
    ds = make_samples()
    f = feats_for(ds, geo)
    labels = RegimeModel.fit(f, 2, seed=0).predict(f)
    wet = np.isin(pd.DatetimeIndex(ds.valid_date.values).day - 1, list(range(0, 12, 3)))
    assert len(set(labels[wet])) == 1 and len(set(labels[~wet])) == 1 and set(labels[wet]) != set(labels[~wet])


def test_select_k_uses_constraints(geo):
    f = feats_for(make_samples(n_days=15), geo)
    k, table = select_k(f, (2, 3, 4), seed=0, n_boot=10)
    assert k in (2, 3, 4)
    assert {"silhouette", "stability_ari", "min_cluster_size", "eligible", "selected"} <= set(table.columns)
    assert table.selected.sum() == 1


def test_scaler_pca_fit_only_on_training(geo):
    ds = make_samples(n_days=12)
    f = feats_for(ds, geo)
    train = f.iloc[:16]
    model = RegimeModel.fit(train, 2, seed=0)
    np.testing.assert_allclose(model.scaler_mean, train.to_numpy().mean(0))
    assert model.metadata["n_fit_samples"] == 16


def test_regime_statistics_and_names(geo):
    land, regions = geo
    ds = make_samples()
    f = feats_for(ds, geo)
    labels = RegimeModel.fit(f, 2, seed=0).predict(f)
    st = regime_statistics(ds, labels, f, regions, land)
    wet = st.sort_values("obs_mean_mm").iloc[-1]
    assert wet.is_heavy_regime and "HEAVY" in wet["name"]
    dry = st.sort_values("obs_mean_mm").iloc[0]
    assert not dry.is_heavy_regime and dry.obs_heavy_frequency == 0
    assert st.n_samples.sum() == ds.sizes["sample"]


# --------------------------------------------------------------------------- weights

def _regime_setup(geo, **kw):
    land, regions = geo
    ds = make_samples(**kw)
    f = feats_for(ds, geo)
    labels = RegimeModel.fit(f, 2, seed=0).predict(f)
    return ds, labels, regions, land, f


def test_regime_weights_differ_by_regime_and_normalise(geo):
    ds, labels, regions, land, f = _regime_setup(geo)
    t = learn_regime_weights(ds, labels, regions, epsilon=0.1, min_samples=2, min_points=10)
    check_weights(t)
    assert ((t.weight >= 0) & (t.weight <= 1)).all()
    wet_regime = int(np.bincount(labels).argmin())  # 1 in 3 days are wet
    w = t[(t.lead_day == 1) & (t.region == "Central")].set_index(["regime", "model"]).weight
    assert w[(wet_regime, "ECMWF")] > 0.9  # ECMWF far better on wet days in this construction
    assert w[(1 - wet_regime, "GFS")] > 0.5  # GFS better (err 1 vs 2) on dry days
    assert set(t.lead_day) == {1, 2}  # separate weights per lead day
    assert {"regime", "lead_day", "model", "weight", "training_error", "sample_count"} <= set(t.columns)


def test_hierarchical_fallback(geo):
    ds, labels, regions, land, f = _regime_setup(geo)
    t = learn_regime_weights(ds, labels, regions, epsilon=0.1, min_samples=1000, min_points=10)
    assert (t.level == LEVEL_STATIC).all()  # too few samples everywhere -> static per lead day
    t2 = learn_regime_weights(ds, labels, regions, epsilon=0.1, min_samples=2, min_points=10**6)
    assert (t2.level == LEVEL_STATIC).all()
    t3 = learn_regime_weights(ds, labels, regions, epsilon=0.1, min_samples=2, min_points=10)
    assert set(t3.level) <= {LEVEL_REGION, LEVEL_INDIA}


def test_missing_model_fallback(geo):
    ds, labels, regions, land, f = _regime_setup(geo)
    ds["forecast"].loc[dict(model="ECMWF")] = np.nan
    t = learn_regime_weights(ds, labels, regions, epsilon=0.1, min_samples=2, min_points=10)
    assert (t[t.model == "GFS"].weight == 1.0).all() and (t[t.model == "ECMWF"].weight == 0.0).all()
    blend = apply_regime_blend(ds, labels, t, "regime", regions)
    xr.testing.assert_allclose(blend, ds["forecast"].sel(model="GFS", drop=True).transpose(*blend.dims))


def test_extreme_aware_is_data_driven(geo):
    ds, labels, regions, land, f = _regime_setup(geo)
    st = regime_statistics(ds, labels, f, regions, land)
    heavy = st[st.is_heavy_regime].regime.tolist()
    base = learn_regime_weights(ds, labels, regions, 0.1, min_samples=2, min_points=10)
    ext = learn_extreme_aware_weights(ds, labels, base, heavy, 64.5, min_samples=2)
    check_weights(ext)
    hv = ext[ext.regime.isin(heavy)]
    assert (hv.level == LEVEL_CSI).all()
    # weights proportional to training CSI, not forced to 1/0
    w = hv[(hv.lead_day == 1) & (hv.region == "North")].set_index("model").weight
    assert 0 < w["GFS"] < w["ECMWF"] < 1 or w["GFS"] == 0.0
    # non-heavy regimes keep the MAE-based regime weights exactly
    pd.testing.assert_series_equal(
        ext[~ext.regime.isin(heavy)].weight.reset_index(drop=True), base[~base.regime.isin(heavy)].weight.reset_index(drop=True))


def test_extreme_aware_keeps_mae_weights_when_csi_undefined(geo):
    ds, labels, regions, land, f = _regime_setup(geo, wet_days=[])  # no heavy rain at all
    base = learn_regime_weights(ds, labels, regions, 0.1, min_samples=2, min_points=10)
    ext = learn_extreme_aware_weights(ds, labels, base, [0, 1], 64.5, min_samples=2)
    np.testing.assert_allclose(ext.weight.to_numpy(), base.weight.to_numpy())
    assert LEVEL_CSI not in set(ext.level)


def test_adaptive_blend_calculation(geo):
    ds, labels, regions, land, f = _regime_setup(geo)
    t = learn_regime_weights(ds, labels, regions, 0.1, min_samples=2, min_points=10)
    blend = apply_regime_blend(ds, labels, t, "regime", regions)
    s = 0
    row = t[(t.regime == labels[s]) & (t.lead_day == int(ds.lead_day.values[s])) & (t.region == "Central")]
    w = row.set_index("model").weight
    point = regions.where(regions == "Central", drop=True)
    lat, lon = float(point.latitude[0]), float(point.longitude[0])
    exp = sum(w[m] * float(ds["forecast"].sel(model=m, latitude=lat, longitude=lon).isel(sample=s)) for m in ("GFS", "ECMWF"))
    assert float(blend.isel(sample=s).sel(latitude=lat, longitude=lon)) == pytest.approx(exp, rel=1e-5)


def test_weight_difference_diagnostic(geo):
    ds, labels, regions, land, f = _regime_setup(geo)
    t = learn_regime_weights(ds, labels, regions, 0.1, min_samples=2, min_points=10)
    static = static_weight_table(ds, 0.1)
    d = weight_differences(t, static)
    assert (d.spread_across_regimes > 0.2).any()
    assert {"static_weight", "spread_across_regimes", "max_abs_diff_from_static", "nearly_identical_across_regimes"} <= set(d.columns)
    same = t.copy()
    same["weight"] = same.groupby(["lead_day", "region", "model"]).weight.transform("mean")
    assert weight_differences(same, static).nearly_identical_across_regimes.all()


# --------------------------------------------------------------------------- leakage

def test_test_observations_change_nothing_fitted(geo, tmp_path):
    land, regions = geo
    ds = make_samples(n_days=16)
    feats = feats_for(ds, geo)
    fit_idx = np.flatnonzero(ds.valid_date.values <= "2025-07-12")
    test_idx = np.flatnonzero(ds.valid_date.values > "2025-07-12")
    cfg = Phase3Config(output_dir=tmp_path, min_samples=2, min_points=10)

    def run(dataset):
        fit = dataset.isel(sample=fit_idx)
        return fit_stage(fit, feats.loc[fit.sample.values], 2, cfg, regions, land)

    a = run(ds)
    tampered = ds.copy(deep=True)
    tampered["observation"][test_idx] = 500.0  # change ONLY test-period observations
    tampered["forecast"][:, test_idx] = 0.0  # and test forecasts
    b = run(tampered)
    np.testing.assert_allclose(a["model"].scaler_mean, b["model"].scaler_mean)
    np.testing.assert_allclose(a["model"].pca_components, b["model"].pca_components)
    np.testing.assert_allclose(a["model"].centers, b["model"].centers)
    np.testing.assert_array_equal(a["labels"], b["labels"])
    for key in ("regime", "extreme_aware", "static"):
        pd.testing.assert_frame_equal(a["weights"][key], b["weights"][key])


def test_validation_observations_affect_only_final_weights(geo, tmp_path):
    land, regions = geo
    ds = make_samples(n_days=16)
    feats = feats_for(ds, geo)
    train_idx = np.flatnonzero(ds.valid_date.values <= "2025-07-08")
    fitall_idx = np.flatnonzero(ds.valid_date.values <= "2025-07-12")
    val_idx = np.setdiff1d(fitall_idx, train_idx)
    cfg = Phase3Config(output_dir=tmp_path, min_samples=2, min_points=10)
    tampered = ds.copy(deep=True)
    tampered["observation"][val_idx] = tampered["observation"][val_idx] * 3 + 50

    dev_a = fit_stage(ds.isel(sample=train_idx), feats.iloc[train_idx], 2, cfg, regions, land)
    dev_b = fit_stage(tampered.isel(sample=train_idx), feats.iloc[train_idx], 2, cfg, regions, land)
    pd.testing.assert_frame_equal(dev_a["weights"]["regime"], dev_b["weights"]["regime"])  # dev stage untouched
    fin_a = fit_stage(ds.isel(sample=fitall_idx), feats.iloc[fitall_idx], 2, cfg, regions, land)
    fin_b = fit_stage(tampered.isel(sample=fitall_idx), feats.iloc[fitall_idx], 2, cfg, regions, land)
    np.testing.assert_allclose(fin_a["model"].centers, fin_b["model"].centers)  # clustering is forecast-only
    assert not fin_a["weights"]["regime"].weight.equals(fin_b["weights"]["regime"].weight)  # final weights use val obs


# --------------------------------------------------------------------------- metrics

def test_pod_far_corr_by_hand():
    f = np.array([70.0, 70.0, 0.0, 0.0, 80.0])
    o = np.array([70.0, 0.0, 70.0, 0.0, 65.0])
    c = metrics.categorical_scores(f, o, 64.5)
    assert c["pod"] == pytest.approx(2 / 3) and c["far"] == pytest.approx(1 / 3) and c["csi"] == pytest.approx(0.5)
    s = metrics.extended_score(xr.DataArray(f), xr.DataArray(o), [64.5])
    assert s["corr"] == pytest.approx(np.corrcoef(f, o)[0, 1])
    assert s["n_obs_heavy"] == 3
    assert s["mae_obs_heavy"] == pytest.approx((0 + 70 + 15) / 3)
    assert np.isnan(metrics.correlation(np.ones(5), o))


def test_daily_counts_and_csi_bootstrap():
    f = xr.DataArray(np.array([[70.0, 0.0], [70.0, 70.0]]), dims=("sample", "latitude"), coords={"sample": [0, 1]})
    o = xr.DataArray(np.array([[70.0, 70.0], [0.0, 70.0]]), dims=("sample", "latitude"), coords={"sample": [0, 1]})
    f = f.expand_dims(longitude=[0.0], axis=-1); o = o.expand_dims(longitude=[0.0], axis=-1)
    mask = xr.ones_like(o, dtype=bool)
    d = daily_counts(f, o, mask, np.array(["a", "b"]), 64.5)
    assert d.loc["a"].tolist() == [1, 0, 1] and d.loc["b"].tolist() == [1, 1, 0]
    same = bootstrap_csi_difference(d, d, 100, 0)
    assert same["csi_difference"] == 0 and same["ci95_low"] == 0 and same["ci95_high"] == 0
