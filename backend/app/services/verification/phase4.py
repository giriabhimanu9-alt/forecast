"""Phase 4: multi-season validation (Experiment A, IMD) and AI/NWP benchmark (Experiment B, ERA5).

Experiment A  India operational: GFS 0.25 + ECMWF IFS (0.4 deg 2023, 0.25 deg 2024-25) vs IMD,
              JJAS 2023/2024/2025, IMD windows (03 UTC -> 03 UTC), train 2023 / val 2024 / test 2025.
Experiment B  AI/NWP benchmark: GraphCast (AI) + IFS HRES vs ERA5, WeatherBench 2 1.5 deg,
              JJAS 2018/2020/2022, windows 00 UTC -> 00 UTC, train 2018 / val 2020 / test 2022.

The two experiments never share a table: B's truth is ERA5, A's is IMD, and no AI
model can be aligned exactly with IMD's 03 UTC day (PHASE4_DATA_AUDIT.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime

from app.config import Settings, TargetGrid
from app.services.data.gfs import GFSAtmosProvider
from app.services.data.observations import IMDRainfallProvider
from app.services.http import HttpClient
from app.services.pipeline import run_forecast_provider
from app.services.verification.accumulation import required_lead_hours, verification_windows
from app.services.verification.experiment import (
    ECMWF_HISTORICAL_URL,
    ExperimentConfig,
    fetch_forecasts,
    plan_init_times,
)

log = logging.getLogger(__name__)

ATMOS_VARIABLES = ("mslp", "u_850hpa", "precipitable_water")


@dataclass
class ExperimentAConfig:
    seasons: tuple[int, ...] = (2023, 2024, 2025)
    season_months: tuple[int, int] = (6, 9)  # JJAS
    train_seasons: tuple[int, ...] = (2023,)
    validation_seasons: tuple[int, ...] = (2024,)
    test_seasons: tuple[int, ...] = (2025,)
    models: tuple[str, ...] = ("GFS", "ECMWF")
    lead_days: tuple[int, ...] = (1, 2, 3)
    ecmwf_base_url: str = ECMWF_HISTORICAL_URL
    http_retries: int = 8
    http_min_interval: float = 0.25

    def season_config(self, season: int, output_dir=None) -> ExperimentConfig:
        m0, m1 = self.season_months
        end = date(season, m1 + 1, 1) if m1 < 12 else date(season + 1, 1, 1)
        return ExperimentConfig(
            name=f"expA_{season}", valid_start=date(season, m0, 1),
            valid_end=date.fromordinal(end.toordinal() - 1), output_dir=output_dir,
            cycles=(0,), lead_days=self.lead_days, models=self.models, ecmwf_base_url=self.ecmwf_base_url,
            imd_source="archive", http_retries=self.http_retries, http_min_interval=self.http_min_interval,
        )


def season_settings(settings: Settings, cfg: ExperimentAConfig, variables=("precipitation",), leads=()) -> Settings:
    return replace(settings, variables=tuple(variables), lead_hours=tuple(leads), ecmwf_base_url=cfg.ecmwf_base_url,
                   http_retries=cfg.http_retries, forecast_init_time=None)


def fetch_atmos(settings: Settings, cfg: ExperimentAConfig, init_times: list[datetime],
                http: HttpClient | None = None) -> tuple[dict, list[dict]]:
    """GFS 1.0 deg MSLP / u850 / PWAT for regime features (not a precipitation member)."""
    out, records = {}, []
    http = http or HttpClient(settings.http_timeout, cfg.http_retries)
    for init in init_times:
        leads = required_lead_hours(verification_windows(init, cfg.lead_days))
        s = season_settings(settings, cfg, ATMOS_VARIABLES, leads)
        res = run_forecast_provider(GFSAtmosProvider(s, http=http), s, init)
        records.append({"model": GFSAtmosProvider.name, "init_time": init.isoformat(), "status": res.status.value,
                        "message": res.message, "downloads": res.downloads, "cache_hits": res.cache_hits})
        if res.ok and res.dataset is not None:
            out[init] = res.dataset.load()
    return out, records


def prefetch_experiment_a(settings: Settings, cfg: ExperimentAConfig, source: str) -> list[str]:
    lines = []
    for season in cfg.seasons:
        scfg = cfg.season_config(season)
        inits = plan_init_times(scfg, "archive")
        if source in ("GFS", "ECMWF"):
            one = replace(scfg, models=(source,))
            _, records = fetch_forecasts(settings, one, inits)
        elif source == "GFS_ATMOS":
            _, records = fetch_atmos(settings, cfg, inits)
        else:  # IMD
            p = IMDRainfallProvider(replace(settings, imd_source="archive", imd_start_date=scfg.valid_start,
                                            imd_end_date=scfg.valid_end))
            st = p.fetch()
            lines.append(f"{season} IMD: {len(st.files)} file(s), {st.downloads} downloaded")
            continue
        ok = sum(r["status"] == "AVAILABLE" for r in records)
        failed = [r["init_time"] for r in records if r["status"] != "AVAILABLE"]
        dl = sum(r["downloads"] for r in records)
        lines.append(f"{season} {source}: {ok}/{len(records)} cycles available, {dl} messages downloaded, failed: {failed}")
        log.info(lines[-1])
    return lines


# =========================================================================== shared evaluation engine

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from app.services.verification import plots  # noqa: E402
from app.services.verification.adaptive import LEVEL_STATIC  # noqa: E402
from app.services.verification.experiment import (  # noqa: E402
    bootstrap_mae_difference,
    daily_abs_error,
    observation_convention_check,
    observation_period_start,
)
from app.services.verification.alignment import build_aligned_dataset  # noqa: E402
from app.services.verification.blending import apply_blend, equal_weights  # noqa: E402
from app.services.verification.metrics import extended_score  # noqa: E402
from app.services.verification.phase3 import (  # noqa: E402
    _verdict,
    bootstrap_csi_difference,
    candidates_for,
    daily_counts,
    fit_stage,
    score_table,
)
from app.services.verification.regimes import RegimeModel, regime_features, select_k  # noqa: E402
from app.services.verification.regions import region_map  # noqa: E402
from app.services.verification.split import assert_no_leakage  # noqa: E402

HEAVY_MM = 64.5
MEANINGFUL_MAE_PCT = 1.0  # |dMAE| must be >= 1 % of the reference MAE
MEANINGFUL_CSI = 0.005  # |dCSI| at 64.5 mm must be >= 0.005


def assess(verdict: str, effect: float, threshold: float, lower_is_better: bool) -> str:
    """Combine the bootstrap verdict with a practical-significance threshold."""
    if verdict.startswith("undefined"):
        return "undefined (no events)"
    if verdict == "not distinguishable":
        return "not distinguishable"
    if not np.isfinite(effect) or abs(effect) < threshold:
        return f"negligible (CI excludes 0 but |effect| < {threshold:g}{'%' if lower_is_better else ''})"
    return "meaningful improvement" if verdict.startswith("better") else "meaningful degradation"


@dataclass
class BlendConfig:
    """Hyper-parameters fixed before looking at validation/test (same values as Phase 3)."""

    epsilon_mm: float = 0.1
    min_samples: int = 3
    min_points: int = 500
    seed: int = 20250701
    ks: tuple[int, ...] = (2, 3, 4)
    bootstrap_samples: int = 2000
    thresholds_mm: tuple[float, ...] = (2.5, 15.6, 64.5)


def _mask(ds: xr.Dataset, models: list[str]) -> xr.DataArray:
    mask = ds["observation"].notnull()
    for m in models:
        mask = mask & ds["forecast"].sel(model=m).notnull()
    return mask


def _by_season(ds: xr.Dataset, seasons) -> xr.Dataset:
    return ds.isel(sample=np.flatnonzero(np.isin(ds.season.values, list(seasons))))


def concat_seasons(parts: list[xr.Dataset]) -> xr.Dataset:
    ds = xr.concat(parts, dim="sample")
    return ds.assign_coords(sample=np.arange(ds.sizes["sample"]))


def evaluate(
    aligned: xr.Dataset,
    feature_sets: dict[str, pd.DataFrame],
    train_seasons, val_seasons, test_seasons,
    bcfg: BlendConfig, land: xr.DataArray, regions: xr.DataArray,
    out: Path, prefix: str,
) -> dict[str, Any]:
    """Development (train -> validation) then final (train+validation -> test, once). Writes CSVs + figures."""
    models = aligned.model.values.tolist()
    train = _by_season(aligned, train_seasons)
    val = _by_season(aligned, val_seasons)
    fit = _by_season(aligned, list(train_seasons) + list(val_seasons))
    test = _by_season(aligned, test_seasons)
    assert_no_leakage(train, val)
    assert_no_leakage(fit, test)
    res: dict[str, Any] = {"models": models}

    # ---- individual model + equal-blend skill for EVERY season (no training involved)
    rows = []
    for season in sorted(set(aligned.season.values.tolist())):
        s = _by_season(aligned, [season])
        m = _mask(s, models)
        cands = {x: s["forecast"].sel(model=x, drop=True) for x in models}
        cands["equal"] = apply_blend(s["forecast"], xr.DataArray(list(equal_weights(models).values()), dims="model",
                                                                 coords={"model": models}))
        t = score_table(s, cands, m, bcfg.thresholds_mm, f"season {season}")
        t.insert(0, "season", season)
        rows.append(t)
    by_season = pd.concat(rows)
    by_season.to_csv(out / "model_scores" / f"{prefix}_scores_by_season.csv", index=False, float_format="%.5f")

    # ---- development: per feature set choose k on TRAIN, evaluate on VALIDATION
    dev_rows, k_tables, dev = [], [], {}
    for fs_name, feats in feature_sets.items():
        f_train, f_val = feats.loc[train.sample.values], feats.loc[val.sample.values]
        k, kt = select_k(f_train, bcfg.ks, seed=bcfg.seed)
        kt.insert(0, "feature_set", fs_name)
        k_tables.append(kt)
        st = fit_stage(train, f_train, k, bcfg, regions, land)
        val_labels = st["model"].predict(f_val)
        vm = _mask(val, models)
        c = candidates_for(val, val_labels, st["weights"], regions)
        sc = score_table(val, c, vm, bcfg.thresholds_mm, "validation")
        sc.insert(0, "feature_set", fs_name)
        dev_rows.append(sc)
        allrow = sc[(sc.method == "regime") & (sc.lead_day == "all") & (sc.regime.astype(str) == "all")].iloc[0]
        dev[fs_name] = {"k": k, "val_mae_regime": float(allrow.mae),
                        "silhouette": float(kt[kt.selected].silhouette.iloc[0]),
                        "stability_ari": float(kt[kt.selected].stability_ari.iloc[0])}
    pd.concat(k_tables).to_csv(out / "regime_analysis" / f"{prefix}_k_selection.csv", index=False, float_format="%.4f")
    val_table = pd.concat(dev_rows)
    val_table.to_csv(out / "verification" / f"{prefix}_validation_comparison.csv", index=False, float_format="%.5f")
    # pre-registered rule: lowest validation MAE of the regime blend; ties (< 0.001 mm) -> fewer features
    order = sorted(dev, key=lambda n: (round(dev[n]["val_mae_regime"], 3), len(feature_sets[n].columns)))
    chosen = order[0]
    res["development"] = dev
    res["chosen_feature_set"] = chosen

    # ---- final: refit on TRAIN+VALIDATION, evaluate once on TEST
    finals, weights_all, stats_all, assign_all = {}, [], [], []
    for fs_name, feats in feature_sets.items():
        k = dev[fs_name]["k"]
        f_fit, f_test = feats.loc[fit.sample.values], feats.loc[test.sample.values]
        st = fit_stage(fit, f_fit, k, bcfg, regions, land)
        st["test_labels"] = st["model"].predict(f_test)
        finals[fs_name] = st
        st["model"].save(out / "regime_analysis" / f"{prefix}_regime_model_{fs_name}.json",
                         extra={"feature_set": fs_name, "fit_seasons": list(train_seasons) + list(val_seasons),
                                "regime_names": {str(r): n for r, n in st["names"].items()}})
        for key in ("regime", "extreme_aware"):
            w = st["weights"][key].copy()
            w.insert(0, "feature_set", fs_name)
            weights_all.append(w)
        s = st["stats"].copy()
        s.insert(0, "feature_set", fs_name)
        stats_all.append(s)
        for period, d, lab in (("fit", fit, st["labels"]), ("test", test, st["test_labels"])):
            a = pd.DataFrame({"feature_set": fs_name, "period": period, "sample": d.sample.values,
                              "season": d.season.values, "valid_date": d.valid_date.values,
                              "lead_day": d.lead_day.values, "regime": lab})
            a["regime_name"] = a.regime.map(st["names"])
            assign_all.append(a)
    pd.concat(stats_all).to_csv(out / "regime_analysis" / f"{prefix}_regime_statistics.csv", index=False, float_format="%.5f")
    pd.concat(assign_all).to_csv(out / "regime_analysis" / f"{prefix}_regime_assignments.csv", index=False)
    weights_df = pd.concat(weights_all)
    weights_df.to_csv(out / "weights" / f"{prefix}_adaptive_weights.csv", index=False, float_format="%.6f")
    primary = finals[chosen]
    primary["weights"]["static"].to_csv(out / "weights" / f"{prefix}_static_inverse_error_weights.csv", index=False,
                                        float_format="%.6f")

    # test candidates: individual, equal, static inverse-error, chosen regime blend (+ extreme-aware),
    # and the non-chosen regime feature set reported as secondary
    tm = _mask(test, models)
    cands = candidates_for(test, primary["test_labels"], primary["weights"], regions)
    for fs_name, st in finals.items():
        if fs_name != chosen:
            other = candidates_for(test, st["test_labels"], st["weights"], regions)
            cands[f"regime[{fs_name}]"] = other["regime"]
    final = score_table(test, cands, tm, bcfg.thresholds_mm, "test", primary["test_labels"])
    final.to_csv(out / "verification" / f"{prefix}_final_comparison.csv", index=False, float_format="%.5f")
    heavy_cols = ["period", "regime", "lead_day", "method", "n_samples", "csi@64.5mm", "pod@64.5mm", "far@64.5mm",
                  "hits@64.5mm", "false_alarms@64.5mm", "misses@64.5mm", "observed_events@64.5mm",
                  "n_obs_heavy", "mae_obs_heavy", "bias_obs_heavy"]
    final[[c for c in heavy_cols if c in final.columns]].to_csv(
        out / "verification" / f"{prefix}_heavy_rain_scores.csv", index=False, float_format="%.5f")

    # per-regime model skill on the fit period (the numbers the weights come from)
    fm = _mask(fit, models)
    fit_c = {x: fit["forecast"].sel(model=x, drop=True) for x in models}
    score_table(fit, fit_c, fm, bcfg.thresholds_mm, "fit", primary["labels"]).to_csv(
        out / "model_scores" / f"{prefix}_model_scores_by_regime_fit.csv", index=False, float_format="%.5f")

    # uncertainty: paired bootstrap over test valid days (all lead days of a day resampled together)
    dates = test.valid_date.values
    obs_t = test["observation"]
    dmae = {n: daily_abs_error(c, obs_t, tm, dates) for n, c in cands.items()}
    dcnt = {n: daily_counts(c, obs_t, tm, dates, HEAVY_MM) for n, c in cands.items()}
    ind = final[(final.lead_day == "all") & (final.regime.astype(str) == "all") & (final.method.isin(models))]
    best = ind.loc[ind.mae.idxmin()].method
    best_heavy = ind.loc[ind["csi@64.5mm"].idxmax()].method if ind["csi@64.5mm"].notna().any() else best
    unc = []
    for a in [c for c in cands if c not in models]:
        for b in dict.fromkeys([*models, "equal", "phase2_inverse_error"]):
            if a == b:
                continue
            m = bootstrap_mae_difference(dmae[a], dmae[b], bcfg.bootstrap_samples, bcfg.seed)
            c = bootstrap_csi_difference(dcnt[a], dcnt[b], bcfg.bootstrap_samples, bcfg.seed)
            unc.append({"method": a, "versus": b, "test_days": m["n_days"], "mae_difference": m["mae_difference"],
                        "mae_ci95_low": m["ci95_low"], "mae_ci95_high": m["ci95_high"],
                        "mae_verdict": _verdict(m["ci95_low"], m["ci95_high"], True),
                        "heavy_csi_difference": c["csi_difference"], "csi_ci95_low": c["ci95_low"],
                        "csi_ci95_high": c["ci95_high"], "heavy_csi_verdict": _verdict(c["ci95_low"], c["ci95_high"], False)})
    uncertainty = pd.DataFrame(unc)
    # Practical significance (Step 19): with ~10^6 correlated grid points, a CI can exclude 0 for a
    # negligible effect. A difference is "meaningful" only if the day-bootstrap CI excludes 0 AND
    # |dMAE| >= 1 % of the comparison MAE (or |dCSI| >= 0.005).
    ref_mae = {m: float(final[(final.method == m) & (final.lead_day == "all") & (final.regime.astype(str) == "all")].mae.iloc[0])
               for m in cands}
    uncertainty["mae_difference_pct"] = 100 * uncertainty.mae_difference / uncertainty.versus.map(ref_mae)
    uncertainty["mae_assessment"] = [assess(v, pct, MEANINGFUL_MAE_PCT, lower_is_better=True)
                                     for v, pct in zip(uncertainty.mae_verdict, uncertainty.mae_difference_pct)]
    uncertainty["heavy_csi_assessment"] = [assess(v, d, MEANINGFUL_CSI, lower_is_better=False)
                                           for v, d in zip(uncertainty.heavy_csi_verdict, uncertainty.heavy_csi_difference)]
    uncertainty.to_csv(out / "verification" / f"{prefix}_uncertainty.csv", index=False, float_format="%.5f")

    # explanations for every test forecast of the chosen adaptive blend
    wt = primary["weights"]["regime"]
    with open(out / "verification" / f"{prefix}_blend_explanations.jsonl", "w", encoding="utf-8") as fh:
        for i in range(test.sizes["sample"]):
            r, d = int(primary["test_labels"][i]), int(test.lead_day.values[i])
            rows_ = wt[(wt.regime == r) & (wt.lead_day == d)]
            avail = [x for x in models if bool(test["forecast"].sel(model=x).isel(sample=i).notnull().any())]
            rec = {
                "valid_date": str(test.valid_date.values[i]), "season": int(test.season.values[i]), "lead_day": d,
                "init_time": str(pd.Timestamp(test.init_time.values[i])), "regime": r,
                "regime_name": primary["names"].get(r), "feature_set": chosen,
                "weights_all_india_mean": {x: round(float(rows_[rows_.model == x].weight.mean()), 4) for x in models},
                "weights_by_region": {reg: {x: round(float(g[g.model == x].weight.iloc[0]), 4) for x in models}
                                      for reg, g in rows_.groupby("region")},
                "training_error_mae_mm": {x: round(float(rows_[rows_.model == x].training_error.mean()), 4) for x in models},
                "sample_count": int(rows_.sample_count.iloc[0]),
                "weight_level": sorted(set(rows_.level)),
                "available_models": avail, "missing_models": [x for x in models if x not in avail],
                "fallback_used": bool(set(rows_.level) & {LEVEL_STATIC}) or len(avail) < len(models),
            }
            fh.write(json.dumps(rec) + "\n")

    # figures
    names = primary["names"]
    feats_chosen = feature_sets[chosen]
    assign = pd.concat([pd.DataFrame({"period": "fit (train+validation)", "sample": fit.sample.values, "regime": primary["labels"]}),
                        pd.DataFrame({"period": "test", "sample": test.sample.values, "regime": primary["test_labels"]})])
    plots.phase3_regime_distribution(assign, primary["model"], feats_chosen, out / "figures" / f"{prefix}_regime_distribution.png", names)
    order_m = models + ["equal", "phase2_inverse_error", "regime", "extreme_aware"] + [c for c in cands if c.startswith("regime[")]
    plots.phase3_method_comparison(final, [m for m in order_m if m in cands], out / "figures" / f"{prefix}_final_comparison.png",
                                   f"{prefix}: test MAE / RMSE ({', '.join(map(str, test_seasons))})")
    plots.phase3_heavy_rain(final, [m for m in order_m if m in cands], out / "figures" / f"{prefix}_heavy_rain.png",
                            f"{prefix}: heavy rain >= {HEAVY_MM:g} mm/24 h, test {', '.join(map(str, test_seasons))}")
    plots.phase3_weights_by_regime(wt, models, names, out / "figures" / f"{prefix}_adaptive_weights.png")
    plots.phase4_scores_by_season(by_season, models + ["equal"], out / "figures" / f"{prefix}_scores_by_season.png", prefix)

    res.update({
        "final": final, "uncertainty": uncertainty, "by_season": by_season, "best_individual": best,
        "best_individual_heavy": best_heavy, "regime_names": names, "regime_stats": primary["stats"],
        "test_regime_counts": {int(r): int(c) for r, c in zip(*np.unique(primary["test_labels"], return_counts=True))},
        "weights": wt, "static_weights": primary["weights"]["static"],
        "n_test_samples": int(test.sizes["sample"]), "n_test_days": len(set(dates.tolist())),
        "n_test_points": int(tm.sum()), "n_test_heavy": int(((obs_t >= HEAVY_MM) & tm).sum()),
        "periods": {name: {"seasons": list(s), "n_samples": int(_by_season(aligned, s).sizes["sample"])}
                    for name, s in (("train", train_seasons), ("validation", val_seasons), ("test", test_seasons))},
    })
    return res


# =========================================================================== Experiment A (IMD)


ATMOS_BOXES = {
    # name: (variable, lat range, lon range, land_only)
    "mslp_monsoon_trough": ("mslp", (20.0, 28.0), (74.0, 88.0), False),
    "u850_arabian_sea_jet": ("u_850hpa", (8.0, 20.0), (68.0, 74.0), False),
    "pwat_india_land": ("precipitable_water", None, None, True),
}


def atmos_features(aligned: xr.Dataset, atmos: dict, land: xr.DataArray) -> pd.DataFrame:
    """Per sample: box means of GFS 1.0 deg fields, averaged over the window's start and end leads."""
    rows = {k: [] for k in ATMOS_BOXES}
    missing = []
    for s in range(aligned.sizes["sample"]):
        init = pd.Timestamp(aligned.init_time.values[s]).to_pydatetime()
        ds = atmos.get(init)
        if ds is None:
            missing.append(str(init))
            for k in rows:
                rows[k].append(np.nan)
            continue
        leads = [np.timedelta64(int(aligned.start_lead_h.values[s]), "h"), np.timedelta64(int(aligned.end_lead_h.values[s]), "h")]
        for name, (var, la, lo, land_only) in ATMOS_BOXES.items():
            f = ds[var].isel(model=0, time=0).sel(lead_time=leads).mean("lead_time")
            if la:
                f = f.sel(latitude=slice(*la), longitude=slice(*lo))
            if land_only:
                f = f.where(land)
            rows[name].append(float(f.mean()))
    if missing:
        raise ValueError(f"GFS 1.0 deg atmospheric fields missing for {len(missing)} samples, e.g. {missing[:3]}")
    return pd.DataFrame(rows, index=pd.Index(aligned.sample.values, name="sample"))


def run_experiment_a(settings: Settings, cfg: ExperimentAConfig, bcfg: BlendConfig, out: Path,
                     http: HttpClient | None = None) -> dict[str, Any]:
    from app.services.verification.experiment import fetch_observations

    parts, manifest_rows, coverage, missing_rows, align_rows, grid_rows = [], [], [], [], [], []
    atmos_all: dict = {}
    for season in cfg.seasons:
        scfg = cfg.season_config(season, out)
        obs, obs_info = fetch_observations(settings, scfg, http)
        inits = plan_init_times(scfg, "archive")
        forecasts, records = fetch_forecasts(settings, scfg, inits, http)
        atmos, atmos_records = fetch_atmos(settings, cfg, inits, http)
        atmos_all.update(atmos)
        models = [m for m in cfg.models if forecasts[m]]
        targets = {observation_period_start(d.date(), "archive") for d in pd.date_range(scfg.valid_start, scfg.valid_end)}
        aligned, alog = build_aligned_dataset(forecasts, obs, inits, cfg.lead_days, list(cfg.models), window_starts=targets)
        aligned = aligned.assign_coords(season=("sample", np.full(aligned.sizes["sample"], season)))
        parts.append(aligned)
        conv = observation_convention_check(aligned, obs)
        for m in cfg.models:
            ok = [r for r in records if r["model"] == m and r["status"] == "AVAILABLE"]
            coverage.append({"experiment": "A", "season": season, "model": m, "cycles_expected": len(inits),
                             "cycles_available": len(ok), "coverage": len(ok) / len(inits),
                             "samples_aligned": int(aligned["forecast"].sel(model=m).notnull().any(("latitude", "longitude")).sum())})
            c = conv.get(m, {})
            align_rows.append({
                "experiment": "A", "season": season, "model": m, "samples": int(aligned.sizes["sample"]),
                "all_windows_24h": bool(((aligned.observation_period_end - aligned.observation_period_start)
                                         == np.timedelta64(24, "h")).all()),
                "all_windows_end_03utc": bool((pd.DatetimeIndex(aligned.observation_period_end.values).hour == 3).all()),
                "init_before_window": bool((aligned.init_time <= aligned.observation_period_start).all()),
                "pattern_corr_label_offsets": json.dumps(c.get("mean_pattern_correlation_by_label_offset_days")),
                "best_label_offset_days": c.get("best_offset_days"),
                "imd_convention_passes": c.get("best_offset_days") == "0",
            })
        ok_atmos = sum(r["status"] == "AVAILABLE" for r in atmos_records)
        coverage.append({"experiment": "A", "season": season, "model": "GFS 1.0 deg (regime features)",
                         "cycles_expected": len(inits), "cycles_available": ok_atmos, "coverage": ok_atmos / len(inits),
                         "samples_aligned": None})
        for r in records + atmos_records:
            if r["status"] != "AVAILABLE":
                missing_rows.append({"experiment": "A", "season": season, "kind": "cycle", **r})
        for w in alog.missing_forecast:
            missing_rows.append({"experiment": "A", "season": season, "kind": "forecast window", "detail": w})
        for w in alog.missing_observation:
            missing_rows.append({"experiment": "A", "season": season, "kind": "observation window", "detail": w})
        ec = ECMWFProvider_resolution(scfg.valid_start)
        grid_rows += [
            {"experiment": "A", "season": season, "model": "GFS", "native_grid": "0.25 deg regular lat/lon",
             "target_grid": "0.25 deg, 5-38N 68-98E (133x121)", "method": "exact-subset (aligned grids)",
             "ocean_missing": "scored only on IMD land (IMD NaN over ocean)"},
            {"experiment": "A", "season": season, "model": "ECMWF", "native_grid": f"{ec} deg regular lat/lon",
             "target_grid": "0.25 deg, 5-38N 68-98E (133x121)",
             "method": "bilinear interpolation" if ec != 0.25 else "exact-subset (aligned grids)",
             "ocean_missing": "scored only on IMD land"},
            {"experiment": "A", "season": season, "model": "GFS_ATMOS_1P00", "native_grid": "1.0 deg",
             "target_grid": "0.25 deg (then box means)", "method": "bilinear interpolation", "ocean_missing": "n/a"},
            {"experiment": "A", "season": season, "model": "IMD", "native_grid": "0.25 deg (6.5-38.5N, 66.5-100E)",
             "target_grid": "0.25 deg", "method": "exact-subset", "ocean_missing": "NaN outside land mask (by design)"},
        ]
        manifest_rows.append({
            "experiment": "A", "season": season,
            "forecast_start": inits[0].isoformat(), "forecast_end": inits[-1].isoformat(),
            "observation_start": str(scfg.valid_start), "observation_end": str(scfg.valid_end),
            "forecast_models": ",".join(models), "observation_source": "IMD gridded rainfall archive (0.25 deg)",
            "missing_days": len(alog.missing_observation),
            "missing_cycles": sum(r["status"] != "AVAILABLE" for r in records),
            "samples": int(aligned.sizes["sample"]),
        })
        save_aligned(aligned, out / "datasets" / f"expA_aligned_{season}.nc")
    aligned = concat_seasons(parts)
    land = aligned["observation"].isel(sample=np.flatnonzero(np.isin(aligned.season.values, cfg.train_seasons))).notnull().any("sample")
    regions = region_map(aligned.latitude.values, aligned.longitude.values)
    rain = regime_features(aligned["forecast"], land, regions)
    atm = atmos_features(aligned, atmos_all, land)
    feature_sets = {"rain": rain, "atmos+rain": pd.concat([rain, atm], axis=1)}
    res = evaluate(aligned, feature_sets, cfg.train_seasons, cfg.validation_seasons, cfg.test_seasons, bcfg, land, regions, out, "expA")
    res.update({"dataset_manifest": manifest_rows, "coverage": coverage, "missing": missing_rows,
                "alignment_checks": align_rows, "grid_info": grid_rows, "land": land,
                "feature_sets": {k: list(v.columns) for k, v in feature_sets.items()}})
    return res


def ECMWFProvider_resolution(day: date) -> float:
    from app.services.data.ecmwf import ECMWFProvider

    return ECMWFProvider.resolution_deg(ECMWFProvider.__new__(ECMWFProvider), datetime(day.year, day.month, day.day))


def save_aligned(ds: xr.Dataset, path: Path) -> None:
    from app.services.pipeline import save_netcdf

    save_netcdf(ds, path)


# =========================================================================== Experiment B (ERA5 benchmark)


@dataclass
class ExperimentBConfig:
    seasons: tuple[int, ...] = (2018, 2020, 2022)
    train_seasons: tuple[int, ...] = (2018,)
    validation_seasons: tuple[int, ...] = (2020,)
    test_seasons: tuple[int, ...] = (2022,)
    lead_days: tuple[int, ...] = (1, 2, 3)
    grid: TargetGrid = field(default_factory=lambda: TargetGrid(6.0, 37.5, 67.5, 97.5, 1.5, name="India (WB2 1.5 deg)"))
    land_fraction_threshold: float = 0.5


def coarse_land_mask(land_025: xr.DataArray, grid: TargetGrid, threshold: float) -> xr.DataArray:
    """1.5 deg cell counts as India land if >= threshold of its 0.25 deg sub-cells are IMD land."""
    half = grid.resolution / 2
    vals = np.zeros(grid.shape)
    for i, la in enumerate(grid.latitudes):
        for j, lo in enumerate(grid.longitudes):
            sub = land_025.sel(latitude=slice(la - half, la + half - 1e-6), longitude=slice(lo - half, lo + half - 1e-6))
            vals[i, j] = float(sub.mean()) if sub.size else 0.0
    frac = xr.DataArray(vals, dims=("latitude", "longitude"), coords={"latitude": grid.latitudes, "longitude": grid.longitudes})
    return frac >= threshold


def run_experiment_b(settings: Settings, cfg: ExperimentBConfig, bcfg: BlendConfig, land_025: xr.DataArray,
                     out: Path) -> dict[str, Any]:
    from app.services.data.wb2 import GraphCastWB2Provider, HRESWB2Provider, era5_reference

    s = replace(settings, grid=cfg.grid, variables=("precipitation",), forecast_init_time=None)
    land = coarse_land_mask(land_025, cfg.grid, cfg.land_fraction_threshold)
    regions = region_map(cfg.grid.latitudes, cfg.grid.longitudes)
    providers = [GraphCastWB2Provider(s), HRESWB2Provider(s)]
    models = [p.name for p in providers]
    parts, manifest_rows, coverage, missing_rows, align_rows = [], [], [], [], []
    for season in cfg.seasons:
        ends = pd.date_range(f"{season}-06-02", f"{season}-10-01", freq="D")  # windows (E-24h, E], E at 00 UTC
        inits = sorted({(e - pd.Timedelta(days=k)).to_pydatetime() for e in ends for k in cfg.lead_days})
        leads = [24 * k for k in cfg.lead_days]
        forecasts: dict = {m: {} for m in models}
        for p in providers:
            p.fetch_many(inits, leads, batch=16)
            ss = replace(s, lead_hours=tuple(leads))
            for init in inits:
                r = run_forecast_provider(p, ss, init)
                if r.ok:
                    forecasts[p.name][init] = r.dataset[["precipitation"]].load()
                else:
                    missing_rows.append({"experiment": "B", "season": season, "kind": "cycle", "model": p.name,
                                         "init_time": init.isoformat(), "status": r.status.value, "message": r.message})
            coverage.append({"experiment": "B", "season": season, "model": p.name, "cycles_expected": len(inits),
                             "cycles_available": len(forecasts[p.name]), "coverage": len(forecasts[p.name]) / len(inits)})
        obs = era5_reference(s, list(ends), cfg.grid, f"era5_{season}")
        obs["precipitation"] = obs["precipitation"].where(land)  # score over India land cells only
        targets = {(e - pd.Timedelta(hours=24)).to_pydatetime() for e in ends}
        aligned, alog = build_aligned_dataset(forecasts, obs, inits, cfg.lead_days, models, window_starts=targets,
                                              window_end_hour_utc=0)
        aligned = aligned.assign_coords(season=("sample", np.full(aligned.sizes["sample"], season)))
        parts.append(aligned)
        for row in coverage:
            if row["experiment"] == "B" and row["season"] == season:
                row["samples_aligned"] = int(aligned["forecast"].sel(model=row["model"]).notnull().any(("latitude", "longitude")).sum())
        align_rows.append({
            "experiment": "B", "season": season, "model": ",".join(models), "samples": int(aligned.sizes["sample"]),
            "all_windows_24h": bool(((aligned.observation_period_end - aligned.observation_period_start) == np.timedelta64(24, "h")).all()),
            "all_windows_end_00utc": bool((pd.DatetimeIndex(aligned.observation_period_end.values).hour == 0).all()),
            "init_before_window": bool((aligned.init_time <= aligned.observation_period_start).all()),
        })
        for w in alog.missing_forecast + alog.missing_observation:
            missing_rows.append({"experiment": "B", "season": season, "kind": "window", "detail": w})
        manifest_rows.append({
            "experiment": "B", "season": season, "forecast_start": inits[0].isoformat(), "forecast_end": inits[-1].isoformat(),
            "observation_start": str(ends[0].date()), "observation_end": str(ends[-1].date()),
            "forecast_models": ",".join(models), "observation_source": "ERA5 (WeatherBench 2, 1.5 deg), India land cells",
            "missing_days": len(alog.missing_observation), "missing_cycles": sum(1 for r in missing_rows if r.get("season") == season and r["kind"] == "cycle"),
            "samples": int(aligned.sizes["sample"]),
        })
        save_aligned(aligned, out / "datasets" / f"expB_aligned_{season}.nc")
    aligned = concat_seasons(parts)
    feats = regime_features(aligned["forecast"], land, regions)
    b = replace(bcfg, min_points=30)  # 1.5 deg grid: ~150 India land cells, regions have tens of cells
    res = evaluate(aligned, {"rain": feats}, cfg.train_seasons, cfg.validation_seasons, cfg.test_seasons, b, land, regions, out, "expB")
    res.update({"dataset_manifest": manifest_rows, "coverage": coverage, "missing": missing_rows, "alignment_checks": align_rows,
                "grid_info": [{"experiment": "B", "season": ",".join(map(str, cfg.seasons)), "model": m,
                               "native_grid": "1.5 deg (WB2 240x121, conservative from 0.25 deg)",
                               "target_grid": "1.5 deg, 6-37.5N 67.5-97.5E (22x21)", "method": "exact-subset",
                               "ocean_missing": f"scored on 1.5 deg cells with >= {cfg.land_fraction_threshold:.0%} IMD land"}
                              for m in models + ["ERA5"]],
                "land_cells": int(land.sum())})
    return res
