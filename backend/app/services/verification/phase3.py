"""Phase 3: regime-conditioned adaptive blending, evaluated on the untouched Phase 2 test period.

Stages (chronological, no shuffling):
  development  fit on TRAIN (1-17 Jul), evaluate on VALIDATION (18-23 Jul): choose k, check methods
  final        refit on TRAIN+VAL (1-23 Jul, = Phase 2 training), evaluate ONCE on TEST (24-31 Jul)

Regimes come from forecast-time features only; observations are used for
weights and regime descriptions strictly inside the fit period.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from app.config import Settings
from app.models.schema import DataPipelineError
from app.services.http import HttpClient
from app.services.verification import plots
from app.services.verification.adaptive import (
    LEVEL_STATIC,
    apply_regime_blend,
    learn_extreme_aware_weights,
    learn_regime_weights,
    static_weight_table,
    weight_differences,
)
from app.services.verification.alignment import build_aligned_dataset
from app.services.verification.blending import apply_blend, equal_weights
from app.services.verification.experiment import (
    ECMWF_HISTORICAL_URL,
    ExperimentConfig,
    bootstrap_mae_difference,
    daily_abs_error,
    fetch_forecasts,
    fetch_observations,
    observation_period_start,
    plan_init_times,
)
from app.services.verification.metrics import extended_score
from app.services.verification.regimes import (
    RegimeModel,
    ocean_box_masks,
    regime_features,
    regime_statistics,
    select_k,
)
from app.services.verification.regions import region_map
from app.services.verification.split import assert_no_leakage

log = logging.getLogger(__name__)

HEAVY_MM = 64.5
METHOD_ORDER = ["GFS", "ECMWF", "equal", "phase2_inverse_error", "regime", "extreme_aware"]


@dataclass
class Phase3Config:
    output_dir: Path
    name: str = "phase3_july2025"
    valid_start: date = date(2025, 7, 1)
    train_end: date = date(2025, 7, 17)
    validation_end: date = date(2025, 7, 23)
    valid_end: date = date(2025, 7, 31)
    models: tuple[str, ...] = ("GFS", "ECMWF")
    lead_days: tuple[int, ...] = (1, 2, 3)
    ks: tuple[int, ...] = (2, 3, 4)
    epsilon_mm: float = 0.1
    min_samples: int = 3
    min_points: int = 500
    thresholds_mm: tuple[float, ...] = (2.5, 15.6, 64.5)
    bootstrap_samples: int = 2000
    seed: int = 20250701
    use_wind: bool = True
    ecmwf_base_url: str = ECMWF_HISTORICAL_URL
    imd_source: str = "archive"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("valid_start", "train_end", "validation_end", "valid_end"):
            d[k] = getattr(self, k).isoformat()
        d["output_dir"] = str(self.output_dir)
        return d

    def experiment_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name=self.name, valid_start=self.valid_start, valid_end=self.valid_end, output_dir=self.output_dir,
            cycles=(0,), lead_days=self.lead_days, models=self.models, ecmwf_base_url=self.ecmwf_base_url,
            imd_source=self.imd_source,
        )


@dataclass
class Phase3Result:
    status: str
    manifest: dict[str, Any]
    summary_lines: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers


def wind_features(aligned: xr.Dataset, forecasts: dict[str, dict[datetime, xr.Dataset]],
                  boxes: dict[str, xr.DataArray]) -> dict[str, np.ndarray] | None:
    """Per-sample ocean-box 10 m wind (mean of window start/end leads, mean of available models)."""
    out: dict[str, list[float]] = {name: [] for name in boxes}
    for s in range(aligned.sizes["sample"]):
        init = pd.Timestamp(aligned.init_time.values[s]).to_pydatetime()
        leads = [np.timedelta64(int(aligned.start_lead_h.values[s]), "h"), np.timedelta64(int(aligned.end_lead_h.values[s]), "h")]
        fields = []
        for model, per_init in forecasts.items():
            ds = per_init.get(init)
            if ds is None or "wind_speed_10m" not in ds:
                continue
            w = ds["wind_speed_10m"].isel(model=0, time=0)
            fields.append(w.sel(lead_time=[lt for lt in leads if lt in w.lead_time.values]).mean("lead_time"))
        if not fields:
            return None
        mean = xr.concat(fields, dim="m").mean("m")
        for name, mask in boxes.items():
            out[name].append(float(mean.where(mask).mean()))
    return {k: np.asarray(v) for k, v in out.items()}


def subset(ds: xr.Dataset, lo: date, hi: date) -> tuple[xr.Dataset, np.ndarray]:
    vd = ds.valid_date.values
    idx = np.flatnonzero((vd >= lo.isoformat()) & (vd <= hi.isoformat()))
    return ds.isel(sample=idx), idx


def static_blend(samples: xr.Dataset, table: pd.DataFrame) -> xr.DataArray:
    models = samples.model.values.tolist()
    parts, order = [], []
    for d in sorted(set(samples.lead_day.values.tolist())):
        idx = np.flatnonzero(samples.lead_day.values == d)
        w = table[table.lead_day == d].set_index("model").weight.reindex(models).fillna(0.0)
        parts.append(apply_blend(samples["forecast"].isel(sample=idx), xr.DataArray(w.values, dims="model", coords={"model": models})))
        order.extend(idx.tolist())
    blend = xr.concat(parts, dim="sample")
    return blend.isel(sample=np.argsort(order))


def candidates_for(samples: xr.Dataset, labels: np.ndarray, weights: dict[str, pd.DataFrame],
                   regions: xr.DataArray) -> dict[str, xr.DataArray]:
    models = samples.model.values.tolist()
    fc = samples["forecast"]
    out = {m: fc.sel(model=m, drop=True) for m in models}
    eq = xr.DataArray(list(equal_weights(models).values()), dims="model", coords={"model": models})
    out["equal"] = apply_blend(fc, eq)
    out["phase2_inverse_error"] = static_blend(samples, weights["static"])
    out["regime"] = apply_regime_blend(samples, labels, weights["regime"], "regime", regions)
    out["extreme_aware"] = apply_regime_blend(samples, labels, weights["extreme_aware"], "extreme_aware", regions)
    return out


def score_table(samples: xr.Dataset, cands: dict[str, xr.DataArray], mask: xr.DataArray, thresholds,
                period: str, labels: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    groups = [(str(d), np.flatnonzero(samples.lead_day.values == d)) for d in sorted(set(samples.lead_day.values.tolist()))]
    groups.append(("all", np.arange(samples.sizes["sample"])))
    regimes = ["all"] + (sorted(set(labels.tolist())) if labels is not None else [])
    for reg in regimes:
        for g, idx in groups:
            if reg != "all":
                idx = idx[labels[idx] == reg]
            if idx.size == 0:
                continue
            for name, fc in cands.items():
                s = extended_score(fc.isel(sample=idx).where(mask.isel(sample=idx)),
                                   samples["observation"].isel(sample=idx), thresholds, HEAVY_MM)
                rows.append({"period": period, "regime": reg, "lead_day": g, "method": name,
                             "n_samples": int(idx.size), **s})
    return pd.DataFrame(rows)


def daily_counts(fc: xr.DataArray, obs: xr.DataArray, mask: xr.DataArray, dates: np.ndarray, t: float) -> pd.DataFrame:
    f, o = fc.where(mask), obs.where(mask)
    fe, oe = (f >= t), (o >= t)
    df = pd.DataFrame({
        "valid_date": dates,
        "hits": (fe & oe).sum(("latitude", "longitude")).values,
        "fa": (fe & ~oe & o.notnull()).sum(("latitude", "longitude")).values,
        "miss": (~fe & oe & f.notnull()).sum(("latitude", "longitude")).values,
    })
    return df.groupby("valid_date")[["hits", "fa", "miss"]].sum()


def bootstrap_csi_difference(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    days = a.index.intersection(b.index)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), size=(n_boot, len(days)))

    def csi(df, ix):
        h, f, m = (df.loc[days, c].to_numpy()[ix].sum(-1) for c in ("hits", "fa", "miss"))
        den = h + f + m
        return np.where(den > 0, h / np.where(den > 0, den, 1), np.nan)

    full = np.arange(len(days))[None, :]
    diff = csi(a, idx) - csi(b, idx)
    point = float((csi(a, full) - csi(b, full))[0])
    if not np.isfinite(diff).any():  # CSI undefined (no events forecast or observed)
        return {"csi_difference": point, "ci95_low": float("nan"), "ci95_high": float("nan")}
    return {"csi_difference": point,
            "ci95_low": float(np.nanpercentile(diff, 2.5)), "ci95_high": float(np.nanpercentile(diff, 97.5))}


def _verdict(lo: float, hi: float, lower_is_better: bool) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined (no events)"
    if lower_is_better:
        return "better (95% CI < 0)" if hi < 0 else "worse (95% CI > 0)" if lo > 0 else "not distinguishable"
    return "better (95% CI > 0)" if lo > 0 else "worse (95% CI < 0)" if hi < 0 else "not distinguishable"


def fit_stage(fit: xr.Dataset, fit_features: pd.DataFrame, k: int, cfg: Phase3Config, regions, land) -> dict[str, Any]:
    """Fit regime model + all weight tables on one fit period (observations of that period only)."""
    model = RegimeModel.fit(fit_features, k, seed=cfg.seed)
    labels = model.predict(fit_features)
    stats = regime_statistics(fit, labels, fit_features, regions, land)
    names = dict(zip(stats.regime, stats.name))
    regime_w = learn_regime_weights(fit, labels, regions, cfg.epsilon_mm, cfg.min_samples, cfg.min_points, names)
    heavy = stats[stats.is_heavy_regime].regime.tolist()
    extreme_w = learn_extreme_aware_weights(fit, labels, regime_w, heavy, HEAVY_MM, cfg.min_samples)
    static_w = static_weight_table(fit, cfg.epsilon_mm, method="phase2_inverse_error")
    return {"model": model, "labels": labels, "stats": stats, "names": names, "heavy_regimes": heavy,
            "weights": {"regime": regime_w, "extreme_aware": extreme_w, "static": static_w}}


# --------------------------------------------------------------------------- main


def run_phase3(settings: Settings, cfg: Phase3Config, http: HttpClient | None = None) -> Phase3Result:
    started = datetime.now(timezone.utc)
    out = cfg.output_dir
    for sub_dir in ("regimes", "weights", "verification", "figures"):
        (out / sub_dir).mkdir(parents=True, exist_ok=True)
    ecfg = cfg.experiment_config()
    route = "archive" if cfg.imd_source == "auto" else cfg.imd_source
    manifest: dict[str, Any] = {"experiment": cfg.name, "phase": 3, "started_at": started.isoformat(timespec="seconds"),
                                "config": cfg.as_dict(), "region": settings.grid.as_dict(),
                                "variable": "precipitation", "units": "mm per 24 h"}

    # ---- data (reuses Phase 1/2 providers, cache, alignment) -------------
    try:
        observations, obs_info = fetch_observations(settings, ecfg, http)
    except DataPipelineError as exc:
        return _finish(out, manifest, "FAILED", [f"IMD observations unavailable: {exc}"])
    manifest["observations"] = obs_info
    init_times = plan_init_times(ecfg, route)
    variables = ("precipitation", "wind_speed_10m") if cfg.use_wind else ("precipitation",)
    forecasts, records = fetch_forecasts(settings, ecfg, init_times, http, variables=variables)
    manifest["forecast_cycles"] = [t.isoformat() for t in init_times]
    manifest["failed_downloads"] = [r for r in records if r["status"] != "AVAILABLE"]
    models = [m for m in cfg.models if forecasts[m]]
    manifest["models_available"] = models
    if not models:
        return _finish(out, manifest, "FAILED", ["No forecast model available."])
    targets = {observation_period_start(d.date(), route) for d in pd.date_range(cfg.valid_start, cfg.valid_end)}
    precip_only = {m: {i: ds[["precipitation"]] for i, ds in per.items()} for m, per in forecasts.items()}
    aligned, align_log = build_aligned_dataset(precip_only, observations, init_times, cfg.lead_days, models,
                                               window_starts=targets)
    manifest["alignment"] = {"aligned_samples": int(aligned.sizes["sample"]),
                             "missing_forecast_windows": align_log.missing_forecast,
                             "missing_observation_windows": align_log.missing_observation,
                             "grid": {"latitude": int(aligned.sizes["latitude"]), "longitude": int(aligned.sizes["longitude"])}}

    # ---- periods ----------------------------------------------------------
    train, _ = subset(aligned, cfg.valid_start, cfg.train_end)
    val, _ = subset(aligned, date.fromordinal(cfg.train_end.toordinal() + 1), cfg.validation_end)
    fitall, _ = subset(aligned, cfg.valid_start, cfg.validation_end)
    test, _ = subset(aligned, date.fromordinal(cfg.validation_end.toordinal() + 1), cfg.valid_end)
    assert_no_leakage(train, val)
    assert_no_leakage(fitall, test)
    periods = {name: {"valid_dates": [str(d.valid_date.values.min()), str(d.valid_date.values.max())],
                      "n_days": len(set(d.valid_date.values.tolist())), "n_samples": int(d.sizes["sample"])}
               for name, d in (("train", train), ("validation", val), ("train_plus_validation", fitall), ("test", test))}
    manifest["periods"] = periods

    # ---- static geography + features (forecast-only) ----------------------
    land = fitall["observation"].notnull().any("sample")  # IMD land mask (static; fit period only)
    regions = region_map(aligned.latitude.values, aligned.longitude.values)
    boxes = ocean_box_masks(aligned.latitude.values, aligned.longitude.values, land)
    wind = wind_features(aligned, forecasts, boxes) if cfg.use_wind else None
    feats = regime_features(aligned["forecast"], land, regions, wind)
    manifest["regime_features"] = list(feats.columns)
    manifest["regime_features_dropped_outside_domain"] = feats.attrs.get("dropped_features", [])
    manifest["regime_feature_source"] = "forecast fields only (mean of available models); no observation of the target window"
    manifest["wind_available"] = wind is not None
    f_train, f_val = feats.loc[train.sample.values], feats.loc[val.sample.values]
    f_fit, f_test = feats.loc[fitall.sample.values], feats.loc[test.sample.values]

    # ---- development: choose k on TRAIN, evaluate on VALIDATION -----------
    k, k_table = select_k(f_train, cfg.ks, seed=cfg.seed)
    k_table.to_csv(out / "regimes" / "k_selection.csv", index=False, float_format="%.4f")
    dev = fit_stage(train, f_train, k, cfg, regions, land)
    val_labels = dev["model"].predict(f_val)
    val_mask = _mask(val, models)
    val_scores = score_table(val, candidates_for(val, val_labels, dev["weights"], regions), val_mask,
                             cfg.thresholds_mm, "validation")
    val_scores.to_csv(out / "verification" / "validation_comparison.csv", index=False, float_format="%.5f")
    # diagnostic only (not used for selection): validation MAE for each candidate k
    k_diag = []
    for kk in k_table.k:
        st = fit_stage(train, f_train, int(kk), cfg, regions, land)
        c = candidates_for(val, st["model"].predict(f_val), st["weights"], regions)
        k_diag.append({"k": int(kk), "val_mae_regime": float(extended_score(c["regime"].where(val_mask), val["observation"], [])["mae"])})
    k_table.merge(pd.DataFrame(k_diag), on="k").to_csv(out / "regimes" / "k_selection.csv", index=False, float_format="%.4f")

    # ---- final: refit on TRAIN+VAL, evaluate once on TEST ------------------
    fin = fit_stage(fitall, f_fit, k, cfg, regions, land)
    fin["model"].save(out / "regimes" / "regime_model.json", extra={
        "fit_period": periods["train_plus_validation"]["valid_dates"],
        "regime_names": {str(r): n for r, n in fin["names"].items()},
        "feature_source": manifest["regime_feature_source"],
        "k_selection": k_table.drop(columns=["selection_note"]).to_dict(orient="records"),
    })
    test_labels = fin["model"].predict(f_test)
    assign = pd.concat([
        pd.DataFrame({"period": p, "sample": d.sample.values, "valid_date": d.valid_date.values,
                      "lead_day": d.lead_day.values, "init_time": d.init_time.values, "regime": lab})
        for p, d, lab in (("train+validation (fit)", fitall, fin["labels"]), ("test", test, test_labels))
    ])
    assign["regime_name"] = assign.regime.map(fin["names"])
    assign = assign.merge(feats.reset_index(), on="sample")
    assign.to_csv(out / "regimes" / "regime_assignments.csv", index=False, float_format="%.5f")
    fin["stats"].to_csv(out / "regimes" / "regime_statistics.csv", index=False, float_format="%.5f")

    # weights
    weights_all = pd.concat([fin["weights"]["regime"], fin["weights"]["extreme_aware"]])
    weights_all.to_csv(out / "weights" / "adaptive_weights.csv", index=False, float_format="%.6f")
    fin["weights"]["static"].to_csv(out / "weights" / "static_weights_phase2_method.csv", index=False, float_format="%.6f")
    diffs = weight_differences(weights_all, fin["weights"]["static"])
    diffs.to_csv(out / "weights" / "weight_difference_by_regime.csv", index=False, float_format="%.6f")

    # model performance by regime on the FIT period (these are the numbers the weights come from)
    fit_mask = _mask(fitall, models)
    fit_cands = {m: fitall["forecast"].sel(model=m, drop=True) for m in models}
    by_regime = score_table(fitall, fit_cands, fit_mask, cfg.thresholds_mm, "train+validation", fin["labels"])
    by_regime.to_csv(out / "verification" / "model_scores_by_regime.csv", index=False, float_format="%.5f")

    # final test comparison
    test_mask = _mask(test, models)
    cands = candidates_for(test, test_labels, fin["weights"], regions)
    final = score_table(test, cands, test_mask, cfg.thresholds_mm, "test", test_labels)
    final.to_csv(out / "verification" / "final_comparison.csv", index=False, float_format="%.5f")
    extreme = final[[c for c in final.columns if c in (
        "period", "regime", "lead_day", "method", "n_samples", "csi@64.5mm", "pod@64.5mm", "far@64.5mm",
        "hits@64.5mm", "false_alarms@64.5mm", "misses@64.5mm", "observed_events@64.5mm",
        "n_obs_heavy", "mae_obs_heavy", "bias_obs_heavy")]]
    extreme.to_csv(out / "verification" / "extreme_rain_scores.csv", index=False, float_format="%.5f")

    # uncertainty (paired day-block bootstrap over test days)
    dates = test.valid_date.values
    obs_t = test["observation"]
    daily_mae = {n: daily_abs_error(c, obs_t, test_mask, dates) for n, c in cands.items()}
    daily_hv = {n: daily_counts(c, obs_t, test_mask, dates, HEAVY_MM) for n, c in cands.items()}
    unc = []
    for a in ("regime", "extreme_aware"):
        for b in ("equal", "phase2_inverse_error", "GFS", "ECMWF"):
            if b not in cands:
                continue
            m = bootstrap_mae_difference(daily_mae[a], daily_mae[b], cfg.bootstrap_samples, cfg.seed)
            c = bootstrap_csi_difference(daily_hv[a], daily_hv[b], cfg.bootstrap_samples, cfg.seed)
            unc.append({"method": a, "versus": b, "test_days": m["n_days"],
                        "mae_difference": m["mae_difference"], "mae_ci95_low": m["ci95_low"], "mae_ci95_high": m["ci95_high"],
                        "mae_verdict": _verdict(m["ci95_low"], m["ci95_high"], True),
                        "heavy_csi_difference": c["csi_difference"], "csi_ci95_low": c["ci95_low"], "csi_ci95_high": c["ci95_high"],
                        "heavy_csi_verdict": _verdict(c["ci95_low"], c["ci95_high"], False)})
    uncertainty = pd.DataFrame(unc)
    uncertainty.to_csv(out / "verification" / "uncertainty.csv", index=False, float_format="%.5f")

    # grid-cell supportability (fit period)
    heavy_per_cell = (fitall["observation"] >= HEAVY_MM).sum("sample").where(land)
    cell_counts = pd.Series(np.bincount(fin["labels"], minlength=k))
    grid_note = {
        "median_heavy_events_per_land_cell_fit_period": float(heavy_per_cell.median()),
        "land_cells_with_ge_10_heavy_events": int((heavy_per_cell >= 10).sum()),
        "land_cells": int(land.sum()),
        "samples_per_regime_fit_period": cell_counts.tolist(),
        "supportable": False,
        "statement": "Grid-cell-level adaptive weights were not statistically supportable with the current sample size.",
    }
    manifest["grid_cell_weights"] = grid_note

    # ---- figures ------------------------------------------------------------
    plots.phase3_regime_distribution(assign, fin["model"], feats, out / "figures" / "regime_distribution.png", fin["names"])
    plots.phase3_model_error_by_regime(by_regime, models, fin["names"], out / "figures" / "model_error_by_regime.png")
    plots.phase3_weights_by_regime(weights_all, models, fin["names"], out / "figures" / "adaptive_weights_by_regime.png")
    plots.phase3_method_comparison(final, METHOD_ORDER, out / "figures" / "forecast_comparison.png",
                                   f"Test MAE vs IMD, {periods['test']['valid_dates'][0]} to {periods['test']['valid_dates'][1]}")
    plots.phase3_heavy_rain(final, METHOD_ORDER, out / "figures" / "heavy_rain_comparison.png",
                            f"Heavy rain (>= {HEAVY_MM:g} mm/24 h), test {periods['test']['valid_dates'][0]} to {periods['test']['valid_dates'][1]}")

    # ---- manifest + report -------------------------------------------------
    manifest.update({
        "k_table": k_table.drop(columns=["selection_note"], errors="ignore").to_dict(orient="records"),
        "k_selected": k, "k_selection_rule": "max training silhouette with min cluster size >= 5 and bootstrap ARI >= 0.6",
        "regimes": {int(r): n for r, n in fin["names"].items()},
        "heavy_regimes": [int(r) for r in fin["heavy_regimes"]],
        "test_regime_counts": {int(r): int(c) for r, c in zip(*np.unique(test_labels, return_counts=True))},
        "n_test_scored_points": int(test_mask.sum()),
        "n_test_heavy_events": int(((obs_t >= HEAVY_MM) & test_mask).sum()),
        "test_dates": sorted(set(test.valid_date.values.tolist())),
        "fallback_rows": {m: int((weights_all[weights_all.method == m].level == LEVEL_STATIC).sum()) for m in ("regime", "extreme_aware")},
        "leakage_controls": [
            "regime features use forecasts only",
            "scaler/PCA/KMeans fitted on fit-period samples only (dev: 1-17 Jul, final: 1-23 Jul)",
            "weights use fit-period observations only; test observations used only for scoring",
            "assert_no_leakage(train, validation) and assert_no_leakage(train+validation, test) passed",
        ],
        "outputs": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
    })
    summary = _summary(cfg, manifest, final, uncertainty, diffs, fin, val_scores, models)
    return _finish(out, manifest, "COMPLETED", summary)


def _mask(ds: xr.Dataset, models: list[str]) -> xr.DataArray:
    mask = ds["observation"].notnull()
    for m in models:
        mask = mask & ds["forecast"].sel(model=m).notnull()
    return mask


def _row(df: pd.DataFrame, method: str, lead: str = "all", regime="all") -> pd.Series:
    return df[(df.method == method) & (df.lead_day == lead) & (df.regime.astype(str) == str(regime))].iloc[0]


def _summary(cfg, manifest, final, unc, diffs, fin, val_scores, models) -> list[str]:
    p = manifest["periods"]
    L = [
        f"Phase 3 experiment: {cfg.name}  (PROOF OF CONCEPT: one monsoon month, {p['test']['n_days']} test days)",
        f"Train {p['train']['valid_dates'][0]}..{p['train']['valid_dates'][1]} | validation "
        f"{p['validation']['valid_dates'][0]}..{p['validation']['valid_dates'][1]} | test "
        f"{p['test']['valid_dates'][0]}..{p['test']['valid_dates'][1]} ({p['test']['n_samples']} forecast cases, "
        f"{manifest['n_test_scored_points']} scored land points, {manifest['n_test_heavy_events']} heavy-rain events)",
        f"Regimes: k = {manifest['k_selected']} (from forecast-time features); "
        + "; ".join(f"{n} [test samples: {manifest['test_regime_counts'].get(r, 0)}]" for r, n in manifest["regimes"].items()),
        "",
        "TEST PERIOD, all lead days (mm/24 h)  MAE / RMSE / bias / corr | heavy>=64.5: CSI / POD / FAR",
    ]
    for m in METHOD_ORDER:
        if m not in set(final.method):
            continue
        r = _row(final, m)
        L.append(f"  {plots.label_of(m):<28} {r.mae:.3f} / {r.rmse:.2f} / {r.bias:+.2f} / {r['corr']:.3f} | "
                 f"{r['csi@64.5mm']:.3f} / {r['pod@64.5mm']:.3f} / {r['far@64.5mm']:.3f}")
    counts = manifest["test_regime_counts"]
    top_r, top_n = max(counts.items(), key=lambda kv: kv[1])
    total = sum(counts.values())
    if top_n / total >= 0.75:
        L.append(f"NOTE: {top_n} of {total} test cases fall in one regime ({manifest['regimes'][top_r].split(':')[0]}); "
                 "on this test period regime conditioning can act only like a single fixed weight set.")
    st = fin["stats"]
    L.append("Heavy regimes for the extreme-aware blend (fit-period heavy-rain frequency above average): "
             + (", ".join(f"REGIME_{r.regime} (ratio {r.heavy_frequency_ratio:.2f})" for r in st[st.is_heavy_regime].itertuples())
                or "none")
             + ("; none reaches the 1.5 ratio required for a HEAVY name" if (st.heavy_frequency_ratio <= 1.5).all() else ""))
    sel = manifest.get("k_table")
    if sel:
        sils = [row["silhouette"] for row in sel if row.get("silhouette") == row.get("silhouette")]
        if sils and max(sils) < 0.5:
            L.append(f"Cluster structure is weak (best training silhouette {max(sils):.2f} < 0.5); regimes are loosely separated.")
    L.append("")
    spread = diffs[diffs.method == "regime"].spread_across_regimes
    L.append(f"Regime weights: spread across regimes max {spread.max():.3f}, median {spread.median():.3f} "
             f"(max |regime - static| {diffs[diffs.method == 'regime'].max_abs_diff_from_static.max():.3f})")
    if spread.max() < 0.02:
        L.append("-> All regimes produce almost identical weights; regime conditioning cannot change the blend much.")
    for a in ("regime", "extreme_aware"):
        for b in ("equal", "phase2_inverse_error"):
            u = unc[(unc.method == a) & (unc.versus == b)].iloc[0]
            L.append(f"{plots.label_of(a)} vs {plots.label_of(b)}: MAE {u.mae_difference:+.4f} mm "
                     f"[{u.mae_ci95_low:+.4f}, {u.mae_ci95_high:+.4f}] -> {u.mae_verdict}; heavy CSI "
                     f"{u.heavy_csi_difference:+.4f} [{u.csi_ci95_low:+.4f}, {u.csi_ci95_high:+.4f}] -> {u.heavy_csi_verdict}")
    ec = unc[(unc.method == "extreme_aware") & (unc.versus == "ECMWF")]
    if len(ec):
        u = ec.iloc[0]
        L.append(f"Extreme-aware vs ECMWF heavy CSI: {u.heavy_csi_difference:+.4f} [{u.csi_ci95_low:+.4f}, "
                 f"{u.csi_ci95_high:+.4f}] -> {u.heavy_csi_verdict}")
    vr = _row(val_scores, "regime"); ve = _row(val_scores, "equal"); vx = _row(val_scores, "extreme_aware")
    L.append(f"Validation (18-23 Jul, dev fit on 1-17 Jul): MAE equal {ve.mae:.3f}, regime {vr.mae:.3f}, "
             f"extreme-aware {vx.mae:.3f}; heavy CSI equal {ve['csi@64.5mm']:.3f}, regime {vr['csi@64.5mm']:.3f}, "
             f"extreme-aware {vx['csi@64.5mm']:.3f}")
    L.append(manifest["grid_cell_weights"]["statement"] + f" (median heavy events per land cell in the fit period: "
             f"{manifest['grid_cell_weights']['median_heavy_events_per_land_cell_fit_period']:.0f})")
    L.append("All statements refer to the 24-31 July 2025 test period only; no generalisation is claimed.")
    return L


def _finish(out: Path, manifest: dict[str, Any], status: str, summary: list[str]) -> Phase3Result:
    manifest["status"] = status
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["summary"] = summary
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text("# " + manifest["experiment"] + "\n\n```\n" + "\n".join(summary) + "\n```\n", encoding="utf-8")
    return Phase3Result(status, manifest, summary)
