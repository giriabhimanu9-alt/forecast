"""Phase 2 historical experiment: forecasts + IMD -> alignment -> skill -> blend -> independent test.

Every number written by this module is computed from downloaded data; when
a source is unavailable it is recorded as such and the experiment continues
with what is genuinely available.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from app.config import Settings
from app.models.schema import DataPipelineError, ProviderStatus
from app.services.data import validation
from app.services.data.ecmwf import ECMWFProvider
from app.services.data.gfs import GFSProvider
from app.services.data.observations import PERIOD_END_OFFSET, IMDRainfallProvider
from app.services.data.tigge import tigge_status
from app.services.http import HttpClient
from app.services.pipeline import run_forecast_provider, save_netcdf
from app.services.verification import plots
from app.services.verification.accumulation import required_lead_hours, verification_windows
from app.services.verification.alignment import build_aligned_dataset
from app.services.verification.blending import METHODS, apply_blend, learn_weights, weight_field
from app.services.verification.metrics import DEFAULT_THRESHOLDS_MM, score
from app.services.verification.regions import region_map
from app.services.verification.split import chronological_split

log = logging.getLogger(__name__)

ECMWF_HISTORICAL_URL = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
PROVIDERS = {"GFS": GFSProvider, "ECMWF": ECMWFProvider}


@dataclass
class ExperimentConfig:
    name: str
    valid_start: date  # first IMD day label verified
    valid_end: date  # last IMD day label verified
    output_dir: Path
    cycles: tuple[int, ...] = (0,)
    lead_days: tuple[int, ...] = (1, 2, 3)
    models: tuple[str, ...] = ("GFS", "ECMWF")
    train_fraction: float = 0.75
    thresholds_mm: tuple[float, ...] = DEFAULT_THRESHOLDS_MM
    epsilon_mm: float = 0.1
    min_days_for_skill_claim: int = 20
    min_test_days: int = 7
    bootstrap_samples: int = 2000
    seed: int = 20250701
    ecmwf_base_url: str = ECMWF_HISTORICAL_URL
    imd_source: str = "archive"
    http_retries: int = 8
    http_min_interval: float = 0.25

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(valid_start=self.valid_start.isoformat(), valid_end=self.valid_end.isoformat(),
                 output_dir=str(self.output_dir))
        return d


@dataclass
class ExperimentResult:
    status: str
    label: str
    manifest: dict[str, Any]
    summary_lines: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- planning


def observation_period_start(day: date, route: str) -> datetime:
    """Start (UTC) of the IMD 24 h window labelled ``day`` under the documented convention."""
    end = pd.Timestamp(day) + PERIOD_END_OFFSET[route]
    return (end - pd.Timedelta(hours=24)).to_pydatetime()


def plan_init_times(cfg: ExperimentConfig, route: str) -> list[datetime]:
    """Every cycle whose lead-day-k window coincides with a target observation window."""
    inits: set[datetime] = set()
    for d in pd.date_range(cfg.valid_start, cfg.valid_end, freq="D"):
        start = observation_period_start(d.date(), route)
        for cycle in cfg.cycles:
            first = (3 - cycle) % 24
            for k in cfg.lead_days:
                init = start - timedelta(hours=first + 24 * (k - 1))
                assert init.hour == cycle
                inits.add(init)
    return sorted(inits)


def _phase2_settings(
    settings: Settings, cfg: ExperimentConfig, cycle: int, lead_hours: list[int],
    variables: tuple[str, ...] = ("precipitation",),
) -> Settings:
    return replace(
        settings,
        variables=variables,
        lead_hours=tuple(lead_hours),
        ecmwf_base_url=cfg.ecmwf_base_url,
        http_retries=cfg.http_retries,
        imd_source=cfg.imd_source,
        imd_start_date=cfg.valid_start - timedelta(days=1),
        imd_end_date=cfg.valid_end + timedelta(days=1),
        forecast_init_time=None,
    )


# --------------------------------------------------------------------------- data


def fetch_observations(
    settings: Settings, cfg: ExperimentConfig, http: HttpClient | None = None
) -> tuple[xr.Dataset, dict[str, Any]]:
    s = _phase2_settings(settings, cfg, cfg.cycles[0], [])
    provider = IMDRainfallProvider(s, http=http or HttpClient(s.http_timeout, cfg.http_retries))
    ds, stats = provider.get_observations()
    ds, value_stats = validation.validate_observation_dataset(ds)
    out = s.processed_dir / "observations" / f"imd_rainfall_{s.imd_start_date:%Y%m%d}_{s.imd_end_date:%Y%m%d}.nc"
    save_netcdf(ds, out)
    info = {
        "source": ds.attrs.get("source"),
        "routes": sorted(set(stats.source_by_day.values())),
        "day_labels": [str(ds.source_date.values[0]), str(ds.source_date.values[-1])],
        "raw_files": sorted({str(p) for p in stats.files}),
        "processed_file": str(out),
        "units": ds.precipitation.attrs.get("units"),
        "valid_points_per_day": value_stats["precipitation"]["valid_points_per_day"],
    }
    return ds, info


def provider_class(model: str):
    """Resolve a model name to a ForecastProvider class: NWP built-ins, then registered AI plug-ins."""
    if model in PROVIDERS:
        return PROVIDERS[model]
    from app.services.data.ai_model import _REGISTRY  # AI plug-ins register themselves here

    if model in _REGISTRY:
        return _REGISTRY[model]
    raise KeyError(f"unknown model {model!r}; built-in: {sorted(PROVIDERS)}, AI plug-ins: {sorted(_REGISTRY)}")


def fetch_forecasts(
    settings: Settings, cfg: ExperimentConfig, init_times: list[datetime], http: HttpClient | None = None,
    variables: tuple[str, ...] = ("precipitation",),
) -> tuple[dict[str, dict[datetime, xr.Dataset]], list[dict[str, Any]]]:
    """Fetch every (model, cycle); ``variables`` must include precipitation (Phase 3 adds wind)."""
    forecasts: dict[str, dict[datetime, xr.Dataset]] = {m: {} for m in cfg.models}
    records: list[dict[str, Any]] = []
    http = http or HttpClient(settings.http_timeout, cfg.http_retries, min_interval=cfg.http_min_interval)
    for model in cfg.models:
        for init in init_times:
            leads = required_lead_hours(verification_windows(init, cfg.lead_days))
            s = _phase2_settings(settings, cfg, init.hour, leads, variables)
            provider = provider_class(model)(s, http=http)
            result = run_forecast_provider(provider, s, init)
            records.append({
                "model": model,
                "init_time": init.isoformat(),
                "lead_hours": leads,
                "status": result.status.value,
                "message": result.message,
                "downloads": result.downloads,
                "cache_hits": result.cache_hits,
            })
            if result.ok and result.dataset is not None:
                forecasts[model][init] = result.dataset[list(variables)].load()
                log.info("%s %s: %s", model, f"{init:%Y-%m-%d %H}Z", result.message)
            else:
                log.warning("%s %s: %s - %s", model, f"{init:%Y-%m-%d %H}Z", result.status.value, result.message)
    return forecasts, records


# --------------------------------------------------------------------------- diagnostics & scoring


def _pattern_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def observation_convention_check(aligned: xr.Dataset, observations: xr.Dataset) -> dict[str, Any]:
    """Which IMD day label best matches each forecast window? (data-driven sanity check)

    For every lead-day-1 sample and model, correlate the forecast 24 h field
    with the IMD fields labelled D-1, D, D+1 where D is the label the
    documented convention assigns. If the convention were off by a day, the
    shifted label would correlate better. Nothing is changed automatically.
    """
    labels = {str(v): i for i, v in enumerate(observations.source_date.values)}
    out: dict[str, Any] = {}
    day1 = aligned.isel(sample=np.flatnonzero(aligned.lead_day.values == 1))
    for model in aligned.model.values.tolist():
        per_offset: dict[int, list[float]] = {-1: [], 0: [], 1: []}
        for s in range(day1.sizes["sample"]):
            fc = day1["forecast"].sel(model=model).isel(sample=s).values
            if not np.isfinite(fc).any():
                continue
            d = date.fromisoformat(str(day1.valid_date.values[s]))
            for off in per_offset:
                key = (d + timedelta(days=off)).isoformat()
                if key in labels:
                    ob = observations["precipitation"].isel(time=labels[key]).values
                    per_offset[off].append(_pattern_corr(fc, ob))
        means = {str(off): (float(np.nanmean(v)) if v and np.isfinite(v).any() else None)
                 for off, v in per_offset.items()}
        valid = {k: v for k, v in means.items() if v is not None}
        best = max(valid, key=valid.get) if valid else None
        out[model] = {"mean_pattern_correlation_by_label_offset_days": means,
                      "n_samples": len(per_offset[0]), "best_offset_days": best}
    offsets = [v["best_offset_days"] for v in out.values() if v["best_offset_days"] is not None]
    out["convention_consistent"] = bool(offsets) and all(o == "0" for o in offsets)
    return out


def _mask_all(ds: xr.Dataset, models: list[str]) -> xr.DataArray:
    mask = ds["observation"].notnull()
    for m in models:
        mask = mask & ds["forecast"].sel(model=m).notnull()
    return mask


def _score_rows(label_period: str, method: str, fc: xr.DataArray, obs: xr.DataArray, lead_day, thresholds) -> dict:
    row = {"period": label_period, "method": method, "lead_day": lead_day}
    row.update(score(fc, obs, thresholds))
    return row


def model_scores(ds: xr.Dataset, period: str, thresholds) -> list[dict]:
    """Per-model scores on each model's own available pairs."""
    rows = []
    for m in ds.model.values.tolist():
        for d in sorted(set(ds.lead_day.values.tolist())):
            sub = ds.isel(sample=np.flatnonzero(ds.lead_day.values == d))
            rows.append(_score_rows(period, m, sub["forecast"].sel(model=m), sub["observation"], int(d), thresholds))
    return rows


def blended_forecasts(test: xr.Dataset, weights: pd.DataFrame, regions: xr.DataArray, models: list[str]) -> dict[str, xr.DataArray]:
    """Blends for the test samples; uses ONLY test forecasts + training weights."""
    out: dict[str, xr.DataArray] = {}
    fc = test["forecast"].sel(model=models)
    for method in METHODS:
        parts = []
        for d in sorted(set(test.lead_day.values.tolist())):
            idx = np.flatnonzero(test.lead_day.values == d)
            w = weight_field(weights, method, int(d), models, regions)
            parts.append(apply_blend(fc.isel(sample=idx), w))
        out[method] = xr.concat(parts, dim="sample").sortby("sample")
    return out


def daily_abs_error(fc: xr.DataArray, obs: xr.DataArray, mask: xr.DataArray, valid_dates: np.ndarray) -> pd.DataFrame:
    err = np.abs(fc - obs).where(mask)
    per_sample = pd.DataFrame({
        "valid_date": valid_dates,
        "abs_sum": err.sum(("latitude", "longitude")).values,
        "n": mask.sum(("latitude", "longitude")).values,
    })
    return per_sample.groupby("valid_date")[["abs_sum", "n"]].sum()


def bootstrap_mae_difference(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    """Paired day-block bootstrap of MAE(a) - MAE(b) over test valid dates."""
    days = a.index.intersection(b.index)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), size=(n_boot, len(days)))
    a_s, a_n = a.loc[days, "abs_sum"].to_numpy(), a.loc[days, "n"].to_numpy()
    b_s, b_n = b.loc[days, "abs_sum"].to_numpy(), b.loc[days, "n"].to_numpy()
    diff = a_s[idx].sum(1) / a_n[idx].sum(1) - b_s[idx].sum(1) / b_n[idx].sum(1)
    return {
        "mae_difference": float(a_s.sum() / a_n.sum() - b_s.sum() / b_n.sum()),
        "ci95_low": float(np.percentile(diff, 2.5)),
        "ci95_high": float(np.percentile(diff, 97.5)),
        "n_days": int(len(days)),
    }


# --------------------------------------------------------------------------- main entry


def run_experiment(settings: Settings, cfg: ExperimentConfig, http: HttpClient | None = None) -> ExperimentResult:
    started = datetime.now(timezone.utc)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    route = cfg.imd_source if cfg.imd_source != "auto" else "archive"
    manifest: dict[str, Any] = {
        "experiment": cfg.name,
        "phase": 2,
        "started_at": started.isoformat(timespec="seconds"),
        "config": cfg.as_dict(),
        "region": settings.grid.as_dict(),
        "variable": "precipitation",
        "units": "mm per 24 h",
        "observation_convention": {
            "route": route,
            "rule": f"IMD {route} day label D covers 03 UTC D-1 -> 03 UTC D (24 h ending 08:30 IST on D)",
            "verified_by": [
                "real-time file for D is published before 08:30 IST on D+1 (Phase 1)",
                "real-time label D == archive label D (r=0.997/0.994, mean|diff| 0.18/0.15 mm, 5 & 10 Jul 2025)",
                "per-run forecast pattern-correlation check (observation_convention_check)",
            ],
            "correction_note": ("Phase 1 assumed archive label D = 03 UTC D -> D+1; corrected on 2026-09-28 after the "
                                "evidence above showed a one-day offset"),
            "period_end_offset_hours": PERIOD_END_OFFSET[route].total_seconds() / 3600,
        },
        "forecast_accumulation": "rain_24h = acc(end_lead) - acc(start_lead); acc(0) = 0; windows end 03 UTC",
        "data_sources": {
            "GFS": "NOAA GFS 0.25 deg pgrb2, https://noaa-gfs-bdp-pds.s3.amazonaws.com (APCP 0-N h accumulation)",
            "ECMWF": f"ECMWF IFS 0.25 deg oper, ECMWF Open Data archive {cfg.ecmwf_base_url} (tp)",
            "IMD": "IMD Pune 0.25 deg gridded daily rainfall",
        },
        "tigge": tigge_status(),
    }

    # 1. observations
    log.info("=== IMD observations (%s) ===", route)
    try:
        observations, obs_info = fetch_observations(settings, cfg, http)
    except DataPipelineError as exc:
        manifest["observations"] = {"status": exc.status.value, "message": str(exc)}
        return _finish(out, manifest, "FAILED", "NO OBSERVATIONS", [f"IMD observations unavailable: {exc}"])
    manifest["observations"] = {"status": ProviderStatus.AVAILABLE.value, **obs_info}

    # 2. forecasts
    init_times = plan_init_times(cfg, route)
    log.info("=== Forecasts: %d cycles %s .. %s ===", len(init_times), init_times[0], init_times[-1])
    forecasts, records = fetch_forecasts(settings, cfg, init_times, http)
    manifest["forecast_cycles"] = [t.isoformat() for t in init_times]
    manifest["forecast_fetch"] = records
    manifest["failed_downloads"] = [r for r in records if r["status"] != ProviderStatus.AVAILABLE.value]
    available_models = [m for m in cfg.models if forecasts[m]]
    manifest["models_available"] = available_models
    manifest["models_unavailable"] = [m for m in cfg.models if not forecasts[m]]
    if not available_models:
        return _finish(out, manifest, "FAILED", "NO FORECASTS", ["No forecast model could be retrieved."])

    # 3. alignment (restricted to the target observation dates)
    targets = {observation_period_start(d.date(), route) for d in pd.date_range(cfg.valid_start, cfg.valid_end)}
    aligned, align_log = build_aligned_dataset(
        forecasts, observations, init_times, cfg.lead_days, available_models, window_starts=targets
    )
    outside = [v for v in aligned.valid_date.values if not cfg.valid_start.isoformat() <= v <= cfg.valid_end.isoformat()]
    if outside:
        raise RuntimeError(f"aligned samples outside the target dates: {outside}")
    save_netcdf(aligned, out / "aligned_samples.nc")
    manifest["alignment"] = {
        "aligned_samples": int(aligned.sizes["sample"]),
        "samples_per_lead_day": {str(k): int(v) for k, v in pd.Series(aligned.lead_day.values).value_counts().sort_index().items()},
        "valid_dates": sorted(set(aligned.valid_date.values.tolist())),
        "lead_times": {
            str(d): sorted({f"{int(a)}-{int(b)} h" for a, b, k in zip(aligned.start_lead_h.values, aligned.end_lead_h.values,
                                                                        aligned.lead_day.values) if k == d})
            for d in sorted(set(aligned.lead_day.values.tolist()))
        },
        "grid": {"latitude": int(aligned.sizes["latitude"]), "longitude": int(aligned.sizes["longitude"])},
        "missing_forecast_windows": align_log.missing_forecast,
        "missing_observation_windows": align_log.missing_observation,
        "file": str(out / "aligned_samples.nc"),
    }
    manifest["observation_convention_check"] = observation_convention_check(aligned, observations)

    # 4. split
    split = chronological_split(aligned, cfg.train_fraction)
    manifest["split"] = split.describe()
    manifest["training_period"] = [str(split.train.observation_period_start.values.min()),
                                   str(split.train.observation_period_end.values.max())]
    manifest["test_period"] = [str(split.test.observation_period_start.values.min()),
                               str(split.test.observation_period_end.values.max())]
    n_days = len(split.train_dates) + len(split.test_dates)
    scientific = n_days >= cfg.min_days_for_skill_claim and len(split.test_dates) >= cfg.min_test_days
    label = "HISTORICAL EXPERIMENT" if scientific else "ALIGNMENT/PIPELINE TEST - NOT A SCIENTIFIC SKILL CLAIM"
    manifest["label"] = label

    # 5. model scores (train + test, each model on its own pairs)
    thresholds = list(cfg.thresholds_mm)
    ms = pd.DataFrame(model_scores(split.train, "train", thresholds) + model_scores(split.test, "test", thresholds))
    ms.to_csv(out / "model_scores.csv", index=False, float_format="%.5f")

    # 6. weights from TRAINING data only
    regions = region_map(aligned.latitude.values, aligned.longitude.values)
    weights = learn_weights(split.train, regions, cfg.epsilon_mm)
    weights.to_csv(out / "weights.csv", index=False, float_format="%.6f")

    # 7. independent test
    blends = blended_forecasts(split.test, weights, regions, available_models)
    candidates: dict[str, xr.DataArray] = {m: split.test["forecast"].sel(model=m, drop=True) for m in available_models}
    candidates.update(blends)
    mask = _mask_all(split.test, available_models)
    obs_test = split.test["observation"]
    test_rows = []
    lead_groups = [(str(d), np.flatnonzero(split.test.lead_day.values == d)) for d in sorted(set(split.test.lead_day.values.tolist()))]
    lead_groups.append(("all", np.arange(split.test.sizes["sample"])))
    for g, idx in lead_groups:
        for name, fc in candidates.items():
            test_rows.append(_score_rows("test", name, fc.isel(sample=idx).where(mask.isel(sample=idx)),
                                         obs_test.isel(sample=idx), g, thresholds))
    test_scores = pd.DataFrame(test_rows)
    test_scores.to_csv(out / "test_scores.csv", index=False, float_format="%.5f")

    # 8. blend comparison with paired bootstrap vs best individual (per lead group)
    comp_rows = []
    for g, idx in lead_groups:
        equal_daily = None
        sub = test_scores[test_scores.lead_day == g]
        indiv = sub[sub.method.isin(available_models)]
        best = indiv.loc[indiv.mae.idxmin()]
        dates = split.test.valid_date.values[idx]
        m_idx = mask.isel(sample=idx)
        best_daily = daily_abs_error(candidates[best.method].isel(sample=idx), obs_test.isel(sample=idx), m_idx, dates)
        for name in candidates:
            row = sub[sub.method == name].iloc[0]
            rec = {"lead_day": g, "method": name, "mae": row.mae, "rmse": row.rmse, "bias": row.bias, "n": row.n,
                   "best_individual": best.method, "best_individual_mae": best.mae,
                   "improvement_vs_best_pct": 100.0 * (best.mae - row.mae) / best.mae}
            if name not in available_models:
                daily = daily_abs_error(candidates[name].isel(sample=idx), obs_test.isel(sample=idx), m_idx, dates)
                if name == "equal":
                    equal_daily = daily
                elif equal_daily is not None:
                    ve = bootstrap_mae_difference(daily, equal_daily, cfg.bootstrap_samples, cfg.seed)
                    rec.update({"mae_minus_equal": ve["mae_difference"], "vs_equal_ci95_low": ve["ci95_low"],
                                "vs_equal_ci95_high": ve["ci95_high"]})
                bs = bootstrap_mae_difference(daily, best_daily, cfg.bootstrap_samples, cfg.seed)
                rec.update({"mae_minus_best": bs["mae_difference"], "ci95_low": bs["ci95_low"],
                            "ci95_high": bs["ci95_high"], "bootstrap_days": bs["n_days"]})
                if bs["n_days"] < cfg.min_test_days:
                    rec["verdict"] = f"not assessed (only {bs['n_days']} test days, need {cfg.min_test_days})"
                else:
                    rec["verdict"] = ("better (95% CI < 0)" if bs["ci95_high"] < 0 else
                                      "worse (95% CI > 0)" if bs["ci95_low"] > 0 else "not distinguishable")
            comp_rows.append(rec)
    comparison = pd.DataFrame(comp_rows)
    comparison.to_csv(out / "blend_comparison.csv", index=False, float_format="%.5f")

    # 9. figures
    _figures(out, cfg, split, weights, regions, candidates, mask, test_scores, available_models, label)

    manifest["metrics"] = ["mae", "rmse", "bias"] + [f"{k}@{t:g}mm" for t in thresholds for k in ("precision", "recall", "f1", "csi")]
    manifest["n_test_scored_points"] = int(mask.sum())
    manifest["outputs"] = sorted(p.name for p in out.iterdir())
    summary = _summary(cfg, manifest, comparison, available_models, label, test_scores)
    return _finish(out, manifest, "COMPLETED", label, summary)


def _figures(out, cfg, split, weights, regions, candidates, mask, test_scores, models, label) -> None:
    methods = models + list(METHODS)
    tag = "pipeline test" if label.startswith("ALIGNMENT") else "independent test period"
    plots.skill_comparison(test_scores, methods, out / "skill_comparison.png",
                           f"{cfg.name}: MAE vs IMD, {tag} {split.test_dates[0]} to {split.test_dates[-1]}")
    day1 = np.flatnonzero(split.test.lead_day.values == min(cfg.lead_days))
    land = split.test["observation"].isel(sample=day1).notnull().any("sample")
    fields = {"IMD observed": split.test["observation"].isel(sample=day1).mean("sample")}
    for name in models + ["regional"]:
        fields[plots.label_of(name)] = candidates[name].isel(sample=day1).mean("sample").where(land)
    plots.rainfall_vs_imd(fields, out / "rainfall_forecast_vs_imd.png",
                          f"Mean daily rainfall, lead day {min(cfg.lead_days)}, test period {split.test_dates[0]} to {split.test_dates[-1]}")
    wmaps = {}
    for d in sorted(set(weights.lead_day)):
        w = weight_field(weights, "regional", int(d), models, regions)
        for m in models:
            wmaps[(m, int(d))] = w.sel(model=m).where(land)
    plots.weight_map(wmaps, regions, out / "weight_map.png",
                     f"Regional inverse-error weights learned on {split.train_dates[0]} to {split.train_dates[-1]}")
    tr = split.train.isel(sample=np.flatnonzero(split.train.lead_day.values == min(cfg.lead_days)))
    tr_mask = _mask_all(tr, models)
    maes = {m: np.abs(tr["forecast"].sel(model=m) - tr["observation"]).where(tr_mask).mean("sample") for m in models}
    plots.error_map(maes, out / "error_map.png",
                    f"Training-period MAE per grid point, lead day {min(cfg.lead_days)}")


def _fmt(x: float) -> str:
    return f"{x:.3f}" if np.isfinite(x) else "n/a"


def _summary(cfg, manifest, comparison: pd.DataFrame, models: list[str], label: str,
             test_scores: pd.DataFrame | None = None) -> list[str]:
    allc = comparison[comparison.lead_day == "all"].set_index("method")
    best = allc.best_individual.iloc[0]
    lines = [
        f"Experiment: {cfg.name}   [{label}]",
        f"Models available: {', '.join(models)}"
        + (f"   (unavailable: {', '.join(manifest['models_unavailable'])})" if manifest["models_unavailable"] else ""),
        f"Aligned samples: {manifest['alignment']['aligned_samples']} "
        f"(lead days {', '.join(manifest['alignment']['samples_per_lead_day'])}), "
        f"train days {manifest['split']['n_train_days']}, test days {manifest['split']['n_test_days']}",
        f"Test period (IMD labels): {manifest['split']['test_valid_dates'][0]} .. {manifest['split']['test_valid_dates'][1]}",
        "",
        "TEST PERIOD, all lead days pooled (mm / 24 h, common grid points only):",
        f"Best individual model by MAE: {best} ({_fmt(allc.loc[best, 'mae'])})",
    ]
    for m in models:
        if m != best:
            lines.append(f"Other individual model {m} MAE: {_fmt(allc.loc[m, 'mae'])}")
    for method, text in (("equal", "Equal blend MAE"), ("inverse_error", "Inverse-error blend MAE"),
                         ("regional", "Regional adaptive blend MAE")):
        lines.append(f"{text}: {_fmt(allc.loc[method, 'mae'])}")
    if len(models) < 2:
        lines.append("Blend improvement versus best individual: not applicable - only one model available, "
                     "every blend equals that model.")
    else:
        top = allc.loc[list(METHODS)].mae.idxmin()
        r = allc.loc[top]
        lines.append(
            f"Blend improvement versus best individual: best blend = {plots.label_of(top)}, "
            f"{r.improvement_vs_best_pct:+.2f}% MAE change "
            f"(MAE difference {r.mae_minus_best:+.3f} mm, 95% CI [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}], "
            f"{int(r.bootstrap_days)} test days) -> {r.verdict}"
        )
        if not label.startswith("HISTORICAL"):
            lines.append("No improvement claim: this run is an alignment/pipeline test (too few days).")
        elif r.verdict.startswith("better"):
            lines.append("The blend beat the best individual model on the independent test period.")
        elif r.verdict.startswith("worse"):
            lines.append("The blend was WORSE than the best individual model on the independent test period.")
        else:
            lines.append("The difference between the best blend and the best individual model is NOT statistically "
                         "distinguishable on this test period; no improvement is claimed.")
        # Did skill-based weighting add anything beyond plain averaging?
        for method in ("inverse_error", "regional"):
            r2 = allc.loc[method]
            if "mae_minus_equal" in allc.columns and np.isfinite(r2.mae_minus_equal):
                lines.append(
                    f"{plots.label_of(method)} vs equal blend: MAE difference {r2.mae_minus_equal:+.4f} mm "
                    f"(95% CI [{r2.vs_equal_ci95_low:+.4f}, {r2.vs_equal_ci95_high:+.4f}])"
                )
        diffs = [abs(allc.loc[m, "mae_minus_equal"]) for m in ("inverse_error", "regional")
                 if "mae_minus_equal" in allc.columns and np.isfinite(allc.loc[m, "mae_minus_equal"])]
        if diffs and max(diffs) < 0.01:
            lines.append("-> Skill-based weights add no measurable benefit over equal weighting here (|diff| < 0.01 mm); "
                         "the blend gain comes from averaging the two models.")
        lines.append("Mean bias (forecast - IMD): " + ", ".join(
            f"{plots.label_of(m)} {allc.loc[m, 'bias']:+.3f}" for m in models + list(METHODS)))
        if test_scores is not None:
            ts = test_scores[test_scores.lead_day.astype(str) == "all"].set_index("method")
            top_t = max(cfg.thresholds_mm)
            col = f"csi@{top_t:g}mm"
            if col in ts.columns and (ts.loc[models, col].isna().all() or ts.loc[list(METHODS), col].isna().all()):
                lines.append(f"Heavy rain (>= {top_t:g} mm/24 h) CSI: undefined (no events forecast or observed)")
            elif col in ts.columns:
                best_ind = ts.loc[models, col].idxmax()
                best_bl = ts.loc[list(METHODS), col].idxmax()
                b_i, b_b = ts.loc[best_ind, col], ts.loc[best_bl, col]
                verdict = "higher" if b_b > b_i else "LOWER" if b_b < b_i else "equal"
                lines.append(
                    f"Heavy rain (>= {top_t:g} mm/24 h) CSI: best individual {best_ind} {b_i:.3f}, best blend "
                    f"{plots.label_of(best_bl)} {b_b:.3f} -> blend is {verdict}"
                    + (" (averaging smooths extremes)" if verdict == "LOWER" else "")
                )
    conv = manifest["observation_convention_check"]
    lines.append("")
    lines.append("IMD period check (lead-day-1 pattern correlation by label offset -1/0/+1 day): " + "; ".join(
        f"{m}: " + ", ".join(f"{k}:{v:.3f}" for k, v in c["mean_pattern_correlation_by_label_offset_days"].items() if v is not None)
        for m, c in conv.items() if isinstance(c, dict)))
    lines.append("Convention consistent with data: " + ("YES" if conv["convention_consistent"] else
                                                         "NO - review the IMD period convention before trusting scores"))
    if not label.startswith("HISTORICAL"):
        lines.append("NOTE: sample too small for a skill claim; these numbers only demonstrate that the pipeline works.")
    return lines


def _finish(out: Path, manifest: dict[str, Any], status: str, label: str, summary: list[str]) -> ExperimentResult:
    manifest["status"] = status
    manifest["label"] = label
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["summary"] = summary
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text("# " + manifest["experiment"] + "\n\n```\n" + "\n".join(summary) + "\n```\n",
                                   encoding="utf-8")
    return ExperimentResult(status, label, manifest, summary)
