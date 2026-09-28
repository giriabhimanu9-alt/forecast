"""Phase 4 orchestration: run Experiment A (IMD) and B (ERA5 benchmark), write data-quality files,
manifests and PHASE4_RESULTS.md. Every number in the report is read from computed tables."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import BACKEND_ROOT, Settings
from app.services.verification import plots
from app.services.verification.phase4 import (
    BlendConfig,
    ExperimentAConfig,
    ExperimentBConfig,
    run_experiment_a,
    run_experiment_b,
)

SUBDIRS = ("data_audit", "datasets", "model_scores", "regime_analysis", "weights", "verification", "figures",
           "manifests", "data_quality")

VARIABLE_METADATA = [
    {"experiment": "A", "model": "GFS", "source_variable": "APCP:surface:0-N hour acc fcst", "source_units": "kg m-2",
     "meaning": "precipitation accumulated since initialisation", "conversion": "x1 -> mm; 24 h = acc(end) - acc(start)"},
    {"experiment": "A", "model": "ECMWF", "source_variable": "tp (sfc)", "source_units": "m",
     "meaning": "precipitation accumulated since initialisation", "conversion": "x1000 -> mm; 24 h = acc(end) - acc(start)"},
    {"experiment": "A", "model": "GFS_ATMOS_1P00", "source_variable": "PRMSL / UGRD 850 mb / PWAT", "source_units": "Pa / m s-1 / kg m-2",
     "meaning": "instantaneous at step", "conversion": "Pa x0.01 -> hPa; m s-1; kg m-2 = mm; regime features only"},
    {"experiment": "A", "model": "IMD", "source_variable": "gridded rainfall (imdlib 'rain')", "source_units": "mm (imdlib label 'mm/day')",
     "meaning": "24 h total ending 08:30 IST (03 UTC) on label day D", "conversion": "x1; -999 -> NaN"},
    {"experiment": "B", "model": "GraphCast", "source_variable": "total_precipitation_6hr", "source_units": "m (verified vs ERA5)",
     "meaning": "precipitation in the 6 h before each step", "conversion": "negative buckets clipped to 0 (counted); cumulative sum -> since init; x1000 -> mm; 24 h difference"},
    {"experiment": "B", "model": "HRES", "source_variable": "total_precipitation", "source_units": "m",
     "meaning": "accumulated since initialisation (0 at step 0)", "conversion": "x1000 -> mm; 24 h difference"},
    {"experiment": "B", "model": "ERA5", "source_variable": "total_precipitation_24hr", "source_units": "m",
     "meaning": "24 h total ending at the time stamp (00 UTC)", "conversion": "x1000 -> mm; masked to India land cells"},
]


def run_phase4(settings: Settings, out: Path, run_a: bool = True, run_b: bool = True) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    for d in SUBDIRS:
        (out / d).mkdir(parents=True, exist_ok=True)
    audit = BACKEND_ROOT / "PHASE4_DATA_AUDIT.md"
    if audit.is_file():
        shutil.copy2(audit, out / "data_audit" / audit.name)
    bcfg, acfg, bexp = BlendConfig(), ExperimentAConfig(), ExperimentBConfig()
    res_a = run_experiment_a(settings, acfg, bcfg, out) if run_a else None
    land = res_a["land"] if res_a else None
    res_b = run_experiment_b(settings, bexp, bcfg, land, out) if (run_b and land is not None) else None

    parts = [r for r in (res_a, res_b) if r]
    pd.DataFrame([x for r in parts for x in r["coverage"]]).to_csv(out / "data_quality" / "coverage.csv", index=False)
    pd.DataFrame([x for r in parts for x in r["missing"]] or [{"experiment": None, "kind": "none"}]).to_csv(
        out / "data_quality" / "missing_data.csv", index=False)
    pd.DataFrame([x for r in parts for x in r["alignment_checks"]]).to_csv(out / "data_quality" / "alignment_checks.csv", index=False)
    pd.DataFrame([x for r in parts for x in r["grid_info"]]).to_csv(out / "data_quality" / "model_grid_info.csv", index=False)
    pd.DataFrame(VARIABLE_METADATA).to_csv(out / "data_quality" / "variable_metadata.csv", index=False)
    for name, r in (("expA", res_a), ("expB", res_b)):
        if r:
            pd.DataFrame(r["dataset_manifest"]).to_csv(out / "manifests" / f"{name}_dataset_manifest.csv", index=False)

    manifest = {
        "phase": 4, "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "blend_config": asdict(bcfg),
        "experiment_a": {**{k: v for k, v in asdict(acfg).items()}, "reference": "IMD", "window": "03 UTC -> 03 UTC"},
        "experiment_b": {"seasons": bexp.seasons, "train": bexp.train_seasons, "validation": bexp.validation_seasons,
                         "test": bexp.test_seasons, "grid": bexp.grid.as_dict(), "reference": "ERA5", "window": "00 UTC -> 00 UTC"},
        "rule": "Experiment A and B scores are never combined; B does not measure skill against IMD.",
        "python": platform.python_version(),
        "packages": _versions(),
        "results": {name: _summary_numbers(r) for name, r in (("A", res_a), ("B", res_b)) if r},
        "outputs": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
    }
    (out / "manifests" / "phase4_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    md = build_results_md(res_a, res_b, acfg, bexp, bcfg)
    (BACKEND_ROOT / "PHASE4_RESULTS.md").write_text(md, encoding="utf-8")
    (out / "PHASE4_RESULTS.md").write_text(md, encoding="utf-8")
    return {"A": res_a, "B": res_b, "manifest": manifest}


def _versions() -> dict[str, str]:
    import importlib

    out = {}
    for pkg in ("xarray", "numpy", "pandas", "scipy", "sklearn", "cfgrib", "eccodes", "zarr", "imdlib"):
        try:
            out[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            out[pkg] = "n/a"
    return out


def _all(final: pd.DataFrame, method: str, lead="all") -> pd.Series | None:
    r = final[(final.method == method) & (final.lead_day == lead) & (final.regime.astype(str) == "all")]
    return r.iloc[0] if len(r) else None


def _summary_numbers(r: dict[str, Any]) -> dict[str, Any]:
    f = r["final"]
    return {m: {k: float(_all(f, m)[k]) for k in ("mae", "rmse", "bias", "corr", "csi@64.5mm", "pod@64.5mm", "far@64.5mm")}
            for m in f.method.unique() if _all(f, m) is not None}


def _fmt(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _score_table(r: dict[str, Any], methods: list[str]) -> str:
    f = r["final"]
    lines = ["| Method | MAE | RMSE | Bias | Corr | CSI ≥64.5 | POD ≥64.5 | FAR ≥64.5 | CSI ≥15.6 |", "|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        x = _all(f, m)
        if x is None:
            continue
        lines.append(f"| {plots.label_of(m)} | {_fmt(x.mae)} | {_fmt(x.rmse, 2)} | {_fmt(x.bias, 2, True)} | {_fmt(x['corr'])} | "
                     f"{_fmt(x['csi@64.5mm'])} | {_fmt(x['pod@64.5mm'])} | {_fmt(x['far@64.5mm'])} | {_fmt(x['csi@15.6mm'])} |")
    return "\n".join(lines)


def _lead_table(r: dict[str, Any], methods: list[str]) -> str:
    f = r["final"]
    leads = [d for d in ("1", "2", "3") if d in set(f.lead_day)]
    head = "| Method | " + " | ".join(f"Day {d} MAE" for d in leads) + " | " + " | ".join(f"Day {d} CSI≥64.5" for d in leads) + " |"
    lines = [head, "|---" * (1 + 2 * len(leads)) + "|"]
    for m in methods:
        vals = [_all(f, m, d) for d in leads]
        if any(v is None for v in vals):
            continue
        lines.append(f"| {plots.label_of(m)} | " + " | ".join(_fmt(v.mae) for v in vals) + " | "
                     + " | ".join(_fmt(v["csi@64.5mm"]) for v in vals) + " |")
    return "\n".join(lines)


def _season_table(r: dict[str, Any], methods: list[str]) -> str:
    s = r["by_season"]
    s = s[(s.lead_day == "all") & (s.regime.astype(str) == "all")]
    lines = ["| Season | Method | MAE | RMSE | Bias | CSI ≥64.5 | POD | FAR | heavy events |", "|---|---|---|---|---|---|---|---|---|"]
    for season in sorted(s.season.unique()):
        for m in methods:
            x = s[(s.season == season) & (s.method == m)]
            if len(x):
                x = x.iloc[0]
                lines.append(f"| {season} | {plots.label_of(m)} | {_fmt(x.mae)} | {_fmt(x.rmse, 2)} | {_fmt(x.bias, 2, True)} | "
                             f"{_fmt(x['csi@64.5mm'])} | {_fmt(x['pod@64.5mm'])} | {_fmt(x['far@64.5mm'])} | {int(x['observed_events@64.5mm'])} |")
    return "\n".join(lines)


def _unc_lines(r: dict[str, Any]) -> list[str]:
    u = r["uncertainty"]
    out = []
    for x in u.itertuples():
        out.append(f"* {plots.label_of(x.method)} vs {plots.label_of(x.versus)}: MAE {x.mae_difference:+.3f} mm "
                   f"({x.mae_difference_pct:+.2f} %) [{x.mae_ci95_low:+.3f}, {x.mae_ci95_high:+.3f}] → **{x.mae_assessment}**; "
                   f"heavy CSI {_fmt(x.heavy_csi_difference, 4, True)} [{_fmt(x.csi_ci95_low, 4, True)}, "
                   f"{_fmt(x.csi_ci95_high, 4, True)}] → **{x.heavy_csi_assessment}** ({x.test_days} test days)")
    return out


def _regime_lines(r: dict[str, Any]) -> list[str]:
    lines = []
    for fs, d in r["development"].items():
        lines.append(f"* feature set `{fs}`: k = {d['k']}, training silhouette {d['silhouette']:.3f}, bootstrap ARI "
                     f"{d['stability_ari']:.3f}, validation MAE of regime blend {d['val_mae_regime']:.3f}")
    lines.append(f"* chosen (lowest validation MAE, pre-registered rule): **`{r['chosen_feature_set']}`**")
    st = r["regime_stats"]
    for x in st.itertuples():
        lines.append(f"  * {x.name}: {x.n_samples} fit samples, mean IMD/ref rain {x.obs_mean_mm:.2f} mm, heavy-rain "
                     f"frequency ratio {x.heavy_frequency_ratio:.2f}, test samples {r['test_regime_counts'].get(int(x.regime), 0)}")
    sil = max(d["silhouette"] for d in r["development"].values())
    lines.append(f"* cluster structure: best silhouette {sil:.2f} → " + ("**weak** (< 0.5)" if sil < 0.5 else "moderate/strong"))
    w = r["weights"]
    spread = w.groupby(["lead_day", "region", "model"]).weight.agg(lambda s: s.max() - s.min())
    lines.append(f"* regime weights: spread across regimes max {spread.max():.3f}, median {spread.median():.3f}"
                 + (" → regimes produce nearly identical weights" if spread.max() < 0.02 else ""))
    return lines


def _weights_table(r: dict[str, Any]) -> str:
    s = r["static_weights"]
    w = r["weights"]
    models = list(dict.fromkeys(s.model))
    lines = ["| Lead day | " + " | ".join(f"static {m}" for m in models) + " | " +
             " | ".join(f"regime {m} (min–max)" for m in models) + " |", "|---" * (1 + 2 * len(models)) + "|"]
    for d in sorted(s.lead_day.unique()):
        st = [f"{float(s[(s.lead_day == d) & (s.model == m)].weight.iloc[0]):.3f}" for m in models]
        rg = [f"{w[(w.lead_day == d) & (w.model == m)].weight.min():.2f}–{w[(w.lead_day == d) & (w.model == m)].weight.max():.2f}" for m in models]
        lines.append(f"| {d} | " + " | ".join(st) + " | " + " | ".join(rg) + " |")
    return "\n".join(lines)


def build_results_md(res_a, res_b, acfg: ExperimentAConfig, bexp: ExperimentBConfig, bcfg: BlendConfig) -> str:
    L = ["# Phase 4 results — multi-season validation + AI forecast benchmark", "",
         f"Generated automatically by `scripts/run_phase4.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. "
         "Every number below is read from the result tables in `results/phase4/`.", "",
         "> **Experiments A and B are separate and must not be compared with each other.** "
         "A verifies against IMD gauge-based rainfall (03→03 UTC); B verifies against ERA5 reanalysis (00→00 UTC, 1.5°). "
         "No AI model could be aligned exactly with IMD (6-hourly output only), so **no AI result here is evidence of skill "
         "against IMD**.", ""]
    L += ["## 1–6. Datasets, dates, models, variables, resolutions", "",
          "| | Experiment A — India operational | Experiment B — AI/NWP benchmark |", "|---|---|---|",
          f"| Truth | IMD 0.25° gridded rainfall (archive), India land | ERA5 `total_precipitation_24hr` (WeatherBench 2), India land cells (≥50 % IMD land) |",
          f"| Seasons (JJAS) | {', '.join(map(str, acfg.seasons))} | {', '.join(map(str, bexp.seasons))} |",
          "| Models | GFS 0.25° (NWP), ECMWF IFS (NWP; 0.4° in 2023, 0.25° 2024–25) | **GraphCast (AI)**, IFS HRES (NWP) |",
          "| Forecast cycles | 00 UTC daily | 00 UTC daily |",
          "| Lead days | 1–3 (3–27, 27–51, 51–75 h) | 1–3 (0–24, 24–48, 48–72 h) |",
          "| Verification window | 24 h ending 03 UTC (08:30 IST), IMD label D | 24 h ending 00 UTC |",
          "| Grid | 0.25°, 5–38°N × 68–98°E (133×121); ECMWF 2023 bilinear from 0.4° | 1.5°, 6–37.5°N × 67.5–97.5°E (22×21), native |",
          "| Variable | 24 h precipitation (mm) | 24 h precipitation (mm) |",
          "| Regime features | forecast rainfall (both models) + GFS 1.0° MSLP, u850, PWAT | forecast rainfall |", ""]
    for name, r, cfg in (("A", res_a, acfg), ("B", res_b, bexp)):
        if not r:
            L += [f"**Experiment {name} was not run.**", ""]
            continue
        p = r["periods"]
        L += [f"## 7. Split — Experiment {name}", "",
              f"* train {p['train']['seasons']} ({p['train']['n_samples']} forecast cases), validation {p['validation']['seasons']} "
              f"({p['validation']['n_samples']}), test **{p['test']['seasons']}** ({p['test']['n_samples']} cases, "
              f"{r['n_test_days']} days, {r['n_test_points']:,} scored grid points, {r['n_test_heavy']:,} heavy-rain events ≥64.5 mm).",
              "* Development: fit on train, choose k and regime feature set on validation. Final: refit on train+validation, "
              "evaluate **once** on test. No shuffling; `assert_no_leakage` checked both boundaries.", ""]
        models = r["models"]
        methods = models + ["equal", "phase2_inverse_error", "regime", "extreme_aware"] + \
            [m for m in r["final"].method.unique() if str(m).startswith("regime[")]
        L += [f"## 8–11. Test scores — Experiment {name} ({', '.join(map(str, p['test']['seasons']))}, all lead days)", "",
              _score_table(r, methods), "", "By lead day:", "", _lead_table(r, methods), "",
              f"Best individual model by MAE: **{r['best_individual']}**; by heavy-rain CSI: **{r['best_individual_heavy']}**.", "",
              "Uncertainty (paired bootstrap over test **days**, all lead days of a day resampled together, "
              f"{bcfg.bootstrap_samples} resamples). An effect is called *meaningful* only if the CI excludes 0 **and** "
              "|ΔMAE| ≥ 1 % (or |ΔCSI| ≥ 0.005); otherwise it is *negligible* even when statistically detectable "
              "(millions of correlated grid points make tiny effects 'significant'):", ""] + _unc_lines(r) + [""]
        L += [f"### By season — Experiment {name} (individual models + equal blend, no training needed)", "",
              _season_table(r, models + ["equal"]), ""]
        L += [f"## 12. Regimes — Experiment {name}", ""] + _regime_lines(r) + ["", "Weights (lead-day static vs regime range across regimes and regions):", "",
                                                                             _weights_table(r), ""]
    L += conclusions(res_a, res_b)
    L += ["## 13. Limitations", "",
          "* **No AI model in the IMD experiment.** Every accessible AI forecast (AIFS, GraphCast, GraphCastGFS, NOAA MLWP) outputs "
          "6-hourly steps, which cannot bracket IMD's 03 UTC day; exact alignment would need temporal interpolation (not done).",
          "* **Experiment B favours GraphCast by construction:** GraphCast is trained on and initialised from ERA5, which is also the "
          "reference; HRES starts from its own analysis. B is a benchmark of agreement with ERA5, not with observed rain.",
          "* **ERA5 is a reanalysis**, not gauge data; its 1.5° rainfall is smoother than IMD; heavy-rain events at 1.5° are rare.",
          "* **Experiment A has three seasons only** (ECMWF open data starts 2023-01-18): one training season, one validation, one test.",
          "* ECMWF 2023 is 0.4° interpolated to 0.25°; its skill may differ from the native 0.25° seasons.",
          "* The 2025 test season contains **July 2025, which was used to fit Phases 2–3** and whose results informed this design; "
          "no Phase 4 parameter is fitted on 2025, but the test is not researcher-blind.",
          "* Regional blocks are coarse lat/lon boxes, not IMD subdivisions; grid-cell weights were not attempted.",
          "* Confidence intervals resample days within one test season; they do not capture season-to-season variability.", "",
          "## 14. Data-access issues", "",
          "* ECMWF open-data AWS mirror throttles (`503 SlowDown`); handled by paced requests + exponential backoff (see `data_quality/missing_data.csv`).",
          "* TIGGE needs an ECMWF account/API key and is 6-hourly; NOAA MLWP files are ~9.3 GB per forecast; Pangu and NeuralGCM "
          "(WB2) have no precipitation variable; FuXi covers 2020 only. Details: `PHASE4_DATA_AUDIT.md`.", "",
          "## 15. Reproduce", "", "```bash", "cd backend", "pytest -q",
          "python scripts/audit_phase4_sources.py      # re-runs the data-access probes -> results/phase4/data_audit/",
          "python scripts/run_phase4.py                # both experiments; first run downloads ~3 GB (cached afterwards)", "```", ""]
    return "\n".join(L)


def conclusions(res_a, res_b) -> list[str]:
    L = ["## Conclusions (13–14 of the brief: did AI / adaptive weighting help?)", ""]
    for name, r in (("A (IMD)", res_a), ("B (ERA5 benchmark)", res_b)):
        if not r:
            continue
        u = r["uncertainty"]

        def verdict(a, b, col="mae_assessment"):
            x = u[(u.method == a) & (u.versus == b)]
            return x.iloc[0][col] if len(x) else "n/a"

        best = r["best_individual"]
        L.append(f"**Experiment {name}**")
        L.append(f"* Equal blend vs best individual ({best}) MAE: {verdict('equal', best)}; "
                 f"heavy CSI vs best heavy-rain model ({r['best_individual_heavy']}): {verdict('equal', r['best_individual_heavy'], 'heavy_csi_assessment')}.")
        L.append(f"* Skill-based static weights (Phase 2 inverse-error) vs equal blend: MAE {verdict('phase2_inverse_error', 'equal')}; "
                 f"heavy CSI {verdict('phase2_inverse_error', 'equal', 'heavy_csi_assessment')}.")
        L.append(f"* Adaptive regime blend vs equal blend: MAE {verdict('regime', 'equal')}; heavy CSI {verdict('regime', 'equal', 'heavy_csi_assessment')}.")
        L.append(f"* **What the regime adds** — regime blend vs static inverse-error weights: MAE {verdict('regime', 'phase2_inverse_error')}; "
                 f"heavy CSI {verdict('regime', 'phase2_inverse_error', 'heavy_csi_assessment')}.")
        L.append(f"* Extreme-aware vs equal: MAE {verdict('extreme_aware', 'equal')}; heavy CSI {verdict('extreme_aware', 'equal', 'heavy_csi_assessment')}.")
        L.append("")
    if res_b:
        u = res_b["uncertainty"]

        def vb(a, b, col="mae_assessment"):
            x = u[(u.method == a) & (u.versus == b)]
            return x.iloc[0][col] if len(x) else "n/a"

        f = res_b["final"]
        g, h, ie = _all(f, "GraphCast"), _all(f, "HRES"), _all(f, "phase2_inverse_error")
        if g is not None and h is not None:
            L.append("**Did the AI model improve the blend?** Measurable only in Experiment B, i.e. agreement with ERA5 — not IMD:")
            L.append(f"* Adding GraphCast to the NWP model: inverse-error GraphCast+HRES blend vs HRES alone: MAE "
                     f"{vb('phase2_inverse_error', 'HRES')} ({ie.mae:.3f} vs {h.mae:.3f}); heavy CSI {vb('phase2_inverse_error', 'HRES', 'heavy_csi_assessment')}.")
            L.append(f"* Adding HRES to GraphCast: the same blend vs GraphCast alone: MAE {vb('phase2_inverse_error', 'GraphCast')} "
                     f"({ie.mae:.3f} vs {g.mae:.3f}); heavy CSI {vb('phase2_inverse_error', 'GraphCast', 'heavy_csi_assessment')}. "
                     "Against ERA5, GraphCast alone beats every blend; ERA5 favours GraphCast by construction (see limitations).")
    L.append("")
    return L
