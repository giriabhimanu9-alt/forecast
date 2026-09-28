"""End-to-end Phase 2 run on SYNTHETIC fixtures served by an in-memory fake HTTP client.

Exercises the real GFS/ECMWF/IMD providers, 24 h accumulation, alignment,
split, weights, blends, CSV/PNG/manifest outputs. The numbers produced are
properties of the analytic test fields, not weather results.
"""

import json
from datetime import date

import pandas as pd
import pytest

from app.services.data.ecmwf import ECMWFProvider
from app.services.data.gfs import GFSProvider
from app.services.data.observations import IMDRainfallProvider
from app.services.verification.accumulation import required_lead_hours, verification_windows
from app.services.verification.experiment import ExperimentConfig, plan_init_times, run_experiment
from tests.synthetic import FakeHttp, publish_ecmwf, publish_gfs, write_imd_realtime

EXPECTED_OUTPUTS = {
    "model_scores.csv", "weights.csv", "test_scores.csv", "blend_comparison.csv",
    "skill_comparison.png", "rainfall_forecast_vs_imd.png", "weight_map.png", "error_map.png",
    "experiment_manifest.json", "report.md", "aligned_samples.nc",
}


@pytest.fixture
def experiment(make_settings, tmp_path):
    settings = make_settings()
    cfg = ExperimentConfig(
        name="synthetic_pipeline_test", valid_start=date(2025, 7, 1), valid_end=date(2025, 7, 6),
        output_dir=tmp_path / "results", lead_days=(1, 2), imd_source="realtime",
        ecmwf_base_url="https://example.invalid/ecmwf", bootstrap_samples=200, http_retries=0,
    )
    http = FakeHttp()
    inits = plan_init_times(cfg, "realtime")
    gfs, ecmwf = GFSProvider(settings, http=http), ECMWFProvider(settings, http=http)
    ecmwf.settings = ecmwf.settings.__class__(**{**ecmwf.settings.__dict__, "ecmwf_base_url": cfg.ecmwf_base_url})
    for init in inits:
        leads = required_lead_hours(verification_windows(init, cfg.lead_days))
        publish_gfs(http, gfs, init, leads)
        publish_ecmwf(http, ecmwf, init, leads)
    imd = IMDRainfallProvider(settings, http=http)
    for d in pd.date_range("2025-06-30", "2025-07-07"):
        write_imd_realtime(imd.realtime_path(d.date()), d.date())
    return settings, cfg, http, inits


def test_phase2_pipeline_end_to_end(experiment):
    settings, cfg, http, inits = experiment
    result = run_experiment(settings, cfg, http=http)
    assert result.status == "COMPLETED", result.summary_lines
    assert result.label.startswith("ALIGNMENT/PIPELINE TEST")  # 6 days is too small for a skill claim
    out = cfg.output_dir
    assert EXPECTED_OUTPUTS <= {p.name for p in out.iterdir()}

    manifest = json.loads((out / "experiment_manifest.json").read_text())
    assert manifest["alignment"]["aligned_samples"] == 12  # 6 dates x 2 lead days
    assert manifest["models_available"] == ["GFS", "ECMWF"]
    assert manifest["failed_downloads"] == []
    assert manifest["split"]["n_train_days"] + manifest["split"]["n_test_days"] == 6
    assert manifest["tigge"]["used"] is False
    for key in ("data_sources", "forecast_cycles", "training_period", "test_period", "metrics", "units"):
        assert key in manifest

    w = pd.read_csv(out / "weights.csv")
    assert ((w.weight >= 0) & (w.weight <= 1)).all()
    assert (w.groupby(["method", "region", "lead_day"]).weight.sum().sub(1).abs() < 1e-6).all()

    comp = pd.read_csv(out / "blend_comparison.csv")
    assert set(comp.method) == {"GFS", "ECMWF", "equal", "inverse_error", "regional"}
    # identical synthetic GFS/ECMWF fields -> every method has the same MAE
    allc = comp[comp.lead_day.astype(str) == "all"]
    assert allc.mae.max() - allc.mae.min() < 1e-4
    assert any("Best individual model by MAE" in line for line in result.summary_lines)


def test_second_run_reuses_cache(experiment):
    settings, cfg, http, inits = experiment
    run_experiment(settings, cfg, http=http)
    before = http.downloads()
    run_experiment(settings, cfg, http=http)
    assert http.downloads() == before


def test_missing_model_is_reported_and_experiment_continues(experiment):
    settings, cfg, http, inits = experiment
    for url in [u for u in http.objects if "example.invalid" in u]:
        del http.objects[url]
    result = run_experiment(settings, cfg, http=http)
    assert result.status == "COMPLETED"
    assert result.manifest["models_unavailable"] == ["ECMWF"]
    assert len(result.manifest["failed_downloads"]) == len(inits)
    assert all(r["status"] == "DOWNLOAD FAILED" for r in result.manifest["failed_downloads"])
    assert any("only one model available" in line for line in result.summary_lines)
