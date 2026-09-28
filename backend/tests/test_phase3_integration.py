"""End-to-end Phase 3 run on SYNTHETIC fixtures (in-memory fake HTTP; never weather data)."""

import json
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from app.models.schema import FORECAST_DIMS, to_standard_forecast
from app.services.data.ecmwf import ECMWFProvider
from app.services.data.gfs import GFSProvider
from app.services.data.observations import IMDRainfallProvider
from app.services.verification.accumulation import required_lead_hours, verification_windows
from app.services.verification.experiment import plan_init_times, provider_class
from app.services.verification.phase3 import Phase3Config, run_phase3
from tests.synthetic import FakeHttp, publish_ecmwf, publish_gfs, write_imd_realtime

EXPECTED = {
    "regimes/regime_assignments.csv", "regimes/regime_statistics.csv", "regimes/regime_model.json",
    "weights/adaptive_weights.csv", "weights/weight_difference_by_regime.csv",
    "verification/model_scores_by_regime.csv", "verification/final_comparison.csv",
    "verification/extreme_rain_scores.csv", "figures/regime_distribution.png", "figures/model_error_by_regime.png",
    "figures/adaptive_weights_by_regime.png", "figures/forecast_comparison.png", "figures/heavy_rain_comparison.png",
    "experiment_manifest.json", "report.md",
}


def test_phase3_end_to_end_on_fixtures(make_settings, tmp_path):
    settings = make_settings()
    cfg = Phase3Config(
        output_dir=tmp_path / "phase3", valid_start=date(2025, 7, 1), train_end=date(2025, 7, 8),
        validation_end=date(2025, 7, 10), valid_end=date(2025, 7, 13), lead_days=(1, 2), ks=(2, 3),
        ecmwf_base_url="https://example.invalid/ecmwf", imd_source="realtime", bootstrap_samples=100,
        min_samples=2, min_points=10,
    )
    http = FakeHttp()
    gfs, ecmwf = GFSProvider(settings, http=http), ECMWFProvider(settings, http=http)
    ecmwf.settings = ecmwf.settings.__class__(**{**ecmwf.settings.__dict__, "ecmwf_base_url": cfg.ecmwf_base_url})
    for init in plan_init_times(cfg.experiment_config(), "realtime"):
        leads = required_lead_hours(verification_windows(init, cfg.lead_days))
        publish_gfs(http, gfs, init, leads)
        publish_ecmwf(http, ecmwf, init, leads)
    imd = IMDRainfallProvider(settings, http=http)
    for d in pd.date_range("2025-06-30", "2025-07-14"):
        write_imd_realtime(imd.realtime_path(d.date()), d.date())

    result = run_phase3(settings, cfg, http=http)
    assert result.status == "COMPLETED", result.summary_lines
    out = cfg.output_dir
    files = {str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file()}
    assert EXPECTED <= files
    m = json.loads((out / "experiment_manifest.json").read_text())
    assert m["periods"]["test"]["valid_dates"] == ["2025-07-11", "2025-07-13"]
    assert m["periods"]["train_plus_validation"]["valid_dates"][1] == "2025-07-10"
    assert m["wind_available"] is True and "wind_arabian_sea" in m["regime_features"]
    assert m["grid_cell_weights"]["supportable"] is False
    w = pd.read_csv(out / "weights" / "adaptive_weights.csv")
    assert (w.groupby(["method", "regime", "lead_day", "region"]).weight.sum().sub(1).abs() < 1e-6).all()
    final = pd.read_csv(out / "verification" / "final_comparison.csv")
    assert set(final.method) == {"GFS", "ECMWF", "equal", "phase2_inverse_error", "regime", "extreme_aware"}
    assert any("no generalisation" in line for line in result.summary_lines)


def test_standard_forecast_view_and_provider_interface(make_settings):
    settings = make_settings()
    for cls in (GFSProvider, ECMWFProvider):
        p = cls(settings)
        assert p.model_name == cls.name
        assert provider_class(cls.name) is cls
    ds = xr.Dataset(
        {"precipitation": (FORECAST_DIMS, np.zeros((1, 1, 2, 2, 2), "float32"), {"units": "mm"})},
        coords={"model": ["GFS"], "time": [np.datetime64("2025-07-01T00", "ns")],
                "lead_time": np.array([3, 27], dtype="timedelta64[h]").astype("timedelta64[ns]"),
                "latitude": [10.0, 11.0], "longitude": [70.0, 71.0]},
    )
    ds = ds.assign_coords(valid_time=ds.time + ds.lead_time)
    da = to_standard_forecast(ds)
    assert da.name == "forecast"
    assert da.dims == ("model_name", "initialization_time", "lead_time", "latitude", "longitude", "variable")
    assert da["variable"].values.tolist() == ["precipitation"]
