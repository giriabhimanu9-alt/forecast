"""Missing-provider fallback: one provider failing must not affect the others."""

import json
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from app.models.schema import DownloadError, InvalidDataError, NotConfiguredError, ProviderStatus
from app.services.data.base import FetchStats, ForecastProvider
from app.services.pipeline import candidate_cycles, resolve_init_time, run_phase1

INIT = datetime(2026, 9, 27, 12)


class StubProvider(ForecastProvider):
    """Stub whose behaviour is chosen per test; values are a synthetic ramp, not weather."""

    def __init__(self, settings, name, behaviour="ok", available=True):
        super().__init__(settings, http=object())
        self.name = name
        self.behaviour = behaviour
        self.available = available

    def is_available(self, init_time, lead_hours):
        return self.available

    def fetch(self, init_time, lead_hours, variables):
        if self.behaviour == "download":
            raise DownloadError(f"{self.name}: server returned 404")
        if self.behaviour == "crash":
            raise RuntimeError("boom")
        return FetchStats(files=[], downloads=1)

    def load_fields(self, files, init_time, lead_hours):
        lats = np.arange(7.0, 21.01, 1.0)
        lons = np.arange(69.0, 81.01, 1.0)
        leads = np.array([np.timedelta64(h, "h") for h in lead_hours], dtype="timedelta64[ns]")
        ramp = np.array(lead_hours, dtype="float32").reshape(-1, 1, 1) * np.ones((1, lats.size, lons.size), "float32")
        if self.behaviour == "negative_rain":
            ramp = -ramp
        da = xr.DataArray(ramp, dims=("lead_time", "latitude", "longitude"),
                          coords={"lead_time": leads, "latitude": lats, "longitude": lons}, attrs={"units": "mm"})
        return {"precipitation": da}


def _settings(make_settings, **kw):
    return make_settings(VARIABLES="precipitation", ENABLE_OBSERVATIONS="false", **kw)


def test_one_failure_does_not_stop_others(make_settings):
    s = _settings(make_settings)
    providers = [
        StubProvider(s, "GOOD"),
        StubProvider(s, "DOWN", behaviour="download"),
        StubProvider(s, "BAD", behaviour="negative_rain"),
        StubProvider(s, "CRASH", behaviour="crash"),
    ]
    report = run_phase1(s, forecast_providers=providers)
    status = {r.name: r.status for r in report.results}
    assert status["GOOD"] is ProviderStatus.AVAILABLE
    assert status["DOWN"] is ProviderStatus.DOWNLOAD_FAILED
    assert status["BAD"] is ProviderStatus.INVALID_DATA
    assert status["CRASH"] is ProviderStatus.INVALID_DATA  # unexpected errors are never "success"
    assert status["IMD"] is ProviderStatus.NOT_CONFIGURED
    assert status["AI"] is ProviderStatus.PLUGIN_ONLY
    assert report.exit_code == 1

    # only the valid dataset is written
    forecasts = sorted(p.name for p in (s.processed_dir / "forecasts").iterdir())
    assert forecasts == ["good_2026092712.nc"]

    saved = json.loads((s.processed_dir / "phase1_report.json").read_text())
    by_name = {p["name"]: p for p in saved["providers"]}
    assert by_name["DOWN"]["status"] == "DOWNLOAD FAILED" and "404" in by_name["DOWN"]["message"]
    assert by_name["GOOD"]["dims"]["lead_time"] == 2


def test_disabled_providers_are_not_configured(make_settings):
    s = _settings(make_settings, ENABLE_GFS="false", ENABLE_ECMWF="false")
    report = run_phase1(s)
    status = {r.name: r.status.value for r in report.results}
    assert status == {"GFS": "NOT CONFIGURED", "ECMWF": "NOT CONFIGURED", "IMD": "NOT CONFIGURED", "AI": "PLUG-IN ONLY"}
    assert report.exit_code == 2  # nothing usable


def test_unknown_ai_provider_reported_not_configured(make_settings):
    s = _settings(make_settings, AI_MODEL_PROVIDER="does-not-exist")
    report = run_phase1(s, forecast_providers=[StubProvider(s, "GOOD")])
    assert report.result("AI").status is ProviderStatus.NOT_CONFIGURED
    assert report.result("GOOD").ok


def test_candidate_cycles():
    now = datetime(2026, 9, 28, 2, 30)
    assert candidate_cycles(now, (0, 12), 3) == [
        datetime(2026, 9, 28, 0), datetime(2026, 9, 27, 12), datetime(2026, 9, 27, 0)
    ]


def test_resolve_latest_prefers_common_cycle(make_settings):
    s = _settings(make_settings, FORECAST_INIT_TIME="latest")

    class Partial(StubProvider):
        def is_available(self, init_time, lead_hours):
            return init_time.hour == 12  # has not published the 00Z run yet

    a, b = StubProvider(s, "A"), Partial(s, "B")
    for p in (a, b):
        p.supported_cycles = (0, 12)
    init, note = resolve_init_time(s, [a, b], now=datetime(2026, 9, 28, 5))
    assert init == datetime(2026, 9, 27, 12)
    assert "all enabled" in note


def test_resolve_latest_falls_back_to_partial(make_settings):
    s = _settings(make_settings, FORECAST_INIT_TIME="latest")
    a, b = StubProvider(s, "A"), StubProvider(s, "B", available=False)
    init, note = resolve_init_time(s, [a, b], now=datetime(2026, 9, 28, 5))
    assert init == datetime(2026, 9, 28, 0)
    assert "only ['A']" in note
