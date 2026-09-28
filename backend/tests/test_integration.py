"""End-to-end run of the real GFS / ECMWF / IMD code paths on SYNTHETIC local fixtures.

The fixtures (tests/synthetic.py) are generated analytic fields served from
an in-memory fake HTTP client; they are not weather data and are only ever
written under pytest's tmp directory.
"""

import json
from datetime import date

import numpy as np
import pytest
import xarray as xr

from app.models.schema import FORECAST_DIMS, ProviderStatus, validate_forecast_schema, validate_observation_schema
from app.services.data.ecmwf import ECMWFProvider
from app.services.data.gfs import GFSProvider
from app.services.data.observations import IMDRainfallProvider
from app.services.pipeline import run_phase1
from tests.synthetic import FakeHttp, publish_ecmwf, publish_gfs, write_imd_realtime

LEADS = [6, 12]


@pytest.fixture
def setup(make_settings):
    s = make_settings(IMD_START_DATE="2026-09-19", IMD_END_DATE="2026-09-21", IMD_SOURCE="realtime")
    http = FakeHttp()
    gfs, ecmwf = GFSProvider(s, http=http), ECMWFProvider(s, http=http)
    init = s.forecast_init_time
    publish_gfs(http, gfs, init, LEADS)
    publish_ecmwf(http, ecmwf, init, LEADS)
    imd = IMDRainfallProvider(s, http=http, today=date(2026, 9, 28))
    for day in (date(2026, 9, 19), date(2026, 9, 20), date(2026, 9, 21)):
        write_imd_realtime(imd.realtime_path(day), day)
    return s, http, gfs, ecmwf, imd


def test_end_to_end_pipeline(setup):
    s, http, gfs, ecmwf, imd = setup
    report = run_phase1(s, forecast_providers=[gfs, ecmwf], observation_provider=imd)
    status = {r.name: r.status for r in report.results}
    assert status == {
        "GFS": ProviderStatus.AVAILABLE,
        "ECMWF": ProviderStatus.AVAILABLE,
        "IMD": ProviderStatus.AVAILABLE,
        "AI": ProviderStatus.PLUGIN_ONLY,
    }, [r.message for r in report.results]
    assert report.exit_code == 0

    g = xr.open_dataset(s.processed_dir / "forecasts" / "gfs_2026092712.nc")
    e = xr.open_dataset(s.processed_dir / "forecasts" / "ecmwf_2026092712.nc")
    for ds in (g, e):
        validate_forecast_schema(ds)
        assert ds.precipitation.dims == FORECAST_DIMS
        assert ds.sizes["latitude"] == 25 and ds.sizes["longitude"] == 21  # 8..20 x 70..80 at 0.5
        assert ds.attrs["regrid_method"] == "bilinear"  # 1 deg source -> 0.5 deg target
        assert float(ds.wind_speed_10m.min()) == pytest.approx(5.0, abs=1e-3)

    # Synthetic accumulation = 0.1 * lead * (1 + lat/30). GFS must have picked the 0-N
    # record (not the bucket = 123) and ECMWF must have converted metres -> mm.
    expected = 0.1 * 12 * (1 + 15.0 / 30.0)
    for ds in (g, e):
        val = float(ds.precipitation.isel(model=0, time=0).sel(lead_time=np.timedelta64(12, "h"), latitude=15.0, longitude=75.0))
        assert val == pytest.approx(expected, rel=1e-3)
    # Kelvin -> Celsius
    assert float(g.temperature_2m.isel(model=0, time=0).sel(lead_time=np.timedelta64(6, "h"), latitude=10.0, longitude=75.0)) == pytest.approx(
        300.0 - 5.0 - 273.15, abs=0.05
    )

    o = xr.open_dataset(s.processed_dir / "observations" / "imd_rainfall_20260919_20260921.nc")
    validate_observation_schema(o)
    assert o.sizes["time"] == 3
    assert o.time.values[0] == np.datetime64("2026-09-19T03:00", "ns")  # real-time label D -> ends 03 UTC D
    assert float(o.precipitation.sel(latitude=15.0, longitude=75.0).isel(time=1)) == 20.0
    assert np.isnan(float(o.precipitation.sel(latitude=8.0, longitude=75.0).isel(time=0)))  # outside "land"

    saved = json.loads((s.processed_dir / "phase1_report.json").read_text())
    assert saved["exit_code"] == 0
    gfs_entry = next(p for p in saved["providers"] if p["name"] == "GFS")
    assert gfs_entry["lead_hours"] == LEADS
    assert gfs_entry["downloads"] == 8 and gfs_entry["cache_hits"] == 0


def test_second_run_uses_cache(setup):
    s, http, gfs, ecmwf, imd = setup
    run_phase1(s, forecast_providers=[gfs, ecmwf], observation_provider=imd)
    before = http.downloads()
    report = run_phase1(s, forecast_providers=[gfs, ecmwf], observation_provider=imd)
    assert http.downloads() == before  # nothing re-downloaded
    assert report.result("GFS").cache_hits == 8 and report.result("GFS").downloads == 0


def test_corrupted_cache_file_is_redownloaded(setup):
    s, http, gfs, ecmwf, imd = setup
    run_phase1(s, forecast_providers=[gfs], observation_provider=imd)
    victim = gfs.cache_path(s.forecast_init_time, 6, "precipitation")
    victim.write_bytes(victim.read_bytes()[:50])  # simulate an interrupted write
    report = run_phase1(s, forecast_providers=[gfs], observation_provider=imd)
    assert report.result("GFS").ok
    assert report.result("GFS").downloads == 1


def test_missing_cycle_is_download_failed_not_success(setup):
    s, http, gfs, ecmwf, imd = setup
    for url in [u for u in http.objects if "ecmwf" in u]:
        del http.objects[url]
    report = run_phase1(s, forecast_providers=[gfs, ecmwf], observation_provider=imd)
    assert report.result("ECMWF").status is ProviderStatus.DOWNLOAD_FAILED
    assert report.result("GFS").ok and report.result("IMD").ok
    assert not (s.processed_dir / "forecasts" / "ecmwf_2026092712.nc").exists()
    assert report.exit_code == 1
