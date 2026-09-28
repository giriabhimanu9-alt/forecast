"""Phase 5: operational blend service and API (hand-built inputs; no network)."""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from app.services import operational as op
from app.services.verification.production import NoForecastAvailable
from app.services.verification.regions import region_map

LATS = np.arange(8.0, 36.0, 2.0)
LONS = np.arange(68.0, 98.0, 2.0)


def _field(v):
    return xr.DataArray(np.full((LATS.size, LONS.size), v, float), dims=("latitude", "longitude"),
                        coords={"latitude": LATS, "longitude": LONS})


def _table():
    rows = []
    for region in op.REGIONS:
        w = 0.8 if region == "South" else 0.4
        rows += [{"method": "regime", "regime": 1, "lead_day": 1, "region": region, "model": "GFS", "weight": w, "training_error": 5.0, "sample_count": 40},
                 {"method": "regime", "regime": 1, "lead_day": 1, "region": region, "model": "ECMWF", "weight": 1 - w, "training_error": 6.0, "sample_count": 40}]
    return pd.DataFrame(rows)


def test_regional_blend_uses_region_weights():
    regions = region_map(LATS, LONS)
    out, meta = op.blend_fields({"GFS": _field(10.0), "ECMWF": _field(20.0)}, 1, 1, _table(), regions)
    south = out.where(regions == "South").max()
    north = out.where(regions == "North").max()
    assert float(south) == pytest.approx(0.8 * 10 + 0.2 * 20)
    assert float(north) == pytest.approx(0.4 * 10 + 0.6 * 20)
    assert meta["regional_weights"]["South"]["GFS"] == 0.8 and not meta["fallback_used"]
    assert meta["available_models"] == ["GFS", "ECMWF"] and meta["sample_count"] == 40


def test_missing_model_fallback_and_no_model():
    regions = region_map(LATS, LONS)
    out, meta = op.blend_fields({"GFS": None, "ECMWF": _field(20.0)}, 1, 1, _table(), regions)
    assert float(out.max()) == 20.0 and meta["missing_models"] == ["GFS"] and meta["fallback_used"]
    with pytest.raises(NoForecastAvailable):
        op.blend_fields({"GFS": None, "ECMWF": None}, 1, 1, _table(), regions)


def test_heavy_rain_alerts():
    regions = region_map(LATS, LONS)
    land = xr.ones_like(_field(0), dtype=bool)
    f = _field(10.0)
    f.loc[dict(latitude=slice(8, 14))] = 80.0  # South is heavy
    a = op.alerts_for(f, land, regions)
    assert a["cells"] > 0 and set(a["regions"]) == {"South"} and a["max_mm"] == 80.0


def test_latest_cycle_is_00utc_and_skips_unpublished():
    class P:
        def __init__(self, ok):
            self.ok = ok

        def is_available(self, init, leads):
            return init in self.ok

    now = datetime(2026, 9, 28, 5)
    d27, d28 = datetime(2026, 9, 27), datetime(2026, 9, 28)
    assert op.latest_cycle([P({d27})], [75], now=now) == d27
    # GFS has today's run, ECMWF not yet: prefer yesterday's complete (validated) blend
    assert op.latest_cycle([P({d27, d28}), P({d27})], [75], now=now) == d27
    # no complete cycle at all: fall back to the newest single-model cycle
    assert op.latest_cycle([P({d28}), P(set())], [75], now=now) == d28
    assert op.latest_cycle([P(set())], [75], now=now) is None


@pytest.fixture
def client(make_settings, monkeypatch):
    from app.api import server

    s = make_settings()
    monkeypatch.setattr(server, "settings", s)
    return TestClient(server.app), s


def test_api_without_runs(client):
    c, _ = client
    assert c.get("/api/health").json()["runs"] == 0
    assert c.get("/api/forecast").status_code == 404
    assert "Monsoon Blend Desk" in c.get("/").text


def test_api_serves_saved_run(client):
    c, s = client
    store = op.store_dir(s)
    store.mkdir(parents=True)
    ds = xr.Dataset({"blend_day1": _field(12.0).where(_field(0) == 0), "GFS_day1": _field(10.0)})
    ds.to_netcdf(store / "2026092700.nc")
    (store / "2026092700.json").write_text(json.dumps({"init_time": "2026-09-27T00:00:00", "lead_days": []}))
    assert c.get("/api/forecast/runs").json() == ["2026092700"]
    assert c.get("/api/forecast").json()["init_time"] == "2026-09-27T00:00:00"
    f = c.get("/api/forecast/field", params={"product": "blend", "lead_day": 1, "stride": 2}).json()
    assert f["units"] == "mm/24h" and f["values"][0][0] == 12.0 and len(f["lat"]) == 7
    assert c.get("/api/forecast/field", params={"product": "ECMWF", "lead_day": 1}).status_code == 404
    assert c.get("/api/forecast", params={"run": "1999010100"}).status_code == 404


@pytest.mark.skipif(not op.REGIME_WEIGHTS.is_file(), reason="Phase 4 results not present")
def test_api_phase4_results_kept_separate(client):
    c, _ = client
    w = c.get("/api/weights/regional").json()
    assert {r["model"] for r in w} == {"GFS", "ECMWF"}  # GraphCast is not an operational member
    v = c.get("/api/validation").json()
    assert "IMD" in v["reference"] and "GraphCast" not in v["scores"]
    b = c.get("/api/benchmark/ai").json()
    assert "ERA5" in b["reference"] and "Not comparable with IMD" in b["warning"] and "GraphCast" in b["scores"]


def test_api_run_without_grids_is_404(client):
    c, s = client
    store = op.store_dir(s)
    store.mkdir(parents=True)
    (store / "2026092700.json").write_text(json.dumps({"init_time": "2026-09-27T00:00:00", "lead_days": [{"lead_day": 1, "error": "no model available"}]}))
    r = c.get("/api/forecast/field", params={"product": "blend", "lead_day": 1})
    assert r.status_code == 404 and "no forecast grids" in r.json()["detail"]


def test_dashboard_map_assets(client):
    c, _ = client
    html = c.get("/").text
    assert 'id="map"' in html and "leaflet" in html.lower() and 'id="opacity"' in html
    r = c.get("/static/india_states.geojson", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip"
    g = r.json()
    assert g["type"] == "FeatureCollection" and len(g["features"]) == 36 and "CC BY 2.5 IN" in g["attribution"]
