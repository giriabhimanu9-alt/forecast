"""Provider unit tests. Index lines mirror the real NOAA / ECMWF index formats."""

import json
from datetime import date, datetime

import pytest

from app.models.schema import DownloadError, InvalidDataError, NotConfiguredError
from app.services.data.ai_model import AIModelProvider, create_ai_provider, register_ai_provider, registered_ai_providers
from app.services.data.base import fields_for
from app.services.data.ecmwf import ECMWFProvider
from app.services.data.gfs import GFSProvider
from app.services.data.observations import IMD_DAY_BYTES, IMD_REALTIME_URL, IMDRainfallProvider
from tests.synthetic import FakeHttp, imd_day_bytes

INIT = datetime(2026, 9, 27, 12)

GFS_IDX_F072 = """\
581:423241357:d=2026092712:TMP:2 m above ground:72 hour fcst:
582:424000000:d=2026092712:SPFH:2 m above ground:72 hour fcst:
588:427837059:d=2026092712:UGRD:10 m above ground:72 hour fcst:
589:428809490:d=2026092712:VGRD:10 m above ground:72 hour fcst:
590:429700000:d=2026092712:PRATE:surface:66-72 hour ave fcst:
596:433121767:d=2026092712:APCP:surface:66-72 hour acc fcst:
597:433498494:d=2026092712:APCP:surface:0-3 day acc fcst:
598:433900000:d=2026092712:ACPCP:surface:0-3 day acc fcst:
"""


def test_fields_for_variables():
    assert fields_for(["precipitation", "wind_speed_10m"]) == ["precipitation", "u_10m", "v_10m"]


def test_gfs_urls(make_settings):
    p = GFSProvider(make_settings())
    assert p.grib_url(INIT, 6) == (
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260927/12/atmos/gfs.t12z.pgrb2.0p25.f006"
    )
    assert p.index_url(INIT, 6).endswith(".f006.idx")


def test_gfs_index_selects_accumulation_from_zero(make_settings):
    p = GFSProvider(make_settings())
    entries = p.parse_index(GFS_IDX_F072, 72, ["precipitation", "temperature_2m", "u_10m", "v_10m"])
    assert entries["precipitation"].byte_range.start == 433498494  # "0-3 day", not "66-72 hour" bucket
    assert entries["precipitation"].byte_range.end == 433900000 - 1
    assert entries["temperature_2m"].byte_range.start == 423241357
    assert entries["temperature_2m"].byte_range.end == 424000000 - 1
    assert entries["v_10m"].byte_range.end == 429700000 - 1


def test_gfs_index_last_record_has_open_range(make_settings):
    idx = "1:0:d=2026092712:TMP:2 m above ground:6 hour fcst:\n2:500:d=2026092712:APCP:surface:0-6 hour acc fcst:\n"
    entries = GFSProvider(make_settings()).parse_index(idx, 6, ["precipitation"])
    assert entries["precipitation"].byte_range.end is None
    assert entries["precipitation"].byte_range.header == "bytes=500-"


def test_gfs_index_missing_field_raises(make_settings):
    idx = "1:0:d=2026092712:TMP:2 m above ground:6 hour fcst:\n"
    with pytest.raises(InvalidDataError, match="APCP"):
        GFSProvider(make_settings()).parse_index(idx, 6, ["precipitation"])
    with pytest.raises(InvalidDataError, match="APCP"):  # accumulation for the wrong lead
        GFSProvider(make_settings()).parse_index(GFS_IDX_F072, 48, ["precipitation"])


def test_gfs_malformed_index_raises(make_settings):
    with pytest.raises(InvalidDataError, match="Malformed"):
        GFSProvider(make_settings()).parse_index("not an index", 6, ["precipitation"])


def test_ecmwf_urls_and_index(make_settings):
    p = ECMWFProvider(make_settings())
    assert p.grib_url(INIT, 6) == (
        "https://data.ecmwf.int/forecasts/20260927/12z/ifs/0p25/oper/20260927120000-6h-oper-fc.grib2"
    )
    assert p.index_url(INIT, 6).endswith("20260927120000-6h-oper-fc.index")
    lines = [
        {"levtype": "sfc", "step": "6", "param": "tp", "_offset": 0, "_length": 100},
        {"levtype": "sfc", "step": "6", "param": "2t", "_offset": 100, "_length": 50},
        {"levtype": "pl", "step": "6", "param": "t", "_offset": 150, "_length": 70},
    ]
    entries = p.parse_index("\n".join(json.dumps(x) for x in lines), 6, ["precipitation", "temperature_2m"])
    assert (entries["precipitation"].byte_range.start, entries["precipitation"].byte_range.end) == (0, 99)
    assert (entries["temperature_2m"].byte_range.start, entries["temperature_2m"].byte_range.end) == (100, 149)
    with pytest.raises(InvalidDataError, match="10u"):
        p.parse_index("\n".join(json.dumps(x) for x in lines), 6, ["u_10m"])


def test_lead_time_support(make_settings):
    g, e = GFSProvider(make_settings()), ECMWFProvider(make_settings())
    assert g.supports_lead(INIT, 1) and not g.supports_lead(INIT, 121)
    assert e.supports_lead(INIT, 6) and not e.supports_lead(INIT, 7)
    with pytest.raises(NotConfiguredError, match="06Z"):
        e.ensure_lead_times(datetime(2026, 9, 27, 6), [6])
    with pytest.raises(NotConfiguredError, match=r"\[7\]"):
        e.ensure_lead_times(INIT, [6, 7])


def test_gfs_fetch_reports_unpublished_cycle(make_settings):
    p = GFSProvider(make_settings(), http=FakeHttp())
    with pytest.raises(DownloadError, match="not published or unreachable"):
        p.fetch(INIT, [6], ["precipitation"])


# --------------------------------------------------------------------------- IMD

def test_imd_default_date_range_and_source(make_settings):
    p = IMDRainfallProvider(make_settings(IMD_LOOKBACK_DAYS="3"), today=date(2026, 9, 28))
    assert p.date_range() == (date(2026, 9, 24), date(2026, 9, 26))
    assert p.source_for(date(2026, 9, 1)) == "realtime"
    assert p.source_for(date(2025, 9, 1)) == "archive"


def test_imd_future_dates_not_configured(make_settings):
    p = IMDRainfallProvider(make_settings(IMD_START_DATE="2026-09-27", IMD_END_DATE="2026-09-28"), today=date(2026, 9, 28))
    with pytest.raises(NotConfiguredError, match="not in the past"):
        p.date_range()


def test_imd_unpublished_day_is_download_failure(make_settings):
    s = make_settings(IMD_START_DATE="2026-09-20", IMD_END_DATE="2026-09-20")
    p = IMDRainfallProvider(s, http=FakeHttp(), today=date(2026, 9, 28))
    with pytest.raises(DownloadError, match="not published"):
        p.fetch()


def test_imd_wrong_size_download_rejected_and_not_cached(make_settings):
    s = make_settings(IMD_START_DATE="2026-09-20", IMD_END_DATE="2026-09-20")
    http = FakeHttp()
    http.posts[(IMD_REALTIME_URL, json.dumps({"rain": "20092026"}))] = b"<html>error</html>"
    p = IMDRainfallProvider(s, http=http, today=date(2026, 9, 28))
    with pytest.raises(InvalidDataError, match=str(IMD_DAY_BYTES)):
        p.fetch()
    assert not p.realtime_path(date(2026, 9, 20)).exists()


def test_imd_download_then_cache(make_settings):
    s = make_settings(IMD_START_DATE="2026-09-20", IMD_END_DATE="2026-09-20")
    http = FakeHttp()
    http.posts[(IMD_REALTIME_URL, json.dumps({"rain": "20092026"}))] = imd_day_bytes(date(2026, 9, 20))
    p = IMDRainfallProvider(s, http=http, today=date(2026, 9, 28))
    first = p.fetch()
    assert (first.downloads, first.cache_hits) == (1, 0)
    assert p.realtime_path(date(2026, 9, 20)).with_suffix(".grd.json").exists()  # provenance sidecar
    second = p.fetch()
    assert (second.downloads, second.cache_hits) == (0, 1)


# --------------------------------------------------------------------------- AI plug-in

def test_ai_plugin_only_by_default(make_settings):
    assert create_ai_provider(make_settings()) is None


def test_ai_unknown_provider_not_configured(make_settings):
    with pytest.raises(NotConfiguredError, match="not a registered provider"):
        create_ai_provider(make_settings(AI_MODEL_PROVIDER="graphcast"))


def test_ai_registry_accepts_plugins(make_settings):
    @register_ai_provider("unit-test-ai")
    class _Dummy(AIModelProvider):
        name = "DUMMY-AI"

        def is_available(self, init_time, lead_hours):
            return False

        def fetch(self, init_time, lead_hours, variables):
            raise DownloadError("dummy")

        def load_fields(self, files, init_time, lead_hours):
            return {}

    assert "unit-test-ai" in registered_ai_providers()
    provider = create_ai_provider(make_settings(AI_MODEL_PROVIDER="unit-test-ai"))
    assert isinstance(provider, _Dummy) and provider.kind == "ai_forecast"
