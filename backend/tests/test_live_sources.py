"""Opt-in smoke tests against the real public servers (no data is saved).

Run with:  PS81_RUN_NETWORK_TESTS=1 pytest -m network
"""

import os
from datetime import date, datetime, timedelta, timezone

import pytest

from app.services.data.ecmwf import ECMWFProvider
from app.services.data.gfs import GFSProvider
from app.services.data.observations import IMD_DAY_BYTES, IMD_REALTIME_URL
from app.services.http import HttpClient
from app.services.pipeline import candidate_cycles

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("PS81_RUN_NETWORK_TESTS") != "1", reason="set PS81_RUN_NETWORK_TESTS=1"),
]


@pytest.mark.parametrize("cls", [GFSProvider, ECMWFProvider])
def test_recent_cycle_index_is_published(make_settings, cls):
    provider = cls(make_settings(HTTP_RETRIES="2"))
    cycles = candidate_cycles(datetime.now(timezone.utc), (0, 12), 4)
    assert any(provider.is_available(c, [6]) for c in cycles), f"{cls.name}: no cycle published in the last 2 days"


def test_imd_realtime_day_has_expected_size():
    day = date.today() - timedelta(days=3)
    content = HttpClient(timeout=60, retries=2).post_bytes(IMD_REALTIME_URL, {"rain": f"{day:%d%m%Y}"})
    assert len(content) == IMD_DAY_BYTES
