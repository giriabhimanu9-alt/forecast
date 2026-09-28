from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, load_settings  # noqa: E402


@pytest.fixture
def make_settings(tmp_path):
    """Build Settings from an explicit env dict (the real .env file is ignored)."""

    def _make(**overrides: str) -> Settings:
        env = {
            "DATA_DIR": str(tmp_path / "data"),
            "TARGET_LAT_MIN": "8",
            "TARGET_LAT_MAX": "20",
            "TARGET_LON_MIN": "70",
            "TARGET_LON_MAX": "80",
            "TARGET_GRID_RESOLUTION": "0.5",
            "FORECAST_INIT_TIME": "2026-09-27T12",
            "FORECAST_LEAD_HOURS": "6,12",
            "HTTP_RETRIES": "0",
        }
        env.update(overrides)
        return load_settings(env=env)

    return _make
