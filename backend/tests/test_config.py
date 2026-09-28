from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from app.config import BACKEND_ROOT, ConfigError, load_settings


def test_defaults_match_india_mvp_grid():
    s = load_settings(env={})
    assert (s.grid.lat_min, s.grid.lat_max, s.grid.lon_min, s.grid.lon_max) == (5, 38, 68, 98)
    assert s.grid.resolution == 0.25
    assert s.grid.shape == (133, 121)
    assert s.lead_hours == tuple(range(6, 73, 6))
    assert s.variables == ("precipitation", "temperature_2m", "wind_speed_10m")
    assert s.forecast_init_time is None  # latest
    assert s.data_dir == (BACKEND_ROOT / "data").resolve()
    assert s.enable_gfs and s.enable_ecmwf and s.enable_observations
    assert s.ai_model_provider == ""


def test_env_overrides(make_settings, tmp_path):
    s = make_settings(TARGET_GRID_RESOLUTION="1.0", FORECAST_LEAD_HOURS="24,48,72", ENABLE_ECMWF="false",
                      IMD_START_DATE="2025-07-01", IMD_END_DATE="2025-07-03")
    assert s.grid.resolution == 1.0
    assert s.lead_hours == (24, 48, 72)
    assert s.enable_ecmwf is False
    assert s.imd_start_date == date(2025, 7, 1) and s.imd_end_date == date(2025, 7, 3)
    assert s.forecast_init_time == datetime(2026, 9, 27, 12)
    assert s.raw_dir == tmp_path / "data" / "raw"


def test_grid_axes_are_exact():
    s = load_settings(env={})
    lats, lons = s.grid.latitudes, s.grid.longitudes
    assert lats[0] == 5.0 and lats[-1] == 38.0
    assert np.allclose(np.diff(lats), 0.25) and np.allclose(np.diff(lons), 0.25)
    assert 20.25 in lats  # no float drift


def test_env_file_is_read_and_env_wins(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nTARGET_LAT_MIN=6\nLOG_LEVEL=debug\n", encoding="utf-8")
    s = load_settings(env={"TARGET_LAT_MIN": "7"}, env_file=env_file)
    assert s.grid.lat_min == 7.0
    assert s.log_level == "DEBUG"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"TARGET_LAT_MIN": "40"}, "latitude bounds"),
        ({"TARGET_LON_MAX": "200"}, "longitude bounds"),
        ({"TARGET_GRID_RESOLUTION": "0.3"}, "whole multiple"),
        ({"TARGET_GRID_RESOLUTION": "-1"}, "must be > 0"),
        ({"FORECAST_LEAD_HOURS": "0,6"}, "lead 0"),
        ({"FORECAST_LEAD_HOURS": "abc"}, "FORECAST_LEAD_HOURS"),
        ({"FORECAST_INIT_TIME": "yesterday"}, "FORECAST_INIT_TIME"),
        ({"FORECAST_INIT_TIME": "2026-09-27T03"}, "cycle"),
        ({"ENABLE_GFS": "maybe"}, "boolean"),
        ({"VARIABLES": "precipitation,humidity"}, "Unsupported"),
        ({"VARIABLES": "temperature_2m"}, "must include 'precipitation'"),
        ({"IMD_START_DATE": "2025-01-01"}, "both"),
        ({"IMD_START_DATE": "2025-02-01", "IMD_END_DATE": "2025-01-01"}, "<="),
        ({"IMD_SOURCE": "ftp"}, "IMD_SOURCE"),
        ({"TARGET_LAT_MIN": "five"}, "must be a number"),
    ],
)
def test_invalid_config_rejected(make_settings, overrides, message):
    with pytest.raises(ConfigError, match=message):
        make_settings(**overrides)


def test_lead_hour_range_syntax(make_settings):
    assert make_settings(FORECAST_LEAD_HOURS="3-12/3,24").lead_hours == (3, 6, 9, 12, 24)


def test_relative_data_dir_resolves_against_backend():
    s = load_settings(env={"DATA_DIR": "some/where"})
    assert s.data_dir == (BACKEND_ROOT / "some" / "where").resolve()
    assert isinstance(s.data_dir, Path)
