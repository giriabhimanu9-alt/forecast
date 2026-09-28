"""Configuration for the Phase 1 ingestion pipeline.

Values come from (highest priority first):
  1. process environment variables
  2. a ``.env`` file in the backend directory (or the path in ``PS81_ENV_FILE``)
  3. the defaults below

Nothing here contains secrets; the Phase 1 sources are all open-access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_VARIABLES = (
    "precipitation", "temperature_2m", "wind_speed_10m",
    # Phase 4 atmospheric regime features (GFS 1.0 deg product)
    "mslp", "u_850hpa", "precipitable_water",
)
# Phase 1 live-run default (unchanged); atmospheric fields are requested explicitly by Phase 4.
DEFAULT_VARIABLES = ("precipitation", "temperature_2m", "wind_speed_10m")


class ConfigError(ValueError):
    """Raised when configuration is missing or inconsistent."""


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"{path}:{lineno}: expected KEY=VALUE, got {raw!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} must be a boolean (true/false), got {raw!r}")


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _parse_lead_hours(raw: str) -> tuple[int, ...]:
    """Parse ``"6,12,24"`` or a range ``"6-72/6"`` into sorted unique hours."""
    raw = raw.strip()
    hours: list[int] = []
    try:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                span, _, step = part.partition("/")
                start, _, stop = span.partition("-")
                hours.extend(range(int(start), int(stop) + 1, int(step or 1)))
            else:
                hours.append(int(part))
    except ValueError as exc:
        raise ConfigError(
            f"FORECAST_LEAD_HOURS must look like '6,12,24' or '6-72/6', got {raw!r}"
        ) from exc
    if not hours:
        raise ConfigError("FORECAST_LEAD_HOURS is empty")
    return tuple(sorted(set(hours)))


def _parse_date(key: str, raw: str | None) -> date | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{key} must be YYYY-MM-DD, got {raw!r}") from exc


@dataclass(frozen=True)
class TargetGrid:
    """Regular lat/lon grid that every dataset is normalised onto."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution: float
    name: str = "India"

    def _axis(self, lo: float, hi: float) -> np.ndarray:
        n = int(round((hi - lo) / self.resolution)) + 1
        return np.round(np.linspace(lo, hi, n), 6)

    @property
    def latitudes(self) -> np.ndarray:
        return self._axis(self.lat_min, self.lat_max)

    @property
    def longitudes(self) -> np.ndarray:
        return self._axis(self.lon_min, self.lon_max)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.latitudes.size, self.longitudes.size)

    def validate(self) -> None:
        if not (-90 <= self.lat_min < self.lat_max <= 90):
            raise ConfigError(
                f"Invalid latitude bounds: TARGET_LAT_MIN={self.lat_min}, TARGET_LAT_MAX={self.lat_max}"
            )
        if not (-180 <= self.lon_min < self.lon_max <= 180):
            raise ConfigError(
                "Invalid longitude bounds (use -180..180): "
                f"TARGET_LON_MIN={self.lon_min}, TARGET_LON_MAX={self.lon_max}"
            )
        if self.resolution <= 0:
            raise ConfigError(f"TARGET_GRID_RESOLUTION must be > 0, got {self.resolution}")
        for lo, hi, label in (
            (self.lat_min, self.lat_max, "latitude"),
            (self.lon_min, self.lon_max, "longitude"),
        ):
            steps = (hi - lo) / self.resolution
            if abs(steps - round(steps)) > 1e-6:
                raise ConfigError(
                    f"The {label} extent {lo}..{hi} is not a whole multiple of "
                    f"TARGET_GRID_RESOLUTION={self.resolution}"
                )

    def as_dict(self) -> dict[str, float | str | list[int]]:
        return {
            "name": self.name,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "resolution": self.resolution,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    grid: TargetGrid
    variables: tuple[str, ...]
    forecast_init_time: datetime | None  # None means "latest available"
    lead_hours: tuple[int, ...]
    latest_search_cycles: int

    enable_gfs: bool
    gfs_base_url: str

    enable_ecmwf: bool
    ecmwf_base_url: str

    enable_observations: bool
    imd_source: str  # auto | realtime | archive
    imd_start_date: date | None
    imd_end_date: date | None
    imd_lookback_days: int

    ai_model_provider: str  # empty = plug-in only

    http_timeout: float
    http_retries: int
    log_level: str

    env_file: Path | None = field(default=None, compare=False)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def validate(self) -> None:
        self.grid.validate()
        unknown = [v for v in self.variables if v not in SUPPORTED_VARIABLES]
        if unknown:
            raise ConfigError(
                f"Unsupported VARIABLES {unknown}; supported: {', '.join(SUPPORTED_VARIABLES)}"
            )
        if "precipitation" not in self.variables:
            raise ConfigError("VARIABLES must include 'precipitation' (primary Phase 1 variable)")
        if any(h <= 0 for h in self.lead_hours):
            raise ConfigError(
                "FORECAST_LEAD_HOURS must be > 0: accumulated precipitation is not "
                "defined at the analysis time (lead 0)"
            )
        if self.forecast_init_time is not None and self.forecast_init_time.hour % 6 != 0:
            raise ConfigError("FORECAST_INIT_TIME must be on a 00/06/12/18 UTC cycle")
        if self.imd_source not in {"auto", "realtime", "archive"}:
            raise ConfigError(f"IMD_SOURCE must be auto|realtime|archive, got {self.imd_source!r}")
        if (self.imd_start_date is None) != (self.imd_end_date is None):
            raise ConfigError("Set both IMD_START_DATE and IMD_END_DATE, or neither")
        if self.imd_start_date and self.imd_end_date and self.imd_start_date > self.imd_end_date:
            raise ConfigError("IMD_START_DATE must be <= IMD_END_DATE")
        if self.imd_lookback_days < 1:
            raise ConfigError("IMD_LOOKBACK_DAYS must be >= 1")
        if self.latest_search_cycles < 1:
            raise ConfigError("LATEST_SEARCH_CYCLES must be >= 1")
        if self.http_timeout <= 0 or self.http_retries < 0:
            raise ConfigError("HTTP_TIMEOUT must be > 0 and HTTP_RETRIES >= 0")

    def as_dict(self) -> dict[str, object]:
        return {
            "data_dir": str(self.data_dir),
            "grid": self.grid.as_dict(),
            "variables": list(self.variables),
            "forecast_init_time": (
                self.forecast_init_time.isoformat() if self.forecast_init_time else "latest"
            ),
            "lead_hours": list(self.lead_hours),
            "enable_gfs": self.enable_gfs,
            "gfs_base_url": self.gfs_base_url,
            "enable_ecmwf": self.enable_ecmwf,
            "ecmwf_base_url": self.ecmwf_base_url,
            "enable_observations": self.enable_observations,
            "imd_source": self.imd_source,
            "imd_start_date": self.imd_start_date.isoformat() if self.imd_start_date else None,
            "imd_end_date": self.imd_end_date.isoformat() if self.imd_end_date else None,
            "imd_lookback_days": self.imd_lookback_days,
            "ai_model_provider": self.ai_model_provider or None,
        }


def load_settings(
    env: Mapping[str, str] | None = None, env_file: Path | None = None
) -> Settings:
    """Build and validate :class:`Settings`.

    ``env`` defaults to ``os.environ`` merged over the ``.env`` file.
    Passing ``env`` explicitly (as tests do) skips the ``.env`` file unless
    ``env_file`` is also given.
    """
    if env is None:
        env_file = env_file or Path(os.environ.get("PS81_ENV_FILE", BACKEND_ROOT / ".env"))
        merged = {**_parse_env_file(env_file), **os.environ}
    else:
        merged = {**(_parse_env_file(env_file) if env_file else {}), **env}

    data_dir = Path(merged.get("DATA_DIR") or "./data")
    if not data_dir.is_absolute():
        data_dir = (BACKEND_ROOT / data_dir).resolve()

    init_raw = (merged.get("FORECAST_INIT_TIME") or "latest").strip()
    if init_raw.lower() == "latest":
        init_time = None
    else:
        try:
            init_time = datetime.fromisoformat(init_raw.replace("Z", ""))
        except ValueError as exc:
            raise ConfigError(
                f"FORECAST_INIT_TIME must be 'latest' or ISO like 2026-09-27T00, got {init_raw!r}"
            ) from exc
        init_time = init_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)

    variables = tuple(
        v.strip()
        for v in (merged.get("VARIABLES") or ",".join(DEFAULT_VARIABLES)).split(",")
        if v.strip()
    )

    settings = Settings(
        data_dir=data_dir,
        grid=TargetGrid(
            lat_min=_get_float(merged, "TARGET_LAT_MIN", 5.0),
            lat_max=_get_float(merged, "TARGET_LAT_MAX", 38.0),
            lon_min=_get_float(merged, "TARGET_LON_MIN", 68.0),
            lon_max=_get_float(merged, "TARGET_LON_MAX", 98.0),
            resolution=_get_float(merged, "TARGET_GRID_RESOLUTION", 0.25),
            name=merged.get("TARGET_REGION_NAME") or "India",
        ),
        variables=variables,
        forecast_init_time=init_time,
        lead_hours=_parse_lead_hours(merged.get("FORECAST_LEAD_HOURS") or "6-72/6"),
        latest_search_cycles=_get_int(merged, "LATEST_SEARCH_CYCLES", 4),
        enable_gfs=_get_bool(merged, "ENABLE_GFS", True),
        gfs_base_url=(merged.get("GFS_BASE_URL") or "https://noaa-gfs-bdp-pds.s3.amazonaws.com").rstrip("/"),
        enable_ecmwf=_get_bool(merged, "ENABLE_ECMWF", True),
        ecmwf_base_url=(merged.get("ECMWF_BASE_URL") or "https://data.ecmwf.int/forecasts").rstrip("/"),
        enable_observations=_get_bool(merged, "ENABLE_OBSERVATIONS", True),
        imd_source=(merged.get("IMD_SOURCE") or "auto").strip().lower(),
        imd_start_date=_parse_date("IMD_START_DATE", merged.get("IMD_START_DATE")),
        imd_end_date=_parse_date("IMD_END_DATE", merged.get("IMD_END_DATE")),
        imd_lookback_days=_get_int(merged, "IMD_LOOKBACK_DAYS", 7),
        ai_model_provider=(merged.get("AI_MODEL_PROVIDER") or "").strip(),
        http_timeout=_get_float(merged, "HTTP_TIMEOUT", 120.0),
        http_retries=_get_int(merged, "HTTP_RETRIES", 3),
        log_level=(merged.get("LOG_LEVEL") or "INFO").upper(),
        env_file=env_file,
    )
    settings.validate()
    return settings
