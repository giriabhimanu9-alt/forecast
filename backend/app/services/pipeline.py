"""Phase 1 pipeline: fetch -> normalise -> regrid -> validate -> save -> report.

Each provider runs in isolation: a failure is recorded with an explicit
status (NOT CONFIGURED / DOWNLOAD FAILED / INVALID DATA) and never stops
the other providers. Only datasets that pass validation are written.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import requests
import xarray as xr

from app.config import Settings
from app.models.schema import DataPipelineError, NotConfiguredError, ProviderResult, ProviderStatus
from app.services.data import validation
from app.services.data.ai_model import create_ai_provider
from app.services.data.base import ForecastProvider
from app.services.data.ecmwf import ECMWFProvider
from app.services.data.gfs import GFSProvider
from app.services.data.observations import IMDRainfallProvider

log = logging.getLogger(__name__)

REPORT_NAME = "phase1_report.json"


@dataclass
class PipelineReport:
    run_id: str
    started_at: str
    settings: dict[str, Any]
    init_time: str | None = None
    init_time_note: str = ""
    results: list[ProviderResult] = field(default_factory=list)
    finished_at: str = ""
    report_path: str = ""

    def result(self, name: str) -> ProviderResult | None:
        return next((r for r in self.results if r.name == name), None)

    @property
    def exit_code(self) -> int:
        """0 = every enabled provider available, 1 = partial, 2 = nothing usable."""
        enabled = [r for r in self.results if r.status not in (ProviderStatus.NOT_CONFIGURED, ProviderStatus.PLUGIN_ONLY)]
        if not any(r.ok for r in self.results):
            return 2
        return 0 if all(r.ok for r in enabled) else 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": 1,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "init_time": self.init_time,
            "init_time_note": self.init_time_note,
            "settings": self.settings,
            "providers": [r.as_dict() for r in self.results],
            "exit_code": self.exit_code,
        }


# --------------------------------------------------------------------------- helpers


def _isoformat(values: Iterable[np.datetime64]) -> list[str]:
    return [pd.Timestamp(v).isoformat() for v in values]


def describe_dataset(ds: xr.Dataset) -> dict[str, Any]:
    info: dict[str, Any] = {
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "variables": {name: da.attrs.get("units") for name, da in ds.data_vars.items()},
    }
    times = ds.time.values
    info["time_range"] = _isoformat([times.min(), times.max()])
    if "lead_time" in ds.coords:
        info["lead_hours"] = [int(h) for h in ds.lead_time.values / np.timedelta64(1, "h")]
        info["valid_time_range"] = _isoformat([ds.valid_time.values.min(), ds.valid_time.values.max()])
    if "period_start" in ds.coords:
        info["accumulation_windows"] = [
            [pd.Timestamp(s).isoformat(), pd.Timestamp(e).isoformat()]
            for s, e in zip(ds.period_start.values, ds.time.values)
        ]
    return info


def save_netcdf(ds: xr.Dataset, path: Path) -> Path:
    """Write atomically with compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in ds.data_vars}
    tmp = path.with_name(path.name + ".part")
    ds.to_netcdf(tmp, encoding=encoding, engine="netcdf4")
    tmp.replace(path)
    return path


def _run_guarded(result: ProviderResult, fn: Callable[[], None]) -> None:
    """Run ``fn`` and map any failure to an explicit provider status."""
    try:
        fn()
    except DataPipelineError as exc:
        result.status = exc.status
        result.message = str(exc)
        log.error("%s: %s - %s", result.name, exc.status.value, exc)
    except requests.RequestException as exc:
        result.status = ProviderStatus.DOWNLOAD_FAILED
        result.message = f"network error: {exc}"
        log.error("%s: DOWNLOAD FAILED - %s", result.name, exc)
    except Exception as exc:  # unexpected: never report as success
        result.status = ProviderStatus.INVALID_DATA
        result.message = f"unexpected {type(exc).__name__}: {exc}"
        log.error("%s: unexpected error\n%s", result.name, traceback.format_exc())


# --------------------------------------------------------------------------- init-time resolution


def candidate_cycles(now: datetime, cycles: Iterable[int], count: int) -> list[datetime]:
    cycles = sorted(set(cycles))
    t = now.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    out: list[datetime] = []
    while len(out) < count:
        if t.hour in cycles:
            out.append(t)
        t -= timedelta(hours=1)
    return out


def resolve_init_time(
    settings: Settings, providers: list[ForecastProvider], now: datetime | None = None
) -> tuple[datetime | None, str]:
    if settings.forecast_init_time is not None:
        return settings.forecast_init_time, "configured via FORECAST_INIT_TIME"
    if not providers:
        return None, "no forecast providers enabled"
    cycles = set.intersection(*(set(getattr(p, "supported_cycles", (0, 12))) for p in providers))
    now = now or datetime.now(timezone.utc)
    candidates = candidate_cycles(now, cycles, settings.latest_search_cycles)
    partial: tuple[datetime, list[str]] | None = None
    for cand in candidates:
        available = [p.name for p in providers if p.is_available(cand, settings.lead_hours)]
        log.info("Cycle %s: published by %s", f"{cand:%Y-%m-%d %H}Z", available or "none")
        if len(available) == len(providers):
            return cand, "latest cycle published by all enabled forecast providers"
        if available and partial is None:
            partial = (cand, available)
    if partial:
        return partial[0], f"no common cycle in last {len(candidates)} checked; only {partial[1]} publish {partial[0]:%Y-%m-%d %H}Z"
    return candidates[0], f"no provider has published any of the last {len(candidates)} cycles"


# --------------------------------------------------------------------------- runners


def run_forecast_provider(
    provider: ForecastProvider, settings: Settings, init_time: datetime | None
) -> ProviderResult:
    result = ProviderResult(provider.name, provider.kind, ProviderStatus.INVALID_DATA)

    def work() -> None:
        if init_time is None:
            raise NotConfiguredError("no forecast initialisation time could be resolved")
        ds, stats = provider.get_forecast(init_time, settings.lead_hours, settings.variables)
        result.raw_files = [str(p) for p in stats.files]
        result.cache_hits, result.downloads = stats.cache_hits, stats.downloads
        ds, value_stats = validation.validate_forecast_dataset(ds, settings.variables)
        out = settings.processed_dir / "forecasts" / f"{provider.name.lower()}_{init_time:%Y%m%d%H}.nc"
        save_netcdf(ds, out)
        result.dataset = ds
        result.output_files = [str(out)]
        result.details.update(describe_dataset(ds), statistics=value_stats)
        result.status = ProviderStatus.AVAILABLE
        result.message = f"{len(stats.files)} raw GRIB messages ({stats.downloads} downloaded, {stats.cache_hits} cached)"

    _run_guarded(result, work)
    return result


def run_observation_provider(provider: IMDRainfallProvider, settings: Settings) -> ProviderResult:
    result = ProviderResult(provider.name, provider.kind, ProviderStatus.INVALID_DATA)

    def work() -> None:
        ds, stats = provider.get_observations()
        result.raw_files = [str(p) for p in stats.files]
        result.cache_hits, result.downloads = stats.cache_hits, stats.downloads
        ds, value_stats = validation.validate_observation_dataset(ds)
        start, end = sorted(stats.source_by_day)[0], sorted(stats.source_by_day)[-1]
        out = settings.processed_dir / "observations" / f"imd_rainfall_{start:%Y%m%d}_{end:%Y%m%d}.nc"
        save_netcdf(ds, out)
        result.dataset = ds
        result.output_files = [str(out)]
        result.details.update(
            describe_dataset(ds),
            statistics=value_stats,
            imd_routes=sorted(set(stats.source_by_day.values())),
            source_dates=[start.isoformat(), end.isoformat()],
        )
        result.status = ProviderStatus.AVAILABLE
        result.message = f"{len(stats.source_by_day)} days ({stats.downloads} files downloaded, {stats.cache_hits} cached)"

    _run_guarded(result, work)
    return result


def run_phase1(
    settings: Settings,
    forecast_providers: list[ForecastProvider] | None = None,
    observation_provider: IMDRainfallProvider | None = None,
    now: datetime | None = None,
) -> PipelineReport:
    started = datetime.now(timezone.utc)
    report = PipelineReport(
        run_id=started.strftime("%Y%m%dT%H%M%SZ"),
        started_at=started.isoformat(timespec="seconds"),
        settings=settings.as_dict(),
    )

    # --- forecast providers
    if forecast_providers is None:
        forecast_providers = []
        for enabled, cls, flag in (
            (settings.enable_gfs, GFSProvider, "ENABLE_GFS"),
            (settings.enable_ecmwf, ECMWFProvider, "ENABLE_ECMWF"),
        ):
            if enabled:
                forecast_providers.append(cls(settings))
            else:
                report.results.append(
                    ProviderResult(cls.name, "forecast", ProviderStatus.NOT_CONFIGURED, f"disabled ({flag}=false)")
                )

    init_time, note = resolve_init_time(settings, forecast_providers, now)
    report.init_time = init_time.isoformat() if init_time else None
    report.init_time_note = note
    log.info("Forecast initialisation time: %s (%s)", report.init_time, note)

    for provider in forecast_providers:
        log.info("=== %s ===", provider.name)
        report.results.append(run_forecast_provider(provider, settings, init_time))

    # --- observations
    log.info("=== IMD observations ===")
    if observation_provider is None and settings.enable_observations:
        observation_provider = IMDRainfallProvider(settings)
    if observation_provider is not None:
        report.results.append(run_observation_provider(observation_provider, settings))
    else:
        report.results.append(
            ProviderResult("IMD", "observation", ProviderStatus.NOT_CONFIGURED, "disabled (ENABLE_OBSERVATIONS=false)")
        )

    # --- AI plug-in
    ai_result = ProviderResult("AI", "ai_forecast", ProviderStatus.PLUGIN_ONLY,
                               "no AI model provider configured (AI_MODEL_PROVIDER is empty); interface only")

    def run_ai() -> None:
        ai = create_ai_provider(settings)
        if ai is not None:
            r = run_forecast_provider(ai, settings, init_time)
            ai_result.__dict__.update(r.__dict__)

    _run_guarded(ai_result, run_ai)
    report.results.append(ai_result)

    # --- report
    report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reports_dir = settings.processed_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.as_dict(), indent=2)
    run_path = reports_dir / f"phase1_report_{report.run_id}.json"
    run_path.write_text(payload, encoding="utf-8")
    (settings.processed_dir / REPORT_NAME).write_text(payload, encoding="utf-8")
    report.report_path = str(run_path)
    return report
