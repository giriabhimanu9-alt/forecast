"""Strict forecast <-> observation time alignment.

A forecast window is paired with an observation only when BOTH ends of the
24 h accumulation period are identical (to the second) and the forecast was
initialised no later than the start of the observed period. There is no
"nearest date" matching, so a one-day (or one-hour) offset raises
:class:`AlignmentError` instead of silently producing wrong scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from app.models.schema import DataPipelineError, InvalidDataError, ProviderStatus
from app.services.verification.accumulation import ForecastWindow, forecast_to_24h_accumulation, verification_windows


class AlignmentError(DataPipelineError):
    status = ProviderStatus.INVALID_DATA


@dataclass(frozen=True)
class ObservationPeriod:
    start: pd.Timestamp  # observation_period_start (UTC)
    end: pd.Timestamp  # observation_period_end (UTC)
    valid_date: str  # the observation's own day label (e.g. IMD date)


@dataclass
class AlignedSample:
    model: str
    init_time: pd.Timestamp
    lead_day: int
    lead_window_h: tuple[int, int]
    valid_period: ObservationPeriod
    forecast_value: xr.DataArray  # (latitude, longitude), mm / 24 h
    observation_value: xr.DataArray  # (latitude, longitude), mm / 24 h


def observation_periods(obs: xr.Dataset) -> list[ObservationPeriod]:
    """Periods of a Phase 1 observation dataset (time = window end, period_start, source_date)."""
    return [
        ObservationPeriod(pd.Timestamp(s), pd.Timestamp(e), str(d))
        for s, e, d in zip(obs.period_start.values, obs.time.values, obs.source_date.values)
    ]


def align_forecast_to_observation(
    forecast_24h: xr.DataArray,
    model: str,
    window: ForecastWindow,
    observations: xr.Dataset,
) -> AlignedSample:
    """Pair one 24 h forecast window with the observation covering exactly the same period."""
    init = pd.Timestamp(window.init_time)
    start, end = pd.Timestamp(window.start), pd.Timestamp(window.end)
    if end - start != pd.Timedelta(hours=24):
        raise AlignmentError(f"{model} {init}: forecast window {start}..{end} is not 24 h")
    if init > start:
        raise AlignmentError(f"{model}: initialisation {init} is after the verified period start {start}")

    matches = np.flatnonzero(observations.time.values == np.datetime64(end, "ns"))
    if matches.size != 1:
        raise AlignmentError(
            f"{model} {init} +{window.start_lead_h}-{window.end_lead_h}h: no unique observation period "
            f"ending at {end} (found {matches.size})"
        )
    obs = observations.isel(time=int(matches[0]))
    period = ObservationPeriod(pd.Timestamp(obs.period_start.values), pd.Timestamp(obs.time.values),
                               str(obs.source_date.values))
    if period.start != start:
        raise AlignmentError(
            f"{model} {init}: forecast window starts {start} but observation period "
            f"{period.valid_date} starts {period.start}"
        )
    for coord in ("latitude", "longitude"):
        if not np.array_equal(forecast_24h[coord].values, obs[coord].values):
            raise AlignmentError(f"{model}: forecast and observation {coord} grids differ")
    return AlignedSample(
        model=model,
        init_time=init,
        lead_day=window.lead_day,
        lead_window_h=(window.start_lead_h, window.end_lead_h),
        valid_period=period,
        forecast_value=forecast_24h,
        observation_value=obs["precipitation"],
    )


@dataclass
class AlignmentLog:
    aligned: int = 0
    missing_forecast: list[str] | None = None
    missing_observation: list[str] | None = None

    def __post_init__(self) -> None:
        self.missing_forecast = self.missing_forecast or []
        self.missing_observation = self.missing_observation or []


def build_aligned_dataset(
    forecasts: Mapping[str, Mapping[datetime, xr.Dataset]],
    observations: xr.Dataset,
    init_times: Iterable[datetime],
    lead_days: Iterable[int],
    models: Iterable[str],
    window_starts: set[datetime] | None = None,
    window_end_hour_utc: int = 3,
) -> tuple[xr.Dataset, AlignmentLog]:
    """Stack every aligned (init, lead_day) window into one dataset.

    dims: model, sample, latitude, longitude. ``forecast`` is NaN for a model
    whose cycle is missing; samples with no observation are dropped and logged.
    ``window_starts`` restricts the result to windows starting at those times
    (the target observation periods).
    """
    models = list(models)
    lead_days = list(lead_days)
    log = AlignmentLog()
    rows: list[dict] = []
    fields: list[xr.DataArray] = []
    obs_fields: list[xr.DataArray] = []
    template = observations["precipitation"].isel(time=0, drop=True)
    for init in init_times:
        for window in verification_windows(init, lead_days, window_end_hour_utc):
            if window_starts is not None and window.start not in window_starts:
                continue
            per_model: list[xr.DataArray] = []
            period: ObservationPeriod | None = None
            obs_value: xr.DataArray | None = None
            obs_missing = False
            for model in models:
                ds = forecasts.get(model, {}).get(init)
                if ds is None:
                    log.missing_forecast.append(f"{model} {init:%Y-%m-%d %H}Z day{window.lead_day}")
                    per_model.append(xr.full_like(template, np.nan))
                    continue
                precip = ds["precipitation"].sel(model=model).isel(time=0, drop=True)
                try:
                    fc = forecast_to_24h_accumulation(precip, window.start_lead_h, window.end_lead_h)
                except InvalidDataError as exc:
                    raise InvalidDataError(f"{model} {init:%Y-%m-%d %H}Z day{window.lead_day}: {exc}") from exc
                try:
                    sample = align_forecast_to_observation(fc, model, window, observations)
                except AlignmentError as exc:
                    if "no unique observation" in str(exc):
                        obs_missing = True
                        break
                    raise
                period, obs_value = sample.valid_period, sample.observation_value
                per_model.append(sample.forecast_value.drop_vars("model", errors="ignore"))
            if obs_missing:
                log.missing_observation.append(f"{init:%Y-%m-%d %H}Z day{window.lead_day} ({window.start}..{window.end})")
                continue
            if period is None:  # no model available for this window
                continue
            rows.append(
                {
                    "init_time": np.datetime64(init, "ns"),
                    "lead_day": window.lead_day,
                    "start_lead_h": window.start_lead_h,
                    "end_lead_h": window.end_lead_h,
                    "observation_period_start": np.datetime64(period.start, "ns"),
                    "observation_period_end": np.datetime64(period.end, "ns"),
                    "valid_date": period.valid_date,
                }
            )
            fields.append(xr.concat(per_model, dim=pd.Index(models, name="model")))
            obs_fields.append(obs_value.drop_vars(["time", "period_start", "source_date"], errors="ignore"))
            log.aligned += 1
    if not rows:
        raise AlignmentError("no forecast/observation pairs could be aligned")
    meta = pd.DataFrame(rows)
    sample_coords = {c: ("sample", meta[c].to_numpy()) for c in meta.columns}
    ds = xr.Dataset(
        {
            "forecast": xr.concat(fields, dim="sample").transpose("model", "sample", "latitude", "longitude"),
            "observation": xr.concat(obs_fields, dim="sample"),
        }
    ).assign_coords(sample=np.arange(len(rows)), **sample_coords)
    ds["forecast"].attrs = {"units": "mm", "long_name": "24 h forecast precipitation"}
    ds["observation"].attrs = {"units": "mm", "long_name": "24 h observed precipitation (IMD)"}
    return ds, log
