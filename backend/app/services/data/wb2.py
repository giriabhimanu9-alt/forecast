"""WeatherBench 2 providers: GraphCast (AI), IFS HRES (NWP) and the ERA5 reference — Experiment B only.

Source: public Zarr stores at https://storage.googleapis.com/weatherbench2/datasets/ (no
credentials), 1.5 deg "240x121_equiangular_with_poles_conservative" grid, 6-hourly steps.

Precipitation semantics (verified on real data, see PHASE4_DATA_AUDIT.md):
  GraphCast  ``total_precipitation_6hr`` = metres fallen in the 6 h before each step.
             Accumulation since init = cumulative sum of the 6 h buckets; buckets
             slightly below zero (model artefact, down to -9e-5 m) are clipped to 0
             before summing and the count is recorded.
  HRES       ``total_precipitation`` = metres accumulated since initialisation.
  ERA5       ``total_precipitation_24hr`` at time T = metres over (T-24 h, T].

These models output 6-hourly steps only, so they can never be aligned exactly with
IMD's 03 UTC day; they are verified against ERA5 in a separate experiment.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from app.config import Settings, TargetGrid
from app.models.schema import DownloadError, InvalidDataError
from app.services.data import normalization
from app.services.data.ai_model import AIModelProvider, register_ai_provider
from app.services.data.base import FetchStats, ForecastProvider

log = logging.getLogger(__name__)

WB2_BASE = "https://storage.googleapis.com/weatherbench2/datasets/"
GRID_SUFFIX = "240x121_equiangular_with_poles_conservative.zarr"
ERA5_STORE = f"era5/1959-2023_01_10-6h-{GRID_SUFFIX}"
BOX = {"latitude": slice(3.0, 40.5), "longitude": slice(64.5, 100.5)}  # India + 1 grid cell margin (0..360 lon)


def _open(path: str) -> xr.Dataset:
    try:
        return xr.open_zarr(WB2_BASE + path, consolidated=True, zarr_format=2, decode_timedelta=True, chunks=None)
    except Exception as exc:  # network / format problems
        raise DownloadError(f"cannot open WeatherBench 2 store {path}: {exc}") from exc


def _clean_attrs(attrs: dict) -> dict:
    """NetCDF attributes must be str/number (WB2 metadata contains booleans and nested values)."""
    return {k: v for k, v in attrs.items()
            if isinstance(v, (str, int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))}


def _atomic_netcdf(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.copy()
    ds.attrs = _clean_attrs(ds.attrs)
    for name in ds.variables:
        ds[name].attrs = _clean_attrs(ds[name].attrs)
        ds[name].encoding = {}
    tmp = path.with_name(path.name + ".part")
    ds.to_netcdf(tmp)
    tmp.replace(path)


class WB2ForecastProvider(ForecastProvider):
    """Forecasts from a WB2 Zarr store, subset to India and cached per initialisation."""

    stores: dict[int, str] = {}
    precip_var: str = ""
    mode: str = "since_init"  # or "sum_6h"
    supported_cycles = (0, 12)

    def __init__(self, settings: Settings, http=None):
        super().__init__(settings, http=http)
        self._open_stores: dict[str, xr.Dataset] = {}

    def store_for(self, init_time: datetime) -> str:
        if init_time.year in self.stores:
            return self.stores[init_time.year]
        if "*" in self.stores:
            return self.stores["*"]
        raise DownloadError(f"{self.name}: no WeatherBench 2 store for {init_time.year} (available: {sorted(self.stores)})")

    def _store(self, path: str) -> xr.Dataset:
        if path not in self._open_stores:
            self._open_stores[path] = _open(path)
        return self._open_stores[path]

    def cache_path(self, init_time: datetime) -> Path:
        return self.settings.raw_dir / "wb2" / self.name.lower() / f"{init_time:%Y%m%d%H}.nc"

    def _leads_to_download(self, lead_hours: Iterable[int]) -> list[int]:
        top = max(lead_hours)
        return list(range(6, top + 1, 6)) if self.mode == "sum_6h" else sorted({h for h in lead_hours if h > 0})

    def is_available(self, init_time: datetime, lead_hours: Iterable[int]) -> bool:
        try:
            ds = self._store(self.store_for(init_time))
        except DownloadError:
            return False
        return bool(np.isin(np.datetime64(init_time, "ns"), ds.time.values))

    def fetch_many(self, init_times: list[datetime], lead_hours: Iterable[int], batch: int = 8) -> FetchStats:
        """Download all missing inits (batched: one Zarr request per chunk, many inits at a time)."""
        lead_hours = list(lead_hours)
        stats = FetchStats(files=[])
        missing = [t for t in init_times if not self.cache_path(t).is_file()]
        stats.cache_hits = len(init_times) - len(missing)
        leads = [np.timedelta64(h, "h") for h in self._leads_to_download(lead_hours)]
        by_store: dict[str, list[datetime]] = {}
        for t in missing:
            by_store.setdefault(self.store_for(t), []).append(t)
        for path, inits in by_store.items():
            ds = self._store(path)
            available = set(pd.DatetimeIndex(ds.time.values).to_pydatetime())
            for i in range(0, len(inits), batch):
                chunk = [t for t in inits[i:i + batch] if t in available]
                if not chunk:
                    continue
                sub = ds[[self.precip_var]].sel(time=[np.datetime64(t, "ns") for t in chunk],
                                                prediction_timedelta=leads, **BOX)
                for attempt in range(4):
                    try:
                        sub = sub.load()
                        break
                    except Exception as exc:
                        if attempt == 3:
                            raise DownloadError(f"{self.name}: WB2 download failed for {chunk[0]}..: {exc}") from exc
                for t in chunk:
                    one = sub.sel(time=np.datetime64(t, "ns"))
                    one.attrs.update(source=f"WeatherBench 2 {path}", variable=self.precip_var, mode=self.mode)
                    _atomic_netcdf(one, self.cache_path(t))
                    stats.downloads += 1
                log.info("%s: cached %d inits (%s..%s)", self.name, len(chunk), chunk[0], chunk[-1])
        stats.files = [self.cache_path(t) for t in init_times if self.cache_path(t).is_file()]
        return stats

    def fetch(self, init_time: datetime, lead_hours: Iterable[int], variables: Iterable[str]) -> FetchStats:
        if list(variables) != ["precipitation"]:
            raise InvalidDataError(f"{self.name}: only precipitation is ingested from WeatherBench 2")
        stats = self.fetch_many([init_time], lead_hours)
        if not stats.files:
            raise DownloadError(f"{self.name}: init {init_time} not in the WeatherBench 2 store")
        return stats

    def load_fields(self, files: list[Path], init_time: datetime, lead_hours: Iterable[int]) -> dict[str, xr.DataArray]:
        with xr.open_dataset(files[0], decode_timedelta=True) as ds:
            raw = ds[self.precip_var].load()
        raw = raw.rename(prediction_timedelta="lead_time").transpose("lead_time", "latitude", "longitude")
        n_clipped = 0
        if self.mode == "sum_6h":
            n_clipped = int((raw < 0).sum())
            acc = raw.clip(min=0.0).cumsum("lead_time")
        else:
            acc = raw
        want = [np.timedelta64(h, "h") for h in lead_hours]
        missing = [h for h in want if h not in acc.lead_time.values]
        if missing:
            raise InvalidDataError(f"{self.name}: leads {missing} not in cached WB2 data")
        acc = acc.sel(lead_time=want).astype("float32")
        acc = acc.assign_coords(lead_time=np.array(want, dtype="timedelta64[ns]"))
        acc.attrs = {"units": "m", "negative_6h_buckets_clipped": n_clipped}
        return {"precipitation": acc}

    def source_attrs(self, init_time: datetime) -> dict[str, object]:
        return {"source": f"WeatherBench 2 {self.store_for(init_time)}", "native_resolution_deg": 1.5,
                "precip_variable": self.precip_var, "precip_conversion": self.mode}


@register_ai_provider("graphcast-wb2")
class GraphCastWB2Provider(WB2ForecastProvider, AIModelProvider):
    """GraphCast (Google DeepMind), ERA5-initialised forecasts published in WeatherBench 2."""

    name = "GraphCast"
    model_family = "GraphCast (DeepMind), WeatherBench 2 graphcast_v2, ERA5 initial conditions"
    stores = {2018: f"graphcast_v2/2018-{GRID_SUFFIX}", 2020: f"graphcast_v2/2020-{GRID_SUFFIX}",
              2022: f"graphcast_v2/2022-{GRID_SUFFIX}"}
    precip_var = "total_precipitation_6hr"
    mode = "sum_6h"


class HRESWB2Provider(WB2ForecastProvider):
    """ECMWF IFS HRES operational forecasts (own analysis) published in WeatherBench 2."""

    name = "HRES"
    stores = {"*": f"hres/2016-2022-0012-{GRID_SUFFIX}"}
    precip_var = "total_precipitation"
    mode = "since_init"


def era5_reference(settings: Settings, window_ends: list[pd.Timestamp], grid: TargetGrid,
                   cache_name: str) -> xr.Dataset:
    """ERA5 24 h precipitation for windows ending at the given times, in the observation contract."""
    path = settings.raw_dir / "wb2" / "era5" / f"{cache_name}.nc"
    times = [np.datetime64(t, "ns") for t in window_ends]
    if path.is_file():
        with xr.open_dataset(path) as ds:
            raw = ds["total_precipitation_24hr"].load()
        if not np.isin(times, raw.time.values).all():
            raw = None
    else:
        raw = None
    if raw is None:
        ds = _open(ERA5_STORE)
        last = ds.time.values[-1]
        times = [t for t in times if t <= last]
        raw = ds["total_precipitation_24hr"].sel(time=times, **BOX).load()
        raw.attrs = {"units": "m", "source": f"WeatherBench 2 {ERA5_STORE}"}
        _atomic_netcdf(raw.to_dataset(), path)
    rain = raw.transpose("time", "latitude", "longitude")
    rain.attrs["units"] = "m"
    labels = pd.DatetimeIndex(rain.time.values)
    # label = end time; period_end_offset 0 h -> canonical time = window end (00 UTC)
    rain = rain.assign_coords(time=labels.normalize())
    obs = normalization.normalize_observations(
        rain, grid, source="ERA5 (WeatherBench 2)", period_end_offset=pd.Timedelta(hours=0),
        source_attrs={"reference": "ERA5 reanalysis 24 h total precipitation (a model product, not gauges)"},
    )
    return obs
