"""Provider interfaces.

Every forecast source (NWP or AI) implements :class:`ForecastProvider` and
returns a dataset that follows ``app.models.schema`` — so later phases
(verification, blending) never need to know which source produced it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from app.config import Settings
from app.models.schema import DownloadError, InvalidDataError, NotConfiguredError
from app.services.data import normalization, validation
from app.services.http import ByteRange, HttpClient, cached_file_ok, write_cached

log = logging.getLogger(__name__)

# Provider-level field keys; see normalization.normalize_forecast.
FIELDS_FOR_VARIABLE: dict[str, tuple[str, ...]] = {
    "precipitation": ("precipitation",),
    "temperature_2m": ("temperature_2m",),
    "wind_speed_10m": ("u_10m", "v_10m"),
    "mslp": ("mslp",),
    "u_850hpa": ("u_850hpa",),
    "precipitable_water": ("precipitable_water",),
}


def fields_for(variables: Iterable[str]) -> list[str]:
    out: list[str] = []
    for var in variables:
        for f in FIELDS_FOR_VARIABLE[var]:
            if f not in out:
                out.append(f)
    return out


@dataclass
class FetchStats:
    files: list[Path]
    cache_hits: int = 0
    downloads: int = 0


class ForecastProvider(ABC):
    """A source of gridded forecasts for one model."""

    name: str
    kind: str = "forecast"

    def __init__(self, settings: Settings, http: HttpClient | None = None):
        self.settings = settings
        self.http = http or HttpClient(settings.http_timeout, settings.http_retries)

    @property
    def model_name(self) -> str:
        """Name used as the ``model`` coordinate; blending code only ever sees this string."""
        return self.name

    def ensure_configured(self) -> None:
        """Raise :class:`NotConfiguredError` if this provider cannot run as configured."""

    @abstractmethod
    def is_available(self, init_time: datetime, lead_hours: Iterable[int]) -> bool:
        """Cheap remote check that a cycle has been published."""

    @abstractmethod
    def fetch(self, init_time: datetime, lead_hours: Iterable[int], variables: Iterable[str]) -> FetchStats:
        """Download (or reuse cached) raw files."""

    @abstractmethod
    def load_fields(self, files: list[Path], init_time: datetime, lead_hours: Iterable[int]) -> dict[str, xr.DataArray]:
        """Decode raw files into provider fields with dims (lead_time, latitude, longitude)."""

    def get_forecast(
        self, init_time: datetime, lead_hours: Iterable[int], variables: Iterable[str]
    ) -> tuple[xr.Dataset, FetchStats]:
        lead_hours = list(lead_hours)
        variables = list(variables)
        self.ensure_configured()
        stats = self.fetch(init_time, lead_hours, variables)
        fields = self.load_fields(stats.files, init_time, lead_hours)
        ds = normalization.normalize_forecast(
            fields, self.name, init_time, self.settings.grid, variables, self.source_attrs(init_time)
        )
        return ds, stats

    def source_attrs(self, init_time: datetime) -> dict[str, object]:
        return {"source": self.name}


# --------------------------------------------------------------------------- GRIB via index files


@dataclass(frozen=True)
class IndexEntry:
    field: str
    byte_range: ByteRange


class IndexedGribProvider(ForecastProvider):
    """Downloads individual GRIB messages via HTTP byte ranges from a published index.

    Both NOAA (``.idx``) and ECMWF open data (``.index``) publish an index
    per forecast file, so we download only the few fields we need (~1 MB
    each) instead of ~500 MB per lead time.
    """

    cache_subdir: str
    supported_cycles: tuple[int, ...] = (0, 6, 12, 18)

    @abstractmethod
    def grib_url(self, init_time: datetime, lead: int) -> str: ...

    @abstractmethod
    def index_url(self, init_time: datetime, lead: int) -> str: ...

    @abstractmethod
    def parse_index(self, text: str, lead: int, fields: list[str]) -> dict[str, IndexEntry]:
        """Return byte ranges for the requested fields; raise InvalidDataError if any is missing."""

    @abstractmethod
    def supports_lead(self, init_time: datetime, lead: int) -> bool: ...

    def ensure_lead_times(self, init_time: datetime, lead_hours: Iterable[int]) -> None:
        if init_time.hour not in self.supported_cycles:
            raise NotConfiguredError(
                f"{self.name} has no {init_time:%H}Z cycle (supported: {self.supported_cycles})"
            )
        bad = [h for h in lead_hours if not self.supports_lead(init_time, h)]
        if bad:
            raise NotConfiguredError(
                f"{self.name} does not publish lead times {bad} h for the {init_time:%H}Z cycle"
            )

    def is_available(self, init_time: datetime, lead_hours: Iterable[int]) -> bool:
        try:
            self.ensure_lead_times(init_time, lead_hours)
        except NotConfiguredError:
            return False
        return self.http.exists(self.index_url(init_time, max(lead_hours)))

    def cache_path(self, init_time: datetime, lead: int, field: str) -> Path:
        return (
            self.settings.raw_dir
            / self.cache_subdir
            / f"{init_time:%Y%m%d}"
            / f"{init_time:%H}"
            / f"f{lead:03d}_{field}.grib2"
        )

    def fetch(self, init_time: datetime, lead_hours: Iterable[int], variables: Iterable[str]) -> FetchStats:
        lead_hours = list(lead_hours)
        self.ensure_lead_times(init_time, lead_hours)
        fields = fields_for(variables)
        stats = FetchStats(files=[])
        for lead in lead_hours:
            missing = [
                f for f in fields if not cached_file_ok(self.cache_path(init_time, lead, f), validation.check_grib_file)
            ]
            stats.cache_hits += len(fields) - len(missing)
            if missing:
                index_url = self.index_url(init_time, lead)
                try:
                    text = self.http.get_text(index_url)
                except DownloadError as exc:
                    raise DownloadError(
                        f"{self.name} {init_time:%Y-%m-%d %H}Z +{lead}h not published or unreachable: {exc}"
                    ) from exc
                entries = self.parse_index(text, lead, missing)
                url = self.grib_url(init_time, lead)
                for f in missing:
                    entry = entries[f]
                    content = self.http.get_bytes(url, entry.byte_range)
                    write_cached(
                        self.cache_path(init_time, lead, f),
                        content,
                        validation.check_grib_file,
                        {"url": url, "range": entry.byte_range.header, "index": index_url, "field": f},
                    )
                    stats.downloads += 1
                    log.info("%s: downloaded +%03dh %s (%.1f kB)", self.name, lead, f, len(content) / 1024)
            stats.files.extend(self.cache_path(init_time, lead, f) for f in fields)
        return stats

    def load_fields(self, files: list[Path], init_time: datetime, lead_hours: Iterable[int]) -> dict[str, xr.DataArray]:
        per_field: dict[str, list[xr.DataArray]] = {}
        for path in files:
            lead = int(path.name[1:4])
            field = path.stem.split("_", 1)[1]
            da = decode_grib_message(path, init_time, lead)
            # Crop right away so we never hold many global fields in memory.
            da = normalization.crop_to_grid(normalization.standardize_coords(da), self.settings.grid).load()
            per_field.setdefault(field, []).append(da)
        out: dict[str, xr.DataArray] = {}
        for field, arrays in per_field.items():
            combined = xr.concat(arrays, dim="lead_time")
            validation.check_no_duplicate_times(combined, ["lead_time"])
            out[field] = combined
        return out


def decode_grib_message(path: Path, init_time: datetime, lead: int) -> xr.DataArray:
    """Decode a single-message GRIB file and check it is the expected init/lead."""
    validation.check_grib_file(path)
    try:
        ds = xr.open_dataset(
            path, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_timedelta=True
        )
    except Exception as exc:  # cfgrib raises a variety of errors on bad input
        raise InvalidDataError(f"Cannot decode GRIB file {path}: {exc}") from exc
    try:
        if len(ds.data_vars) != 1:
            raise InvalidDataError(f"{path.name}: expected 1 GRIB variable, found {list(ds.data_vars)}")
        da = ds[next(iter(ds.data_vars))].load()
    finally:
        ds.close()
    actual_init = np.datetime64(da["time"].values, "ns")
    if actual_init != normalization.to_utc_naive(init_time):
        raise InvalidDataError(f"{path.name}: GRIB init time {actual_init} != requested {init_time}")
    step = da["step"].values
    if step != np.timedelta64(lead, "h"):
        raise InvalidDataError(f"{path.name}: GRIB step {step} != requested lead {lead} h")
    units = da.attrs.get("GRIB_units") or da.attrs.get("units")
    da = da.drop_vars([c for c in da.coords if c not in ("latitude", "longitude")])
    da = da.expand_dims(lead_time=np.array([np.timedelta64(lead, "h")], dtype="timedelta64[ns]"))
    da.attrs = {"units": units, "grib_short_name": da.attrs.get("GRIB_shortName", "")}
    return da
