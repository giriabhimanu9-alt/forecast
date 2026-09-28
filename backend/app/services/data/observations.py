"""IMD gridded daily rainfall observations (0.25°, India land).

Two official IMD Pune routes, the same ones ``imdlib`` uses:

* real-time daily files   POST https://imdpune.gov.in/cmpg/Realtimedata/Rainfall/rain.php  rain=DDMMYYYY
* yearly archive (QC'd)   POST https://imdpune.gov.in/cmpg/Griddata/rainfall.php          rain=YYYY
  (Pai et al. 2014, MAUSAM 65(1); 1901 to the last completed year)

Both are open access. We download with our own strict client (timeouts,
retries, exact-size checks; imdlib's downloader returns ``None`` on HTTP
errors and would cache an error page) but write imdlib's file layout and
decode with ``imdlib.open_real_data`` / ``imdlib.open_data``.

Accumulation windows (all 24 h ending/starting 08:30 IST = 03:00 UTC):

* real-time day D  -> 03 UTC D-1 .. 03 UTC D. Verified empirically: the
  file for D is published before 08:30 IST on D+1, so it cannot cover the
  24 h ending on D+1.
* archive day D    -> 03 UTC D-1 .. 03 UTC D, i.e. the SAME convention.
  Phase 1 originally assumed 03 UTC D .. D+1 (a reading of Pai et al. 2014);
  Phase 2 disproved that with real July 2025 data (2026-09-28):
    - real-time file D vs archive label D: r = 0.997 / 0.994, mean |diff|
      0.18 / 0.15 mm (5 and 10 July), i.e. the same field after QC; labels
      D-1 and D+1 give r = 0.27-0.47;
    - GFS and ECMWF 24 h forecasts for 03 UTC D -> 03 UTC D+1 correlate best
      with archive label D+1 (0.42 / 0.50) rather than D (0.29 / 0.33).
  The Phase 2 experiment re-runs the forecast check on every run
  (``observation_convention_check`` in the manifest).

The canonical ``time`` coordinate is the window END, so both sources line up
with forecast ``valid_time``; see ``normalization.normalize_observations``.
"""

from __future__ import annotations

import calendar
import contextlib
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import xarray as xr

from app.config import Settings
from app.models.schema import DownloadError, InvalidDataError, NotConfiguredError
from app.services.data import normalization, validation
from app.services.http import HttpClient, cached_file_ok, write_cached

log = logging.getLogger(__name__)

IMD_REALTIME_URL = "https://imdpune.gov.in/cmpg/Realtimedata/Rainfall/rain.php"
IMD_ARCHIVE_URL = "https://imdpune.gov.in/cmpg/Griddata/rainfall.php"
IMD_NLAT, IMD_NLON = 129, 135
IMD_DAY_BYTES = IMD_NLAT * IMD_NLON * 4
IMD_FILL_VALUE = -999.0

PERIOD_END_OFFSET = {
    "realtime": pd.Timedelta(hours=3),  # label D -> window ends 03 UTC on D
    "archive": pd.Timedelta(hours=3),  # label D -> window ends 03 UTC on D (verified, see module doc)
}


@dataclass
class ObservationFetch:
    files: list[Path]
    source_by_day: dict[date, str]
    cache_hits: int = 0
    downloads: int = 0


class IMDRainfallProvider:
    name = "IMD"
    kind = "observation"

    def __init__(self, settings: Settings, http: HttpClient | None = None, today: date | None = None):
        self.settings = settings
        self.http = http or HttpClient(settings.http_timeout, settings.http_retries)
        self.today = today or datetime.now(timezone.utc).date()

    # ------------------------------------------------------------------ configuration

    def date_range(self) -> tuple[date, date]:
        s = self.settings
        if s.imd_start_date and s.imd_end_date:
            start, end = s.imd_start_date, s.imd_end_date
        else:
            # Real-time files appear with a lag of up to a day; stay 2 days back.
            end = self.today - timedelta(days=2)
            start = end - timedelta(days=s.imd_lookback_days - 1)
        if end >= self.today:
            raise NotConfiguredError(f"IMD_END_DATE {end} is not in the past; observations do not exist yet")
        if start.year < 1901:
            raise NotConfiguredError("IMD gridded rainfall starts in 1901")
        return start, end

    def source_for(self, day: date) -> str:
        mode = self.settings.imd_source
        if mode != "auto":
            return mode
        return "archive" if day.year < self.today.year else "realtime"

    # ------------------------------------------------------------------ download

    @property
    def raw_dir(self) -> Path:
        return self.settings.raw_dir / "imd"

    def realtime_path(self, day: date) -> Path:
        # imdlib.open_real_data naming
        return self.raw_dir / "realtime" / f"rain_ind0.25_{day:%y_%m_%d}.grd"

    def archive_path(self, year: int) -> Path:
        # imdlib.open_data(..., fn_format="yearwise") naming
        return self.raw_dir / "archive" / "rain" / f"{year}.grd"

    def _fetch_realtime(self, day: date, stats: ObservationFetch) -> None:
        path = self.realtime_path(day)

        def check(p: Path) -> None:
            validation.check_binary_size(p, IMD_DAY_BYTES)

        if cached_file_ok(path, check):
            stats.cache_hits += 1
        else:
            content = self.http.post_bytes(IMD_REALTIME_URL, {"rain": f"{day:%d%m%Y}"})
            if len(content) == 0:
                raise DownloadError(f"IMD real-time rainfall for {day} is not published yet (empty response)")
            try:
                write_cached(path, content, check, {"url": IMD_REALTIME_URL, "form": {"rain": f"{day:%d%m%Y}"}})
            except InvalidDataError as exc:
                raise InvalidDataError(f"IMD real-time file for {day}: {exc}") from exc
            stats.downloads += 1
            log.info("IMD: downloaded real-time rainfall for %s", day)
        stats.files.append(path)

    def _fetch_archive(self, year: int, stats: ObservationFetch) -> None:
        path = self.archive_path(year)
        expected = (366 if calendar.isleap(year) else 365) * IMD_DAY_BYTES

        def check(p: Path) -> None:
            validation.check_binary_size(p, expected)

        if cached_file_ok(path, check):
            stats.cache_hits += 1
        else:
            log.info("IMD: downloading %d archive (~%.0f MB)", year, expected / 1e6)
            content = self.http.post_bytes(IMD_ARCHIVE_URL, {"rain": str(year)})
            if len(content) == 0:
                raise DownloadError(f"IMD archive for {year} is not published (empty response)")
            write_cached(path, content, check, {"url": IMD_ARCHIVE_URL, "form": {"rain": str(year)}})
            stats.downloads += 1
        stats.files.append(path)

    def fetch(self) -> ObservationFetch:
        start, end = self.date_range()
        days = [d.date() for d in pd.date_range(start, end, freq="D")]
        stats = ObservationFetch(files=[], source_by_day={})
        for year in sorted({d.year for d in days if self.source_for(d) == "archive"}):
            try:
                self._fetch_archive(year, stats)
            except DownloadError as exc:
                if self.settings.imd_source == "archive":
                    raise
                log.warning("IMD archive %d unavailable (%s); using real-time files", year, exc)
                for d in days:
                    if d.year == year:
                        stats.source_by_day[d] = "realtime"
        for d in days:
            src = stats.source_by_day.setdefault(d, self.source_for(d))
            if src == "realtime":
                self._fetch_realtime(d, stats)
        return stats

    # ------------------------------------------------------------------ decode

    def _read(self, source: str, start: date, end: date) -> xr.DataArray:
        import imdlib  # imported lazily: only needed when IMD is enabled

        # imdlib prints progress to stdout; keep our CLI output clean.
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                if source == "realtime":
                    obj = imdlib.open_real_data("rain", f"{start}", f"{end}", str(self.realtime_path(start).parent))
                else:
                    obj = imdlib.open_data(
                        "rain", f"{start}", f"{end}", "yearwise", str(self.archive_path(start.year).parent.parent)
                    )
            except Exception as exc:
                raise InvalidDataError(f"imdlib could not read IMD {source} data {start}..{end}: {exc}") from exc
        ds = obj.get_xarray()
        rain = ds["rain"].astype("float32")
        units = ds["rain"].attrs.get("units")
        # imdlib labels the values "mm/day"; each value is the 24 h total for
        # that day, i.e. a depth in mm over the (one-day) accumulation window.
        if units == "mm/day":
            units = "mm"
        rain.attrs["units"] = units
        return rain

    def get_observations(self) -> tuple[xr.Dataset, ObservationFetch]:
        stats = self.fetch()
        # Decode contiguous runs of days that share a source.
        runs: list[tuple[str, date, date]] = []
        for d in sorted(stats.source_by_day):
            src = stats.source_by_day[d]
            if runs and runs[-1][0] == src and runs[-1][2] + timedelta(days=1) == d:
                runs[-1] = (src, runs[-1][1], d)
            else:
                runs.append((src, d, d))
        parts = []
        for src, start, end in runs:
            rain = self._read(src, start, end)
            parts.append(
                normalization.normalize_observations(
                    rain,
                    self.settings.grid,
                    source="IMD",
                    period_end_offset=PERIOD_END_OFFSET[src],
                    fill_values=(IMD_FILL_VALUE,),
                )
            )
        ds = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        ds.attrs.update(
            {
                "source": "India Meteorological Department (IMD) Pune gridded daily rainfall 0.25 deg",
                "institution": "India Meteorological Department",
                "imd_routes": ",".join(sorted({r[0] for r in runs})),
                "references": "Pai et al. (2014), MAUSAM 65(1), 1-18",
            }
        )
        return ds, stats
