"""SYNTHETIC TEST FIXTURES — NOT WEATHER DATA.

Generates small GRIB2 messages, index files and IMD-layout binaries from
simple analytic formulas so the real provider code paths (index parsing,
byte-range download, GRIB decoding, imdlib reading, normalisation) can be
exercised offline. Nothing here is ever written to ``data/`` or presented
as a forecast or observation; files live only in pytest's tmp directories.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import eccodes
import numpy as np

from app.services.http import ByteRange

# Coarse 1° source grid covering the test target region.
SRC_LAT0, SRC_LAT1 = 30.0, 0.0  # north -> south, like GFS/ECMWF
SRC_LON0, SRC_LON1 = 60.0, 100.0
SRC_RES = 1.0
SRC_LATS = np.arange(SRC_LAT0, SRC_LAT1 - 0.5 * SRC_RES, -SRC_RES)
SRC_LONS = np.arange(SRC_LON0, SRC_LON1 + 0.5 * SRC_RES, SRC_RES)


def synthetic_field(kind: str, lead: int) -> np.ndarray:
    """Analytic test pattern on the source grid, shape (nlat, nlon)."""
    lat2d, lon2d = np.meshgrid(SRC_LATS, SRC_LONS, indexing="ij")
    if kind == "precip_mm":  # accumulated since init: grows linearly with lead
        return 0.1 * lead * (1.0 + lat2d / 30.0)
    if kind == "temp_k":
        return 300.0 - 0.5 * lat2d + 0.0 * lon2d
    if kind == "u":
        return np.full_like(lat2d, 3.0)
    if kind == "v":
        return np.full_like(lat2d, 4.0)  # => wind speed exactly 5 m/s
    if kind == "bucket":
        return np.full_like(lat2d, 123.0)  # must never be selected
    raise ValueError(kind)


def grib_message(centre: str, short_name: str, init: datetime, lead: int, values: np.ndarray,
                 accum_start: int | None = None) -> bytes:
    h = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        s = eccodes.codes_set
        s(h, "centre", centre)
        s(h, "dataDate", int(init.strftime("%Y%m%d")))
        s(h, "dataTime", init.hour * 100)
        s(h, "gridType", "regular_ll")
        s(h, "Ni", SRC_LONS.size)
        s(h, "Nj", SRC_LATS.size)
        s(h, "latitudeOfFirstGridPointInDegrees", SRC_LAT0)
        s(h, "latitudeOfLastGridPointInDegrees", SRC_LAT1)
        s(h, "longitudeOfFirstGridPointInDegrees", SRC_LON0)
        s(h, "longitudeOfLastGridPointInDegrees", SRC_LON1)
        s(h, "iDirectionIncrementInDegrees", SRC_RES)
        s(h, "jDirectionIncrementInDegrees", SRC_RES)
        if accum_start is not None:
            s(h, "productDefinitionTemplateNumber", 8)
            s(h, "typeOfFirstFixedSurface", 1)
            s(h, "shortName", short_name)
            s(h, "stepType", "accum")
            s(h, "startStep", accum_start)
            s(h, "endStep", lead)
        else:
            s(h, "shortName", short_name)
            s(h, "stepUnits", 1)
            s(h, "forecastTime", lead)
        eccodes.codes_set_values(h, np.asarray(values, dtype=float).ravel())
        return eccodes.codes_get_message(h)
    finally:
        eccodes.codes_release(h)


class FakeHttp:
    """In-memory stand-in for :class:`app.services.http.HttpClient`."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.posts: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str]] = []

    def exists(self, url: str) -> bool:
        self.calls.append(("HEAD", url))
        return url in self.objects

    def get_text(self, url: str) -> str:
        return self.get_bytes(url).decode()

    def get_bytes(self, url: str, byte_range: ByteRange | None = None) -> bytes:
        from app.models.schema import DownloadError

        self.calls.append(("GET", url))
        if url not in self.objects:
            raise DownloadError(f"GET {url} returned HTTP 404")
        data = self.objects[url]
        if byte_range is None:
            return data
        end = len(data) if byte_range.end is None else byte_range.end + 1
        return data[byte_range.start:end]

    def post_bytes(self, url: str, data: dict[str, str]) -> bytes:
        self.calls.append(("POST", f"{url} {data}"))
        key = (url, json.dumps(data, sort_keys=True))
        if key not in self.posts:
            return b""  # IMD answers unpublished dates with an empty 200 response
        return self.posts[key]

    def downloads(self) -> int:
        return sum(1 for m, u in self.calls if m in ("GET", "POST") and not u.endswith((".idx", ".index")))


def publish_gfs(http: FakeHttp, provider, init: datetime, leads: list[int]) -> None:
    """Serve synthetic GFS files + .idx in NOAA's real index format."""
    for lead in leads:
        records = [
            ("TMP", "2 m above ground", f"{lead} hour fcst", grib_message("kwbc", "2t", init, lead, synthetic_field("temp_k", lead))),
            ("UGRD", "10 m above ground", f"{lead} hour fcst", grib_message("kwbc", "10u", init, lead, synthetic_field("u", lead))),
            ("VGRD", "10 m above ground", f"{lead} hour fcst", grib_message("kwbc", "10v", init, lead, synthetic_field("v", lead))),
            ("APCP", "surface", f"{lead - 6}-{lead} hour acc fcst",
             grib_message("kwbc", "tp", init, lead, synthetic_field("bucket", lead), accum_start=lead - 6)),
            ("APCP", "surface", f"0-{lead} hour acc fcst",
             grib_message("kwbc", "tp", init, lead, synthetic_field("precip_mm", lead), accum_start=0)),
        ]
        blob, lines, offset = b"", [], 0
        for n, (var, level, trange, msg) in enumerate(records, start=1):
            lines.append(f"{n}:{offset}:d={init:%Y%m%d%H}:{var}:{level}:{trange}:")
            blob += msg
            offset += len(msg)
        http.objects[provider.grib_url(init, lead)] = blob
        http.objects[provider.index_url(init, lead)] = ("\n".join(lines) + "\n").encode()


def publish_ecmwf(http: FakeHttp, provider, init: datetime, leads: list[int]) -> None:
    """Serve synthetic ECMWF open-data files + JSON-lines .index (tp in metres)."""
    for lead in leads:
        records = [
            ("tp", grib_message("ecmf", "tp", init, lead, synthetic_field("precip_mm", lead) / 1000.0, accum_start=0)),
            ("2t", grib_message("ecmf", "2t", init, lead, synthetic_field("temp_k", lead))),
            ("10u", grib_message("ecmf", "10u", init, lead, synthetic_field("u", lead))),
            ("10v", grib_message("ecmf", "10v", init, lead, synthetic_field("v", lead))),
        ]
        blob, lines = b"", []
        for param, msg in records:
            lines.append(json.dumps({"date": f"{init:%Y%m%d}", "time": f"{init:%H}00", "step": str(lead),
                                     "levtype": "sfc", "param": param, "_offset": len(blob), "_length": len(msg)}))
            blob += msg
        http.objects[provider.grib_url(init, lead)] = blob
        http.objects[provider.index_url(init, lead)] = ("\n".join(lines) + "\n").encode()


IMD_LATS = np.linspace(6.5, 38.5, 129)
IMD_LONS = np.linspace(66.5, 100.0, 135)


def imd_day_bytes(day: date) -> bytes:
    """IMD real-time binary layout: float32 (lat, lon), lon fastest, -999 outside 'land'."""
    lat2d, lon2d = np.meshgrid(IMD_LATS, IMD_LONS, indexing="ij")
    land = (lat2d >= 10) & (lat2d <= 30) & (lon2d >= 72) & (lon2d <= 90)
    values = np.where(land, float(day.day) + 0.0 * lat2d, -999.0).astype("<f4")
    return values.tobytes()


def write_imd_realtime(path: Path, day: date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(imd_day_bytes(day))
