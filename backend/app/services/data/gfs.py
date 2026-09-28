"""NOAA/NCEP GFS 0.25° forecasts.

Source: NOAA Open Data Dissemination bucket (public, no credentials)
    https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
Archive back to 2021 is kept there, which Phase 2 (historical skill) relies on.
NOMADS uses the same path layout for the last ~10 days, so
``GFS_BASE_URL=https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod`` also works.

Each ``.idx`` line is ``n:byte_offset:d=YYYYMMDDHH:VAR:LEVEL:TIMERANGE:``;
a record's length is the next record's offset minus its own.
"""

from __future__ import annotations

import re
from datetime import datetime

from app.models.schema import InvalidDataError
from app.services.data.base import IndexedGribProvider, IndexEntry
from app.services.http import ByteRange

# field -> (VAR, LEVEL). APCP additionally must be the accumulation starting at hour 0.
GFS_FIELDS: dict[str, tuple[str, str]] = {
    "precipitation": ("APCP", "surface"),
    "temperature_2m": ("TMP", "2 m above ground"),
    "u_10m": ("UGRD", "10 m above ground"),
    "v_10m": ("VGRD", "10 m above ground"),
    "mslp": ("PRMSL", "mean sea level"),
    "u_850hpa": ("UGRD", "850 mb"),
    "precipitable_water": ("PWAT", "entire atmosphere (considered as a single layer)"),
}

_ACC_FROM_ZERO = re.compile(r"^0-(\d+) (hour|day) acc")


def _parse_idx_lines(text: str) -> list[tuple[int, str, str, str]]:
    records = []
    for line in text.strip().splitlines():
        parts = line.split(":")
        if len(parts) < 7:
            raise InvalidDataError(f"Malformed GFS index line: {line!r}")
        records.append((int(parts[1]), parts[3], parts[4], parts[5]))
    if not records:
        raise InvalidDataError("Empty GFS index file")
    return records


class GFSProvider(IndexedGribProvider):
    name = "GFS"
    cache_subdir = "gfs"
    supported_cycles = (0, 6, 12, 18)
    product = "0p25"

    def grib_url(self, init_time: datetime, lead: int) -> str:
        return (
            f"{self.settings.gfs_base_url}/gfs.{init_time:%Y%m%d}/{init_time:%H}/atmos/"
            f"gfs.t{init_time:%H}z.pgrb2.{self.product}.f{lead:03d}"
        )

    def index_url(self, init_time: datetime, lead: int) -> str:
        return self.grib_url(init_time, lead) + ".idx"

    def supports_lead(self, init_time: datetime, lead: int) -> bool:
        return (1 <= lead <= 120) or (120 < lead <= 384 and lead % 3 == 0)

    def parse_index(self, text: str, lead: int, fields: list[str]) -> dict[str, IndexEntry]:
        records = _parse_idx_lines(text)
        out: dict[str, IndexEntry] = {}
        for field in fields:
            var, level = GFS_FIELDS[field]
            matches = []
            for i, (offset, rec_var, rec_level, timerange) in enumerate(records):
                if rec_var != var or rec_level != level:
                    continue
                if field == "precipitation":
                    m = _ACC_FROM_ZERO.match(timerange)
                    if not m:
                        continue  # skip the 6-hourly bucket, keep 0-N accumulation
                    hours = int(m.group(1)) * (24 if m.group(2) == "day" else 1)
                    if hours != lead:
                        continue
                end = records[i + 1][0] - 1 if i + 1 < len(records) else None
                matches.append(ByteRange(offset, end))
            if not matches:
                raise InvalidDataError(f"GFS index for +{lead}h has no {var}:{level} record")
            # At 6-hourly leads GFS repeats the identical 0-N APCP record; any copy works.
            if field != "precipitation" and len(matches) > 1:
                raise InvalidDataError(f"GFS index for +{lead}h has {len(matches)} {var}:{level} records")
            out[field] = IndexEntry(field, matches[-1])
        return out

    def source_attrs(self, init_time: datetime) -> dict[str, object]:
        return {
            "source": f"NOAA/NCEP Global Forecast System (GFS) {self.product}, pgrb2",
            "source_url": self.grib_url(init_time, 0).rsplit(".f", 1)[0],
            "institution": "NOAA/NCEP",
        }


class GFSAtmosProvider(GFSProvider):
    """GFS 1.0 deg product for large-scale atmospheric regime features (MSLP, u850, PWAT).

    Not a precipitation ensemble member: it only feeds regime detection. The
    1.0 deg files have 3-hourly steps and are ~15x smaller than 0.25 deg.
    """

    name = "GFS_ATMOS_1P00"
    cache_subdir = "gfs_1p00"
    product = "1p00"

    def supports_lead(self, init_time: datetime, lead: int) -> bool:
        return 3 <= lead <= 384 and lead % 3 == 0
