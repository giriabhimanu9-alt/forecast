"""ECMWF IFS 0.25° forecasts from ECMWF Open Data.

Source: https://data.ecmwf.int/forecasts/YYYYMMDD/HHz/ifs/0p25/oper/
        YYYYMMDDHH0000-<step>h-oper-fc.grib2   (+ ``.index`` JSON-lines)
Licence: CC-BY-4.0, no API key or account required.

``data.ecmwf.int`` keeps only the most recent few days. For older cycles
(Phase 2 historical verification) point ``ECMWF_BASE_URL`` at the AWS mirror
``https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com`` which has the same
path layout. Full-archive (MARS) access needs an ECMWF account and licence
and is intentionally NOT used here.

Units: ``tp`` is metres of water accumulated since initialisation;
``2t`` is K; ``10u``/``10v`` are m s-1.
"""

from __future__ import annotations

import json
from datetime import date, datetime

from app.models.schema import InvalidDataError
from app.services.data.base import IndexedGribProvider, IndexEntry
from app.services.http import ByteRange

ECMWF_PARAMS: dict[str, str] = {
    "precipitation": "tp",
    "temperature_2m": "2t",
    "u_10m": "10u",
    "v_10m": "10v",
}


class ECMWFProvider(IndexedGribProvider):
    name = "ECMWF"
    cache_subdir = "ecmwf"
    model_path = "ifs"  # "aifs-single" is the same layout for ECMWF's AI model
    stream = "oper"
    supported_cycles = (0, 12)  # the 06/18Z runs use the short "scda" stream

    def product_path(self, init_time: datetime) -> str:
        """Directory layout by date (verified against the AWS mirror, PHASE4_DATA_AUDIT.md):
        2023-01-18..2024-01-31 ``0p4-beta/<stream>`` (0.4 deg); 2024-02 ``0p25/<stream>``;
        from 2024-03-01 ``<model>/0p25/<stream>``."""
        d = init_time.date()
        if d < date(2024, 2, 1):
            return f"0p4-beta/{self.stream}"
        if d < date(2024, 3, 1):
            return f"0p25/{self.stream}"
        return f"{self.model_path}/0p25/{self.stream}"

    def resolution_deg(self, init_time: datetime) -> float:
        return 0.4 if "0p4" in self.product_path(init_time) else 0.25

    def grib_url(self, init_time: datetime, lead: int) -> str:
        return (
            f"{self.settings.ecmwf_base_url}/{init_time:%Y%m%d}/{init_time:%H}z/{self.product_path(init_time)}/"
            f"{init_time:%Y%m%d%H}0000-{lead}h-{self.stream}-fc.grib2"
        )

    def index_url(self, init_time: datetime, lead: int) -> str:
        return self.grib_url(init_time, lead)[: -len(".grib2")] + ".index"

    def supports_lead(self, init_time: datetime, lead: int) -> bool:
        return (0 <= lead <= 144 and lead % 3 == 0) or (144 < lead <= 360 and lead % 6 == 0)

    def parse_index(self, text: str, lead: int, fields: list[str]) -> dict[str, IndexEntry]:
        entries: dict[str, list[dict]] = {}
        for line in text.strip().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvalidDataError(f"Malformed ECMWF index line: {line[:120]!r}") from exc
            if rec.get("levtype") == "sfc" and str(rec.get("step")) == str(lead):
                entries.setdefault(rec.get("param"), []).append(rec)
        out: dict[str, IndexEntry] = {}
        for field in fields:
            param = ECMWF_PARAMS[field]
            recs = entries.get(param, [])
            if len(recs) != 1:
                raise InvalidDataError(
                    f"ECMWF index for +{lead}h has {len(recs)} records for {param} (expected 1)"
                )
            offset, length = int(recs[0]["_offset"]), int(recs[0]["_length"])
            out[field] = IndexEntry(field, ByteRange(offset, offset + length - 1))
        return out

    def source_attrs(self, init_time: datetime) -> dict[str, object]:
        return {
            "source": f"ECMWF Open Data, {self.model_path.upper()} {self.resolution_deg(init_time)} deg {self.stream}",
            "native_resolution_deg": self.resolution_deg(init_time),
            "source_url": self.grib_url(init_time, 0).rsplit("/", 1)[0],
            "institution": "European Centre for Medium-Range Weather Forecasts",
            "license": "CC-BY-4.0",
        }
