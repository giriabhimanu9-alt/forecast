"""Re-run the Phase 4 data-access probes (metadata/index requests only) and save machine-readable results.

    python scripts/audit_phase4_sources.py      -> results/phase4/data_audit/source_audit.json

The human-readable audit with conclusions is PHASE4_DATA_AUDIT.md.
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BACKEND_ROOT  # noqa: E402

S = requests.Session()
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
EC = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
WB2 = "https://storage.googleapis.com/weatherbench2/datasets/"
GRID = "240x121_equiangular_with_poles_conservative.zarr"


def get(url, method="GET"):
    for i in range(8):
        try:
            r = S.request(method, url, timeout=90)
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(2 * (i + 1))
            continue
        if r.status_code in (429, 503):
            time.sleep(3 * (i + 1))
            continue
        return r
    return None


def s3_prefixes(base, prefix):
    r = get(f"{base}/?list-type=2&prefix={prefix}&delimiter=/&max-keys=1000")
    return re.findall(r"<Prefix>([^<]*)</Prefix>", r.text)[1:] if r is not None else []


def ec_steps_and_params(product_dir):
    r = get(f"{EC}/?list-type=2&prefix={product_dir}&max-keys=1000")
    keys = re.findall(r"<Key>([^<]*)</Key>", r.text) if r is not None else []
    steps = sorted({int(m) for k in keys for m in re.findall(r"-(\d+)h-", k)})
    idx = next((k for k in keys if k.endswith(".index") and "-24h-" in k), None)
    params = []
    if idx:
        txt = get(f"{EC}/{idx}").text
        params = sorted({json.loads(l)["param"] for l in txt.splitlines() if l.startswith("{")})
    return {"steps_first": steps[:6], "step_interval_h": (steps[1] - steps[0]) if len(steps) > 1 else None,
            "max_step": steps[-1] if steps else None, "has_tp": "tp" in params}


def wb2_meta(path):
    r = get(WB2 + path + "/.zmetadata")
    if r is None or r.status_code != 200:
        return {"accessible": False}
    md = r.json()["metadata"]
    arrays = sorted({k.split("/")[0] for k in md if k.endswith("/.zarray")})
    return {"accessible": True, "precip_variables": [a for a in arrays if "precip" in a],
            "n_time": md.get("time/.zarray", {}).get("shape", [None])[0],
            "n_lead": md.get("prediction_timedelta/.zarray", {}).get("shape", [None])[0]}


def main() -> int:
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out["gfs"] = {}
    for y in (2021, 2022, 2023, 2024, 2025):
        r = get(f"{GFS}/gfs.{y}0710/00/atmos/gfs.t00z.pgrb2.0p25.f027.idx")
        r1 = get(f"{GFS}/gfs.{y}0710/00/atmos/gfs.t00z.pgrb2.1p00.f027.idx")
        out["gfs"][y] = {"0p25_idx": r is not None and r.status_code == 200, "1p00_idx": r1 is not None and r1.status_code == 200,
                         "apcp_0_27h": bool(r is not None and "APCP:surface:0-27 hour acc" in r.text)}
    out["ecmwf_mirror"] = {"first_dates": s3_prefixes(EC, "")[:3]}
    for day in ("20230710", "20240710", "20250710"):
        prods = []
        for p in s3_prefixes(EC, f"{day}/00z/"):
            prods += s3_prefixes(EC, p) or [p]
        info = {}
        for p in prods:
            sub = s3_prefixes(EC, p)
            for d in (sub or [p]):
                if d.rstrip("/").endswith("oper") or d.rstrip("/").endswith(("0p25", "0p4-beta")):
                    target = d if d.rstrip("/").endswith("oper") else d + "oper/"
                    info[target] = ec_steps_and_params(target)
        out["ecmwf_mirror"][day] = info
    out["weatherbench2"] = {
        "graphcast_2018": wb2_meta(f"graphcast_v2/2018-{GRID}"), "graphcast_2020": wb2_meta(f"graphcast_v2/2020-{GRID}"),
        "graphcast_2022": wb2_meta(f"graphcast_v2/2022-{GRID}"), "hres": wb2_meta(f"hres/2016-2022-0012-{GRID}"),
        "era5": wb2_meta(f"era5/1959-2023_01_10-6h-{GRID}"), "pangu": wb2_meta(f"pangu/2018-2022_0012_{GRID}"),
        "neuralgcm_2020": wb2_meta(f"neuralgcm_deterministic/2020-{GRID}"), "fuxi_2020": wb2_meta(f"fuxi/2020-{GRID}"),
    }
    r = get("https://noaa-oar-mlwp-data.s3.amazonaws.com/?list-type=2&prefix=GRAP_v100_GFS/2024/0710/&max-keys=5")
    sizes = [int(x) for x in re.findall(r"<Size>(\d+)</Size>", r.text)] if r is not None else []
    out["noaa_mlwp"] = {"graphcast_file_bytes": sizes[:2], "steps": "6 h (README)"}
    r = get("https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html")
    out["imd"] = {"archive_page_reachable": r is not None and r.status_code == 200,
                  "cached_archives": sorted(p.name for p in (BACKEND_ROOT / "data/raw/imd/archive/rain").glob("*.grd"))}
    dest = BACKEND_ROOT / "results" / "phase4" / "data_audit" / "source_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1)[:3000])
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
