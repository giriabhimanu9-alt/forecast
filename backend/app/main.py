"""Command-line entry point for the Phase 1 ingestion pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import ConfigError, load_settings
from app.models.schema import ProviderStatus
from app.services.pipeline import PipelineReport, run_phase1


def format_summary(report: PipelineReport) -> str:
    s = report.settings
    grid = s["grid"]
    lines = ["", "=" * 64, "PHASE 1 PIPELINE SUMMARY", "=" * 64]
    labels = {"GFS": "GFS", "ECMWF": "ECMWF", "IMD": "OBSERVED", "AI": "AI"}
    for r in report.results:
        lines.append(f"{labels.get(r.name, r.name) + ':':<11}{r.status.value}")
        if r.message and r.status is not ProviderStatus.AVAILABLE:
            lines.append(f"{'':<11}-> {r.message}")
    lines += [
        "",
        f"Target grid: {grid['resolution']} deg ({grid['shape'][0]} lat x {grid['shape'][1]} lon)",
        f"Region:      {grid['name']} (lat {grid['lat_min']}..{grid['lat_max']}, lon {grid['lon_min']}..{grid['lon_max']})",
        f"Variables:   {', '.join(s['variables'])} (primary: precipitation)",
        f"Init time:   {report.init_time or 'n/a'}  [{report.init_time_note}]",
    ]
    for r in report.results:
        if not r.ok:
            continue
        d = r.details
        lines.append("")
        lines.append(f"[{r.name}] {r.message}")
        lines.append(f"  dims:       {d['dims']}")
        if "lead_hours" in d:
            lines.append(f"  lead times: {d['lead_hours'][0]}..{d['lead_hours'][-1]} h ({len(d['lead_hours'])} steps)")
            lines.append(f"  valid:      {d['valid_time_range'][0]} -> {d['valid_time_range'][1]}")
        else:
            lines.append(f"  time range: {d['time_range'][0]} -> {d['time_range'][1]} (24 h windows ending 03 UTC)")
        pr = d["statistics"]["precipitation"]
        lines.append(f"  precip mm:  min {pr['min']}  mean {pr['mean']}  max {pr['max']}")
        for f in r.output_files:
            lines.append(f"  saved:      {f}")
    files = sum(len(r.raw_files) for r in report.results if r.ok)
    lines += ["", f"Files processed: {files} raw files", f"Report:          {report.report_path}", "=" * 64]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIH PS 26081 - Phase 1 data ingestion & normalisation")
    parser.add_argument("--log-level", default=None, help="override LOG_LEVEL")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 3

    logging.basicConfig(
        level=getattr(logging, (args.log_level or settings.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "cfgrib", "ecmwf"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    report = run_phase1(settings)
    print(format_summary(report))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
