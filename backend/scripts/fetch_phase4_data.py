"""Pre-download Phase 4 Experiment A inputs into the normal cache (safe to re-run; cached files are skipped).

    python scripts/fetch_phase4_data.py --source GFS
    python scripts/fetch_phase4_data.py --source ECMWF
    python scripts/fetch_phase4_data.py --source GFS_ATMOS
    python scripts/fetch_phase4_data.py --source IMD

Each source can run in its own process (different servers). The Phase 4 experiment
reuses exactly these cached files; nothing here is required, it only saves time.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.services.verification.phase4 import ExperimentAConfig, prefetch_experiment_a  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=["GFS", "ECMWF", "GFS_ATMOS", "IMD"])
    p.add_argument("--seasons", default="2023,2024,2025")
    a = p.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "cfgrib", "app.services.data.base"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = ExperimentAConfig(seasons=tuple(int(s) for s in a.seasons.split(",")))
    summary = prefetch_experiment_a(load_settings(), cfg, a.source)
    for line in summary:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
