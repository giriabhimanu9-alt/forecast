"""Phase 2 historical verification + first blend.

    python scripts/run_phase2.py --preset pilot      # July 1-10 2025: alignment/pipeline test
    python scripts/run_phase2.py --preset july2025   # July 1-31 2025: first historical experiment

Run from the backend/ directory. Results go to results/<name>/.
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_ROOT, ConfigError, load_settings  # noqa: E402
from app.services.verification.experiment import ExperimentConfig, run_experiment  # noqa: E402

PRESETS = {
    "pilot": ("july2025_pilot", date(2025, 7, 1), date(2025, 7, 10)),
    "july2025": ("july2025", date(2025, 7, 1), date(2025, 7, 31)),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="pilot")
    p.add_argument("--start", type=date.fromisoformat, help="first IMD day label (overrides preset)")
    p.add_argument("--end", type=date.fromisoformat, help="last IMD day label (overrides preset)")
    p.add_argument("--name", help="experiment name / results sub-folder")
    p.add_argument("--models", default="GFS,ECMWF")
    p.add_argument("--cycles", default="0", help="UTC cycles, e.g. 0 or 0,12")
    p.add_argument("--lead-days", default="1,2,3")
    p.add_argument("--train-fraction", type=float, default=0.75)
    p.add_argument("--thresholds", default="2.5,15.6,64.5", help="event thresholds, mm/24h")
    p.add_argument("--epsilon", type=float, default=0.1, help="inverse-error epsilon, mm")
    p.add_argument("--output-root", default=str(BACKEND_ROOT / "results"))
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)

    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("urllib3", "cfgrib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 3

    name, start, end = PRESETS[a.preset]
    cfg = ExperimentConfig(
        name=a.name or name,
        valid_start=a.start or start,
        valid_end=a.end or end,
        output_dir=Path(a.output_root) / (a.name or name),
        models=tuple(m.strip().upper() for m in a.models.split(",") if m.strip()),
        cycles=tuple(int(c) for c in a.cycles.split(",")),
        lead_days=tuple(int(d) for d in a.lead_days.split(",")),
        train_fraction=a.train_fraction,
        thresholds_mm=tuple(float(t) for t in a.thresholds.split(",")),
        epsilon_mm=a.epsilon,
    )
    result = run_experiment(settings, cfg)
    print("\n" + "=" * 72)
    print("\n".join(result.summary_lines))
    print("=" * 72)
    print(f"Results: {cfg.output_dir}")
    return 0 if result.status == "COMPLETED" else 2


if __name__ == "__main__":
    sys.exit(main())
