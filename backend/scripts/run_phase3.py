"""Phase 3: regime-conditioned adaptive blending on real July 2025 data.

    python scripts/run_phase3.py

Run from the backend/ directory. Results go to results/phase3/.
Train 1-17 Jul, validation 18-23 Jul, final fit 1-23 Jul, untouched test 24-31 Jul (IMD day labels).
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_ROOT, ConfigError, load_settings  # noqa: E402
from app.services.verification.phase3 import Phase3Config, run_phase3  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=str(BACKEND_ROOT / "results" / "phase3"))
    p.add_argument("--no-wind", action="store_true", help="rainfall-only regime features")
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
    result = run_phase3(settings, Phase3Config(output_dir=Path(a.output), use_wind=not a.no_wind))
    print("\n" + "=" * 100 + "\n" + "\n".join(result.summary_lines) + "\n" + "=" * 100)
    print(f"Results: {a.output}")
    return 0 if result.status == "COMPLETED" else 2


if __name__ == "__main__":
    sys.exit(main())
