"""Phase 4: multi-season IMD validation (Experiment A) + AI/NWP benchmark vs ERA5 (Experiment B).

    python scripts/run_phase4.py              # both experiments
    python scripts/run_phase4.py --only A     # IMD experiment only
    python scripts/run_phase4.py --only B     # needs the IMD land mask, so A runs first anyway

Run from backend/. Results: results/phase4/ and PHASE4_RESULTS.md.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_ROOT, load_settings  # noqa: E402
from app.services.verification.phase4_report import run_phase4  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=str(BACKEND_ROOT / "results" / "phase4"))
    p.add_argument("--only", choices=["A", "B"])
    a = p.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "cfgrib", "app.services.data.base", "app.services.verification.experiment"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    res = run_phase4(load_settings(), Path(a.output), run_a=True, run_b=a.only != "A")
    print("\nPhase 4 finished. Summary numbers (test period, all lead days):")
    for name, block in res["manifest"]["results"].items():
        print(f"  Experiment {name}:")
        for m, v in block.items():
            print(f"    {m:<24} MAE {v['mae']:.3f}  RMSE {v['rmse']:.2f}  bias {v['bias']:+.2f}  "
                  f"CSI64.5 {v['csi@64.5mm']:.3f}  POD {v['pod@64.5mm']:.3f}  FAR {v['far@64.5mm']:.3f}")
    print(f"Report: {BACKEND_ROOT / 'PHASE4_RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
