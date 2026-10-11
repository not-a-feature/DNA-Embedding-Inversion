"""Pick ``optim.scheduler.max_lr`` from the calibrate.sbatch sweep, unattended.

Encodes the rule stated in calibrate.sbatch / calibrate_report.sh: take the
LARGEST max_lr that is stable for ALL THREE FMs. A run counts as stable only if
it finished the whole epoch budget with no divergence, no adaptive-stabiliser
fire, and a best epoch after the first.

Prints the chosen LR on stdout (and nothing else, so the caller can capture it);
diagnostics go to stderr. Exits non-zero and prints NOTHING on stdout when no LR
qualifies -- the caller must then NOT launch the benchmark. Running 72h at an LR
the calibration just showed to be unstable is strictly worse than waiting for a
human, so this refuses rather than falls back to a "least bad" option.

Usage:  python select_max_lr.py [--calib-dir outputs/calib] [--epochs 20]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List

# Must match FMS/LRS/EPOCHS in calibrate.sbatch.
FMS = ["dnabert2", "evo2", "ntv2"]
LRS = ["2.5e-4", "5e-4", "1e-3"]

EPOCH_RE = re.compile(
    r"epoch=(\d+) \| train_loss=(\S+) \| val_loss=(\S+) \| time_sec=(\S+)"
)


def assess(log_path: str, epochs: int) -> Dict[str, object]:
    """Judge one calibration run from its train.log."""
    if not os.path.isfile(log_path):
        return {"stable": False, "why": "no train.log"}

    with open(log_path) as fh:
        text = fh.read()

    rows = EPOCH_RE.findall(text)
    if not rows:
        return {"stable": False, "why": "no completed epochs"}

    # A requeued run resumes and appends, so the log can carry an epoch twice;
    # judge by the highest epoch reached rather than the row count.
    seen = {int(e) for e, _, _, _ in rows}
    finite = [(int(e), float(v)) for e, _, v, _ in rows if v.lower() != "nan"]

    if not finite:
        return {"stable": False, "why": "every epoch diverged (nan)"}

    best_epoch, best_val = min(finite, key=lambda r: r[1])
    reasons: List[str] = []

    if max(seen) < epochs:
        # OneCycle only anneals to its minimum at total_steps; an unfinished run
        # has not exercised the schedule it is being judged on.
        reasons.append(f"incomplete ({max(seen)}/{epochs} epochs)")
    if any(v.lower() == "nan" for _, _, v, _ in rows):
        reasons.append("diverged (nan)")
    if "stabiliser" in text:
        reasons.append("adaptive stabiliser fired")
    if best_epoch == 1 and len(finite) > 3:
        # The exact signature of the previous wasted run: the model never beat
        # its epoch-1 loss, i.e. the LR peak degraded it.
        reasons.append("best epoch = 1 (never improved)")

    return {
        "stable": not reasons,
        "why": ", ".join(reasons) if reasons else "ok",
        "best_val": best_val,
        "best_epoch": best_epoch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib-dir", default="outputs/calib")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    # Largest first, so the first fully-stable LR wins.
    ordered = sorted(LRS, key=float, reverse=True)
    chosen = None

    print("Calibration verdicts:", file=sys.stderr)
    for lr in ordered:
        verdicts = {fm: assess(os.path.join(args.calib_dir, f"{fm}_lr{lr}", "train.log"), args.epochs) for fm in FMS}
        for fm in FMS:
            v = verdicts[fm]
            detail = ""
            if "best_val" in v:
                detail = f" best_val={v['best_val']:.4f} @ep{v['best_epoch']}"
            print(f"  max_lr={lr:<8} {fm:<9} {'STABLE' if v['stable'] else 'unstable'}: {v['why']}{detail}", file=sys.stderr)
        if all(v["stable"] for v in verdicts.values()) and chosen is None:
            chosen = lr

    if chosen is None:
        print(
            "\nNo max_lr was stable for all three FMs. Refusing to launch the 72h "
            "benchmark on an LR the calibration just rejected -- inspect with "
            "`bash calibrate_report.sh` and submit run_experiments.sbatch by hand.",
            file=sys.stderr,
        )
        return 1

    print(f"\nSelected max_lr={chosen} (largest stable across {', '.join(FMS)})", file=sys.stderr)
    print(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
