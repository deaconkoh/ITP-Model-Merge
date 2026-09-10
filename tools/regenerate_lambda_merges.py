"""Regenerate the lambda-grid merged checkpoints from the specialists.

The merges are pure deterministic functions of the two specialists and a weight, so they are
derived artifacts rather than experimental results: regeneration is bit-identical (verified,
max abs diff 0.0) and costs CPU only -- no GPU, no retraining.

They are therefore not kept in the working tree. Run this to rebuild the full grid, e.g.
before re-evaluating a lambda sweep on a new pool.

    python tools/regenerate_lambda_merges.py                 # full 21-point grid, both sizes
    python tools/regenerate_lambda_merges.py --sizes 10x5    # one size
    python tools/regenerate_lambda_merges.py --check         # report what is missing, build nothing
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "daniel/trained_network/SD2"
SEEDS = [111, 222, 333, 444]
# interior points of the 21-point grid; lambda=0 and lambda=1 are the specialists themselves
PCTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["10x5", "20x10"])
    ap.add_argument("--pcts", nargs="+", type=int, default=PCTS)
    ap.add_argument("--check", action="store_true", help="report missing merges, build nothing")
    args = ap.parse_args()

    missing, built, failed = [], 0, []
    for size in args.sizes:
        for seed in SEEDS:
            for pct in args.pcts:
                name = f"{size}+carbon+priority+soup_c{pct}_s{seed}"
                if (CKPT / f"{name}.pth").exists():
                    continue
                missing.append(name)
                if args.check:
                    continue
                lam = pct / 100
                cmd = [sys.executable, str(REPO / "tools/merge_weight_soup.py"),
                       "--data-source", "SD2",
                       "--models", f"{size}+carbon+priority+c_s{seed}",
                                   f"{size}+carbon+priority+m_s{seed}",
                       "--weights", str(lam), str(round(1 - lam, 10)),
                       "--out-name", name]
                r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
                if r.returncode != 0 or not (CKPT / f"{name}.pth").exists():
                    failed.append(name)
                else:
                    built += 1

    if args.check:
        print(f"missing merges: {len(missing)}")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
        return

    print(f"rebuilt {built} merged checkpoints")
    if failed:
        print(f"FAILED ({len(failed)}): {failed[:5]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
