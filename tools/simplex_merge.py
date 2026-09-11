"""Build 3-specialist compositions on the 2-simplex.

Composition weights (w_m, w_c, w_p) sum to 1 and are encoded in the checkpoint name as
permille integers, e.g. (1/3,1/3,1/3) -> simplex_m333c333p334_s111. Permille keeps the
centroid exact and supports refinement down to 0.1% without ambiguity.

Resolution policy (carried forward from the resolution finding): do NOT default to a coarse
grid. The programme uses coarse-to-fine --
    stage 1: step 0.10 -> 66 points, locates the optimum region
    stage 2: step 0.05 within a neighbourhood of the stage-1 optimum
with the refinement genuinely fine rather than a token second pass.

    python tools/simplex_merge.py --list --step 0.1
    python tools/simplex_merge.py --build --step 0.1 --sizes 10x5 --seeds 111
    python tools/simplex_merge.py --build --points 333,333,334 500,250,250 --smoke
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "daniel/trained_network/SD2"
SEEDS = [111, 222, 333, 444]

# real specialists; --smoke substitutes stand-ins so the machinery can be tested
# before the priority specialists exist
SPECIALISTS = {"m": "m_s{seed}", "c": "c_s{seed}", "p": "p_s{seed}"}
SMOKE_SPECIALISTS = {"m": "m_s{seed}", "c": "c_s{seed}", "p": "c_s222"}


def simplex_points(step):
    """All (w_m, w_c, w_p) on the 2-simplex at the given step, as permille integers."""
    K = int(round(1 / step))
    pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            k = K - i - j
            pts.append(_permille(i / K, j / K, k / K))
    return pts


def _permille(wm, wc, wp):
    a, b = int(round(wm * 1000)), int(round(wc * 1000))
    return (a, b, 1000 - a - b)      # last absorbs rounding so the triple sums to 1000 exactly


def name_for(size, pts, seed):
    return f"{size}+carbon+priority+simplex_m{pts[0]:03d}c{pts[1]:03d}p{pts[2]:03d}_s{seed}"


def neighbourhood(centre, radius, step):
    """Points within `radius` (in permille, L1/2) of a centre, on a finer grid."""
    out = []
    for p in simplex_points(step):
        if sum(abs(a - b) for a, b in zip(p, centre)) / 2 <= radius:
            out.append(p)
    return out


def build(size, seed, pts, specialists, dry_run=False):
    name = name_for(size, pts, seed)
    if (CKPT / f"{name}.pth").exists():
        return "exists"
    models, weights = [], []
    for key, w in zip(("m", "c", "p"), pts):
        if w == 0:
            continue                       # a zero-weight specialist is simply omitted
        tmpl = specialists[key]
        tag = tmpl.format(seed=seed) if "{seed}" in tmpl else tmpl
        models.append(f"{size}+carbon+priority+{tag}")
        weights.append(w / 1000)
    if dry_run:
        return f"would build {name} from {list(zip(models, weights))}"
    cmd = [sys.executable, str(REPO / "tools/merge_weight_soup.py"), "--data-source", "SD2",
           "--models", *models, "--weights", *[str(w) for w in weights], "--out-name", name]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return "ok" if r.returncode == 0 and (CKPT / f"{name}.pth").exists() else f"FAIL: {r.stderr[-200:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--points", nargs="+", help="explicit permille triples, e.g. 333,333,334")
    ap.add_argument("--sizes", nargs="+", default=["10x5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--smoke", action="store_true",
                    help="use stand-in specialists (no priority specialist needed yet)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pts = ([tuple(int(x) for x in p.split(",")) for p in args.points] if args.points
           else simplex_points(args.step))
    for p in pts:
        assert sum(p) == 1000, f"weights must sum to 1000 permille, got {p}"

    if args.list or not args.build:
        print(f"{len(pts)} simplex points at step {args.step} (permille, sum=1000)")
        for p in pts[:12]:
            print(f"   m={p[0]/10:5.1f}%  c={p[1]/10:5.1f}%  p={p[2]/10:5.1f}%")
        if len(pts) > 12:
            print(f"   ... and {len(pts)-12} more")
        edges = sum(1 for p in pts if 0 in p)
        print(f"   {edges} of {len(pts)} points lie on an edge (one specialist unused)")
        return

    spec = SMOKE_SPECIALISTS if args.smoke else SPECIALISTS
    if args.smoke:
        print("SMOKE MODE: priority slot substituted with a stand-in checkpoint")
    n_ok = 0
    for size in args.sizes:
        for seed in args.seeds:
            for p in pts:
                r = build(size, seed, p, spec, dry_run=args.dry_run)
                if r in ("ok", "exists") or r.startswith("would"):
                    n_ok += 1
                else:
                    print(f"  {name_for(size,p,seed)}: {r}")
    print(f"built/verified {n_ok} compositions")


if __name__ == "__main__":
    main()
