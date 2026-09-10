"""Verification of the oracle-headroom ceiling that every negative conclusion rests on.

Addresses:
  Step 2  denser lambda grid (5-point vs 11-point)
  Step 4  preference sensitivity (w_carbon swept finely)
  Step 5a is the oracle genuinely an upper bound?
  Step 5b DISTRIBUTION of per-instance gains, not just the mean
  Step 5c per-seed vs seed-averaged oracle gain

Also corrects a bias found in the original computation: the fixed-lambda baseline was
chosen by argmin over the mean of the SAME instances it was then scored on, giving the
baseline hindsight and understating the oracle gap. Both the original (hindsight) and a
leave-one-out-selected baseline are reported.

Evaluation-only; reads existing .npy results. Run from the repository root.
    python tools/verify_oracle.py --grid 5
    python tools/verify_oracle.py --grid 11
"""
import argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = [111, 222, 333, 444]
GRID5 = [0.0, 0.3, 0.5, 0.7, 1.0]
GRID11 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def tag_for(lam):
    if lam == 0.0:
        return "m_s{s}"
    if lam == 1.0:
        return "c_s{s}"
    return f"soup_c{int(round(lam*100))}_s{{s}}"


def load_metrics(size, grid):
    pool = f"trainvali_{size}+carbon+priority"
    d = REPO / "daniel/test_results/SD2" / pool
    out = {}
    for lam in grid:
        per_seed = []
        for s in SEEDS:
            f = d / f"Result_DANIELG+{size}+carbon+priority+{tag_for(lam).format(s=s)}_{pool}.npy"
            if not f.exists():
                return None, str(f)
        for s in SEEDS:
            f = d / f"Result_DANIELG+{size}+carbon+priority+{tag_for(lam).format(s=s)}_{pool}.npy"
            per_seed.append(np.load(f)[:, :2])
        out[lam] = np.stack(per_seed)          # (n_seed, n_inst, 2)
    return out, None


def scalarize(mets, grid, w):
    ref_m = mets[0.5][:, :, 0]; ref_c = mets[0.5][:, :, 1]
    return np.stack([w * (mets[l][:, :, 1] / ref_c) + (1 - w) * (mets[l][:, :, 0] / ref_m)
                     for l in grid])           # (n_lam, n_seed, n_inst)


def gain_hindsight(O_seed):
    """O_seed: (n_lam, n_inst). Baseline lambda chosen on the same instances (original method)."""
    static = O_seed.mean(1)
    bi = int(np.argmin(static))
    oracle = O_seed.min(0).mean()
    return 100 * (static[bi] - oracle) / static[bi], bi


def gain_loo(O_seed):
    """Baseline lambda chosen leave-one-out, so it never sees the instance it is scored on."""
    n = O_seed.shape[1]
    base = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        bi = int(np.argmin(O_seed[:, m].mean(1)))
        base[i] = O_seed[bi, i]
    baseline = base.mean()
    oracle = O_seed.min(0).mean()
    return 100 * (baseline - oracle) / baseline


def per_instance_gains(O_seed):
    """% gain per instance from per-instance lambda vs the LOO-selected fixed lambda."""
    n = O_seed.shape[1]
    g = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        bi = int(np.argmin(O_seed[:, m].mean(1)))
        best = O_seed[:, i].min()
        g[i] = 100 * (O_seed[bi, i] - best) / O_seed[bi, i]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, choices=[5, 11], default=5)
    ap.add_argument("--sizes", nargs="+", default=["10x5", "20x10"])
    args = ap.parse_args()
    grid = GRID5 if args.grid == 5 else GRID11

    for size in args.sizes:
        mets, missing = load_metrics(size, grid)
        if mets is None:
            print(f"### {size}: SKIP — missing {missing}\n")
            continue
        n_inst = mets[0.5].shape[1]
        print("=" * 80)
        print(f"{size} — {len(grid)}-point grid {grid}, n={n_inst}, {len(SEEDS)} seeds")
        print("=" * 80)

        O = scalarize(mets, grid, 0.5)

        # ---- 5a: upper-bound sanity
        per_inst_min = O.min(0)
        viol = 0
        for li in range(O.shape[0]):
            viol += int((O[li] < per_inst_min - 1e-12).sum())
        print(f"\n[5a UPPER BOUND] per-instance oracle is <= every fixed lambda everywhere: "
              f"{'PASS' if viol == 0 else f'FAIL ({viol} violations)'}")
        print(f"       oracle picks argmin over lambda per (seed, instance) with full hindsight,")
        print(f"       so no deployable rule can beat it by construction.")

        # ---- 5c + bias correction
        print(f"\n[ORACLE GAIN @ w=0.5]")
        hind, loo = [], []
        for si, s in enumerate(SEEDS):
            g_h, bi = gain_hindsight(O[:, si, :])
            g_l = gain_loo(O[:, si, :])
            hind.append(g_h); loo.append(g_l)
            print(f"  seed {s}: hindsight-baseline {g_h:5.2f}%   LOO-baseline {g_l:5.2f}%   "
                  f"(best fixed lambda = {grid[bi]})")
        print(f"  per-seed mean : hindsight {np.mean(hind):5.2f}% (sd {np.std(hind, ddof=1):.2f})"
              f"   LOO {np.mean(loo):5.2f}% (sd {np.std(loo, ddof=1):.2f})")
        O_avg = O.mean(1)                      # seed-averaged objectives, then oracle
        g_avg, _ = gain_hindsight(O_avg)
        print(f"  seed-AVERAGED objectives first: {g_avg:5.2f}%   "
              f"(vs {np.mean(hind):.2f}% per-seed -> averaging "
              f"{'suppresses' if g_avg < np.mean(hind) else 'inflates'} headroom)")

        # ---- 5b: distribution
        allg = np.concatenate([per_instance_gains(O[:, si, :]) for si in range(len(SEEDS))])
        print(f"\n[5b DISTRIBUTION of per-instance gains] (LOO baseline, pooled over seeds, "
              f"n={len(allg)})")
        pct = np.percentile(allg, [50, 75, 90, 95, 99])
        print(f"  mean={allg.mean():.2f}%  median={pct[0]:.2f}%  p75={pct[1]:.2f}%  "
              f"p90={pct[2]:.2f}%  p95={pct[3]:.2f}%  p99={pct[4]:.2f}%  max={allg.max():.2f}%")
        for thr in [0.5, 1, 2, 5, 10]:
            frac = 100 * (allg > thr).mean()
            print(f"    instances with gain > {thr:4.1f}% : {frac:5.1f}%   "
                  f"(they hold {100*allg[allg>thr].sum()/allg.sum():5.1f}% of all available gain)")

        # ---- Step 4: preference sensitivity
        print(f"\n[STEP 4 PREFERENCE SENSITIVITY] oracle gain (LOO baseline) vs w_carbon")
        for w in [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]:
            Ow = scalarize(mets, grid, w)
            gs = [gain_loo(Ow[:, si, :]) for si in range(len(SEEDS))]
            bl = [grid[int(np.argmin(Ow[:, si, :].mean(1)))] for si in range(len(SEEDS))]
            print(f"  w={w:.1f}: oracle gain {np.mean(gs):5.2f}% (sd {np.std(gs, ddof=1):.2f})   "
                  f"best fixed lambda per seed {bl}")
        print()


if __name__ == "__main__":
    main()
