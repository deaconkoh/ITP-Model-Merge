"""Has the lambda grid converged? 5 vs 11 vs 21 points, plus practical-resolution ceilings.

Steps:
  2  convergence: oracle gain at 5 / 11 / 21 points, LOO-selected fixed-lambda baseline
  3  where the per-instance optima actually sit (and how many need the half-steps)
  4  skew of per-instance gains at the new resolution
  5  restricted-selection ceiling: best achievable if a mechanism may only pick among k
     well-spaced lambdas (k = 1,3,5,7), chosen by greedy forward selection
     (submodular objective, so greedy carries a 1-1/e guarantee; verified against
     exhaustive search at k=3)

Evaluation only; reads existing .npy results. Run from the repository root.
"""
import itertools
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = [111, 222, 333, 444]
W = 0.5

GRID5 = [0.0, 0.3, 0.5, 0.7, 1.0]
GRID11 = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
GRID21 = [round(x, 2) for x in np.arange(0.0, 1.001, 0.05)]


def tag_for(lam):
    if lam == 0.0:
        return "m_s{s}"
    if lam == 1.0:
        return "c_s{s}"
    return f"soup_c{int(round(lam*100))}_s{{s}}"


def load(size, grid):
    pool = f"trainvali_{size}+carbon+priority"
    d = REPO / "daniel/test_results/SD2" / pool
    mets = {}
    for lam in grid:
        per_seed = []
        for s in SEEDS:
            f = d / f"Result_DANIELG+{size}+carbon+priority+{tag_for(lam).format(s=s)}_{pool}.npy"
            if not f.exists():
                return None, str(f)
            per_seed.append(np.load(f)[:, :2])
        mets[lam] = np.stack(per_seed)
    return mets, None


def scalarize(mets, grid):
    ref_m = mets[0.5][:, :, 0]; ref_c = mets[0.5][:, :, 1]
    return np.stack([W * (mets[l][:, :, 1] / ref_c) + (1 - W) * (mets[l][:, :, 0] / ref_m)
                     for l in grid])


def loo_baseline_vals(O_seed):
    """Per-instance objective under a fixed lambda chosen leave-one-out."""
    n = O_seed.shape[1]
    base = np.empty(n); idx = np.empty(n, int)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        bi = int(np.argmin(O_seed[:, m].mean(1)))
        base[i] = O_seed[bi, i]; idx[i] = bi
    return base, idx


_BASE_CACHE = {}


def base_mean_for(O_seed, key):
    """LOO fixed-lambda baseline. Independent of any subset, so computed once per (size, seed)."""
    if key not in _BASE_CACHE:
        _BASE_CACHE[key] = loo_baseline_vals(O_seed)[0].mean()
    return _BASE_CACHE[key]


def gain_loo(O_seed, subset=None, key=None):
    """Oracle gain (%) of per-instance selection (restricted to `subset` if given).

    The fixed-lambda baseline is always selected from the FULL grid, since a real
    deployment would pick its single best lambda from anywhere.
    """
    bm = base_mean_for(O_seed, key) if key is not None else loo_baseline_vals(O_seed)[0].mean()
    sel = O_seed if subset is None else O_seed[subset]
    return 100 * (bm - sel.min(0).mean()) / bm


def per_instance_gains(O_seed):
    base, _ = loo_baseline_vals(O_seed)
    return 100 * (base - O_seed.min(0)) / base


def greedy_subset(O, k, size):
    """Greedy forward selection of k lambda indices maximising mean oracle gain over seeds."""
    n_lam = O.shape[0]
    chosen = []
    for _ in range(k):
        best, best_g = None, -np.inf
        for c in range(n_lam):
            if c in chosen:
                continue
            trial = chosen + [c]
            g = np.mean([gain_loo(O[:, si, :], subset=trial, key=(size, si))
                         for si in range(len(SEEDS))])
            if g > best_g:
                best_g, best = g, c
        chosen.append(best)
    return sorted(chosen), best_g


def main():
    for size in ["10x5", "20x10"]:
        print("=" * 82)
        print(f"{size}")
        print("=" * 82)

        avail = {}
        for name, grid in [("5", GRID5), ("11", GRID11), ("21", GRID21)]:
            mets, missing = load(size, grid)
            if mets is None:
                print(f"  {name}-point: SKIP (missing {Path(missing).name})")
                continue
            avail[name] = (mets, grid)

        # ---- Step 2: convergence
        print("\n[STEP 2 CONVERGENCE] oracle gain @ w=0.5, LOO-selected fixed-lambda baseline")
        gains = {}
        for name in ["5", "11", "21"]:
            if name not in avail:
                continue
            mets, grid = avail[name]
            O = scalarize(mets, grid)
            gs = [gain_loo(O[:, si, :], key=(size, name, si)) for si in range(len(SEEDS))]
            gains[name] = np.mean(gs)
            print(f"  {name:>2}-point grid: {np.mean(gs):5.2f}%  (sd {np.std(gs, ddof=1):.2f})  "
                  f"per-seed {np.round(gs,2)}")
        if "5" in gains and "11" in gains:
            print(f"   5 -> 11 : {100*(gains['11']-gains['5'])/gains['5']:+6.1f}%")
        if "11" in gains and "21" in gains:
            d = 100 * (gains["21"] - gains["11"]) / gains["11"]
            print(f"  11 -> 21 : {d:+6.1f}%   -> "
                  f"{'CONVERGED (under 10%)' if abs(d) < 10 else 'NOT converged (>=10%)'}")

        if "21" not in avail:
            print()
            continue
        mets21, grid21 = avail["21"]
        O21 = scalarize(mets21, grid21)

        # ---- Step 3: where the optima sit
        print("\n[STEP 3 WHERE THE OPTIMA SIT] per-instance argmin lambda, 21-point grid")
        argm = O21.argmin(0)                       # (n_seed, n_inst)
        lams = np.array(grid21)[argm].ravel()
        halfsteps = {0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95}
        on_half = np.mean([l in halfsteps for l in lams])
        print(f"  optima landing on a NEW half-step (unreachable at 11 points): {100*on_half:5.1f}%")
        print(f"  distribution of optimal lambda:")
        vals, cnts = np.unique(lams, return_counts=True)
        for v, c in zip(vals, cnts):
            bar = "#" * int(round(60 * c / cnts.max()))
            print(f"    {v:.2f}: {100*c/len(lams):5.1f}%  {bar}")
        print(f"  spread: mean={lams.mean():.3f} sd={lams.std():.3f} "
              f"IQR=[{np.percentile(lams,25):.2f}, {np.percentile(lams,75):.2f}]  "
              f"p10={np.percentile(lams,10):.2f} p90={np.percentile(lams,90):.2f}")
        for lo, hi in [(0.1, 0.5), (0.2, 0.6), (0.0, 0.5)]:
            frac = np.mean((lams >= lo) & (lams <= hi))
            print(f"  fraction of optima within [{lo}, {hi}]: {100*frac:5.1f}%")

        # ---- Step 4: skew at 21 points
        allg = np.concatenate([per_instance_gains(O21[:, si, :]) for si in range(len(SEEDS))])
        print(f"\n[STEP 4 SKEW at 21 points] n={len(allg)}")
        print(f"  mean={allg.mean():.2f}%  median={np.median(allg):.2f}%  "
              f"p90={np.percentile(allg,90):.2f}%  p95={np.percentile(allg,95):.2f}%  "
              f"max={allg.max():.2f}%")
        for thr in [1, 2, 5, 10]:
            frac = 100 * (allg > thr).mean()
            share = 100 * allg[allg > thr].sum() / allg.sum() if (allg > thr).any() else 0.0
            print(f"    >{thr:2d}%: {frac:5.1f}% of instances, holding {share:5.1f}% of all gain")
        top10 = np.sort(allg)[::-1][:max(1, len(allg)//10)]
        print(f"  top decile of instances holds {100*top10.sum()/allg.sum():.1f}% of all available gain")

        # ---- Step 5: restricted selection
        print(f"\n[STEP 5 PRACTICAL-RESOLUTION CEILING] best achievable with only k lambdas")
        full = np.mean([gain_loo(O21[:, si, :], key=(size, si)) for si in range(len(SEEDS))])
        for k in [1, 3, 5, 7]:
            sub, g = greedy_subset(O21, k, size)
            chosen = [grid21[i] for i in sub]
            print(f"  k={k}: {g:5.2f}%  ({100*g/full:5.1f}% of the unrestricted {full:.2f}%)  "
                  f"lambdas={chosen}")
        # exhaustive check at k=3
        best_ex, best_g_ex = None, -np.inf
        for combo in itertools.combinations(range(len(grid21)), 3):
            g = np.mean([gain_loo(O21[:, si, :], subset=list(combo), key=(size, si)) for si in range(len(SEEDS))])
            if g > best_g_ex:
                best_g_ex, best_ex = g, combo
        print(f"  k=3 exhaustive check: {best_g_ex:5.2f}% with "
              f"lambdas={[grid21[i] for i in best_ex]}  (greedy was within "
              f"{100*(best_g_ex-greedy_subset(O21,3,size)[1])/best_g_ex:.1f}%)")
        print()


if __name__ == "__main__":
    main()
