"""Track B1: seed-repeat of the v2 structured-annotation pre-check.

The claim under test is a SCALING claim:
    oracle headroom under structured annotation does not decline with instance size,
    whereas under i.i.d. annotation it does.

Earlier v2 numbers were a single RNG draw at n=50. Here both schemes are resampled across
several draws at n=100 so that the contrast can be compared against draw-to-draw noise.
Both schemes are resampled (not just the structured one) so the i.i.d. arm also gets a
spread; routing and processing times are identical throughout.

CPU only. No checkpoints, no GPU, nothing written to the data tree.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import precheck_structured_headroom as P

N_DRAWS = 4
N_INST = 100
ALPHA = 2.0
BASE_SEED = 20260910


def iid_carbon(jobs, rng):
    """Resample the current convention: carbon ~ U{1..100} i.i.d. per (op, machine) pair."""
    return [[[(m, pt, float(rng.integers(1, 101))) for m, pt, _ in opts] for opts in ops]
            for ops in jobs]


def structured_v2(jobs, n_mch, rng, alpha_max=ALPHA):
    """carbon(o,m) = P_m * base_o : machine rate anti-correlated with machine speed, carbon
    magnitude set by an operation-level term independent of processing time."""
    kappa = rng.uniform(0.0, 1.0)
    tot = np.zeros(n_mch); cnt = np.zeros(n_mch)
    for ops in jobs:
        for opts in ops:
            for m, pt, _ in opts:
                tot[m] += pt; cnt[m] += 1
    tbar = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    valid = ~np.isnan(tbar)
    z = np.zeros(n_mch)
    if valid.sum() > 1 and np.nanstd(tbar[valid]) > 0:
        z[valid] = -(tbar[valid] - np.nanmean(tbar[valid])) / np.nanstd(tbar[valid])
    Pm = np.clip(1.0 + kappa * alpha_max * z, 0.05, None)
    out = []
    for ops in jobs:
        job = []
        for opts in ops:
            base = rng.uniform(1, 100)
            job.append([(m, pt, max(1e-6, Pm[m] * base * (1 + rng.normal(0, 0.05))))
                        for m, pt, _ in opts])
        out.append(job)
    return out


def headroom(instances):
    """Oracle gain (%) of per-instance lambda selection over the best single static lambda."""
    n = len(instances)
    MK = np.zeros((len(P.LAMBDAS), n)); CB = np.zeros((len(P.LAMBDAS), n))
    for idx, (n_jobs, n_mch, jobs) in enumerate(instances):
        pts = [pt for ops in jobs for opts in ops for _, pt, _ in opts]
        cbs = [cb for ops in jobs for opts in ops for _, _, cb in opts]
        pm, cm = float(np.mean(pts)), float(np.mean(cbs))
        for li, lam in enumerate(P.LAMBDAS):
            mk, cb = P.dispatch(jobs, n_mch, lam, pm, cm)
            MK[li, idx] = mk; CB[li, idx] = cb
    half = P.LAMBDAS.index(0.5)
    obj = P.W_CARBON * (CB / CB[half]) + (1 - P.W_CARBON) * (MK / MK[half])
    static = obj.mean(1); bi = int(np.argmin(static))
    return 100 * (static[bi] - obj.min(0).mean()) / static[bi]


def main():
    results = {}
    for size in ["10x5", "20x10"]:
        pool = P.REPO / "daniel/data/data_train_vali/SD2" / f"{size}+carbon+priority"
        files = sorted(pool.glob("*.fjs"), key=lambda p: int("".join(filter(str.isdigit, p.stem))))
        parsed = [P.parse(f) for f in files][:N_INST]
        print(f"=== {size}: n={len(parsed)}, {N_DRAWS} draws per scheme, alpha={ALPHA} ===", flush=True)
        for scheme in ["iid", "structured"]:
            gains = []
            for d in range(N_DRAWS):
                rng = np.random.default_rng(BASE_SEED + 1000 * d)
                if scheme == "iid":
                    inst = [(a, b, iid_carbon(c, rng)) for a, b, c in parsed]
                else:
                    inst = [(a, b, structured_v2(c, b, rng)) for a, b, c in parsed]
                g = headroom(inst)
                gains.append(g)
                print(f"  {scheme:11s} draw {d}: oracle gain = {g:5.2f}%", flush=True)
            results[(size, scheme)] = np.array(gains)
            print(f"  {scheme:11s} MEAN = {np.mean(gains):5.2f}%  sd = {np.std(gains, ddof=1):.2f}", flush=True)
        print(flush=True)

    print("=" * 72)
    print("SCALING CONTRAST (the pre-registered claim)")
    print("=" * 72)
    for scheme in ["iid", "structured"]:
        a = results[("10x5", scheme)]; b = results[("20x10", scheme)]
        ratio = b.mean() / a.mean()
        print(f"  {scheme:11s} 10x5 = {a.mean():5.2f}% (sd {a.std(ddof=1):.2f})   "
              f"20x10 = {b.mean():5.2f}% (sd {b.std(ddof=1):.2f})   "
              f"20x10/10x5 = {ratio:.2f}")
    ai, bi_ = results[("10x5", "iid")], results[("20x10", "iid")]
    as_, bs = results[("10x5", "structured")], results[("20x10", "structured")]
    print()
    print(f"  i.i.d.     retains {100*bi_.mean()/ai.mean():5.1f}% of its 10x5 headroom at 20x10")
    print(f"  structured retains {100*bs.mean()/as_.mean():5.1f}% of its 10x5 headroom at 20x10")
    print()
    print(f"  20x10 head-to-head: structured {bs.mean():.2f}% vs i.i.d. {bi_.mean():.2f}% "
          f"-> ratio {bs.mean()/bi_.mean():.2f}x")
    print(f"  separation vs noise: difference = {bs.mean()-bi_.mean():.2f} pp, "
          f"pooled sd = {np.sqrt((bs.var(ddof=1)+bi_.var(ddof=1))/2):.2f} pp")
    PRINT_FLUSH = True


if __name__ == "__main__":
    main()
