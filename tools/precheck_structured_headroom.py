"""Track B pre-check: does structured carbon annotation create more lambda-selection
headroom than i.i.d. annotation? Training-free, CPU only, no checkpoints touched.

Method: a lambda-parameterised greedy dispatcher schedules each instance, choosing at each
step the ready (operation, machine) pair minimising

    lambda * carbon/carbon_mean + (1 - lambda) * pt/pt_mean

Routing and processing times are IDENTICAL between the two schemes; only the carbon
annotation differs. We then compute, per scheme, the oracle gain of per-instance lambda
selection over the single best static lambda under the same scalarized preference used
elsewhere (w_carbon = 0.5, referenced to each instance's own lambda=0.5 result).

LIMITATIONS (stated up front):
  * a heuristic dispatcher's lambda-response is not identical to a trained policy's, so
    this is a screening indicator, not a guarantee;
  * it exercises only the carbon/machine-character part of the proposed spec, not the
    job-level priority clustering (priority does not enter this two-objective preference).
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
W_CARBON = 0.5
LAMBDAS = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
ALPHA_MAX = 0.9          # max machine power spread; modelling choice
NOISE_SD = 0.05          # multiplicative noise on carbon; modelling choice
RNG_SEED = 20260910


def parse(path):
    lines = Path(path).read_text().splitlines()
    h = lines[0].split()
    n_jobs, n_mch = int(h[0]), int(h[1])
    jobs = []
    for line in lines[1:]:
        t = line.strip().split()
        if not t:
            continue
        i, ops = 1, []
        while i < len(t):
            mc = int(t[i]); i += 2          # skip machine_count and priority
            opts = []
            for _ in range(mc):
                opts.append((int(t[i]) - 1, float(t[i + 1]), float(t[i + 2]))); i += 3
            ops.append(opts)
        jobs.append(ops)
    return n_jobs, n_mch, jobs


def structured_carbon(jobs, n_mch, rng):
    """Per B2: instance latent kappa -> machine power P_m anti-correlated with machine speed;
    carbon(o,m) = P_m * pt(o,m) * (1+eps)."""
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
    P = 1.0 + kappa * ALPHA_MAX * z          # fast machine (low tbar) -> high power
    P = np.clip(P, 0.1, None)
    out = []
    for ops in jobs:
        job = []
        for opts in ops:
            job.append([(m, pt, max(1e-6, P[m] * pt * (1 + rng.normal(0, NOISE_SD))))
                        for m, pt, _ in opts])
        out.append(job)
    return out


def dispatch(jobs, n_mch, lam, pt_mean, cb_mean):
    n_jobs = len(jobs)
    nxt = [0] * n_jobs
    job_ready = [0.0] * n_jobs
    mach_free = [0.0] * n_mch
    total_ops = sum(len(o) for o in jobs)
    carbon = 0.0
    for _ in range(total_ops):
        best = None
        for j in range(n_jobs):
            if nxt[j] >= len(jobs[j]):
                continue
            for m, pt, cb in jobs[j][nxt[j]]:
                score = lam * (cb / cb_mean) + (1 - lam) * (pt / pt_mean)
                start = job_ready[j] if job_ready[j] > mach_free[m] else mach_free[m]
                key = (score, start, j, m)
                if best is None or key < best[0]:
                    best = (key, j, m, pt, cb, start)
        _, j, m, pt, cb, start = best
        end = start + pt
        job_ready[j] = end; mach_free[m] = end; nxt[j] += 1; carbon += cb
    return max(job_ready), carbon


def evaluate(instances, label):
    n = len(instances)
    MK = np.zeros((len(LAMBDAS), n)); CB = np.zeros((len(LAMBDAS), n))
    for idx, (n_jobs, n_mch, jobs) in enumerate(instances):
        pts = [pt for ops in jobs for opts in ops for _, pt, _ in opts]
        cbs = [cb for ops in jobs for opts in ops for _, _, cb in opts]
        pm, cm = float(np.mean(pts)), float(np.mean(cbs))
        for li, lam in enumerate(LAMBDAS):
            mk, cb = dispatch(jobs, n_mch, lam, pm, cm)
            MK[li, idx] = mk; CB[li, idx] = cb
    half = LAMBDAS.index(0.5)
    obj = W_CARBON * (CB / CB[half]) + (1 - W_CARBON) * (MK / MK[half])
    static = obj.mean(1); bi = int(np.argmin(static))
    oracle = obj.min(0).mean()
    gain = 100 * (static[bi] - oracle) / static[bi]
    argmins = obj.argmin(0)
    distinct = len(set(argmins.tolist()))
    spread = np.std([LAMBDAS[k] for k in argmins])
    print(f"  {label:26s} best_static_lambda={LAMBDAS[bi]:.1f}  ORACLE GAIN={gain:5.2f}%  "
          f"distinct_optima={distinct:2d}  sd(optimal_lambda)={spread:.3f}")
    return gain, distinct, spread


def main():
    rng = np.random.default_rng(RNG_SEED)
    for size in ["10x5", "20x10"]:
        pool = REPO / "daniel/data/data_train_vali/SD2" / f"{size}+carbon+priority"
        files = sorted(pool.glob("*.fjs"), key=lambda p: int("".join(filter(str.isdigit, p.stem))))
        parsed = [parse(f) for f in files]
        print(f"=== {size}: {len(parsed)} instances, lambda grid {LAMBDAS} ===")
        iid = [(a, b, c) for a, b, c in parsed]
        struct = [(a, b, structured_carbon(c, b, rng)) for a, b, c in parsed]
        g1, *_ = evaluate(iid, "i.i.d. (current)")
        g2, *_ = evaluate(struct, "structured (proposed)")
        print(f"  --> headroom ratio structured/iid = {g2/g1 if g1 > 0 else float('inf'):.2f}x\n")


if __name__ == "__main__":
    main()
