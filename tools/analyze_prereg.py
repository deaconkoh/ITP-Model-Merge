"""Track A: the pre-registered analysis, exactly as specified in preregistration_h3_static.md.

Target  : per-instance lambda-response under the fixed preference w_carbon=0.5,
          seed-averaged across the 4 specialist seeds in OUTPUT space.
Primary : carb_pt_corr_global, positive sign, both sizes.
Secondary family (Holm-Bonferroni within family, m=2):
          carb_pt_corr_withinop_mean (+), clean_is_slow_frac (-).
Everything else is exploratory and labelled as such.

Grid conventions (fixed here to remove a researcher degree of freedom):
  * response slope  -> 3-point MERGE grid {0.3,0.5,0.7}, matching the definition used for
    the earlier carbon-only analysis so the "which numbers transfer" comparison is
    apples-to-apples.
  * oracle / selection -> 5-point grid {0,0.3,0.5,0.7,1.0}, matching prereg section 1.

Run from the repository root.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel"))

SEEDS = [111, 222, 333, 444]
W_CARBON = 0.5
MERGE_GRID = [0.3, 0.5, 0.7]
FULL_GRID = [0.0, 0.3, 0.5, 0.7, 1.0]
PRIMARY = "carb_pt_corr_global"
SECONDARY = {"carb_pt_corr_withinop_mean": +1, "clean_is_slow_frac": -1}

TAG = {0.0: "m_s{s}", 0.3: "soup_c30_s{s}", 0.5: "soup_c50_s{s}", 0.7: "soup_c70_s{s}", 1.0: "c_s{s}"}


# ----------------------------------------------------------------- features
def parse_instance(path):
    lines = Path(path).read_text().splitlines()
    h = lines[0].split()
    n_jobs, n_mch = int(h[0]), int(h[1])
    ops, ops_per_job = [], []
    for line in lines[1:]:
        t = line.strip().split()
        if not t:
            continue
        ops_per_job.append(int(t[0]))
        i = 1
        while i < len(t):
            mc = int(t[i]); prio = int(t[i + 1]); i += 2
            opts = []
            for _ in range(mc):
                opts.append((int(t[i]), int(t[i + 1]), int(t[i + 2]))); i += 3
            ops.append({"priority": prio, "options": opts})
    return n_jobs, n_mch, ops, ops_per_job


def skew(x):
    x = np.asarray(x, float); s = x.std()
    return 0.0 if s == 0 else float(np.mean(((x - x.mean()) / s) ** 3))


def features_for(path):
    n_jobs, n_mch, ops, opj = parse_instance(path)
    prio = np.array([o["priority"] for o in ops], float)
    pts, cbs, flex, per_op, clean_slow = [], [], [], [], []
    for o in ops:
        a = np.array(o["options"], float)
        pt, cb = a[:, 1], a[:, 2]
        pts += pt.tolist(); cbs += cb.tolist(); flex.append(len(a))
        if len(a) >= 3 and pt.std() > 0 and cb.std() > 0:
            per_op.append(float(np.corrcoef(pt, cb)[0, 1]))
        if len(a) >= 2:
            clean_slow.append(1.0 if pt[int(np.argmin(cb))] >= np.median(pt) else 0.0)
    pts = np.array(pts); cbs = np.array(cbs); flex = np.array(flex, float); opj = np.array(opj, float)
    return {
        "prio_mean": prio.mean(), "prio_var": prio.var(ddof=1), "prio_skew": skew(prio),
        "prio_spread": prio.max() - prio.min(),
        "carbon_mean": cbs.mean(), "carbon_var": cbs.var(ddof=1), "carbon_spread": cbs.max() - cbs.min(),
        "carb_pt_corr_global": float(np.corrcoef(pts, cbs)[0, 1]),
        "carb_pt_corr_withinop_mean": float(np.mean(per_op)) if per_op else np.nan,
        "carb_pt_corr_withinop_std": float(np.std(per_op, ddof=1)) if len(per_op) > 1 else np.nan,
        "clean_is_slow_frac": float(np.mean(clean_slow)) if clean_slow else np.nan,
        "flex_mean": flex.mean(), "flex_var": flex.var(ddof=1),
        "pt_mean": pts.mean(), "pt_var": pts.var(ddof=1), "pt_spread": pts.max() - pts.min(),
        "n_jobs": float(n_jobs), "n_machines": float(n_mch), "n_ops": float(len(ops)),
        "ops_per_job_mean": opj.mean(), "ops_per_job_var": opj.var(ddof=1) if len(opj) > 1 else 0.0,
    }


# ----------------------------------------------------------------- data loading
def load_metrics(size):
    """returns dict lam -> (n_seeds, n_inst, 2) array of [makespan, carbon]"""
    d = REPO / "daniel/test_results/SD2" / f"trainvali_{size}+carbon+priority"
    out = {}
    for lam in FULL_GRID:
        per_seed = []
        for s in SEEDS:
            f = d / f"Result_DANIELG+{size}+carbon+priority+{TAG[lam].format(s=s)}_trainvali_{size}+carbon+priority.npy"
            per_seed.append(np.load(f)[:, :2])
        out[lam] = np.stack(per_seed)
    return out


def scalarized(mets):
    """obj(lam) = w*carbon/carbon(0.5) + (1-w)*makespan/makespan(0.5); shape (n_seed,n_inst) per lam"""
    ref_m = mets[0.5][:, :, 0]; ref_c = mets[0.5][:, :, 1]
    return {lam: W_CARBON * (v[:, :, 1] / ref_c) + (1 - W_CARBON) * (v[:, :, 0] / ref_m)
            for lam, v in mets.items()}


# ----------------------------------------------------------------- stats helpers
def fisher_ci(r, n, kind="pearson", alpha=0.05):
    if not np.isfinite(r) or n < 5 or abs(r) >= 1:
        return (np.nan, np.nan)
    se = (1.0 if kind == "pearson" else 1.06) / np.sqrt(n - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(np.arctanh(r) - z * se)), float(np.tanh(np.arctanh(r) + z * se))


def power_for(r, n, alpha=0.05):
    za = stats.norm.ppf(1 - alpha / 2)
    z = np.arctanh(abs(r)) * np.sqrt(n - 3)
    return float(stats.norm.sf(za - z) + stats.norm.cdf(-za - z))


def anova2(Y):
    n_i, n_s = Y.shape
    gm = Y.mean()
    ie = Y.mean(1) - gm; se_ = Y.mean(0) - gm
    inter = Y - gm - ie[:, None] - se_[None, :]
    SSi = n_s * (ie ** 2).sum(); SSs = n_i * (se_ ** 2).sum(); SSint = (inter ** 2).sum()
    SSt = ((Y - gm) ** 2).sum()
    dfi, dfs = n_i - 1, n_s - 1; dfint = dfi * dfs
    MSi, MSs, MSint = SSi / dfi, SSs / dfs, SSint / dfint
    Fi, Fs = MSi / MSint, MSs / MSint
    return dict(pct_i=100 * SSi / SSt, pct_s=100 * SSs / SSt, pct_int=100 * SSint / SSt,
                F_i=Fi, p_i=1 - stats.f.cdf(Fi, dfi, dfint),
                F_s=Fs, p_s=1 - stats.f.cdf(Fs, dfs, dfint),
                dfi=dfi, dfs=dfs, dfint=dfint, reliability=1 - 1 / Fi)


def loo_linear(x, y):
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        sl, ic = np.polyfit(x[m], y[m], 1)
        pred[i] = sl * x[i] + ic
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), pred


def loo_nested(X, y):
    pred = np.empty(len(y)); chosen = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        best, br = None, -1
        for c in X.columns:
            v = X[c].values.astype(float)
            r = abs(np.corrcoef(v[m], y[m])[0, 1])
            if np.isfinite(r) and r > br:
                br, best = r, c
        chosen.append(best)
        v = X[best].values.astype(float)
        sl, ic = np.polyfit(v[m], y[m], 1)
        pred[i] = sl * v[i] + ic
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), pd.Series(chosen).value_counts()


# ----------------------------------------------------------------- main
def run(size):
    print("=" * 84)
    print(f"SIZE {size}")
    print("=" * 84)
    pool = REPO / "daniel/data/data_train_vali/SD2" / f"{size}+carbon+priority"
    files = sorted(pool.glob("*.fjs"), key=lambda p: int("".join(filter(str.isdigit, p.stem))))
    feats = pd.DataFrame([features_for(f) for f in files], index=[f.stem for f in files])
    n = len(feats)

    mets = load_metrics(size)
    obj = scalarized(mets)

    # ---- target: response slope on the 3-point merge grid, % of own lambda=0.5 baseline
    o30, o50, o70 = obj[0.3], obj[0.5], obj[0.7]          # each (n_seed, n_inst); o50 == 1
    pct30 = (o30 - o50) / o50 * 100
    pct70 = (o70 - o50) / o50 * 100
    slope = (pct70 - pct30) / 0.4                          # (n_seed, n_inst)
    target = slope.mean(0)                                 # seed-averaged, output space

    print(f"\n[TARGET] n={n} instances, {len(SEEDS)} seeds, preference w_carbon={W_CARBON}")
    print(f"  seed-avg response slope: mean={target.mean():.2f} sd={target.std(ddof=1):.2f}")
    for i, s in enumerate(SEEDS):
        print(f"    seed {s}: mean={slope[i].mean():7.2f} sd={slope[i].std(ddof=1):6.2f}")

    # ---- variance decomposition against the NEW target
    Y = slope.T                                            # (n_inst, n_seed)
    a = anova2(Y)
    print(f"\n[VARIANCE DECOMPOSITION vs preference target]")
    print(f"  instance   {a['pct_i']:5.1f}%  F({a['dfi']},{a['dfint']})={a['F_i']:6.2f}  p={a['p_i']:.4g}")
    print(f"  seed       {a['pct_s']:5.1f}%  F({a['dfs']},{a['dfint']})={a['F_s']:6.2f}  p={a['p_s']:.4g}")
    print(f"  interaction{a['pct_int']:5.1f}%")
    print(f"  target reliability (1-1/F) = {a['reliability']:.3f}")

    # ---- direction vs magnitude stability
    sgn = np.sign(slope)
    sign_agree = (sgn == sgn[0]).all(0).mean()
    argmin_full = np.stack([np.argmin(np.stack([obj[l][i] for l in FULL_GRID]), 0) for i in range(len(SEEDS))])
    argmax_agree = (argmin_full == argmin_full[0]).all(0).mean()
    print(f"\n[DIRECTION vs MAGNITUDE STABILITY]")
    print(f"  sign of slope agrees across all seeds : {sign_agree:.1%}")
    print(f"  argmin lambda agrees across all seeds : {argmax_agree:.1%}  (5-point grid)")
    marg = pd.Series(argmin_full.ravel()).value_counts(normalize=True).sort_index()
    print(f"  marginal argmin distribution: " + ", ".join(f"lam={FULL_GRID[k]}:{v:.1%}" for k, v in marg.items()))

    # ---- oracle headroom on this pool
    O = np.stack([obj[l] for l in FULL_GRID])              # (5, n_seed, n_inst)
    gains = []
    for si in range(len(SEEDS)):
        st = O[:, si, :].mean(1); bi = int(np.argmin(st))
        gains.append(100 * (st[bi] - O[:, si, :].min(0).mean()) / st[bi])
    print(f"\n[ORACLE HEADROOM on this pool] mean={np.mean(gains):.2f}% sd={np.std(gains, ddof=1):.2f}%")

    # ---- PRIMARY
    x = feats[PRIMARY].values.astype(float); y = target
    r, p = stats.pearsonr(x, y); rho, prho = stats.spearmanr(x, y)
    lo, hi = fisher_ci(r, n)
    print(f"\n[PRIMARY] {PRIMARY} (pre-registered, positive sign expected, uncorrected)")
    print(f"  Pearson  r={r:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  p={p:.4g}   R^2={r**2:.3f}")
    print(f"  Spearman r={rho:+.3f}  p={prho:.4g}")
    print(f"  achieved power at n={n}: for |r|=0.3 -> {power_for(0.3,n):.1%}, |r|=0.4 -> {power_for(0.4,n):.1%}")
    print(f"  per-seed breakdown:")
    for i, s in enumerate(SEEDS):
        rs, ps = stats.pearsonr(x, slope[i])
        print(f"    seed {s}: r={rs:+.3f} p={ps:.4g}")
    r2_loo, _ = loo_linear(x, y)
    print(f"  LOO R^2 (pre-specified feature) = {r2_loo:+.3f}")

    # ---- SECONDARY family, Holm-Bonferroni m=2
    print(f"\n[SECONDARY FAMILY] Holm-Bonferroni within family (m=2)")
    rows = []
    for f_, expected in SECONDARY.items():
        v = feats[f_].values.astype(float)
        rr, pp = stats.pearsonr(v, y)
        rows.append((f_, rr, pp, expected, np.sign(rr) == expected))
    rows.sort(key=lambda t: t[2])
    m = len(rows)
    for j, (f_, rr, pp, exp, ok) in enumerate(rows):
        adj = min(1.0, pp * (m - j))
        lo2, hi2 = fisher_ci(rr, n)
        print(f"  {f_:28s} r={rr:+.3f} CI[{lo2:+.3f},{hi2:+.3f}] p={pp:.4g} p_holm={adj:.4g} "
              f"sign_as_predicted={ok}")

    # ---- exploratory
    drop = [c for c in feats.columns if feats[c].std(ddof=1) == 0 or
            (feats[c].mean() != 0 and abs(feats[c].std(ddof=1) / feats[c].mean()) < 0.01)]
    Xe = feats.drop(columns=drop)
    print(f"\n[EXPLORATORY — not pre-registered, needs independent replication]")
    print(f"  dropped as constant/near-constant: {drop}")
    ex = []
    for c in Xe.columns:
        v = Xe[c].values.astype(float)
        rr, pp = stats.pearsonr(v, y)
        ex.append((c, rr, pp))
    ex.sort(key=lambda t: -abs(t[1]))
    for c, rr, pp in ex[:6]:
        print(f"  {c:28s} r={rr:+.3f} p={pp:.4g}")
    r2n, counts = loo_nested(Xe, y)
    print(f"  nested LOO R^2 (selection inside folds) = {r2n:+.3f}; picks={dict(counts)}")

    # ---- CRITERION 3: realized benefit of a feature-driven lambda rule (LOO)
    print(f"\n[CRITERION 3] feature-driven lambda selection, leave-one-out, vs oracle")
    rec = []
    for si in range(len(SEEDS)):
        objs = O[:, si, :]                                  # (5, n_inst)
        best_lam_idx = objs.argmin(0)
        static_idx = int(np.argmin(objs.mean(1)))
        static_val = objs[static_idx].mean()
        oracle_val = objs.min(0).mean()
        pred_vals = np.empty(n)
        for i in range(n):
            msk = np.ones(n, bool); msk[i] = False
            sl_, ic_ = np.polyfit(x[msk], best_lam_idx[msk].astype(float), 1)
            k = int(np.clip(round(sl_ * x[i] + ic_), 0, len(FULL_GRID) - 1))
            pred_vals[i] = objs[k, i]
        rule_val = pred_vals.mean()
        denom = static_val - oracle_val
        recovered = 100 * (static_val - rule_val) / denom if denom > 0 else np.nan
        rec.append(recovered)
        print(f"  seed {SEEDS[si]}: static={static_val:.4f} rule={rule_val:.4f} oracle={oracle_val:.4f} "
              f"| oracle gain={100*denom/static_val:.2f}% | recovered={recovered:6.1f}%")
    print(f"  --> mean oracle gain recovered by the feature rule: {np.nanmean(rec):.1f}%  "
          f"(criterion 3 threshold: >= 50%)")

    return dict(size=size, n=n, r=r, p=p, ci=(lo, hi), loo=r2_loo, anova=a,
                recovered=float(np.nanmean(rec)), oracle=float(np.mean(gains)),
                sign_agree=sign_agree, argmax_agree=argmax_agree)


if __name__ == "__main__":
    out = {s: run(s) for s in ["10x5", "20x10"]}
    print("\n" + "=" * 84)
    print("VERDICT AGAINST PRE-REGISTERED DECISION RULE")
    print("=" * 84)
    for s, o in out.items():
        c1 = (o["r"] > 0) and (o["p"] < 0.05)
        c2 = o["loo"] >= 0.10
        c3 = o["recovered"] >= 50.0
        print(f"{s}: C1 sign+sig={'PASS' if c1 else 'FAIL'} (r={o['r']:+.3f}, p={o['p']:.4g}) | "
              f"C2 LOO R^2>=0.10={'PASS' if c2 else 'FAIL'} ({o['loo']:+.3f}) | "
              f"C3 >=50% oracle={'PASS' if c3 else 'FAIL'} ({o['recovered']:.1f}%)")
    allpass = all((o["r"] > 0 and o["p"] < 0.05 and o["loo"] >= 0.10 and o["recovered"] >= 50.0)
                  for o in out.values())
    print(f"\nSTATIC-FEATURE ROUTE: {'SUPPORTED' if allpass else 'NOT SUPPORTED'}")
