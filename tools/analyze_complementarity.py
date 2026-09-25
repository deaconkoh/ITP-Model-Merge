"""RQ1 complementarity pilot -- analysis of the rollouts written by tools/complementarity.py.

Fixed before looking at results:
  * Normalisation: per instance, each objective is min-max normalised by the ideal (min) and nadir (max)
    observed across BOTH specialists and ALL 4 seeds on that instance (separately for the evaluation
    pool and the validation split). Score = w_m * norm_makespan + w_c * norm_carbon, lower is better.
  * Hypervolume in normalised space, reference point (1.1, 1.1), computed on pool-mean points.
  * "Best baseline" at preference w: the best Step 1 method run AT that preference (alpha = w; weight
    merge only where lambda = w exists). A stricter variant allowing any alpha is reported alongside.
  * Oracle: per instance, the best of the 32 segment assignments at preference w.
  * Verdict. Meaningful headroom: oracle beats the best baseline by >= 2% score on average over
    w_c in [0.3, 0.7] at >= 3 of 4 seeds. Learnable structure: the transferred fixed assignment keeps a
    clearly positive share of that headroom -- operationalised as a positive mean share at >= 3 of 4
    held-out seeds AND a mean share across held-out seeds >= 20%.
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results/complementarity/raw"
OUT = REPO / "results/complementarity"
SEEDS = [111, 222, 333, 444]
WCS = [round(0.05 * i, 2) for i in range(21)]
MID = [w for w in WCS if 0.3 - 1e-9 <= w <= 0.7 + 1e-9]
BITS = ["".join(b) for b in itertools.product("MC", repeat=5)]
REF = (1.1, 1.1)
R = {s: dict(np.load(RAW / f"seed{s}.npz")) for s in SEEDS}


def norm_bounds(prefix=""):
    keys = [f"{prefix}assign_MMMMM", f"{prefix}assign_CCCCC"] if prefix else ["spec_m", "spec_c"]
    X = np.stack([R[s][k] for s in SEEDS for k in keys])          # [8, 2, n]
    lo, hi = X.min(0), X.max(0)
    return lo, np.where(hi - lo > 0, hi - lo, 1.0)


LO, RG = norm_bounds()
LOV, RGV = norm_bounds("vali_")


def normed(s, key, vali=False):
    lo, rg = (LOV, RGV) if vali else (LO, RG)
    return (R[s][key] - lo) / rg                                  # [2, n]


def inst_score(s, key, w, vali=False):
    n = normed(s, key, vali)
    return (1 - w) * n[0] + w * n[1]


def stoch_inst(s, w):
    return np.mean([inst_score(s, f"stoch_{w:.2f}_r{r}", w) for r in range(5)], 0)


def baselines_at(s, w):
    """Matched baselines at preference w: name -> per-instance score."""
    b = {"makespan spec": inst_score(s, "spec_m", w), "carbon spec": inst_score(s, "spec_c", w),
         "prob mixture": inst_score(s, f"prob_{w:.2f}", w), "logit mixture": inst_score(s, f"logit_{w:.2f}", w),
         "stochastic selector": stoch_inst(s, w)}
    c = int(round(w * 1000))
    if c % 100 == 0 and f"merge_{c:04d}" in R[s]:
        b["weight merge"] = inst_score(s, f"merge_{c:04d}", w)
    return b


def any_alpha_best(s, w):
    keys = [k for k in R[s] if not k.startswith(("assign_", "vali_", "stoch_"))]
    vals = [inst_score(s, k, w).mean() for k in keys]
    vals += [np.mean([inst_score(s, f"stoch_{a:.2f}_r{r}", w) for r in range(5)], 0).mean() for a in WCS]
    return min(vals)


def hv(points):
    P = np.array([p for p in points if p[0] < REF[0] and p[1] < REF[1]])
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    front, best_y = [], np.inf
    for x, y in P:
        if y < best_y:
            front.append((x, y)); best_y = y
    area, prev_y = 0.0, REF[1]
    for x, y in front:
        area += (REF[0] - x) * (prev_y - y); prev_y = y
    return area


def pt(s, key):
    return tuple(normed(s, key).mean(1))


def stoch_pt(s, w):
    return tuple(np.mean([normed(s, f"stoch_{w:.2f}_r{r}") for r in range(5)], 0).mean(1))


def main():
    rep, res = [], {}
    P = lambda *a: rep.append(" ".join(str(x) for x in a))
    # ------------------------------------------------------------------ Step 1: scores and fronts
    methods = ["makespan spec", "carbon spec", "prob mixture", "logit mixture", "stochastic selector", "weight merge"]
    rows = {m: {w: [] for w in WCS} for m in methods + ["best baseline", "oracle", "best fixed assignment"]}
    for s in SEEDS:
        for w in WCS:
            b = baselines_at(s, w)
            for m, v in b.items():
                rows[m][w].append(v.mean())
            rows["best baseline"][w].append(min(v.mean() for v in b.values()))
            A = np.stack([inst_score(s, f"assign_{x}", w) for x in BITS])     # [32, n]
            rows["oracle"][w].append(A.min(0).mean())
            rows["best fixed assignment"][w].append(A.mean(1).min())
    res["scores"] = {m: {f"{w:.2f}": [float(x) for x in v] for w, v in d.items()} for m, d in rows.items()}

    fronts = {}
    for s in SEEDS:
        F = {"specialists": [pt(s, "spec_m"), pt(s, "spec_c")],
             "prob mixture": [pt(s, f"prob_{w:.2f}") for w in WCS],
             "logit mixture": [pt(s, f"logit_{w:.2f}") for w in WCS],
             "stochastic selector": [stoch_pt(s, w) for w in WCS],
             "weight merge": [pt(s, k) for k in sorted(R[s]) if k.startswith("merge_")],
             "32 fixed assignments": [pt(s, f"assign_{x}") for x in BITS]}
        F["all baselines"] = sum((F[k] for k in ("specialists", "prob mixture", "logit mixture",
                                                  "stochastic selector", "weight merge")), [])
        orc = []
        for w in WCS:
            A = np.stack([inst_score(s, f"assign_{x}", w) for x in BITS])
            win = A.argmin(0)
            N = np.stack([normed(s, f"assign_{x}") for x in BITS])           # [32, 2, n]
            orc.append(tuple(N[win, :, np.arange(N.shape[2])].mean(0)))
        F["per-instance oracle"] = orc
        fronts[s] = F
    res["hv"] = {str(s): {k: hv(v) for k, v in F.items()} for s, F in fronts.items()}

    # ------------------------------------------------------------------ Step 3a/b: headroom
    head = {s: {} for s in SEEDS}
    head_any = {s: {} for s in SEEDS}
    for i, s in enumerate(SEEDS):
        for w in WCS:
            B, O = rows["best baseline"][w][i], rows["oracle"][w][i]
            head[s][w] = 100 * (B - O) / B if B > 0 else np.nan
            Ba = any_alpha_best(s, w)
            head_any[s][w] = 100 * (Ba - O) / Ba if Ba > 0 else np.nan
    mid_head = {s: float(np.mean([head[s][w] for w in MID])) for s in SEEDS}
    mid_head_any = {s: float(np.mean([head_any[s][w] for w in MID])) for s in SEEDS}
    res["headroom_mid"] = mid_head; res["headroom_mid_any_alpha"] = mid_head_any

    # fixed-assignment (in-sample, pool level) vs best baseline -- how much needs per-instance choice?
    fixed_head = {s: float(np.mean([100 * (rows["best baseline"][w][i] - rows["best fixed assignment"][w][i])
                                    / rows["best baseline"][w][i] for w in MID])) for i, s in enumerate(SEEDS)}

    # ------------------------------------------------------------------ Step 3c: structure
    struct = {}
    for w in (0.3, 0.5, 0.7):
        cnt, ties, n_tot = {}, [], 0
        best_fixed = {}
        for s in SEEDS:
            A = np.stack([inst_score(s, f"assign_{x}", w) for x in BITS])
            mn = A.min(0)
            win = np.isclose(A, mn[None, :], atol=1e-12)
            ties.append(win.sum(0).mean())
            for x_i, x in enumerate(BITS):
                cnt[x] = cnt.get(x, 0) + int(win[x_i].sum())
            n_tot += A.shape[1]
            best_fixed[s] = BITS[int(A.mean(1).argmin())]
        top = sorted(cnt.items(), key=lambda kv: -kv[1])[:5]
        struct[w] = {"top": [(x, c / n_tot) for x, c in top], "mean_tied_winners": float(np.mean(ties)),
                     "best_fixed_per_seed": best_fixed}

    # ------------------------------------------------------------------ Step 3d: transfer
    transfer = {}
    for h in SEEDS:
        tr = [s for s in SEEDS if s != h]
        shares, shares_fixed, gains, picks = [], [], [], {}
        hi = SEEDS.index(h)
        for w in WCS:
            vali = np.stack([np.mean([inst_score(s, f"vali_assign_{x}", w, vali=True).mean() for s in tr])
                             for x in BITS])
            x_star = BITS[int(vali.argmin())]
            picks[f"{w:.2f}"] = x_star
            T = inst_score(h, f"assign_{x_star}", w).mean()
            B, O, F_ = rows["best baseline"][w][hi], rows["oracle"][w][hi], rows["best fixed assignment"][w][hi]
            if w in MID:
                shares.append((B - T) / (B - O) if B - O > 1e-12 else np.nan)
                shares_fixed.append((B - T) / (B - F_) if B - F_ > 1e-12 else np.nan)
                gains.append(100 * (B - T) / B)
        transfer[h] = {"share_of_oracle_headroom": float(np.nanmean(shares)),
                       "share_of_fixed_headroom": (float(np.nanmean(shares_fixed)) if np.any(np.isfinite(shares_fixed))
                                            else None),   # None: no fixed assignment beat the best baseline
                       "gain_vs_best_baseline_pct": float(np.mean(gains)), "picks": picks}

    # ------------------------------------------------------------------ Step 3e: state dependence
    sd = {}
    for i, s in enumerate(SEEDS):
        pure_or_mixed, beats_all = [], []
        for w in MID:
            A = np.stack([inst_score(s, f"assign_{x}", w) for x in BITS])
            win = A.argmin(0)
            mixed = np.array([BITS[k] not in ("MMMMM", "CCCCC") for k in win])
            b = baselines_at(s, w)
            bb = np.min(np.stack(list(b.values())), 0)                   # per-instance best matched baseline
            pure_or_mixed.append(mixed.mean())
            beats_all.append(((A.min(0) < bb - 1e-12) & mixed).mean())
        # best MIXED fixed assignment vs every fixed-alpha combination at ANY alpha and the selector
        diffs = []
        for w in MID:
            mixed_best = min(inst_score(s, f"assign_{x}", w).mean() for x in BITS if x not in ("MMMMM", "CCCCC"))
            diffs.append(100 * (any_alpha_best(s, w) - mixed_best) / any_alpha_best(s, w))
        sd[s] = {"instances_whose_best_is_a_switch": float(np.mean(pure_or_mixed)),
                 "instances_where_a_switch_beats_every_matched_baseline": float(np.mean(beats_all)),
                 "best_fixed_switch_vs_best_any_alpha_baseline_pct": float(np.mean(diffs))}

    # ------------------------------------------------------------------ selection control
    # The oracle chooses the best of 32 rollouts PER INSTANCE. To separate switching from selection, give
    # fixed-alpha combinations the same per-instance freedom: per instance, the best probability or logit
    # mixture over ALL 21 alphas (42 candidates, scored at w), and the best of the matched baselines
    # including each stochastic repeat separately.
    ctrl = {}
    for i, s in enumerate(SEEDS):
        vs_alpha, vs_base, frac_better = [], [], []
        for w in MID:
            A = np.stack([inst_score(s, f"assign_{x}", w) for x in BITS]).min(0)
            fa = np.stack([inst_score(s, f"{m}_{a:.2f}", w) for m in ("prob", "logit") for a in WCS]).min(0)
            b = baselines_at(s, w); b.pop("stochastic selector")
            cands = list(b.values()) + [inst_score(s, f"stoch_{w:.2f}_r{r}", w) for r in range(5)]
            bo = np.stack(cands).min(0)
            vs_alpha.append(100 * (fa.mean() - A.mean()) / fa.mean())
            vs_base.append(100 * (bo.mean() - A.mean()) / bo.mean())
            frac_better.append(float((A < fa - 1e-12).mean()))
        ctrl[s] = {"oracle_vs_per_instance_best_fixed_alpha_pct": float(np.mean(vs_alpha)),
                   "oracle_vs_per_instance_best_matched_baseline_pct": float(np.mean(vs_base)),
                   "instances_where_best_switch_beats_best_fixed_alpha": float(np.mean(frac_better))}

    # ------------------------------------------------------------------ Step 2: diagnostics
    diag = {}
    for src in ("spec_m", "spec_c", "prob_0.50"):
        per = {k: [] for k in ("differ", "jsd", "H_m", "H_c", "top3", "n_valid")}
        for s in SEEDS:
            D = np.load(RAW / f"diag_seed{s}.npz")
            for k in per:
                per[k].append(D[f"{src}__{k}"])                          # [n, 50]
        diag[src] = {k: np.concatenate(v, 0) for k, v in per.items()}

    # ------------------------------------------------------------------ plots
    for s in SEEDS:
        F = fronts[s]
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        style = {"prob mixture": ("tab:blue", "o-"), "logit mixture": ("tab:orange", "s-"),
                 "stochastic selector": ("tab:purple", "^-"), "weight merge": ("tab:brown", "D-")}
        for k, (col, mk) in style.items():
            Q = np.array(F[k]); ax.plot(Q[:, 0], Q[:, 1], mk, color=col, ms=3, lw=1, label=k)
        Q = np.array(F["32 fixed assignments"]); ax.scatter(Q[:, 0], Q[:, 1], s=14, c="tab:green",
                                                            label="32 fixed segment assignments", zorder=3)
        Q = np.array(F["per-instance oracle"]); ax.plot(Q[:, 0], Q[:, 1], "*-", color="tab:red", ms=5, lw=1,
                                                        label="per-instance oracle", zorder=4)
        Q = np.array(F["specialists"]); ax.scatter(Q[:, 0], Q[:, 1], s=90, marker="X", c="k",
                                                   label="specialists (M, C)", zorder=5)
        ax.set_xlabel("normalised makespan (pool mean)"); ax.set_ylabel("normalised carbon (pool mean)")
        ax.set_title(f"10x5, seed {s}: fronts in normalised objective space")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / f"pareto_seed{s}.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------------------ write
    json.dump({"scores": res["scores"], "hv": res["hv"], "headroom_mid": mid_head,
               "headroom_mid_any_alpha": mid_head_any, "fixed_assignment_headroom_mid": fixed_head,
               "structure": {str(k): v for k, v in struct.items()}, "transfer": {str(k): v for k, v in transfer.items()},
               "state_dependence": {str(k): v for k, v in sd.items()},
               "selection_control": {str(k): v for k, v in ctrl.items()}},
              open(OUT / "complementarity_summary.json", "w"), indent=1)
    np.savez_compressed(OUT / "diagnostics_pooled.npz",
                        **{f"{src}__{k}": v for src, d in diag.items() for k, v in d.items()})
    # human-readable dump
    print("MID-PREFERENCE HEADROOM (oracle vs best matched baseline, mean over w_c 0.3-0.7), per seed:",
          {s: round(v, 2) for s, v in mid_head.items()})
    print("  vs best ANY-alpha baseline:", {s: round(v, 2) for s, v in mid_head_any.items()})
    print("  best FIXED assignment (in-sample, pool level) vs best matched baseline:",
          {s: round(v, 2) for s, v in fixed_head.items()})
    print("\nSCORES (mean over seeds; lower is better)")
    print("  w_c  " + "".join(f"{m[:16]:>18}" for m in rows))
    for w in WCS:
        print(f"  {w:.2f} " + "".join(f"{np.mean(rows[m][w]) if rows[m][w] else float('nan'):>18.4f}" for m in rows))
    print("\nPER-SEED HEADROOM (%) at each w_c:")
    for s in SEEDS:
        print(f"  {s}: " + " ".join(f"{head[s][w]:+.1f}" for w in WCS))
    print("\nHYPERVOLUME (ref 1.1,1.1)")
    for s in SEEDS:
        print(f"  {s}: " + "  ".join(f"{k}={v:.4f}" for k, v in res['hv'][str(s)].items()))
    print("\nSTRUCTURE")
    for w, v in struct.items():
        print(f"  w_c={w}: top winners {[(x, round(f, 3)) for x, f in v['top']]}  mean tied winners/instance "
              f"{v['mean_tied_winners']:.2f}  best fixed per seed {v['best_fixed_per_seed']}")
    print("\nTRANSFER (held-out seed)")
    for h, v in transfer.items():
        print(f"  held out {h}: share of oracle headroom {v['share_of_oracle_headroom']:+.3f}   share of fixed-"
              f"assignment headroom {('n/a (no fixed assignment beat the baseline)' if v['share_of_fixed_headroom'] is None else format(v['share_of_fixed_headroom'], '+.3f'))}   gain vs best baseline "
              f"{v['gain_vs_best_baseline_pct']:+.2f}%   picks@0.3/0.5/0.7 "
              f"{v['picks']['0.30']}/{v['picks']['0.50']}/{v['picks']['0.70']}")
    print("\nSTATE DEPENDENCE")
    for s, v in sd.items():
        print(f"  {s}: {v}")
    print("\nSELECTION CONTROL (mean over w_c 0.3-0.7; positive = 32-assignment oracle better)")
    for k, v in ctrl.items():
        print(f"  {k}: {v}")
    print("\nDIAGNOSTICS by stage (5 segments of 10 decisions), pooled over seeds and instances")
    for src, d in diag.items():
        print(f"  on states visited by {src}:")
        for k in ("differ", "jsd", "H_m", "H_c", "top3", "n_valid"):
            segs = [np.nanmean(d[k][:, i * 10:(i + 1) * 10]) for i in range(5)]
            print(f"    {k:<8} overall {np.nanmean(d[k]):.3f}   by stage " + " ".join(f"{x:.3f}" for x in segs))


if __name__ == "__main__":
    main()
