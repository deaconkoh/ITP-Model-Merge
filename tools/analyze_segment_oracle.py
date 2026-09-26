"""Analysis of the soft-weight segment oracle (pre-registered in results/segment_blend_oracle.md).

Reads results/segment_oracle/raw/seed<s>.npz and a6.npz; writes results/segment_oracle/analysis_tables.md,
results/segment_oracle/summary.json and results/segment_oracle/schedule_shape.png. Runs nothing on the GPU.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/segment_oracle"
RAW = OUT / "raw"
PILOT = REPO / "results/complementarity/raw"
SEEDS = [111, 222, 333, 444]
WC = np.array([round(0.05 * i, 2) for i in range(21)])
G21 = np.array([round(0.05 * i, 2) for i in range(21)])
G21_ROWS = [int(round(c * 100)) for c in G21]
MID = (WC >= 0.3 - 1e-9) & (WC <= 0.7 + 1e-9)
K, REF, NBOOT, BSEED = 5, (1.1, 1.1), 10_000, 20260926

split = json.loads((OUT / "split.json").read_text())
FIT, HELD = np.array(split["fit_idx"]), np.array(split["held_idx"])
X = np.stack([np.load(PILOT / f"seed{s}.npz")[k] for s in SEEDS for k in ("spec_m", "spec_c")])
LO, RG = X.min(0), np.where(X.max(0) - X.min(0) > 0, X.max(0) - X.min(0), 1.0)
D = {s: dict(np.load(RAW / f"seed{s}.npz")) for s in SEEDS}
N = D[SEEDS[0]]["FMK"].shape[1]


def norm(inst, mk, cb):
    return (mk - LO[0][inst]) / RG[0][inst], (cb - LO[1][inst]) / RG[1][inst]


def score(inst, w, mk, cb):
    nm, nc = norm(inst, mk, cb)
    return (1 - w) * nm + w * nc


def hv(points):
    P = np.array([p for p in points if p[0] < REF[0] and p[1] < REF[1]])
    if not len(P):
        return 0.0
    P = P[np.argsort(P[:, 0])]
    area, prev = 0.0, REF[1]
    for x, y in P:
        if y < prev:
            area += (REF[0] - x) * (prev - y); prev = y
    return area


# ------------------------------------------------------------------------------------------ arm scores
A = {}
for s in SEEDS:
    d = D[s]
    FS = np.stack([score(np.arange(N), w, d["FMK"], d["FCB"]) for w in WC])      # [w, 101, n]
    a1 = FS[:, G21_ROWS, :].min(1); a2 = FS.min(1); a3 = d["a3_score"]
    a1_c = np.array(G21_ROWS)[FS[:, G21_ROWS, :].argmin(1)]                        # row in G101 of A1 choice
    a1_mk = np.take_along_axis(d["FMK"][None].repeat(len(WC), 0), a1_c[:, None, :], 1)[:, 0]
    a1_cb = np.take_along_axis(d["FCB"][None].repeat(len(WC), 0), a1_c[:, None, :], 1)[:, 0]
    a4_row = np.array(G21_ROWS)[d["a4_idx"]]
    a4 = np.stack([FS[j, a4_row[j], HELD] for j in range(len(WC))])               # [w, held]
    a5 = np.stack([score(HELD, w, d["a5_held_mk"][j], d["a5_held_cb"][j]) for j, w in enumerate(WC)])
    A[s] = dict(a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a1_mk=a1_mk, a1_cb=a1_cb, a4_row=a4_row)

lines, summary = [], {}
L = lines.append

# ------------------------------------------------------------------------------------------ headroom A3 vs A2
a2 = np.stack([A[s]["a2"] for s in SEEDS]); a3 = np.stack([A[s]["a3"] for s in SEEDS]); a1 = np.stack([A[s]["a1"] for s in SEEDS])
ok = a2 >= 0.02
rel = (a2 - a3) / np.where(ok, a2, np.nan)
H_primary = float(np.nanmean(rel)); H_rom = float((a2.mean() - a3.mean()) / a2.mean())
beat05 = float(np.nanmean(rel > 0.005))
gap_share = float((a1.mean() - a2.mean()) / (a1.mean() - a3.mean())) if a1.mean() > a3.mean() else float("nan")
summary.update(H_primary=H_primary, H_ratio_of_means=H_rom, share_beating_A2_by_0p5=beat05,
               excluded_A2_lt_0p02=int((~ok).sum()), total_triples=int(ok.size), A2_closes_share_of_A1_A3_gap=gap_share)
L("### Per-instance headroom: segment oracle A3 vs matched-budget fixed control A2\n")
L(f"* Primary (pre-registered): mean over (instance, w, seed) of (A2 - A3)/A2, excluding A2 < 0.02 "
  f"({int((~ok).sum())} of {ok.size} triples excluded): **{100 * H_primary:+.2f}%**")
L(f"* Ratio of means (mean A2 - mean A3)/mean A2: {100 * H_rom:+.2f}%")
L(f"* Share of (instance, w, seed) where A3 beats A2 by more than 0.5%: {100 * beat05:.1f}%")
L(f"* Share of the A1 -> A3 gap closed by A2 (G101 instead of G21, fixed): **{100 * gap_share:.0f}%**\n")
L("| w_c | A1 fixed, G21 | A2 fixed, G101 | A3 segment oracle | headroom (A2-A3)/A2, per-instance mean | A3 beats A2 by >0.5% |")
L("|---|---|---|---|---|---|")
for j, w in enumerate(WC):
    r = rel[:, j]
    L(f"| {w:.2f} | {a1[:, j].mean():.4f} | {a2[:, j].mean():.4f} | {a3[:, j].mean():.4f} | "
      f"{100 * np.nanmean(r):+.2f}% | {100 * np.nanmean(r > 0.005):.0f}% |")
L("\nPer seed (primary headroom): " + ", ".join(f"{s}: {100 * np.nanmean(rel[i]):+.2f}%" for i, s in enumerate(SEEDS)))
ev = np.stack([D[s]["a3_evals"] for s in SEEDS])
L(f"\nEvaluations used by A3 per (instance, w): median {np.median(ev):.0f}, mean {ev.mean():.0f}, "
  f"range {ev.min()}-{ev.max()} (A2 uses 101 fixed-alpha rollouts per instance, shared across w).\n")
summary["a3_evals_median"] = float(np.median(ev))

# ------------------------------------------------------------------------------------------ bootstrap helper
rng = np.random.default_rng(BSEED)
BOOT = rng.integers(0, len(HELD), size=(NBOOT, len(HELD)))


def gain_ci(base_list, new_list, mask=None):
    """Ratio-of-means gain (base - new)/base over (w, held instance), bootstrap over held instances.
    Lists hold [w, held] arrays; several seeds are pooled with the same resampled instances."""
    m = mask if mask is not None else np.ones(len(WC), bool)
    B = np.stack([b[m] for b in base_list]); Nw = np.stack([x[m] for x in new_list])     # [S, w, held]
    g = (B.mean() - Nw.mean()) / B.mean()
    bs = B[:, :, BOOT].mean(axis=(0, 1, 3)); ns = Nw[:, :, BOOT].mean(axis=(0, 1, 3))
    gb = (bs - ns) / bs
    return float(g), float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))


# ------------------------------------------------------------------------------------------ deployable A5 vs A4
L("### Deployable gain on HELD-OUT: segment schedule A5 vs fixed alpha* A4, (A4 - A5)/A4\n")
L("| seed | all 21 preferences: gain [95% CI] | w_c 0.3-0.7: gain [95% CI] | passes criterion (CI > 0 and gain >= 2%) |")
L("|---|---|---|---|")
dep, n_pass = {}, 0
for s in SEEDS:
    g, lo, hi = gain_ci([A[s]["a4"]], [A[s]["a5"]]); gm, lom, him = gain_ci([A[s]["a4"]], [A[s]["a5"]], MID)
    p = lo > 0 and g >= 0.02; n_pass += p
    dep[s] = dict(gain=g, lo=lo, hi=hi, mid=gm, mid_lo=lom, mid_hi=him, passes=bool(p))
    L(f"| {s} | {100 * g:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}] | {100 * gm:+.2f}% [{100 * lom:+.2f}, {100 * him:+.2f}] | {'yes' if p else 'no'} |")
g, lo, hi = gain_ci([A[s]["a4"] for s in SEEDS], [A[s]["a5"] for s in SEEDS])
gm, lom, him = gain_ci([A[s]["a4"] for s in SEEDS], [A[s]["a5"] for s in SEEDS], MID)
L(f"| **pooled** | {100 * g:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}] | {100 * gm:+.2f}% [{100 * lom:+.2f}, {100 * him:+.2f}] | |")
summary.update(deployable=dep, deployable_pooled=dict(gain=g, lo=lo, hi=hi, mid=gm), seeds_passing_transfer=int(n_pass))

# ------------------------------------------------------------------------------------------ A6 cross-seed
if (RAW / "a6.npz").exists():
    A6 = dict(np.load(RAW / "a6.npz"))
    L("\n### Cross-seed transfer on HELD-OUT: seed s's A5 schedule on seed s''s specialists vs seed s''s own A4\n")
    L("| source -> target | all preferences: gain [95% CI] | w_c 0.3-0.7 |")
    L("|---|---|---|")
    bl, nl = [], []
    for src in SEEDS:
        for tgt in SEEDS:
            if src == tgt:
                continue
            new = np.stack([score(HELD, w, A6[f"{src}_on_{tgt}_mk"][j], A6[f"{src}_on_{tgt}_cb"][j]) for j, w in enumerate(WC)])
            g, lo, hi = gain_ci([A[tgt]["a4"]], [new]); gm, _, _ = gain_ci([A[tgt]["a4"]], [new], MID)
            bl.append(A[tgt]["a4"]); nl.append(new)
            L(f"| {src} -> {tgt} | {100 * g:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}] | {100 * gm:+.2f}% |")
    g, lo, hi = gain_ci(bl, nl)
    L(f"| **pooled (12 pairs)** | {100 * g:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}] | |")
    summary["a6_pooled"] = dict(gain=g, lo=lo, hi=hi)

# ------------------------------------------------------------------------------------------ hypervolume
L("\n### Hypervolume across preferences (reference (1.1, 1.1), normalised pool-mean points)\n")
L("| seed | A1 (all 100) | A3 (all 100) | A1 (held-out) | A3 (held-out) | A4 (held-out) | A5 (held-out) |")
L("|---|---|---|---|---|---|---|")
hvs = {}
for s in SEEDS:
    d, a = D[s], A[s]
    def pts(mk, cb, idx):
        return [tuple(np.array(norm(idx, mk[j][idx] if mk.shape[1] == N else mk[j], cb[j][idx] if cb.shape[1] == N else cb[j])).mean(1))
                for j in range(len(WC))]
    all_i = np.arange(N)
    h = dict(A1=hv(pts(a["a1_mk"], a["a1_cb"], all_i)), A3=hv(pts(d["a3_mk"], d["a3_cb"], all_i)),
             A1h=hv(pts(a["a1_mk"], a["a1_cb"], HELD)), A3h=hv(pts(d["a3_mk"], d["a3_cb"], HELD)),
             A4h=hv([tuple(np.array(norm(HELD, d["FMK"][a["a4_row"][j]][HELD], d["FCB"][a["a4_row"][j]][HELD])).mean(1))
                     for j in range(len(WC))]),
             A5h=hv([tuple(np.array(norm(HELD, d["a5_held_mk"][j], d["a5_held_cb"][j])).mean(1)) for j in range(len(WC))]))
    hvs[s] = h
    L(f"| {s} | {h['A1']:.4f} | {h['A3']:.4f} | {h['A1h']:.4f} | {h['A3h']:.4f} | {h['A4h']:.4f} | {h['A5h']:.4f} |")
summary["hv"] = hvs

# ------------------------------------------------------------------------------------------ schedule shape
alpha3 = np.stack([1 - G21[D[s]["a3_idx"]] for s in SEEDS])          # [S, w, n, K]
alpha5 = np.stack([1 - G21[D[s]["a5_idx"]] for s in SEEDS])          # [S, w, K]
m3, m5 = alpha3.mean(axis=(0, 2)), alpha5.mean(axis=0)              # [w, K]
L("\n### Schedule shape: mean alpha_k (weight on the MAKESPAN expert) by segment\n")
L("| w_c | A3 seg 1..5 | A3 first - last | A5 seg 1..5 | A5 first - last |")
L("|---|---|---|---|---|")
for j, w in enumerate(WC):
    L(f"| {w:.2f} | {' '.join(f'{x:.2f}' for x in m3[j])} | {m3[j, 0] - m3[j, -1]:+.2f} | "
      f"{' '.join(f'{x:.2f}' for x in m5[j])} | {m5[j, 0] - m5[j, -1]:+.2f} |")
d3 = alpha3[..., 0] - alpha3[..., -1]
L(f"\nA3, first minus last segment alpha: mean {d3.mean():+.3f}; share of (instance, w, seed) with more makespan weight "
  f"early {np.mean(d3 > 0):.2f}, later {np.mean(d3 < 0):.2f}, equal {np.mean(d3 == 0):.2f}. A5 (w_c 0.3-0.7): first minus "
  f"last per seed " + ", ".join(f"{(alpha5[i][MID, 0] - alpha5[i][MID, -1]).mean():+.2f}" for i in range(4)) + ".")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
cmap = plt.get_cmap("viridis")
for ax, M, ttl in ((axs[0], m3, "A3 per-instance segment oracle (mean)"), (axs[1], m5, "A5 deployable schedule (mean over seeds)")):
    for j, w in enumerate(WC):
        ax.plot(range(1, K + 1), M[j], "-o", ms=3, color=cmap(j / (len(WC) - 1)), lw=1)
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("segment (fifth of the schedule)"); ax.set_xticks(range(1, K + 1)); ax.grid(alpha=0.3)
axs[0].set_ylabel("alpha_k (weight on makespan expert)")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); sm.set_array([])
fig.colorbar(sm, ax=axs, label="preference w_c (carbon weight)")
fig.savefig(OUT / "schedule_shape.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------------------ state dependence
def ridge_cv(Xf, y, groups, lam=1.0, folds=5, seed=0):
    ug = np.unique(groups); r = np.random.default_rng(seed); r.shuffle(ug)
    fold_of = {g: i % folds for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups]); pred = np.zeros_like(y)
    for k in range(folds):
        tr, te = f != k, f == k
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0); sd[sd == 0] = 1
        Xt = (Xf[tr] - mu) / sd; ym = y[tr].mean()
        beta = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ (y[tr] - ym))
        pred[te] = ((Xf[te] - mu) / sd) @ beta + ym
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


rows = []
for si, s in enumerate(SEEDS):
    d = D[s]
    for j, w in enumerate(WC):
        for i in range(N):
            for k in range(K):
                win = slice(max(0, SEG0 := 10 * (k - 1)), 10 * k) if k > 0 else slice(0, 1)
                rows.append((si, j, i, k, alpha3[si, j, i, k], d["feat_differ"][j, i, win].mean(), d["feat_H_m"][j, i, win].mean(),
                             d["feat_H_c"][j, i, win].mean(), k / K, d["feat_rem_work"][j, i, k], d["feat_mk_ratio"][j, i, k],
                             d["feat_cb_ratio"][j, i, k], w))
R_ = np.array(rows, dtype=float)
S_, W_, I_, K_, Y = R_[:, 0], R_[:, 1], R_[:, 2], R_[:, 3], R_[:, 4]
Xall = R_[:, 5:]
grp = (S_ * 1000 + I_).astype(int)       # group = (seed, instance): an instance never spans train and test folds
strata = (S_ * 100 + W_) * 10 + K_
Yw = Y.copy()
for st in np.unique(strata):
    m = strata == st; Yw[m] = Y[m] - Y[m].mean()
L("\n### State-dependence check: predicting the A3 alpha_k from cheap state features (ridge, 5-fold CV grouped by instance)\n")
L("Features: disagreement rate and both entropies over the previous segment, fraction scheduled, remaining-work ratio, "
  "partial makespan / initial lower bound, partial carbon / minimum-carbon cost so far, w. No critic values.\n")
L("| segment | R^2, raw alpha_k (w included; reported, not decisive) | null 95th pct | R^2, WITHIN-preference alpha_k (decisive) | null 95th pct | beats null |")
L("|---|---|---|---|---|---|")
rng2 = np.random.default_rng(BSEED)
sd_res, n_beat = {}, 0
for k in list(range(K)) + ["pooled"]:
    m = (K_ == k) if k != "pooled" else np.ones(len(Y), bool)
    r_raw = ridge_cv(Xall[m], Y[m], grp[m]); r_w = ridge_cv(Xall[m], Yw[m], grp[m])
    null_raw, null_w = [], []
    for _ in range(200):
        null_raw.append(ridge_cv(Xall[m], rng2.permutation(Y[m]), grp[m]))
        yp = Yw[m].copy(); st = strata[m]
        for u in np.unique(st):
            ii = np.flatnonzero(st == u); yp[ii] = yp[rng2.permutation(ii)]
        null_w.append(ridge_cv(Xall[m], yp, grp[m]))
    q_raw, q_w = np.percentile(null_raw, 95), np.percentile(null_w, 95)
    beat = r_w > q_w
    if k != "pooled":
        n_beat += beat
    sd_res[str(k)] = dict(r2_raw=float(r_raw), null95_raw=float(q_raw), r2_within=float(r_w), null95_within=float(q_w), beats=bool(beat))
    lab = f"{k + 1}" if k != "pooled" else "pooled"
    L(f"| {lab} | {r_raw:.3f} | {q_raw:.3f} | {r_w:.3f} | {q_w:.3f} | {'yes' if beat else 'no'} |")
summary.update(state_dependence=sd_res, segments_beating_null=int(n_beat))

# ------------------------------------------------------------------------------------------ verdict
H_ok = H_primary >= 0.03
transfer_ok = n_pass >= 3
if H_ok and transfer_ok:
    verdict = "A"
elif H_ok:
    verdict = "B -> router (state)" if n_beat >= 2 else "B -> treated as C (no state signal)"
else:
    verdict = "C"
summary["verdict"] = verdict
L(f"\n### Verdict inputs\n\n* A3 vs A2 per-instance headroom: {100 * H_primary:+.2f}% (threshold 3%) -> {'met' if H_ok else 'NOT met'}")
L(f"* Seeds where A5 beats A4 on HELD-OUT with CI > 0 and gain >= 2%: {n_pass}/4 (need 3) -> {'met' if transfer_ok else 'NOT met'}")
L(f"* Segments where within-preference R^2 beats the null: {n_beat}/5 (need 2 for Outcome B -> router)")
L(f"* A2 closes {100 * gap_share:.0f}% of the A1 -> A3 gap")
L(f"* **Outcome: {verdict}**")
(OUT / "analysis_tables.md").write_text("\n".join(lines) + "\n")
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
print("\n".join(lines))
