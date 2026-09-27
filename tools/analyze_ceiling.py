"""Analysis of the ceiling check (pre-registered in results/ceiling_check.md). No GPU, no solver.

Reads results/ceiling_check/raw/{c1_greedy_s*.npz, sampling_s*.npz, cpsat_t60/*.json, cpsat_t600/*.json} and the
blend_finetune / Step 4 curves; writes analysis_tables.md, summary.json and score_vs_updates_ceiling.png under
results/ceiling_check/.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import ceiling_scorer as SC                      # noqa: E402

OUT = SC.OUT
RAW = OUT / "raw"
SEEDS, HELD, TAG = [int(x) for x in os.environ.get("CEIL_SEEDS", "111 222 333 444").split()], [int(i) for i in SC.HELD], SC.TAG
NS = len(HELD)
NBOOT, BSEED = 10_000, 20260928
BUDGETS = [0, 50, 100, 250, 500, 1000]
INSTS, DUES = SC.instances(), SC.due_dates()
REF = {s: SC.reference(s) for s in SEEDS}


def replay(inst, machine, start):
    """DANIEL's append rule: operations in order of start time; start = max(job ready, machine free)."""
    order = sorted(range(inst.n_ops), key=lambda o: (start[o], o))
    job_of = {o: j for j, ops in enumerate(inst.job_ops) for o in ops}
    jr, mf, st = [0] * inst.n_j, [0] * inst.n_m, [0] * inst.n_ops
    for o in order:
        m, j = machine[o], job_of[o]
        st[o] = max(jr[j], mf[m]); jr[j] = mf[m] = st[o] + inst.pt[o][m]
    return st


# ------------------------------------------------------------------------------------------ load
G, B, CP, CP6 = {}, {}, {}, {}
for s in SEEDS:
    G[s] = dict(np.load(RAW / f"c1_greedy_s{s}.npz"))
    B[s] = dict(np.load(RAW / f"sampling_s{s}.npz"))
    assert list(G[s]["held"]) == HELD and list(B[s]["held"]) == HELD
    CP[s] = [json.loads((RAW / "cpsat_t60" / f"s{s}_i{i}.json").read_text()) for i in HELD]
for f in sorted((RAW / "cpsat_t600").glob("*.json")):
    r = json.loads(f.read_text()); CP6[(r["seed"], r["inst"])] = r
assert all(r["objective_int"] is not None for s in SEEDS for r in CP[s]), "a 60 s solve returned no solution"

cp_score = np.array([[r["score"] for r in CP[s]] for s in SEEDS])                  # [4, 49]
cp_bound = np.array([[r["bound_score"] for r in CP[s]] for s in SEEDS])
status = np.array([[r["status"] for r in CP[s]] for s in SEEDS])
policy = {"C1@1000 greedy": np.stack([G[s]["score"] for s in SEEDS])}
for n in (10, 30, 100):
    policy[f"C1@1000 best-of-{n}"] = np.stack([B[s]["score"][:, :n].min(1) for s in SEEDS])
# sanity: the scorer's greedy scores equal the evaluator's (Step 0) and bounds never exceed solutions
assert (cp_bound <= cp_score + 1e-12).all()

rng = np.random.default_rng(BSEED)
BOOT = rng.integers(0, NS, size=(NBOOT, NS))
L, summary = [], {"n_instances": NS, "seeds": SEEDS}
P_ = L.append


def gap_stats(x, ref):
    g = (x - ref) / ref                                                               # [4, 49]
    gb = g[:, BOOT].mean(axis=(0, 2))
    return dict(mean=float(g.mean()), median=float(np.median(g)), lo=float(np.percentile(gb, 2.5)),
                hi=float(np.percentile(gb, 97.5)), per_seed=[float(v) for v in g.mean(1)])


# ------------------------------------------------------------------------------------------ gap table
P_(f"### Gap to CP-SAT (60 s best solution), HELD-OUT, {len(SEEDS)} seeds x {NS} instances\n")
P_("Gap = (policy score - CP-SAT score) / CP-SAT score per (seed, instance); pooled mean, median, 95% bootstrap CI "
   "over instances (each carrying its seeds). Mean scores are pooled means over seeds and instances.\n")
P_("| method | mean score | gap mean | gap median | 95% CI | " + " | ".join(f"seed {s}" for s in SEEDS) + " |")
P_("|---|---|---|---|---|" + "---|" * len(SEEDS))
P_(f"| CP-SAT 60 s best | {cp_score.mean():.4f} | - | - | - | " + " | ".join(f"{v:.4f}" for v in cp_score.mean(1)) + " |")
P_(f"| CP-SAT proven bound | {cp_bound.mean():.4f} | - | - | - | " + " | ".join(f"{v:.4f}" for v in cp_bound.mean(1)) + " |")
summary["gaps"] = {}
for k, v in policy.items():
    st = gap_stats(v, cp_score); summary["gaps"][k] = st | {"mean_score": float(v.mean())}
    P_(f"| {k} | {v.mean():.4f} | {100*st['mean']:+.2f}% | {100*st['median']:+.2f}% | "
       f"[{100*st['lo']:+.2f}, {100*st['hi']:+.2f}] | " + " | ".join(f"{100*x:+.2f}%" for x in st["per_seed"]) + " |")
P_("\nSame methods against the PROVEN BOUND (upper limit on the gap to the true optimum): (policy - bound) / bound\n")
P_("| method | gap-to-bound mean | 95% CI |")
P_("|---|---|---|")
summary["gaps_to_bound"] = {}
for k, v in [("CP-SAT 60 s best", cp_score)] + list(policy.items()):
    st = gap_stats(v, cp_bound); summary["gaps_to_bound"][k] = st
    P_(f"| {k} | {100*st['mean']:+.2f}% | [{100*st['lo']:+.2f}, {100*st['hi']:+.2f}] |")

# share of the greedy->CP-SAT gap closed by sampling
closed = {k: float((policy["C1@1000 greedy"].mean() - v.mean()) / (policy["C1@1000 greedy"].mean() - cp_score.mean()))
          for k, v in policy.items() if "best-of" in k}
summary["share_of_greedy_gap_closed"] = closed
P_("\nShare of the greedy-to-CP-SAT score difference closed by best-of-N: "
   + ", ".join(f"{k.split()[-1]} {100*v:.0f}%" for k, v in closed.items()) + ".\n")

# ------------------------------------------------------------------------------------------ solver status
sg = (cp_score - cp_bound) / cp_score
n_opt = int((status == "OPTIMAL").sum())
hint_viol = [(s, HELD[k], CP[s][k]["score"] - CP[s][k]["hint_score"]) for si, s in enumerate(SEEDS) for k in range(NS)
             if CP[s][k]["score"] > CP[s][k]["hint_score"] + 1e-6]
hint_same = int(sum(abs(CP[s][k]["score"] - CP[s][k]["hint_score"]) <= 1e-6 for s in SEEDS for k in range(NS)))
walls = np.array([[r["wall_s"] for r in CP[s]] for s in SEEDS])
summary["solver"] = dict(optimal=n_opt, pairs=len(SEEDS) * NS, share_optimal=n_opt / (len(SEEDS) * NS), solver_gap_mean=float(sg.mean()),
                         solver_gap_median=float(np.median(sg)), solver_gap_max=float(sg.max()),
                         hint_violations=len(hint_viol), hint_not_improved=hint_same, wall_mean=float(walls.mean()))
P_("### CP-SAT status (60 s, 8 workers, warm-started with C1@1000 greedy)\n")
P_(f"* OPTIMAL: **{n_opt} / {len(SEEDS)*NS}** pairs ({100*n_opt/(len(SEEDS)*NS):.0f}%); the rest FEASIBLE "
   f"({ {str(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))} }).")
P_(f"* Solver gap (best - bound) / best: mean {100*sg.mean():.1f}%, median {100*np.median(sg):.1f}%, "
   f"range [{100*sg.min():.1f}%, {100*sg.max():.1f}%]; per seed " + ", ".join(f"{100*v:.1f}%" for v in sg.mean(1)) + ".")
P_(f"* Hint check: solution worse than the C1 hint in **{len(hint_viol)}** pairs (tolerance 1e-6); equal to the hint "
   f"in {hint_same}. Mean wall time {walls.mean():.1f} s.")

# reachability: replay through DANIEL's append rule
worse = 0; rep_sc = np.zeros_like(cp_score)
for si, s in enumerate(SEEDS):
    for k, i in enumerate(HELD):
        r = CP[s][k]
        st = replay(INSTS[i], r["machine"], r["start"])
        rep_sc[si, k] = SC.score(s, i, *SC.raw_objectives(INSTS[i], DUES[i], r["machine"], st), ref=REF[s])
        worse += rep_sc[si, k] > r["score"] + 1e-12
summary["reachability"] = dict(replay_worse=int(worse), replay_better=int((rep_sc < cp_score - 1e-12).sum()),
                               replay_mean=float(rep_sc.mean()))
P_(f"* Reachability: replaying every CP-SAT schedule through DANIEL's append rule gives a score worse than CP-SAT's in "
   f"**{worse}** pairs (strictly better in {summary['reachability']['replay_better']}; mean {rep_sc.mean():.4f}). "
   "The CP-SAT schedules lie inside DANIEL's action space.\n")

P_("### 60 s vs 600 s (10 pre-registered pairs)\n")
P_("| seed | instance | status 60 / 600 | score 60 | score 600 | change | bound 60 | bound 600 | change | solver gap 60 / 600 |")
P_("|---|---|---|---|---|---|---|---|---|---|")
d_sc, d_bd = [], []
for (s, i), r6 in CP6.items():
    r = CP[s][HELD.index(i)]
    d_sc.append((r6["score"] - r["score"]) / r["score"]); d_bd.append((r6["bound_score"] - r["bound_score"]) / r["bound_score"])
    P_(f"| {s} | {i} | {r['status']} / {r6['status']} | {r['score']:.4f} | {r6['score']:.4f} | {100*d_sc[-1]:+.2f}% | "
       f"{r['bound_score']:.4f} | {r6['bound_score']:.4f} | {100*d_bd[-1]:+.2f}% | "
       f"{100*(r['score']-r['bound_score'])/r['score']:.1f}% / {100*(r6['score']-r6['bound_score'])/r6['score']:.1f}% |")
if CP6:
    summary["t600"] = dict(n=len(CP6), score_change_mean=float(np.mean(d_sc)), score_change_min=float(np.min(d_sc)),
                           bound_change_mean=float(np.mean(d_bd)), bound_change_max=float(np.max(d_bd)))
    P_(f"\nMean change 60 -> 600 s: best score {100*np.mean(d_sc):+.2f}% (largest improvement {100*np.min(d_sc):+.2f}%), "
       f"proven bound {100*np.mean(d_bd):+.2f}% (largest rise {100*np.max(d_bd):+.2f}%).\n")

# ------------------------------------------------------------------------------------------ raw objectives
P_("### Raw objectives, HELD-OUT (mean over instances, then seeds)\n")
P_("| schedule | makespan | carbon | tardiness | score |")
P_("|---|---|---|---|---|")
cp_raw = np.array([[[r["mk"], r["cb"], r["tard"]] for r in CP[s]] for s in SEEDS])          # [4, 49, 3]
gr_raw = np.stack([np.stack([G[s]["mk"], G[s]["cb"], G[s]["tard"]], 1) for s in SEEDS])
bi = {s: B[s]["score"].argmin(1) for s in SEEDS}
b_raw = np.stack([np.stack([B[s][k][np.arange(NS), bi[s]] for k in ("mk", "cb", "tard")], 1) for s in SEEDS])
summary["raw"] = {}
for lab, R, sc in (("CP-SAT 60 s best", cp_raw, cp_score), ("C1@1000 greedy", gr_raw, policy["C1@1000 greedy"]),
                   ("C1@1000 best-of-100", b_raw, policy["C1@1000 best-of-100"])):
    m = R.mean(axis=(0, 1)); summary["raw"][lab] = [float(x) for x in m] + [float(sc.mean())]
    P_(f"| {lab} | {m[0]:.1f} | {m[1]:.1f} | {m[2]:.1f} | {sc.mean():.4f} |")
# per-objective contribution to the score gap (weighted, normalised terms)
def terms(R):
    out = []
    for si, s in enumerate(SEEDS):
        mr, cr, tr = REF[s]
        out.append(np.stack([SC.W[0] * R[si, :, 0] / mr[HELD], SC.W[1] * R[si, :, 1] / cr[HELD],
                             SC.W[2] * R[si, :, 2] / tr], 1))
    return np.stack(out).mean(axis=(0, 1))
tg, tc, tb = terms(gr_raw), terms(cp_raw), terms(b_raw)
summary["score_terms"] = {"greedy": tg.tolist(), "cpsat": tc.tolist(), "best100": tb.tolist()}
P_("\nWeighted score terms (w x objective / normaliser; they sum to the score):\n")
P_("| schedule | makespan term | carbon term | tardiness term |")
P_("|---|---|---|---|")
for lab, t in (("CP-SAT 60 s best", tc), ("C1@1000 greedy", tg), ("C1@1000 best-of-100", tb)):
    P_(f"| {lab} | {t[0]:.4f} | {t[1]:.4f} | {t[2]:.4f} |")
dt = tg - tc
P_(f"\nGreedy minus CP-SAT, per term: makespan {dt[0]:+.4f}, carbon {dt[1]:+.4f}, tardiness {dt[2]:+.4f} "
   f"(share of the total difference: " + ", ".join(f"{100*x/dt.sum():.0f}%" for x in dt) + ").\n")

# ------------------------------------------------------------------------------------------ verdict
g, b = summary["gaps"]["C1@1000 greedy"], summary["gaps"]["C1@1000 best-of-100"]
within = lambda st: st["mean"] <= 0.02 and st["hi"] <= 0.03
if within(g):
    verdict = "SATURATED"
elif g["mean"] > 0.02 and within(b):
    verdict = "DECODING-LIMITED"
elif g["mean"] > 0.02 and b["mean"] > 0.02:
    verdict = "POLICY-OR-BUDGET-LIMITED"
else:
    verdict = "NOT DECIDED"
summary["verdict"] = verdict
summary["proven_optimum"] = n_opt / (len(SEEDS) * NS) >= 0.8
P_(f"### Verdict: **{verdict}** (greedy gap {100*g['mean']:+.2f}% [{100*g['lo']:+.2f}, {100*g['hi']:+.2f}], "
   f"best-of-100 gap {100*b['mean']:+.2f}% [{100*b['lo']:+.2f}, {100*b['hi']:+.2f}]; "
   f"{'proven optimum' if summary['proven_optimum'] else 'best-known solution, NOT a proven optimum'}).")

(OUT / "analysis_tables.md").write_text("\n".join(L) + "\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("\n".join(L))

# ------------------------------------------------------------------------------------------ figure
S4 = SC.S4
BF = REPO / "results/blend_finetune/raw"


def curve_score(s, out3):
    mr, cr, tr = REF[s]
    return SC.W[0] * (out3[0] / mr[HELD]) + SC.W[1] * (out3[1] / cr[HELD]) + SC.W[2] * (out3[2] / tr)


curves = {}
for s in SEEDS:
    for lab, stem in (("C1 makespan spec (Step 4)", "spec_m"), ("Step 4 merge", "merge"), ("Step 4 scratch", "scratch")):
        for bb in BUDGETS:
            f = S4 / f"Result_DANIELG+10x5+carbon+priority+s4t_{stem}_{TAG}_s{s}@u{bb}_trainvali-{TAG}_10x5+carbon+priority.npy"
            curves.setdefault(lab, {}).setdefault(s, {})[bb] = curve_score(s, np.load(f)[:, [0, 1, 3]][HELD].T).mean()
    for arm, lab in (("F1", "F1 blend, joint fine-tune"), ("F2", "F2 blend + residual"),
                     ("C2", "C2 makespan spec + fresh critic"), ("C3", "C3 merge + fresh critic")):
        d = np.load(BF / f"{arm}_s{s}.npz")
        curves.setdefault(lab, {})[s] = {bb: curve_score(s, d[f"u{bb}"]).mean() for bb in BUDGETS}
style = {"C1 makespan spec (Step 4)": ("tab:orange", "-o"), "C2 makespan spec + fresh critic": ("tab:orange", "--s"),
         "C3 merge + fresh critic": ("tab:blue", "--s"), "F1 blend, joint fine-tune": ("purple", "-D"),
         "F2 blend + residual": ("tab:pink", "-D"), "Step 4 merge": ("tab:blue", "-o"), "Step 4 scratch": ("tab:gray", "-o")}
fig, ax = plt.subplots(figsize=(9, 5.2))
for lab, (col, mk) in style.items():
    Y = np.array([[curves[lab][s][bb] for bb in BUDGETS] for s in SEEDS])
    ax.plot(BUDGETS, Y.mean(0), mk, ms=3, color=col, lw=1.3, label=lab)
ax.axhline(cp_score.mean(), color="k", lw=1.5, label=f"CP-SAT best, 60 s ({cp_score.mean():.3f})")
ax.axhline(cp_bound.mean(), color="k", lw=1.2, ls="--", label=f"CP-SAT proven bound ({cp_bound.mean():.3f})")
for n, m in ((10, "v"), (30, "^"), (100, "*")):
    v = policy[f"C1@1000 best-of-{n}"].mean()
    ax.plot([1000], [v], m, color="tab:red", ms=8 if n == 100 else 6, label=f"C1@1000 best-of-{n} ({v:.3f})")
ax.set_xscale("symlog", linthresh=50); ax.set_xticks(BUDGETS); ax.set_xticklabels([str(bb) for bb in BUDGETS])
ax.set_xlabel("PPO updates (symlog, linthresh 50)"); ax.set_ylabel("Step 4 score, HELD-OUT (lower is better)")
ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.set_title("10x5 HELD-OUT: training curves (mean over 4 seeds) vs the CP-SAT ceiling", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "score_vs_updates_ceiling.png", dpi=130); plt.close(fig)
