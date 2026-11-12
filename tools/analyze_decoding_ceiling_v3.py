"""Report of docs/decoding_ceiling_protocol_v3.md: results/decoding_ceiling_v3.md (+ figures, summary.json).

    python tools/analyze_decoding_ceiling_v3.py
"""
from __future__ import annotations
import gzip, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import decoding_v3 as D                                             # noqa: E402  (CPU import)
import benchmark_v3 as B                                            # noqa: E402
import mo_metrics as MM                                             # noqa: E402
import tardiness_diagnosis_v3 as TD                                 # noqa: E402

RES = REPO / "results" / "decoding_ceiling_v3"
G = 66
W = D.PREFS[:G]
NORM = {"evaluation": REPO / "results/baselines_v3/normalisation_points.json",
        "validation": REPO / "results/warmstart_v3/norm/validation_normalisation_points_10x5.json"}
B2 = REPO / "results/baselines_v3/cpsat/b2/evaluation_10x5"
METHODS = ["greedy", "sample2", "sample4", "sample8", "sample16", "sample32", "sample64", "sample128",
           "beam2", "beam4", "beam8", "beam16", "beam32", "la_g2", "la_g3", "la_s2"]
NAME = {"greedy": "Greedy", **{f"sample{k}": f"Best-of-{k} sampling" for k in (2, 4, 8, 16, 32, 64, 128)},
        **{f"beam{k}": f"Beam search, width {k}" for k in (2, 4, 8, 16, 32)},
        "la_g2": "Lookahead, top 2, greedy continuation", "la_g3": "Lookahead, top 3, greedy continuation",
        "la_s2": "Lookahead, top 2, greedy + 3 sampled continuations"}
EXT = {"sample32", "sample64", "sample128", "beam16", "beam32"}           # budget-matched extensions
GROUPS = ["makespan-heavy", "carbon-heavy", "tardiness-heavy", "mixed"]


def group(w):
    k = int(np.argmax(w))
    return ["makespan-heavy", "carbon-heavy", "tardiness-heavy"][k] if w[k] >= 0.8 - 1e-9 else "mixed"


GRP = np.array([group(w) for w in W])


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"


class Split:
    def __init__(self, split):
        self.split = split
        self.data = B.load_split(split, "10x5")
        self.I = D.N_INST[split]
        self.norm = json.loads(NORM[split].read_text())["instances"]
        self.idn = [MM.ideal_nadir(self.norm[self.data.keys[i]]) for i in range(self.I)]
        self.runs = {}
        for key in D.CKPTS:
            for m in METHODS:
                p = RES / split / f"{key}__{m}.npz"
                if p.exists():
                    z = np.load(p)
                    self.runs[key, m] = {k: z[k] for k in z.files}
        self.front = None
        if split == "evaluation":
            self.front = [np.array([[p[k] for k in MM.OBJECTIVES] for p in
                                    json.loads((B2 / f"{Path(self.data.keys[i]).stem}.json").read_text())["front"]])
                          for i in range(self.I)]

    def score(self, i, obj, w):
        return (w[..., 0] * obj[..., 0] / self.data.mk_ref[i] + w[..., 1] * obj[..., 1] / self.data.cb_ref[i]
                + w[..., 2] * obj[..., 2] / self.data.t_ref)

    def hv(self, i, pts):
        return MM.instance_hv(pts, *self.idn[i])

    def metrics(self, obj):
        """obj [I, 66, 3] -> per-instance hv, igd+, unique non-dominated."""
        hv, igd, nd = [], [], []
        for i in range(self.I):
            p = obj[i, :G]
            hv.append(self.hv(i, p))
            q, _ = MM.normalise(p, *self.idn[i])
            keep = MM.nondominated(q)
            nd.append(int(len(np.unique(np.round(p[keep], 9), axis=0))))
            if self.front is not None:
                igd.append(MM.instance_igd_plus(p, self.front[i], *self.idn[i]))
        return np.array(hv), (np.array(igd) if igd else None), np.array(nd)


def norm_w_dot(sp, i, pts, w):
    """lambda . f~ in normalised space (kept objectives), for regret."""
    q, keep = MM.normalise(np.atleast_2d(pts), *sp.idn[i])
    return q @ np.asarray(w)[keep]


def render():
    out = RES.parent / "decoding_ceiling_v3.md"
    figs = RES / "figures"; figs.mkdir(parents=True, exist_ok=True)
    S = {s: Split(s) for s in ("validation", "evaluation")}
    lat = json.loads((RES / "latency.json").read_text()) if (RES / "latency.json").exists() else {}
    check = json.loads((RES / "nondelay_cpsat" / "check.json").read_text())
    summary = {}
    # ---------------- reference sets for regret: everything found for the instance (+ CP-SAT front on evaluation)
    srch = np.load(RES / "search" / "evaluation.npz") if (RES / "search" / "evaluation.npz").exists() else None
    pool = []
    ev = S["evaluation"]
    for i in range(ev.I):
        pts = [r["obj"][i, :G] for (k, m), r in ev.runs.items()]
        pool.append(np.concatenate(pts))
    gap_pool = _audit_pool_objs(ev)
    subset = _subset(ev)
    for i in range(ev.I):
        extra = [gap_pool[i]]
        if srch is not None:
            extra.append(srch["obj"][i])
        extra.append(subset["restricted_obj"][i])
        pool[i] = np.concatenate([pool[i]] + [e for e in extra if len(e)])
    nd_known = [p for p in pool]                                       # all DANIEL-reachable (checked below)
    ref_eval = [np.concatenate([nd_known[i], ev.front[i], subset["unrestricted_obj"][i]] if len(subset["unrestricted_obj"][i])
                               else [nd_known[i], ev.front[i]]) for i in range(ev.I)]
    va = S["validation"]
    ref_val = [np.concatenate([r["obj"][i, :G] for r in va.runs.values()]) for i in range(va.I)]

    def regret(sp, ref, obj):
        out = []
        for i in range(sp.I):
            for g, w in enumerate(W):
                out.append(float(norm_w_dot(sp, i, obj[i, g], w)[0] - norm_w_dot(sp, i, ref[i], w).min()))
        return float(np.mean(out))

    table = {}
    for s, sp in S.items():
        for (key, m), r in sp.runs.items():
            hv, igd, nd = sp.metrics(r["obj"])
            ev_pairs = r["evals"][:, :G]
            table[s, key, m] = {"hv": float(hv.mean()), "igd": float(igd.mean()) if igd is not None else None,
                                "nd": float(nd.mean()), "regret": regret(sp, ref_eval if s == "evaluation" else ref_val, r["obj"]),
                                "evals": float(ev_pairs.mean()), "complete": float(r["complete"][:, :G].mean()),
                                "sec_per_pair": float(r["seconds"]) / (sp.I * len(D.PREFS)),
                                "score": float(np.mean([sp.score(i, r["obj"][i, :G], W) for i in range(sp.I)])),
                                "group_score": {gname: float(np.mean([sp.score(i, r["obj"][i, :G], W)[GRP == gname] for i in range(sp.I)]))
                                                for gname in GROUPS}}
    summary["table"] = {f"{s}|{k}|{m}": v for (s, k, m), v in table.items()}

    # ---------------- validation selection (protocol)
    def val_mean(m, key="hv"):
        v = [table["validation", k, m][key] for k in D.CKPTS if ("validation", k, m) in table]
        return float(np.mean(v)) if len(v) == len(D.CKPTS) else None
    cand = [m for m in METHODS if val_mean(m) is not None and val_mean(m, "evals") <= 6400 + 1e-9]
    selected = max(cand, key=lambda m: val_mean(m))
    cand16 = [m for m in cand if val_mean(m, "evals") <= 800 + 1e-9]
    selected16 = max(cand16, key=lambda m: val_mean(m))
    summary["selected"] = selected; summary["selected_800"] = selected16

    L = []; w_ = L.append
    w_("# Decoding strategies and the non-delay ceiling, benchmark v3 (10x5)")
    w_("")
    w_("Inference and CP-SAT only. No training; network weights, input features, the action mask and the simulator are "
       "unchanged. Protocol, fixed before the runs and amended once after the validation pilot: "
       "`docs/decoding_ceiling_protocol_v3.md`.")
    w_("")
    w_(f"- **Settings chosen on:** the 18 validation instances. **Reported on:** the 20 evaluation instances with a "
       "CP-SAT reference front, as in the CP-SAT audit.")
    w_("- **Preferences:** the 66 grid preferences.")
    w_("- **Checkpoints (final):** World-model base mix (`pd_wm_u2250`), Random shared base mix (`pd_ctrl_u3650`), "
       "From scratch (`pd_fs_u4550`).")
    w_("- **Selection score:** every schedule is ranked by the benchmark's preference-weighted normalised score, using "
       "the simulator's exact objectives.")
    w_("- **Measurement:** single seed, numbers only. The reserved final-test sets were not used.")
    w_("")
    # ---------------- summary
    g_wm = table["evaluation", "wm", "greedy"]["hv"]; s16 = table["evaluation", "wm", "sample16"]["hv"]
    sel = table["evaluation", "wm", selected]["hv"]
    best_nd_hv = float(np.mean([ev.hv(i, nd_known[i]) for i in range(ev.I)]))
    cps_hv = float(np.mean([ev.hv(i, ev.front[i]) for i in range(ev.I)]))
    cps_plus = float(np.mean([ev.hv(i, np.concatenate([ev.front[i], subset["unrestricted_obj"][i]])
                                     if len(subset["unrestricted_obj"][i]) else ev.front[i]) for i in range(ev.I)]))
    old_nd = 0.913
    summary.update(best_nd_hv=best_nd_hv, cpsat_hv=cps_hv, cpsat_plus_hv=cps_plus)
    w_("## Summary")
    w_("")
    w_("| World-model base mix, evaluation (20 instances x 66 preferences) | Hypervolume | Network state evaluations per pair |")
    w_("|---|---|---|")
    w_(f"| Greedy | {f(g_wm)} | {f(table['evaluation', 'wm', 'greedy']['evals'], 0)} |")
    w_(f"| Best-of-16 sampling | {f(s16)} | {f(table['evaluation', 'wm', 'sample16']['evals'], 0)} |")
    w_(f"| **Selected on validation: {NAME[selected]}** | **{f(sel)}** | {f(table['evaluation', 'wm', selected]['evals'], 0)} |")
    w_(f"| Best within the best-of-16 budget on validation: {NAME[selected16]} | {f(table['evaluation', 'wm', selected16]['hv'])} | {f(table['evaluation', 'wm', selected16]['evals'], 0)} |")
    w_(f"| Best known DANIEL-reachable (non-delay) schedules: CP-SAT audit pool + every decoder here + new search | {f(best_nd_hv)} | - |")
    w_(f"| CP-SAT reference front (unrestricted) | {f(cps_hv)} | - |")
    w_("")
    rec_frac = (sel - g_wm) / (best_nd_hv - g_wm) if best_nd_hv > g_wm else float("nan")
    w_(f"- **Question A (better decoding):**")
    w_(f"  - The decoder chosen on validation raises the World-model base mix from {f(g_wm)} to {f(sel)}, against "
       f"{f(s16)} for best-of-16.")
    w_(f"  - That recovers {pct(sel - g_wm, best_nd_hv - g_wm)} of the gap between greedy and the best known "
       f"DANIEL-reachable schedules ({f(g_wm)} to {f(best_nd_hv)}).")
    w_("  - Budget-matched comparisons are in Part A.")
    w_(f"- **Question B (the non-delay ceiling):**")
    w_(f"  - Best known DANIEL-reachable hypervolume: {f(old_nd)} before this study, {f(best_nd_hv)} now.")
    w_(f"  - No hypervolume ceiling could be proven: the exact restricted model is too hard to solve.")
    w_(f"  - On the 200 instance-preference pairs of the proven subset, see B2/B3 for how much of the gap to "
       "unrestricted CP-SAT is proven to come from the non-delay rule.")
    w_("")

    # ---------------- Part A
    w_("## Part A: decoding the existing policies")
    w_("")
    w_("**Methods:**")
    w_("- **Greedy:** the argmax action.")
    w_("- **Best-of-K sampling:** the best of K sampled schedules. K = 2-16 are the first K of the committed best-of-16 "
       "stream (seed 50), so best-of-16 reproduces the committed evaluation. K = 32-128 come from a separate "
       "stream (seed 51).")
    w_("- **Beam search:** partial schedules are kept by accumulated log-probability, and identical partial schedules "
       "are merged. Finished schedules are ranked by the selection score.")
    w_("- **Lookahead:** at each step, DANIEL's top-k legal actions are each completed by the frozen policy, and the best "
       "by the selection score is executed.")
    w_("- Starred methods (*) are budget-matched extensions beyond the requested settings.")
    w_("")
    w_("**Cost columns**, per instance-preference pair:")
    w_("- **State evals:** network evaluations of a simulator state.")
    w_("- **Schedules:** completed schedules evaluated.")
    w_("- **Batched s/pair:** wall-clock time with all pairs batched on the GPU.")
    w_("- **Latency:** one pair at a time, mean and worst over 20 fixed evaluation pairs (RTX 5060 Ti).")
    w_("")
    w_("Regret: mean preference-weighted regret in normalised space, against every schedule found for the instance (and the CP-SAT front on evaluation).")
    w_("")
    for s in ("validation", "evaluation"):
        w_(f"### {s.capitalize()} split")
        w_("")
        for key, (lab, _) in D.CKPTS.items():
            if (s, key, "greedy") not in table:
                continue
            w_(f"**{lab}**")
            w_("")
            hdr = "| Method | HV | " + ("IGD+ | " if s == "evaluation" else "") + "Unique non-dominated | Regret | State evals | Schedules | Batched s/pair |" + (" Latency mean / worst (s) |" if s == "evaluation" else "")
            w_(hdr)
            w_("|" + "---|" * (hdr.count("|") - 1))
            for m in METHODS:
                t = table.get((s, key, m))
                if t is None:
                    continue
                row = f"| {NAME[m]}{'*' if m in EXT else ''} | {f(t['hv'])} | " + (f"{f(t['igd'])} | " if s == "evaluation" else "") + \
                      f"{f(t['nd'], 1)} | {f(t['regret'], 4)} | {f(t['evals'], 0)} | {f(t['complete'], 1)} | {f(t['sec_per_pair'], 4)} |"
                if s == "evaluation":
                    lm = lat.get(key, {}).get(m if not m.startswith("sample") or m in ("sample16", "sample128") else "", None)
                    row += f" {f(lm['mean'], 2)} / {f(lm['max'], 2)} |" if lm else " - |"
                w_(row)
            w_("")
    w_("Latency was measured for greedy, best-of-16, best-of-128, every beam width and every lookahead variant.")
    w_("")
    # selection
    w_("### Settings chosen on validation")
    w_("")
    w_("| Method | Validation HV (mean of 3 checkpoints) | State evals per pair |")
    w_("|---|---|---|")
    for m in METHODS:
        if val_mean(m) is not None:
            w_(f"| {NAME[m]}{'*' if m in EXT else ''} | {f(val_mean(m))} | {f(val_mean(m, 'evals'), 0)} |")
    w_("")
    w_(f"- **Selected (highest validation HV within 6,400 state evaluations per pair):** {NAME[selected]}.")
    w_(f"- **Best within the best-of-16 budget (800):** {NAME[selected16]}.")
    w_("")
    # budget-matched comparison on evaluation
    w_("### At similar budgets (evaluation)")
    w_("")
    w_("For each beam and lookahead setting, the best-of-K sampling run with the nearest budget at or above it is used. "
       "Values are the HV difference (method minus sampling).")
    w_("")
    w_("| Method | Its budget | Sampling compared | " + " | ".join(D.CKPTS[k][0] for k in D.CKPTS) + " |")
    w_("|---|---|---|" + "---|" * len(D.CKPTS))
    ks = [2, 4, 8, 16, 32, 64, 128]
    for m in ["beam2", "beam4", "beam8", "beam16", "beam32", "la_g2", "la_g3", "la_s2"]:
        if ("evaluation", "wm", m) not in table:
            continue
        b = table["evaluation", "wm", m]["evals"]
        kk = next((k for k in ks if 50 * k >= b - 1e-9), 128)
        diffs = [table["evaluation", key, m]["hv"] - table["evaluation", key, f"sample{kk}"]["hv"]
                 for key in D.CKPTS if ("evaluation", key, m) in table and ("evaluation", key, f"sample{kk}") in table]
        w_(f"| {NAME[m]} | {f(b, 0)} | best-of-{kk} ({50 * kk}) | " + " | ".join(f"{d:+.3f}" for d in diffs) + " |")
    w_("")
    _fig_budget(table, figs / "hv_vs_budget.png")
    w_("![HV against budget](decoding_ceiling_v3/figures/hv_vs_budget.png)")
    w_("")

    # ---------------- Part B
    w_("## Part B: the non-delay ceiling")
    w_("")
    w_("### B1. DANIEL's exact action space")
    w_("")
    w_("From `daniel/fjsp_env_same_op_nums.py`.")
    w_("- **Allowed actions.** At each step, the actions are pairs (job j's next unscheduled operation, machine m) "
       "with m eligible for that operation (processing time > 0).")
    w_("- **Pair start time.** A pair can start at max(job ready time, machine free time) (`pair_free_time`).")
    w_("- **The rule.** A pair is allowed only if that start equals the step's `next_schedule_time`, the minimum over all "
       "eligible pairs (lines 584-591 and 638-642). Every other logit is set to minus infinity "
       "(`model/main_model.py:192`). The chosen operation then starts at exactly that time.")
    w_("- **Definitions from the literature:**")
    w_("  - **Non-delay schedule:** no machine is ever idle while an operation it can process is ready. Every schedule "
       "the rule produces is non-delay (`results/diagnose_v3.md`).")
    w_("  - **Converse.** Every non-delay schedule can be produced, in exact arithmetic. Take the unscheduled operation that "
       "the target schedule starts first, at time s. If the earliest start t* over all eligible pairs were below s, some "
       "machine would be idle in the target from t* to s while an operation it could process is ready, which contradicts "
       "non-delay. So the earliest start equals s, and that operation's pair is allowed.")
    w_("  - **Active schedules** (Giffler-Thompson: an operation may start whenever no other operation could finish before "
       "it starts) and **semi-active schedules** (any start time that cannot be moved earlier without changing the "
       "order) are wider classes. DANIEL cannot produce them.")
    mc = check["mask_check"]
    w_(f"- **A floating-point detail.** The environment compares start times in scaled floating point (raw time / 99). "
       f"Against the same rule in exact integer arithmetic, at every step of {mc['states']:,} states (random decodings and "
       "every CP-SAT audit pool schedule):")
    w_(f"  - the float rule masks {mc['pairs_allowed_only_by_exact']:,} pairs that tie exactly for the earliest start, in "
       f"{mc['states_with_difference']:,} states ({pct(mc['states_with_difference'], mc['states'])});")
    w_(f"  - it allows {mc['pairs_allowed_only_by_float']} pairs that the exact rule forbids.")
    w_("  - So **DANIEL's action space is the non-delay rule with some exact ties lost**. Its schedules are a subset of "
       "the non-delay schedules, and any bound on non-delay schedules is also a bound for DANIEL.")
    w_("")
    w_("### B2. CP-SAT restricted to DANIEL's rule")
    w_("")
    w_("**Model.** The B1/B2 CP-SAT model (`tools/cpsat_baselines_v3.py`) with additional constraints:")
    w_("- a circuit of consecutive operations per machine;")
    w_("- for every machine m, every operation o eligible on m and every idle gap of m (before its first operation, "
       "between consecutive operations, after its last operation), the gap may not overlap the interval [r_o, s_o) during "
       "which o is ready and waiting.")
    w_("")
    w_("The code is `tools/nondelay_cpsat_v3.py`.")
    w_("")
    fx = check["fixed_daniel_schedules"]; fl = check["fixed_left_shifted_cpsat"]
    w_("**Checked equivalent before use:**")
    w_(f"- Pool schedules, all made by DANIEL's rule, are feasible when fixed in the model: {', '.join(f'{k} {v}' for k, v in fx.items())}.")
    w_(f"- Left-shifted CP-SAT schedules, which are not non-delay, are infeasible when fixed: {', '.join(f'{k} {v}' for k, v in fl.items())}.")
    w_("")
    w_("**Pilot** (validation instances 0-1, 5 preferences, up to 60 s, no hint):")
    w_("- no solve was proven optimal;")
    w_("- schedules were often worse than DANIEL's;")
    w_("- bounds were weak.")
    w_("")
    w_("Solving all 1,320 evaluation pairs exactly was out of reach, so the protocol was amended, before any evaluation "
       "solve, to two parts (below).")
    w_("")
    sb = subset
    w_(f"**1. Proven bounds on a subset.** {len(sb['rows'])} pairs: 20 instances x 10 preferences (the three corners, the "
       "three edge midpoints, and four interior points). Each pair is solved twice, for 60 s with 6 workers:")
    w_("- **restricted:** non-delay, hinted with the best known DANIEL schedule;")
    w_("- **unrestricted:** the plain model, hinted with CP-SAT's best front point.")
    w_("")
    rows = sb["rows"]
    st_r = {}; st_u = {}
    for r in rows:
        st_r[r["r_status"]] = st_r.get(r["r_status"], 0) + 1; st_u[r["u_status"]] = st_u.get(r["u_status"], 0) + 1
    w_(f"- Restricted solve status: {', '.join(f'{k} {v}' for k, v in sorted(st_r.items()))}.")
    w_(f"- Unrestricted solve status: {', '.join(f'{k} {v}' for k, v in sorted(st_u.items()))}.")
    w_(f"- Restricted solves that found a better schedule than the best known: {sum(r['r_improved'] for r in rows)}.")
    w_(f"  - Of these, DANIEL's rule reproduces {sum(r['r_improved'] and r['r_reproduced'] for r in rows)}; only those count as DANIEL-reachable.")
    w_("")
    w_("All values are mean selection scores (lower is better).")
    w_("- **Best known DANIEL** includes the restricted solves.")
    w_("- **Proven DANIEL lower bound** is a valid lower bound on the score of any schedule DANIEL can produce.")
    w_("- **Proven cost of the rule:** the restricted bound minus the best unrestricted schedule found, where positive. "
       "The unrestricted optimum is at most that schedule's score.")
    w_("")
    w_("| Preferences | Pairs | Unrestricted best (proven optimal) | Best known DANIEL | Proven DANIEL lower bound | Total gap (DANIEL - unrestricted) | Proven cost of the rule | Pairs with a proven cost | Unresolved |")
    w_("|---|---|---|---|---|---|---|---|---|")
    for gname in GROUPS + [None]:
        rr = [r for r in rows if gname is None or r["group"] == gname]
        if not rr:
            continue
        tot = np.mean([r["nd_best"] - r["u_best"] for r in rr]); pc = np.mean([max(0.0, r["r_lb"] - r["u_best"]) for r in rr])
        w_(f"| {gname or 'All 10'} | {len(rr)} | {f(np.mean([r['u_best'] for r in rr]))} ({sum(r['u_status'] == 'OPTIMAL' for r in rr)}) | "
           f"{f(np.mean([r['nd_best'] for r in rr]))} | {f(np.mean([r['r_lb'] for r in rr]))} | {f(tot)} | {f(pc)} ({pct(pc, tot)}) | "
           f"{sum(r['r_lb'] > r['u_best'] + 1e-9 for r in rr)} | {f(tot - pc)} ({pct(tot - pc, tot)}) |")
    w_("")
    w_("Reading the table:")
    w_("- **Proven cost** is a minimum: the restricted bound is a lower bound, so the true cost of the rule is at least this.")
    w_("- **Unresolved** is the part of the gap that the bounds cannot assign. It is either more cost of the rule, or "
       "better DANIEL schedules that no method has found yet.")
    w_("")
    if srch is not None:
        sset = json.loads(str(srch["settings"]))
        imp = (srch["score"] < srch["seed_score"] - 1e-9)
        w_(f"**2. Search inside DANIEL's rule** (all 66 preferences).")
        w_(f"- **Method:** random-key genetic search decoded by DANIEL's own rule (population {sset['population']}, "
           f"{sset['generations']} generations, seeded with the {sset['seeds']} best known schedules per pair).")
        w_(f"- **Result:** it improved the best known schedule in {int(imp.sum())} of {imp.size} pairs; the mean score "
           f"went from {f(float(srch['seed_score'].mean()), 4)} to {f(float(srch['score'].mean()), 4)}.")
        w_("- Every result was replayed through the environment and re-scored.")
        w_("")
    # ---------------- B3
    w_("### B3. The remaining gap (evaluation, identical normalisation)")
    w_("")
    w_("| Schedules | HV | Mean score: makespan-heavy | carbon-heavy | tardiness-heavy | mixed | all 66 |")
    w_("|---|---|---|---|---|---|---|")

    def gscore(objs):                                                   # objs [I, 66, 3]
        sc = np.stack([ev.score(i, objs[i], W) for i in range(ev.I)])
        return [float(sc[:, GRP == g].mean()) for g in GROUPS] + [float(sc.mean())]
    best_nd_obj = np.stack([nd_known[i][np.argmin(ev.score(i, nd_known[i][None, :, :], W[:, None, :]), 1)] for i in range(ev.I)])
    cp_obj = np.stack([ev.front[i][np.argmin(ev.score(i, ev.front[i][None, :, :], W[:, None, :]), 1)] for i in range(ev.I)])
    rows3 = [("DANIEL greedy, World-model base mix", table["evaluation", "wm", "greedy"]["hv"], ev.runs["wm", "greedy"]["obj"][:, :G]),
             (f"{NAME[selected]}, World-model base mix", table["evaluation", "wm", selected]["hv"], ev.runs["wm", selected]["obj"][:, :G])]
    for key in ("ctrl", "fs"):
        rows3.append((f"{NAME[selected]}, {D.CKPTS[key][0]}", table["evaluation", key, selected]["hv"], ev.runs[key, selected]["obj"][:, :G]))
    rows3 += [("Best known DANIEL-reachable (all sources)", best_nd_hv, best_nd_obj),
              ("CP-SAT reference front (unrestricted)", cps_hv, cp_obj)]
    for lab, h, o in rows3:
        gs = gscore(o)
        w_(f"| {lab} | {f(h)} | " + " | ".join(f(x) for x in gs) + " |")
    w_("")
    w_("For the two set rows (best known DANIEL-reachable, CP-SAT), each preference uses the set's best point for it.")
    w_("")
    w_("**Gap shares:**")
    w_(f"- Greedy to best known DANIEL-reachable ({f(g_wm)} to {f(best_nd_hv)}): the selected decoder recovers "
       f"{pct(sel - g_wm, best_nd_hv - g_wm)} for the World-model base mix. The best decoder on any checkpoint recovers "
       f"{pct(max(table['evaluation', k, m]['hv'] for k in D.CKPTS for m in METHODS if ('evaluation', k, m) in table) - g_wm, best_nd_hv - g_wm)}.")
    w_(f"- The search and the restricted solves raise the best known DANIEL front from {f(old_nd)} to {f(best_nd_hv)}.")
    w_(f"- Best known DANIEL-reachable to CP-SAT ({f(best_nd_hv)} to {f(cps_hv)}): see the proven-subset table. "
       "No hypervolume-level ceiling is proven.")
    w_("")
    _fig_groups(rows3, gscore, figs / "groups.png")
    w_("![By preference group](decoding_ceiling_v3/figures/groups.png)")
    w_("")
    w_("## Files")
    w_("")
    w_("- `tools/decoding_v3.py`: decoders and Part A runs. `tools/nondelay_cpsat_v3.py`: restricted CP-SAT, checks, "
       "subset bounds and search. `tools/analyze_decoding_ceiling_v3.py`: this report.")
    w_("- `results/decoding_ceiling_v3/<split>/<checkpoint>__<method>.npz`: every schedule (machine, start, objectives) and its cost.")
    w_("- `results/decoding_ceiling_v3/nondelay_cpsat/`: checks and subset solves (every solve's schedule, bound and status).")
    w_("- `results/decoding_ceiling_v3/search/evaluation.npz`: search results.")
    w_("- `results/decoding_ceiling_v3/latency.json`, `summary.json`.")
    w_("- `tests/test_decoding_ceiling_v3.py`, `tests/test_nondelay_cpsat_v3.py`.")
    out.write_text("\n".join(L) + "\n")
    summary["subset_rows"] = rows
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"written: {out}")


def _audit_pool_objs(ev):
    """Objectives of the CP-SAT audit's DANIEL-reachable pool (evaluation), per instance."""
    with gzip.open(REPO / "results/cpsat_gap_v3/analysis.json.gz", "rt") as fh:
        A = json.loads(fh.read())
    out = [[] for _ in range(ev.I)]
    for r in A["rows"]:
        out[r["i"]].append(r["best_obj"])
    pool = REPO / "results/cpsat_gap_v3/pool"
    fx = np.load(pool / "fixed.npz")
    for i in range(ev.I):
        for k in fx.files:
            if k.endswith("__machine"):
                m, s = fx[k][i], fx[k[: -len("__machine")] + "__start"][i]
                r = B.schedule_objectives(ev.data.job_length[i], ev.data.op_pt[i], ev.data.carbon[i], ev.data.due[i], m, s)
                out[i].append([r["makespan"], r["carbon"], r["tardiness"]])
        for p in sorted(pool.glob("mo__*.npz")):
            z = np.load(p)
            out[i] += z["greedy_obj"][i, :G].tolist()
            if "s16_obj" in z.files:
                out[i] += z["s16_obj"][i, :G].tolist()
    return [np.asarray(o) for o in out]


def _subset(ev):
    """Subset solves: per pair the unrestricted best and status, the restricted bound, best known DANIEL (incl. the
    restricted solve if DANIEL's rule reproduces it)."""
    import nondelay_cpsat_v3 as N
    d = REPO / "results/decoding_ceiling_v3/nondelay_cpsat/subset"
    rows, u_obj, r_obj = [], [[] for _ in range(ev.I)], [[] for _ in range(ev.I)]
    for i in range(ev.I):
        stem = Path(ev.data.keys[i]).stem
        fu, fr = d / f"unrestricted_{stem}.json", d / f"restricted_{stem}.json"
        if not (fu.exists() and fr.exists()):
            continue
        U = json.loads(fu.read_text())["solves"]; R = json.loads(fr.read_text())["solves"]
        inst = N.Inst(ev.data, i)
        for g in N.SUBSET:
            u, r = U.get(str(g)), R.get(str(g))
            if u is None or r is None or "score" not in u:
                continue
            w = N.PREFS[g]
            reproduced = False
            if "machine" in r:
                m = np.asarray(r["machine"])[None]; s = np.asarray(r["start"], float)[None]
                dd = TD.NonDelay(ev.data, np.array([i]))
                res = dd.run(TD.key_policy(TD.keys_from_schedule(m, s, 5)))
                reproduced = bool(np.array_equal(res["machine"], m) and np.allclose(res["start"], s))
                u_obj[i].append(u["obj"])
                if reproduced:
                    r_obj[i].append(r["obj"])
            r_score = r.get("score", np.inf)
            nd_best = min(r["hint_score"], r_score if reproduced else np.inf)
            rows.append({"i": i, "g": g, "group": group(w), "u_best": u["score"], "u_status": u["status"],
                         "u_lb": u.get("lower_bound"), "r_status": r["status"], "r_lb": r.get("lower_bound", -np.inf),
                         "nd_best": nd_best, "hint": r["hint_score"],
                         "r_improved": bool(r_score < r["hint_score"] - 1e-9), "r_reproduced": reproduced})
    return {"rows": rows, "unrestricted_obj": [np.asarray(x).reshape(-1, 3) for x in u_obj],
            "restricted_obj": [np.asarray(x).reshape(-1, 3) for x in r_obj]}


def _fig_budget(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, key in zip(axes, D.CKPTS):
        for fam, ms, mk in (("sampling", ["sample2", "sample4", "sample8", "sample16", "sample32", "sample64", "sample128"], "o-"),
                            ("beam", ["beam2", "beam4", "beam8", "beam16", "beam32"], "s-"),
                            ("lookahead", ["la_g2", "la_g3", "la_s2"], "^")):
            pts = [(table["evaluation", key, m]["evals"], table["evaluation", key, m]["hv"]) for m in ms if ("evaluation", key, m) in table]
            if pts:
                x, y = zip(*pts); ax.plot(x, y, mk, label=fam)
        if ("evaluation", key, "greedy") in table:
            ax.axhline(table["evaluation", key, "greedy"]["hv"], color="grey", ls=":", label="greedy")
        ax.set_xscale("log"); ax.set_title(D.CKPTS[key][0], fontsize=9); ax.set_xlabel("network state evaluations per pair")
    axes[0].set_ylabel("hypervolume (evaluation)"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_groups(rows3, gscore, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(GROUPS) + 1); wd = 0.8 / len(rows3)
    for k, (lab, _, o) in enumerate(rows3):
        ax.bar(x + (k - len(rows3) / 2 + 0.5) * wd, gscore(o), wd, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(GROUPS + ["all"]); ax.set_ylabel("mean selection score (lower is better)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    render()
