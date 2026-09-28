"""Analysis of Part C of results/v2_fixes.md (pre-registered decision rule).

Arms on the 49 HELD-OUT instances, 4 seeds: V1 (the original v1 tardiness specialist t_k125s060_s<seed>; evaluated
here once, cached in raw/eval_V1_s<seed>.npz), V2 (retrained on v2), V1CONT (v1 control, 1000 further updates).
CP-SAT per-job results from input_reward_audit for reference. Writes results/v2_fixes/{partC_tables.md,
partC_summary.json}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import ceiling_scorer as SC                        # noqa: E402

OUT = REPO / "results/v2_fixes"
RAW = OUT / "raw"
SEEDS, HELD = SC.SEEDS, [int(i) for i in SC.HELD]
NBOOT, BSEED = 10_000, 20260930
ARMS = {"V1": "v1 tardiness specialist", "V2": "V2: retrained on v2", "V1CONT": "V1-CONT: v1 + 1000 updates (control)"}


def ensure_v1_eval():
    missing = [s for s in SEEDS if not (RAW / f"eval_V1_s{s}.npz").exists()]
    if not missing:
        return
    import v2_common as V
    data = V.P.load_data("data_train_vali"); due = SC.due_dates()
    for s in missing:
        o = V.run(V.load_policy(f"10x5+carbon+priority+t_k125s060_s{s}", "v1"), data, due, SC.HELD, obs="v1")
        np.savez_compressed(RAW / f"eval_V1_s{s}.npz", held=SC.HELD, **o)


def main():
    ensure_v1_eval()
    sys.path.insert(0, str(REPO / "daniel"))
    from due_dates import work_content
    insts, dues = SC.instances(), SC.due_dates()
    P = np.zeros((len(HELD), 10)); D = np.zeros_like(P)
    for k, i in enumerate(HELD):
        ins = insts[i]; pt = np.zeros((ins.n_ops, ins.n_m))
        for o, d in enumerate(ins.pt):
            for m, p in d.items():
                pt[o, m] = p
        P[k] = work_content(np.array([len(o) for o in ins.job_ops]), pt); D[k] = dues[i]
    tight = D / P
    q1 = tight <= np.percentile(np.tile(tight.ravel(), 4), 25)       # the audit's pooled Q1 (same jobs every seed)
    R = {a: {s: dict(np.load(RAW / f"eval_{a}_s{s}.npz")) for s in SEEDS} for a in ARMS
         if all((RAW / f"eval_{a}_s{s}.npz").exists() for s in SEEDS)}
    import pandas as pd
    jobs = pd.read_csv(REPO / "results/input_reward_audit/jobs.csv")
    cp = jobs[jobs.schedule == "cpsat"]
    L, S = [], {"arms_available": list(R)}
    P_ = L.append

    def score(s, o):
        ref = SC.reference(s)
        return np.array([SC.score(s, i, o["mk"][k], o["cb"][k], o["tard"][k], ref=ref) for k, i in enumerate(HELD)])

    P_("### Part C: tardiness specialists on the 49 HELD-OUT instances (means over instances; pooled = over 4 seeds)\n")
    P_("| arm | seed | mean total tardiness | jobs late | tightest-quartile jobs late | Spearman(d/P, C/P) | makespan | carbon | Step 4 score |")
    P_("|---|---|---|---|---|---|---|---|---|")
    S["metrics"] = {}
    for a, lab in ARMS.items():
        if a not in R:
            continue
        rows = []
        for s in SEEDS + ["pooled"]:
            ss = SEEDS if s == "pooled" else [s]
            C = np.concatenate([R[a][x]["job_completion"] for x in ss]); Tj = np.maximum(0, C - np.tile(D, (len(ss), 1)))
            tt, pp = np.tile(tight, (len(ss), 1)), np.tile(P, (len(ss), 1)); qq = np.tile(q1, (len(ss), 1))
            m = dict(T=float(np.mean([R[a][x]["tard"].mean() for x in ss])), late=float((Tj > 0).mean()),
                     q1_late=float((Tj[qq] > 0).mean()), rho=float(stats.spearmanr(tt.ravel(), (C / pp).ravel()).statistic),
                     mk=float(np.mean([R[a][x]["mk"].mean() for x in ss])), cb=float(np.mean([R[a][x]["cb"].mean() for x in ss])),
                     score=float(np.mean([score(x, R[a][x]).mean() for x in ss])))
            S["metrics"].setdefault(a, {})[str(s)] = m
            P_(f"| {lab} | {s} | {m['T']:.1f} | {100*m['late']:.1f}% | {100*m['q1_late']:.1f}% | {m['rho']:+.3f} | "
               f"{m['mk']:.1f} | {m['cb']:.1f} | {m['score']:.4f} |")
    Ccp = cp.completion.values; rho_cp = stats.spearmanr(cp.due / cp.P_j, Ccp / cp.P_j.values).statistic
    P_(f"| CP-SAT 60 s best (ceiling_check) | pooled | {cp.groupby(['seed','pool_idx']).tardiness.sum().mean():.1f} | "
       f"{100*(cp.tardiness > 0).mean():.1f}% | - | {rho_cp:+.3f} | - | - | 0.6140 |")

    rng = np.random.default_rng(BSEED)
    BOOT = rng.integers(0, len(HELD), size=(NBOOT, len(HELD)))

    def reduction(x, y):          # (mean T of y - mean T of x) / mean T of y, positive = x lower tardiness
        g = (y.mean() - x.mean()) / y.mean(); gb = (y[BOOT].mean(1) - x[BOOT].mean(1)) / y[BOOT].mean(1)
        return float(g), float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))

    S["reductions"] = {}
    for x, y in (("V2", "V1"), ("V1CONT", "V1"), ("V2", "V1CONT")):
        if x not in R or y not in R:
            continue
        P_(f"\n**Tardiness reduction of {ARMS[x]} vs {ARMS[y]}** (paired over instances; 95% bootstrap CI, 10,000 resamples)\n")
        P_("| seed | reduction | 95% CI | counts (>= 20% and CI excludes 0) |")
        P_("|---|---|---|---|")
        for s in SEEDS:
            g, lo, hi = reduction(R[x][s]["tard"], R[y][s]["tard"])
            c = g >= 0.20 and (lo > 0 or hi < 0)
            S["reductions"].setdefault(f"{x}_vs_{y}", {})[str(s)] = dict(reduction=g, lo=lo, hi=hi, counts=bool(c))
            P_(f"| {s} | {100*g:+.1f}% | [{100*lo:+.1f}, {100*hi:+.1f}] | {'yes' if c else 'no'} |")
    if "V2" in R and "V1" in R:
        n = sum(v["counts"] for v in S["reductions"]["V2_vs_V1"].values())
        rho = S["metrics"]["V2"]["pooled"]["rho"]
        a1, a2 = n >= 3, rho >= 0.3
        verdict = "SUCCESS" if a1 and a2 else ("PARTIAL" if a1 or a2 else "NO EFFECT")
        S.update(verdict=verdict, seeds_counting=n, pooled_rho_v2=rho)
        P_(f"\n### Verdict: **{verdict}** (tardiness criterion {n}/4 seeds {'met' if a1 else 'not met'}; pooled "
           f"tightness correlation {rho:+.3f} {'>=' if a2 else '<'} +0.3)")
    (OUT / "partC_tables.md").write_text("\n".join(L) + "\n")
    (OUT / "partC_summary.json").write_text(json.dumps(S, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
