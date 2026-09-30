"""Analysis of the v2 re-run (pre-registered in results/v2_rerun_seed111.md). CPU only.

Reads <results>/eval/<arm>_s<seed>@u<N>.npz, <results>/provenance/ (chunk timings, .done/.FAILED markers), the
seed's CP-SAT 60 s solutions (results/ceiling_check/raw/cpsat_t60), and the Step 4 result files for the v1 C1 and
scratch curves. Writes <results>/analysis_tables.md, <results>/summary.json, <results>/score_vs_updates.png, and the
generated section of the report (between the GENERATED markers; the pre-registration is never touched).

An arm counts only if it is not FAILED and every planned snapshot is evaluated; a rule that depends on an arm that
does not count is NOT DETERMINED.

    python tools/analyze_v2_rerun.py [--seeds 111] [--results-dir ...] [--report ...] [--smoke --out-root DIR]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import ceiling_scorer as SC                              # noqa: E402
from run_v2_rerun import ARMS, plan                      # noqa: E402

S4 = REPO / "daniel/test_results/SD2/trainvali-k125s060_10x5+carbon+priority"
BEGIN, END = "<!-- BEGIN GENERATED RESULTS (tools/analyze_v2_rerun.py) -->", "<!-- END GENERATED RESULTS -->"
NBOOT, BSEED = 10_000, 20261001
LABEL = {"V2-M": "V2-M makespan spec, v2", "V2-S": "V2-S scratch, v2", "V1-CONT": "V1-CONT C1 continued, v1",
         "V2-T": "V2-T v2 tardiness spec, v2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[111])
    ap.add_argument("--results-dir", default="results/v2_rerun_seed111")
    ap.add_argument("--report", default="results/v2_rerun_seed111.md")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default="")
    a = ap.parse_args()
    if a.smoke:
        root = Path(a.out_root).resolve()
        res, report = root / "results", root / "v2_rerun_seed111.md"
    else:
        res, report = REPO / a.results_dir, REPO / a.report
    seeds = a.seeds
    # rule checkpoints; in smoke mode (2 updates per arm) they map onto the smoke snapshots to exercise the code
    U = (lambda u, arm: ({0: 0, 50: 1, 100: 1, 250: 1, 500: 2, 1000: 1, 1500: 2, 2000: 2, 2500: 2, 3000: 2}[u]
                         + (1000 if arm == "V1-CONT" else 0))) if a.smoke else (lambda u, arm: u)

    # ------------------------------------------------------------------ load
    E, status = {}, {}
    for arm in ARMS:
        snaps, _ = plan(arm, a.smoke)
        ok = True
        for s in seeds:
            failed = (res / "provenance" / f"{arm}_s{s}.FAILED").exists()
            for u in snaps:
                f = res / "eval" / f"{arm}_s{s}@u{u}.npz"
                if f.exists():
                    E.setdefault(arm, {}).setdefault(s, {})[u] = dict(np.load(f))
            have = set(E.get(arm, {}).get(s, {}))
            ok &= (not failed) and set(snaps) <= have
        status[arm] = "complete" if ok else ("FAILED" if any((res / "provenance" / f"{arm}_s{s}.FAILED").exists()
                                                             for s in seeds) else "incomplete")
    any_eval = next((d for arm in E for s in E[arm] for d in E[arm][s].values()), None)
    held = any_eval["held"] if any_eval is not None else SC.HELD
    hidx = [list(SC.HELD).index(i) for i in held]
    n = len(held)

    def sc(arm, u):                                   # [seeds, n] per-instance scores, or None
        try:
            return np.stack([E[arm][s][U(u, arm)]["score"] for s in seeds])
        except KeyError:
            return None

    # references: v1 Step 4 curves, CP-SAT, v1 C1@1000
    def s4(name, s):
        r = np.load(S4 / f"Result_DANIELG+10x5+carbon+priority+{name}_trainvali-k125s060_10x5+carbon+priority.npy")[:, [0, 1, 3]]
        ref = SC.reference(s)
        return r[held], np.array([SC.score(s, i, *r[i], ref=ref) for i in held])
    v1c1 = {s: s4(f"s4t_spec_m_k125s060_s{s}@u1000", s) for s in seeds}
    C1_1000 = np.stack([v1c1[s][1] for s in seeds])
    cp = np.stack([np.array([json.loads((REPO / "results/ceiling_check/raw/cpsat_t60" / f"s{s}_i{int(i)}.json")
                                        .read_text())["score"] for i in held]) for s in seeds])
    rng = np.random.default_rng(BSEED)
    BOOT = rng.integers(0, n, size=(NBOOT, n))

    def gain(X, Y):                                   # relative gain of X over Y, positive = X better
        g = (Y.mean() - X.mean()) / Y.mean()
        xb, yb = X[:, BOOT].mean(axis=(0, 2)), Y[:, BOOT].mean(axis=(0, 2))
        gb = (yb - xb) / yb
        return float(g), float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))

    def gap(X):
        g = (X - cp) / cp
        gb = g[:, BOOT].mean(axis=(0, 2))
        return float(g.mean()), float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))

    L, S = [], {"seeds": seeds, "n_instances": n, "arm_status": status, "smoke": a.smoke}
    P_ = L.append
    P_(f"Seeds: {seeds}; {n} HELD-OUT instances. EXPLORATORY: confidence intervals are paired bootstrap intervals "
       "over instances (10,000 resamples) and reflect instance variation only, not seed variation.\n")
    P_("Arm status: " + ", ".join(f"{LABEL[k]}: **{v}**" for k, v in status.items()) + ".\n")

    # ------------------------------------------------------------------ score table
    ups = sorted({u for arm in ARMS for u in plan(arm, a.smoke)[0]})
    P_("### Step 4 score per checkpoint (mean over instances; lower is better)\n")
    P_("| arm | " + " | ".join(f"u{u}" for u in ups) + " |")
    P_("|---|" + "---|" * len(ups))
    S["scores"] = {}
    for arm in ARMS:
        row = []
        for u in ups:
            v = [E[arm][s][u]["score"].mean() for s in seeds if u in E.get(arm, {}).get(s, {})]
            row.append(f"{np.mean(v):.4f}" if len(v) == len(seeds) else "-")
            if len(v) == len(seeds):
                S["scores"].setdefault(arm, {})[str(u)] = float(np.mean(v))
        P_(f"| {LABEL[arm]} | " + " | ".join(row) + " |")
    v1curves = {}
    for nm, stem in (("v1 Step 4 C1 (makespan spec)", "spec_m"), ("v1 Step 4 scratch", "scratch")):
        row = []
        for u in ups:
            if u <= 1000 and u in (0, 50, 100, 250, 500, 1000):
                v = np.mean([s4(f"s4t_{stem}_k125s060_s{s}@u{u}", s)[1].mean() for s in seeds])
                v1curves.setdefault(nm, {})[u] = v; row.append(f"{v:.4f}")
            else:
                row.append("-")
        P_(f"| {nm} (reference) | " + " | ".join(row) + " |")
    P_(f"| CP-SAT 60 s best (ceiling_check) | " + " | ".join(f"{cp.mean():.4f}" if k == 0 else "" for k in range(len(ups))) + " |")

    # ------------------------------------------------------------------ gap table
    P_("\n### Gap to the CP-SAT best per checkpoint: mean of (score - CP-SAT) / CP-SAT, 95% bootstrap CI\n")
    P_("| arm | checkpoint | gap | 95% CI |")
    P_("|---|---|---|---|")
    S["gaps"] = {}
    for arm in ARMS:
        for u in plan(arm, a.smoke)[0]:
            X = sc(arm, u) if not a.smoke else (np.stack([E[arm][s][u]["score"] for s in seeds])
                                                if all(u in E.get(arm, {}).get(s, {}) for s in seeds) else None)
            if X is None:
                continue
            g, lo, hi = gap(X); S["gaps"].setdefault(arm, {})[str(u)] = [g, lo, hi]
            P_(f"| {LABEL[arm]} | u{u} | {100*g:+.1f}% | [{100*lo:+.1f}, {100*hi:+.1f}] |")
    g, lo, hi = gap(C1_1000)
    P_(f"| v1 Step 4 C1 (reference) | u1000 | {100*g:+.1f}% | [{100*lo:+.1f}, {100*hi:+.1f}] |")

    # ------------------------------------------------------------------ raw objectives and due-date behaviour
    sys.path.insert(0, str(REPO / "daniel"))
    from due_dates import work_content
    insts, dues = SC.instances(), SC.due_dates()
    Pj, Dj = np.zeros((len(SC.HELD), 10)), np.zeros((len(SC.HELD), 10))
    for k, i in enumerate(SC.HELD):
        ins = insts[i]; pt = np.zeros((ins.n_ops, ins.n_m))
        for o, d in enumerate(ins.pt):
            for m, p in d.items():
                pt[o, m] = p
        Pj[k] = work_content(np.array([len(o) for o in ins.job_ops]), pt); Dj[k] = dues[i]
    tight_all = Dj / Pj
    q1_all = tight_all <= np.percentile(tight_all, 25)          # the audit's tightest-quartile job set
    tight, q1, Ph, Dh = tight_all[hidx], q1_all[hidx], Pj[hidx], Dj[hidx]

    def behaviour(C):                                    # C: [seeds, n, J] job completions
        Tj = np.maximum(0, C - Dh[None]); tt = np.broadcast_to(tight, C.shape)
        rho = stats.spearmanr(tt.ravel(), (C / Ph[None]).ravel()).statistic
        return float(rho), float((Tj[:, q1] > 0).mean())

    import pandas as pd
    jobs = pd.read_csv(REPO / "results/input_reward_audit/jobs.csv")
    cpj = jobs[(jobs.schedule == "cpsat") & jobs.seed.isin(seeds) & jobs.pool_idx.isin([int(i) for i in held])]
    rho_cp = stats.spearmanr(cpj.due / cpj.P_j, cpj.completion / cpj.P_j).statistic
    P_("\n### Raw objectives and due-date behaviour at updates 1,000 and 3,000\n")
    P_("Spearman = Spearman(d_j/P_j, C_j/P_j) over the HELD-OUT jobs, as input_reward_audit (v1 C1 about -0.02, "
       "CP-SAT +0.74); Q1 late = share of the tightest-quartile jobs that are late.\n")
    P_("| arm | update | makespan | carbon | tardiness | score | Spearman | Q1 late |")
    P_("|---|---|---|---|---|---|---|---|")
    S["behaviour"] = {}
    for arm in ARMS:
        for u in (1000, 3000):
            try:
                d = [E[arm][s][U(u, arm)] for s in seeds]
            except KeyError:
                continue
            C = np.stack([x["job_completion"] for x in d]); rho, q1l = behaviour(C)
            m = [float(np.mean([x[k].mean() for x in d])) for k in ("mk", "cb", "tard", "score")]
            S["behaviour"].setdefault(arm, {})[str(u)] = dict(mk=m[0], cb=m[1], tard=m[2], score=m[3], rho=rho, q1_late=q1l)
            P_(f"| {LABEL[arm]} | {u} | {m[0]:.1f} | {m[1]:.1f} | {m[2]:.1f} | {m[3]:.4f} | {rho:+.3f} | {100*q1l:.1f}% |")
    r = np.mean([v1c1[s][0].mean(0) for s in seeds], 0)
    P_(f"| v1 Step 4 C1 (reference) | 1000 | {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} | {C1_1000.mean():.4f} | (V1-CONT u1000) | (V1-CONT u1000) |")
    cpr = [json.loads((REPO / "results/ceiling_check/raw/cpsat_t60" / f"s{s}_i{int(i)}.json").read_text()) for s in seeds for i in held]
    P_(f"| CP-SAT 60 s best | - | {np.mean([x['mk'] for x in cpr]):.1f} | {np.mean([x['cb'] for x in cpr]):.1f} | "
       f"{np.mean([x['tard'] for x in cpr]):.1f} | {cp.mean():.4f} | {rho_cp:+.3f} | {100*(cpj[cpj.due/cpj.P_j <= np.percentile(tight_all, 25)].tardiness > 0).mean():.1f}% |")

    # ------------------------------------------------------------------ comparisons and verdicts
    P_("\n### Paired comparisons (relative gain of X over Y = (mean Y - mean X) / mean Y; positive = X better)\n")
    P_("| rule | X | Y | gain | 95% CI | criterion | met |")
    P_("|---|---|---|---|---|---|---|")
    C = {}

    def comp(key, xa, xu, ya, yu, thr, rule, Yarr=None):
        X = sc(xa, xu); Y = Yarr if Yarr is not None else sc(ya, yu)
        need = [xa] + ([ya] if Yarr is None else [])
        if X is None or Y is None or any(status[k] != "complete" for k in need):
            C[key] = None
            P_(f"| {rule} | {xa}@{xu} | {ya}@{yu} | - | - | gain >= {100*thr:.0f}%, CI > 0 | NOT DETERMINED |"); return None
        g, lo, hi = gain(X, Y)
        met = g >= thr and lo > 0
        C[key] = dict(gain=g, lo=lo, hi=hi, met=bool(met))
        P_(f"| {rule} | {xa}@{xu} | {ya}@{yu} | {100*g:+.2f}% | [{100*lo:+.2f}, {100*hi:+.2f}] | gain >= {100*thr:.0f}%, CI > 0 | {'yes' if met else 'no'} |")
        return C[key]

    q1a = comp("q1a", "V2-M", 1000, "v1 C1 (Step 4)", 1000, 0.02, "Q1", Yarr=C1_1000)
    q1b = comp("q1b", "V2-M", 3000, "V1-CONT", 3000, 0.02, "Q1")
    q2 = comp("q2", "V2-M", 3000, "V2-M", 1000, 0.01, "Q2")
    q2c = comp("q2c", "V1-CONT", 3000, "V1-CONT", 1000, 0.01, "Q2 (report)")
    q3a = comp("q3a", "V2-M", 250, "V2-S", 250, 0.02, "Q3")
    q3b = comp("q3b", "V2-M", 500, "V2-S", 500, 0.02, "Q3")
    q3c = comp("q3c", "V2-S", 3000, "V2-M", 3000, 0.0, "Q3 (catch-up, report)")
    q4a = comp("q4a", "V2-T", 1000, "V2-M", 1000, 0.01, "Q4")
    q4b = comp("q4b", "V2-T", 3000, "V2-M", 3000, 0.01, "Q4")

    V = {}
    V["Q1"] = "NOT DETERMINED" if q1a is None or q1b is None else ("LIFTED" if q1a["met"] and q1b["met"] else "NOT LIFTED")
    V["Q2"] = "NOT DETERMINED" if q2 is None else ("BUDGET-LIMITED" if q2["met"] else "PLATEAU")
    V["Q2 (V1-CONT, reported)"] = "NOT DETERMINED" if q2c is None else ("BUDGET-LIMITED" if q2c["met"] else "PLATEAU")
    V["Q3"] = "NOT DETERMINED" if q3a is None or q3b is None else ("YES" if q3a["met"] and q3b["met"] else "NO")
    if q3c is not None:
        V["Q3 catch-up (V2-S@3000 within +-1% of V2-M@3000, CI includes 0)"] = (
            "CAUGHT UP" if abs(q3c["gain"]) <= 0.01 and q3c["lo"] <= 0 <= q3c["hi"] else "NOT CAUGHT UP")
    if status["V2-M"] == "complete" and status["V2-S"] == "complete":
        target = S["scores"]["V2-M"][str(U(1000, "V2-M"))]
        first = next((u for u in plan("V2-S", a.smoke)[0] if S["scores"]["V2-S"][str(u)] <= target), None)
        V["Q3 first V2-S checkpoint <= V2-M@1000"] = f"u{first}" if first is not None else "none up to 3000"
    for key, q in (("Q4 @1000", q4a), ("Q4 @3000", q4b)):
        V[key] = ("NOT DETERMINED" if q is None else "BETTER" if q["met"] else
                  "WORSE" if q["gain"] <= -0.01 and q["hi"] < 0 else "TIED")
    S["comparisons"], S["verdicts"] = C, V

    # ------------------------------------------------------------------ cost
    P_("\n### Cost (GPU wall-clock of training chunks; evaluation excluded)\n")
    P_("| arm | updates trained | hours | s/update |")
    P_("|---|---|---|---|")
    S["cost"] = {}
    for arm in ARMS:
        recs = []
        for s in seeds:
            f = res / "provenance" / f"{arm}_s{s}_chunks.jsonl"
            if f.exists():
                recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if recs:
            upd = sum(r["to"] - r["from"] for r in recs); sec = sum(r["wall_seconds"] for r in recs)
            S["cost"][arm] = dict(updates=upd, hours=sec / 3600)
            P_(f"| {LABEL[arm]} | {upd} | {sec/3600:.2f} | {sec/max(1, upd):.3f} |")

    P_("\n### Verdicts (exploratory, provisional until the 4-seed confirmation)\n")
    for k, v in V.items():
        P_(f"* **{k}: {v}**")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    col = {"V2-M": "tab:orange", "V2-S": "tab:gray", "V1-CONT": "tab:blue", "V2-T": "tab:green"}
    for arm in ARMS:
        pts = sorted((u, v) for u, v in ((int(k), v) for k, v in S["scores"].get(arm, {}).items()))
        if pts:
            ax.plot(*zip(*pts), "-o", ms=3.5, lw=1.5, color=col[arm], label=LABEL[arm] + ("" if status[arm] == "complete" else f" ({status[arm]})"))
    for (nm, pts), c in zip(v1curves.items(), ("tab:orange", "tab:gray")):
        ax.plot(list(pts), list(pts.values()), "--", lw=1.1, color=c, alpha=0.8, label=nm + " (0-1000)")
    ax.axhline(cp.mean(), color="k", lw=1.4, label=f"CP-SAT best, 60 s ({cp.mean():.3f})")
    ax.set_xscale("symlog", linthresh=50)
    ticks = [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000] if not a.smoke else ups
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("PPO updates (symlog, linthresh 50)"); ax.set_ylabel("Step 4 score, HELD-OUT (lower is better)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title(f"v2 re-run, seed{'s' if len(seeds) > 1 else ''} {', '.join(map(str, seeds))} (EXPLORATORY)", fontsize=10)
    fig.tight_layout(); fig.savefig(res / "score_vs_updates.png", dpi=130); plt.close(fig)
    rel = (res / "score_vs_updates.png").relative_to(report.parent) if not a.smoke else Path("results/score_vs_updates.png")
    P_(f"\n![score vs updates]({rel.as_posix()})\n")
    P_("*Solid: this study (mean over seeds). Dashed: the existing v1 Step 4 curves (0-1,000) for reference. Black: "
       "CP-SAT best (ceiling_check, 60 s).*")

    (res / "analysis_tables.md").write_text("\n".join(L) + "\n")
    (res / "summary.json").write_text(json.dumps(S, indent=1))
    txt = report.read_text()
    if BEGIN not in txt or END not in txt:
        raise SystemExit(f"report {report} lacks the GENERATED markers")
    head, rest = txt.split(BEGIN, 1); _, tail = rest.split(END, 1)
    report.write_text(head + BEGIN + "\n\n" + "\n".join(L) + "\n\n" + END + tail)
    print("\n".join(L))
    print("ANALYSIS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
