"""Analysis of the converged contrastive arms (docs/contrastive_converged_protocol_v3.md): every contrastive arm under the
PPO stopping rule (cap 8,000 updates), next to its earlier fixed-budget result (read-only from results/contrastive_v3 and
results/merge_categories_v3). Writes results/contrastive_converged_v3.md. Reuses tools/analyze_warmstart_v3.py unchanged
(loading, regret) and daniel/mo_metrics.py. Plain arm names. Single seed: numbers only, no significance claims.

    python tools/analyze_contrastive_converged_v3.py [--eval_split evaluation] [--results ...] [--out ...]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
_ARGV, sys.argv = sys.argv, sys.argv[:1]
import analyze_warmstart_v3 as AW                           # noqa: E402  (unchanged; its globals are pointed below)
sys.argv = _ARGV
MM, OBJ = AW.MM, AW.OBJ

MANIFEST = REPO / "configs" / "contrastive_converged_v3.json"
ARMS = ["CCB", "CCD", "CCE", "CCSBAVG", "CCTA", "CCTIES"]
OLD = {"CCB": ("contrastive_v3", "CLB"), "CCD": ("contrastive_v3", "CLD"), "CCE": ("contrastive_v3", "CLE"),
       "CCSBAVG": ("merge_categories_v3", "SBAVG"), "CCTA": ("merge_categories_v3", "TA"),
       "CCTIES": ("merge_categories_v3", "TIES")}
SEED = 111
AW.NAME_RE = re.compile(r"^(\d+x\d+)\+mix\+bv3_(?:cls?|mcs?|ccs?)_([A-Z0-9]+)_s(\d+)$")
AW.ARMS = ARMS + [o for _, o in OLD.values()]

DESIGN = [
    "Stopping rule: the PPO warm-start rule (daniel/convergence.py), checked at every validation (every 50 updates): "
    "stop once the best validation HV has improved by less than 0.5% (relative) over the last 1,000 updates, after at "
    "least 1,000 updates. Hard cap 8,000 updates; a run that reaches it is reported as capped, not converged.",
    "Step 1: the rule was replayed on each fixed-budget run's logged validation history before any run (frozen in "
    "configs/contrastive_converged_v3.json). If it never stopped the history, the arm continues from its saved final "
    "state; if it would have stopped it earlier, the arm reruns from its saved starting network.",
    "A continued run loads the saved weights, optimiser, RNG states, logs, best checkpoint and generator counters, and "
    "may differ from its source only in the model name and the stopping rule (otherwise train.py refuses). Before "
    "continuing, the resumed policy must reproduce the last logged validation HV exactly; otherwise the arm reruns from "
    "its starting network (exit code 3, recorded).",
    "Two fixed-budget runs ended between two instance resamplings (update 3,350; resampling every 20 updates), and their "
    "resume states predate the saved instance batch: the batch is rebuilt by replaying the instance generator from the "
    "seed, accepted only if the replay ends in exactly the saved numpy state and generator counters. Resume states now "
    "also store the batch.",
    "Wall-clock and environment steps of a continued run include its fixed-budget part (same machine, --jobs 3); a "
    "rerun counts from zero.",
    "Contrastive settings unchanged (configs/contrastive_v3.json); validation, checkpoint selection and evaluation as "
    "before.",
    "Regret is computed over the converged and the fixed-budget models together (plus the specialists, the rules and "
    "CP-SAT B1/B2), so its values differ from the earlier reports; HV, IGD+ and unique non-dominated counts of the "
    "fixed-budget models are the earlier values.",
]


def f3(v, fmt="{:.3f}"):
    return "-" if v is None or not np.isfinite(v) else fmt.format(v)


def load(res, split):
    AW.RES = Path(res)
    return AW.load(split)


def facts(r):
    """Per-run training facts from its logs."""
    u, hv, sec, _ = AW.val_series(r)
    st = r["status"] or {}
    steps = [e["cl"]["env_steps"] for e in (r["reward"] or []) if "cl" in e]
    b = int(np.argmax(hv))
    stop = st.get("update", u[-1])
    return {"best_u": u[b], "best": hv[b], "stop": stop, "status": st.get("status"),
            "steps_best": float(sum(steps[:u[b]])), "steps_stop": float(sum(steps[:stop])),
            "h_best": sec[b] / 3600, "h_stop": st.get("train_seconds", sec[-1]) / 3600, "u": u, "hv": hv}


def reason(status):
    return {"CONVERGED": "converged", "NOT_CONVERGED": "capped"}.get(status, status or "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_split", default="evaluation")
    ap.add_argument("--norm", nargs="+", default=[str(AW.BRES / "normalisation_points.json")])
    ap.add_argument("--results", default=str(REPO / "results" / "contrastive_converged_v3"))
    ap.add_argument("--out", default=str(REPO / "results" / "contrastive_converged_v3.md"))
    a = ap.parse_args()
    res = Path(a.results)
    man = json.loads(MANIFEST.read_text())
    NAME = {k: v["name"] for k, v in man["arms"].items()}
    norm = {}
    for f in a.norm:
        norm.update(json.loads(Path(f).read_text())["instances"])
    pick = lambda d, keep: {v["arm"]: v for v in d.values() if v["seed"] == SEED and v["arm"] in keep}
    new_runs, new_ev = load(res, a.eval_split)
    R, E = pick(new_runs, ARMS), pick(new_ev, ARMS)
    OR, OE = {}, {}
    for study in ("contrastive_v3", "merge_categories_v3"):
        runs, ev = load(REPO / "results" / study, a.eval_split) if a.eval_split == "evaluation" else ({}, {})
        keep = [o for s, o in OLD.values() if s == study]
        OR.update(pick(runs, keep)); OE.update(pick(ev, keep))
    AW.RES = res
    plan = json.loads((res / "plan.json").read_text()) if (res / "plan.json").exists() else {}
    rule = man["rule"]

    L = ["# Contrastive fine-tuning to convergence on benchmark v3", "",
         "Generated by `tools/analyze_contrastive_converged_v3.py` (docs/contrastive_converged_protocol_v3.md). Every "
         "contrastive arm judged at convergence under the PPO warm-start stopping rule, next to its earlier "
         "fixed-budget result (results/contrastive_v3.md, results/merge_categories_v3.md; unchanged).", "",
         f"**Stopping rule:** {rule['statement']}.", "",
         "**Single seed (111): numbers only, no significance claims.** Higher HV is better; lower IGD+ and regret are "
         f"better. Evaluation split: `{a.eval_split}`.", "",
         f"Hardware: {plan.get('host', {}).get('gpu', '?')}, {plan.get('host', {}).get('logical_cpus', '?')} logical CPUs, "
         f"--jobs {plan.get('jobs', '?')}.", ""]

    # ---------------- step 1
    L += ["## Step 1: replay of the rule on the fixed-budget histories", "",
          "| Arm | Fixed budget (updates) | Rule would stop at | Mode | Exactness check (last logged vs reproduced validation HV) |",
          "|---|---|---|---|---|"]
    for x in ARMS:
        src = man["arms"][x]
        rdir = res / "runs" / f"10x5+mix+bv3_{'ccs' if res.name.endswith('_smoke') else 'cc'}_{x}_s{SEED}"
        mode = json.loads((rdir / "mode.json").read_text()) if (rdir / "mode.json").exists() else {}
        chk = json.loads((rdir / "extension_check.json").read_text()) if (rdir / "extension_check.json").exists() else None
        m = mode.get("mode", src["mode"])
        mtxt = (f"continued from update {src['fixed_budget']}" if m == "continue" else "rerun from the starting network")
        if mode.get("fallback"):
            mtxt += f" ({mode['fallback']})"
        ctxt = ("-" if chk is None else f"{chk['logged_hv']:.10f} vs {chk['reproduced_hv']:.10f} at update "
                f"{chk['logged_update']}: {'exact' if chk['exact'] else 'NOT exact'}")
        L.append(f"| {NAME[x]} | {src['fixed_budget']} | {src['replay_stop_update'] or 'never'} | {mtxt} | {ctxt} |")
    L.append("")

    # ---------------- training to convergence
    L += ["## Training to convergence (validation split, 15 preferences)", "",
          "| Arm | Update of best | Stop update | Stop reason | Best validation HV | Env steps to best (M) | Env steps to "
          "stop (M) | Wall-clock to best (h) | Wall-clock to stop (h) |", "|---|---|---|---|---|---|---|---|---|"]
    for x in ARMS:
        if x not in R or not R[x]["valiquality"]:
            L.append(f"| {NAME[x]} | - | - | - | - | - | - | - | - |"); continue
        t = facts(R[x])
        L.append(f"| {NAME[x]} | {t['best_u']} | {t['stop']} | {reason(t['status'])} | {t['best']:.3f} | "
                 f"{t['steps_best'] / 1e6:.2f} | {t['steps_stop'] / 1e6:.2f} | {t['h_best']:.2f} | {t['h_stop']:.2f} |")
    L += ["", "Continued runs count their fixed-budget part in steps and wall-clock; reruns count from zero.", ""]
    rr = []
    for x in ARMS:
        o = OLD[x][1]
        if x in R and o in OR and man["arms"][x]["mode"] == "rerun" and R[x]["valiquality"] and OR[o]["valiquality"]:
            fn, fo = facts(R[x]), facts(OR[o])
            new, old = dict(zip(fn["u"], fn["hv"])), dict(zip(fo["u"], fo["hv"]))
            common = sorted(set(new) & set(old))
            d = max(abs(new[u] - old[u]) for u in common) if common else float("nan")
            rr.append(f"{NAME[x]}: {len(common)} common validations, largest |ΔHV| {d:.2e}; stopped at "
                      f"{facts(R[x])['stop']} (replay predicted {man['arms'][x]['replay_stop_update']})")
    if rr:
        L += ["Reruns vs their fixed-budget runs (same seed and start): " + "; ".join(rr) + ".", ""]

    # ---------------- evaluation, converged vs fixed budget
    every = {**{x: E[x] for x in E}, **{o: OE[o] for o in OE}}
    any_ev = next((v for v in every.values() if v["npz"] is not None), None)
    keys = [str(q) for q in any_ev["npz"]["keys"]] if any_ev else []
    keys = keys if keys and all(q in norm for q in keys) else []
    if keys:
        size = "10x5"
        grid = MM.simplex_grid(AW._grid_step(int(any_ev["json"]["grid_size"])))
        b1, b2 = AW.load_cpsat("b1", a.eval_split, size), AW.load_cpsat("b2", a.eval_split, size)
        sub = [q for q in keys if q in b2]
        sev = {x: v for x, v in every.items() if v["npz"] is not None and [str(q) for q in v["npz"]["keys"]] == keys}
        reg = AW.regret_table(sev, keys, norm, grid, a.eval_split, size, b1, b2) if sev else None
        L += [f"## Evaluation ({a.eval_split} split, greedy, {len(grid)} preferences, {len(keys)} instances; IGD+ on the "
              f"{len(sub)}-instance CP-SAT B2 subset)", "",
              "| Arm | Training | Updates (stop) | Update of best | Best validation HV | Evaluation HV | IGD+ | "
              "Unique non-dominated | Mean regret |", "|---|---|---|---|---|---|---|---|---|"]
        for x in ARMS:
            for tag, run, ev in (("fixed budget", OR.get(OLD[x][1]), OE.get(OLD[x][1])),
                                 ("to convergence", R.get(x), E.get(x))):
                k = OLD[x][1] if tag == "fixed budget" else x
                if run is None or not run["valiquality"]:
                    L.append(f"| {NAME[x]} | {tag} | - | - | - | - | - | - | - |"); continue
                t = facts(run)
                stop = f"{t['stop']}" + (f" ({reason(t['status'])})" if tag == "to convergence" else "")
                if ev is None or ev["npz"] is None or [str(q) for q in ev["npz"]["keys"]] != keys:
                    L.append(f"| {NAME[x]} | {tag} | {stop} | {t['best_u']} | {t['best']:.3f} | - | - | - | - |"); continue
                L.append(f"| {NAME[x]} | {tag} | {stop} | {t['best_u']} | {t['best']:.3f} | "
                         f"{np.asarray(ev['npz']['hv']).mean():.3f} | {f3(ev['json']['igd_plus_mean'])} | "
                         f"{ev['json']['nd_size_mean']:.1f} | {f3(reg[0][k]['all']) if reg and k in reg[0] else '-'} |")
        L.append("")
    else:
        L += ["## Evaluation", "", "_No evaluation results for this split yet._", ""]

    # ---------------- setup cost
    L += ["## Setup cost (listed separately; not in the wall-clock above)", "",
          "| Arm | Setup | Cost (h, this machine) |", "|---|---|---|"]
    dist = REPO / man["setup_costs"]["distillation"]["source"]
    d_h = json.loads(dist.read_text())["total_seconds"] / 3600 if dist.exists() else None
    sb = REPO / man["setup_costs"]["shared_base"]["source"]
    sb_h = (sum(json.loads((sb / f"{t}.status.json").read_text())["train_seconds"] for t in ("base", "ft_m", "ft_c", "ft_t"))
            / 3600 if (sb / "base.status.json").exists() else None)
    spec = [REPO / "results" / "baselines_v3" / "runs" / f"10x5+mix+bv3_spec_{o}_s111" / "status.json" for o in "mct"]
    sp_h = sum(json.loads(f.read_text())["train_seconds"] for f in spec) / 3600 if all(f.exists() for f in spec) else None
    L += ["| From scratch | none | 0.00 |",
          f"| Distillation | distillation of the student (states and training) | {f3(d_h, '{:.2f}')} |",
          "| Naive weight average (independent specialists) | uniform average of three specialists | ~0 |",
          *(f"| {n} | shared base and M, C, T fine-tunes (PPO), shared by the three | {f3(sb_h, '{:.2f}')} |"
            for n in ("Model Soup (shared base)", "Task Arithmetic", "TIES")),
          "", f"Distillation and Naive weight average also use the independently trained seed-111 specialists "
          f"(M, C, T: {f3(sp_h, '{:.2f}')} h of PPO in total, trained earlier for the baselines).", ""]

    L += ["## Design choices", ""] + [f"{i}. {d}" for i, d in enumerate(DESIGN, 1)] + [""]
    L += ["## Frozen inputs", "", "```json", json.dumps(man, indent=1), "```", ""]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
