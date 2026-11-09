"""Analysis of the merge-failure test (docs/merge_failure_test_protocol_v3.md): Part A, update-0 diagnostics of six
starting networks; Part B, three one-change contrastive arms from the Model Soup start next to the converged Model Soup
and From scratch runs (read-only from results/contrastive_converged_v3). Writes results/merge_failure_test_v3.md. Reuses
tools/analyze_warmstart_v3.py and tools/analyze_contrastive_converged_v3.py unchanged. Plain arm names. Single seed:
numbers only, no significance claims.

    python tools/analyze_merge_failure_test_v3.py [--eval_split evaluation] [--results ...] [--out ...]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
_ARGV, sys.argv = sys.argv, sys.argv[:1]
import analyze_warmstart_v3 as AW                           # noqa: E402
import analyze_contrastive_converged_v3 as AC               # noqa: E402  (facts, reason)
sys.argv = _ARGV

MANIFEST = REPO / "configs" / "merge_failure_test_v3.json"
ARMS = ["MFSP", "MFNA", "MFPR"]
CTRL = ["CCSBAVG", "CCB"]
DIAG = ["SOUP", "DIST", "RAND", "MFSP", "MFNA", "MFPR"]
SEED = 111
AW.NAME_RE = re.compile(r"^(\d+x\d+)\+mix\+bv3_(?:ccs?|mfs?)_([A-Z0-9]+)_s(\d+)$")
AW.ARMS = ARMS + CTRL

DESIGN = [
    "Every arm is the converged Model Soup run with exactly one change (its flags in the manifest): the same start "
    "network, seed 111, contrastive settings (K 8, beta 0.1, margin 0.01, clip 1.0, lr 3e-4, 20 environments), "
    "instance and preference streams, validation and checkpoint selection, and the same stopping rule. None of the "
    "changes draws random numbers, so every random stream is that of the Model Soup run.",
    "Shrink and perturb (Ash & Adams, NeurIPS 2020, Section 4): theta <- 0.3 theta + 0.0001 theta_fresh, where "
    "theta_fresh is the freshly initialised network of seed 111 (the From-scratch initialisation). The paper adds noise "
    "'by adding parameters from a scaled, randomly-initialized network'; 0.3 and 0.0001 are its values for the "
    "pre-training-then-fine-tuning case (Section 4.2, Figure 9). Every parameter is shrunk, including the critic, which "
    "the contrastive objective never uses. The contrastive reference is the shrunk-and-perturbed network (the network "
    "fine-tuning starts from).",
    "No anchor: the loss -log sigmoid(beta (l_better - l_worse)), the DPO loss with the reference log-probabilities set "
    "to zero. The start network is the Model Soup start, so its update-0 validation HV equals Model Soup's.",
    "Random preference input: the preference-input columns of the first actor and critic layers keep the fresh seed-111 "
    "values (as in From scratch) instead of zero; every other weight is the merge's.",
    "Part A uses no training: each network is built by train.py's own code with the arm's flags (same seed and "
    "transfer), measured on the validation split (all instances, the 15 validation preferences), and saved (starts/). "
    "The gradient is one contrastive gradient on the run's first batch (same instances and preferences for every "
    "network; schedules sampled from the network), before clipping; the weights are discarded afterwards.",
    "Near-inactive unit: |activation| <= 0.01 on at least 99% of the entries of the validation batch (every state of "
    "the greedy validation rollouts, unfinished schedules only; attention-block outputs are ELU, actor hidden layers "
    "tanh). Saturated tanh unit: |activation| >= 0.99 on at least 99% of those entries.",
    "Closing the gap: an arm closes the gap if its evaluation HV is at least From scratch's converged evaluation HV; "
    "otherwise the share closed is (arm - Model Soup) / (From scratch - Model Soup) on evaluation HV.",
    "Wall-clock: same machine and --jobs 3 for all runs. From scratch was continued from its fixed-budget run, so its "
    "wall-clock includes that part; Model Soup and the new arms count from zero.",
]


def f3(v, fmt="{:.3f}"):
    return "-" if v is None or not np.isfinite(v) else fmt.format(v)


def short(layer):
    return (layer.replace("feature_exact.", "").replace("op_attention_blocks.", "op-attn ")
            .replace("mch_attention_blocks.", "mch-attn ").replace("linears.", ""))


def part_a(res, man):
    ddir = res / "diagnostics"
    D = {k: json.loads((ddir / f"{k}.json").read_text()) for k in DIAG if (ddir / f"{k}.json").exists()}
    if not D:
        return ["## Part A: update-0 measurements (no training)", "", "_No diagnostics yet._", ""], D
    name = {k: man["diagnostics"]["subjects"][k]["name"] for k in DIAG}
    first = next(iter(D.values()))
    L = ["## Part A: update-0 measurements (no training)", "",
         f"Validation split: {first['validation']['instances']} instances x {first['validation']['preferences']} "
         f"preferences; K = {first['sampled']['k']} sampled schedules per (instance, preference). Rows 4-6 are the "
         "starting networks of the Part B arms.", "",
         "| Network | Validation HV | Policy entropy | Distinct schedules (of 8) | Contrastive loss | Gradient norm "
         "(pre-clip) | Greedy action changes with w (% of states) | Mean TV between preferences | Distinct greedy "
         "schedules (of 15) |", "|---|---|---|---|---|---|---|---|---|"]
    for k in DIAG:
        if k not in D:
            continue
        d = D[k]; p = d["preference_sensitivity"]; g = d["gradient"]
        L.append(f"| {name[k]} | {d['validation']['hv_update0']:.3f} | {d['sampled']['entropy']:.3f} | "
                 f"{d['sampled']['distinct_schedules_of_k']:.2f} | {g['loss']:.4f} | {g['total']:.4f} | "
                 f"{100 * p['share_greedy_action_changes']:.1f} | {p['mean_tv_between_preferences']:.4f} | "
                 f"{p['distinct_greedy_schedules_of_15']:.2f} |")
    L += ["", "Entropy: mean per-step policy entropy along the sampled schedules (nats). The contrastive loss is log 2 = "
          "0.6931 for every anchored network at update 0 (policy = reference); the gradient norm is that of the run's "
          "first update. Preference sensitivity: on every state of the greedy validation rollouts, the policy under "
          "each of the 15 preferences.", ""]
    layers = list(first["activations"])
    L += ["### Near-inactive units per layer (% of units)", "",
          "| Network | " + " | ".join(short(x) for x in layers) + " | actor tanh saturated (layer 0 / 1) |",
          "|---|" + "---|" * (len(layers) + 1)]
    for k in DIAG:
        if k in D:
            a = D[k]["activations"]
            sat = " / ".join(f"{100 * a[x]['saturated']:.1f}" for x in layers if "saturated" in a[x])
            L.append(f"| {name[k]} | " + " | ".join(f"{100 * a[x]['near_inactive']:.1f}" for x in layers) + f" | {sat} |")
    L += ["", "Units per layer: " + ", ".join(f"{short(x)} {first['activations'][x]['units']}" for x in layers) + ".", ""]
    wl = list(first["weight_norms"]["per_layer"])
    L += ["### Weight norm per layer (Frobenius norm of all its parameters)", "",
          "| Network | " + " | ".join(short(x) for x in wl) + " | total | preference columns (actor / critic) |",
          "|---|" + "---|" * (len(wl) + 2)]
    for k in DIAG:
        if k in D:
            w = D[k]["weight_norms"]; pc = w["preference_columns"]
            L.append(f"| {name[k]} | " + " | ".join(f"{w['per_layer'][x]:.2f}" for x in wl) + f" | {w['total']:.2f} | "
                     f"{pc['actor.linears.0.weight']:.3g} / {pc['critic.linears.0.weight']:.3g} |")
    gl = list(first["gradient"]["per_layer"])
    L += ["", "### Gradient norm of the contrastive loss per layer (one batch, before clipping at 1.0)", "",
          "| Network | " + " | ".join(short(x) for x in gl) + " | total | pairs | groups with a pair |",
          "|---|" + "---|" * (len(gl) + 3)]
    for k in DIAG:
        if k in D:
            g = D[k]["gradient"]
            L.append(f"| {name[k]} | " + " | ".join(f"{g['per_layer'].get(x, 0.0):.4f}" for x in gl) + f" | "
                     f"{g['total']:.4f} | {g['pairs']} | {100 * g['groups_with_pair']:.0f}% |")
    L += ["", "The critic receives no gradient (the contrastive objective does not use it).", ""]
    return L, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_split", default="evaluation")
    ap.add_argument("--norm", nargs="+", default=[str(AW.BRES / "normalisation_points.json")])
    ap.add_argument("--results", default=str(REPO / "results" / "merge_failure_test_v3"))
    ap.add_argument("--out", default=str(REPO / "results" / "merge_failure_test_v3.md"))
    a = ap.parse_args()
    res = Path(a.results)
    man = json.loads(MANIFEST.read_text())
    NAME = {**{k: v["name"] for k, v in man["arms"].items()}, **{k: v["name"] for k, v in man["controls"].items()}}
    norm = {}
    for f in a.norm:
        norm.update(json.loads(Path(f).read_text())["instances"])
    pick = lambda d, keep: {v["arm"]: v for v in d.values() if v["seed"] == SEED and v["arm"] in keep}
    runs, ev = AC.load(res, a.eval_split)
    R, E = pick(runs, ARMS), pick(ev, ARMS)
    if a.eval_split == "evaluation":
        cr, ce = AC.load(REPO / "results" / "contrastive_converged_v3", a.eval_split)
        R.update(pick(cr, CTRL)); E.update(pick(ce, CTRL))
    AW.RES = res
    plan = json.loads((res / "plan.json").read_text()) if (res / "plan.json").exists() else {}

    L = ["# Why merged starts end below random initialisation: merge-failure test on benchmark v3", "",
         "Generated by `tools/analyze_merge_failure_test_v3.py` (docs/merge_failure_test_protocol_v3.md). Test case: the "
         "Model Soup (shared base) start, 10x5, seed 111, contrastive fine-tuning under the converged runs' stopping "
         "rule. Model Soup and From scratch are the converged runs (results/contrastive_converged_v3.md; unchanged).", "",
         f"**Stopping rule:** {man['rule']['statement']}.", "",
         "**Single seed (111): numbers only, no significance claims.** Higher HV is better; lower IGD+ is better. "
         f"Evaluation split: `{a.eval_split}`.", "",
         f"Hardware: {plan.get('host', {}).get('gpu', '?')}, {plan.get('host', {}).get('logical_cpus', '?')} logical CPUs, "
         f"--jobs {plan.get('jobs', '?')}.", ""]
    A, D = part_a(res, man)
    L += A

    # ---------------- part B
    L += ["## Part B: one-change arms from the Model Soup start", "",
          "| Arm | Change | HV at update 0 | Update of best | Best validation HV | Stop update | Evaluation HV | IGD+ | "
          "Unique non-dominated | Wall-clock to best / stop (h) | Closed the gap to From scratch? |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    evhv = {}
    for x in ARMS + CTRL:
        e = E.get(x)
        if e is not None and e["npz"] is not None:
            evhv[x] = float(np.asarray(e["npz"]["hv"]).mean())
    soup, scratch = evhv.get("CCSBAVG"), evhv.get("CCB")
    change = {"MFSP": "weights x 0.3 + 0.0001 x fresh network", "MFNA": "reference term removed from the loss",
              "MFPR": "preference columns fresh, not zero", "CCSBAVG": "none (control)",
              "CCB": "random initialisation (reference)"}
    for x in ARMS + ["CCSBAVG", "CCB"]:
        r = R.get(x)
        if r is None or not r["valiquality"]:
            L.append(f"| {NAME[x]} | {change[x]} | - | - | - | - | - | - | - | - | - |"); continue
        t = AC.facts(r)
        e = E.get(x)
        if x in ARMS:
            if x in evhv and scratch is not None and soup is not None:
                gap = ("yes" if evhv[x] >= scratch else
                       f"no ({100 * (evhv[x] - soup) / (scratch - soup):.0f}% of the gap closed)")
            else:
                gap = "-"
        else:
            gap = "(control)" if x == "CCSBAVG" else "(reference)"
        L.append(f"| {NAME[x]} | {change[x]} | {t['hv'][0]:.3f} | {t['best_u']} | {t['best']:.3f} | {t['stop']} "
                 f"({AC.reason(t['status'])}) | {f3(evhv.get(x))} | "
                 f"{f3(e['json']['igd_plus_mean']) if e else '-'} | {f3(e['json']['nd_size_mean'], '{:.1f}') if e else '-'} | "
                 f"{t['h_best']:.2f} / {t['h_stop']:.2f} | {gap} |")
    L += ["", "HV at update 0, best and stop: validation split, 15 preferences. Evaluation: 100 instances, 66 preferences, "
          "greedy; IGD+ against CP-SAT B2 on the same 20 instances as every earlier report.", ""]
    chk = []
    tag = "mfs" if res.name.endswith("_smoke") else "mf"
    for x in ARMS:
        f = res / "runs" / f"10x5+mix+bv3_{tag}_{x}_s{SEED}" / "reference.json"
        if f.exists() and x in D:
            ok = json.loads(f.read_text())["cl_reference_sha256"] == D[x]["start_sha256"]
            chk.append(f"{NAME[x]}: {'match' if ok else 'MISMATCH'}")
    if chk:
        L += ["Saved start networks (Part A, starts/) vs each run's contrastive reference network (SHA-256 of the "
              "weights): " + "; ".join(chk) + ".", ""]
    if "MFNA" in R and "CCSBAVG" in R and R["MFNA"]["valiquality"] and R["CCSBAVG"]["valiquality"]:
        a0, b0 = AC.facts(R["MFNA"])["hv"][0], AC.facts(R["CCSBAVG"])["hv"][0]
        L += [f"No anchor starts from the Model Soup network: update-0 validation HV {a0:.10f} vs {b0:.10f} "
              f"({'identical' if a0 == b0 else 'DIFFERENT'}).", ""]

    L += ["## Arms and design choices", ""] + [f"{i}. {d}" for i, d in enumerate(DESIGN, 1)] + [""]
    L += ["## Frozen inputs", "", "```json", json.dumps(man, indent=1), "```", ""]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
