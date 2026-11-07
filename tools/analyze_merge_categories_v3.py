"""Analysis of the merge-category arms (docs/merge_categories_protocol_v3.md): SB-AVG, Task Arithmetic and TIES, each as a
zero-training merge and after contrastive fine-tuning, next to the seed-111 contrastive arms CL-B, CL-D and CL-E
(read-only from results/contrastive_v3) and E's zero-training merge. Writes results/merge_categories_v3.md and figures
in results/merge_categories_v3/figures/. Reuses tools/analyze_warmstart_v3.py unchanged (loading, regret, references)
and daniel/mo_metrics.py. Single seed: numbers only, no significance claims.

    python tools/analyze_merge_categories_v3.py [--eval_split evaluation] [--results results/merge_categories_v3] [--out ...]
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
import matplotlib.pyplot as plt                              # noqa: E402
MM, OBJ = AW.MM, AW.OBJ

CL_RES = REPO / "results" / "contrastive_v3"
MANIFEST = REPO / "configs" / "merge_categories_v3.json"
SEED = 111
MC = ["SBAVG", "TA", "TIES"]
CL = ["CLB", "CLD", "CLE"]
ZERO = {"SBAVG": "ZSBAVG", "TA": "ZTA", "TIES": "ZTIES", "CLE": "ZE"}
SHORT = {"SBAVG": "SB-AVG", "TA": "TA", "TIES": "TIES", "CLB": "CL-B", "CLD": "CL-D", "CLE": "CL-E",
         "ZSBAVG": "SB-AVG merge (zero training)", "ZTA": "TA merge (zero training)",
         "ZTIES": "TIES merge (zero training)", "ZE": "E merge (zero training)"}
LABEL = {"SBAVG": "SB-AVG + contrastive", "TA": "TA + contrastive", "TIES": "TIES + contrastive",
         "CLB": "CL-B (contrastive, scratch)", "CLD": "CL-D (contrastive from the distilled student)",
         "CLE": "CL-E (contrastive from E's naive merge)", **{v: SHORT[v] for v in ZERO.values()}}
START = {"SBAVG": "shared-base average", "TA": "Task Arithmetic merge", "TIES": "TIES merge",
         "CLB": "random initialisation", "CLD": "distilled student", "CLE": "naive merge (independent specialists)"}
AW.NAME_RE = re.compile(r"^(\d+x\d+)\+mix\+bv3_(?:cls?|mcs?)_([A-Z0-9]+)_s(\d+)$")
AW.ARMS = MC + CL + list(ZERO.values())
AW.LABEL = {**AW.LABEL, **LABEL}

DESIGN = [
    "Shared base: the pre-registered E-shared path (`tools/ec2_runner.py: u_shared_base`, unchanged): an unconditioned "
    "DANIEL (the specialists' architecture and schema) trained with PPO on the equal-weight `mct` objective for 500 "
    "updates; its best-validation checkpoint is the base. Chosen because it has exactly the specialists' architecture "
    "and sits at the equal-weight middle of the three objectives, so each task vector points from the middle toward one "
    "corner; a random-initialisation base would make the task vectors whole networks.",
    "Specialists: M, C and T fine-tuned from that base with the specialist training settings (PPO, configs "
    "benchmark_v3/{m,c,t}.json, seed 111, generator stream) for a fixed 1,000 updates each (the E-shared budget); "
    "best-validation checkpoints. They are new specialists, not the committed checkpoints/specialists_v3 (which were "
    "trained independently from random initialisations and stopped by the convergence rule).",
    "SB-AVG = (θ_M + θ_C + θ_T) / 3 (`merge_specialists_v3.py --shared_base`); it equals TA with λ = 1/3.",
    "TA = θ_base + λ Σ_x τ_x with τ_x = θ_x − θ_base.",
    "TIES: each τ_x is trimmed to its top 20% entries by |value| over the whole flattened vector (all tensors "
    "together, as in the official implementation), the rest set to 0; per entry the elected sign is the sign of the "
    "sum of the trimmed values; the merged value is the mean over the tasks whose kept value is non-zero and has the "
    "elected sign (0 if none); result θ_base + λ τ_TIES.",
    "λ ∈ {0.3, 0.5, 0.7, 1.0} for TA and TIES separately, chosen by update-0 validation HV (the CL arms' 15-preference "
    "validation grid and frozen validation normalisation points); tie → smaller λ. Evaluation data never used.",
    "Preference input and critic as arm E: every tensor (extractor, actor, critic) is merged by the same rule into an "
    "unconditioned v3 bundle; `--init_mode full` copies it into the preference-conditioned network with the three "
    "preference columns of the first actor and critic layers set to zero, so the update-0 network ignores w. The "
    "contrastive objective never uses or trains the critic.",
    "Fine-tuning: the existing contrastive option unchanged (configs/contrastive_v3.json constants: K 8, β 0.1, margin "
    "0.01, gradient clip 1.0, lr 3e-4); reference = the start network, frozen; 3,350 updates (E's final update, seed "
    "111), no early stopping; validation at update 0 and every 50 updates; best-validation-HV checkpoint.",
    "Zero-training value: a 0-update contrastive run from the merge (validation at update 0, network saved) evaluated "
    "like the fine-tuned arms; for E the merge is regenerated from the committed seed-111 specialists (check: its "
    "update-0 validation HV equals CL-E's).",
    "Every constant was fixed in configs/merge_categories_v3.json before any run; no hyperparameter was tuned.",
]


def f3(v):
    return "-" if v is None or not np.isfinite(v) else f"{v:.3f}"


def cl_log(r):
    return [e for e in (r["reward"] or []) if "cl" in e]


def summary(r):
    u, hv, sec, _ = AW.val_series(r)
    if not u:
        return None
    b = int(np.argmax(hv))
    lg = cl_log(r)
    steps = [e["cl"]["env_steps"] for e in lg]
    st = r["status"] or {}
    return {"u0": hv[0] if u[0] == 0 else None, "best": hv[b], "best_u": u[b], "best_h": sec[b] / 3600,
            "best_steps": float(sum(steps[:u[b]])) if len(steps) >= u[b] else None,
            "final": hv[-1], "final_u": u[-1], "train_h": st.get("train_seconds", sec[-1]) / 3600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_split", default="evaluation")
    ap.add_argument("--norm", nargs="+", default=[str(AW.BRES / "normalisation_points.json")])
    ap.add_argument("--results", default=str(REPO / "results" / "merge_categories_v3"))
    ap.add_argument("--out", default=str(REPO / "results" / "merge_categories_v3.md"))
    a = ap.parse_args()
    res = Path(a.results); fig_dir = res / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    norm = {}
    for f in a.norm:
        norm.update(json.loads(Path(f).read_text())["instances"])
    AW.RES = res; mc_runs, mc_ev = AW.load(a.eval_split)
    AW.RES = CL_RES; cl_runs, cl_ev = AW.load(a.eval_split) if a.eval_split == "evaluation" else ({}, {})
    AW.RES = res
    pick = lambda d: {v["arm"]: v for v in d.values() if v["seed"] == SEED and v["arm"] in AW.ARMS}
    R = {**pick(cl_runs), **pick(mc_runs)}
    E = {**pick(cl_ev), **pick(mc_ev)}
    zero_val = {}
    for d in sorted((res / "zero").glob("*")) if (res / "zero").exists() else []:
        m = AW.NAME_RE.match(d.name)
        if m and (d / "valiquality.json").exists():
            zero_val[m.group(2)] = json.loads((d / "valiquality.json").read_text())[0]["hv"]
    man = json.loads(MANIFEST.read_text())
    sel_f = res / "lambda_selection.json"
    sel = json.loads(sel_f.read_text()) if sel_f.exists() else None
    plan = json.loads((res / "plan.json").read_text()) if (res / "plan.json").exists() else {}

    L = ["# Merge categories on benchmark v3: shared-base average, Task Arithmetic and TIES, with contrastive fine-tuning",
         "", "Generated by `tools/analyze_merge_categories_v3.py` (docs/merge_categories_protocol_v3.md). Three merges of "
         "the same three specialists, all fine-tuned from one shared base, each evaluated with zero training and after "
         "contrastive fine-tuning (no PPO fine-tuning in this study). Compared with the seed-111 contrastive arms of "
         "results/contrastive_v3.md.", "",
         "**Single seed (111): numbers only, no significance claims.** Higher HV is better; lower IGD+ and regret are "
         f"better. Evaluation split: `{a.eval_split}`.", "",
         f"Hardware: {plan.get('host', {}).get('gpu', '?')}, {plan.get('host', {}).get('logical_cpus', '?')} logical CPUs, "
         f"--jobs {plan.get('jobs', '?')} (the CL arms ran on the same machine with --jobs 3).", ""]

    # ---------------- shared base
    L += ["## Shared base and specialists (PPO specialist training; not arms)", "",
          "| Network | Init | Objective | Updates | Best validation value (lower is better) | Update of best | Training (h) |",
          "|---|---|---|---|---|---|---|"]
    sbdir = res / "shared_base" / f"10x5_s{SEED}"
    total_h = 0.0
    for tag, init, goal in (("base", "random (seed 111)", "mct (equal weights)"), ("ft_m", "base", "makespan"),
                            ("ft_c", "base", "carbon"), ("ft_t", "base", "tardiness")):
        f = sbdir / f"{tag}.status.json"
        if not f.exists():
            L.append(f"| {tag} | {init} | {goal} | - | - | - | - |"); continue
        s = json.loads(f.read_text()); total_h += s["train_seconds"] / 3600
        L.append(f"| {tag} | {init} | {goal} | {s['update']} | {s['best_value']:.4f} | {s['best_update']} | "
                 f"{s['train_seconds'] / 3600:.2f} |")
    L += ["", f"Shared-base preparation cost: {total_h:.2f} h of PPO training in total (shared by SB-AVG, TA and TIES).", ""]

    # ---------------- lambda
    L += ["## λ selection (validation split only)", ""]
    if sel:
        L += ["| Merge | " + " | ".join(f"λ = {l:g}" for l in sel["lambda_grid"]) + " | Selected |",
              "|---|" + "---|" * (len(sel["lambda_grid"]) + 1)]
        for arm in ("TA", "TIES"):
            c = {x["lambda"]: x["validation_hv_update0"] for x in sel["candidates"].get(arm, [])}
            L.append(f"| {arm} | " + " | ".join(f3(c.get(l)) for l in sel["lambda_grid"]) +
                     f" | λ = {sel['selected'][arm]['lambda']:g} |")
        L += ["", f"Update-0 validation HV (15 preferences); rule: {sel['rule']}. TIES keeps {sel['ties_keep']:.0%} of each "
              "task vector.", ""]
    else:
        L += ["_Not run yet._", ""]

    # ---------------- raised / lowered
    L += ["## Did contrastive fine-tuning raise or lower validation HV relative to update 0?", ""]
    for x in MC + ["CLE"]:
        s = summary(R[x]) if x in R else None
        if s is None or s["u0"] is None:
            L.append(f"- **{SHORT[x]}**: no validation curve yet."); continue
        verdict = ("never raised it (best checkpoint = update 0)" if s["best_u"] == 0 else
                   f"raised it to a best of {s['best']:.3f} at update {s['best_u']} ({s['best'] - s['u0']:+.3f})")
        d = s["final"] - s["u0"]
        L.append(f"- **{SHORT[x]}** (start: {START[x]}): update 0 HV {s['u0']:.3f}; fine-tuning {verdict}; final HV at "
                 f"update {s['final_u']} is {s['final']:.3f} ({d:+.3f}, {'raised' if d > 0 else 'lowered' if d < 0 else 'unchanged'}).")
    L.append("")
    chk = [f"{SHORT[x]} {summary(R[x])['u0']:.4f} vs probe {zero_val[ZERO[x]]:.4f}" for x in MC + ["CLE"]
           if x in R and ZERO[x] in zero_val and summary(R[x]) and summary(R[x])["u0"] is not None]
    if chk:
        L += ["Check, update-0 validation HV of each fine-tuning run vs its zero-training probe (same network): "
              + "; ".join(chk) + ".", ""]

    # ---------------- training
    L += ["## Training (validation split, 15 preferences; same machine for every arm)", "",
          "| Arm | Start | Updates | HV at update 0 | Best HV | Update of best | Env steps to best (M) | Wall-clock to best (h) | "
          "Final HV | Training (h) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x in MC + CL:
        s = summary(R[x]) if x in R else None
        if s is None:
            L.append(f"| {SHORT[x]} | {START[x]} | - | - | - | - | - | - | - | - |"); continue
        L.append(f"| {SHORT[x]} | {START[x]} | {s['final_u']} | {f3(s['u0'])} | {s['best']:.3f} | {s['best_u']} | "
                 f"{'-' if s['best_steps'] is None else f'{s['best_steps'] / 1e6:.2f}'} | {s['best_h']:.2f} | "
                 f"{s['final']:.3f} | {s['train_h']:.2f} |")
    L += ["", "Wall-clock excludes the shared-base preparation above (SB-AVG, TA, TIES) and the distillation (CL-D).", ""]

    # ---------------- evaluation
    rows = ["ZE", "CLE", "ZSBAVG", "SBAVG", "ZTA", "TA", "ZTIES", "TIES", "CLB", "CLD"]
    any_ev = next((E[x] for x in rows if x in E and E[x]["npz"] is not None), None)
    keys = [str(q) for q in any_ev["npz"]["keys"]] if any_ev else []
    keys = keys if all(q in norm for q in keys) else []
    if keys:
        size = "10x5"
        grid = MM.simplex_grid(AW._grid_step(int(any_ev["json"]["grid_size"])))
        b1, b2 = AW.load_cpsat("b1", a.eval_split, size), AW.load_cpsat("b2", a.eval_split, size)
        sub = [q for q in keys if q in b2]
        sev = {f"{x}": v for x, v in E.items() if v["npz"] is not None and [str(q) for q in v["npz"]["keys"]] == keys}
        reg = AW.regret_table(sev, keys, norm, grid, a.eval_split, size, b1, b2) if sev else None
        L += [f"## Evaluation ({a.eval_split} split, greedy, {len(grid)} preferences, {len(keys)} instances; IGD+ on the "
              f"{len(sub)}-instance CP-SAT B2 subset)", "",
              "| Arm | HV | HV on the B2 subset | IGD+ | Unique non-dominated | Mean regret | Best-of-16 sampling HV |",
              "|---|---|---|---|---|---|---|"]
        for x in rows:
            v = E.get(x)
            if v is None or v["npz"] is None or [str(q) for q in v["npz"]["keys"]] != keys:
                L.append(f"| {LABEL[x]} | - | - | - | - | - | - |"); continue
            hv = np.asarray(v["npz"]["hv"])
            hv_sub = np.mean([hv[keys.index(q)] for q in sub]) if sub else None
            smp = v["json"].get("sample") or {}
            L.append(f"| {LABEL[x]} | {hv.mean():.3f} | {f3(hv_sub)} | {f3(v['json']['igd_plus_mean'])} | "
                     f"{v['json']['nd_size_mean']:.1f} | {f3(reg[0][x]['all']) if reg and x in reg[0] else '-'} | "
                     f"{f3(smp.get('hv_mean'))} |")
        if sub:
            hvb2 = [MM.instance_hv(np.array([[p[o] for o in OBJ] for p in b2[q]["front"]]), *MM.ideal_nadir(norm[q]))
                    for q in sub]
            L.append(f"| CP-SAT B2 reference front (subset only) | - | {np.mean(hvb2):.3f} | 0 | - | - | - |")
        L += ["", "A zero-training merge ignores the preference (its preference columns are zero), so its 66 greedy schedules "
              "per instance are identical and its unique non-dominated count is 1. Regret is computed against "
              "every point of the arms in this table, the specialists, the rules and CP-SAT B1/B2 (so values are not "
              "comparable with the other reports).", ""]
    else:
        L += ["## Evaluation", "", "_No evaluation results for this split yet._", ""]

    # ---------------- contrastive diagnostics
    L += ["## Contrastive diagnostics (mean over the first and the last 100 updates)", "",
          "| Arm | Loss | Groups with ≥ 1 pair | Distinct schedules per group (of K) | Policy entropy | Grad norm (pre-clip) | "
          "Skipped updates | s / update |", "|---|---|---|---|---|---|---|---|"]
    for x in MC + ["CLE"]:
        lg = cl_log(R[x]) if x in R else []
        if not lg:
            continue
        g = lambda sl, k: np.nanmean([np.nan if e["cl"][k] is None else e["cl"][k] for e in sl])
        cells = [f"{fmt.format(g(lg[:100], k))} → {fmt.format(g(lg[-100:], k))}" for k, fmt in
                 (("loss", "{:.4f}"), ("groups_with_pair", "{:.2f}"), ("distinct_schedules", "{:.2f}"),
                  ("entropy", "{:.3f}"), ("grad_norm", "{:.3f}"))]
        L.append(f"| {SHORT[x]} | " + " | ".join(cells) + f" | {sum(e['cl']['skipped'] for e in lg)} | "
                 f"{np.mean([e['cl']['seconds'] for e in lg]):.2f} |")
    L.append("")

    # ---------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
    col = {"SBAVG": "#2a7fd4", "TA": "#d4762a", "TIES": "#3a9a4a", "CLE": "#7a7a7a", "CLB": "#b0b0b0", "CLD": "#555555"}
    for x in MC + CL:
        if x not in R or not R[x]["valiquality"]:
            continue
        u, hv, sec, _ = AW.val_series(R[x])
        ls = "-" if x in MC or x == "CLE" else ":"
        ax[0].plot(u, hv, ls, color=col[x], lw=1.6 if x in MC else 1.1, label=SHORT[x])
        ax[1].plot(np.asarray(sec) / 3600, hv, ls, color=col[x], lw=1.6 if x in MC else 1.1, label=SHORT[x])
    for axis, xl in zip(ax, ("updates", "training wall-clock (h, this machine)")):
        axis.set_xlabel(xl); axis.set_ylabel("validation HV"); axis.grid(alpha=.3)
    ax[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Validation hypervolume, contrastive fine-tuning from each start, 10x5, seed 111"); fig.tight_layout()
    fig.savefig(fig_dir / "curves_10x5.png", dpi=120); plt.close(fig)
    L += [f"![curves]({res.name}/figures/curves_10x5.png)", ""]

    L += ["## Design choices (fixed before any run)", ""] + [f"{i}. {d}" for i, d in enumerate(DESIGN, 1)] + [""]
    L += ["## Known limitation", "", "DANIEL's action mask only admits non-delay schedules (results/diagnose_v3.md, "
          "Test 2), kept as published for every arm; part of every gap to CP-SAT is a gap of the action space.", "",
          "## Frozen inputs", "", "```json", json.dumps(man, indent=1), "```", ""]
    if sel:
        L += ["## Committed networks", "", "Starting networks and best checkpoints: `checkpoints/merge_categories_v3/` "
              "(SHA-256 in `SHA256SUMS.json`). Selected starts:", "", "```json",
              json.dumps(sel.get("start_sha256", {}), indent=1), "```", ""]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
