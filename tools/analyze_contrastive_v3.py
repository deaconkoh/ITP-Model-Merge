"""Analysis of the contrastive fine-tuning arms (docs/contrastive_protocol_v3.md): each CL arm side by side with its PPO
counterpart (B, D, E; seed 111, read-only from results/warmstart_v3). Writes results/contrastive_v3.md and figures in
results/contrastive_v3/figures/. Reuses tools/analyze_warmstart_v3.py unchanged (loading, regret, references) and
daniel/mo_metrics.py. Single seed: numbers only, no significance claims.

    python tools/analyze_contrastive_v3.py [--eval_split evaluation] [--results results/contrastive_v3] [--out ...]
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
MM, B, OBJ = AW.MM, AW.B, AW.OBJ

WS_RES = REPO / "results" / "warmstart_v3"
MANIFEST = REPO / "configs" / "contrastive_v3.json"
PAIRS = [("CLB", "B"), ("CLD", "D"), ("CLE", "E")]
LABEL = {"CLB": "CL-B (contrastive, scratch)", "CLD": "CL-D (contrastive from D's update-0 network)",
         "CLE": "CL-E (contrastive from the naive merge)", "B": "B (PPO, scratch)", "D": "D (PPO from the distilled student)",
         "E": "E (PPO from the naive merge)"}
SHORT = {"CLB": "CL-B", "CLD": "CL-D", "CLE": "CL-E", "B": "B (PPO)", "D": "D (PPO)", "E": "E (PPO)"}
AW.NAME_RE = re.compile(r"^(\d+x\d+)\+mix\+bv3_(?:wss?|cls?)_([A-Z0-9]+)_s(\d+)$")
AW.ARMS = [a for p in PAIRS for a in p]
AW.LABEL = {**AW.LABEL, **LABEL}
SEED = 111
UPD_BUDGETS = (0, 250, 500, 1000, 2000)
STEP_BUDGETS = (250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000)

DESIGN = [
    "Groups: each of the 20 training instances of an update gets one preference w (the PPO arms' preference stream, "
    "`pref_rng`), shared by its K = 8 sampled schedules; instances are the PPO arms' generator stream (resampled every "
    "20 updates), each repeated K times (160 environments).",
    "Objectives are scored exactly by the environment (makespan, carbon, total tardiness), as in validation.",
    "Normalisation for the Tchebycheff score: f̂ = (MK/mk_ref, C/cb_ref, T/T_ref), the per-instance training references "
    "of the PPO reward (generator references and the frozen T_ref); ideal point 0. No validation, evaluation or test "
    "references are used in training.",
    "Pairs: (a) Pareto dominance on the raw objectives (≤ on all three, < on at least one; tolerance 1e-9), any w; "
    "(b) Tchebycheff g(x|w) = max_k w_k f̂_k, pair when g(worse) − g(better) > 0.01. Union, a pair counted once; "
    "identical objective vectors never form a pair.",
    "Loss: mean over all pairs of the update (every pair weighted equally, not per group) of "
    "−log σ(β[(ℓ_i − ℓ_i^ref) − (ℓ_j − ℓ_j^ref)]), β = 0.1, ℓ = Σ_t log π(a_t|s_t) over the full schedule.",
    "Reference: the starting network, frozen (CL-B: the seed-111 random initialisation; CL-D: the distilled student, "
    "i.e. D's update-0 network; CL-E: the uniform merge with zero preference columns).",
    "Optimisation: one gradient step per update (on-policy; no epochs, no importance ratios), PPO's Adam optimiser at "
    "lr 3e-4, gradient-norm clipping at 1.0. Exact gradient in two passes (trajectory log-probabilities, then per-state "
    "log-probabilities in minibatches of 1,024 weighted by dL/dℓ). An update without any pair takes no step (logged).",
    "Negatives are the policy's own samples only.",
    "Budget: the PPO counterpart's final update count (frozen in configs/contrastive_v3.json); no early stopping "
    "(status FINISHED). Validation at update 0 and every 50 updates; best-validation-HV checkpoint, as for PPO.",
    "Distinct schedules per group = distinct action sequences among the K samples.",
    "Evaluation uses daniel/evaluate_mo_v3.py (the multi-objective evaluator of the PPO arms; the brief's "
    "`evaluate_v3.py` is the single-objective specialist evaluator).",
    "Environment steps: CL = 160 environments × operations per update; PPO = 20 × operations. Both see the same "
    "instances at the same update, so PPO's steps at update u are CL's steps at u divided by K.",
    "No hyperparameter was tuned; every constant was fixed in configs/contrastive_v3.json before any run.",
]


def runs_and_evals(res, split):
    AW.RES = Path(res)
    return AW.load(split)


def cl_log(r):
    return [e for e in (r["reward"] or []) if "cl" in e]


def steps_per_update(cl_runs):
    """Environment steps per update of CL (from its log) and of PPO (CL / K): same instances at the same update."""
    k = json.loads(MANIFEST.read_text())["constants"]["cl_k"]
    for r in cl_runs.values():
        lg = cl_log(r)
        if lg:
            return np.array([e["cl"]["env_steps"] for e in lg], float), k
    return None, k


def steps_at(per_update, u):
    return float(per_update[:u].sum()) if per_update is not None and u <= len(per_update) else None


def f3(v):
    return "-" if v is None or not np.isfinite(v) else f"{v:.3f}"


def summary_row(r):
    u, hv, sec, _ = AW.val_series(r)
    if not u:
        return None
    b = int(np.argmax(hv))
    st = r["status"] or {}
    return {"u0": hv[0] if u[0] == 0 else None, "best": hv[b], "best_u": u[b], "best_h": sec[b] / 3600,
            "final": hv[-1], "final_u": u[-1], "train_h": st.get("train_seconds", sec[-1]) / 3600,
            "spu": (st.get("train_seconds", sec[-1]) / max(1, st.get("update", u[-1])))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_split", default="evaluation")
    ap.add_argument("--norm", nargs="+", default=[str(AW.BRES / "normalisation_points.json")])
    ap.add_argument("--results", default=str(REPO / "results" / "contrastive_v3"))
    ap.add_argument("--out", default=str(REPO / "results" / "contrastive_v3.md"))
    a = ap.parse_args()
    res = Path(a.results); fig_dir = res / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    norm = {}
    for f in a.norm:
        norm.update(json.loads(Path(f).read_text())["instances"])
    cl_runs, cl_ev = runs_and_evals(res, a.eval_split)
    ws_runs, ws_ev = runs_and_evals(WS_RES, a.eval_split)
    AW.RES = res
    pick = lambda d: {v["arm"]: v for v in d.values() if v["seed"] == SEED and v["arm"] in AW.ARMS} if d else {}
    R = {**pick(ws_runs), **pick(cl_runs)}                    # arm -> run record
    E = {**pick(ws_ev), **pick(cl_ev)}                        # arm -> evaluation record
    per_update, k = steps_per_update({x: R[x] for x in ("CLB", "CLD", "CLE") if x in R})
    man = json.loads(MANIFEST.read_text())
    plan = json.loads((res / "plan.json").read_text()) if (res / "plan.json").exists() else {}
    calib = res / "calibration_10x5.json"
    factor = json.loads(calib.read_text())["factor"] if calib.exists() else None

    L = ["# Contrastive fine-tuning on benchmark v3: results", "",
         "Generated by `tools/analyze_contrastive_v3.py`. Each contrastive (CL) arm replaces only PPO's update with a "
         "DPO-style pairwise-ranking objective; model, environment, non-delay mask, preference sampling, training "
         "instances, training normalisation, validation, checkpoint selection and evaluation are those of the PPO arms "
         "(docs/contrastive_protocol_v3.md). PPO counterparts: results/warmstart_v3, seed 111.", "",
         "**Single seed (111): numbers only, no significance claims.** Higher HV is better; lower IGD+ and regret are "
         f"better. Evaluation split: `{a.eval_split}`.", "",
         f"Hardware: {plan.get('host', {}).get('gpu', '?')}, {plan.get('host', {}).get('logical_cpus', '?')} logical CPUs, "
         f"--jobs {plan.get('jobs', '?')}" + (f"; calibration factor (this machine / local DANIEL-M) {factor:.2f}."
                                               if factor else "."), "",
         "E0 (zero-training merge front) is not part of this study: it is a family of networks θ(λ) = Σ λ_x θ_x, one "
         "unconditioned network per preference, not a single network that could be fine-tuned.", ""]

    # ---------------- did fine-tuning raise or lower HV?
    L += ["## Did fine-tuning raise or lower validation HV relative to update 0?", ""]
    for cl, ppo in PAIRS:
        for x in (cl, ppo):
            s = summary_row(R[x]) if x in R else None
            if s is None or s["u0"] is None:
                L.append(f"- **{SHORT[x]}**: no validation curve yet."); continue
            d_best, d_final = s["best"] - s["u0"], s["final"] - s["u0"]
            verdict = ("never raised it (best checkpoint = update 0)" if s["best_u"] == 0 else
                       f"raised it to a best of {s['best']:.3f} at update {s['best_u']} ({d_best:+.3f})")
            L.append(f"- **{SHORT[x]}**: update 0 HV {s['u0']:.3f}; fine-tuning {verdict}; final HV at update "
                     f"{s['final_u']} is {s['final']:.3f} ({d_final:+.3f} vs update 0, "
                     f"{'raised' if d_final > 0 else 'lowered' if d_final < 0 else 'unchanged'}).")
    L.append("")
    if "CLD" in R and "D" in R:
        a0, b0 = summary_row(R["CLD"]), summary_row(R["D"])
        if a0 and b0 and a0["u0"] is not None and b0["u0"] is not None:
            L += [f"Reproduction check: CL-D's update-0 validation HV {a0['u0']:.4f} vs PPO D's {b0['u0']:.4f} (the D "
                  "student was regenerated with the same seed on this machine).", ""]

    # ---------------- training summary
    L += ["## Training (validation split, 15 preferences)", "",
          "| Arm | Budget (updates) | HV at update 0 | Best HV | Update of best | Wall-clock to best (h) | Final HV | "
          "Training (h) | s / update | Env steps (total) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cl, ppo in PAIRS:
        for x in (ppo, cl):
            s = summary_row(R[x]) if x in R else None
            if s is None:
                L.append(f"| {LABEL[x]} | - | - | - | - | - | - | - | - | - |"); continue
            mult = 1 if x.startswith("CL") else 1 / k
            steps = steps_at(per_update, s["final_u"])
            L.append(f"| {LABEL[x]} | {man['arms'][cl]['updates']} | {f3(s['u0'])} | {s['best']:.3f} | {s['best_u']} | "
                     f"{s['best_h']:.2f} | {s['final']:.3f} | {s['train_h']:.2f} | {s['spu']:.2f} | "
                     f"{'-' if steps is None else f'{steps * mult / 1e6:.2f} M'} |")
    L.append("")

    # ---------------- equal updates and equal environment steps
    L += ["### Validation HV at equal updates", "", "| Arm | " + " | ".join(f"u = {u}" for u in UPD_BUDGETS) + " |",
          "|---|" + "---|" * len(UPD_BUDGETS)]
    for cl, ppo in PAIRS:
        for x in (ppo, cl):
            if x not in R or not R[x]["valiquality"]:
                continue
            u, hv, _, _ = AW.val_series(R[x])
            L.append(f"| {SHORT[x]} | " + " | ".join(f3(MM.curve_at(u, hv, b)) if b <= u[-1] else "-" for b in UPD_BUDGETS)
                     + " |")
    L += ["", "### Validation HV at equal environment steps", "",
          f"CL takes {k}× PPO's environment steps per update (same instances), so CL at update u is compared with PPO at "
          f"update {k}u. '-' = the run did not reach that many steps.", "",
          "| Arm | " + " | ".join(f"{s / 1e6:g} M steps" for s in STEP_BUDGETS) + " |", "|---|" + "---|" * len(STEP_BUDGETS)]
    curves_steps = {}
    if per_update is not None:
        cum = np.concatenate([[0.0], np.cumsum(per_update)])
        for cl, ppo in PAIRS:
            for x in (ppo, cl):
                if x not in R or not R[x]["valiquality"]:
                    continue
                u, hv, _, _ = AW.val_series(R[x])
                scale = 1.0 if x.startswith("CL") else 1.0 / k
                xs = [cum[min(v, len(cum) - 1)] * scale if v < len(cum) else np.nan for v in u]
                curves_steps[x] = (xs, hv)
                row = []
                for sb in STEP_BUDGETS:
                    ok = [i for i, s in enumerate(xs) if np.isfinite(s) and s <= sb]
                    reached = any(np.isfinite(s) and s >= sb for s in xs)
                    row.append(f3(hv[ok[-1]]) if ok and reached else "-")
                L.append(f"| {SHORT[x]} | " + " | ".join(row) + " |")
    L.append("")

    # ---------------- evaluation
    any_ev = next(iter(E.values()), None)
    keys = [str(q) for q in any_ev["npz"]["keys"]] if any_ev and any_ev["npz"] is not None else []
    keys = keys if all(q in norm for q in keys) else []
    if keys:
        size = "10x5"
        grid = MM.simplex_grid(AW._grid_step(int(any_ev["json"]["grid_size"])))
        b1, b2 = AW.load_cpsat("b1", a.eval_split, size), AW.load_cpsat("b2", a.eval_split, size)
        sub = [q for q in keys if q in b2]
        sev = {v_name: v for v_name, v in {**{n: v for n, v in ws_ev.items()}, **cl_ev}.items()
               if v["seed"] == SEED and v["arm"] in E and v["npz"] is not None
               and [str(q) for q in v["npz"]["keys"]] == keys}
        reg = AW.regret_table(sev, keys, norm, grid, a.eval_split, size, b1, b2) if sev else None
        L += [f"## Evaluation ({a.eval_split} split, greedy, {len(grid)} preferences, {len(keys)} instances; IGD+ on the "
              f"{len(sub)}-instance CP-SAT B2 subset)", "",
              "| Arm | HV | HV on the B2 subset | IGD+ | Unique non-dominated | Mean regret | Best-of-16 sampling HV |",
              "|---|---|---|---|---|---|---|"]
        for cl, ppo in PAIRS:
            for x in (ppo, cl):
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
        L += ["", "Regret is computed against every point of the arms in this table, the specialists, the rules and "
              "CP-SAT B1/B2 (the warm-start study's definition over a smaller set of arms, so values are not "
              "comparable with results/warmstart_v3.md).", ""]
        L += [f"### Objectives at the corner and equal-weight preferences", "",
              "| Arm | MK at w=(1,0,0) | C at w=(0,1,0) | T at w=(0,0,1) | MK / C / T at w=(1/3,1/3,1/3) |",
              "|---|---|---|---|---|"]
        for cl, ppo in PAIRS:
            for x in (ppo, cl):
                v = E.get(x)
                if v is None:
                    continue
                q = v["json"]["objectives_at"]
                L.append(f"| {SHORT[x]} | {q['w=(1,0,0)']['makespan']:.1f} | {q['w=(0,1,0)']['carbon']:.1f} | "
                         f"{q['w=(0,0,1)']['tardiness']:.1f} | "
                         + " / ".join(f"{q['w=(1/3,1/3,1/3)'][o]:.1f}" for o in OBJ) + " |")
        L.append("")
    else:
        L += ["## Evaluation", "", "_No evaluation results for this split yet._", ""]

    # ---------------- contrastive diagnostics
    L += ["## Contrastive diagnostics (per update; mean over the first and the last 100 updates)", "",
          "| Arm | Loss | Groups with ≥ 1 pair | Pairs per update (dominance / Tchebycheff) | Distinct schedules per group "
          "(of K) | Policy entropy | Grad norm (pre-clip) | Skipped updates | s / update (rollout + step) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in ("CLB", "CLD", "CLE"):
        lg = cl_log(R[x]) if x in R else []
        if not lg:
            continue
        def win(part):
            sl = lg[:100] if part == 0 else lg[-100:]
            g = lambda key: np.nanmean([np.nan if e["cl"][key] is None else e["cl"][key] for e in sl])
            return g, sl
        cells = []
        for key, fmt in (("loss", "{:.4f}"), ("groups_with_pair", "{:.2f}"), ("pairs", "{:.0f}"),
                         ("distinct_schedules", "{:.2f}"), ("entropy", "{:.3f}"), ("grad_norm", "{:.3f}")):
            (g0, s0), (g1, s1) = win(0), win(1)
            if key == "pairs":
                cells.append(f"{g0('pairs'):.0f} ({g0('pairs_dominance'):.0f} / {g0('pairs_tchebycheff'):.0f}) → "
                             f"{g1('pairs'):.0f} ({g1('pairs_dominance'):.0f} / {g1('pairs_tchebycheff'):.0f})")
            else:
                cells.append(f"{fmt.format(g0(key))} → {fmt.format(g1(key))}")
        L.append(f"| {SHORT[x]} | " + " | ".join(cells) + f" | {sum(e['cl']['skipped'] for e in lg)} | "
                 f"{np.mean([e['cl']['seconds'] for e in lg]):.2f} |")
    L.append("")

    # ---------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    cmap = plt.get_cmap("tab10")
    for c, (cl, ppo) in enumerate(PAIRS):
        for x, ls in ((ppo, "--"), (cl, "-")):
            if x not in R or not R[x]["valiquality"]:
                continue
            u, hv, sec, _ = AW.val_series(R[x])
            ax[0].plot(u, hv, ls, color=cmap(c), label=SHORT[x])
            if x in curves_steps:
                ax[1].plot(np.asarray(curves_steps[x][0]) / 1e6, hv, ls, color=cmap(c), label=SHORT[x])
            ax[2].plot(np.asarray(sec) / 3600, hv, ls, color=cmap(c), label=SHORT[x])
    for axis, xl in zip(ax, ("updates", "environment steps (M)", "training wall-clock (h)")):
        axis.set_xlabel(xl); axis.set_ylabel("validation HV"); axis.grid(alpha=.3)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Validation hypervolume: contrastive (solid) vs PPO (dashed), seed 111"); fig.tight_layout()
    fig.savefig(fig_dir / "curves_10x5.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for c, (cl, _) in enumerate(PAIRS):
        lg = cl_log(R[cl]) if cl in R else []
        if not lg:
            continue
        u = [e["update"] for e in lg]
        for axis, key in zip(ax, ("loss", "groups_with_pair", "distinct_schedules", "entropy")):
            axis.plot(u, [np.nan if e["cl"][key] is None else e["cl"][key] for e in lg], color=cmap(c), lw=.8,
                      label=SHORT[cl])
            axis.set_title(key.replace("_", " ")); axis.set_xlabel("updates"); axis.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "cl_diagnostics_10x5.png", dpi=110); plt.close(fig)
    L += [f"![curves]({res.name}/figures/curves_10x5.png)", "", f"![diagnostics]({res.name}/figures/cl_diagnostics_10x5.png)",
          ""]

    L += ["## Design choices (fixed before any run)", ""] + [f"{i}. {d}" for i, d in enumerate(DESIGN, 1)] + [""]
    L += ["## Known limitation", "", "DANIEL's action mask only admits non-delay schedules (results/diagnose_v3.md, "
          "Test 2), kept as published for every arm; part of every gap to CP-SAT is a gap of the action space.", "",
          "## Frozen inputs", "", "```json", json.dumps({k: v for k, v in man.items()}, indent=1), "```", ""]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
