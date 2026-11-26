"""Consolidate the "why did D not beat C" investigation (10x5, waiting allowed, seed 111).

Reads only existing results (Experiment 1: results/wm_merge_diagnosis_v3; Experiments 2/3:
results/wm_distill_direct_v3; references: rs_four_bases_v3, specialist_reuse_v3,
specialist_reuse_wm_td_v3, daniel_wm_direct_v3). Writes
results/wm_diagnosis_summary_v3/{summary.json, *.png}. No training, no evaluation, no final-test access.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import rs_four_bases_v3 as RS                                           # noqa: E402

OUT = REPO / "results" / "wm_diagnosis_summary_v3"
DX = REPO / "results" / "wm_merge_diagnosis_v3"
DD = REPO / "results" / "wm_distill_direct_v3"
RS4 = REPO / "results" / "rs_four_bases_v3"
SR = REPO / "results" / "specialist_reuse_v3"
SRB = REPO / "results" / "specialist_reuse_wm_td_v3"
WMD = REPO / "results" / "daniel_wm_direct_v3"
L = lambda p: json.loads(Path(p).read_text())                            # noqa: E731
K = "ABCD"
LABEL = {"A": "A random", "B": "B outcome prediction", "C": "C immediate physical", "D": "D joint latent + physical"}
METHODS = ("direct", "rs", "distill")
MLABEL = {"direct": "Direct MO", "rs": "Rewarded Soups", "distill": "Distillation"}
H = 3600.0


def evals():
    f = {
        "direct": {"A": SR / "eval/scratch.npz", "B": WMD / "eval/wmdirect.npz",
                   "C": DD / "eval/C_direct.npz", "D": DD / "eval/D_direct.npz"},
        "rs": {k: (RS4 / f"eval/{k}_final.npz") for k in "CD"} | {"A": SR / "eval/rs.npz", "B": REPO / "results/specialist_reuse_wm_v3/eval/rs.npz"},
        "distill": {"A": SR / "eval/distill.npz", "B": SRB / "eval/distill.npz",
                    "C": DD / "eval/C_distill.npz", "D": DD / "eval/D_distill.npz"},
        "rs_before": {"A": REPO / "results/eval_curve_v3/eval/rs_random_u0.npz",
                      "B": REPO / "results/eval_curve_v3/eval/rs_world_model_u0.npz",
                      "C": RS4 / "eval/C_merged_start.npz", "D": RS4 / "eval/D_merged_start.npz"},
        "distill_before": {k: DD / f"eval/{k}_student.npz" for k in K},
        "matched": {f"{k}@t{u}": DX / f"eval/{k}_t{u}.npz" for k in K for u in (1000, 2000)},
    }
    out = {}
    for g, d in f.items():
        out[g] = {}
        for k, p in d.items():
            m = RS.metrics(p)
            out[g][k] = {"hv": m["best_of_128_hv"], "igd_plus": m["best_of_128_igd_plus"], "greedy_hv": m["greedy_hv"],
                         "igd_instances": m["igd_instances"], "eval_seconds": m["eval_seconds"],
                         "file": str(Path(p).relative_to(REPO))}
    return out


def costs():
    rs4 = L(RS4 / "summary.json")["rows"]
    sa, sb = L(SR / "state.json"), L(SRB / "state.json")
    st = {f"{k}_{m}": L(DD / f"{k}_{m}/state.json") for k in "CD" for m in ("direct", "distill")}
    pre = {k: rs4[k]["pretraining_hours"] for k in K}
    spec = {k: rs4[k]["specialist_hours_total"] for k in K}
    stage1 = {"A": sa["distill_stage1"]["total_seconds"] / H, "B": sb["distill_stage1"]["total_seconds"] / H,
              "C": st["C_distill"]["distill_stage1"]["total_seconds"] / H,
              "D": st["D_distill"]["distill_stage1"]["total_seconds"] / H}
    dft = {"A": sa["units"]["final:distill"]["train_seconds"] / H, "B": sb["units"]["final:distill"]["train_seconds"] / H,
           "C": st["C_distill"]["units"]["final:distill"]["train_seconds"] / H,
           "D": st["D_distill"]["units"]["final:distill"]["train_seconds"] / H}
    direct = {"A": sa["units"]["final:scratch"]["train_seconds"] / H,
              "B": L(WMD / "summary.json")["world_model_encoder"]["train_hours"],
              "C": st["C_direct"]["units"]["final:direct"]["train_seconds"] / H,
              "D": st["D_direct"]["units"]["final:direct"]["train_seconds"] / H}
    rs = {k: rs4[k]["rs_hours"] for k in K}
    rows = {}
    for k in K:
        rows[k] = {
            "pretraining": pre[k], "specialists": spec[k],
            "direct": {"mo_training": direct[k], "total": pre[k] + direct[k], "additional_given_specialists": direct[k]},
            "rs": {"mo_training": rs[k], "total": pre[k] + spec[k] + rs[k], "additional_given_specialists": rs[k]},
            "distill": {"teacher_data_and_stage1": stage1[k], "mo_training": dft[k],
                        "total": pre[k] + spec[k] + stage1[k] + dft[k],
                        "additional_given_specialists": stage1[k] + dft[k]},
        }
    tun = {m: sum(u["train_seconds"] for n, u in sa["units"].items() if n.startswith(f"trial:{m}:")) / H
           for m in ("rs", "distill", "scratch")}
    runs = {"C_direct": st["C_direct"]["units"]["final:direct"], "C_distill": st["C_distill"]["units"]["final:distill"],
            "D_direct": st["D_direct"]["units"]["final:direct"], "D_distill": st["D_distill"]["units"]["final:distill"]}
    wall = L(DD / "wall.json")
    summed = sum(u["train_seconds"] for u in runs.values()) + sum(
        st[f"{k}_distill"]["distill_stage1"]["total_seconds"] for k in "CD")
    return rows, {"tuning_hours_on_random_base_reused_by_all": tun,
                  "runs": {n: {kk: u[kk] for kk in ("status", "update", "best_update", "train_seconds")} | {
                      "best_validation_hv": -u["best"]} for n, u in runs.items()},
                  "exp23_summed_training_hours": summed / H, "exp23_elapsed_training_hours": wall["training_s"] / H,
                  "exp23_elapsed_stage1_hours": wall["stage1_s"] / H,
                  "exp23_evaluation_elapsed_hours": wall["evaluation_s"] / H,
                  "exp23_concurrency": wall["concurrency"]}


def diag():
    w, p, it, mt = L(DX / "weights.json"), L(DX / "probes.json"), L(DX / "interpolation.json"), L(DX / "matched.json")
    drift = {k: {o: {part: w["drift"][k][o]["parts"][part]["relative"] for part in ("encoder", "actor")} for o in "mct"}
             for k in K}
    tv = {k: {part: {pp: {x: w["task_vectors"][k][part][pp][x] for x in ("cosine", "sign_agreement", "norm_ratio")}
                     for pp in ("mc", "mt", "ct")} | {"norm": w["task_vectors"][k][part]["norm"]}
              for part in ("encoder", "actor", "all")} for k in K}
    rep = p["representation"]
    cka = {k: {o: {t: rep[k][o][t]["cka_to_base"] for t in rep[k][o]} for o in "mct"} for k in K}
    cka_pairs = {k: {pp: rep[k][pp] for pp in ("mc", "mt", "ct")} for k in K}
    ret = {}
    for n, v in p["retention"].items():
        r = {"relational_mean": v["relational"]["mean"],
             "pair_earliest_start_r2": v["relational"]["pair_earliest_start_r2"],
             "best_machine_accuracy": v["relational"]["best_machine_accuracy"],
             "machine_earliest_start_r2": v["relational"]["machine_earliest_start_r2"],
             "availability_r2": v["availability_next_state"]["absolute_centred_free_time"]["r2"],
             "availability_sign_accuracy": v["availability_next_state"]["same_operation_branch_difference"][
                 "sign_accuracy_nonzero"],
             "machine_free_time_now_r2": v["current_state_readouts"]["machine_free_time_now"]["r2"],
             "job_ready_time_now_r2": v["current_state_readouts"]["job_ready_time_now"]["r2"],
             "refit_change_mae": v["physical_refit_head"]["mean_change_mae_changed"],
             "refit_F_r2": v["physical_refit_head"]["F_change_r2_changed"],
             "refit_R_r2": v["physical_refit_head"]["R_change_r2_changed"],
             "refit_T_r2": v["physical_refit_head"]["T_change_r2_changed"]}
        if "physical_original_head" in v:
            o = v["physical_original_head"]
            r |= {"original_change_mae": o["mean_change_mae_changed"], "original_F_r2": o["F_change_r2_changed"],
                  "original_R_r2": o["R_change_r2_changed"], "original_T_r2": o["T_change_r2_changed"]}
        if "latent_original_predictor" in v:
            o = v["latent_original_predictor"]
            r |= {"latent_change_explained": o["changed_tokens"]["change_explained"],
                  "latent_discrimination": o["discrimination"]["overall"],
                  "latent_same_op_other_machine": o["discrimination"]["same_operation_other_machine"]}
        ret[n] = r
    interp = {k: {pp: {"mean_interior_gap": it[k][pp]["mean_interior_gap"], "max_interior_gap": it[k][pp]["max_interior_gap"]}
                  for pp in ("mc", "mt", "ct")} for k in K}
    matched = {n: {"tardiness_update": v["tardiness_update"], "validation_hv": v["validation_hv"]}
               for n, v in mt["merges"].items()}
    return {"drift": drift, "task_vectors": tv, "cka_to_base": cka, "cka_between_specialists": cka_pairs,
            "retention": ret, "interpolation": interp, "matched_validation": matched}


# --------------------------------------------------------------------------------------------- plots
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
COL = RS.COL
MCOL = {"direct": "#184f95", "rs": "#3987e5", "distill": "#9ec5f4"}


def _ax(ax):
    ax.grid(color=GRID, lw=.8, axis="y"); ax.set_axisbelow(True); ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plots(ev, cost, dg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.facecolor": SURF})
    x = np.arange(4)

    # 1 grouped HV bar
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); _ax(ax)
    bw = .26
    for i, m in enumerate(METHODS):
        v = [ev[m][k]["hv"] for k in K]
        b = ax.bar(x + (i - 1) * bw, v, bw - .02, color=MCOL[m], label=MLABEL[m], edgecolor=SURF, linewidth=1)
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val + .002, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, [LABEL[k] for k in K]); ax.set_ylim(1.0, 1.13); ax.set_ylabel("best-of-128 HV (100 instances × 66 preferences)")
    ax.set_title("Final HV by base and method (seed 111, 10x5)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "hv_grouped.png", dpi=150); plt.close(fig)

    # 2 stacked cost
    fig, ax = plt.subplots(figsize=(10, 4.6)); _ax(ax)
    parts = (("pretraining", "world-model pretraining", "#0d366b"), ("specialists", "specialists (sum of 3)", "#3987e5"),
             ("stage1", "teacher data + distillation stage 1", "#eda100"), ("mo", "multi-objective training", "#9ec5f4"))
    labels, stacks = [], []
    for k in K:
        for m in METHODS:
            c = cost[k]
            d = {"pretraining": c["pretraining"], "specialists": c["specialists"] if m != "direct" else 0,
                 "stage1": c[m].get("teacher_data_and_stage1", 0), "mo": c[m]["mo_training"]}
            labels.append(f"{k}\n{MLABEL[m].replace(' ', chr(10), 1) if m == 'rs' else MLABEL[m]}"); stacks.append(d)
    xs = np.arange(len(labels)) + np.repeat(np.arange(4) * .6, 3)
    bottom = np.zeros(len(labels))
    for key, lab, col in parts:
        v = np.array([s[key] for s in stacks])
        ax.bar(xs, v, .8, bottom=bottom, color=col, label=lab, edgecolor=SURF, linewidth=1)
        bottom += v
    for xx, t in zip(xs, bottom):
        ax.text(xx, t + .1, f"{t:.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(xs, labels, fontsize=7.5); ax.set_ylabel("hours (summed run durations)"); ax.set_ylim(0, bottom.max() * 1.25)
    ax.set_title("Training cost by base and method (tuning and evaluation excluded)", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "cost_stacked.png", dpi=150); plt.close(fig)

    # 3 HV vs cost
    fig, ax = plt.subplots(figsize=(7.6, 4.8)); _ax(ax); ax.grid(color=GRID, lw=.8, axis="x")
    mk = {"direct": "s", "rs": "o", "distill": "^"}
    for k in K:
        for m in METHODS:
            ax.scatter(cost[k][m]["total"], ev[m][k]["hv"], s=70, marker=mk[m], color=COL[k], edgecolor=SURF, linewidth=2, zorder=3)
            ax.annotate(f"{k}", (cost[k][m]["total"], ev[m][k]["hv"]), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK)
    for k in K:
        ax.scatter([], [], color=COL[k], s=50, label=LABEL[k])
    for m in METHODS:
        ax.scatter([], [], marker=mk[m], color=MUTED, s=50, label=MLABEL[m])
    ax.set_xlabel("total training hours (pretraining + specialists + MO training)"); ax.set_ylabel("best-of-128 HV")
    ax.set_title("HV vs training cost", loc="left"); ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "hv_vs_cost.png", dpi=150); plt.close(fig)

    # 4 drift and task vectors
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    ow = .2
    for j, (title, getter, yl) in enumerate((
            ("Encoder relative drift (specialist vs base)", lambda k, o: dg["drift"][k][o]["encoder"], "||θ_s − θ_0|| / ||θ_0||"),
            ("Encoder task-vector cosine", lambda k, o: dg["task_vectors"][k]["encoder"][o]["cosine"], "cosine"),
            ("Actor task-vector cosine", lambda k, o: dg["task_vectors"][k]["actor"][o]["cosine"], "cosine"))):
        ax = axs[j]; _ax(ax)
        groups = "mct" if j == 0 else ("mc", "mt", "ct")
        for i, k in enumerate(K):
            ax.bar(np.arange(3) + (i - 1.5) * ow, [getter(k, o) for o in groups], ow - .02, color=COL[k], label=LABEL[k],
                   edgecolor=SURF, linewidth=1)
        names = {"m": "makespan", "c": "carbon", "t": "tardiness", "mc": "m–c", "mt": "m–t", "ct": "c–t"}
        ax.set_xticks(np.arange(3), [names[g] for g in groups]); ax.set_ylabel(yl); ax.set_title(title, loc="left", fontsize=10)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "drift_task_vectors.png", dpi=150); plt.close(fig)

    # 5 interpolation curves, C and D (A, B faint)
    it = L(DX / "interpolation.json")
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for j, pp in enumerate(("mc", "mt", "ct")):
        ax = axs[j]; _ax(ax)
        for k in K:
            d = it[k][pp]
            faint = k in "AB"
            ax.plot(d["alpha"], d["merged_score"], color=COL[k], lw=1.2 if faint else 2.2, alpha=.45 if faint else 1,
                    marker=None if faint else "o", ms=4, label=f"{LABEL[k]} merged")
            if not faint:
                ax.plot(d["alpha"], d["best_endpoint_score"], color=COL[k], lw=1.2, ls="--", label=f"{k} better endpoint")
        names = {"mc": ("makespan", "carbon"), "mt": ("makespan", "tardiness"), "ct": ("carbon", "tardiness")}[pp]
        ax.set_xlabel(f"α  (0 = {names[0]} specialist, 1 = {names[1]} specialist)")
        ax.set_ylabel("preference-weighted score (lower is better)")
        ax.set_title(f"{names[0]} ↔ {names[1]}", loc="left", fontsize=10)
    axs[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Pairwise weight interpolation, greedy on 18 validation instances", x=.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "interpolation.png", dpi=150); plt.close(fig)

    # 6 validation curves of the new policies (+ A/B references, faint)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    cur = {"direct": {"A": SR / "runs/10x5+mix+bv3_sr_scratch_s111", "B": WMD / "runs/10x5+mix+bv3_srwmd_wmdirect_s111",
                      "C": DD / "C_direct/runs/10x5+mix+bv3_srip_direct_s111", "D": DD / "D_direct/runs/10x5+mix+bv3_srjp_direct_s111"},
           "distill": {"A": SR / "runs/10x5+mix+bv3_sr_distill_s111", "B": SRB / "runs/10x5+mix+bv3_srwmtd_distill_s111",
                       "C": DD / "C_distill/runs/10x5+mix+bv3_srip_distill_s111", "D": DD / "D_distill/runs/10x5+mix+bv3_srjp_distill_s111"}}
    for j, m in enumerate(("direct", "distill")):
        ax = axs[j]; _ax(ax)
        for k in K:
            q = L(cur[m][k] / "valiquality.json")
            faint = k in "AB"
            ax.plot([r["update"] for r in q], [r["hv"] for r in q], color=COL[k], lw=1.2 if faint else 2,
                    alpha=.5 if faint else 1, label=LABEL[k] + (" (reference)" if faint else ""))
        ax.set_xlabel("update"); ax.set_title(f"{MLABEL[m]}: validation HV (18 instances, greedy)", loc="left", fontsize=10)
        ax.set_ylim(0.5 if m == "direct" else 0.8, 1.0)
    axs[0].set_ylabel("validation HV"); axs[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "validation_curves.png", dpi=150); plt.close(fig)

    # extra: matched merges
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); _ax(ax)
    for k in K:
        pts = sorted((v["tardiness_update"], v["validation_hv"]) for n, v in dg["matched_validation"].items() if n[0] == k)
        ax.plot(*zip(*pts), color=COL[k], marker="o", ms=5, lw=2, label=LABEL[k])
    ax.set_xlabel("tardiness-specialist updates merged (makespan, carbon at selected checkpoints)")
    ax.set_ylabel("merged-start validation HV"); ax.set_title("Merges at matched tardiness budgets (no fine-tuning)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "matched_merges.png", dpi=150); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = evals(); cost, extra = costs(); dg = diag()
    out = {"hv": ev, "cost_hours": cost, "cost_notes": extra, "diagnostics": dg,
           "checks": {"exp23": L(DD / "checks.json"), "exp1_audit_all_pass": L(DX / "audit.json").get("all_pass")},
           "notes": {"hv": "best-of-128, 100 instances x 66 preferences, seed 51, frozen normalisation; IGD+ on the 20-instance CP-SAT subset",
                     "matched_reproduces_existing_C_start": abs(ev["matched"]["C@t1000"]["hv"] - ev["rs_before"]["C"]["hv"]) < 1e-12}}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    plots(ev, cost, dg)
    for m in ("direct", "rs", "distill", "rs_before", "distill_before"):
        print(m, {k: round(ev[m][k]["hv"], 4) for k in K})
    print("matched", {n: round(v["hv"], 4) for n, v in ev["matched"].items()})
    for k in K:
        print(k, {m: round(cost[k][m]["total"], 2) for m in METHODS}, {m: round(cost[k][m]["additional_given_specialists"], 2) for m in METHODS})
    print(json.dumps(extra, indent=1))


if __name__ == "__main__":
    main()
