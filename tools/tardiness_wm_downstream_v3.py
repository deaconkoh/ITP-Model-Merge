"""Downstream tardiness-specialist experiment: do world-model encoders translate into better specialist learning?
(10x5, waiting allowed, seed 111; exploratory, not a revision of any earlier decision rule.)

Four arms, identical except the encoder weights (feature_exact) of the start bundle:
  A  random       the earlier tardiness pilot's random start bundle (checkpoints/world_model_m1_v3_tardiness_pilot/start_rnd.pth)
  B  outcome      the earlier pilot's revised outcome-prediction encoder start (start_pre.pth; action pilot, pairs + abs+diff)
  C  immediate    immediate physical prediction only (world_model_immediate_pilot_v3 model B, matched 60-epoch checkpoint)
  D  joint        latent + immediate physical prediction (world_model_immediate_pilot_v3 model E, matched 60-epoch checkpoint)
Everything else is the earlier tardiness pilot's recipe, launched with its own code (tools/world_model_m1_tardiness_pilot.py:
args(), run_arm()): configs/benchmark_v3/t.json, contrastive fine-tuning constants, seed 111, generator training stream,
--dispatch_rule semi_active, 2,000 updates, validation every 50 updates (greedy, 18 validation instances, T / Tref),
no early stopping, encoder fine-tuned with the rest; plus --budget_checkpoints every 50 updates (snapshots only).
No world-model loss is used during specialist training. The four arms run concurrently (3 CPU cores each, one GPU).

The common threshold, budget, AUC definition and evaluation protocol are written to settings.json before any arm
starts. Selected checkpoints (best validation) and the final 2,000-update snapshots are evaluated on the 10x5
EVALUATION split (held out; the reserved final-test sets are never loaded), greedy and best-of-128 (seed 51).

    python tools/tardiness_wm_downstream_v3.py     # -> results/tardiness_wm_downstream_v3/ (+ .md)
"""
from __future__ import annotations
import json, shutil, subprocess, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import world_model_m1_tardiness_pilot as T                          # noqa: E402

NAME = "tardiness_wm_downstream_v3"
EARLIER, EARLIER_CK = T.RES, T.CK                                    # results/world_model_m1_v3/tardiness_pilot
IMM_CK = REPO / "checkpoints" / "world_model_immediate_pilot_v3"
IMM_RES = REPO / "results" / "world_model_immediate_pilot_v3"
JP_RES = REPO / "results" / "world_model_joint_pilot_v3"
RES, CK = REPO / "results" / NAME, REPO / "checkpoints" / NAME
T.RES, T.CK = RES, CK
ARMS = {"dsA": ("A random encoder", "0-2"), "dsB": ("B outcome-prediction encoder", "3-5"),
        "dsC": ("C immediate physical encoder", "6-8"), "dsD": ("D joint latent + physical encoder", "9-11")}
T.ARMS = ARMS
LABEL = {"dsA": "A random", "dsB": "B outcome prediction", "dsC": "C immediate physical", "dsD": "D joint latent + physical"}
COL = {"dsA": "#8a8a85", "dsB": "#eb6834", "dsC": "#2a78d6", "dsD": "#1baf7a"}
SNAPS = list(range(0, 2001, 50))
ALLOWED_DIFF = {"model_suffix", "init_from", "budget_checkpoints"}
_args = T.args


def args_snap(arm, s):
    return [*_args(arm, s), "--budget_checkpoints", ",".join(map(str, SNAPS))]


T.args = args_snap


def settings():
    f = RES / "settings.json"
    if f.exists():
        return json.loads(f.read_text())
    es = json.loads((EARLIER / "settings.json").read_text())
    tB, tE = (json.loads((IMM_RES / f"train_{k}.json").read_text())["matched"] for k in ("B", "E"))
    jp = json.loads((JP_RES / "coverage.json").read_text())
    xo = json.loads((IMM_RES / "extraction_old.json").read_text())
    pc = json.loads((REPO / "results/world_model_m1_v3/action_pilot.json").read_text())
    s = {**{k: es[k] for k in ("budget_updates", "target", "target_reached_by_reference_at", "target_rule", "reference", "recipe")},
         "threshold_source": "the earlier tardiness pilot's frozen common target (results/world_model_m1_v3/tardiness_pilot/"
                             "settings.json), i.e. existing validation evidence; reached = first validation point with "
                             "T / Tref <= target",
         "auc": {"primary": "mean of log10(validation T / Tref) over updates 0..2000 (trapezoid, divided by 2000)",
                 "secondary": "mean of validation T / Tref over updates 50..2000 (trapezoid, divided by 1950; excludes the "
                              "untrained start, whose value dominates a linear-scale area)"},
         "selection": "best validation T / Tref within the budget (train.py's own selection: <model>.pth)",
         "evaluation": {"split": "evaluation", "size": "10x5", "greedy": True, "sample": 128, "sample_seed": 51,
                        "checkpoints": ["selected (best validation)", "final (2,000-update snapshot)"],
                        "dispatch_rule": "semi_active", "reserved_final_test": "never loaded"},
         "snapshots": SNAPS, "seed": 111, "cpu_sets": {a: v[1] for a, v in ARMS.items()},
         "pretraining_seconds": {
             "dsA": {"total": 0.0},
             "dsB": {"data": pc["pilot_data"]["train_regeneration_seconds"],
                     "encoder_training": pc["pilot"]["pairs|abs+diff"]["seconds"]},
             "dsC": {"data_episodes_and_branching": jp["seconds"]["episodes_and_branching"],
                     "physical_target_extraction": xo["seconds"], "encoder_training_60_epochs": tB["train_seconds"]},
             "dsD": {"data_episodes_and_branching": jp["seconds"]["episodes_and_branching"],
                     "physical_target_extraction": xo["seconds"], "encoder_training_60_epochs": tE["train_seconds"]}},
         "pretraining_notes": "C and D: the joint pilot's continuation rollouts are not needed for immediate targets and are "
                              "excluded; their epoch-60 training time is the time to the matched snapshot within longer runs.",
         "sources": {"dsA": str((EARLIER_CK / "start_rnd.pth").relative_to(REPO)), "dsB": str((EARLIER_CK / "start_pre.pth").relative_to(REPO)),
                     "dsC": str((IMM_CK / "B" / "epoch60" / "online_encoder.pth").relative_to(REPO)),
                     "dsD": str((IMM_CK / "E" / "epoch60" / "online_encoder.pth").relative_to(REPO))},
         "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    for a, d in s["pretraining_seconds"].items():
        d["total"] = float(sum(v for k, v in d.items() if k != "total"))
    s["source_sha256"] = {a: T.sha(REPO / p) for a, p in s["sources"].items()}
    RES.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(s, indent=1) + "\n")
    return s


def build_starts():
    import torch
    CK.mkdir(parents=True, exist_ok=True)
    rnd = torch.load(EARLIER_CK / "start_rnd.pth", map_location="cpu", weights_only=False)
    sd = rnd["state_dict"]
    fe = {k for k in sd if k.startswith("feature_exact.")}
    shutil.copy2(EARLIER_CK / "start_rnd.pth", CK / "start_dsA.pth")
    shutil.copy2(EARLIER_CK / "start_pre.pth", CK / "start_dsB.pth")
    for arm, k in (("dsC", "B"), ("dsD", "E")):
        enc = torch.load(IMM_CK / k / "epoch60" / "online_encoder.pth", map_location="cpu")
        assert {f"feature_exact.{q}" for q in enc} == fe
        new = dict(rnd); new["state_dict"] = {q: (enc[q[len("feature_exact."):]].clone() if q in fe else v.clone()) for q, v in sd.items()}
        torch.save(new, CK / f"start_{arm}.pth")
    st = {a: torch.load(CK / f"start_{a}.pth", map_location="cpu", weights_only=False)["state_dict"] for a in ARMS}
    chk = {"start_dsA_equals_earlier_start_rnd": T.sha(CK / "start_dsA.pth") == T.sha(EARLIER_CK / "start_rnd.pth"),
           "start_dsB_equals_earlier_start_pre": T.sha(CK / "start_dsB.pth") == T.sha(EARLIER_CK / "start_pre.pth")}
    for a in ARMS:
        chk[f"{a}_identical_outside_encoder_vs_A"] = all(torch.equal(st[a][q], sd[q]) for q in sd if q not in fe)
        chk[f"{a}_same_keys_and_shapes"] = {q: v.shape for q, v in st[a].items()} == {q: v.shape for q, v in sd.items()}
    for a, b in (("dsA", "dsB"), ("dsA", "dsC"), ("dsA", "dsD"), ("dsB", "dsC"), ("dsB", "dsD"), ("dsC", "dsD")):
        chk[f"encoders_differ_{a}_{b}"] = any(not torch.equal(st[a][q], st[b][q]) for q in fe)
    for arm, k in (("dsC", "B"), ("dsD", "E")):
        enc = torch.load(IMM_CK / k / "epoch60" / "online_encoder.pth", map_location="cpu")
        chk[f"{arm}_encoder_equals_epoch60_online_encoder"] = all(torch.equal(st[arm]["feature_exact." + q], v) for q, v in enc.items())
    assert all(chk.values()), chk
    (CK / "SHA256SUMS.json").write_text(json.dumps({p.name: T.sha(p) for p in sorted(CK.glob("start_*.pth"))}, indent=1) + "\n")
    return chk


def protocol_check(s):
    """Command identical to the earlier random arm's except run name, start file and the snapshot flag."""
    T.CK = EARLIER_CK
    old = _args("rnd", s)
    T.CK = CK
    out = {}
    for arm in ARMS:
        new = T.args(arm, s)
        assert new[:len(old)] != old and len(new) == len(old) + 2 and new[len(old)] == "--budget_checkpoints"
        diff = [(x, y) for x, y in zip(old, new[:len(old)]) if x != y]
        assert len(diff) == 2, diff
        out[arm] = [list(d) for d in diff]
    return out


def collect_outputs():
    dan = T.DAN / "trained_network" / "SD2"
    for arm in ARMS:
        d = CK / arm; d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dan / f"{T.name(arm)}.pth", d / "selected.pth")
        for u in SNAPS:
            shutil.copy2(dan / f"{T.name(arm)}@u{u}.pth", d / f"u{u}.pth")


def record_check():
    out = {}
    old = json.loads((EARLIER / "rnd_run.json").read_text())
    for arm in ARMS:
        new = json.loads((RES / f"{arm}_run.json").read_text())
        keys = set(old["config"]) | set(new["config"])
        diff = sorted(k for k in keys if old["config"].get(k) != new["config"].get(k))
        out[arm] = {"config_keys_differing_vs_earlier_random_arm": diff,
                    "data_fingerprints_equal": old["data_fingerprints"] == new["data_fingerprints"],
                    "protocol_matches": set(diff) <= ALLOWED_DIFF and old["data_fingerprints"] == new["data_fingerprints"]}
    for arm, e in (("dsA", "rnd"), ("dsB", "pre")):
        a = json.loads((RES / f"{arm}_valiquality.json").read_text()); b = json.loads((EARLIER / f"{e}_valiquality.json").read_text())
        out[f"{arm}_reproduces_earlier_{e}_curve"] = [x["selection_metric"] for x in a] == [x["selection_metric"] for x in b]
    return out


def evaluate():
    f = RES / "evaluation.json"
    if f.exists():
        return json.loads(f.read_text())
    models = [str(CK / arm / n) for arm in ARMS for n in ("selected.pth", "u2000.pth")]
    out_dir = RES / "eval"
    t0 = time.time()
    with open(RES / "eval.stdout.log", "w") as log:
        r = subprocess.run([T.PY, "evaluate_v3.py", "--models", *models, "--split", "evaluation", "--size", "10x5",
                            "--sample", "128", "--sample_seed", "51", "--out", str(out_dir), "--dispatch_rule", "semi_active"],
                           cwd=T.DAN, env=T.ENV, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit("evaluate_v3.py failed; see eval.stdout.log")
    summ = json.loads(next(out_dir.rglob("summary.json")).read_text())
    E = {"n": summ["n"], "t_ref_mean": summ.get("t_ref"), "manifest_sha256": summ["manifest_sha256"], "seconds": time.time() - t0, "arms": {}}
    for arm in ARMS:
        E["arms"][arm] = {n.split(".")[0]: summ["models"][str(CK / arm / n)] for n in ("selected.pth", "u2000.pth")}
    f.write_text(json.dumps(E, indent=1) + "\n")
    return E


def curve(arm):
    return json.loads((RES / f"{arm}_valiquality.json").read_text())


def summarise(s, E):
    import numpy as np
    rows = {}
    for arm in ARMS:
        v = curve(arm)
        u = np.array([e["update"] for e in v], float); y = np.array([e["selection_metric"] for e in v])
        tard = np.array([e["tardiness"] for e in v]); sec = np.array([e["train_seconds"] for e in v])
        hit = next((i for i, e in enumerate(v) if e["selection_metric"] <= s["target"] + 1e-12), None)
        b = int(np.argmin(y))
        pre = s["pretraining_seconds"][arm]["total"]
        m50 = u >= 50
        rows[arm] = {"start": float(y[0]), "start_tardiness": float(tard[0]),
                     "at": {int(k): float(y[u == k][0]) for k in (250, 500, 1000, 1500, 2000)},
                     "best": float(y[b]), "best_update": int(u[b]), "best_tardiness": float(tard[b]),
                     "final": float(y[-1]), "final_tardiness": float(tard[-1]),
                     "threshold_update": int(u[hit]) if hit is not None else None,
                     "threshold_specialist_hours": float(sec[hit] / 3600) if hit is not None else None,
                     "threshold_total_hours": float((sec[hit] + pre) / 3600) if hit is not None else None,
                     "specialist_hours": float(sec[-1] / 3600), "pretraining_hours": pre / 3600,
                     "total_hours": float((sec[-1] + pre) / 3600),
                     "auc_log10": float(np.trapezoid(np.log10(y), u) / (u[-1] - u[0])),
                     "auc_linear_from_50": float(np.trapezoid(y[m50], u[m50]) / (u[m50][-1] - u[m50][0]))}
    out = {"rows": rows, "target": s["target"], "budget": s["budget_updates"], "evaluation": E}
    (RES / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def plot(s, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=BG, sharey=True)
    for ax, (key, xl) in zip(axs, (("update", "specialist training updates"), ("spec_h", "specialist training time (hours)"),
                                    ("tot_h", "cumulative time incl. world-model pretraining (hours)"))):
        for arm in ARMS:
            v = curve(arm); pre = s["pretraining_seconds"][arm]["total"]
            x = [e["update"] if key == "update" else e["train_seconds"] / 3600 + (pre / 3600 if key == "tot_h" else 0) for e in v]
            ax.plot(x, [e["selection_metric"] for e in v], color=COL[arm], lw=1.8, marker="o", ms=2.3, label=LABEL[arm])
        ax.axhline(s["target"], color=MUTED, ls="--", lw=.9)
        ax.set_yscale("log"); ax.set_xlabel(xl); ax.grid(color=GRID, lw=.7); ax.set_facecolor(BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axs[0].set_ylabel("validation tardiness, T / Tref (log scale)")
    axs[0].legend(frameon=False, fontsize=8.5)
    axs[0].text(2000, s["target"] * 1.06, f"common threshold {s['target']:.4f}", ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.suptitle("Tardiness specialist from four encoder initialisations (10x5, waiting allowed, seed 111)", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.01, "18 validation instances, greedy, every 50 updates. Single seed: treat differences as indicative only.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(RES / "curves.png", dpi=140, facecolor=BG); plt.close(fig)
    # zoomed late-phase panel (updates 500..2000, linear scale)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    for arm in ARMS:
        v = [e for e in curve(arm) if e["update"] >= 500]
        ax.plot([e["update"] for e in v], [e["selection_metric"] for e in v], color=COL[arm], lw=1.8, marker="o", ms=2.5, label=LABEL[arm])
    ax.axhline(s["target"], color=MUTED, ls="--", lw=.9, label=f"common threshold {s['target']:.4f}")
    ax.set_xlabel("specialist training updates"); ax.set_ylabel("validation tardiness, T / Tref")
    ax.set_title("Late phase (updates 500-2,000, linear scale)", fontsize=11, loc="left")
    ax.grid(color=GRID, lw=.7); ax.legend(frameon=False, fontsize=8.5); ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "curves_late.png", dpi=140, facecolor=BG); plt.close(fig)


def report(s, S, chk):
    f3 = lambda x: "-" if x is None else f"{x:.4f}"
    f2 = lambda x: "-" if x is None else f"{x:.2f}"
    R = S["rows"]; E = S["evaluation"]
    L = ["# Downstream tardiness specialist from world-model encoders (10x5, waiting allowed, seed 111)", "",
         "Exploratory downstream experiment following `results/world_model_joint_pilot_v3.md` and "
         "`results/world_model_immediate_pilot_v3.md`; it does not revise either experiment's decision rule. Single seed.", "",
         "## Arms (identical except the encoder weights of the start bundle)", "",
         "| arm | encoder | source | pretraining (s) |", "|---|---|---|---|"]
    desc = {"dsA": "untrained DANIEL encoder (seed-111 start)", "dsB": "revised outcome-prediction world model (action pilot: pair features, abs+diff; final outcomes under continuation rules)",
            "dsC": "immediate physical prediction only (immediate pilot B, 60 epochs)", "dsD": "joint latent + immediate physical prediction (immediate pilot E, 60 epochs)"}
    for a in ARMS:
        L.append(f"| {LABEL[a]} | {desc[a]} | `{s['sources'][a]}` | {s['pretraining_seconds'][a]['total']:.0f} |")
    L += ["", "- Recipe: the earlier tardiness pilot's own command (contrastive fine-tuning, cl_k 8, cl_beta 0.1, lr 3e-4, "
              "20 environments, generator training stream, seed 111, `--dispatch_rule semi_active`), 2,000 updates, "
              "validation every 50 updates (greedy, 18 validation instances, T / Tref), no early stopping; encoder fine-tuned "
              "with the policy; no world-model loss during specialist training. Snapshots every 50 updates.",
          f"- Checks: start bundles identical outside the encoder, encoders differ pairwise, C/D encoders equal the epoch-60 "
          f"checkpoints; commands differ from the earlier random arm only in run name, start file and the snapshot flag; "
          f"saved run configs differ only in {sorted(ALLOWED_DIFF)} with equal data fingerprints: "
          f"{all(v['protocol_matches'] for k, v in chk['record'].items() if k in ARMS)}.",
          f"- Reproducibility: A reproduces the earlier random arm's validation curve exactly: {chk['record']['dsA_reproduces_earlier_rnd_curve']}; "
          f"B reproduces the earlier pretrained arm's curve exactly: {chk['record']['dsB_reproduces_earlier_pre_curve']}.",
          f"- Common threshold (fixed before launch, from existing validation evidence): T / Tref <= {s['target']:.4f}, "
          f"the reference tardiness specialist's best validation value within 2,000 updates (same recipe and protocol).",
          f"- AUC: {s['auc']['primary']}; secondary {s['auc']['secondary']}.",
          "- Pretraining cost: B = data regeneration + encoder training of the action pilot; C, D = the joint pilot's episodes "
          "and branching + physical-target extraction + 60 epochs of training (continuation rollouts not needed, excluded).", "",
          "![curves](tardiness_wm_downstream_v3/curves.png)", "", "![late phase](tardiness_wm_downstream_v3/curves_late.png)", "",
          "## Validation (18 instances, greedy; T / Tref, lower is better)", "",
          "| arm | before training | @250 | @500 | @1000 | @1500 | @2000 (final) | best (update) | AUC log10 | AUC linear (50-2000) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        r = R[a]
        L.append(f"| {LABEL[a]} | {r['start']:.3f} | {f3(r['at'][250])} | {f3(r['at'][500])} | {f3(r['at'][1000])} | {f3(r['at'][1500])} | "
                 f"{f3(r['final'])} | {f3(r['best'])} ({r['best_update']}) | {r['auc_log10']:.4f} | {r['auc_linear_from_50']:.4f} |")
    L += ["", "Actual mean total tardiness on the validation instances (minutes): " +
          "; ".join(f"{LABEL[a]}: start {R[a]['start_tardiness']:.0f}, best {R[a]['best_tardiness']:.1f}, final {R[a]['final_tardiness']:.1f}" for a in ARMS) + ".", "",
          "## Threshold and cost", "",
          "| arm | threshold reached at update | specialist hours to threshold | incl. pretraining | specialist hours (2,000 updates) | pretraining hours | total hours |",
          "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        r = R[a]
        L.append(f"| {LABEL[a]} | {r['threshold_update'] if r['threshold_update'] is not None else 'not reached'} | "
                 f"{f2(r['threshold_specialist_hours'])} | {f2(r['threshold_total_hours'])} | {r['specialist_hours']:.2f} | "
                 f"{r['pretraining_hours']:.3f} | {r['total_hours']:.2f} |")
    L += ["", "The four arms ran concurrently on the same machine (3 CPU cores each, one shared GPU), so their wall-clock times are "
              "comparable with each other, not with the earlier pilot's runs.", "",
          f"## Held-out evaluation ({E['n']} instances of the 10x5 evaluation split; reserved final-test sets not used)", "",
          "| arm | checkpoint | greedy T / Tref | greedy mean tardiness (min) | greedy tardy-job share | best-of-128 T / Tref | best-of-128 mean tardiness (min) | greedy makespan | greedy carbon |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for c, lab in (("selected", f"selected (update {E['arms'][a]['selected'].get('update')})"), ("u2000", "final (2,000)")):
            g = E["arms"][a][c]["greedy"]; b = E["arms"][a][c]["sample128"]
            L.append(f"| {LABEL[a]} | {lab} | {g['tardiness_ratio']:.4f} | {g['tardiness']:.1f} | {g['tardy_share']:.3f} | "
                     f"{b['tardiness_ratio']:.4f} | {b['tardiness']:.1f} | {g['makespan']:.1f} | {g['carbon']:.2f} |")
    L += ["", "## Interpretation", "", "INTERPRETATION_PLACEHOLDER", "",
          "## Limitations", "",
          "- One seed (111) for every arm: run-to-run variation of the specialist is not measured, so gaps of a few hundredths "
          "in T / Tref are not evidence on their own.",
          "- 18 validation instances drive selection and the threshold; the held-out evaluation uses the evaluation split.",
          "- Encoders are fine-tuned with the policy, so the comparison is about the starting point, not a frozen representation.",
          "- B's encoder was pretrained with a different (no-delay) rule and a different data set from C and D; pretraining "
          "budgets differ between B and C/D (C and D are matched to each other).",
          "- Only the tardiness specialist; makespan and carbon specialists and Rewarded Soups are not tested here.", ""]
    (REPO / "results" / f"{NAME}.md").write_text("\n".join(L) + "\n")


def main():
    s = settings()
    chk = {"start": build_starts(), "command": protocol_check(s)}
    (RES / "start_check.json").write_text(json.dumps(chk, indent=1) + "\n")
    print(json.dumps(chk), flush=True)
    out = {}
    th = [threading.Thread(target=T.run_arm, args=(a, s, out)) for a in ARMS]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if any(out.get(a) != "ok" for a in ARMS):
        raise SystemExit(f"an arm failed: {out}")
    collect_outputs()
    chk["record"] = record_check()
    (RES / "start_check.json").write_text(json.dumps(chk, indent=1) + "\n")
    E = evaluate()
    S = summarise(s, E)
    plot(s, S)
    report(s, S, chk)
    print(json.dumps(S["rows"], indent=1), flush=True)


if __name__ == "__main__":
    main()
