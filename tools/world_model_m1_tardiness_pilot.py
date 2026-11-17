"""Matched 10x5 tardiness specialist pilot, waiting allowed: does the revised world-model encoder (pilot checkpoint with
job-machine pair features and the action-difference loss, pretrained under the no-delay rule) help DANIEL learn
faster? Two arms, identical except the encoder weights:
  pretrained  the M1 start DANIEL bundle (seed-111 actor and critic) with feature_exact replaced by the pilot encoder
  random      the M1 start DANIEL bundle itself (the seed-111 random encoder of the frozen-encoder comparison)
Both: the M2 single-objective contrastive recipe (configs/contrastive_v3.json constants, goal t, seed 111, generator
stream), --dispatch_rule semi_active for training and validation, the encoder trained with the policy, validation
every 50 updates (greedy, 18 validation instances, T / Tref), fixed budget, no stopping rule.
Budget and the common target are written to the settings file BEFORE either arm starts; the target is the reference
run's best validation tardiness within the same budget (results/specialist_reuse_v3, bv3_sr_spec_t_s111: the same
recipe, validation protocol and waiting rule). Earlier results and checkpoints are only read; the reserved final-test
sets are never loaded.

    python tools/world_model_m1_tardiness_pilot.py     # -> results/world_model_m1_v3/tardiness_pilot/
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
PY = str(REPO / ".venv" / "bin" / "python")
RES = REPO / "results" / "world_model_m1_v3" / "tardiness_pilot"
CK = REPO / "checkpoints" / "world_model_m1_v3_tardiness_pilot"
REF = REPO / "results" / "specialist_reuse_v3" / "runs" / "10x5+mix+bv3_sr_spec_t_s111"
PILOT_CK = REPO / "checkpoints" / "world_model_m1_v3_action_pilot" / "pilot_pairs_abs+diff.pth"
START = REPO / "checkpoints" / "world_model_m1_v3" / "world_model_trunk_start.pth"
BUDGET = 2000
ARMS = {"pre": ("pretrained encoder", "0-3"), "rnd": ("random encoder", "4-7")}
ENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", WANDB_MODE="offline", PYTHONUNBUFFERED="1")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name(arm):
    return f"10x5+mix+bv3_wmtp_{arm}_s111"


def settings():
    f = RES / "settings.json"
    if f.exists():
        return json.loads(f.read_text())
    v = json.loads((REF / "valiquality.json").read_text())
    within = [e for e in v if e["update"] <= BUDGET]
    best = min(within, key=lambda e: e["selection_metric"])
    c = json.loads((REPO / "configs" / "contrastive_v3.json").read_text())["constants"]
    s = {"budget_updates": BUDGET, "target": best["selection_metric"], "target_reached_by_reference_at": best["update"],
         "target_rule": "the reference run's best validation tardiness (T / Tref, greedy, 18 validation instances, "
                        "every 50 updates, waiting rule) within the same update budget",
         "reference": {"run": str(REF.relative_to(REPO)), "valiquality_sha256": sha(REF / "valiquality.json")},
         "recipe": {**{k: c[k] for k in ("cl_k", "cl_beta", "cl_margin", "cl_grad_clip", "lr", "minibatch_size",
                                         "num_envs", "reset_env_timestep", "validate_timestep")},
                    "goal": "t", "seed_train": 111, "dispatch_rule": "semi_active", "train_source": "generator"},
         "sources": {"pilot_encoder": {"path": str(PILOT_CK.relative_to(REPO)), "sha256": sha(PILOT_CK)},
                     "start_bundle": {"path": str(START.relative_to(REPO)), "sha256": sha(START)}},
         "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(s, indent=1) + "\n")
    return s


def build_starts():
    """Two DANIEL bundles identical outside feature_exact; checked."""
    import torch
    CK.mkdir(parents=True, exist_ok=True)
    b = torch.load(START, map_location="cpu", weights_only=False)
    pilot = torch.load(PILOT_CK, map_location="cpu", weights_only=False)
    trunk = {k[len("trunk."):]: v for k, v in pilot.items() if k.startswith("trunk.")}
    sd = b["state_dict"]
    fe = {k for k in sd if k.startswith("feature_exact.")}
    assert {f"feature_exact.{k}" for k in trunk} == fe, "pilot encoder keys differ from DANIEL's feature_exact"
    pre = dict(b); pre["state_dict"] = {k: (trunk[k[len("feature_exact."):]].clone() if k in fe else v.clone())
                                       for k, v in sd.items()}
    assert all(pre["state_dict"][k].shape == sd[k].shape for k in sd)
    torch.save(pre, CK / "start_pre.pth"); shutil.copy2(START, CK / "start_rnd.pth")
    a, r = torch.load(CK / "start_pre.pth", weights_only=False)["state_dict"], torch.load(CK / "start_rnd.pth", weights_only=False)["state_dict"]
    same_outside = all(torch.equal(a[k], r[k]) for k in a if k not in fe)
    differ_inside = any(not torch.equal(a[k], r[k]) for k in fe)
    assert same_outside and differ_inside
    (CK / "SHA256SUMS.json").write_text(json.dumps({p.name: sha(p) for p in sorted(CK.glob("*.pth"))}, indent=1) + "\n")
    return {"identical_outside_encoder": same_outside, "encoders_differ": differ_inside}


def args(arm, s):
    c = s["recipe"]
    return ["train.py", "--config", str(REPO / "configs/benchmark_v3/t.json"), "--n_j", "10", "--n_m", "5",
            "--model_suffix", f"bv3_wmtp_{arm}_s111", "--seed_train", "111", "--v3_train_source", "generator",
            "--finetune_objective", "contrastive", "--cl_k", str(c["cl_k"]), "--cl_beta", str(c["cl_beta"]),
            "--cl_margin", str(c["cl_margin"]), "--cl_grad_clip", str(c["cl_grad_clip"]), "--lr", str(c["lr"]),
            "--minibatch_size", str(c["minibatch_size"]), "--num_envs", str(c["num_envs"]),
            "--reset_env_timestep", str(c["reset_env_timestep"]), "--validate_timestep", str(c["validate_timestep"]),
            "--converge_window", "0", "--max_updates", str(s["budget_updates"]), "--chunk_updates", "500",
            "--init_from", str(CK / f"start_{arm}.pth"), "--init_mode", "exact", "--resume", "True", "--device", "cuda",
            "--dispatch_rule", "semi_active"]


def run_arm(arm, s, out):
    st = DAN / "train_log" / "SD2" / f"{name(arm)}.status.json"
    log = open(RES / f"{arm}.stdout.log", "a")
    while not (st.exists() and json.loads(st.read_text())["status"] in ("FINISHED", "CONVERGED")):
        r = subprocess.run(["taskset", "-c", ARMS[arm][1], PY, *args(arm, s)], cwd=DAN, env=ENV, stdout=log,
                           stderr=subprocess.STDOUT)
        if r.returncode != 0:
            out[arm] = f"train.py failed (rc {r.returncode})"; return
    for f, dst in ((f"valiquality_{name(arm)}.json", "valiquality"), (f"reward_{name(arm)}.json", "reward"),
                   (f"{name(arm)}.status.json", "status"), (f"{name(arm)}.run.json", "run")):
        shutil.copy2(DAN / "train_log" / "SD2" / f, RES / f"{arm}_{dst}.json")
    out[arm] = "ok"


def summarise(s):
    pre_cost = json.loads((REPO / "results/world_model_m1_v3/action_pilot.json").read_text())
    rows = {}
    for arm in ARMS:
        v = json.loads((RES / f"{arm}_valiquality.json").read_text())
        hit = next((e for e in v if e["selection_metric"] <= s["target"] + 1e-12), None)
        last = v[-1]
        rows[arm] = {"start": v[0]["selection_metric"], "reached_update": hit["update"] if hit else None,
                     "reached_hours": hit["train_seconds"] / 3600 if hit else None,
                     "final_update": last["update"], "final": last["selection_metric"],
                     "best": min(e["selection_metric"] for e in v), "hours_total": last["train_seconds"] / 3600}
    s2 = {"rows": rows, "pretraining_cost": {
        "pilot_data_regeneration_s": pre_cost["pilot_data"]["train_regeneration_seconds"],
        "pilot_encoder_training_s": pre_cost["pilot"]["pairs|abs+diff"]["seconds"]}}
    (RES / "summary.json").write_text(json.dumps(s2, indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for arm, col in (("pre", "#d62728"), ("rnd", "#7f7f7f")):
        v = json.loads((RES / f"{arm}_valiquality.json").read_text())
        ax.plot([e["update"] for e in v], [e["selection_metric"] for e in v], color=col, lw=1.8, marker="o", ms=2.5,
                label=ARMS[arm][0])
    ax.axhline(s["target"], color="#555555", ls="--", lw=.9, label=f"common target {s['target']:.4f}")
    ax.set_yscale("log"); ax.set_xlabel("updates"); ax.set_ylabel("validation tardiness (T / Tref, log scale)")
    ax.set_title("Tardiness specialist, waiting allowed: pretrained vs random encoder (10x5, seed 111)", fontsize=11)
    ax.grid(alpha=.3); ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "curves.png", dpi=140, facecolor="white"); plt.close(fig)
    return s2


def main():
    s = settings()
    print("settings:", json.dumps({k: s[k] for k in ("budget_updates", "target", "target_reached_by_reference_at")}), flush=True)
    chk = build_starts()
    (RES / "start_check.json").write_text(json.dumps(chk, indent=1) + "\n")
    out = {}
    th = [threading.Thread(target=run_arm, args=(a, s, out)) for a in ARMS]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if any(v != "ok" for v in out.values()):
        raise SystemExit(f"an arm failed: {out}")
    print(json.dumps(summarise(s), indent=1))


if __name__ == "__main__":
    main()
