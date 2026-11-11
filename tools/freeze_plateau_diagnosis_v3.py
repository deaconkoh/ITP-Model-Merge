"""Frozen settings of the plateau diagnosis (docs/plateau_diagnosis_protocol_v3.md): the five converged runs whose logs
are read (SHA-256), the Part A checkpoints along training (source, how obtained, expected update and start-network
hash), the evaluation settings (greedy and best-of-16), and the Part B arms (World-model base mix start, one change
each, 1,500 updates fixed). Every step reads configs/plateau_diagnosis_v3.json through load(), which fails loudly if a
source changed.

    python tools/freeze_plateau_diagnosis_v3.py freeze      # once, before any run
    python tools/freeze_plateau_diagnosis_v3.py verify
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs" / "plateau_diagnosis_v3.json"
M3 = "configs/world_model_m3_v3.json"
CC, CL, WM3 = "results/contrastive_converged_v3/runs", "results/contrastive_v3/runs", "results/world_model_m3_v3/runs"
TN = "daniel/trained_network/SD2"
RUNS = {  # key: (label, run dir with reward/valiquality/status, start-network hash from the resume identity)
    "fs": ("From scratch", f"{CC}/10x5+mix+bv3_cc_CCB_s111"),
    "dist": ("Distillation", f"{CC}/10x5+mix+bv3_cc_CCD_s111"),
    "wm": ("World-model base mix", f"{WM3}/10x5+mix+bv3_wm3_wm_s111"),
    "ctrl": ("Random shared base mix", f"{WM3}/10x5+mix+bv3_wm3_ctrl_s111"),
    "orig": ("Original specialists mix", f"{WM3}/10x5+mix+bv3_wm3_orig_s111"),
}
# Part A checkpoints. kind: copy = a saved best bundle; resume = the policy of a saved final (resume) state, wrapped
# as a bundle; rebuild = the start network re-created by train.py with --max_updates 0 from the run's own settings
# (no training) and checked against the run's logged start-network hash.
CKPTS = [
    ("fs", 0, "rebuild", f"{CL}/10x5+mix+bv3_cl_CLB_s111/run.json"),
    ("fs", 2450, "copy", f"{TN}/10x5+mix+bv3_cl_CLB_s111.pth"),
    ("fs", 2600, "resume", "checkpoints/contrastive_converged_v3/sources/10x5+mix+bv3_cl_CLB_s111.resume.pth"),
    ("fs", 4150, "copy", "checkpoints/contrastive_converged_v3/10x5+mix+bv3_cc_CCB_s111.pth"),
    ("fs", 4550, "resume", f"{TN}/10x5+mix+bv3_cc_CCB_s111.resume.pth"),
    ("dist", 0, "rebuild", f"{CL}/10x5+mix+bv3_cl_CLD_s111/run.json"),
    ("dist", 900, "copy", "checkpoints/contrastive_converged_v3/10x5+mix+bv3_cc_CCD_s111.pth"),
    ("dist", 1000, "resume", "checkpoints/contrastive_converged_v3/sources/10x5+mix+bv3_cl_CLD_s111.resume.pth"),
    ("dist", 1900, "resume", f"{TN}/10x5+mix+bv3_cc_CCD_s111.resume.pth"),
    ("wm", 0, "copy", "checkpoints/world_model_m3_v3/10x5+mix+bv3_wm3_wm_s111_start.pth"),
    ("wm", 1650, "copy", "checkpoints/world_model_m3_v3/10x5+mix+bv3_wm3_wm_s111.pth"),
    ("wm", 2250, "resume", f"{TN}/10x5+mix+bv3_wm3_wm_s111.resume.pth"),
    ("ctrl", 0, "copy", "checkpoints/world_model_m3_v3/10x5+mix+bv3_wm3_ctrl_s111_start.pth"),
    ("ctrl", 2650, "copy", "checkpoints/world_model_m3_v3/10x5+mix+bv3_wm3_ctrl_s111.pth"),
    ("ctrl", 3650, "resume", f"{TN}/10x5+mix+bv3_wm3_ctrl_s111.resume.pth"),
    ("orig", 0, "copy", "checkpoints/world_model_m3_v3/10x5+mix+bv3_wm3_orig_s111_start.pth"),
    ("orig", 1650, "copy", "checkpoints/world_model_m3_v3/10x5+mix+bv3_wm3_orig_s111.pth"),
    ("orig", 2650, "resume", f"{TN}/10x5+mix+bv3_wm3_orig_s111.resume.pth"),
]
START_SHA_FROM = {"fs": f"{TN}/10x5+mix+bv3_cc_CCB_s111.resume.pth", "dist": f"{TN}/10x5+mix+bv3_cc_CCD_s111.resume.pth",
                  "wm": f"{TN}/10x5+mix+bv3_wm3_wm_s111.resume.pth", "ctrl": f"{TN}/10x5+mix+bv3_wm3_ctrl_s111.resume.pth",
                  "orig": f"{TN}/10x5+mix+bv3_wm3_orig_s111.resume.pth"}
ARMS = {  # Part B: one change each from the M3 World-model base mix run
    "beta05": ("Stronger push (beta 0.5)", {"cl_beta": 0.5}),
    "lr1e3": ("Faster learning (lr 1e-3)", {"lr": 0.001}),
    "k16": ("More samples (K 16)", {"cl_k": 16}),
    "ctrl": ("Control (M3 settings)", {}),
}


def sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def torch_load(path):
    """torch.load of a training state: its pickle imports daniel modules (benchmark_v3, which reads params), so daniel
    goes first on the path (tools/benchmark_v3 is a directory) and the command line is hidden from params."""
    import torch
    if str(REPO / "daniel") != sys.path[0]:
        sys.path.insert(0, str(REPO / "daniel"))
    if "benchmark_v3" in sys.modules and getattr(sys.modules["benchmark_v3"], "__file__", None) is None:
        del sys.modules["benchmark_v3"]
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        sys.argv = argv


def start_hashes():
    """The start-network hash each run logged (cl_reference_sha256 of its resume identity)."""
    return {k: torch_load(REPO / p)["identity"]["cl_reference_sha256"] for k, p in START_SHA_FROM.items()}


def build():
    m3 = json.loads((REPO / M3).read_text())
    runs = {k: {"label": lab, "dir": d, **{f: sha(f"{d}/{f}") for f in ("reward.json", "valiquality.json",
                                                                          "status.json")}}
            for k, (lab, d) in RUNS.items()}
    starts = start_hashes()
    ck = [{"run": r, "update": u, "kind": kind, "source": src, "source_sha256": sha(src),
           "name": f"pd_{r}_u{u}"} for r, u, kind, src in CKPTS]
    c = m3["part_b"]["contrastive"]
    return {
        "size": "10x5", "seed": 111, "runs": runs, "start_network_sha256": starts,
        "start_network_sha256_source": START_SHA_FROM,
        "part_a": {
            "checkpoints": ck,
            "signals": ["loss", "entropy", "distinct_schedules", "pairs_per_group", "groups_with_pair", "grad_norm",
                        "weight_drift"],
            "smooth_updates": 50, "log2": 0.6931471805599453,
            "evaluation": {"split": "evaluation", "grid_step": 0.1, "sample": 16, "sample_seed": 50,
                           "norm": m3["part_a"]["eval_norm"], "norm_sha256": m3["part_a"]["eval_norm_sha256"]},
            "noise": {"bootstrap": 10000, "bootstrap_seed": 2026},
            "b2_root": m3["part_a"]["b2_root"]},
        "part_b": {
            "base": {"source": M3, "sha256": sha(M3), "set": "wm", "specialists": m3["sets"]["wm"]["specialists"],
                     "contrastive": c},
            "arms": {k: {"label": lab, "change": ch} for k, (lab, ch) in ARMS.items()},
            "updates": 1500, "validate_timestep": 50, "chunk": 500, "stopping_rule": "none (fixed length)",
            "budget_checkpoints": [0, 250, 500, 750, 1000, 1250, 1500],
            "evaluation": "final (update 1500) and best-validation checkpoints: greedy and best-of-16 on the "
                          "evaluation split (66 preferences, 100 instances)",
            "order": ["k16", "beta05", "lr1e3", "ctrl"]},
        "smoke": {"updates": 5, "validate_timestep": 5, "mo_val_first": 3, "budget_checkpoints": [0, 5],
                  "eval_split": "validation", "eval_first": 3, "norm": m3["part_a"]["val_norm"]},
    }


def load(check=True):
    m = json.loads(MANIFEST.read_text())
    if check:
        cur = build()
        for k in cur:
            if m.get(k) != cur[k]:
                raise SystemExit(f"configs/plateau_diagnosis_v3.json: '{k}' no longer matches its sources; nothing started")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["freeze", "verify"])
    a = ap.parse_args()
    if a.action == "freeze":
        if MANIFEST.exists():
            sys.exit(f"{MANIFEST} exists (frozen); delete it deliberately to re-freeze")
        m = build(); m["frozen"] = time.strftime("%Y-%m-%d")
        MANIFEST.write_text(json.dumps(m, indent=1) + "\n")
        print(f"frozen: {MANIFEST}")
    load()
    print("plateau diagnosis settings OK")


if __name__ == "__main__":
    main()
