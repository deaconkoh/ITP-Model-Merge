"""Frozen settings of world-model M3 (docs/world_model_m3_protocol_v3.md): the four specialist sets (SHA-256), the merge,
Part A settings (grids, corner threshold, ridge test, regret pool), the Part B contrastive constants and stopping rule,
the pre-registered criteria, references and cost sources. Every M3 step reads configs/world_model_m3_v3.json through
load(), which fails loudly if a source changed.

    python tools/freeze_world_model_m3_v3.py freeze      # once, before any run
    python tools/freeze_world_model_m3_v3.py verify
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs" / "world_model_m3_v3.json"
CL_CONFIG = "configs/contrastive_v3.json"
SETS = {
    "wm": ("World-model base mix", "World-model base", "checkpoints/world_model_m2_v3/10x5+mix+bv3_wm2_wm_{o}_s111.pth",
           True),
    "ctrl": ("Random shared base mix", "Random shared base",
             "checkpoints/world_model_m2_v3/10x5+mix+bv3_wm2_ctrl_{o}_s111.pth", True),
    "orig": ("Original specialists mix", "Original specialists",
             "checkpoints/specialists_v3/10x5+mix+bv3_spec_{o}_s111.pth", True),
    "old": ("Old shared-base specialists mix", "Old shared-base specialists",
            "checkpoints/merge_categories_v3/10x5+mix+bv3_mcft_{o}_s111.pth", False),
}
COST = {      # training hours of the specialists (status.json train_seconds), per set
    "wm": ["results/world_model_m2_v3/runs/10x5+mix+bv3_wm2_wm_{o}_s111/status.json"],
    "ctrl": ["results/world_model_m2_v3/runs/10x5+mix+bv3_wm2_ctrl_{o}_s111/status.json"],
    "orig": ["results/baselines_v3/runs/10x5+mix+bv3_spec_{o}_s111/status.json"],
    "old": ["results/merge_categories_v3/shared_base/10x5_s111/ft_{o}.status.json"],
}
OLD_BASE_STATUS = "results/merge_categories_v3/shared_base/10x5_s111/base.status.json"
VAL_NORM = "results/warmstart_v3/norm/validation_normalisation_points_10x5.json"
EVAL_NORM = "results/baselines_v3/normalisation_points.json"
E0 = {s: f"results/warmstart_v3/eval/evaluation_10x5/10x5+mix+bv3_ws_E0_s{s}" for s in (111, 222, 333)}
CC = {"From scratch": ("results/contrastive_converged_v3/eval/evaluation_10x5/10x5+mix+bv3_cc_CCB_s111",
                       "results/contrastive_converged_v3/runs/10x5+mix+bv3_cc_CCB_s111"),
      "Distillation": ("results/contrastive_converged_v3/eval/evaluation_10x5/10x5+mix+bv3_cc_CCD_s111",
                       "results/contrastive_converged_v3/runs/10x5+mix+bv3_cc_CCD_s111")}


def sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def build():
    cl = json.loads((REPO / CL_CONFIG).read_text())["constants"]
    sets = {}
    for k, (label, short, pat, part_b) in SETS.items():
        sp = {o: pat.format(o=o) for o in "mct"}
        sets[k] = {"label": label, "short": short, "specialists": sp, "sha256": {o: sha(p) for o, p in sp.items()},
                   "part_b": part_b, "cost_status": {o: [c.format(o=o) for c in COST[k]][0] for o in "mct"},
                   "cost_status_sha256": {o: sha(COST[k][0].format(o=o)) for o in "mct"}}
    sets["old"]["shared_base_status"] = OLD_BASE_STATUS
    sets["old"]["shared_base_status_sha256"] = sha(OLD_BASE_STATUS)
    refs = {"e0": {str(s): {"json": p + ".json", "sha256": sha(p + ".json"), "npz": p + ".npz", "npz_sha256": sha(p + ".npz"),
                            "hv": json.loads((REPO / (p + ".json")).read_text())["hv_mean"]} for s, p in E0.items()}}
    for name, (ev, run) in CC.items():
        refs[name] = {"eval_json": ev + ".json", "eval_sha256": sha(ev + ".json"), "eval_npz": ev + ".npz",
                      "eval_npz_sha256": sha(ev + ".npz"), "status": run + "/status.json",
                      "status_sha256": sha(run + "/status.json"), "valiquality": run + "/valiquality.json",
                      "valiquality_sha256": sha(run + "/valiquality.json")}
    return {
        "size": "10x5", "seed": 111, "sets": sets,
        "merge": {"definition": "for preference w = (w_M, w_C, w_T): theta(w) = w_M theta_M + w_C theta_C + w_T theta_T, "
                                "a separate mix for every preference (every sample in Part B)",
                  "actor_and_trunk": "mixed per preference; trainable in Part B (theta_x receives w_x * dL/dtheta(w))",
                  "critic": "mixed the same way and kept in the checkpoints; never read by the contrastive loss, so it "
                            "receives no gradient",
                  "preference_columns": "every specialist is copied into a preference-conditioned DANIEL with its three "
                                        "w-input columns (first actor and critic layers) at zero, so the mixed columns "
                                        "start at zero and the update-0 network is the plain weight mix for every w; "
                                        "trainable in Part B; w is fed both as mixing weights and as the preference input"},
        "part_a": {"grid_step": 0.1, "val_grid_step": 0.25, "corner": 0.8, "equal_weights": [1 / 3, 1 / 3, 1 / 3],
                   "ridge_t": [round(i / 10, 1) for i in range(11)], "pairs": ["mc", "mt", "ct"],
                   "ridge_split": "validation",
                   "ridge_score": "s(t) = t f_a + (1-t) f_b of theta_t = t theta_a + (1-t) theta_b, f = objective "
                                  "normalised per instance (validation ideal/nadir), mean over instances; barrier = "
                                  "max_t [s(t) - (t s(1) + (1-t) s(0))]",
                   "val_norm": VAL_NORM, "val_norm_sha256": sha(VAL_NORM),
                   "eval_norm": EVAL_NORM, "eval_norm_sha256": sha(EVAL_NORM),
                   "b2_root": "results/baselines_v3/cpsat/b2",
                   "regret_pool": "per split and instance: the references of analyze_warmstart_v3.reference_points "
                                  "(specialists of all seeds, rules, CP-SAT B1), the CP-SAT B2 front where it exists, "
                                  "converged From scratch and Distillation (evaluation split only) and the 66-point "
                                  "fronts of all four sets; Part B arms are scored against this pool plus their own "
                                  "points",
                   "nondominated_share": "per preference, the share of instances whose schedule is not strictly "
                                         "dominated by another schedule of the same set's 66-point front"},
        "part_b": {"contrastive": {"source": CL_CONFIG, "sha256": sha(CL_CONFIG),
                                   **{k: cl[k] for k in ("cl_k", "cl_beta", "cl_margin", "cl_grad_clip", "lr",
                                                         "minibatch_size", "num_envs", "reset_env_timestep",
                                                         "validate_timestep", "mo_val_grid_step", "pref_dirichlet")}},
                   "rule": {"converge_window": 1000, "converge_tol": 0.005, "min_updates": 1000, "cap": 8000,
                            "chunk": 500,
                            "statement": "stop once the best validation HV has improved by less than 0.5% over the last "
                                         "1,000 updates, after at least 1,000 updates; hard cap 8,000 updates, reported "
                                         "as 'capped'"},
                   "evaluation": {"split": "evaluation", "instances": 100, "grid_step": 0.1}},
        "criteria": {
            "mergeable": {"regret_ratio": 0.90, "regret_split": "evaluation", "barrier_min_pairs": 2,
                          "statement": "before fine-tuning, the world-model set has (a) in-between-preference mean "
                                       "regret <= 0.90 x the random-shared-base set's (evaluation split) AND (b) a "
                                       "smaller barrier on at least 2 of 3 pairs (validation)"},
            "better": {"margin": 0.05, "statement": "the fine-tuned World-model base mix has evaluation HV >= 1.05 x "
                                                    "the fine-tuned Random shared base mix's (best checkpoints)"},
            "faster": {"target_hv": 0.542, "ratio": 0.70,
                       "statement": "the World-model base mix first reaches validation HV 0.542 (From scratch's best) "
                                    "in <= 70% of the updates the Random shared base mix needs; 'no' if it never "
                                    "reaches it; 'yes (control not reached)' if only the control never does"}},
        "references": refs,
        "world_model_cost_h": {"data": 0.22, "pretrain": 1.05, "total": 1.27,
                               "source": "results/world_model_m1_v3/pretrain.json, data_train_meta.json"},
        "smoke": {"cap": 5, "chunk": 0, "validate_timestep": 5, "mo_val_first": 3, "eval_split": "validation",
                  "eval_first": 3, "part_a_first": 3},
    }


def load(check=True):
    m = json.loads(MANIFEST.read_text())
    if check:
        cur = build()
        for k in cur:
            if m.get(k) != cur[k]:
                raise SystemExit(f"configs/world_model_m3_v3.json: '{k}' no longer matches its sources; nothing started")
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
    print("world-model M3 settings OK")


if __name__ == "__main__":
    main()
