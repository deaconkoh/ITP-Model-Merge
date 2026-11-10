"""Frozen settings of world-model M2 (docs/world_model_m2_protocol_v3.md): the contrastive constants, stopping rule,
seeds, the M1 world-model trunk, the pre-registered criteria and the reference sources, with SHA-256. Every M2 step
reads configs/world_model_m2_v3.json through load(), which fails loudly if a source changed.

    python tools/freeze_world_model_m2_v3.py freeze      # once, before any run
    python tools/freeze_world_model_m2_v3.py verify
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs" / "world_model_m2_v3.json"
CL_CONFIG = "configs/contrastive_v3.json"
TRUNK = "checkpoints/world_model_m1_v3/world_model_trunk.pth"
SEEDS_ALL = (111, 222, 333)


def sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def build():
    cl = json.loads((REPO / CL_CONFIG).read_text())["constants"]
    m1 = json.loads((REPO / "checkpoints/world_model_m1_v3/SHA256SUMS.json").read_text())
    if m1["world_model_trunk.pth"] != sha(TRUNK):
        raise SystemExit("the M1 trunk does not match its SHA256SUMS.json")
    refs = {}
    for o in "mct":
        for s in SEEDS_ALL:
            run = f"10x5+mix+bv3_spec_{o}_s{s}"
            ev, st = f"results/baselines_v3/eval/evaluation_10x5/{run}.json", f"results/baselines_v3/runs/{run}/status.json"
            refs[run] = {"eval": ev, "eval_sha256": sha(ev), "status": st, "status_sha256": sha(st)}
    return {
        "size": "10x5", "objectives": ["m", "c", "t"], "seeds_available": list(SEEDS_ALL), "seeds_default": [111],
        "contrastive": {"source": CL_CONFIG, "sha256": sha(CL_CONFIG),
                        **{k: cl[k] for k in ("cl_k", "cl_beta", "cl_margin", "cl_grad_clip", "lr", "minibatch_size",
                                              "num_envs", "reset_env_timestep", "validate_timestep")},
                        "pairs": "single objective: (i, j) is a pair when f_j - f_i > margin, f = the goal's objective "
                                 "normalised per instance (MK / MKref_i, C / Cref_i, T / Tref); K schedules per "
                                 "instance; no preference input, no dominance rule"},
        "rule": {"converge_window": 1000, "converge_tol": 0.005, "min_updates": 1000, "cap": 8000, "chunk": 500,
                 "statement": "stop once the best validation value has improved by less than 0.5% over the last 1,000 "
                              "updates, after at least 1,000 updates; hard cap 8,000 updates, reported as 'capped'"},
        "world_model_trunk": {"path": TRUNK, "sha256": sha(TRUNK)},
        "world_model_cost": {"pretrain": "results/world_model_m1_v3/pretrain.json",
                             "pretrain_sha256": sha("results/world_model_m1_v3/pretrain.json"),
                             "data": "results/world_model_m1_v3/data_train_meta.json",
                             "data_sha256": sha("results/world_model_m1_v3/data_train_meta.json")},
        "evaluation": {"split": "evaluation", "instances": 100, "sample": 100, "sample_seed": 50,
                       "primary": "greedy, the specialist's own raw objective (mean over instances)"},
        "criteria": {"faster": {"ratio": 0.70, "min_objectives": 2,
                                "statement": "the world-model specialist reaches the control's best validation value "
                                             "in <= 70% of the control's updates to its best, on at least 2 of 3 "
                                             "objectives"},
                     "better": {"margin": 0.05, "min_objectives": 2,
                                "statement": "final evaluation value (greedy, primary objective) at least 5% better "
                                             "than the control's on at least 2 of 3 objectives"}},
        "references": {"ppo_specialists": refs, "cpsat_b1": "results/baselines_v3/cpsat/b1/evaluation_10x5"},
        "smoke": {"cap": 5, "chunk": 0, "validate_timestep": 5, "eval_split": "development", "eval_first": 3,
                  "sample": 4},
    }


def load(check=True):
    m = json.loads(MANIFEST.read_text())
    if check:
        cur = build()
        for k in cur:
            if m.get(k) != cur[k]:
                raise SystemExit(f"configs/world_model_m2_v3.json: '{k}' no longer matches its sources; nothing started")
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
    print("world-model M2 settings OK")


if __name__ == "__main__":
    main()
