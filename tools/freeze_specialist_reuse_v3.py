"""Frozen settings of the specialist-reuse experiment (docs/specialist_reuse_protocol_v3.md): the contrastive recipe,
the stopping rule, the waiting option, the four methods, the equal tuning budget and search spaces, the TIES rule and
screen, the decoders and the evaluation, and the smoke settings. Every step reads configs/specialist_reuse_v3.json
through load(), which fails loudly if a source changed.

    python tools/freeze_specialist_reuse_v3.py freeze      # once, before any run
    python tools/freeze_specialist_reuse_v3.py verify
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs" / "specialist_reuse_v3.json"
RECIPE = "configs/contrastive_v3.json"
VAL_NORM = "results/warmstart_v3/norm/validation_normalisation_points_10x5.json"
EVAL_NORM = "results/baselines_v3/normalisation_points.json"
B2 = "results/baselines_v3/cpsat/b2/evaluation_10x5"
SPEC_CONFIGS = {o: f"configs/benchmark_v3/{o}.json" for o in "mct"}
MO_CONFIG = "configs/benchmark_v3/mct.json"
METHODS = {"scratch": "From scratch", "distill": "Distillation", "rs": "Rewarded Soups", "ties": "Preference-weighted TIES"}
LR = [1e-4, 3e-4, 1e-3]
BETA = [0.1, 0.5]


def sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def b2_sha():
    h = hashlib.sha256()
    for f in sorted((REPO / B2).glob("*.json")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def build():
    rec = json.loads((REPO / RECIPE).read_text())["constants"]
    trials = [{"lr": lr, "cl_beta": b} for lr, b in itertools.product(LR, BETA)]
    return {
        "name": "specialist_reuse_v3", "size": "10x5", "seed": 111, "tag": "sr",
        "dispatch_rule": {"flag": "--dispatch_rule semi_active",
                          "where": ["specialist training and its validation (train.py)",
                                    "distillation stage 1: teacher rollouts and held-out check (distill_v3.py)",
                                    "multi-objective training and its validation (train.py), TIES screen",
                                    "evaluation and timing (tools/decoding_v3.py environments)"]},
        "recipe": {"source": RECIPE, "sha256": sha(RECIPE), "constants": rec},
        "stopping": {"converge_window": 1000, "converge_tol": 0.005, "min_updates": 1000, "max_updates": 8000,
                     "chunk": 500, "rule": "stop when the best validation value improves by less than 0.5% over 1,000 "
                                           "updates; minimum 1,000, cap 8,000 (a run that reaches the cap is 'capped')"},
        "validation": {"decoder": "greedy", "split": "validation", "instances": "all 18", "grid_step": 0.25,
                       "preferences": 15, "every": rec["validate_timestep"], "norm": VAL_NORM, "norm_sha256": sha(VAL_NORM),
                       "use": "tuning score, checkpoint selection (best), stopping, time to target"},
        "base": {"what": "unconditioned DANIEL after setup_seed(111) (tools/world_model_m2_bases.random_base_sd)"},
        "specialists": {"objectives": ["m", "c", "t"],
                        "configs": {o: {"path": p, "sha256": sha(p)} for o, p in SPEC_CONFIGS.items()},
                        "training": "single-objective contrastive from a copy of the base (--init_mode exact)",
                        "selection": "the specialist's own validation objective (score_v3 of its goal); best checkpoint"},
        "mo_config": {"path": MO_CONFIG, "sha256": sha(MO_CONFIG)},
        "methods": {
            "scratch": {"label": METHODS["scratch"], "start": "fresh preference-conditioned DANIEL, setup_seed(111)"},
            "distill": {"label": METHODS["distill"],
                        "stage1": {"script": "daniel/distill_v3.py", "instances": 400, "epochs": 15, "batch": 256,
                                   "lr": 3e-4, "wall_cap": 1200},
                        "stage2": "contrastive fine-tuning from the student (--init_mode no_critic)"},
            "rs": {"label": METHODS["rs"], "flags": "--merge_mix m,c,t (--merge_rule mix)",
                   "rule": "theta(w) = w_M theta_M + w_C theta_C + w_T theta_T per sample"},
            "ties": {"label": METHODS["ties"], "flags": "--merge_mix m,c,t --merge_rule ties --merge_base base",
                     "rule": "daniel/mixture.py:pref_ties_flat (trim per vector, weight before election, sum of "
                             "agreeing entries, lambda; zero weight: no vote, no contribution; exact corner: the "
                             "specialist itself)",
                     "screen": {"keep": [0.2, 0.5, 0.8], "lambda": [0.8, 1.0, 1.5],
                                "score": "validation HV of the merged start (the fixed validation decoder, update 0)"}}},
        "tuning": {"trials_per_method": len(trials), "updates_per_trial": 1000, "trials": trials,
                   "score": "best validation HV within the trial", "tie_break": "closest to the recipe (lr 3e-4, beta 0.1)",
                   "full_run": "the selected setting, trained fresh with the stopping rule"},
        "evaluation": {"split": "evaluation", "instances": 100, "grid_step": 0.1, "preferences": 66,
                       "extra": "equal weights also decoded (batch layout of evaluate_mo_v3), not in the metrics",
                       "decoders": {"greedy": {}, "sample128": {"K": 128, "seed": 51}},
                       "norm": EVAL_NORM, "norm_sha256": sha(EVAL_NORM), "hv_reference": 1.1,
                       "igd_front": B2, "igd_front_sha256": b2_sha(), "igd_instances": 20,
                       "timing": "batched seconds per instance-preference pair, GPU alone, same batching for all"},
        "smoke": {"updates": 5, "validate_timestep": 5, "mo_val_first": 3, "trials": 2,
                  "distill": {"instances": 4, "epochs": 1, "wall_cap": 120, "val_first": 2},
                  "screen": {"keep": [0.2, 0.5], "lambda": [1.0]}, "eval_first": 2, "sample_K": 4},
    }


def load(check=True):
    m = json.loads(MANIFEST.read_text())
    if check:
        cur = build()
        for k in cur:
            if m.get(k) != cur[k]:
                raise SystemExit(f"configs/specialist_reuse_v3.json: '{k}' no longer matches its sources; nothing started")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["freeze", "verify"])
    a = ap.parse_args()
    if a.action == "freeze":
        if MANIFEST.exists():
            sys.exit(f"{MANIFEST} exists (frozen); delete it deliberately to re-freeze")
        m = build(); m["frozen"] = time.strftime("%Y-%m-%d")
        assert len({len(m["tuning"]["trials"])}) == 1
        MANIFEST.write_text(json.dumps(m, indent=1) + "\n")
        print(f"frozen: {MANIFEST}")
    load()
    print("specialist reuse settings OK")


if __name__ == "__main__":
    main()
