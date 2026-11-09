"""Frozen inputs of the merge-failure test (docs/merge_failure_test_protocol_v3.md): the Model Soup (shared base)
starting network, the converged control runs (Model Soup, From scratch), the stopping rule, the three one-change arms
and the update-0 diagnostics, with SHA-256 hashes. Every consumer reads configs/merge_failure_test_v3.json; `verify`
fails loudly when a source no longer matches.

    python tools/freeze_merge_failure_test_v3.py freeze      # once, before any run
    python tools/freeze_merge_failure_test_v3.py verify      # pre-flight and runner
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from freeze_contrastive_v3 import sha                      # noqa: E402  (SHA-256 of LF bytes, text files)

MANIFEST = REPO / "configs" / "merge_failure_test_v3.json"
CONVERGED = "configs/contrastive_converged_v3.json"
CC_RES = "results/contrastive_converged_v3"
SIZE, SEED = "10x5", 111

ARMS = {
    "MFSP": {"name": "Shrink and perturb", "flags": ["--init_shrink", "0.3", "--init_perturb", "0.0001"],
             "change": "before fine-tuning, every weight of the Model Soup start becomes 0.3 x weight + 0.0001 x the "
                       "same weight of a freshly initialised network (seed 111, the From-scratch initialisation); the "
                       "contrastive reference is this shrunk-and-perturbed network",
             "source": "Ash & Adams, 'On Warm-Starting Neural Network Training', NeurIPS 2020, Section 4: "
                       "theta <- lambda * theta + p, p ~ N(0, sigma^2), 'in practice ... adding parameters from a scaled, "
                       "randomly-initialized network'; lambda = 0.3 and noise scale 0.0001 are the values the paper "
                       "uses for the pre-training-then-fine-tuning case (Section 4.2, Figure 9)"},
    "MFNA": {"name": "No anchor", "flags": ["--cl_anchor", "none"],
             "change": "the same pairwise loss with the reference log-probabilities removed: "
                       "-log sigmoid(beta * (l_better - l_worse)) instead of "
                       "-log sigmoid(beta * [(l_better - l_better_ref) - (l_worse - l_worse_ref)])"},
    "MFPR": {"name": "Random preference input", "flags": ["--init_pref_columns", "fresh"],
             "change": "the preference-input columns of the first actor and critic layers keep the values of a freshly "
                       "initialised network (seed 111, as in From scratch) instead of zero; every other weight from the "
                       "merge"},
}
DIAG = {  # update-0 diagnostics subjects: (plain name, how the network is built)
    "SOUP": ("Model Soup (shared base)", "the Model Soup start, --init_mode full (preference columns zero)"),
    "DIST": ("Distillation", "the distilled student, --init_mode no_critic (as the Distillation arm)"),
    "RAND": ("Random initialisation", "no --init_from: the fresh seed-111 network (the From-scratch start)"),
    **{k: (v["name"], f"the arm's start: {v['change']}") for k, v in ARMS.items()},
}
GAP = ("an arm closes the gap to From scratch if its evaluation HV is at least From scratch's converged evaluation HV; "
       "otherwise the report gives the share of the gap closed, (arm - Model Soup) / (From scratch - Model Soup), on "
       "evaluation HV (Model Soup and From scratch: the converged runs). Single seed: numbers only, no significance claims")


def bsha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(prev=None):
    cc = json.loads((REPO / CONVERGED).read_text())
    soup, dist = cc["arms"]["CCSBAVG"], cc["arms"]["CCD"]
    ctrl = {}
    for arm in ("CCSBAVG", "CCB"):
        run = f"{SIZE}+mix+bv3_cc_{arm}_s{SEED}"
        rdir, ev = f"{CC_RES}/runs/{run}", f"{CC_RES}/eval/evaluation_{SIZE}/{run}.json"
        ctrl[arm] = {"name": cc["arms"][arm]["name"], "run": run, "dir": rdir,
                     "status_sha256": sha(f"{rdir}/status.json"), "valiquality_sha256": sha(f"{rdir}/valiquality.json"),
                     "eval_sha256": sha(ev)}
    dist_path = Path(prev["diagnostics"]["distillation_start"]["path"]) if prev else REPO / dist["rerun_start"]["path"]
    return {"size": SIZE, "seed": SEED, "rule": cc["rule"],
            "converged_manifest": {"source": CONVERGED, "sha256": sha(CONVERGED)},
            "contrastive": cc["contrastive"], "validation_norm": cc["validation_norm"],
            "start": {"name": "Model Soup (shared base)", "path": soup["rerun_start"]["path"],
                      "sha256": bsha(REPO / soup["rerun_start"]["path"]), "init_mode": "full",
                      "note": "the start of the converged Model Soup run (a rerun from this network under the same rule)"},
            "controls": ctrl,
            "arms": {k: {**v, "base": "the converged Model Soup run's flags (tools/ec2_runner.py: cc_args CCSBAVG, rerun), "
                                      "plus the flags above"} for k, v in ARMS.items()},
            "diagnostics": {"subjects": {k: {"name": n, "network": d} for k, (n, d) in DIAG.items()},
                            "distillation_start": {"path": str(dist_path), "sha256": bsha(dist_path),
                                                   "init_mode": dist["rerun_start"]["init_mode"]},
                            "k": cc["contrastive"]["constants"]["cl_k"], "seed": SEED,
                            "near_inactive": "|activation| <= 0.01 on at least 99% of the entries of the validation batch "
                                             "(every state of the greedy validation rollouts, unfinished schedules only)",
                            "saturated_tanh": "|tanh activation| >= 0.99 on at least 99% of those entries",
                            "gradient": "the run's first contrastive batch (same instances and preferences for every "
                                        "network), before clipping",
                            "preference_sensitivity": "on every state of the greedy validation rollouts, the policy under "
                                                      "each of the 15 validation preferences"},
            "gap_criterion": GAP}


def load(check=True):
    m = json.loads(MANIFEST.read_text())
    if check:
        cur = build(m)
        for k in [k for k in cur if k != "frozen"]:
            if m[k] != cur[k]:
                raise SystemExit(f"configs/merge_failure_test_v3.json: '{k}' no longer matches its sources; nothing started")
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
    m = load()
    print("merge-failure manifest OK: " + ", ".join(v["name"] for v in m["arms"].values()))


if __name__ == "__main__":
    main()
