"""Frozen inputs of the converged contrastive arms (docs/contrastive_converged_protocol_v3.md): the stopping rule, and per
arm the fixed-budget source run, the replay of the rule on its logged validation history, the resulting mode (continue
from the saved final state, or rerun from the saved starting network) and every file it needs, with SHA-256 hashes.
Every consumer reads configs/contrastive_converged_v3.json; `verify` fails loudly when a source no longer matches.

    python tools/freeze_contrastive_converged_v3.py freeze      # once, before any run
    python tools/freeze_contrastive_converged_v3.py verify      # pre-flight and runner
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO / "daniel"))
from freeze_contrastive_v3 import sha                      # noqa: E402  (SHA-256 of LF bytes, text files)
from convergence import is_converged                       # noqa: E402

MANIFEST = REPO / "configs" / "contrastive_converged_v3.json"
TN = REPO / "daniel" / "trained_network" / "SD2"
SIZE, SEED = "10x5", 111
RULE = {"converge_window": 1000, "converge_tol": 0.005, "min_updates": 1000, "cap": 8000,
        "statement": "stop once the best validation HV has improved by less than 0.5% over the last 1,000 updates, after "
                     "at least 1,000 updates (the PPO warm-start rule, daniel/convergence.py); hard cap 8,000 updates, "
                     "reported as 'capped'"}
# arm token in the run command -> (plain name, fixed-budget source run, its results folder, start network for a rerun)
ARMS = {
    "CCB": ("From scratch", "10x5+mix+bv3_cl_CLB_s111", "results/contrastive_v3", None),
    "CCD": ("Distillation", "10x5+mix+bv3_cl_CLD_s111", "results/contrastive_v3",
            "daniel/trained_network/SD2/10x5+mix+bv3_clD_student_s111.pth"),
    "CCE": ("Naive weight average (independent specialists)", "10x5+mix+bv3_cl_CLE_s111", "results/contrastive_v3",
            "daniel/trained_network/SD2/10x5+mix+bv3_clE_merged_s111.pth"),
    "CCSBAVG": ("Model Soup (shared base)", "10x5+mix+bv3_mc_SBAVG_s111", "results/merge_categories_v3",
                "checkpoints/merge_categories_v3/10x5+mix+bv3_mcSBAVG_start_s111.pth"),
    "CCTA": ("Task Arithmetic", "10x5+mix+bv3_mc_TA_s111", "results/merge_categories_v3",
             "checkpoints/merge_categories_v3/10x5+mix+bv3_mcTA_start_s111.pth"),
    "CCTIES": ("TIES", "10x5+mix+bv3_mc_TIES_s111", "results/merge_categories_v3",
               "checkpoints/merge_categories_v3/10x5+mix+bv3_mcTIES_start_s111.pth"),
}


def bsha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(val):
    """First validation update at which the rule stops the logged history, or None."""
    log = [(e["update"], e["selection_metric"]) for e in val]
    for i in range(len(log)):
        if is_converged(log[:i + 1], RULE["converge_window"], RULE["converge_tol"], RULE["min_updates"])[0]:
            return log[i][0]
    return None


def build():
    arms = {}
    for arm, (plain, run, res, start) in ARMS.items():
        rdir = f"{res}/runs/{run}"
        val = json.loads((REPO / rdir / "valiquality.json").read_text())
        cfg = json.loads((REPO / rdir / "run.json").read_text())["config"]
        budget = val[-1]["update"]
        stop = replay(val)
        mode = "continue" if stop is None else "rerun"
        rec = {"name": plain, "source_run": run, "source_dir": rdir,
               "source_valiquality_sha256": sha(f"{rdir}/valiquality.json"), "fixed_budget": budget,
               "replay_stop_update": stop, "mode": mode,
               "reason": (f"the rule never stops the logged history (0-{budget}); continue from the saved final state"
                          if stop is None else f"the rule would have stopped at update {stop}, before the fixed budget "
                                               f"{budget}; rerun from the saved starting network"),
               "source_init": {"init_from": cfg.get("init_from") or "", "init_mode": cfg.get("init_mode", "exact"),
                               "mo_val_norm": cfg["mo_val_norm"], "mo_val_first": cfg["mo_val_first"],
                               "init_from_sha256": bsha(cfg["init_from"]) if cfg.get("init_from") else None}}
        resume = TN / f"{run}.resume.pth"
        if mode == "continue":
            rec["resume_from"] = str(resume)
            rec["resume_from_sha256"] = bsha(resume)
        rec["rerun_start"] = ({"path": start, "sha256": bsha(REPO / start), "init_mode": cfg.get("init_mode", "full")}
                              if start else {"path": None, "note": "random initialisation, seed 111 (as the source run)"})
        arms[arm] = rec
    cl = "configs/contrastive_v3.json"
    clm = json.loads((REPO / cl).read_text())
    return {"size": SIZE, "seed": SEED, "rule": RULE, "arms": arms,
            "contrastive": {"source": cl, "source_sha256": sha(cl), "constants": clm["constants"]},
            "validation_norm": clm["validation_norm"],
            "setup_costs": {"distillation": {"source": "results/contrastive_v3/distill/10x5+mix+bv3_clD_student_s111.distill.json"},
                            "shared_base": {"source": "results/merge_categories_v3/shared_base/10x5_s111"}}}


def load(check=True):
    m = json.loads(MANIFEST.read_text())
    if check:
        cur = build()
        for k in ("size", "seed", "rule", "arms", "contrastive", "validation_norm"):
            if m[k] != cur[k]:
                raise SystemExit(f"configs/contrastive_converged_v3.json: '{k}' no longer matches its sources; "
                                 "nothing started")
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
    print("contrastive-converged manifest OK: " + ", ".join(f"{v['name']} {v['mode']}" for v in m["arms"].values()))


if __name__ == "__main__":
    main()
