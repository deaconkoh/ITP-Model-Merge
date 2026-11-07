"""Frozen inputs of the contrastive fine-tuning arms (docs/contrastive_protocol_v3.md): the update budget of each arm
(the final update of its PPO counterpart, seed 111), the validation normalisation points and the method constants, with
SHA-256 hashes of the files they were read from. Every consumer reads configs/contrastive_v3.json; `verify` fails
loudly when a source file no longer matches.

    python tools/freeze_contrastive_v3.py freeze      # once, before any run
    python tools/freeze_contrastive_v3.py verify      # pre-flight and runner
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs" / "contrastive_v3.json"
WS = "results/warmstart_v3"
SIZE, SEED = "10x5", 111
COUNTERPART = {"CLB": "B", "CLD": "D", "CLE": "E"}
NORM = f"{WS}/norm/validation_normalisation_points_{SIZE}.json"
CONSTANTS = {"cl_k": 8, "cl_beta": 0.1, "cl_margin": 0.01, "cl_grad_clip": 1.0, "lr": 3e-4, "minibatch_size": 1024,
             "validate_timestep": 50, "converge_window": 0, "mo_val_grid_step": 0.25, "pref_dirichlet": 1.0,
             "num_envs": 20, "reset_env_timestep": 20}


def sha(path):
    """SHA-256 of the file with LF line endings (independent of the working copy's line-ending conversion)."""
    return hashlib.sha256((REPO / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build():
    arms = {}
    for arm, ppo in COUNTERPART.items():
        name = f"{SIZE}+mix+bv3_ws_{ppo}_s{SEED}"
        st = f"{WS}/runs/{name}/status.json"
        s = json.loads((REPO / st).read_text())
        arms[arm] = {"counterpart": ppo, "counterpart_run": name, "updates": int(s["update"]),
                     "counterpart_status": s["status"], "source": st, "source_sha256": sha(st)}
    return {"size": SIZE, "seed": SEED, "arms": arms,
            "budget_rule": "updates = final update of the PPO counterpart run (seed 111); no early stopping",
            "validation_norm": {"path": NORM, "sha256": sha(NORM)}, "constants": CONSTANTS}


def load(check=True):
    m = json.loads(MANIFEST.read_text())
    if check:
        cur = build()
        for k in ("arms", "validation_norm", "constants", "size", "seed"):
            if m[k] != cur[k]:
                raise SystemExit(f"configs/contrastive_v3.json: '{k}' no longer matches its sources; nothing started")
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
    print("contrastive manifest OK: " + ", ".join(f"{k} {v['updates']} updates" for k, v in m["arms"].items()))


if __name__ == "__main__":
    main()
