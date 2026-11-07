"""Frozen inputs of the merge-category arms SB-AVG, TA and TIES (docs/merge_categories_protocol_v3.md): the update budget
(arm E's final update, seed 111), the contrastive constants (copied from configs/contrastive_v3.json), the lambda grid,
the TIES keep fraction, the shared-base budgets and the validation normalisation points, with SHA-256 hashes of the
files they were read from. Every consumer reads configs/merge_categories_v3.json; `verify` fails loudly when a source
file no longer matches.

    python tools/freeze_merge_categories_v3.py freeze      # once, before any run
    python tools/freeze_merge_categories_v3.py verify      # pre-flight and runner
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from freeze_contrastive_v3 import sha                      # noqa: E402  (SHA-256 of LF bytes)

MANIFEST = REPO / "configs" / "merge_categories_v3.json"
CL = "configs/contrastive_v3.json"
SIZE, SEED = "10x5", 111
E_STATUS = f"results/warmstart_v3/runs/{SIZE}+mix+bv3_ws_E_s{SEED}/status.json"
METHOD = {"lambda_grid": [0.3, 0.5, 0.7, 1.0], "ties_keep": 0.2,
          "lambda_rule": "highest update-0 validation HV (15-preference grid, frozen validation points); tie -> smaller "
                         "lambda; evaluation data never used",
          "shared_base": {"base_goal": "mct", "base_updates": 500, "ft_updates": 1000,
                          "source": "tools/ec2_runner.py u_shared_base (pre-registered E-shared path)"}}


def build():
    cl = json.loads((REPO / CL).read_text())
    e = json.loads((REPO / E_STATUS).read_text())
    return {"size": SIZE, "seed": SEED,
            "budget": {"updates": int(e["update"]), "rule": "arm E's final update (seed 111); no early stopping",
                       "source": E_STATUS, "source_sha256": sha(E_STATUS)},
            "contrastive": {"source": CL, "source_sha256": sha(CL), "constants": cl["constants"]},
            "validation_norm": cl["validation_norm"], "method": METHOD}


def load(check=True):
    m = json.loads(MANIFEST.read_text())
    if check:
        cur = build()
        for k in ("size", "seed", "budget", "contrastive", "validation_norm", "method"):
            if m[k] != cur[k]:
                raise SystemExit(f"configs/merge_categories_v3.json: '{k}' no longer matches its sources; nothing started")
        if sha(m["validation_norm"]["path"]) != m["validation_norm"]["sha256"]:
            raise SystemExit("the frozen validation normalisation file changed; nothing started")
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
    print(f"merge-categories manifest OK: {m['budget']['updates']} updates, lambda grid {m['method']['lambda_grid']}, "
          f"TIES keep {m['method']['ties_keep']}")


if __name__ == "__main__":
    main()
