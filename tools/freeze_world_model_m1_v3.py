"""Frozen settings of the world-model trunk, Milestone 1 (docs/world_model_m1_protocol_v3.md): data, pretraining, the
two checks, the outcome rule and the comparison trunks (with SHA-256). Every step reads configs/world_model_m1_v3.json
through load(), which fails loudly if a source no longer matches.

    python tools/freeze_world_model_m1_v3.py freeze      # once, before any run
    python tools/freeze_world_model_m1_v3.py verify
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs" / "world_model_m1_v3.json"
TRUNKS = {"shared_base": "checkpoints/merge_categories_v3/10x5+mix+bv3_mcbase_s111.pth",
          "makespan_specialist": "checkpoints/specialists_v3/10x5+mix+bv3_spec_m_s111.pth"}

SETTINGS = {
    "size": "10x5", "feature_set": "current",          # "current" (12 operation features) or "lateness" (17)
    # data
    "train_seed": 111, "train_instances": 100000, "shard_instances": 500, "states_per_episode": 6, "n_candidates": 4,
    "val_seed": 2111, "val_instances": 18, "val_step_stride": 1,
    "t_norm": 990.0,                                   # 99 x N / M: the benchmark's time scale x operations per machine
    "e_norm": 133.65,                                  # benchmark CARBON_SCALE (13.365 kg) x N / M
    "t_ref": 557.4180230253888,                        # the 10x5 Tref of the frozen benchmark manifest
    # pretraining
    "lr": 1e-3, "batch_states": 256, "max_epochs": 100, "patience": 5, "head_hidden": 64,
    # probes
    "probe_seeds": 5, "probe_hidden": 64, "probe_lr": 1e-3, "probe_max_epochs": 2000, "probe_patience": 50,
    # outcome rule (MLP probe, mean MAE over probe seeds)
    "rule": {"probe": "mlp", "pass_margin": 0.05, "pass_min_objectives": 2, "max_worse": 0.05,
             "fail_sanity_groups": 2},
    "smoke": {"train_instances": 40, "shard_instances": 10, "val_instances": 4, "val_step_stride": 7,
              "max_epochs": 2, "probe_seeds": 1, "probe_max_epochs": 20, "probe_patience": 5},
}


def sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def build():
    from_manifest = json.loads((REPO / "daniel" / "data" / "benchmark_v3" / "manifest_benchmark_v3.json").read_text())
    t_ref = float(from_manifest["sizes"]["10x5"]["t_ref"])
    if abs(t_ref - SETTINGS["t_ref"]) > 1e-9:
        raise SystemExit(f"Tref changed: {t_ref}")
    return {**SETTINGS, "trunks": {k: {"path": p, "sha256": sha(p)} for k, p in TRUNKS.items()}}


def load(check=True):
    m = json.loads(MANIFEST.read_text())
    if check:
        cur = build()
        for k in cur:
            if m.get(k) != cur[k]:
                raise SystemExit(f"configs/world_model_m1_v3.json: '{k}' no longer matches its sources; nothing started")
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
    print("world-model M1 settings OK")


if __name__ == "__main__":
    main()
