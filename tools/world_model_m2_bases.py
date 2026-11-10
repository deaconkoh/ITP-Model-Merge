"""Starting networks of world-model M2 (docs/world_model_m2_protocol_v3.md), per seed s:

    shared_random_base_s{s}.pth   DANIEL (single-objective, schema benchmark_v3_f12_m9_p10) after setup_seed(s)
    world_model_base_s{s}.pth     the SAME network with its trunk (feature_exact) replaced by the M1 world-model trunk

The two differ only in feature_exact; actor and critic are identical. Every specialist of an arm starts from a copy of
its arm's file (train.py --init_from <base> --init_mode exact).

    python tools/world_model_m2_bases.py --seeds 111 --out checkpoints/world_model_m2_v3/bases
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel"))
_argv, sys.argv = sys.argv, [sys.argv[0], "--device", "cpu", "--obs_version", "v3", "--reward_version", "v3",
                             "--goal", "t"]
import torch                                                        # noqa: E402
from params import configs                                          # noqa: E402
from common_utils import setup_seed                                 # noqa: E402
from checkpointing import make_checkpoint_bundle, json_safe_config, unwrap_checkpoint, validate_checkpoint_schema  # noqa
from feature_schemas import SCHEMA_V3                               # noqa: E402
from model.main_model import DANIEL                                 # noqa: E402
sys.argv = _argv


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wm_trunk(path, expect_sha):
    if sha(path) != expect_sha:
        raise SystemExit(f"world-model trunk {path} does not match its frozen SHA-256")
    sd, meta = unwrap_checkpoint(torch.load(path, map_location="cpu", weights_only=False))
    validate_checkpoint_schema(meta, SCHEMA_V3, sd)
    return {k: v for k, v in sd.items() if k.startswith("feature_exact.")}


def build(seed, trunk_path, trunk_sha, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    trunk = wm_trunk(trunk_path, trunk_sha)
    setup_seed(seed)
    sd = {k: v.detach().cpu().clone() for k, v in DANIEL(configs).state_dict().items()}
    wm = dict(sd); wm.update({k: v.clone() for k, v in trunk.items()})
    assert set(trunk) == {k for k in sd if k.startswith("feature_exact.")}
    files = {}
    for name, state, parent in ((f"shared_random_base_s{seed}.pth", sd, None),
                                (f"world_model_base_s{seed}.pth", wm, {"path": str(Path(trunk_path).relative_to(REPO)),
                                                                        "sha256": trunk_sha, "part": "feature_exact"})):
        bundle = make_checkpoint_bundle(state, config=json_safe_config(configs), data_fingerprints={}, update=0,
                                        validation_metrics={}, selection_metric=float("nan"),
                                        parents=[parent] if parent else [], repo_root=REPO)
        torch.save(bundle, out / name)
        files[name] = sha(out / name)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="111")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(REPO / "configs" / "world_model_m2_v3.json"))
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out = out if out.is_absolute() else REPO / out
    sums = {}
    for s in [int(x) for x in a.seeds.split(",")]:
        sums.update(build(s, REPO / cfg["world_model_trunk"]["path"], cfg["world_model_trunk"]["sha256"], out))
    f = out / "SHA256SUMS.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **sums}, indent=1) + "\n")
    print(json.dumps(sums, indent=1))


if __name__ == "__main__":
    main()
