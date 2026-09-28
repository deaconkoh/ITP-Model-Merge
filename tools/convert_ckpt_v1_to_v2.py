"""Function-preserving conversion of v1 checkpoints (schema canonical_f11_p9_v2) to v2 observations
(canonical_f13_p10_obsv2; results/v2_fixes.md, Part B).

  * first operation-attention block, every head: W [11 x 32] -> [13 x 32], the 2 new rows (slack, time to due
    date) exactly 0;
  * first machine-attention block, every head: W_edge [11 x 32] (operation features as edge features) -> [13 x 32],
    new rows exactly 0;
  * actor first Linear: weight [64 x 41] -> [64 x 42], new last column (projected lateness) exactly 0; the carbon
    column (index 32 + 8 = 40) multiplied by CARBON_SCALE = 100, offsetting v2's carbon / 100;
  * everything else, the critic included, copied unchanged.
Why zero weights preserve the function: operation features reach W / W_edge only through the environment's
per-feature standardisation (never mixes features), input dropout (p = 0, identity in eval) and the linear
comp_idx @ fea_j; pair features enter the actor's first Linear directly. See results/v2_fixes.md.

    python tools/convert_ckpt_v1_to_v2.py <v1 name> [...]     # names without .pth, in daniel/trained_network/SD2
Writes <name>_obsv2.pth next to the source; refuses to overwrite.
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
CK = REPO / "daniel/trained_network/SD2"
V1, V2 = "canonical_f11_p9_v2", "canonical_f13_p10_obsv2"
CARBON_SCALE = 100.0
N_OP_NEW, EMB = 2, 32          # new operation features; actor input = 4 x 8 embedding dims before the pair features
CARBON_COL = EMB + 8           # pair feature 9 (carbon) in the actor input


def convert_state_dict(sd):
    out = {}
    for k, v in sd.items():
        v = v.clone()
        if k.startswith("feature_exact.op_attention_blocks.0.") and k.endswith(".W"):
            assert v.shape == (11, 32), (k, v.shape)
            v = torch.cat([v, torch.zeros(N_OP_NEW, v.shape[1], dtype=v.dtype, device=v.device)], 0)
        elif k.startswith("feature_exact.mch_attention_blocks.0.") and k.endswith(".W_edge"):
            assert v.shape == (11, 32), (k, v.shape)
            v = torch.cat([v, torch.zeros(N_OP_NEW, v.shape[1], dtype=v.dtype, device=v.device)], 0)
        elif k == "actor.linears.0.weight":
            assert v.shape == (64, 41), (k, v.shape)
            v[:, CARBON_COL] = v[:, CARBON_COL] * CARBON_SCALE
            v = torch.cat([v, torch.zeros(v.shape[0], 1, dtype=v.dtype, device=v.device)], 1)
        out[k] = v
    return out


def convert(name):
    src = CK / f"{name}.pth"
    dst = CK / f"{name}_obsv2.pth"
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst.name}")
    ck = torch.load(src, map_location="cpu")
    assert isinstance(ck, dict) and "state_dict" in ck, "versioned checkpoint bundle expected"
    assert ck["config"]["feature_schema"] == V1, ck["config"]["feature_schema"]
    new = dict(ck)
    new["state_dict"] = convert_state_dict(ck["state_dict"])
    cfg = dict(ck["config"])
    cfg.update(feature_schema=V2, fea_j_input_dim=13, fea_pair_input_dim=10, obs_version="v2")
    new["config"] = cfg
    new["parents"] = [{"path": str(src), "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                       "conversion": "v1 -> v2 observations, function-preserving (tools/convert_ckpt_v1_to_v2.py)",
                       "carbon_scale": CARBON_SCALE}]
    torch.save(new, dst)
    print(f"{src.name} -> {dst.name}")
    return dst


if __name__ == "__main__":
    for n in sys.argv[1:]:
        convert(n)
