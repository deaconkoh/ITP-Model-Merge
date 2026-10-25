"""Weight transfer and merging for the warm-start arms (docs/warmstart_protocol_v3.md).

Sources are benchmark-v3 specialists (schema benchmark_v3_f12_m9_p10, no preference input). Targets are
preference-conditioned DANIELs (schema benchmark_v3_f12_m9_p10_w3), whose first actor and critic layers have three
extra input columns for w, appended LAST in both inputs (daniel/model/main_model.py). Every other tensor has the same
shape in both.

  transfer(src, dst, "full")      copy every tensor; the preference columns of the two first layers are set to zero, so
                                  the conditioned network computes exactly the specialist's function for every w
  transfer(src, dst, "backbone")  copy only the dual-attention feature extractor (feature_exact.*); actor, critic and
                                  preference weights keep dst's (fresh) values
  transfer(src, dst, "no_critic") copy everything except the critic (arm D: a distilled preference-conditioned student,
                                  same schema as dst; the critic keeps dst's fresh values)
  uniform_merge(sds)              element-wise mean of several state dicts with identical shapes
  interpolate(sds, lam)           sum_x lam_x * theta_x (E0)
"""
from __future__ import annotations
import torch

PREF_LAYERS = ("actor.linears.0.weight", "critic.linears.0.weight")
BACKBONE_PREFIX = "feature_exact."


def transfer(src_sd: dict, dst_net, mode: str) -> dict:
    """Load src_sd into dst_net (in place) by `mode`; returns a report {copied, padded, kept}."""
    if mode not in ("full", "backbone", "no_critic"):
        raise ValueError(f"transfer mode must be full, backbone or no_critic, got {mode!r}")
    dst = dst_net.state_dict()
    unknown = sorted(set(src_sd) - set(dst))
    if unknown:
        raise KeyError(f"source tensors absent from the target network: {unknown[:5]}")
    report = {"copied": [], "padded": [], "kept": []}
    for k, d in dst.items():
        if (mode == "backbone" and not k.startswith(BACKBONE_PREFIX)) or (mode == "no_critic" and k.startswith("critic.")):
            report["kept"].append(k); continue
        if k not in src_sd:
            raise KeyError(f"target tensor {k} has no source")
        s = src_sd[k].to(d.device, d.dtype)
        if s.shape == d.shape:
            dst[k] = s.clone(); report["copied"].append(k)
        elif k in PREF_LAYERS and s.dim() == 2 and s.shape[0] == d.shape[0] and d.shape[1] > s.shape[1]:
            new = torch.zeros_like(d)
            new[:, :s.shape[1]] = s                     # the preference columns come last and stay zero
            dst[k] = new; report["padded"].append(k)
        else:
            raise ValueError(f"incompatible shapes for {k}: source {tuple(s.shape)}, target {tuple(d.shape)}")
    dst_net.load_state_dict(dst)
    return report


def zero_pref_columns(net, n_pref: int = 3):
    """Set the preference-input columns of the first actor and critic layers to zero (in place)."""
    with torch.no_grad():
        for k, p in net.named_parameters():
            if k in PREF_LAYERS:
                p[:, -n_pref:] = 0.0


def interpolate(sds: list, lam) -> dict:
    """sum_x lam_x * sd_x over identically shaped state dicts (lam sums to 1 for a convex combination)."""
    if len(sds) != len(lam):
        raise ValueError("one coefficient per state dict")
    keys = set(sds[0])
    if any(set(s) != keys or any(s[k].shape != sds[0][k].shape for k in keys) for s in sds):
        raise ValueError("state dicts to merge must have identical keys and shapes")
    return {k: sum(float(l) * s[k].to(torch.float64) for l, s in zip(lam, sds)).to(sds[0][k].dtype) for k in sds[0]}


def uniform_merge(sds: list) -> dict:
    return interpolate(sds, [1.0 / len(sds)] * len(sds))
