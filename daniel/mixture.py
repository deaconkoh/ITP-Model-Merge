"""Preference-weighted merging (docs/world_model_m3_protocol_v3.md).

Three specialists theta_M, theta_C, theta_T; for preference w the policy is the network with weights
theta(w) = w_M theta_M + w_C theta_C + w_T theta_T, a separate mix for every sample.

  mix_state_dicts(sds, w)       the plain mix of state dicts for one w (= warmstart.interpolate)
  MixturePolicy(nets)           three preference-conditioned DANIELs; forward(..., pref) runs every row with its own mix
                                theta(pref_row) and also feeds pref_row as the preference input. Gradients reach all
                                three networks (each theta_x gets w_x * dL/dtheta(w)).
  load_specialists(paths, cfg)  the three unconditioned v3 specialists, each copied into a preference-conditioned DANIEL
                                with warmstart.transfer(..., "full"): the preference columns of the first actor and critic
                                layers are zero, so at the start theta(w) computes the plain weight mix for every w.

Exactness: every DANIEL parameter enters the computation linearly (Linear layers and attention matrices; no
normalisation layers), so per-sample parameters are well defined. The batched path materialises per-row parameters
[B, *shape] and relies on the layers accepting them (daniel/model/attention_layer.py, sub_layers.py); with the modules'
own 2-D parameters the layers compute exactly as before. `grouped=True` is the slow reference (one functional_call per
distinct preference), used by the tests.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.func import functional_call

KEYS = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs")


def mix_state_dicts(sds, w):
    """sum_x w_x sd_x (float64 accumulation, as warmstart.interpolate)."""
    from warmstart import interpolate
    return interpolate(sds, w)


class MixturePolicy(nn.Module):
    def __init__(self, nets, grouped=False):
        super().__init__()
        if len(nets) != 3 or any(not getattr(n, "preference_dim", 0) for n in nets):
            raise ValueError("MixturePolicy needs three preference-conditioned DANIELs")
        self.nets = nn.ModuleList(nets)
        self.preference_dim = nets[0].preference_dim
        self.grouped = grouped
        self.names = [k for k, _ in nets[0].named_parameters()]
        if any(len(list(b)) for b in (n.buffers() for n in nets)):
            raise ValueError("DANIEL has no buffers; a mixture would have to mix them too")

    def stacked(self):
        """{name: [3, *shape]} of the three networks' parameters (differentiable)."""
        ps = [dict(n.named_parameters()) for n in self.nets]
        return {k: torch.stack([p[k] for p in ps]) for k in self.names}

    def mixed(self, w):
        """{name: [B, *shape]}: row b's parameters sum_x w[b, x] theta_x."""
        return {k: torch.einsum("bx,x...->b...", w, v) for k, v in self.stacked().items()}

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs, pref=None):
        if pref is None or pref.dim() != 2 or pref.shape[1] != 3:
            raise ValueError("a MixturePolicy needs pref [B, 3]: the mixing weights (also fed as the preference input)")
        w = pref.to(torch.float32)
        x = dict(fea_j=fea_j, op_mask=op_mask, candidate=candidate, fea_m=fea_m, mch_mask=mch_mask, comp_idx=comp_idx,
                 dynamic_pair_mask=dynamic_pair_mask, fea_pairs=fea_pairs, pref=pref)
        if self.grouped:
            return self._grouped(w, x)
        return functional_call(self.nets[0], self.mixed(w), kwargs=x)

    def _grouped(self, w, x):
        """Reference implementation: one ordinary forward per distinct preference row."""
        uniq, inv = torch.unique(w, dim=0, return_inverse=True)
        st = self.stacked()
        pi_out = v_out = None
        for g in range(len(uniq)):
            rows = (inv == g).nonzero().squeeze(1)
            params = {k: torch.einsum("x,x...->...", uniq[g], v) for k, v in st.items()}
            pi, v = functional_call(self.nets[0], params, kwargs={k: t[rows] for k, t in x.items()})
            if pi_out is None:
                pi_out = pi.new_zeros((len(w), pi.shape[1])); v_out = v.new_zeros((len(w), v.shape[1]))
            pi_out = pi_out.index_copy(0, rows, pi); v_out = v_out.index_copy(0, rows, v)
        return pi_out, v_out


def load_specialists(paths, config, device):
    """Three v3 specialists (schema benchmark_v3_f12_m9_p10) -> three preference-conditioned DANIELs, preference columns
    zero. Returns (nets, parents) with each source's path and SHA-256."""
    from pathlib import Path
    from checkpointing import file_sha256, unwrap_checkpoint, validate_checkpoint_schema
    from feature_schemas import SCHEMA_V3
    from model.main_model import DANIEL
    from warmstart import transfer
    if len(paths) != 3:
        raise ValueError("--merge_mix needs three specialists: makespan, carbon, tardiness (in that order)")
    nets, parents = [], []
    for p in paths:
        sd, meta = unwrap_checkpoint(torch.load(p, map_location=device, weights_only=False))
        validate_checkpoint_schema(meta, SCHEMA_V3, sd)
        net = DANIEL(config).to(device)
        report = transfer(sd, net, "full")
        nets.append(net)
        parents.append({"path": str(Path(p).resolve()), "sha256": file_sha256(p), "init_mode": "merge_mix",
                        "transfer": {k: len(v) for k, v in report.items()}})
    return nets, parents
