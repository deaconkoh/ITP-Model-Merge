"""Update-0 diagnostics of a starting network (docs/merge_failure_test_protocol_v3.md, Part A). No training.

The network is built exactly as train.py builds it at update 0 (same flags, same seed, same --init_from transfer), so
the measured network is the one a contrastive run starts from. Measurements, on the validation split (all instances x
the 15 validation preferences; never the final-test sets):
  1. policy entropy (mean per-step entropy along sampled schedules) and distinct schedules out of K sampled per
     (instance, preference);
  2. per layer, the fraction of near-inactive units (|activation| <= INACTIVE_EPS on at least INACTIVE_FRAC of the
     entries of the validation batch: every state visited by the greedy validation rollouts), and for tanh layers the
     fraction of saturated units (|activation| >= SAT_EPS on at least INACTIVE_FRAC of the entries);
  3. per layer, the weight norm (Frobenius norm of all its parameters), plus the norm of the preference-input columns;
  4. the gradient norm of the contrastive loss on one batch: the first training batch of the run (same instances and
     preferences for every network, K schedules sampled from the network), before clipping, total and per layer;
  5. preference sensitivity: on every state of the greedy validation rollouts, the policy is evaluated under each of the
     15 validation preferences: share of states whose greedy action changes with the preference, mean total-variation
     distance between the action distributions of two preferences, and distinct greedy schedules per instance out of 15.

    cd daniel && python merge_diagnostics.py --diag_out OUT.json --diag_label NAME --diag_seed 111 [--diag_save_start 1] \
        -- <train.py flags>
"""
import sys

_own, _rest = sys.argv[1:], []
if "--" in _own:
    i = _own.index("--"); _own, _rest = _own[:i], _own[i + 1:]
_opts = dict(zip(_own[::2], _own[1::2]))
sys.argv = [sys.argv[0]] + _rest                         # train.py's flags, parsed by params on import

import json, time                                        # noqa: E402
from collections import OrderedDict                      # noqa: E402
from pathlib import Path                                 # noqa: E402
import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
import train as T                                        # noqa: E402
import contrastive as CL                                 # noqa: E402
from common_utils import sample_action, greedy_select_action, setup_seed   # noqa: E402

INACTIVE_EPS, SAT_EPS, INACTIVE_FRAC = 1e-2, 0.99, 0.99


def layer_of(name):
    """Group parameters into layers: attention blocks, actor/critic linear layers."""
    p = name.split(".")
    if p[0] == "feature_exact":
        return ".".join(p[:3])                           # feature_exact.op_attention_blocks.0
    return ".".join(p[:3])                               # actor.linears.0


class Activations:
    """Forward hooks: attention-block outputs (ELU) and actor/critic hidden layers (tanh)."""

    def __init__(self, net):
        self.stats, self.handles, self.on = OrderedDict(), [], True
        fe = net.feature_exact
        for kind, blocks in (("op_attention_blocks", fe.op_attention_blocks), ("mch_attention_blocks", fe.mch_attention_blocks)):
            for i, b in enumerate(blocks):
                self.add(b, f"feature_exact.{kind}.{i}", "elu", lambda o: o)
        for head in ("actor",):                          # the critic is never used by the contrastive objective
            mlp = getattr(net, head)
            for i in range(mlp.num_layers - 1):
                self.add(mlp.linears[i], f"{head}.linears.{i}", "tanh", torch.tanh)

    def add(self, mod, name, act, f):
        self.stats[name] = {"activation": act, "n": 0, "small": None, "sat": None}

        def hook(_m, _inp, out):
            if not self.on:
                return
            a = f(out.detach()).abs().reshape(-1, out.shape[-1])
            s = self.stats[name]
            small, sat = (a <= INACTIVE_EPS).sum(0).double(), (a >= SAT_EPS).sum(0).double()
            s["small"] = small if s["small"] is None else s["small"] + small
            s["sat"] = sat if s["sat"] is None else s["sat"] + sat
            s["n"] += a.shape[0]
        self.handles.append(mod.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()

    def summary(self):
        out = OrderedDict()
        for k, s in self.stats.items():
            small, sat = s["small"] / s["n"], s["sat"] / s["n"]
            out[k] = {"activation": s["activation"], "units": int(len(small)), "entries": int(s["n"]),
                      "near_inactive": float((small >= INACTIVE_FRAC).double().mean()),
                      "mean_share_small": float(small.mean()),
                      **({"saturated": float((sat >= INACTIVE_FRAC).double().mean())} if s["activation"] == "tanh" else {})}
        return out


def policy_call(policy, state, pref=None, rows=None):
    x = CL.policy_inputs(state)
    if rows is not None:
        x = {k: v[rows] for k, v in x.items()}
    if pref is not None:
        x["pref"] = pref
    return policy(**x)[0]


@torch.no_grad()
def greedy_part(tr):
    """Greedy validation rollouts: activations on every visited state, and preference sensitivity on the same states."""
    net, env = tr.ppo.policy, tr.vali_env
    grid = torch.as_tensor(np.asarray(tr.mo_grid), dtype=torch.float32)
    n_i, n_g = len(tr.mo_val_keys), len(tr.mo_grid)
    acts = Activations(net)
    state, changed, tv, n_states, seqs = env.reset(), 0, 0.0, 0, []
    alive = np.ones(n_i * n_g, bool)                     # rows whose schedule is not finished yet
    acts.on = False
    while True:
        pi = policy_call(net, state)
        rows = torch.as_tensor(np.flatnonzero(alive), device=pi.device)
        acts.on = True
        policy_call(net, state, rows=rows)               # hooks record the unfinished rows only
        acts.on = False
        probs = torch.stack([policy_call(net, state, g.expand(len(rows), 3).to(pi.device), rows=rows)
                             for g in grid])             # [G, alive, A]
        arg = probs.argmax(-1)
        changed += int((arg != arg[:1]).any(0).sum())
        G = len(grid)
        d = sum(0.5 * (probs[i] - probs[j]).abs().sum(-1) for i in range(G) for j in range(i + 1, G)) / (G * (G - 1) / 2)
        tv += float(d.sum()); n_states += len(rows)
        a = greedy_select_action(pi)
        seqs.append(np.where(alive, a.cpu().numpy(), -1))
        state, _, done = env.step(a.cpu().numpy())
        alive = ~np.asarray(done, bool).reshape(-1)
        if done.all():
            break
    acts.remove()
    seq = np.stack(seqs, 1).reshape(n_i, n_g, -1)
    distinct = [len({seq[i, g].tobytes() for g in range(n_g)}) for i in range(n_i)]
    return acts.summary(), {"states": n_states, "share_greedy_action_changes": changed / n_states,
                            "mean_tv_between_preferences": tv / n_states,
                            "distinct_greedy_schedules_of_15": float(np.mean(distinct))}


@torch.no_grad()
def sampled_part(tr, k, seed):
    """K sampled schedules per (validation instance, preference): mean per-step entropy, distinct schedules out of K."""
    net, env = tr.ppo.policy, tr.vali_env
    n_i, n_g = len(tr.mo_val_keys), len(tr.mo_grid)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    runs, ent_sum, ent_n = [], 0.0, 0
    for _ in range(k):
        state, seq = env.reset(), []
        alive = np.ones(n_i * n_g, bool)
        while True:
            pi = policy_call(net, state)
            a, _ = sample_action(pi)
            ent = -(pi * torch.log(pi.clamp_min(1e-30))).sum(1)
            live = torch.as_tensor(alive, device=ent.device)
            ent_sum += float(ent[live].sum()); ent_n += int(alive.sum())
            seq.append(np.where(alive, a.cpu().numpy(), -1))
            state, _, done = env.step(a.cpu().numpy())
            alive = ~np.asarray(done, bool).reshape(-1)
            if done.all():
                break
        runs.append(np.stack(seq, 1))                    # [N, T]
    R = np.stack(runs, 0)                                # [K, N, T]
    distinct = [len({R[r, e].tobytes() for r in range(k)}) for e in range(R.shape[1])]
    per_pref = np.asarray(distinct, float).reshape(n_i, n_g).mean(0)
    return {"k": k, "entropy": ent_sum / ent_n, "distinct_schedules_of_k": float(np.mean(distinct)),
            "distinct_by_preference": [float(x) for x in per_pref]}


def weight_norms(net, n_pref):
    from warmstart import PREF_LAYERS
    groups = OrderedDict()
    for k, p in net.named_parameters():
        groups.setdefault(layer_of(k), 0.0)
        groups[layer_of(k)] += float((p.detach().double() ** 2).sum())
    out = OrderedDict((k, v ** 0.5) for k, v in groups.items())
    sd = dict(net.named_parameters())
    pref = {k: float(sd[k][:, -n_pref:].detach().double().norm()) for k in PREF_LAYERS}
    total = sum(groups.values()) ** 0.5
    return {"per_layer": out, "preference_columns": pref, "total": total}


def gradient_part(tr):
    """One contrastive update's gradient, exactly as train.py computes it at update 0 (same batch and preferences);
    the pre-clip norms are captured inside clip_grad_norm_. The weights change afterwards, so this runs last."""
    setup_seed(tr.seed_train)
    tr.log = []
    tr.pref_rng = np.random.default_rng([tr.seed_train, 20261019])
    rec = {}
    orig = torch.nn.utils.clip_grad_norm_

    def capture(params, clip, *a, **kw):
        params = list(params)
        names = {id(p): n for n, p in tr.ppo.policy.named_parameters()}
        g = OrderedDict()
        for p in params:
            if p.grad is not None:
                g.setdefault(layer_of(names[id(p)]), 0.0)
                g[layer_of(names[id(p)])] += float((p.grad.detach().double() ** 2).sum())
        rec["per_layer"] = OrderedDict((k, v ** 0.5) for k, v in g.items())
        rec["total"] = sum(g.values()) ** 0.5
        return orig(params, clip, *a, **kw)
    torch.nn.utils.clip_grad_norm_ = capture
    try:
        tr.contrastive_update(0)
    finally:
        torch.nn.utils.clip_grad_norm_ = orig
    cl = tr.log[-1]["cl"]
    rec.update(logged_grad_norm=cl["grad_norm"], loss=cl["loss"], pairs=cl["pairs"],
               groups_with_pair=cl["groups_with_pair"], train_entropy=cl["entropy"],
               train_distinct_schedules=cl["distinct_schedules"], anchor=tr.config.cl_anchor)
    return rec


def main():
    t0 = time.time()
    tr = T.Trainer(T.configs)
    if _opts.get("--diag_save_start", "0") == "1":     # the exact update-0 network, as <model>@u0.pth
        tr.save_budget_checkpoint(0)
    seed = int(_opts.get("--diag_seed", tr.seed_train))
    net = tr.ppo.policy
    net.eval()
    start_sha = tr.cl_ref_sha
    vhv = tr.validate_mo()["hv"]                         # the update-0 validation HV, as train.py logs it
    net.eval()
    acts, pref = greedy_part(tr)
    samp = sampled_part(tr, tr.config.cl_k, seed)
    norms = weight_norms(net, tr.config.preference_dim)
    net.train()
    grad = gradient_part(tr)
    out = {"label": _opts.get("--diag_label", tr.model_name), "model_name": tr.model_name,
           "start_sha256": start_sha, "parents": tr.parent_checkpoints,
           "validation": {"instances": len(tr.mo_val_keys), "preferences": len(tr.mo_grid),
                          "hv_update0": vhv},
           "sampled": samp, "activations": acts, "thresholds": {"inactive_eps": INACTIVE_EPS, "saturated_eps": SAT_EPS,
                                                                "share_of_entries": INACTIVE_FRAC},
           "weight_norms": norms, "gradient": grad, "preference_sensitivity": pref,
           "seconds": time.time() - t0}
    Path(_opts["--diag_out"]).write_text(json.dumps(out, indent=1) + "\n")
    print(f"diagnostics written: {_opts['--diag_out']}")


if __name__ == "__main__":
    main()
