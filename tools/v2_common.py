"""Shared helpers for the v2 study (results/v2_fixes.md): v1/v2 policy loading, canonical grouped rollouts that
also return per-job completions, and the lock-step v1-vs-v2 check of Part B. EVALUATION ONLY.

Importing this module fixes the process config to the canonical v1 config (via complementarity); v2 networks are
built from a copy of that config with the v2 dimensions, and v2 environments are requested per instance
(FJSPEnvForSameOpNums(..., obs_version="v2")).
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import complementarity as P                           # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums          # noqa: E402
from model.main_model import DANIEL                             # noqa: E402
from common_utils import load_checkpoint_state_dict             # noqa: E402

V2_SCHEMA = "canonical_f13_p10_obsv2"
CHUNK = 5000


def v2_config():
    cfg = copy.copy(P.configs)
    cfg.fea_j_input_dim, cfg.fea_pair_input_dim = 13, 10
    cfg.feature_schema, cfg.obs_version = V2_SCHEMA, "v2"
    return cfg


def load_policy(name, obs="v1"):
    if obs == "v1":
        return P.load_policy(name)
    net = DANIEL(v2_config())
    net.load_state_dict(load_checkpoint_state_dict(P.CK / f"{name}.pth", map_location=P.configs.device,
                                                   expected_schema=V2_SCHEMA))
    net.eval()
    return net


def _groups(data, inst):
    ub = np.array([float(data[1][i][data[1][i] > 0].max()) for i in inst])
    for u in np.unique(ub):
        rows = np.flatnonzero(ub == u)
        for c0 in range(0, len(rows), CHUNK):
            yield rows[c0:c0 + CHUNK]


def _env(sub, dues, obs):
    env = FJSPEnvForSameOpNums(n_j=sub[0][0].shape[0], n_m=sub[1][0].shape[1], obs_version=obs, reward_version="v1")
    state = env.set_initial_data(*[list(d) for d in sub], due_date_list=list(dues))
    return env, state


def _finish(env, dues):
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - np.asarray(dues)).sum(1)
    return dict(mk=env.current_makespan.copy(), cb=env.total_carbon.copy(), tard=tard, job_completion=comp.copy())


@torch.no_grad()
def run(pol, data, due, inst, obs="v1", gen=None):
    """Greedy (gen None) or sampled rollouts; results in the order of `inst` (pool indices, repeats allowed)."""
    out = None
    for r in _groups(data, inst):
        sub = tuple([d[i] for i in inst[r]] for d in data); dues = [due[i] for i in inst[r]]
        env, state = _env(sub, dues, obs)
        while True:
            s = pol.forward_logits(**P.args_of(state))[0]
            a = (s.argmax(1) if gen is None else torch.multinomial(F.softmax(s, 1), 1, generator=gen).squeeze(1))
            state, _, done = env.step(actions=a.cpu().numpy())
            if np.all(done):
                break
        res = _finish(env, dues)
        if out is None:
            out = {k: np.zeros((len(inst),) + v.shape[1:]) for k, v in res.items()}
        for k, v in res.items():
            out[k][r] = v
    return out


@torch.no_grad()
def lockstep(pol1, pol2, data, due, inst):
    """v1 policy in the v1 env and v2 policy in the v2 env, side by side, each following its own greedy action.
    Returns per-instance results for both, the number of steps whose greedy actions differ, and the maximum
    absolute difference in action probabilities (over all valid and invalid actions, all steps)."""
    o1 = o2 = None; n_diff, max_dp, n_steps = 0, 0.0, 0
    for r in _groups(data, inst):
        sub = tuple([d[i] for i in inst[r]] for d in data); dues = [due[i] for i in inst[r]]
        e1, s1 = _env(sub, dues, "v1"); e2, s2 = _env(sub, dues, "v2")
        while True:
            p1 = F.softmax(pol1.forward_logits(**P.args_of(s1))[0], 1)
            p2 = F.softmax(pol2.forward_logits(**P.args_of(s2))[0], 1)
            a1, a2 = p1.argmax(1), p2.argmax(1)
            n_diff += int((a1 != a2).sum()); n_steps += len(a1)
            max_dp = max(max_dp, float((p1 - p2).abs().max()))
            s1, _, d1 = e1.step(actions=a1.cpu().numpy()); s2, _, d2 = e2.step(actions=a2.cpu().numpy())
            if np.all(d1):
                assert np.all(d2)
                break
        for which, e in ((1, e1), (2, e2)):
            res = _finish(e, dues)
            tgt = o1 if which == 1 else o2
            if tgt is None:
                tgt = {k: np.zeros((len(inst),) + v.shape[1:]) for k, v in res.items()}
                if which == 1: o1 = tgt
                else: o2 = tgt
            for k, v in res.items():
                tgt[k][r] = v
    return o1, o2, n_diff, n_steps, max_dp
