"""Shared helpers for the v3 specialist diagnosis (results/diagnose_v3.md).

Read-only use of the existing code: the v3 environment, the DANIEL network and the frozen v3 loaders. Never loads
the reserved final-test sets; tests use the validation split (the train pool for behaviour cloning, the evaluation
split only for the rule baselines of Test 1, next to the existing baselines table).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DANIEL_DIR = REPO / "daniel"
OUT = REPO / "results" / "diagnose_v3"
sys.path.insert(0, str(DANIEL_DIR))
sys.path.insert(0, str(REPO / "tools" / "objective_validation"))

_argv = sys.argv
sys.argv = [_argv[0], "--device", "cuda", "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct"]
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from params import configs                                  # noqa: E402
import benchmark_v3 as B                                    # noqa: E402
from checkpointing import unwrap_checkpoint                 # noqa: E402
from common_utils import load_checkpoint_state_dict         # noqa: E402
from feature_schemas import SCHEMA_V3                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums      # noqa: E402
from model.main_model import DANIEL                         # noqa: E402
import common as OV                                         # noqa: E402  (tools/objective_validation)
import schedules as S                                       # noqa: E402
sys.argv = _argv

DEVICE = configs.device
SIZES = ("10x5", "20x10")
SEEDS = (111, 222, 333)
GOALS = "mct"
# pair-feature columns (fjsp_env_same_op_nums.construct_pair_features): 8 time features, carbon, lateness
PAIR_NAMES = ["pt", "pt/op_max", "pt/mch_max_cand", "pt/max_remain", "pt/mch_max_remain", "pt/pair_max",
              "pt/job_remain_work", "wait_time", "carbon/13.365", "lateness/99"]
OP_NAMES = ["scheduled", "ct_lb", "min_pt", "pt_span", "mean_pt", "waiting", "remain_work", "job_left_ops",
            "job_remain_work", "avail_mch_frac", "slack", "time_to_due"]
MCH_NAMES = ["avail_jc", "avail_ops", "min_pt", "mean_pt", "waiting", "remain_work", "free_time", "working",
             "power/15"]


def run_name(size, goal, seed):
    return f"{size}+mix+bv3_spec_{goal}_s{seed}"


def load_net(name, which="best"):
    """which='best': the selected checkpoint (trained_network/SD2/<name>.pth); 'final': the policy at the last update,
    from the resume state <name>.resume.pth."""
    net = DANIEL(configs).to(DEVICE)
    if which == "best":
        path = DANIEL_DIR / "trained_network" / "SD2" / f"{name}.pth"
        net.load_state_dict(load_checkpoint_state_dict(path, map_location=DEVICE, expected_schema=SCHEMA_V3))
    else:
        st = torch.load(DANIEL_DIR / "trained_network" / "SD2" / f"{name}.resume.pth", map_location=DEVICE,
                        weights_only=False)
        net.load_state_dict(st["policy"])
    net.eval()
    return net


def load_resume(name):
    return torch.load(DANIEL_DIR / "trained_network" / "SD2" / f"{name}.resume.pth", map_location="cpu",
                      weights_only=False)


def make_env(data, idx, goal="mct"):
    n_j, n_m = B.SIZES[data.size]
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    state = env.set_initial_data(**data.env_kwargs(list(idx), goal))
    return env, state


def forward(net, state):
    return net(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
               fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
               dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)


def env_objectives(env):
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
    return env.current_makespan.copy(), env.total_carbon.copy(), tard


def ratios(data, idx, mk, cb, td):
    idx = list(idx)
    return {"mk_ratio": float(np.mean(np.asarray(mk) / np.asarray(data.mk_ref)[idx])),
            "carbon_ratio": float(np.mean(np.asarray(cb) / np.asarray(data.cb_ref)[idx])),
            "tardiness_ratio": float(np.mean(td) / data.t_ref),
            "makespan": float(np.mean(mk)), "carbon": float(np.mean(cb)), "tardiness": float(np.mean(td))}


def ov_instance(data, i):
    jl = np.asarray(data.job_length[i])
    return OV.Instance(data.keys[i], jl, np.asarray(data.op_pt[i]), np.repeat(np.arange(len(jl)), jl),
                       np.concatenate([[0], np.cumsum(jl)[:-1]]))


def save_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
