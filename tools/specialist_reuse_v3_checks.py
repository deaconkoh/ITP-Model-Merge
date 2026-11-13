"""Checks before any multi-objective stage of the specialist-reuse experiment (docs/specialist_reuse_protocol_v3.md):
for Rewarded Soups and for preference-weighted TIES (every screened keep and lambda),
  1. changing the preference changes the merged weights (two interior preferences give different weights);
  2. each exact corner e_x reproduces specialist x: merged weights equal its weights, and the merged policy's action
     probabilities on real validation states (waiting rule) equal the original unconditioned specialist's.

    python tools/specialist_reuse_v3_checks.py --specs m.pth,c.pth,t.pth --base base.pth --keep 0.2,0.5 --lam 1.0 --out f.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--specs", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--keep", required=True)
ap.add_argument("--lam", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--states", type=int, default=6, help="validation instances whose states are compared")
a = ap.parse_args()
sys.path.insert(0, str(REPO / "daniel"))
sys.argv = [sys.argv[0], "--device", "cpu", "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct",
            "--preference_dim", "3", "--dispatch_rule", "semi_active"]
import copy                                                         # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from params import configs                                          # noqa: E402
import benchmark_v3 as B                                            # noqa: E402
from common_utils import load_checkpoint_state_dict                 # noqa: E402
from feature_schemas import SCHEMA_V3                               # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums              # noqa: E402
from mixture import MixturePolicy, TiesMixturePolicy, load_specialists, load_base   # noqa: E402
from model.main_model import DANIEL                                 # noqa: E402

KEYS = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs")
EYE = torch.eye(3)
W1, W2 = torch.tensor([[0.6, 0.2, 0.2]]), torch.tensor([[0.2, 0.2, 0.6]])


def states():
    """Validation states along a greedy-free random legal trajectory (waiting rule), for output comparisons."""
    data = B.load_split("validation", "10x5")
    env = FJSPEnvForSameOpNums(10, 5)
    st = env.set_initial_data(**data.env_kwargs(range(a.states), "mct"))
    rng = np.random.default_rng(0)
    out = []
    for t in range(50):
        if t % 10 == 0:
            out.append({k: getattr(st, f"{k}_tensor").clone() for k in KEYS})
        m = ~env.dynamic_pair_mask.reshape(env.number_of_envs, -1)
        st, _, _ = env.step(np.array([rng.choice(np.flatnonzero(r)) for r in m]))
    return out


def main():
    paths = a.specs.split(",")
    nets, _ = load_specialists(paths, configs, "cpu")
    base = load_base(a.base, configs, "cpu")
    plain = []
    for p in paths:                                                  # the original unconditioned specialists
        c = copy.copy(configs); c.preference_dim = 0; c.feature_schema = SCHEMA_V3
        n = DANIEL(c); n.load_state_dict(load_checkpoint_state_dict(p, map_location="cpu", expected_schema=SCHEMA_V3))
        plain.append(n.eval())
    S = states()
    policies = {"rewarded_soups": MixturePolicy(copy.deepcopy(nets))}
    for k in a.keep.split(","):
        for l in a.lam.split(","):
            policies[f"ties_keep{k}_lambda{l}"] = TiesMixturePolicy(copy.deepcopy(nets), base, float(k), float(l))
    rec = {"checks": {}}
    ok_all = True
    for name, pol in policies.items():
        pol.eval()
        with torch.no_grad():
            m1, m2 = pol.mixed(W1), pol.mixed(W2)
            diff = max(float((m1[k] - m2[k]).abs().max()) for k in m1)
            corners = {}
            for x, o in enumerate("mct"):
                mx = pol.mixed(EYE[x:x + 1])
                own = dict(pol.nets[x].named_parameters())
                weights_equal = all(torch.equal(mx[k][0], own[k]) for k in mx)
                out_diff = 0.0
                for s in S:
                    pref = EYE[x].repeat(s["fea_j"].shape[0], 1)
                    pi_m, _ = pol(**s, pref=pref)
                    pi_s, _ = plain[x](**s)
                    out_diff = max(out_diff, float((pi_m - pi_s).abs().max()))
                corners[o] = {"weights_equal": weights_equal, "max_prob_difference": out_diff,
                              "pass": weights_equal and out_diff < 1e-5}
        ok = diff > 0 and all(c["pass"] for c in corners.values())
        ok_all &= ok
        rec["checks"][name] = {"preference_changes_weights": diff > 0, "max_weight_difference_w1_w2": diff,
                               "w1": W1[0].tolist(), "w2": W2[0].tolist(), "corners": corners, "pass": ok}
        print(name, "PASS" if ok else "FAIL", f"weight change {diff:.3g}",
              {o: (c["weights_equal"], f"{c['max_prob_difference']:.1e}") for o, c in corners.items()}, flush=True)
    rec["all_pass"] = bool(ok_all)
    rec["states_compared"] = sum(int(s["fea_j"].shape[0]) for s in S)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
