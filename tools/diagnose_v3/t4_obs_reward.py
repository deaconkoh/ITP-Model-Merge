"""Test 4: observations and reward (hypotheses A2, A3, B3).

(a) dump: carbon, power and due-date features of every candidate pair in 3 states per size (steps 0, N/2, N-3 of a
    DANIEL-C greedy rollout on validation instance 0), next to their raw sources;
(b) trace: a forward hook on the actor checks that its input's last 10 columns are exactly the pair features (carbon and
    lateness reach the actor directly) and that the machine-power column enters the machine attention blocks;
    ablation: on the states of each specialist's own greedy rollout, the share of decisions that change when one input
    column is replaced by its per-state mean over candidate pairs / machines (carbon, power, lateness, processing time);
(c) reward: per-step rewards summed over 50 episodes per size and goal (uniform-random allowed actions) vs
    -objective/ref;
(d) ranges: percentiles of the raw (unstandardised) pair features and of the raw due-date operation features over
    greedy rollouts of the whole validation split, 10x5 vs 20x10.

    python tools/diagnose_v3/t4_obs_reward.py
"""
import numpy as np
import torch
import dv3
from dv3 import B
from common_utils import greedy_select_action


def raw_due_features(env):
    est, work, pair_start, finished = env.v2_job_state()
    return env.due_dates - est - work, env.due_dates - est                    # slack, time_to_due [E,J] (/99 units)


@torch.no_grad()
def dump_states(net, data):
    env, state = dv3.make_env(data, [0])
    N = env.number_of_ops
    want = {0, N // 2, N - 3}
    out = []
    step = 0
    while True:
        if step in want:
            e = 0
            slack, ttd = raw_due_features(env)
            allowed = ~env.dynamic_pair_mask[e].astype(bool)
            rows = []
            for j in range(env.number_of_jobs):
                if env.mask[e, j]:
                    continue
                o = env.candidate[e, j]
                for m in np.flatnonzero(env.process_relation[e, o]):
                    rows.append({
                        "job": j, "op": int(o), "machine": int(m), "allowed": bool(allowed[j, m]),
                        "raw_pt": float(env.true_op_pt[e, o, m]),
                        "raw_carbon_kg": float(env.op_carbon[e, o, m]),
                        "fea_carbon": float(env.fea_pairs[e, j, m, 8]),
                        "raw_power_kw": float(env.machine_power[e, m]),
                        "fea_power_after_standardisation": float(env.fea_m[e, m, 8]),
                        "raw_due": float(env.true_due_dates[e, j]),
                        "raw_slack_/99": float(slack[e, j]), "fea_slack_std": float(env.fea_j[e, o, 10]),
                        "raw_time_to_due_/99": float(ttd[e, j]), "fea_time_to_due_std": float(env.fea_j[e, o, 11]),
                        "fea_lateness_/99": float(env.fea_pairs[e, j, m, 9])})
            out.append({"step": step, "pairs": rows})
        pi, _ = dv3.forward(net, state)
        a = greedy_select_action(pi).cpu().numpy()
        state, _, done = env.step(a)
        step += 1
        if np.all(done):
            break
    return out


@torch.no_grad()
def trace_and_ablate(net, data, idx):
    """Actor-input identity check plus per-column ablation agreement on the model's own greedy trajectory."""
    captured = {}
    h = net.actor.register_forward_hook(lambda mod, inp, outp: captured.__setitem__("x", inp[0].detach()))
    env, state = dv3.make_env(data, idx)
    abl = {"carbon(pair 8)": ("pair", 8), "lateness(pair 9)": ("pair", 9), "pt(pair 0)": ("pair", 0),
           "power(mch 8)": ("mch", 8), "slack(op 10)": ("op", 10), "time_to_due(op 11)": ("op", 11)}
    same = {k: 0 for k in abl}
    n = 0
    identity_ok = True
    while True:
        pi, _ = dv3.forward(net, state)
        x = captured["x"]
        fp = state.fea_pairs_tensor.reshape(x.shape[0], -1, x.shape[-1] - 32)
        identity_ok &= bool(torch.equal(x[..., 32:], fp))
        a = greedy_select_action(pi)
        allowed = ~state.dynamic_pair_mask_tensor.bool()                     # [E,J,M]
        for k, (kind, col) in abl.items():
            fj, fm, fpairs = state.fea_j_tensor.clone(), state.fea_m_tensor.clone(), state.fea_pairs_tensor.clone()
            if kind == "pair":
                v = fpairs[..., col]
                mean = (v * allowed).sum((1, 2), keepdim=True) / allowed.sum((1, 2), keepdim=True).clamp(min=1)
                fpairs[..., col] = mean.expand_as(v)
            elif kind == "mch":
                fm[..., col] = fm[..., col].mean(1, keepdim=True)
            else:
                fj[..., col] = fj[..., col].mean(1, keepdim=True)
            pi2, _ = net(fea_j=fj, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor, fea_m=fm,
                         mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                         dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=fpairs)
            same[k] += int((greedy_select_action(pi2) == a).sum())
        n += len(a)
        state, _, done = env.step(a.cpu().numpy())
        if np.all(done):
            break
    h.remove()
    return identity_ok, {k: 1 - v / n for k, v in same.items()}


def reward_check(size, goal, data, episodes=50):
    idx = [i % len(data) for i in range(episodes)]
    env, state = dv3.make_env(data, idx, goal)
    rng = np.random.default_rng(7)
    total = np.zeros(episodes)
    while True:
        allowed = ~env.dynamic_pair_mask.reshape(episodes, -1).astype(bool)
        a = np.array([rng.choice(np.flatnonzero(r)) for r in allowed])
        _, r, done = env.step(a)
        total += r
        if np.all(done):
            break
    mk, cb, td = dv3.env_objectives(env)
    score = B.score(B.goal_weights(goal), mk, cb, td, np.asarray(data.mk_ref)[idx], np.asarray(data.cb_ref)[idx],
                    data.t_ref)
    return {"max_abs_diff": float(np.max(np.abs(total + score))), "mean_return": float(total.mean()),
            "mean_score": float(score.mean()), "return_range": [float(total.min()), float(total.max())]}


@torch.no_grad()
def ranges(net, data):
    env, state = dv3.make_env(data, range(len(data)))
    acc = {k: [] for k in dv3.PAIR_NAMES}
    acc["raw_slack_/99"], acc["raw_time_to_due_/99"] = [], []
    while True:
        live = ~env.mask.astype(bool)
        allowed = ~env.dynamic_pair_mask.astype(bool)
        for c, k in enumerate(dv3.PAIR_NAMES):
            acc[k].append(env.fea_pairs[..., c][allowed])
        slack, ttd = raw_due_features(env)
        acc["raw_slack_/99"].append(slack[live]); acc["raw_time_to_due_/99"].append(ttd[live])
        pi, _ = dv3.forward(net, state)
        state, _, done = env.step(greedy_select_action(pi).cpu().numpy())
        if np.all(done):
            break
    return {k: {f"p{q}": float(np.percentile(np.concatenate(v), q)) for q in (1, 50, 99)} |
            {"max": float(np.concatenate(v).max()), "min": float(np.concatenate(v).min())} for k, v in acc.items()}


def main():
    out = {"dump": {}, "trace": {}, "reward": {}, "ranges": {}}
    for size in dv3.SIZES:
        data = B.load_split("validation", size)
        idx = list(range(len(data)))
        out["dump"][size] = dump_states(dv3.load_net(dv3.run_name(size, "c", 111)), data)
        for goal in "mct":
            for seed in dv3.SEEDS:
                name = dv3.run_name(size, goal, seed)
                ok, ab = trace_and_ablate(dv3.load_net(name), data, idx)
                out["trace"][name] = {"actor_input_last10_equal_pair_features": ok, "decisions_changed": ab}
                print(name, ok, {k: round(v, 3) for k, v in ab.items()}, flush=True)
            out["reward"][f"{size}_{goal}"] = reward_check(size, goal, data)
            print(size, goal, out["reward"][f"{size}_{goal}"], flush=True)
        out["ranges"][size] = ranges(dv3.load_net(dv3.run_name(size, "c", 111)), data)
        for k, v in out["ranges"][size].items():
            print(size, f"{k:22s}", {a: round(b, 3) for a, b in v.items()})
    # the power column is consumed by the machine attention block's first layer (input width 9)
    net = dv3.load_net(dv3.run_name("10x5", "c", 111))
    out["power_enters_machine_attention"] = {
        n: list(p.shape) for n, p in net.named_parameters() if n.startswith("feature_exact.mch_attention_blocks.0.")
        and p.dim() == 2 and 9 in p.shape}
    dv3.save_json("t4_obs_reward.json", out)


if __name__ == "__main__":
    main()
