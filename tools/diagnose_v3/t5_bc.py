"""Test 5: learnability (A2 vs A5). Behaviour-clone a FRESH DANIEL network (same architecture and v3 inputs) on the
Test 2 rule "lowest-energy allowed pair" at 10x5, supervised (cross-entropy on the masked logits), no RL.

Training states: rule rollouts on the v3 train pool (63 instances) with epsilon-random actions (eps 0.2) for state
diversity, labelled with the rule's action. Held-out: the validation split (18 instances), states from the pure rule
trajectory. Reported: top-1 imitation accuracy, "carbon-equivalent" accuracy (chosen pair has the rule's carbon), and
the carbon of greedy rollouts of the cloned policy vs the rule and vs DANIEL-C. Budget: <= 15 GPU-minutes.

    python tools/diagnose_v3/t5_bc.py
"""
import random, time
import numpy as np
import torch
import torch.nn.functional as F
import dv3
from dv3 import B, configs
from model.main_model import DANIEL
from common_utils import greedy_select_action
from t2_mask import min_energy_allowed, pair_view

SEED, EPS, PASSES, BUDGET_S, BATCH, LR = 20261018, 0.2, 12, 15 * 60, 256, 1e-3
KEYS = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs")


def collect(data, eps, rng):
    env, state = dv3.make_env(data, range(len(data)))
    buf = {k: [] for k in KEYS}; labels, cmin = [], []
    while True:
        for k in KEYS:
            buf[k].append(getattr(state, f"{k}_tensor"))
        a = min_energy_allowed(env)
        carbon, _, allowed, _, _ = pair_view(env)
        flat_c = np.where(allowed, carbon, np.inf).reshape(env.number_of_envs, -1)
        labels.append(torch.as_tensor(a, device=dv3.DEVICE)); cmin.append(torch.as_tensor(flat_c, device=dv3.DEVICE))
        act = a.copy()
        if eps > 0:
            flip = rng.random(len(a)) < eps
            for e in np.flatnonzero(flip):
                act[e] = rng.choice(np.flatnonzero(np.isfinite(flat_c[e])))
        state, _, done = env.step(act)
        if np.all(done):
            break
    return {k: torch.cat(v) for k, v in buf.items()}, torch.cat(labels), torch.cat(cmin)


def logits(net, X, sl):
    sc, _, _ = net.forward_logits(**{k: X[k][sl] for k in KEYS})
    return sc


@torch.no_grad()
def accuracy(net, X, y, cflat):
    hit = eq = 0
    for s in range(0, len(y), 1024):
        sl = slice(s, s + 1024)
        p = logits(net, X, sl).argmax(1)
        hit += int((p == y[sl]).sum())
        c = cflat[sl]
        eq += int((c.gather(1, p[:, None]).squeeze(1) <= c.min(1).values + 1e-9).sum())
    return hit / len(y), eq / len(y)


@torch.no_grad()
def rollout_carbon(net_or_rule, data):
    env, state = dv3.make_env(data, range(len(data)))
    while True:
        if callable(net_or_rule) and not isinstance(net_or_rule, torch.nn.Module):
            a = net_or_rule(env)
        else:
            pi, _ = dv3.forward(net_or_rule, state); a = greedy_select_action(pi).cpu().numpy()
        state, _, done = env.step(a)
        if np.all(done):
            break
    return dv3.ratios(data, range(len(data)), *dv3.env_objectives(env))


def main():
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    train, val = B.load_split("train", "10x5"), B.load_split("validation", "10x5")
    parts = [collect(train, EPS, rng) for _ in range(PASSES)]
    X = {k: torch.cat([p[0][k] for p in parts]) for k in KEYS}
    y = torch.cat([p[1] for p in parts]); cy = torch.cat([p[2] for p in parts])
    Xv, yv, cv = collect(val, 0.0, rng)
    net = DANIEL(configs).to(dv3.DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    hist = []
    acc0 = accuracy(net, Xv, yv, cv)
    hist.append({"epoch": 0, "val_top1": acc0[0], "val_carbon_equiv": acc0[1], "seconds": time.time() - t0})
    epoch = 0
    while time.time() - t0 < BUDGET_S - 60 and epoch < 40:
        epoch += 1
        net.train()
        perm = torch.randperm(len(y), device=dv3.DEVICE)
        tot = 0.0
        for s in range(0, len(y), BATCH):
            b = perm[s:s + BATCH]
            sc, _, _ = net.forward_logits(**{k: X[k][b] for k in KEYS})
            loss = F.cross_entropy(sc, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        net.eval()
        acc = accuracy(net, Xv, yv, cv)
        hist.append({"epoch": epoch, "train_loss": tot / len(y), "val_top1": acc[0], "val_carbon_equiv": acc[1],
                     "seconds": time.time() - t0})
        print(hist[-1], flush=True)
        if acc[1] > 0.995:
            break
    out = {"train_states": int(len(y)), "val_states": int(len(yv)), "history": hist,
           "bc_policy": rollout_carbon(net, val), "rule": rollout_carbon(min_energy_allowed, val),
           "daniel_c": {s: rollout_carbon(dv3.load_net(dv3.run_name("10x5", "c", s)), val) for s in dv3.SEEDS},
           "seconds": time.time() - t0, "budget_s": BUDGET_S}
    print({k: v for k, v in out.items() if k != "history"})
    dv3.save_json("t5_bc.json", out)


if __name__ == "__main__":
    main()
