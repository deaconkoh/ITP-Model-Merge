"""Test 6: training dynamics of all 18 runs (hypotheses A5, B1, B2, B4).

What train.py logged per update: the mean training return (train_log/SD2/reward_<run>.json), the mean total PPO loss,
training makespan and carbon (stdout only), and every 50 updates the validation objectives and selection score
(valiquality_<run>.json). NOT logged: policy entropy, approximate KL, clip fraction, value loss on its own, explained
variance, gradient norm. Reconstructed here WITHOUT training, from the two policies each run kept (the selected "best"
checkpoint and the "final" policy of the resume state):
  * policy entropy (raw and normalised by log #allowed pairs) and KL(best || final) on the same states,
  * critic explained variance and value/return scale on sampled episodes with the run's own reward,
  * Adam's second-moment estimate at the last update (RMS gradient over roughly the last 1,000 steps) for actor and
    critic parameters,
plus, from the logs: the training return and total loss around the best update and at the end (B4: does training
return keep improving while validation degrades?).

    python tools/diagnose_v3/t6_logs.py
"""
import json, re
import numpy as np
import torch
import dv3
from dv3 import B

LOG = dv3.DANIEL_DIR / "train_log" / "SD2"
EP = re.compile(r"Episode (\d+)\t reward: ([-\d.]+)\t makespan: ([-\d.]+)\t Mean_loss: ([-\d.e+]+)")


def parse(name):
    ret = json.loads((LOG / f"reward_{name}.json").read_text())
    val = json.loads((LOG / f"valiquality_{name}.json").read_text())
    loss = {}
    for line in (LOG / f"{name}.stdout.log").read_text(errors="replace").splitlines():
        for m in EP.finditer(line):
            loss[int(m.group(1))] = float(m.group(4))                       # the last occurrence wins (resumed chunks)
    r = np.array([e["mean_return"] for e in ret])
    lo = np.array([loss.get(u + 1, np.nan) for u in range(len(r))])
    return r, lo, val


def window(x, a, b):
    a, b = max(0, a), min(len(x), b)
    return float(np.nanmean(x[a:b])) if b > a else float("nan")


@torch.no_grad()
def probe(net, data, goal, reps=4, seed=0):
    """Sampled episodes (reps per validation instance) with the run's reward: per-state entropy and normalised entropy,
    critic values and returns-to-go; also returns the states' policies for a KL comparison."""
    idx = [i for i in range(len(data)) for _ in range(reps)]
    env, state = dv3.make_env(data, idx, goal)
    gen = torch.Generator(device=dv3.DEVICE); gen.manual_seed(seed)
    ent, nent, vals, rews, states = [], [], [], [], []
    while True:
        pi, v = dv3.forward(net, state)
        logp = torch.log(pi.clamp_min(1e-30))
        h = -(pi * logp).sum(1)
        n_allowed = (~state.dynamic_pair_mask_tensor.bool()).reshape(len(idx), -1).sum(1).float()
        ent.append(h); nent.append(torch.where(n_allowed > 1, h / torch.log(n_allowed.clamp_min(2)), torch.zeros_like(h)))
        vals.append(v.squeeze(1))
        states.append({k: getattr(state, f"{k}_tensor").clone() for k in
                       ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask",
                        "fea_pairs")})
        a = torch.multinomial(pi, 1, generator=gen).squeeze(1)
        state, r, done = env.step(a.cpu().numpy())
        rews.append(torch.as_tensor(r, device=dv3.DEVICE, dtype=torch.float32))
        if np.all(done):
            break
    R = torch.stack(rews)                                                      # [T, E]
    G = torch.flip(torch.cumsum(torch.flip(R, [0]), 0), [0])                   # returns-to-go, gamma = 1
    V = torch.stack(vals)
    ev = 1 - torch.var(G - V) / torch.var(G)
    return {"entropy": float(torch.cat(ent).mean()), "entropy_normalised": float(torch.cat(nent).mean()),
            "explained_variance": float(ev), "value_mean": float(V.mean()), "return_to_go_mean": float(G.mean()),
            "value_abs_error": float((G - V).abs().mean()), "episode_return_mean": float(G[0].mean()),
            "episode_return_std": float(G[0].std())}, states


@torch.no_grad()
def kl(net_p, net_q, states):
    tot, n = 0.0, 0
    for s in states:
        p, _ = net_p(**s); q, _ = net_q(**s)
        m = p > 0
        tot += float((torch.where(m, p * (torch.log(p.clamp_min(1e-30)) - torch.log(q.clamp_min(1e-30))), 0)).sum())
        n += p.shape[0]
    return tot / n


def adam_rms(resume, net):
    """sqrt of Adam's exp_avg_sq averaged over each parameter group of the network (actor, critic, feature extractor)."""
    names = [n for n, _ in net.named_parameters()]
    st = resume["optimizer"]["state"]
    out = {"actor": [], "critic": [], "feature_exact": []}
    for i, n in enumerate(names):
        if i in st:
            out[n.split(".")[0]].append(float(st[i]["exp_avg_sq"].sqrt().mean()))
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def main():
    out = {}
    for size in dv3.SIZES:
        data = B.load_split("validation", size)
        for goal in "mct":
            for seed in dv3.SEEDS:
                name = dv3.run_name(size, goal, seed)
                r, lo, val = parse(name)
                st = json.loads((LOG / f"{name}.status.json").read_text())
                best_u, last_u = st["best_update"], st["update"]
                vs = {e["update"]: e["selection_metric"] for e in val}
                resume = dv3.load_resume(name)
                nb, nf = dv3.load_net(name, "best"), dv3.load_net(name, "final")
                pb, states = probe(nb, data, goal)
                pf, _ = probe(nf, data, goal)
                rec = {
                    "best_update": best_u, "last_update": last_u,
                    "val_at_best": vs[best_u], "val_at_last": vs[max(vs)], "val_at_0": vs[0],
                    "val_min_last_500": min(v for u, v in vs.items() if u > last_u - 500),
                    "train_return_first_100": window(r, 0, 100),
                    "train_return_around_best": window(r, best_u - 100, best_u + 100),
                    "train_return_last_200": window(r, len(r) - 200, len(r)),
                    "train_return_best_window": max(window(r, a, a + 100) for a in range(0, len(r) - 99, 50)),
                    "loss_first_100": window(lo, 0, 100), "loss_around_best": window(lo, best_u - 100, best_u + 100),
                    "loss_last_200": window(lo, len(lo) - 200, len(lo)),
                    "loss_max_window": max(window(lo, a, a + 50) for a in range(0, len(lo) - 49, 50)),
                    "probe_best": pb, "probe_final": pf, "kl_best_to_final": kl(nb, nf, states),
                    "adam_rms_final": adam_rms(resume, nf),
                    "curve": {"train_return_per50": [window(r, a, a + 50) for a in range(0, len(r), 50)],
                              "loss_per50": [window(lo, a, a + 50) for a in range(0, len(lo), 50)],
                              "val": [[u, vs[u]] for u in sorted(vs)]}}
                out[name] = rec
                print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                             if k not in ("curve", "probe_best", "probe_final", "adam_rms_final")}, flush=True)
                print("   best ", {k: round(v, 4) for k, v in pb.items()})
                print("   final", {k: round(v, 4) for k, v in pf.items()}, rec["adam_rms_final"], flush=True)
    dv3.save_json("t6_logs.json", out)


if __name__ == "__main__":
    main()
