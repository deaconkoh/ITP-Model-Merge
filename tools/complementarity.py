"""RQ1 complementarity pilot -- rollout engine (EVALUATION ONLY: no training, no new checkpoints).

For each seed, the frozen makespan and carbon specialists (10x5) are run side by side on a batch of
instances. At every step BOTH specialists score the same set of valid operation-machine pairs; a
per-step "chooser" turns the two distributions into one action. Every method in the pilot is a chooser:

  spec_m / spec_c        one specialist alone (greedy)
  prob_<wc>              static probability mixture  p = w_m p_M + w_c p_C           (greedy)
  logit_<wc>             static logit mixture  log p = w_m log p_M + w_c log p_C, renormalised (greedy)
  stoch_<wc>_r<k>        stochastic expert selection: per step, expert M with prob w_m, else C;
                         take that expert's greedy action (5 repeats, seeded)
  merge_<c>              weight merge (Model Soup) of M and C at lambda = c/1000 (existing checkpoints)
  assign_<XXXXX>         switching oracle: K = 5 equal segments of decisions, segment i uses expert X_i

Masked / zero probabilities. The network's masked logits are -inf on invalid pairs. Log-probabilities come
from log_softmax of those logits, so every VALID pair has a finite log-probability and every invalid pair is
-inf in both specialists. The logit mixture is formed only over valid pairs (invalid entries are set to
-inf explicitly, never computed as 0 * -inf), then renormalised with log_softmax. At w = 0 or 1 the mixture
is exactly that specialist. The probability mixture is 0 on invalid pairs by construction.

Outputs results/complementarity/raw/seed<s>.npz (makespan, carbon per method and instance) and
results/complementarity/raw/diag_seed<s>.npz (per-step disagreement diagnostics).

    python tools/complementarity.py --seeds 111 222 333 444
"""
from __future__ import annotations
import argparse, itertools, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
sys.path.insert(0, str(DAN))
_ARGV, sys.argv = sys.argv[1:], [sys.argv[0], "--config", str(REPO / "configs/canonical/mc.json"),
                                 "--device", "cuda"]
import torch                                                         # noqa: E402
import torch.nn.functional as F                                      # noqa: E402
from params import configs                                           # noqa: E402
from model.PPO import PPO_initialize                                 # noqa: E402
from common_utils import load_checkpoint_state_dict                  # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums               # noqa: E402
from data_utils import load_priority_carbon_data_from_files          # noqa: E402
from objectives import ObjectiveSpec                                 # noqa: E402

CK = DAN / "trained_network/SD2"
OUT = REPO / "results/complementarity/raw"
WCS = [round(0.05 * i, 2) for i in range(21)]
K = 5
DEV = torch.device(configs.device)


def load_policy(name):
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(load_checkpoint_state_dict(
        CK / f"{name}.pth", map_location=configs.device, expected_schema=configs.feature_schema))
    ppo.policy.eval()
    return ppo.policy


def load_data(split):
    return load_priority_carbon_data_from_files(str(DAN / f"data/{split}/SD2/10x5+carbon+priority"))


def args_of(state):
    return dict(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)


def rollout(data, policies, chooser, diag=None, values=None, objective=None):
    """Batch instances in groups that share the same maximum processing time.

    The environment normalises processing times by the maximum over the WHOLE batch, so batching instances
    with different maxima changes the policy's inputs relative to the canonical one-instance-at-a-time
    evaluation (observed: 3/100 instances decided differently). Within a group of equal maxima the batch
    normalisation equals the per-instance one, so grouped batching reproduces the canonical evaluation.
    Results are returned in the original instance order; diagnostics and values are concatenated by group.
    """
    ub = np.array([float(pt[pt > 0].max()) for pt in data[1]])
    mk, cb = np.zeros(len(ub)), np.zeros(len(ub))
    for u in np.unique(ub):
        idx = np.flatnonzero(ub == u)
        sub = tuple([d[i] for i in idx] for d in data)
        d_g = {} if diag is not None else None
        v_g = [] if values is not None else None
        m, c = _rollout(sub, policies, chooser, diag=d_g, values=v_g, objective=objective)
        mk[idx], cb[idx] = m, c
        if diag is not None:
            for k, v in d_g.items():
                full = diag.setdefault(k, [np.full(len(ub), np.nan) for _ in v])
                for t, arr in enumerate(v):
                    full[t][idx] = arr
        if values is not None:
            if not values:
                values.extend([(np.full(len(ub), np.nan), np.full(len(ub), np.nan)) for _ in v_g])
            for t, (vv, rr) in enumerate(v_g):
                values[t][0][idx] = vv; values[t][1][idx] = rr
    return mk, cb


@torch.no_grad()
def _rollout(data, policies, chooser, diag=None, values=None, objective=None):
    """Batched greedy rollout. policies: list of networks whose log-probs are handed to chooser(t, logps).
    Returns (makespan, carbon) arrays over instances. diag: dict to fill with per-step diagnostics
    (requires two policies). values: list to fill with (value estimates, rewards) per step for policy 0."""
    n_j, n_m = data[0][0].shape[0], data[1][0].shape[1]
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    if objective is not None:
        env.objective = objective
    state = env.set_initial_data(list(data[0]), list(data[1]), list(data[2]), list(data[3]))
    n_steps = env.number_of_ops
    t = 0
    while True:
        logps, vals = [], []
        for pol in policies:
            scores, gj, gm = pol.forward_logits(**args_of(state))
            logps.append(F.log_softmax(scores, dim=1))
            if values is not None:
                vals.append(pol.critic(torch.cat((gj, gm), dim=-1)).squeeze(-1))
        if diag is not None:
            record_diag(diag, t, logps[0], logps[1], args_of(state)["dynamic_pair_mask"])
        action = chooser(t, logps)
        state, reward, done = env.step(actions=action.cpu().numpy())
        if values is not None:
            values.append((vals[0].cpu().numpy(), np.asarray(reward, dtype=np.float64)))
        t += 1
        if np.all(done):
            break
    assert t == n_steps
    return env.current_makespan.copy(), env.total_carbon.copy()


def record_diag(diag, t, lm, lc, mask):
    valid = ~mask.reshape(mask.shape[0], -1)
    pm, pc = lm.exp(), lc.exp()
    top_m, top_c = pm.argmax(1), pc.argmax(1)
    mix = 0.5 * (pm + pc)

    def kl(p, lp):     # KL(p || mix) in bits over valid pairs
        lmix = torch.log(mix.clamp_min(1e-30))
        term = torch.where(valid & (p > 0), p * (lp - lmix), torch.zeros_like(p))
        return term.sum(1) / np.log(2)
    jsd = 0.5 * kl(pm, lm) + 0.5 * kl(pc, lc)

    def ent(p, lp):
        return -torch.where(valid & (p > 0), p * lp, torch.zeros_like(p)).sum(1) / np.log(2)
    nv = valid.sum(1)
    kk = torch.clamp(nv, max=3)
    overlap = []
    for e in range(pm.shape[0]):
        k = int(kk[e])
        a = set(torch.topk(pm[e], k).indices.tolist()); b = set(torch.topk(pc[e], k).indices.tolist())
        overlap.append(len(a & b) / k)
    for key, v in (("differ", (top_m != top_c).float()), ("jsd", jsd), ("H_m", ent(pm, lm)),
                   ("H_c", ent(pc, lc)), ("top3", torch.tensor(overlap, device=pm.device)),
                   ("n_valid", nv.float())):
        diag.setdefault(key, []).append(v.cpu().numpy())


# ------------------------------------------------------------------ choosers
def greedy(i):
    return lambda t, lp: lp[i].argmax(1)


def prob_mix(wc):
    return lambda t, lp: ((1 - wc) * lp[0].exp() + wc * lp[1].exp()).argmax(1)


def logit_mix(wc):
    def f(t, lp):
        if wc == 0.0:
            return lp[0].argmax(1)
        if wc == 1.0:
            return lp[1].argmax(1)
        invalid = torch.isinf(lp[0]) | torch.isinf(lp[1])
        z = (1 - wc) * torch.where(invalid, torch.zeros_like(lp[0]), lp[0]) \
            + wc * torch.where(invalid, torch.zeros_like(lp[1]), lp[1])
        z = torch.where(invalid, torch.full_like(z, float("-inf")), z)
        return F.log_softmax(z, dim=1).argmax(1)
    return f


def stoch(wc, rng):
    def f(t, lp):
        pick_c = torch.as_tensor(rng.random(lp[0].shape[0]) < wc, device=lp[0].device)
        return torch.where(pick_c, lp[1].argmax(1), lp[0].argmax(1))
    return f


def assign(bits, n_steps=50):
    seg = n_steps // K
    return lambda t, lp: lp[0 if bits[min(t // seg, K - 1)] == "M" else 1].argmax(1)


def run_seed(seed, log):
    pm, pc = load_policy(f"10x5+carbon+priority+m_s{seed}"), load_policy(f"10x5+carbon+priority+c_s{seed}")
    pool, vali = load_data("data_train_vali"), load_data("data_validation")
    R, D = {}, {}
    t0 = time.time()
    n = 0

    def go(key, data, chooser, pols=(pm, pc), diag=None):
        nonlocal n
        R[key] = np.stack(rollout(data, list(pols), chooser, diag=diag))
        n += 1

    for key, i in (("spec_m", 0), ("spec_c", 1)):
        d = {}
        go(key, pool, greedy(i), diag=d)
        D[key] = {k: np.stack(v, 1) for k, v in d.items()}
    for wc in WCS:
        d = {} if wc == 0.5 else None
        go(f"prob_{wc:.2f}", pool, prob_mix(wc), diag=d)
        if d is not None:
            D["prob_0.50"] = {k: np.stack(v, 1) for k, v in d.items()}
        go(f"logit_{wc:.2f}", pool, logit_mix(wc))
        for r in range(5):
            rng = np.random.default_rng([seed, int(round(wc * 100)), r])
            go(f"stoch_{wc:.2f}_r{r}", pool, stoch(wc, rng))
    for c in range(0, 1001, 100):
        name = f"10x5+carbon+priority+simplex_m{1000 - c:03d}c{c:03d}p000_s{seed}"
        if (CK / f"{name}.pth").exists():
            go(f"merge_{c:04d}", pool, greedy(0), pols=(load_policy(name),))
    for bits in ("".join(b) for b in itertools.product("MC", repeat=K)):
        go(f"assign_{bits}", pool, assign(bits))
        go(f"vali_assign_{bits}", vali, assign(bits))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"seed{seed}.npz", **R)
    np.savez_compressed(OUT / f"diag_seed{seed}.npz",
                        **{f"{src}__{k}": v for src, dd in D.items() for k, v in dd.items()})
    log(f"seed {seed}: {n} batched rollouts in {time.time() - t0:.0f}s")
    return n, time.time() - t0


def critic_check(seed, log):
    """Step 0: does each specialist's critic predict its own return-to-go?"""
    pool = load_data("data_train_vali")
    out = {}
    for tag, spec in (("m", ObjectiveSpec("m")), ("c", ObjectiveSpec("c", carbon_weight=0.01))):
        pol = load_policy(f"10x5+carbon+priority+{tag}_s{seed}")
        steps = []
        rollout(pool, [pol], greedy(0), values=steps, objective=spec)
        V = np.stack([v for v, _ in steps], 1)                     # [E, T]
        Rw = np.stack([r for _, r in steps], 1)
        G = np.cumsum(Rw[:, ::-1], axis=1)[:, ::-1]                # return-to-go (gamma = 1)
        pooled = float(np.corrcoef(V.ravel(), G.ravel())[0, 1])
        initial = float(np.corrcoef(V[:, 0], G[:, 0])[0, 1])
        within = float(np.nanmean([np.corrcoef(V[:, t], G[:, t])[0, 1] for t in range(V.shape[1] - 1)]))
        out[tag] = (pooled, initial, within, float(np.mean(np.abs(V - G))), float(np.mean(np.abs(G))))
        log(f"seed {seed} critic {tag}: r(pooled over states)={pooled:+.3f}  r(initial state vs episode "
            f"return, across instances)={initial:+.3f}  mean r within each step={within:+.3f}  "
            f"mean|V-G|={out[tag][3]:.3f} vs mean|G|={out[tag][4]:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333, 444])
    ap.add_argument("--critic-only", action="store_true")
    args = ap.parse_args(_ARGV)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    for s in args.seeds:
        critic_check(s, log)
    if args.critic_only:
        return
    tot_n, tot_t = 0, 0.0
    for s in args.seeds:
        n, t = run_seed(s, log)
        tot_n += n; tot_t += t
    log(f"TOTAL: {tot_n} batched rollouts ({tot_n} x up to 100 instances x 50 decisions) in {tot_t / 60:.1f} min "
        f"on one GPU")
    log("COMPLEMENTARITY_ROLLOUTS_DONE")


if __name__ == "__main__":
    main()
