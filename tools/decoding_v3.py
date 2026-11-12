"""Decoding strategies for the trained multi-objective DANIEL policies (docs/decoding_ceiling_protocol_v3.md, Part A).
Inference only: network weights, features, action mask and simulator are unchanged.

Decoders (all over DANIEL's legal operation-machine actions, every branch with its own simulator state):
  greedy                 argmax action
  sample K               best of K sampled schedules by the preference-weighted selection score
  beam B                 beam search on accumulated log-probability, identical partial schedules merged, final
                         ranking by the selection score
  lookahead k [m]        rollout action selection: top-k actions, each completed by the frozen policy (greedy, plus m
                         sampled continuations), the best by the selection score is executed

    python tools/decoding_v3.py run --split validation|evaluation [--quick]
    python tools/decoding_v3.py latency
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["run", "latency"])
    ap.add_argument("--split", default="validation", choices=["validation", "evaluation"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="comma-separated method names")
    ap.add_argument("--ckpts", default=None, help="comma-separated checkpoint keys")
    A = ap.parse_args()
    _dev = A.device
else:
    import os
    A = None
    _dev = os.environ.get("DECODING_DEVICE", "cpu")
sys.path.insert(0, str(REPO / "daniel")); sys.path.insert(0, str(REPO / "tools"))
_argv = sys.argv
sys.argv = [sys.argv[0], "--device", _dev, "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct",
            "--preference_dim", "3"]
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from params import configs                                          # noqa: E402
import benchmark_v3 as B                                            # noqa: E402
import mo_metrics as MM                                             # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums              # noqa: E402
sys.argv = _argv

SIZE = "10x5"
CKPTS = {"wm": ("World-model base mix", "checkpoints/plateau_diagnosis_v3/part_a/pd_wm_u2250.pth"),
         "ctrl": ("Random shared base mix", "checkpoints/plateau_diagnosis_v3/part_a/pd_ctrl_u3650.pth"),
         "fs": ("From scratch", "checkpoints/plateau_diagnosis_v3/part_a/pd_fs_u4550.pth")}
N_INST = {"validation": 18, "evaluation": 20}
EQUAL = np.array([[1 / 3, 1 / 3, 1 / 3]])
PREFS = np.vstack([MM.simplex_grid(0.1), EQUAL])
G = 66
METHODS = (["greedy", "sample16", "sample128", "beam2", "beam4", "beam8", "beam16", "beam32", "la_g2", "la_g3", "la_s2"])
OUT = REPO / "results" / "decoding_ceiling_v3"
STATIC = ("env_idxs", "env_job_idx", "multi_env_mch_diag", "op_idx")


# ------------------------------------------------------------------------------------------------ simulator copies

def select(env, idx):
    """A new environment whose row r is an independent copy of row idx[r] of `env` (repeats allowed)."""
    E = env.number_of_envs
    idx = np.asarray(idx, dtype=int)
    new = copy.copy(env)
    for k, v in vars(env).items():
        if k in STATIC:
            continue
        if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == E:
            setattr(new, k, v[idx])
    it = None
    for sk in ("state", "old_state"):
        st = copy.copy(getattr(env, sk))
        for k, v in vars(st).items():
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == E:
                if it is None:
                    it = torch.as_tensor(idx, device=v.device)
                setattr(st, k, v[it])
        setattr(new, sk, st)
    new.number_of_envs = len(idx)
    new.set_static_properties()
    return new


def make_env(data, idx, prefs):
    env = FJSPEnvForSameOpNums(*B.SIZES[data.size])
    env.set_initial_data(**data.env_kwargs(idx, "mct"))
    env.set_preference(prefs)
    return env


def forward(net, env):
    s = env.state
    with torch.no_grad():
        pi, _ = net(fea_j=s.fea_j_tensor, op_mask=s.op_mask_tensor, candidate=s.candidate_tensor, fea_m=s.fea_m_tensor,
                    mch_mask=s.mch_mask_tensor, comp_idx=s.comp_idx_tensor, dynamic_pair_mask=s.dynamic_pair_mask_tensor,
                    fea_pairs=s.fea_pairs_tensor, pref=s.pref_tensor)
    return pi


def result(env):
    """objectives [E, 3], machine [E, N], start [E, N] of finished environments."""
    machine = env.sched_machine
    pt = env.true_op_pt[env.env_idxs[:, None], np.arange(env.number_of_ops)[None, :], machine]
    start = env.true_op_ct - pt
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
    return np.stack([env.current_makespan, env.total_carbon, tard], 1), machine.copy(), start


def step(env, a):
    """env.step that also records the machine of every operation (kept row-wise, so copies keep it)."""
    if not hasattr(env, "sched_machine") or env.sched_machine.shape[0] != env.number_of_envs:
        env.sched_machine = np.full((env.number_of_envs, env.number_of_ops), -1)
    op = env.candidate[env.env_idxs, a // env.number_of_machines]
    env.sched_machine[env.env_idxs, op] = a % env.number_of_machines
    return env.step(a)


def score(data, idx, obj, w):
    idx = np.asarray(idx)
    mk = np.asarray(data.mk_ref)[idx]; cb = np.asarray(data.cb_ref)[idx]
    return w[:, 0] * obj[:, 0] / mk + w[:, 1] * obj[:, 1] / cb + w[:, 2] * obj[:, 2] / data.t_ref


class Cost:
    def __init__(self, n):
        self.evals = np.zeros(n); self.complete = np.zeros(n); self.seconds = 0.0

    def as_dict(self):
        return {"evals": self.evals.tolist(), "complete": self.complete.tolist(), "seconds": self.seconds}


# ------------------------------------------------------------------------------------------------ decoders

def rollout_to_end(net, env, greedy_rows=None, gen=None, owner=None, cost=None):
    """Completes every row: greedy where greedy_rows (default all), sampled elsewhere. Counts state evaluations per
    owner (item index of each row)."""
    E = env.number_of_envs
    greedy_rows = np.ones(E, bool) if greedy_rows is None else greedy_rows
    gt = torch.as_tensor(greedy_rows, device=configs.device)
    while not np.all(env.done()):
        pi = forward(net, env)
        if cost is not None:
            np.add.at(cost.evals, owner, 1)
        a = pi.argmax(1)
        if not greedy_rows.all():
            s = torch.multinomial(pi[~gt], 1, generator=gen).squeeze(1)
            a = a.clone(); a[~gt] = s
        step(env, a.cpu().numpy())
    return result(env)


def greedy(net, data, idx, prefs, cost):
    env = make_env(data, idx, prefs)
    o, m, s = rollout_to_end(net, env, owner=np.arange(len(idx)), cost=cost)
    cost.complete += 1
    return o, m, s


def sample_best(net, data, idx, prefs, K, seed, cost, keep=(2, 4, 8, 16)):
    """Best of K samples per row, drawn per instance exactly as evaluate_mo_v3 (one instance per batch, all its
    preferences, K consecutive samples per preference). Returns {k: (obj, machine, start)} for k in keep."""
    gen = torch.Generator(device=configs.device); gen.manual_seed(seed)
    n = len(idx)
    out = {k: [np.zeros((n, 3)), np.zeros((n, 50), int), np.zeros((n, 50))] for k in keep}
    inst = sorted(set(idx))
    rows_of = {i: [r for r in range(n) if idx[r] == i] for i in inst}
    for i in inst:
        rows = rows_of[i]
        w = np.repeat(prefs[rows], K, axis=0)
        env = make_env(data, [i] * len(w), w)
        o, m, s = rollout_to_end(net, env, greedy_rows=np.zeros(len(w), bool), gen=gen,
                                 owner=np.repeat(rows, K), cost=cost)
        sc = score(data, [i] * len(w), o, w).reshape(len(rows), K)
        for k in keep:
            pick = sc[:, :k].argmin(1) + np.arange(len(rows)) * K
            out[k][0][rows], out[k][1][rows], out[k][2][rows] = o[pick], m[pick], s[pick]
    cost.complete += K
    return out


_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def zobrist(op, m, st):
    """64-bit hash of a placement (operation, machine, integer start), order-free when XOR-combined."""
    x = (op.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + m.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
         + np.rint(st).astype(np.int64).astype(np.uint64) * np.uint64(0x94D049BB133111EB) + np.uint64(1))
    x ^= x >> np.uint64(30); x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27); x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def beam(net, data, idx, prefs, width, cost, stats=None):
    n = len(idx)
    env = make_env(data, idx, prefs)
    item = np.arange(n); cum = np.zeros(n); key = np.zeros(n, np.uint64)
    M = env.number_of_machines
    merged = 0
    with np.errstate(over="ignore"):
        for t in range(env.number_of_ops):
            pi = forward(net, env).double()
            np.add.at(cost.evals, item, 1)
            logp = torch.log(pi).cpu().numpy()
            legal = pi.cpu().numpy() > 0
            e_idx, a_idx = np.nonzero(legal)
            sc = cum[e_idx] + logp[e_idx, a_idx]
            job, m = a_idx // M, a_idx % M
            op = env.candidate[e_idx, job]
            st = np.maximum(env.true_candidate_free_time[e_idx, job], env.true_mch_free_time[e_idx, m])
            ck = key[e_idx] ^ zobrist(op, m, st)
            it = item[e_idx]
            order = np.lexsort((a_idx, e_idx, -sc, it))                      # per item: best first, ties by index
            e_idx, a_idx, sc, ck, it = e_idx[order], a_idx[order], sc[order], ck[order], it[order]
            # merge identical partial schedules (same item, same placement set): keep the first (best)
            pair = np.stack([it.astype(np.uint64), ck], 1)
            _, first = np.unique(pair, axis=0, return_index=True)
            keepm = np.zeros(len(it), bool); keepm[first] = True
            merged += int((~keepm).sum())
            e_idx, a_idx, sc, ck, it = e_idx[keepm], a_idx[keepm], sc[keepm], ck[keepm], it[keepm]
            order = np.lexsort((a_idx, e_idx, -sc, it))
            e_idx, a_idx, sc, ck, it = e_idx[order], a_idx[order], sc[order], ck[order], it[order]
            start_of = np.r_[0, np.flatnonzero(np.diff(it)) + 1]
            rank = np.arange(len(it)) - np.repeat(start_of, np.diff(np.r_[start_of, len(it)]))
            sel = rank < width
            env = select(env, e_idx[sel])
            step(env, a_idx[sel])
            item, cum, key = it[sel], sc[sel], ck[sel]
    o, mch, s = result(env)
    np.add.at(cost.complete, item, 1)
    w = prefs[item]
    sc = score(data, np.asarray(idx)[item], o, w)
    best = {}
    for r in np.lexsort((-cum, sc)):                                         # lowest score; ties: higher log-prob
        best.setdefault(item[r], r)
    pick = np.array([best[i] for i in range(n)])
    if stats is not None:
        stats["merged"] = merged
    return o[pick], mch[pick], s[pick]


def lookahead(net, data, idx, prefs, k, m_samples, seed, cost):
    n = len(idx)
    idx = np.asarray(idx)
    gen = torch.Generator(device=configs.device); gen.manual_seed(seed)
    env = make_env(data, idx, prefs)
    env.sched_machine = np.full((n, env.number_of_ops), -1)
    # incumbent: the greedy continuation of the current state
    inc = rollout_to_end(net, select(env, np.arange(n)), owner=np.arange(n), cost=cost)
    cost.complete += 1
    inc_sc = score(data, idx, inc[0], prefs)
    M = env.number_of_machines
    for t in range(env.number_of_ops):
        pi = forward(net, env).double()
        cost.evals += 1
        p = pi.cpu().numpy()
        legal = p > 0
        nleg = legal.sum(1)
        order = np.argsort(-np.where(legal, p, -1.0), axis=1, kind="stable")[:, :k]
        keff = np.minimum(nleg, k)
        dec = np.flatnonzero(nleg > 1)
        chosen = order[:, 0].copy()
        if len(dec):
            rows_item, rows_cand, rows_greedy = [], [], []
            for c in range(k):
                its = dec[keff[dec] > c]
                if c > 0:                                                      # fresh greedy continuation
                    rows_item.append(its); rows_cand.append(np.full(len(its), c)); rows_greedy.append(np.ones(len(its), bool))
                for _ in range(m_samples):
                    rows_item.append(its); rows_cand.append(np.full(len(its), c)); rows_greedy.append(np.zeros(len(its), bool))
            ri, rc, rg = np.concatenate(rows_item), np.concatenate(rows_cand), np.concatenate(rows_greedy)
            if len(ri):
                R = select(env, ri)
                R.sched_machine = env.sched_machine[ri].copy()
                step(R, order[ri, rc])
                o, mm, ss = rollout_to_end(net, R, greedy_rows=rg, gen=gen, owner=ri, cost=cost)
                np.add.at(cost.complete, ri, 1)
                sc = score(data, idx[ri], o, prefs[ri])
            else:
                sc = np.zeros(0)
            val = np.full((n, k), np.inf)
            val[dec, 0] = inc_sc[dec]
            np.minimum.at(val, (ri, rc), sc)
            best_c = np.argmin(val[dec], 1)                                    # ties: the higher-probability action
            chosen[dec] = order[dec, best_c]
            # the chosen candidate's greedy continuation becomes the incumbent
            for d, c in zip(dec, best_c):
                if c == 0:
                    continue
                r = np.flatnonzero((ri == d) & (rc == c) & rg)[0]
                inc = (_set(inc[0], d, o[r]), _set(inc[1], d, mm[r]), _set(inc[2], d, ss[r]))
                inc_sc[d] = sc[r]
        step(env, chosen)
    o, m, s = result(env)
    assert np.allclose(score(data, idx, o, prefs), inc_sc), "lookahead: executed schedule differs from its incumbent"
    return o, m, s


def _set(arr, i, v):
    arr[i] = v
    return arr


# ------------------------------------------------------------------------------------------------ running

def load_net(path):
    from common_utils import load_checkpoint_state_dict
    from checkpointing import unwrap_checkpoint
    from feature_schemas import SCHEMA_V3_MO
    from model.main_model import DANIEL
    _, meta = unwrap_checkpoint(torch.load(path, map_location=configs.device, weights_only=False))
    if meta.get("config", {}).get("merge_mix"):
        from mixture import MixturePolicy
        net = MixturePolicy([DANIEL(configs).to(configs.device) for _ in range(3)])
    else:
        net = DANIEL(configs).to(configs.device)
    net.load_state_dict(load_checkpoint_state_dict(path, map_location=configs.device, expected_schema=SCHEMA_V3_MO))
    return net.eval()


def run_method(net, data, idx, prefs, name, chunk=None):
    """Returns {method name: (obj, machine, start, cost dict)} (sampling returns several K at once)."""
    n = len(idx)
    out = {}
    if name == "greedy":
        c = Cost(n); t0 = time.time(); r = greedy(net, data, idx, prefs, c); c.seconds = time.time() - t0
        out["greedy"] = (*r, c.as_dict())
    elif name in ("sample16", "sample128"):
        K, seed, keep = (16, 50, (2, 4, 8, 16)) if name == "sample16" else (128, 51, (32, 64, 128))
        c = Cost(n); t0 = time.time(); r = sample_best(net, data, idx, prefs, K, seed, c, keep); c.seconds = time.time() - t0
        for k in keep:
            d = c.as_dict(); d["evals"] = (np.asarray(d["evals"]) * k / K).tolist(); d["complete"] = [k] * n
            d["seconds"] = c.seconds * k / K
            out[f"sample{k}"] = (*r[k], d)
    elif name.startswith("beam"):
        width = int(name[4:])
        c = Cost(n); st = {}; t0 = time.time()
        parts = []
        step_n = chunk or n
        for c0 in range(0, n, step_n):
            sl = np.arange(c0, min(n, c0 + step_n))
            cc = Cost(len(sl))
            parts.append(beam(net, data, list(np.asarray(idx)[sl]), prefs[sl], width, cc, st))
            c.evals[sl] = cc.evals; c.complete[sl] = cc.complete
        c.seconds = time.time() - t0
        r = tuple(np.concatenate([p[j] for p in parts]) for j in range(3))
        d = c.as_dict(); d["merged"] = st.get("merged")
        out[name] = (*r, d)
    elif name.startswith("la_"):
        k = int(name[-1]); ms = 3 if name.startswith("la_s") else 0
        c = Cost(n); t0 = time.time()
        r = lookahead(net, data, idx, prefs, k, ms, 52, c); c.seconds = time.time() - t0
        out[name] = (*r, c.as_dict())
    return out


def run(split, quick, only=None, ckpts=None):
    data = B.load_split(split, SIZE)
    I = 2 if quick else N_INST[split]
    idx = [i for i in range(I) for _ in range(len(PREFS))]
    prefs = np.tile(PREFS, (I, 1))
    out_dir = OUT / ("quick" if quick else "") / split
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = only.split(",") if only else METHODS
    for key in (ckpts.split(",") if ckpts else CKPTS):
        lab, path = CKPTS[key]
        net = load_net(REPO / path)
        for name in methods:
            f = out_dir / f"{key}__{name}.npz"
            if name.startswith("sample"):
                f = out_dir / f"{key}__{name}.done"
            if f.exists():
                continue
            t0 = time.time()
            res = run_method(net, data, idx, prefs, name, chunk=max(1, 12000 // int(name[4:])) if name.startswith("beam") else None)
            for mname, (o, m, s, c) in res.items():
                for r, i in enumerate(idx):                                   # independent re-score and feasibility
                    ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i], m[r], s[r])
                    if max(abs(ind["makespan"] - o[r, 0]), abs(ind["carbon"] - o[r, 1]), abs(ind["tardiness"] - o[r, 2])) > 1e-6:
                        raise AssertionError(f"{key} {mname}: simulator and independent scorer disagree")
                np.savez_compressed(out_dir / f"{key}__{mname}.npz", obj=o.reshape(I, len(PREFS), 3),
                                    machine=m.reshape(I, len(PREFS), -1), start=s.reshape(I, len(PREFS), -1),
                                    evals=np.asarray(c["evals"]).reshape(I, len(PREFS)),
                                    complete=np.asarray(c["complete"]).reshape(I, len(PREFS)),
                                    seconds=c["seconds"], merged=-1 if c.get("merged") is None else c["merged"])
            if name.startswith("sample"):
                f.write_text("ok\n")
            print(f"[{time.strftime('%H:%M:%S')}] {split} {key} {name}: {time.time() - t0:.0f} s", flush=True)
        del net
        torch.cuda.empty_cache()


def latency(ckpts=None):
    """Single-pair latency (one instance-preference pair at a time) on 20 fixed evaluation pairs."""
    data = B.load_split("evaluation", SIZE)
    pairs = [(i, (7 * i) % G) for i in range(20)]
    out = {}
    for key in (ckpts.split(",") if ckpts else CKPTS):
        net = load_net(REPO / CKPTS[key][1])
        for name in ["greedy", "sample16", "sample128", "beam2", "beam4", "beam8", "beam16", "beam32", "la_g2", "la_g3", "la_s2"]:
            ts = []
            for i, g in pairs:
                w = PREFS[g][None]
                torch.cuda.synchronize()
                t0 = time.time()
                if name.startswith("sample"):
                    K, seed = (16, 50) if name == "sample16" else (128, 51)
                    gen = torch.Generator(device=configs.device); gen.manual_seed(seed)
                    env = make_env(data, [i] * K, np.repeat(w, K, 0))
                    o, _, _ = rollout_to_end(net, env, greedy_rows=np.zeros(K, bool), gen=gen)
                    score(data, [i] * K, o, np.repeat(w, K, 0)).argmin()
                else:
                    run_method(net, data, [i], w, name)
                torch.cuda.synchronize()
                ts.append(time.time() - t0)
            out.setdefault(key, {})[name] = {"mean": float(np.mean(ts)), "max": float(np.max(ts)), "pairs": pairs}
            print(f"latency {key} {name}: mean {np.mean(ts):.3f} s, max {np.max(ts):.3f} s", flush=True)
        del net
    (OUT / "latency.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if A.action == "run":
        run(A.split, A.quick, A.only, A.ckpts)
    else:
        latency(A.ckpts)
