"""Ceiling check, Part B: best-of-N stochastic decoding of C1@1000 (results/ceiling_check.md). EVALUATION ONLY.

For each seed, C1@1000 (= Step 4 makespan-specialist arm after 1000 updates, s4t_spec_m_k125s060_s<seed>@u1000)
decodes every HELD-OUT instance 100 times by sampling from the policy's own softmax (no temperature change;
torch.multinomial with a fixed per-seed generator). The complete schedule of every sample (machine and start time
of every operation) is stored and re-scored with the independent Step 0 scorer; best-of-N uses the first N samples.

Rollouts batch only instances sharing the same maximum processing time (the environment normalises by the batch
maximum), so greedy decoding reproduces the canonical evaluator exactly (checked in Step 0).

    python tools/ceiling_sampling.py [seeds...]      # writes results/ceiling_check/raw/sampling_s<seed>.npz
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import complementarity as P                           # noqa: E402  (sets the canonical config; loads policies)
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums          # noqa: E402
import ceiling_scorer as SC                                     # noqa: E402

RAW = REPO / "results/ceiling_check/raw"
TAG = "k125s060"
N_SAMPLES = 100
CHUNK = 5000
GEN_SEED = 20260928


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def c1_name(seed):
    return f"10x5+carbon+priority+s4t_spec_m_{TAG}_s{seed}@u1000"


@torch.no_grad()
def _rollout(data, dues, pol, gen=None):
    env = FJSPEnvForSameOpNums(n_j=data[0][0].shape[0], n_m=data[1][0].shape[1])
    state = env.set_initial_data(list(data[0]), list(data[1]), list(data[2]), list(data[3]), due_date_list=list(dues))
    E, n_m, rows = env.number_of_envs, env.number_of_machines, np.arange(env.number_of_envs)
    machine = np.full((E, env.number_of_ops), -1, dtype=np.int64)
    while True:
        scores = pol.forward_logits(**P.args_of(state))[0]
        if gen is None:
            a = F.log_softmax(scores, dim=1).argmax(1)                     # greedy, as the canonical evaluator
        else:
            a = torch.multinomial(F.softmax(scores, dim=1), 1, generator=gen).squeeze(1)
        a = a.cpu().numpy()
        op = env.candidate[rows, a // n_m]
        assert (machine[rows, op] == -1).all()
        machine[rows, op] = a % n_m
        state, _, done = env.step(actions=a)
        if np.all(done):
            break
    assert (machine >= 0).all()
    comp = env.true_op_ct.copy()
    start = comp - env.true_op_pt[rows[:, None], np.arange(env.number_of_ops)[None, :], machine]
    completion = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, completion - np.asarray(dues)).sum(1)
    return dict(mk=env.current_makespan.copy(), cb=env.total_carbon.copy(), tard=tard, machine=machine, start=start)


def run(pol, data, due, inst, gen=None):
    """inst: pool indices (repeats allowed). Grouped by maximum processing time; results in input order."""
    ub = np.array([float(data[1][i][data[1][i] > 0].max()) for i in inst])
    out = None
    for u in np.unique(ub):
        rows = np.flatnonzero(ub == u)
        for c0 in range(0, len(rows), CHUNK):
            r = rows[c0:c0 + CHUNK]
            sub = tuple([d[i] for i in inst[r]] for d in data)
            res = _rollout(sub, [due[i] for i in inst[r]], pol, gen)
            if out is None:
                out = {k: np.zeros((len(inst),) + v.shape[1:], dtype=v.dtype) for k, v in res.items()}
            for k, v in res.items():
                out[k][r] = v
    return out


_CACHE = {}


def _data():
    if not _CACHE:
        _CACHE["data"] = P.load_data("data_train_vali")
        _CACHE["due"] = SC.due_dates()
    return _CACHE["data"], _CACHE["due"]


def greedy_c1(seed):
    data, due = _data()
    return run(P.load_policy(c1_name(seed)), data, due, SC.HELD)


def sample_seed(seed):
    f = RAW / f"sampling_s{seed}.npz"
    if f.exists():
        log(f"seed {seed}: complete, skipping"); return
    t0 = time.time()
    data, due = _data()
    pol = P.load_policy(c1_name(seed))
    gen = torch.Generator(device=P.DEV); gen.manual_seed(GEN_SEED + seed)
    inst = np.repeat(SC.HELD, N_SAMPLES)                                     # row k*100 + r = sample r of HELD[k]
    o = run(pol, data, due, inst, gen)
    torch.cuda.synchronize(); gpu = time.time() - t0
    # re-score every sampled schedule with the independent scorer and cross-check the environment's numbers
    insts, dues, ref = SC.instances(), SC.due_dates(), SC.reference(seed)
    mine = np.array([SC.raw_objectives(insts[i], dues[i], o["machine"][k], o["start"][k]) for k, i in enumerate(inst)])
    env3 = np.stack([o["mk"], o["cb"], o["tard"]], 1)
    d = float(np.abs(mine - env3).max())
    if d > 1e-6:
        raise SystemExit(f"seed {seed}: scorer and environment disagree on sampled schedules (max diff {d})")
    sc = np.array([SC.score(seed, i, *mine[k], ref=ref) for k, i in enumerate(inst)])
    n = len(SC.HELD)
    np.savez_compressed(f, held=SC.HELD, mk=mine[:, 0].reshape(n, N_SAMPLES), cb=mine[:, 1].reshape(n, N_SAMPLES),
                        tard=mine[:, 2].reshape(n, N_SAMPLES), score=sc.reshape(n, N_SAMPLES),
                        machine=o["machine"].reshape(n, N_SAMPLES, -1).astype(np.int8),
                        start=o["start"].reshape(n, N_SAMPLES, -1).astype(np.int32),
                        gpu_seconds=gpu, max_abs_diff_scorer_vs_env=d)
    b = sc.reshape(n, N_SAMPLES)
    log(f"seed {seed}: {N_SAMPLES} samples x {n} instances in {gpu:.1f} s; scorer vs env max diff {d:.2g}; "
        f"mean sample score {b.mean():.4f}, best-of-10/30/100 {b[:, :10].min(1).mean():.4f}/"
        f"{b[:, :30].min(1).mean():.4f}/{b.min(1).mean():.4f}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [int(a) for a in P._ARGV if a.isdigit()] or SC.SEEDS
    for s in todo:
        sample_seed(s)
    log("SAMPLING_DONE")


if __name__ == "__main__":
    main()
