"""Soft-weight segment oracle, arms A1-A6 (pre-registered in results/segment_blend_oracle.md, commit ec8e77e).

Logit mixture of the frozen makespan (M) and carbon (C) specialists with a per-segment weight. Internally the
weight is carried as the pilot's carbon weight c = 1 - alpha, and the mixture is computed with the pilot's exact
arithmetic ((1 - c) * log p_M + c * log p_C over valid pairs, invalid = -inf, log_softmax, argmax; c = 0 or 1 is
the pure specialist), so fixed-alpha runs reproduce the pilot's logit-mixture results bit for bit (checked).

Every rollout row is (instance, 5-segment schedule). Rows are batched ONLY with rows whose instance has the same
maximum processing time (the environment scales by the batch maximum), in chunks of <= 9000 environments.

Outputs results/segment_oracle/raw/seed<s>.npz per seed (resume skips finished seeds) and a6.npz.

    python tools/segment_blend_oracle.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import complementarity as P                       # noqa: E402  (pilot helpers; sets configs as the pilot did)
import torch                                      # noqa: E402
import torch.nn.functional as F                   # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums      # noqa: E402
from data_utils import sorted_instance_files                # noqa: E402

OUT = REPO / "results/segment_oracle"
RAW = OUT / "raw"
PILOT = REPO / "results/complementarity/raw"
SEEDS = [111, 222, 333, 444]
WC = np.array([round(0.05 * i, 2) for i in range(21)])      # preferences w_c (pilot grid)
G21 = np.array([round(0.05 * i, 2) for i in range(21)])      # carbon-weight grid c = 1 - alpha
G101 = np.array([round(0.01 * i, 2) for i in range(101)])
K, SEG, CHUNK = 5, 10, 9000
SPLIT_SEED = 20260926
# COST RULE (pre-registered): upper-bound estimate 8.4 GPU-h > 6 h, so A3 restarts 2 -> 1 (the alpha = w_m
# restart is kept, the random restart dropped) and passes 4 -> 3; A5 uses the same reduced search.
PASSES, RESTART_W = 3, True
GPU_SECONDS = [0.0]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------ rollout engine
@torch.no_grad()
def _rollout_rows(data, pols, cw, feat=None):
    """data: (jl, pt, prio, carbon) lists for one batch; cw: [E, 5] carbon weights per segment."""
    env = FJSPEnvForSameOpNums(n_j=data[0][0].shape[0], n_m=data[1][0].shape[1])
    state = env.set_initial_data(list(data[0]), list(data[1]), list(data[2]), list(data[3]))
    E = len(data[0])
    cw_t = torch.as_tensor(cw, dtype=torch.float64)
    cm = (1.0 - cw_t).to(torch.float32).to(P.DEV)       # f32(1 - c) from double, as the pilot's python scalar
    cc = cw_t.to(torch.float32).to(P.DEV)
    zero_c, one_c = (cw_t == 0.0).to(P.DEV), (cw_t == 1.0).to(P.DEV)
    lb0 = env.op_ct_lb.max(1) * env.pt_upper_bound
    total_work = env.op_mean_pt.sum(1)
    elig = env.process_relation
    cmin = np.where(elig, env.op_carbon, np.inf).min(2)                     # [E, N] min carbon per op
    t = 0
    while True:
        lp = [F.log_softmax(pol.forward_logits(**P.args_of(state))[0], dim=1) for pol in pols]
        k = min(t // SEG, K - 1)
        if feat is not None:
            P.record_diag(feat.setdefault("_diag", {}), t, lp[0], lp[1], P.args_of(state)["dynamic_pair_mask"])
            if t % SEG == 0:
                sched = env.op_scheduled_flag.astype(bool)
                rem = (env.op_mean_pt * (~sched)).sum(1) / total_work
                mk = np.maximum(env.current_makespan, 0.0) / lb0
                cmin_done = np.where(sched, cmin, 0.0).sum(1)
                cb = np.where(cmin_done > 0, env.total_carbon / np.maximum(cmin_done, 1e-9), 1.0)
                feat.setdefault("rem_work", []).append(rem)
                feat.setdefault("mk_ratio", []).append(mk)
                feat.setdefault("cb_ratio", []).append(cb)
        invalid = torch.isinf(lp[0]) | torch.isinf(lp[1])
        z = cm[:, k:k + 1] * torch.where(invalid, torch.zeros_like(lp[0]), lp[0]) \
            + cc[:, k:k + 1] * torch.where(invalid, torch.zeros_like(lp[1]), lp[1])
        z = torch.where(invalid, torch.full_like(z, float("-inf")), z)
        a = F.log_softmax(z, dim=1).argmax(1)
        a = torch.where(zero_c[:, k], lp[0].argmax(1), torch.where(one_c[:, k], lp[1].argmax(1), a))
        state, _, done = env.step(actions=a.cpu().numpy())
        t += 1
        if np.all(done):
            break
    return env.current_makespan.copy(), env.total_carbon.copy()


def run_rows(pols, data, inst, cw, feat=False):
    """inst: [R] instance indices; cw: [R, 5] carbon weights. Returns makespan, carbon [R] (+ features)."""
    t0 = time.time()
    ub = np.array([float(data[1][i][data[1][i] > 0].max()) for i in inst])
    mk, cb = np.zeros(len(inst)), np.zeros(len(inst))
    F_ = {} if feat else None
    for u in np.unique(ub):
        rows = np.flatnonzero(ub == u)
        for c0 in range(0, len(rows), CHUNK):
            r = rows[c0:c0 + CHUNK]
            sub = tuple([d[i] for i in inst[r]] for d in data)
            f = {} if feat else None
            m, c = _rollout_rows(sub, pols, cw[r], feat=f)
            mk[r], cb[r] = m, c
            if feat:
                for key, v in f.items():
                    if key == "_diag":
                        for dk, dv in v.items():
                            arr = F_.setdefault(dk, np.full((len(inst), len(dv)), np.nan))
                            arr[r] = np.stack(dv, 1)
                    else:
                        arr = F_.setdefault(key, np.full((len(inst), len(v)), np.nan))
                        arr[r] = np.stack(v, 1)
    torch.cuda.synchronize()
    GPU_SECONDS[0] += time.time() - t0
    return (mk, cb, F_) if feat else (mk, cb)


# ------------------------------------------------------------------------------------------ scoring
def norm_bounds():
    X = np.stack([np.load(PILOT / f"seed{s}.npz")[k] for s in SEEDS for k in ("spec_m", "spec_c")])
    lo, hi = X.min(0), X.max(0)
    return lo, np.where(hi - lo > 0, hi - lo, 1.0)


LO, RG = norm_bounds()                                         # [2, 100], the pilot's pool normalisation


def score(inst, w, mk, cb):
    nm = (mk - LO[0][inst]) / RG[0][inst]
    nc = (cb - LO[1][inst]) / RG[1][inst]
    return (1 - w) * nm + w * nc


def make_split(n_inst, data):
    ub = np.array([float(pt[pt > 0].max()) for pt in data[1]])
    rng = np.random.default_rng(SPLIT_SEED)
    fit = []
    for u in np.unique(ub):
        g = np.flatnonzero(ub == u); rng.shuffle(g)
        fit += list(g[: (len(g) + 1) // 2])
    fit = np.sort(np.array(fit)); held = np.setdiff1d(np.arange(n_inst), fit)
    return fit, held


# ------------------------------------------------------------------------------------------ coordinate descent
def coordinate_descent(n_units, start, eval_fn, tag):
    """start: [U, 5] indices into G21. eval_fn(unit_ids, sched_idx[n, 5]) -> scores[n]. Minimises score.
    Returns best schedule indices, best scores, evaluations used per unit."""
    cur = start.copy()
    cur_s = eval_fn(np.arange(n_units), cur)
    evals = np.ones(n_units, dtype=int)
    active = np.ones(n_units, dtype=bool)
    for p in range(PASSES):
        improved = np.zeros(n_units, dtype=bool)
        for k in range(K):
            U = np.flatnonzero(active)
            if len(U) == 0:
                break
            cand = np.repeat(cur[U], len(G21), axis=0)
            cand[:, k] = np.tile(np.arange(len(G21)), len(U))
            s = eval_fn(np.repeat(U, len(G21)), cand).reshape(len(U), len(G21))
            evals[U] += len(G21)
            b = s.argmin(1); bs = s[np.arange(len(U)), b]
            better = bs < cur_s[U] - 1e-12
            cur[U[better], k] = b[better]; cur_s[U[better]] = bs[better]; improved[U[better]] = True
            log(f"      {tag}: pass {p + 1} segment {k + 1}: {better.sum()}/{len(U)} improved, "
                f"GPU {GPU_SECONDS[0] / 60:.1f} min")      # per-sweep heartbeat for the 7-minute hang guard
        log(f"    {tag}: pass {p + 1} done, {improved.sum()}/{n_units} units improved "
            f"(active {active.sum()}), GPU {GPU_SECONDS[0] / 60:.1f} min")
        active &= improved
        if not active.any():
            break
    return cur, cur_s, evals


def best_of_starts(n_units, starts, eval_fn, tag):
    best, best_s, ev = None, None, np.zeros(n_units, dtype=int)
    for i, st in enumerate(starts):
        c, s, e = coordinate_descent(n_units, st, eval_fn, f"{tag} start {i + 1}/{len(starts)}")
        ev += e
        if best is None:
            best, best_s = c, s
        else:
            m = s < best_s - 1e-12
            best[m], best_s[m] = c[m], s[m]
    return best, best_s, ev


# ------------------------------------------------------------------------------------------ per seed
def run_seed(seed, data, fit, held):
    f = RAW / f"seed{seed}.npz"
    if f.exists():
        log(f"seed {seed}: complete, skipping"); return
    pols = [P.load_policy(f"10x5+carbon+priority+m_s{seed}"), P.load_policy(f"10x5+carbon+priority+c_s{seed}")]
    n = len(data[0]); nw = len(WC)
    # ---- A1 / A2: fixed alpha over G101 (preference-independent rollouts) -------------------------------
    inst = np.tile(np.arange(n), len(G101)); cw = np.repeat(np.repeat(G101, n)[:, None], K, 1)
    mk, cb = run_rows(pols, data, inst, cw)
    FMK, FCB = mk.reshape(len(G101), n), cb.reshape(len(G101), n)
    pilot = np.load(PILOT / f"seed{seed}.npz")
    g21_rows = [int(round(c * 100)) for c in G21]
    bad = sum(int(not (np.array_equal(FMK[r], pilot[f"logit_{c:.2f}"][0]) and
                       np.array_equal(FCB[r], pilot[f"logit_{c:.2f}"][1]))) for r, c in zip(g21_rows, G21))
    if bad:
        raise SystemExit(f"REPRODUCTION FAILED at seed {seed}: {bad}/21 fixed-alpha runs differ from the pilot")
    log(f"seed {seed}: fixed alpha x 101 done; all 21 G21 runs reproduce the pilot's logit mixture exactly")
    FS = np.stack([score(np.arange(n), w, FMK, FCB) for w in WC])            # [21 w, 101 c, n]
    a1_idx = FS[:, g21_rows, :].argmin(1)                                     # [21, n] index into G21
    # ---- A3: segment oracle per (instance, w) ------------------------------------------------------------
    pi = np.tile(np.arange(n), nw); pw = np.repeat(np.arange(nw), n)          # unit u -> (instance, w index)

    def eval_a3(units, sched):
        m, c = run_rows(pols, data, pi[units], G21[sched])
        return score(pi[units], WC[pw[units]], m, c)
    starts = [np.repeat(a1_idx.reshape(-1)[:, None], K, 1)]                   # a1_idx[w, i] -> unit order w*n+i
    if RESTART_W:
        starts.append(np.repeat(np.array([int(round(w * 20)) for w in WC[pw]])[:, None], K, 1))
    a3, a3_s, a3_ev = best_of_starts(n * nw, starts, eval_a3, f"seed {seed} A3")
    mk3, cb3, feat = run_rows(pols, data, pi, G21[a3], feat=True)
    assert np.allclose(score(pi, WC[pw], mk3, cb3), a3_s), "A3 re-evaluation mismatch (non-determinism)"
    # ---- A4 / A5 on FIT, evaluated on HELD-OUT ----------------------------------------------------------
    fit_mean = FS[:, g21_rows, :][:, :, fit].mean(2)                          # [21 w, 21 c]
    a4_idx = fit_mean.argmin(1)

    def eval_a5(units, sched):
        u = np.repeat(units, len(fit)); ii = np.tile(fit, len(units)); sc = np.repeat(sched, len(fit), axis=0)
        m, c = run_rows(pols, data, ii, G21[sc])
        return score(ii, WC[u], m, c).reshape(len(units), len(fit)).mean(1)
    starts5 = [np.repeat(a4_idx[:, None], K, 1)]
    if RESTART_W:
        starts5.append(np.repeat(np.array([int(round(w * 20)) for w in WC])[:, None], K, 1))
    a5, a5_s, a5_ev = best_of_starts(nw, starts5, eval_a5, f"seed {seed} A5")
    hi = np.tile(held, nw); hw = np.repeat(np.arange(nw), len(held))
    mk5, cb5 = run_rows(pols, data, hi, G21[a5[hw]])
    np.savez_compressed(f, FMK=FMK, FCB=FCB, a1_idx=a1_idx, a3_idx=a3.reshape(nw, n, K), a3_score=a3_s.reshape(nw, n),
                        a3_evals=a3_ev.reshape(nw, n), a3_mk=mk3.reshape(nw, n), a3_cb=cb3.reshape(nw, n),
                        a4_idx=a4_idx, a5_idx=a5, a5_fit_score=a5_s, a5_evals=a5_ev,
                        a5_held_mk=mk5.reshape(nw, len(held)), a5_held_cb=cb5.reshape(nw, len(held)),
                        **{f"feat_{k}": v.reshape(nw, n, -1) for k, v in feat.items()})
    log(f"seed {seed}: saved; cumulative GPU {GPU_SECONDS[0] / 60:.1f} min")


def run_a6(data, held):
    f = RAW / "a6.npz"
    if f.exists():
        log("A6 complete, skipping"); return
    sched = {s: np.load(RAW / f"seed{s}.npz")["a5_idx"] for s in SEEDS}
    out = {}
    nw = len(WC); hi = np.tile(held, nw); hw = np.repeat(np.arange(nw), len(held))
    for tgt in SEEDS:
        pols = [P.load_policy(f"10x5+carbon+priority+m_s{tgt}"), P.load_policy(f"10x5+carbon+priority+c_s{tgt}")]
        for src in SEEDS:
            if src == tgt:
                continue
            m, c = run_rows(pols, data, hi, G21[sched[src][hw]])
            out[f"{src}_on_{tgt}_mk"] = m.reshape(nw, len(held)); out[f"{src}_on_{tgt}_cb"] = c.reshape(nw, len(held))
        log(f"A6: target seed {tgt} done")
    np.savez_compressed(f, **out)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    data = P.load_data("data_train_vali")
    names = [Path(p).name for p in sorted_instance_files(str(REPO / "daniel/data/data_train_vali/SD2/10x5+carbon+priority"))]
    fit, held = make_split(len(data[0]), data)
    split = {"rng_seed": SPLIT_SEED, "stratified_by": "maximum processing time group",
             "fit": [names[i] for i in fit], "held_out": [names[i] for i in held],
             "fit_idx": fit.tolist(), "held_idx": held.tolist()}
    sp = OUT / "split.json"
    if sp.exists():
        assert json.loads(sp.read_text())["fit_idx"] == fit.tolist(), "split changed between runs"
    sp.write_text(json.dumps(split, indent=1))
    t0 = time.time()
    todo = [int(a) for a in sys.argv[1:] if a.isdigit()] or SEEDS
    for s in todo:
        run_seed(s, data, fit, held)
    if "--a6" in sys.argv or len(sys.argv) == 1:
        run_a6(data, held)
    log(f"SEGMENT_ORACLE_DONE: rollout GPU time this run {GPU_SECONDS[0] / 60:.1f} min, wall {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
