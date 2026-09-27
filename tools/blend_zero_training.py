"""Zero-training three-specialist blend on the Step 4 metric (pre-registered in results/blend_zero_training.md).

EVALUATION ONLY. For each seed, the frozen makespan (M), carbon (C) and tardiness (T) specialists are combined
per decision:
  logit mixture:  log p = sum_i w_i log p_i over valid actions (invalid = -inf, set explicitly), log_softmax, argmax
  prob mixture:   p = sum_i w_i p_i, argmax
A row whose weights put 1 on a single expert takes that expert's argmax directly, as the pilot did, so vertex rows
are exactly the specialist. Single checkpoints (Step 4 re-scoring, arm R) are rolled out the same way.

Due dates come from the frozen manifest (k125s060), hash-checked. Instances are batched only with instances sharing
the same maximum processing time. Tardiness is computed exactly as the canonical evaluator does (per-job completion
= true completion of the job's last operation).

Outputs results/blend_zero_training/raw/seed<s>.npz (resume skips finished seeds).
    python tools/blend_zero_training.py [seeds...]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import complementarity as P                           # noqa: E402  pilot helpers (configs as in the pilot)
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums          # noqa: E402
from data_utils import sorted_instance_files                    # noqa: E402
from due_dates import due_dates_for_directory, load_manifest, manifest_tag   # noqa: E402
from simplex_merge import simplex_points                        # noqa: E402

OUT = REPO / "results/blend_zero_training"
RAW = OUT / "raw"
S4 = REPO / "daniel/test_results/SD2/trainvali-k125s060_10x5+carbon+priority"
POOL = REPO / "daniel/data/data_train_vali/SD2/10x5+carbon+priority"
SEEDS = [111, 222, 333, 444]
TAG = "k125s060"
SPLIT_SEED = 20260927
CHUNK = 9000
GPU_SECONDS = [0.0]
ARMS_R = ["merge", "scratch", "spec_m", "spec_c", "spec_t"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def _rollout(data, dues, pols, W, mode, diag=None):
    env = FJSPEnvForSameOpNums(n_j=data[0][0].shape[0], n_m=data[1][0].shape[1])
    state = env.set_initial_data(list(data[0]), list(data[1]), list(data[2]), list(data[3]), due_date_list=list(dues))
    Wt = torch.as_tensor(np.asarray(W, dtype=np.float64))
    coef = Wt.to(torch.float32).to(P.DEV)                                   # [E, n_experts]
    vertex = (Wt == 1.0).to(P.DEV)                                          # rows that are a single expert
    t = 0
    while True:
        lp = [F.log_softmax(pol.forward_logits(**P.args_of(state))[0], dim=1) for pol in pols]
        invalid = torch.isinf(lp[0])
        for x in lp[1:]:
            invalid = invalid | torch.isinf(x)
        if mode == "logit":
            z = sum(coef[:, i:i + 1] * torch.where(invalid, torch.zeros_like(x), x) for i, x in enumerate(lp))
            z = torch.where(invalid, torch.full_like(z, float("-inf")), z)
            a = F.log_softmax(z, dim=1).argmax(1)
        else:
            a = sum(coef[:, i:i + 1] * x.exp() for i, x in enumerate(lp)).argmax(1)
        for i, x in enumerate(lp):
            a = torch.where(vertex[:, i], x.argmax(1), a)
        if diag is not None:
            valid = ~invalid
            tops = torch.stack([x.argmax(1) for x in lp], 1)
            ents = torch.stack([-torch.where(valid & (x.exp() > 0), x.exp() * x, torch.zeros_like(x)).sum(1) / np.log(2)
                                for x in lp], 1)
            diag.setdefault("top", []).append(tops.cpu().numpy())
            diag.setdefault("ent", []).append(ents.cpu().numpy())
            diag.setdefault("chosen", []).append(a.cpu().numpy())
        state, _, done = env.step(actions=a.cpu().numpy())
        t += 1
        if np.all(done):
            break
    completion = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, completion - np.asarray(dues)).sum(1)
    return env.current_makespan.copy(), env.total_carbon.copy(), tard


def run_rows(pols, data, due, inst, W, mode="logit", diag=False):
    t0 = time.time()
    ub = np.array([float(data[1][i][data[1][i] > 0].max()) for i in inst])
    out = np.zeros((3, len(inst)))
    D = {} if diag else None
    for u in np.unique(ub):
        rows = np.flatnonzero(ub == u)
        for c0 in range(0, len(rows), CHUNK):
            r = rows[c0:c0 + CHUNK]
            sub = tuple([d[i] for i in inst[r]] for d in data)
            dg = {} if diag else None
            m, c, td = _rollout(sub, [due[i] for i in inst[r]], pols, np.asarray(W)[r], mode, dg)
            out[:, r] = m, c, td
            if diag:
                for k, v in dg.items():
                    arr = np.stack(v, 1)                                     # [E_chunk, T(, experts)]
                    full = D.setdefault(k, np.zeros((len(inst),) + arr.shape[1:], dtype=arr.dtype))
                    full[r] = arr
    torch.cuda.synchronize()
    GPU_SECONDS[0] += time.time() - t0
    return (out, D) if diag else out


def make_split(data):
    ub = np.array([float(pt[pt > 0].max()) for pt in data[1]])
    rng = np.random.default_rng(SPLIT_SEED)
    fit = []
    for u in np.unique(ub):
        g = np.flatnonzero(ub == u); rng.shuffle(g); fit += list(g[: (len(g) + 1) // 2])
    fit = np.sort(np.array(fit)); return fit, np.setdiff1d(np.arange(len(ub)), fit)


def s4_file(name):
    return S4 / f"Result_DANIELG+10x5+carbon+priority+{name}_trainvali-{TAG}_10x5+carbon+priority.npy"


def run_seed(seed, data, due):
    f = RAW / f"seed{seed}.npz"
    if f.exists():
        log(f"seed {seed}: complete, skipping"); return
    n = len(data[0]); allI = np.arange(n)
    names = {"M": f"10x5+carbon+priority+m_s{seed}", "C": f"10x5+carbon+priority+c_s{seed}",
             "T": f"10x5+carbon+priority+t_{TAG}_s{seed}"}
    pols = [P.load_policy(names[k]) for k in "MCT"]
    grid = np.array(simplex_points(0.1))                                   # [66, 3] permille (m, c, t)
    inst = np.tile(allI, len(grid)); W = np.repeat(grid / 1000.0, n, axis=0)
    G = run_rows(pols, data, due, inst, W).reshape(3, len(grid), n)
    log(f"seed {seed}: 66-point logit grid done")
    b1, diag = run_rows(pols, data, due, allI, np.full((n, 3), 1 / 3), diag=True)
    b2 = run_rows(pols, data, due, allI, np.full((n, 3), 1 / 3), mode="prob")
    # ---- equivalence check: vertices must equal Step 4's specialist evaluations exactly ------------------
    bad = []
    for k, v in (("M", (1000, 0, 0)), ("C", (0, 1000, 0)), ("T", (0, 0, 1000))):
        g = int(np.flatnonzero((grid == v).all(1))[0])
        ref = np.load(s4_file(names[k].split("+")[-1]))[:, [0, 1, 3]].T
        if not np.array_equal(G[:, g, :], ref):
            bad.append(f"vertex {k}: {int((G[:, g, :] != ref).any(0).sum())} instances differ")
    # ---- R: re-score Step 4 checkpoints (single policy) -------------------------------------------------
    R = {}
    for arm in ARMS_R:
        for b in (0, 1000):
            nm = f"s4t_{arm}_{TAG}_s{seed}@u{b}"
            ck = P.CK / f"10x5+carbon+priority+{nm}.pth"
            if not ck.exists():
                log(f"MISSING checkpoint {ck.name}"); continue
            out = run_rows([P.load_policy(ck.stem)], data, due, allI, np.ones((n, 1)))
            R[f"{arm}@{b}"] = out
            ref = np.load(s4_file(nm))[:, [0, 1, 3]].T
            if not np.array_equal(out, ref):
                bad.append(f"R {nm}: {int((out != ref).any(0).sum())} instances differ")
    if bad:
        raise SystemExit(f"EQUIVALENCE CHECK FAILED at seed {seed}: " + "; ".join(bad))
    log(f"seed {seed}: equivalence check passed (3 vertices + {len(R)} Step 4 checkpoints reproduce exactly)")
    np.savez_compressed(f, grid=grid, G=G, b1=b1, b2=b2, **{f"R_{k}": v for k, v in R.items()},
                        diag_top=diag["top"], diag_ent=diag["ent"], diag_chosen=diag["chosen"])
    log(f"seed {seed}: saved; cumulative GPU {GPU_SECONDS[0] / 60:.2f} min")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    data = P.load_data("data_train_vali")
    man = load_manifest()
    assert manifest_tag(POOL, man) == TAG
    due = due_dates_for_directory(POOL, man)                                # hash-checked, loader order
    names = [Path(p).name for p in sorted_instance_files(str(POOL))]
    fit, held = make_split(data)
    (OUT / "split.json").write_text(json.dumps({"rng_seed": SPLIT_SEED, "stratified_by": "maximum processing time group",
                                                "fit": [names[i] for i in fit], "held_out": [names[i] for i in held],
                                                "fit_idx": fit.tolist(), "held_idx": held.tolist()}, indent=1))
    todo = [int(a) for a in P._ARGV if a.isdigit()] or SEEDS
    for s in todo:
        run_seed(s, data, due)
    log(f"BLEND_EVAL_DONE: GPU {GPU_SECONDS[0] / 60:.2f} min this run")


if __name__ == "__main__":
    main()
