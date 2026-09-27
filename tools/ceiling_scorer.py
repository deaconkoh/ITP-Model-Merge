"""Independent schedule scorer for the ceiling check (results/ceiling_check.md, Step 0).

Takes a COMPLETE schedule (machine and start time of every operation) and computes makespan, total carbon, total
tardiness and the Step 4 score, without touching the DANIEL environment:

  * instance data parsed directly from the .fjs text (own parser; raw processing times and carbon, no scaling --
    the environment's processing-time normalisation only feeds the network, its reported makespan, carbon and
    tardiness are all in raw units);
  * completion C_o = start_o + pt[o, machine_o]; makespan = max_o C_o; carbon = sum_o carbon[o, machine_o];
    tardiness = sum_j max(0, C_(last op of j) - d_j) with d_j from the frozen due-date manifest (k125s060,
    hash-checked);
  * Step 4 score, preference (1/3, 1/3, 1/3), per seed s:
        w_m * mk / mk_ref[s, i] + w_c * cb / cb_ref[s, i] + w_t * T / mean_100(T_ref[s])
    where ref = the seed's centroid composition simplex_m333c333t334_k125s060_s<s> (per-instance makespan and
    carbon; tardiness by the centroid's mean over ALL 100 pool instances), exactly as analyze_arms_t.py /
    analyze_blend_finetune.py.
  * feasibility is verified (eligible machine, job precedence, no overlap per machine, start >= 0); an infeasible
    schedule raises.

    python tools/ceiling_scorer.py --check      # Step 0: reproduce the evaluator on C1@1000 greedy schedules
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "daniel/data/data_train_vali/SD2/10x5+carbon+priority"
S4 = REPO / "daniel/test_results/SD2/trainvali-k125s060_10x5+carbon+priority"
OUT = REPO / "results/ceiling_check"
TAG = "k125s060"
SEEDS = [111, 222, 333, 444]
W = (1 / 3, 1 / 3, 1 / 3)            # Preference(1/3, 1/3, 1/3, third="t") as Step 4 (checked in --check)
HELD = np.array(json.loads((REPO / "results/blend_zero_training/split.json").read_text())["held_idx"])


class Instance:
    """One .fjs instance: jobs -> operations (global index in job order), per-op eligible machines, pt, carbon."""

    def __init__(self, path):
        lines = Path(path).read_text().splitlines()
        n_j, n_m = (int(x) for x in lines[0].split()[:2])
        self.name, self.n_j, self.n_m = Path(path).name, n_j, n_m
        self.job_ops, self.pt, self.cb = [], [], []            # pt/cb: list of dict machine -> value
        for j in range(n_j):
            v = [int(x) for x in re.findall(r"\d+", lines[j + 1])]
            n_ops, k, ops = v[0], 1, []
            for _ in range(n_ops):
                n_mch = v[k]                                     # then the operation priority (unused here)
                pt, cb = {}, {}
                for q in range(n_mch):
                    m, p, c = v[k + 2 + 3 * q: k + 5 + 3 * q]
                    pt[m - 1], cb[m - 1] = p, c
                k += 2 + 3 * n_mch
                ops.append(len(self.pt)); self.pt.append(pt); self.cb.append(cb)
            self.job_ops.append(ops)
        self.n_ops = len(self.pt)


def instances():
    files = sorted(POOL.glob("*.fjs"))
    sys.path.insert(0, str(REPO / "daniel"))
    from data_utils import sorted_instance_files                      # loader order = manifest order
    assert [Path(f).name for f in sorted_instance_files(str(POOL))] == [f.name for f in files]
    return [Instance(f) for f in files]


def due_dates():
    sys.path.insert(0, str(REPO / "daniel"))
    from due_dates import due_dates_for_directory, load_manifest, manifest_tag
    man = load_manifest()
    assert manifest_tag(POOL, man) == TAG
    return [np.asarray(d, dtype=np.float64) for d in due_dates_for_directory(POOL, man)]   # hash-checked


def reference(seed):
    f = S4 / f"Result_DANIELG+10x5+carbon+priority+simplex_m333c333t334_{TAG}_s{seed}_trainvali-{TAG}_10x5+carbon+priority.npy"
    r = np.load(f)[:, [0, 1, 3]]
    return r[:, 0], r[:, 1], float(np.mean(r[:, 2]))                  # mk_ref[i], cb_ref[i], pooled T_ref


def raw_objectives(inst: Instance, due, machine, start):
    """machine[o], start[o] for every operation o (global index). Returns (makespan, carbon, tardiness)."""
    machine = [int(x) for x in machine]; start = [float(x) for x in start]
    assert len(machine) == len(start) == inst.n_ops
    comp = []
    for o in range(inst.n_ops):
        if machine[o] not in inst.pt[o]:
            raise ValueError(f"{inst.name}: op {o} on ineligible machine {machine[o]}")
        if start[o] < 0:
            raise ValueError(f"{inst.name}: op {o} starts before 0")
        comp.append(start[o] + inst.pt[o][machine[o]])
    for ops in inst.job_ops:
        for a, b in zip(ops, ops[1:]):
            if start[b] < comp[a]:
                raise ValueError(f"{inst.name}: precedence violated between ops {a} and {b}")
    for m in range(inst.n_m):
        on = sorted((start[o], comp[o]) for o in range(inst.n_ops) if machine[o] == m)
        for (s0, c0), (s1, _) in zip(on, on[1:]):
            if s1 < c0:
                raise ValueError(f"{inst.name}: overlap on machine {m}")
    mk = max(comp)
    cb = float(sum(inst.cb[o][machine[o]] for o in range(inst.n_ops)))
    tard = float(sum(max(0.0, comp[ops[-1]] - due[j]) for j, ops in enumerate(inst.job_ops)))
    return float(mk), cb, tard


def score(seed, i, mk, cb, tard, ref=None):
    """Step 4 score of instance i (pool index) at seed; same arithmetic order as preference3.scalarise."""
    mk_ref, cb_ref, t_ref = ref if ref is not None else reference(seed)
    return W[0] * (mk / mk_ref[i]) + W[1] * (cb / cb_ref[i]) + W[2] * (tard / t_ref)


def check():
    """Step 0: C1@1000 greedy schedules (all HELD-OUT instances, 4 seeds) -> scorer must reproduce the evaluator."""
    sys.path.insert(0, str(REPO / "tools"))
    import ceiling_sampling as CS                                     # GPU rollout (imports torch)
    from preference3 import Preference
    p = Preference(1 / 3, 1 / 3, 1 / 3, third="t"); assert (p.w_m, p.w_c, p.w_p) == W
    raw = OUT / "raw"; raw.mkdir(parents=True, exist_ok=True)
    insts, dues = instances(), due_dates()
    rep, worst = {}, 0.0
    for s in SEEDS:
        g = CS.greedy_c1(s)                                          # schedules + the rollout's own objectives
        ev = np.load(S4 / f"Result_DANIELG+10x5+carbon+priority+s4t_spec_m_{TAG}_s{s}@u1000_trainvali-{TAG}"
                          f"_10x5+carbon+priority.npy")[:, [0, 1, 3]][HELD]
        env_same = bool(np.array_equal(np.stack([g["mk"], g["cb"], g["tard"]], 1), ev))
        ref = reference(s)
        ev_score = p.w_m * (ev[:, 0] / ref[0][HELD]) + p.w_c * (ev[:, 1] / ref[1][HELD]) + p.w_p * (ev[:, 2] / ref[2])
        mine = np.array([raw_objectives(insts[i], dues[i], g["machine"][k], g["start"][k]) for k, i in enumerate(HELD)])
        sc = np.array([score(s, i, *mine[k], ref=ref) for k, i in enumerate(HELD)])
        d_obj = np.abs(mine - ev).max(0); d_sc = float(np.abs(sc - ev_score).max())
        worst = max(worst, d_sc)
        rep[s] = {"n": int(len(HELD)), "rollout_equals_step4_file": env_same,
                  "max_abs_diff_makespan": float(d_obj[0]), "max_abs_diff_carbon": float(d_obj[1]),
                  "max_abs_diff_tardiness": float(d_obj[2]), "max_abs_diff_score": d_sc,
                  "pooled_score_evaluator": float(ev_score.mean()), "pooled_score_scorer": float(sc.mean())}
        np.savez_compressed(raw / f"c1_greedy_s{s}.npz", held=HELD, machine=g["machine"], start=g["start"],
                            mk=mine[:, 0], cb=mine[:, 1], tard=mine[:, 2], score=sc)
        print(f"seed {s}: {rep[s]}", flush=True)
    ok = worst < 1e-9 and all(r["rollout_equals_step4_file"] for r in rep.values())
    (OUT / "step0_check.json").write_text(json.dumps({"pass": ok, "max_abs_diff_score": worst, "per_seed": rep},
                                                     indent=1))
    print("STEP0", "PASS" if ok else "FAIL", f"max |score diff| = {worst:.3g}")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(check())
