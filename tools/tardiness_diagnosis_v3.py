"""Why is the tardiness specialist DANIEL-T so far from CP-SAT? Part A of docs/tardiness_diagnosis_protocol_v3.md
(no training): reward density, the gap in plain units, and a split of the gap into "no waiting" (DANIEL's non-delay
action mask) and "learning".

Every schedule below is built by DANIEL's own decision rule: at each step only the operation-machine pairs that can
start at the earliest possible start time are allowed (fjsp_env_same_op_nums.py: `unavailable_pairs = pair_free_time >
next_schedule_time`). For speed the search uses `NonDelay`, a lean batched copy of that mask in the environment's own
scaled arithmetic; every reported schedule is replayed through the real environment and re-scored by the independent
scorer (benchmark_v3.schedule_objectives), and the run stops if either disagrees.

    python tools/tardiness_diagnosis_v3.py run        # writes results/tardiness_diagnosis_v3/*.json (about 1 h)
    python tools/tardiness_diagnosis_v3.py render     # writes results/tardiness_diagnosis_v3.md
    python tools/tardiness_diagnosis_v3.py run --quick   # smoke: 3 instances, 3 generations, separate output

Evaluation split only (first 20 instances per size, all with a CP-SAT single-objective tardiness result); the reserved
final-test sets are never loaded. Reads the committed specialists' evaluations and CP-SAT results; writes nothing else.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "objective_validation"))
_argv = sys.argv
sys.argv = [sys.argv[0], "--device", "cpu", "--obs_version", "v3", "--reward_version", "v3", "--goal", "t"]
import numpy as np                                                  # noqa: E402
from params import configs                                          # noqa: E402,F401
import benchmark_v3 as B                                            # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums              # noqa: E402
import common as OV                                                 # noqa: E402
import schedules as S                                               # noqa: E402
sys.argv = _argv

OUT = REPO / "results" / "tardiness_diagnosis_v3"
CPSAT = REPO / "results" / "baselines_v3" / "cpsat" / "b1"
EVAL = REPO / "results" / "baselines_v3" / "eval"
SPLIT, N_INST, SEEDS = "evaluation", 20, (111, 222, 333)
# search settings, fixed before the run (docs/tardiness_diagnosis_protocol_v3.md)
GENERATIONS = {"10x5": 2000, "20x10": 1000}           # about 5 min and 50 min on the RTX 5060 Ti PC (CPU numpy)
SEARCH = {"population": 200, "elite": 20, "crossover": "uniform, per operation row",
          "mutation_rate": 0.05, "mutation_sigma": 0.1, "immigrants": 0.05, "seed": 2026}
RULES = {"EDD+ECT": "earliest due date", "MWKR+ECT": "most work remaining (mean processing time, as the environment)",
         "LST+ECT": "least slack: due date - earliest start - remaining minimum work (the network's slack input)"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------- the non-delay decoder

class NonDelay:
    """Batched copy of DANIEL's state transition restricted to what the action mask needs, in the same scaled float
    arithmetic (times divided by the fixed scale 99), so the allowed set at every step is bit-identical to the
    environment's dynamic_pair_mask (tested in tests/test_tardiness_diagnosis_v3.py)."""

    def __init__(self, data, idx):
        self.idx = np.asarray(idx)
        self.E = len(self.idx)
        raw = np.stack([np.asarray(data.op_pt[i], dtype=np.float64) for i in self.idx])          # [E, N, M]
        self.raw = raw
        self.elig = raw > 0
        self.pt = np.where(self.elig, raw / B.PT_SCALE, 0.0)            # environment: raw / max(upper, 1e-8)
        jl = np.stack([np.asarray(data.job_length[i]) for i in self.idx])
        self.first = np.concatenate([np.zeros((self.E, 1), int), np.cumsum(jl, 1)[:, :-1]], 1)
        self.last = self.first + jl - 1
        self.due = np.stack([np.asarray(data.due[i], dtype=np.float64) for i in self.idx])        # raw [E, J]
        self.E_, self.J, self.N, self.M = self.E, jl.shape[1], raw.shape[1], raw.shape[2]
        self.mean_pt = np.ma.array(self.pt, mask=~self.elig).mean(2).data
        self.min_pt = np.ma.array(self.pt, mask=~self.elig).min(2).data
        self.ei = np.arange(self.E)

    def reset(self):
        self.cand = self.first.copy()
        self.done = np.zeros((self.E, self.J), bool)
        self.r = np.zeros((self.E, self.J)); self.F = np.zeros((self.E, self.M))
        self.rr = np.zeros((self.E, self.J)); self.Fr = np.zeros((self.E, self.M))      # raw times
        self.machine = np.full((self.E, self.N), -1); self.start = np.zeros((self.E, self.N))
        self.sched = np.zeros((self.E, self.N), bool)
        self.actions = np.zeros((self.E, self.N), int)
        self.step_i = 0

    def allowed(self):
        """[E, J, M] pairs the environment allows now, and the earliest start time (scaled)."""
        el = self.elig[self.ei[:, None], self.cand] & ~self.done[:, :, None]
        pf = np.maximum(self.r[:, :, None], self.F[:, None, :])
        nst = np.min(np.where(el, pf, np.inf).reshape(self.E, -1), 1)
        return el & ~(pf > nst[:, None, None]), nst

    def step(self, action):
        j, m = action // self.M, action % self.M
        op = self.cand[self.ei, j]
        st = np.maximum(self.r[self.ei, j], self.F[self.ei, m]); ct = st + self.pt[self.ei, op, m]
        self.r[self.ei, j] = ct; self.F[self.ei, m] = ct
        st_r = np.maximum(self.rr[self.ei, j], self.Fr[self.ei, m]); ct_r = st_r + self.raw[self.ei, op, m]
        self.rr[self.ei, j] = ct_r; self.Fr[self.ei, m] = ct_r
        self.machine[self.ei, op] = m; self.start[self.ei, op] = st_r; self.sched[self.ei, op] = True
        self.actions[:, self.step_i] = action; self.step_i += 1
        fin = op == self.last[self.ei, j]
        self.done[self.ei[fin], j[fin]] = True
        self.cand[self.ei, j] += ~fin

    def run(self, policy):
        self.reset()
        for _ in range(self.N):
            ok, nst = self.allowed()
            a = policy(self, ok, nst)
            assert ok.reshape(self.E, -1)[self.ei, a].all(), "policy chose a masked pair"
            self.step(a)
        comp = self.rr                                                         # completion of each job's last op
        return {"tardiness": np.maximum(0.0, comp - self.due).sum(1), "tardy_share": (comp > self.due + 1e-9).mean(1),
                "makespan": comp.max(1), "machine": self.machine.copy(), "start": self.start.copy(),
                "actions": self.actions.copy()}


def key_policy(keys):
    """Random-key decoding: among the allowed pairs, the one whose (operation, machine) key is smallest."""
    def pol(d, ok, nst):
        k = keys[d.ei[:, None, None], d.cand[:, :, None], np.arange(d.M)[None, None, :]]
        return np.where(ok, k, np.inf).reshape(d.E, -1).argmin(1)
    return pol


def rule_policy(rule):
    """Job by the rule among jobs with an allowed pair (ties: lowest job index); machine by ECT among its allowed
    machines (all allowed pairs start at the same time, so ECT = shortest processing time; ties: lowest index)."""
    def pol(d, ok, nst):
        can = ok.any(2)
        if rule == "EDD+ECT":
            score = d.due / B.PT_SCALE
        elif rule == "MWKR+ECT":
            rem = d.mean_pt * ~d.sched
            score = -np.stack([np.add.reduceat(rem[e], d.first[e]) for e in range(d.E)])
        elif rule == "LST+ECT":
            rem = d.min_pt * ~d.sched
            work = np.stack([np.add.reduceat(rem[e], d.first[e]) for e in range(d.E)])
            score = d.due / B.PT_SCALE - nst[:, None] - work
        else:
            raise ValueError(rule)
        j = np.where(can, score, np.inf).argmin(1)
        pt = d.pt[d.ei, d.cand[d.ei, j]]
        m = np.where(ok[d.ei, j], pt, np.inf).argmin(1)
        return j * d.M + m
    return pol


def keys_from_schedule(machine, start, n_m):
    """Keys that make the decoder follow a given schedule's order: the pair it used gets its start time, every other
    pair of that operation comes after all used pairs. Returned as ranks scaled to [0, 1)."""
    E, N = machine.shape
    big = start.max() + 1.0
    k = np.full((E, N, n_m), 0.0)
    k += big + np.arange(n_m)[None, None, :] + start[:, :, None]
    k[np.arange(E)[:, None], np.arange(N)[None, :], machine] = start + np.arange(N)[None, :] * 1e-6
    return rank01(k)


def keys_from_actions(actions, cand_seq_op, n_m):
    """Keys that replay an action sequence exactly: the t-th chosen pair gets key t / N."""
    E, N = actions.shape
    k = np.ones((E, N, n_m)) + np.random.default_rng(0).random((E, N, n_m))
    k[np.arange(E)[:, None], cand_seq_op, actions % n_m] = np.arange(N)[None, :] / N
    return rank01(k)


def rank01(k):
    E = k.shape[0]
    flat = k.reshape(E, -1)
    r = np.empty_like(flat); r[np.arange(E)[:, None], flat.argsort(1, kind="stable")] = np.arange(flat.shape[1])
    return (r / flat.shape[1]).reshape(k.shape)


# ------------------------------------------------------------------------------------------- replay and re-score

def replay(data, idx, actions, goal="t", record=False):
    """Replays action sequences through the REAL environment; returns its objectives (and, with record, per-step
    tardiness and makespan potential changes)."""
    env = FJSPEnvForSameOpNums(*B.SIZES[data.size])
    env.set_initial_data(**data.env_kwargs(idx, goal))
    rec = {"t_reward": [], "mk_reward": [], "late_completion": []}
    for t in range(actions.shape[1]):
        a = actions[:, t]
        if record:
            mk_before = np.max(env.op_ct_lb, axis=1); job = a // env.number_of_machines
            last = env.candidate[env.env_idxs, job] == env.job_last_op_id[env.env_idxs, job]
        _, rew, done = env.step(a)
        if record:
            rec["t_reward"].append(np.asarray(rew).copy())
            rec["mk_reward"].append(mk_before - np.max(env.op_ct_lb, axis=1))
            jc = env.true_candidate_free_time[env.env_idxs, job]
            rec["late_completion"].append(last & (jc > env.true_due_dates[env.env_idxs, job] + 1e-9))
    assert done.all()
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    out = {"tardiness": np.maximum(0.0, comp - env.true_due_dates).sum(1), "makespan": env.current_makespan.copy()}
    return (out, {k: np.stack(v, 1) for k, v in rec.items()}) if record else out


def check(data, idx, res, label):
    """Real environment and independent scorer must both agree with the decoder."""
    env = replay(data, idx, res["actions"])
    for r, i in enumerate(idx):
        ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i],
                                    res["machine"][r], res["start"][r])
        for a, b in ((ind["tardiness"], res["tardiness"][r]), (env["tardiness"][r], res["tardiness"][r]),
                     (ind["makespan"], res["makespan"][r]), (env["makespan"][r], res["makespan"][r])):
            if abs(a - b) > 1e-6:
                raise AssertionError(f"{label}: decoder, environment and scorer disagree on {data.keys[i]}")


# ------------------------------------------------------------------------------------------------------- the search

def search(data, idx, seeds, cfg, log):
    """Random-key genetic search over non-delay schedules, all instances in one batch. seeds: [I, S, N, M] keys
    placed in the initial population. Returns the best result per instance and its trajectory."""
    I, P = len(idx), cfg["population"]
    rng = np.random.default_rng(cfg["seed"])
    dec = NonDelay(data, np.repeat(idx, P))
    N, M = dec.N, dec.M
    pop = rng.random((I, P, N, M))
    pop[:, :seeds.shape[1]] = seeds
    best = {"tardiness": np.full(I, np.inf)}
    traj, t0 = [], time.time()
    for g in range(cfg["generations"]):
        res = dec.run(key_policy(pop.reshape(I * P, N, M)))
        fit = res["tardiness"].reshape(I, P)
        order = np.argsort(fit, 1, kind="stable")
        top = order[:, 0]
        for i in range(I):
            v = fit[i, top[i]]
            if v < best["tardiness"][i] - 1e-9:
                best["tardiness"][i] = v
                for k in ("tardy_share", "makespan", "machine", "start", "actions"):
                    best.setdefault(k, [None] * I)[i] = res[k][i * P + top[i]]
                best.setdefault("generation", [0] * I)[i] = g
        traj.append(float(best["tardiness"].mean()))
        if g % 50 == 0 or g == cfg["generations"] - 1:
            log(f"  generation {g}: mean best tardiness {traj[-1]:.2f} ({time.time() - t0:.0f} s)")
        ne = cfg["elite"]
        elite = np.take_along_axis(pop, order[:, :ne, None, None], 1)                         # [I, ne, N, M]
        nc = P - ne
        pa = elite[np.arange(I)[:, None], rng.integers(0, ne, (I, nc))]
        pb = elite[np.arange(I)[:, None], rng.integers(0, ne, (I, nc))]
        child = np.where(rng.random((I, nc, N, 1)) < 0.5, pa, pb)
        mut = rng.random(child.shape) < cfg["mutation_rate"]
        child = np.where(mut, child + rng.normal(0, cfg["mutation_sigma"], child.shape), child)
        imm = rng.random((I, nc)) < cfg["immigrants"]
        child[imm] = rng.random((int(imm.sum()), N, M))
        pop = np.concatenate([elite, child], 1)
    out = {k: (np.stack(v) if k in ("machine", "start", "actions") else np.asarray(v)) for k, v in best.items()}
    out["trajectory"] = traj
    out["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------------------------------------------- Part A

def cpsat(size, keys):
    out = []
    for k in keys:
        rec = json.loads((CPSAT / f"{SPLIT}_{size}" / f"{Path(k).stem}.json").read_text())
        assert rec["key"] == k
        t = rec["points"]["tard"]
        out.append({"status": t["primary"]["status"], "machine": np.asarray(t["final"]["machine"]),
                    "start": np.asarray(t["final"]["start"], dtype=float), **t["final"]["exact"]})
    return out


def specialist(size, seed, mode):
    f = EVAL / f"{SPLIT}_{size}" / f"{size}+mix+bv3_spec_t_s{seed}{'' if mode == 'greedy' else '_' + mode}.npz"
    d = np.load(f)
    return {k: d[k] for k in d.files}, sha(f)


def reference_rule(data, idx):
    """The benchmark's frozen MWKR+ECT reference schedule (active-style, may wait; tools/objective_validation)."""
    out = {"tardiness": [], "tardy_share": []}
    for i in idx:
        jl, pt = data.job_length[i], data.op_pt[i]
        inst = OV.Instance(data.keys[i], jl, pt, np.repeat(np.arange(len(jl)), jl),
                           np.concatenate([[0], np.cumsum(jl)[:-1]]))
        m, s = S.build(inst, *B.REFERENCE_RULE)
        r = B.schedule_objectives(jl, pt, data.carbon[i], data.due[i], m, s)
        assert abs(r["makespan"] - data.mk_ref[i]) < 1e-9
        last = np.cumsum(jl) - 1
        out["tardiness"].append(r["tardiness"]); out["tardy_share"].append(float((r["completion"][last] > data.due[i]).mean()))
    return {k: np.asarray(v) for k, v in out.items()}


def active_rule(data, idx, job_rule):
    """The same job rule with active-style dispatching (allowed to wait), for context."""
    out = []
    for i in idx:
        jl, pt = data.job_length[i], data.op_pt[i]
        inst = OV.Instance(data.keys[i], jl, pt, np.repeat(np.arange(len(jl)), jl),
                           np.concatenate([[0], np.cumsum(jl)[:-1]]))
        m, s = S.build(inst, job_rule, "ECT", due=np.asarray(data.due[i]))
        out.append(B.schedule_objectives(jl, pt, data.carbon[i], data.due[i], m, s)["tardiness"])
    return np.asarray(out)


def stats(t, share=None, makespan=None):
    t = np.asarray(t, dtype=float)
    d = {"mean": float(t.mean()), "per_instance": [float(x) for x in t], "zero": int((t < 1e-9).sum())}
    if share is not None:
        d["tardy_share"] = float(np.mean(share))
    if makespan is not None:
        d["makespan"] = float(np.mean(makespan))
    return d


def part_a(size, quick, log):
    data = B.load_split(SPLIT, size)
    n_all = len(data)
    sub = np.arange(3 if quick else N_INST)
    keys = [data.keys[i] for i in sub]
    log(f"{size}: {len(sub)} instances ({keys[0]} .. {keys[-1]})")
    R = {"size": size, "keys": keys, "t_ref": data.t_ref, "inputs": {}}

    # ---- A3: the gap in plain units, all 100 evaluation instances
    all_idx = np.arange(n_all)
    ref_all = reference_rule(data, all_idx)
    cp_all = cpsat(size, [data.keys[i] for i in all_idx])
    spec = {}
    for seed in SEEDS:
        for mode in ("greedy", "sample100"):
            d, h = specialist(size, seed, mode)
            assert list(d["keys"]) == data.keys
            spec[(seed, mode)] = d
            R["inputs"][f"spec_t_s{seed}_{mode}"] = h
    a3 = {"rule": stats(ref_all["tardiness"], ref_all["tardy_share"]),
          "cpsat": {**stats([c["tardiness"] for c in cp_all], [c["tardy_share"] for c in cp_all]),
                    "optimal": int(sum(c["status"] == "OPTIMAL" for c in cp_all))}}
    for mode in ("greedy", "sample100"):
        per_seed = [stats(spec[(s, mode)]["tardiness"], spec[(s, mode)]["tardy_share"]) for s in SEEDS]
        a3[f"daniel_{mode}"] = {"per_seed": per_seed, "mean": float(np.mean([p["mean"] for p in per_seed])),
                                "tardy_share": float(np.mean([p["tardy_share"] for p in per_seed]))}
    a3["n"] = n_all
    R["a3"] = a3

    # ---- A4: the gap split on the first instances
    cp = cpsat(size, keys)
    t0 = time.time()
    a4 = {"cpsat": {**stats([c["tardiness"] for c in cp], [c["tardy_share"] for c in cp], [c["makespan"] for c in cp]),
                    "optimal": int(sum(c["status"] == "OPTIMAL" for c in cp))},
          "reference_rule": stats(reference_rule(data, sub)["tardiness"], reference_rule(data, sub)["tardy_share"])}
    # rules under the environment
    seeds, seed_names = [], []
    for rule in RULES:
        d = NonDelay(data, sub)
        res = d.run(rule_policy(rule))
        check(data, sub, res, rule)
        a4[f"env_{rule}"] = stats(res["tardiness"], res["tardy_share"], res["makespan"])
        seeds.append(keys_from_actions(res["actions"], op_sequence(d, res), d.M)); seed_names.append(rule)
    for rule, job in (("EDD+ECT", "EDD"), ("MWKR+ECT", "MWKR")):
        a4[f"active_{rule}"] = stats(active_rule(data, sub, job))
    # CP-SAT's order decoded through the environment
    d = NonDelay(data, sub)
    k_cp = keys_from_schedule(np.stack([c["machine"] for c in cp]), np.stack([c["start"] for c in cp]), d.M)
    res = d.run(key_policy(k_cp))
    check(data, sub, res, "CP-SAT order")
    a4["cpsat_order_in_env"] = stats(res["tardiness"], res["tardy_share"], res["makespan"])
    seeds.append(k_cp); seed_names.append("CP-SAT order")
    # DANIEL-T (committed evaluations): greedy and best of 100 samples; their schedules seed the search too
    for mode in ("greedy", "sample100"):
        a4[f"daniel_{mode}"] = {"per_seed": [stats(spec[(s, mode)]["tardiness"][sub], spec[(s, mode)]["tardy_share"][sub])
                                             for s in SEEDS]}
        a4[f"daniel_{mode}"]["mean"] = float(np.mean([p["mean"] for p in a4[f"daniel_{mode}"]["per_seed"]]))
        a4[f"daniel_{mode}"]["tardy_share"] = float(np.mean([p["tardy_share"] for p in a4[f"daniel_{mode}"]["per_seed"]]))
        for s in SEEDS:
            k = keys_from_schedule(spec[(s, mode)]["machine"][sub], spec[(s, mode)]["start"][sub], d.M)
            r2 = NonDelay(data, sub).run(key_policy(k))
            same = np.abs(r2["tardiness"] - spec[(s, mode)]["tardiness"][sub]) < 1e-6
            a4.setdefault("daniel_schedules_reproduced", {})[f"s{s}_{mode}"] = int(same.sum())
            seeds.append(k); seed_names.append(f"DANIEL-T s{s} {mode}")
    a4["seconds_rules_and_seeds"] = time.time() - t0
    seed_t = np.stack([NonDelay(data, sub).run(key_policy(k))["tardiness"] for k in seeds], 1)       # [I, S]
    a4["best_seed"] = stats(seed_t.min(1))
    a4["best_seed_source"] = [seed_names[j] for j in seed_t.argmin(1)]
    # the search, seeded with the rules, CP-SAT's order and DANIEL-T's schedules
    cfg = dict(SEARCH, generations=3, population=24, elite=4) if quick else dict(SEARCH, generations=GENERATIONS[size])
    log(f"{size}: search, population {cfg['population']} x {len(sub)} instances, {cfg['generations']} generations")
    sr = search(data, sub, np.stack(seeds, 1), cfg, log)
    check(data, sub, sr, "search")
    a4["search"] = {**stats(sr["tardiness"], sr["tardy_share"], sr["makespan"]), "seconds": sr["seconds"],
                    "trajectory": sr["trajectory"], "generation_of_best": [int(g) for g in sr["generation"]],
                    "beat_every_seed": int((sr["tardiness"] < seed_t.min(1) - 1e-9).sum()), "settings": cfg}
    R["a4"] = a4

    # ---- A1 (measured): how often the tardiness reward is non-zero along DANIEL-T s111's greedy schedules
    acts = NonDelay(data, sub).run(key_policy(keys_from_schedule(spec[(111, "greedy")]["machine"][sub],
                                                                  spec[(111, "greedy")]["start"][sub], d.M)))["actions"]
    _, rec = replay(data, sub, acts, goal="t", record=True)
    tr, mr, lc = rec["t_reward"], rec["mk_reward"], rec["late_completion"]
    half = tr.shape[1] // 2
    R["a1"] = {"steps": int(tr.shape[1]),
               "tardiness_reward_nonzero_share": float((np.abs(tr) > 1e-12).mean()),
               "late_job_completion_share": float(lc.mean()),
               "makespan_reward_nonzero_share": float((np.abs(mr) > 1e-12).mean()),
               "tardiness_return_first_half_share": float(tr[:, :half].sum() / tr.sum()) if tr.sum() else None,
               "makespan_return_first_half_share": float(mr[:, :half].sum() / mr.sum()) if mr.sum() else None,
               "schedules": "DANIEL-T s111 greedy, replayed through the environment with goal t"}
    return R


def op_sequence(d, res):
    """Operation chosen at each step of a decoder run, recovered from the action sequence."""
    E, N = res["actions"].shape
    cand = d.first.copy(); seq = np.zeros((E, N), int)
    for t in range(N):
        j = res["actions"][:, t] // d.M
        seq[:, t] = cand[d.ei, j]; cand[d.ei, j] += 1
    return seq


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["run", "render"])
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 instances, tiny search, separate output")
    ap.add_argument("--sizes", nargs="+", default=["10x5", "20x10"])
    a = ap.parse_args()
    out = OUT.with_name(OUT.name + "_smoke") if a.quick else OUT
    if a.action == "render":
        import analyze_tardiness_diagnosis_v3 as AN
        AN.render(out, quick=a.quick)
        return
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    for size in a.sizes:
        t0 = time.time()
        R = part_a(size, a.quick, log)
        R["seconds"] = time.time() - t0
        R["manifest_sha256"] = B.file_sha256(B.MANIFEST)
        (out / f"part_a_{size}.json").write_text(json.dumps(R, indent=1) + "\n")
        log(f"{size}: done in {R['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
