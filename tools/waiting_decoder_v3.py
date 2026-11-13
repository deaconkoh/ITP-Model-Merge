"""No-delay vs unrestricted hypervolume on the 200-pair solver subset (results/decoding_ceiling_v3.md, Part B2):
CP-SAT (restricted and unrestricted subset solves, as saved) and the From-scratch DANIEL checkpoint with best-of-128
sampling, decoded with the current no-delay rule (saved) and with a waiting-allowed rule (run here).

Waiting-allowed rule: the environment unchanged except the one line that masks pairs which cannot start at the
earliest possible time (fjsp_env_same_op_nums.py:640, `unavailable_pairs`). Every eligible pair of every job's next
operation is then legal; the chosen operation starts at max(job ready, machine free), so job precedence, machine
eligibility and machine capacity are kept. A machine can therefore stay idle while another operation is ready
(deliberate waiting). Operations are appended to their machine, never inserted into an earlier idle gap: the
reachable schedules are exactly the semi-active schedules, which include an optimal schedule for every regular
objective (makespan, carbon, total tardiness).

    python tools/waiting_decoder_v3.py check     # small example: a valid schedule the no-delay rule rules out
    python tools/waiting_decoder_v3.py run       # From scratch, best-of-128, evaluation (GPU)
    python tools/waiting_decoder_v3.py table
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ACTION = sys.argv[1] if len(sys.argv) > 1 else "table"
os.environ.setdefault("DECODING_DEVICE", "cuda" if ACTION == "run" else "cpu")
sys.argv = sys.argv[:1]
sys.path.insert(0, str(REPO / "tools"))
import numpy as np                                                  # noqa: E402
import decoding_v3 as D                                             # noqa: E402
import benchmark_v3 as B                                            # noqa: E402
import mo_metrics as MM                                             # noqa: E402
import tardiness_diagnosis_v3 as TD                                 # noqa: E402
import nondelay_cpsat_v3 as N                                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums              # noqa: E402

RES = REPO / "results" / "decoding_ceiling_v3"
OUT = RES / "evaluation_waiting"


class WaitingEnv(FJSPEnvForSameOpNums):
    """DANIEL's environment without the no-delay restriction (only `unavailable_pairs` is emptied)."""

    @property
    def unavailable_pairs(self):
        return self._unavailable_pairs

    @unavailable_pairs.setter
    def unavailable_pairs(self, v):
        self._unavailable_pairs = np.zeros_like(v)


def use_waiting():
    D.FJSPEnvForSameOpNums = WaitingEnv                              # make_env builds the relaxed environment


def deliberate_wait(data, i, m, s):
    """Total time a machine is idle while an operation it can process is ready (0 = non-delay)."""
    pt = np.asarray(data.op_pt[i], float); n, M = pt.shape
    jl = np.asarray(data.job_length[i]); first = np.r_[0, np.cumsum(jl)[:-1]]
    pred = np.arange(n) - 1; pred[first] = -1
    comp = s + pt[np.arange(n), m]; r = np.where(pred >= 0, comp[np.maximum(pred, 0)], 0.0); mk = comp.max()
    tot = 0.0
    for mm in range(M):
        ops = np.flatnonzero(m == mm); ops = ops[np.argsort(s[ops])]
        gaps, prev = [], 0.0
        for o in ops:
            if s[o] > prev + 1e-9:
                gaps.append((prev, s[o]))
            prev = comp[o]
        if mk > prev + 1e-9:
            gaps.append((prev, mk))
        for a, b in gaps:
            iv = sorted((max(r[o], a), min(s[o], b)) for o in np.flatnonzero(pt[:, mm] > 0) if min(s[o], b) > max(r[o], a) + 1e-9)
            if iv:
                lo, hi = iv[0]
                for x, y in iv[1:]:
                    if x > hi:
                        tot += hi - lo; lo, hi = x, y
                    else:
                        hi = max(hi, y)
                tot += hi - lo
    return tot


def check():
    """Small example (evaluation instance 0): random legal actions under the waiting rule give a valid schedule with
    deliberate waiting, which the no-delay rule cannot produce."""
    use_waiting()
    data = B.load_split("evaluation", "10x5")
    rng = np.random.default_rng(1)
    env = D.make_env(data, [0] * 8, np.tile(D.PREFS[[41]], (8, 1)))
    legal_counts = []
    while not np.all(env.done()):
        mask = ~env.dynamic_pair_mask.reshape(env.number_of_envs, -1)
        legal_counts.append(mask.sum(1).mean())
        D.step(env, np.array([rng.choice(np.flatnonzero(r)) for r in mask]))
    o, m, s = D.result(env)
    out = []
    for r in range(8):
        ind = B.schedule_objectives(data.job_length[0], data.op_pt[0], data.carbon[0], data.due[0], m[r], s[r])
        valid = abs(ind["makespan"] - o[r, 0]) < 1e-9 and abs(ind["tardiness"] - o[r, 2]) < 1e-6
        dw = deliberate_wait(data, 0, m[r], s[r])
        rep = TD.NonDelay(data, np.array([0])).run(TD.key_policy(TD.keys_from_schedule(m[r][None], s[r][None], 5)))
        same = bool(np.array_equal(rep["machine"][0], m[r]) and np.allclose(rep["start"][0], s[r]))
        out.append({"valid": bool(valid), "deliberate_waiting": float(dw), "no_delay_rule_reproduces": same,
                    "objectives": o[r].tolist()})
    # the no-delay environment never offers more pairs than the waiting one; compare the legal-pair counts
    env2 = FJSPEnvForSameOpNums(10, 5)
    env2.set_initial_data(**data.env_kwargs([0], "mct")); env2.set_preference(D.PREFS[[41]])
    rec = {"examples": out, "mean_legal_pairs_waiting": float(np.mean(legal_counts))}
    print(json.dumps(rec, indent=1))
    ok = [e for e in out if e["valid"] and e["deliberate_waiting"] > 0 and not e["no_delay_rule_reproduces"]]
    assert all(e["valid"] for e in out) and ok, "waiting decoder check failed"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(rec, indent=1) + "\n")


def run():
    use_waiting()
    data = B.load_split("evaluation", "10x5")
    I = 20
    idx = [i for i in range(I) for _ in range(len(D.PREFS))]
    prefs = np.tile(D.PREFS, (I, 1))
    net = D.load_net(REPO / D.CKPTS["fs"][1])
    t0 = time.time()
    res = D.run_method(net, data, idx, prefs, "sample128")
    o, m, s, c = res["sample128"]
    for r, i in enumerate(idx):
        ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i], m[r], s[r])
        assert max(abs(ind["makespan"] - o[r, 0]), abs(ind["carbon"] - o[r, 1]), abs(ind["tardiness"] - o[r, 2])) < 1e-6
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "fs__sample128.npz", obj=o.reshape(I, len(D.PREFS), 3), machine=m.reshape(I, len(D.PREFS), -1),
                        start=s.reshape(I, len(D.PREFS), -1), evals=np.asarray(c["evals"]).reshape(I, len(D.PREFS)),
                        seconds=c["seconds"])
    print(f"waiting decoder, From scratch, best-of-128: {time.time() - t0:.0f} s", flush=True)


def table():
    data = B.load_split("evaluation", "10x5")
    norm = json.loads((REPO / "results/baselines_v3/normalisation_points.json").read_text())["instances"]
    sub = N.SUBSET
    cells, counts, nd_wait = {}, {}, []
    sets = {"cpsat_nodelay": [], "cpsat_unres": [], "fs_nodelay": [], "fs_unres": [], "cpsat_nodelay_daniel": []}
    stat = {"restricted": {}, "unrestricted": {}}
    for i in range(20):
        stem = Path(data.keys[i]).stem
        for which, key in (("restricted", "cpsat_nodelay"), ("unrestricted", "cpsat_unres")):
            sol = json.loads((RES / "nondelay_cpsat/subset" / f"{which}_{stem}.json").read_text())["solves"]
            pts, ptsd = [], []
            for g in sub:
                r = sol[str(g)]
                stat[which][r["status"]] = stat[which].get(r["status"], 0) + 1
                pts.append(r["obj"])
                if which == "restricted":
                    mm, ss = np.asarray(r["machine"])[None], np.asarray(r["start"], float)[None]
                    rep = TD.NonDelay(data, np.array([i])).run(TD.key_policy(TD.keys_from_schedule(mm, ss, 5)))
                    if np.array_equal(rep["machine"], mm) and np.allclose(rep["start"], ss):
                        ptsd.append(r["obj"])
            sets[key].append(np.asarray(pts))
            if which == "restricted":
                sets["cpsat_nodelay_daniel"].append(np.asarray(ptsd).reshape(-1, 3))
        a = np.load(RES / "evaluation" / "fs__sample128.npz"); b = np.load(OUT / "fs__sample128.npz")
        sets["fs_nodelay"].append(a["obj"][i, sub]); sets["fs_unres"].append(b["obj"][i, sub])
        nd_wait.append(np.mean([deliberate_wait(data, i, b["machine"][i, g], b["start"][i, g]) > 1e-9 for g in sub]))
    for k, v in sets.items():
        cells[k] = float(np.mean([MM.instance_hv(p, *MM.ideal_nadir(norm[data.keys[i]])) if len(p) else 0.0
                                  for i, p in enumerate(v)]))
    fs_ev = (float(np.load(RES / "evaluation" / "fs__sample128.npz")["evals"][:, sub].mean()),
             float(np.load(OUT / "fs__sample128.npz")["evals"][:, sub].mean()))
    rec = {"hv": cells, "solver_status": stat, "fs_waiting_share_with_deliberate_wait": float(np.mean(nd_wait)),
           "fs_state_evals_per_pair": fs_ev, "preferences": [D.PREFS[g].tolist() for g in sub]}
    (RES / "hv_table.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    {"check": check, "run": run, "table": table}[ACTION]()
