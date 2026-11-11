"""Why does CP-SAT schedule better than DANIEL? CP-SAT evaluation and analysis only (no training, no change to DANIEL,
the benchmark or the CP-SAT results). 10x5, evaluation split, the 20 instances with a CP-SAT B2 reference front.

Stages
  pool      (GPU) the known NON-DELAY schedules of each instance, i.e. schedules DANIEL's own action mask can produce:
              - every existing multi-objective checkpoint that is still on disk and has a committed evaluation-split
                result: greedy and best-of-16 (seed 50) schedules for the 67 evaluation preferences, regenerated with
                the evaluation's own settings and checked against the committed objectives;
              - the committed single-objective specialists' greedy and best-of-100 schedules (stored in their npz);
              - the earlier no-waiting rules, CP-SAT's tardiness order decoded without waiting, and the earlier
                no-waiting search (results/tardiness_diagnosis_v3.md), re-run with its fixed seed and checked against
                its committed per-instance values;
            and, for the World-model base mix final and From scratch final checkpoints, DANIEL's greedy trajectories
            with the action probabilities at every step (Part 4).
  analyse   (CPU) Parts 1-4 -> results/cpsat_gap_v3/analysis.json.gz
  render    the report -> results/cpsat_gap_v3.md (tools/analyze_cpsat_gap_v3.py)

    python tools/cpsat_gap_v3.py pool [--quick]
    python tools/cpsat_gap_v3.py analyse [--quick]
    python tools/cpsat_gap_v3.py render [--quick]

Reads only the evaluation split; the reserved final-test sets are never loaded.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("stage", choices=["pool", "analyse", "render"])
ap.add_argument("--quick", action="store_true", help="smoke test: 3 instances, 2 checkpoints, tiny search")
ap.add_argument("--device", default="cuda")
ap.add_argument("--only", choices=["rollouts", "fixed"], default=None, help="pool: one half only")
A = ap.parse_args()
sys.path.insert(0, str(REPO / "daniel")); sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "objective_validation"))
sys.argv = [sys.argv[0], "--device", A.device if A.stage == "pool" else "cpu", "--obs_version", "v3",
            "--reward_version", "v3", "--goal", "mct", "--preference_dim", "3"]
import numpy as np                                                  # noqa: E402
from params import configs                                          # noqa: E402
import benchmark_v3 as B                                            # noqa: E402
import mo_metrics as MM                                             # noqa: E402
import tardiness_diagnosis_v3 as TD                                 # noqa: E402

SPLIT, SIZE = "evaluation", "10x5"
N_INST = 3 if A.quick else 20
OUT = REPO / "results" / ("cpsat_gap_v3_smoke" if A.quick else "cpsat_gap_v3")
POOL = OUT / "pool"
B1 = REPO / "results" / "baselines_v3" / "cpsat" / "b1" / f"{SPLIT}_{SIZE}"
B2 = REPO / "results" / "baselines_v3" / "cpsat" / "b2" / f"{SPLIT}_{SIZE}"
EQUAL = np.array([[1 / 3, 1 / 3, 1 / 3]])
PREFS = np.vstack([MM.simplex_grid(0.1), EQUAL])                    # 66 grid preferences + equal weights (as evaluated)
G = 66
SAMPLE_K, SAMPLE_SEED, BATCH = 16, 50, 1400                         # evaluate_mo_v3 defaults used by every evaluation
TRACE = {"wm_final": "pd_wm_u2250", "fs_final": "pd_fs_u4550"}      # Part 4: plateau_diagnosis_v3 Part A checkpoints
LABEL = {"wm_final": "World-model base mix, final (update 2250)", "fs_final": "From scratch, final (update 4550)"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


# ------------------------------------------------------------------------------------------------ pool: rollouts

def mo_sources():
    """Every multi-objective evaluation-split result (10x5) whose checkpoint is still on disk (found by SHA-256)."""
    files = [f for f in glob.glob(str(REPO / "daniel" / "trained_network" / "SD2" / "*.pth"))
             + glob.glob(str(REPO / "checkpoints" / "**" / "*.pth"), recursive=True) if not f.endswith(".resume.pth")]
    by_sha = {}
    for f in sorted(files):
        by_sha.setdefault(sha(f), f)
    found, missing = [], []
    for j in sorted(glob.glob(str(REPO / "results" / "**" / f"{SPLIT}_{SIZE}" / "*.json"), recursive=True)):
        if "/cpsat/" in j:
            continue
        try:
            d = json.loads(Path(j).read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "checkpoint_sha256" not in d or "grid_size" not in d:
            continue                                                 # single-objective summaries, merges without files
        npz = Path(j).with_suffix(".npz")
        s16 = npz.with_name(npz.stem + "_sample16.npz")
        rec = {"name": d["model"], "result": rel(j), "greedy_npz": rel(npz),
               "s16_npz": rel(s16) if s16.exists() else None, "checkpoint_sha256": d["checkpoint_sha256"]}
        if d["checkpoint_sha256"] in by_sha:
            found.append({**rec, "checkpoint": rel(by_sha[d["checkpoint_sha256"]])})
        else:
            missing.append(rec)
    return found, missing


def load_net(path):
    import torch
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


def rollout(net, data, idx, prefs, gen=None, record=False):
    """evaluate_mo_v3.rollout, optionally recording the action, the allowed pairs and the action probabilities."""
    import torch
    from common_utils import greedy_select_action
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    env = FJSPEnvForSameOpNums(*B.SIZES[data.size])
    state = env.set_initial_data(**data.env_kwargs(idx, "mct"))
    state = env.set_preference(prefs)
    machine = np.full((len(idx), env.number_of_ops), -1)
    rec = {"action": [], "prob": [], "allowed": []}
    with torch.no_grad():
        while True:
            pi, _ = net(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                        fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor,
                        pref=state.pref_tensor)
            a = (greedy_select_action(pi) if gen is None else torch.multinomial(pi, 1, generator=gen).squeeze(1)).cpu().numpy()
            if record:
                rec["action"].append(a.copy()); rec["prob"].append(pi.float().cpu().numpy().astype(np.float32))
                rec["allowed"].append(~state.dynamic_pair_mask_tensor.cpu().numpy().reshape(len(idx), -1))
            op = env.candidate[env.env_idxs, a // env.number_of_machines]
            machine[env.env_idxs, op] = a % env.number_of_machines
            state, _, done = env.step(a)
            if np.all(done):
                break
    pt = env.true_op_pt[env.env_idxs[:, None], np.arange(env.number_of_ops)[None, :], machine]
    start = env.true_op_ct - pt
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
    obj = np.stack([env.current_makespan, env.total_carbon, tard], 1)
    if record:
        return obj, machine, start, {k: np.stack(v, 1) for k, v in rec.items()}
    return obj, machine, start


def score(data, i, obj, w):
    """The benchmark's preference-weighted normalised score (lower is better); obj [..., 3], w [..., 3]."""
    return (w[..., 0] * obj[..., 0] / data.mk_ref[i] + w[..., 1] * obj[..., 1] / data.cb_ref[i]
            + w[..., 2] * obj[..., 2] / data.t_ref)


def pool_rollouts(data, log):
    import torch
    found, missing = mo_sources()
    if A.quick:
        found = [f for f in found if f["name"] in TRACE.values()]
    I = N_INST
    meta = {"found": found, "missing_checkpoint": missing}
    for f in found:
        out = POOL / f"mo__{f['name']}.npz"
        if out.exists():
            continue
        t0 = time.time()
        net = load_net(REPO / f["checkpoint"])
        idx = [i for i in range(I) for _ in range(len(PREFS))]
        trace = f["name"] in TRACE.values()
        r = rollout(net, data, idx, np.tile(PREFS, (I, 1)), record=trace)
        obj_g, mach_g, st_g = r[:3]
        ref = np.load(REPO / f["greedy_npz"])["objectives"][:I]
        same_g = np.abs(obj_g.reshape(I, len(PREFS), 3) - ref).max(2) < 1e-9
        save = {"greedy_obj": obj_g.reshape(I, len(PREFS), 3), "greedy_machine": mach_g.reshape(I, len(PREFS), -1),
                "greedy_start": st_g.reshape(I, len(PREFS), -1), "greedy_reproduced": same_g}
        if trace:
            save.update({f"trace_{k}": v.reshape(I, len(PREFS), *v.shape[1:]) for k, v in r[3].items()})
        if f["s16_npz"]:
            gen = torch.Generator(device=configs.device); gen.manual_seed(SAMPLE_SEED)
            per = max(1, BATCH // (len(PREFS) * SAMPLE_K))
            assert per == 1
            o16 = np.zeros((I, len(PREFS), 3)); m16 = np.zeros((I, len(PREFS), mach_g.shape[1]), int)
            s16 = np.zeros((I, len(PREFS), mach_g.shape[1]))
            for i in range(I):
                idx = [i] * (len(PREFS) * SAMPLE_K)
                w = np.repeat(PREFS, SAMPLE_K, axis=0)
                o, m, s = rollout(net, data, idx, w, gen)
                sc = score(data, i, o, w).reshape(len(PREFS), SAMPLE_K)
                pick = sc.argmin(1) + np.arange(len(PREFS)) * SAMPLE_K
                o16[i], m16[i], s16[i] = o[pick], m[pick], s[pick]
            ref16 = np.load(REPO / f["s16_npz"])["objectives"][:I]
            save.update({"s16_obj": o16, "s16_machine": m16, "s16_start": s16,
                         "s16_reproduced": np.abs(o16 - ref16).max(2) < 1e-9})
        tmp = out.with_suffix(".tmp.npz"); np.savez_compressed(tmp, **save); tmp.replace(out)
        log(f"{f['name']}: greedy reproduced {int(same_g.sum())}/{same_g.size}"
            + (f", best-of-16 reproduced {int(save['s16_reproduced'].sum())}/{save['s16_reproduced'].size}"
               if "s16_reproduced" in save else "") + f" ({time.time() - t0:.0f} s)")
        del net
        torch.cuda.empty_cache()
    (POOL / "mo_sources.json").write_text(json.dumps(meta, indent=1) + "\n")


# ------------------------------------------------------------------------------------- pool: fixed (no network)

def pool_fixed(data, log):
    sub = np.arange(N_INST)
    keys = [data.keys[i] for i in sub]
    out, info = {}, {}
    seeds = []
    for rule in TD.RULES:                                                       # the earlier no-waiting rules
        d = TD.NonDelay(data, sub); res = d.run(TD.rule_policy(rule)); TD.check(data, sub, res, rule)
        out[f"rule__{rule}"] = res
        seeds.append(TD.keys_from_actions(res["actions"], TD.op_sequence(d, res), d.M))
    cp = TD.cpsat(SIZE, keys)                                                   # CP-SAT tardiness order, no waiting
    d = TD.NonDelay(data, sub)
    k_cp = TD.keys_from_schedule(np.stack([c["machine"] for c in cp]), np.stack([c["start"] for c in cp]), d.M)
    res = d.run(TD.key_policy(k_cp)); TD.check(data, sub, res, "CP-SAT order")
    out["cpsat_tard_order_no_waiting"] = res
    seeds.append(k_cp)
    spec = {}
    for mode in ("greedy", "sample100"):                                        # DANIEL-T specialists seed the search
        for s in TD.SEEDS:
            dd, h = TD.specialist(SIZE, s, mode)
            seeds.append(TD.keys_from_schedule(dd["machine"][sub], dd["start"][sub], d.M))
    cfg = dict(TD.SEARCH, generations=3, population=24, elite=4) if A.quick else dict(TD.SEARCH, generations=TD.GENERATIONS[SIZE])
    log(f"no-waiting search, {cfg['generations']} generations (as results/tardiness_diagnosis_v3.md)")
    sr = TD.search(data, sub, np.stack(seeds, 1), cfg, lambda m: None)
    TD.check(data, sub, sr, "search")
    out["search_tardiness"] = sr
    if not A.quick:
        ref = json.loads((REPO / "results" / "tardiness_diagnosis_v3" / "part_a_10x5.json").read_text())["a4"]["search"]
        same = np.abs(np.asarray(ref["per_instance"]) - sr["tardiness"]) < 1e-9
        info["search_reproduced"] = int(same.sum())
        log(f"search reproduced {int(same.sum())}/{len(same)} instances (mean {sr['tardiness'].mean():.3f}, "
            f"committed {ref['mean']:.3f})")
    save = {}
    for k, r in out.items():
        save[f"{k}__machine"] = np.asarray(r["machine"]); save[f"{k}__start"] = np.asarray(r["start"], dtype=float)
    # committed single-objective specialists (greedy and best of 100; their npz hold the schedules)
    for f in sorted(glob.glob(str(REPO / "results" / "*" / "eval" / f"{SPLIT}_{SIZE}" / "*.npz"))):
        z = np.load(f)
        if "machine" not in z.files or "start" not in z.files or z["machine"].ndim != 2:
            continue
        assert list(z["keys"][:N_INST]) == keys, f
        name = Path(f).stem
        save[f"so__{name}__machine"] = z["machine"][sub]; save[f"so__{name}__start"] = z["start"][sub].astype(float)
        info.setdefault("single_objective", {})[name] = rel(f)
    np.savez_compressed(POOL / "fixed.npz", **save)
    (POOL / "fixed.json").write_text(json.dumps(info, indent=1) + "\n")
    log(f"fixed pool: {len([k for k in save if k.endswith('__machine')])} sources")


def pool(log):
    data = B.load_split(SPLIT, SIZE)
    POOL.mkdir(parents=True, exist_ok=True)
    if A.only in (None, "fixed"):
        pool_fixed(data, log)
    if A.only in (None, "rollouts"):
        pool_rollouts(data, log)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"{A.stage}.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    if A.stage == "pool":
        pool(log)
    elif A.stage == "analyse":
        import cpsat_gap_v3_analysis as AN
        AN.run(A, log)
    else:
        import analyze_cpsat_gap_v3 as R
        R.render(OUT, quick=A.quick)


if __name__ == "__main__":
    main()
