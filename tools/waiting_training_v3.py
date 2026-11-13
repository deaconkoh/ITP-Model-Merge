"""From-scratch DANIEL trained with waiting allowed (docs/waiting_training_protocol_v3.md).

Control: the From-scratch run behind the 0.754 no-delay cell of results/decoding_ceiling_v3_hv_table.md:
  phase 1  10x5+mix+bv3_cl_CLB_s111   2,600 updates, no stopping rule (configs/contrastive_v3.json)
  phase 2  10x5+mix+bv3_cc_CCB_s111   continued from phase 1's saved state with the convergence rule (window 1000,
                                      tol 0.005, min 1000, max 8000); stopped CONVERGED at 4,550; final state used
The new run repeats both phases with every argument taken from the control's saved run.json, except the run name and
--dispatch_rule semi_active (training, in-training validation, and evaluation all use the waiting-allowed rule).

    python tools/waiting_training_v3.py verify       # baseline HV from saved schedules; command reconstruction test
    python tools/waiting_training_v3.py train        # phase 1 then phase 2 (resumable: re-run)
    python tools/waiting_training_v3.py evaluate     # best-of-128 (seed 51) on the 20 x 10 subset; table
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
PY = str(REPO / ".venv" / "bin" / "python")
CLB = REPO / "results/contrastive_v3/runs/10x5+mix+bv3_cl_CLB_s111"
CCB = REPO / "results/contrastive_converged_v3/runs/10x5+mix+bv3_cc_CCB_s111"
P1, P2 = "bv3_wt_W1_s111", "bv3_wt_W2_s111"                    # new run names (phase 1, phase 2)
OUT = REPO / "results" / "waiting_training_v3"
SKIP = {"resume_from",                                         # set explicitly below
        # derived at start-up from --config (configs/benchmark_v3/mct.json), not command-line settings
        "feature_schema", "fea_j_input_dim", "fea_m_input_dim", "fea_pair_input_dim", "enable_priority", "config"}


_DEFAULTS = None


def defaults():
    """The parser's defaults (daniel/params.py), read in a separate process."""
    global _DEFAULTS
    if _DEFAULTS is None:
        code = "import sys,json; sys.argv=['x']; import params; print(json.dumps(vars(params.parser.parse_args([]))))"
        r = subprocess.run([PY, "-c", code], cwd=DAN, capture_output=True, text=True)
        _DEFAULTS = json.loads(r.stdout.strip().splitlines()[-1])
    return _DEFAULTS


def cfg_args(cfg, **over):
    """The command line of a saved run: --config plus every setting that differs from the parser default (settings
    derived from --config at start-up are left to it)."""
    c = dict(cfg); c.update(over)
    d = defaults()
    a = ["--config", str((DAN / cfg["config"]).resolve())]
    for k, v in sorted(c.items()):
        if k in SKIP or v is None or (k in d and d[k] == v and k not in over):
            continue
        a += [f"--{k}", *map(str, v)] if isinstance(v, list) else [f"--{k}", str(v)]
    return a


def control():
    c1 = json.loads((CLB / "run.json").read_text())["config"]
    c2 = json.loads((CCB / "run.json").read_text())["config"]
    return c1, c2


def train_cmd(phase, cfg, waiting=True, **over):
    extra = {"dispatch_rule": "semi_active"} if waiting else {}
    return [PY, str(DAN / "train.py"), *cfg_args(cfg, **extra, **over)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    out = {}
    # 1. the 0.754 baseline, rescored from its saved schedules
    sys.argv = sys.argv[:1]; sys.path.insert(0, str(REPO / "tools"))
    import numpy as np
    import nondelay_cpsat_v3 as N
    import benchmark_v3 as B, mo_metrics as MM
    data = B.load_split("evaluation", "10x5")
    norm = json.loads((REPO / "results/baselines_v3/normalisation_points.json").read_text())["instances"]
    z = np.load(REPO / "results/decoding_ceiling_v3/evaluation/fs__sample128.npz")
    hv, worst = [], 0.0
    for i in range(20):
        pts = []
        for g in N.SUBSET:
            r = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i],
                                      z["machine"][i, g], z["start"][i, g])
            o = np.array([r["makespan"], r["carbon"], r["tardiness"]]); worst = max(worst, float(np.abs(o - z["obj"][i, g]).max()))
            pts.append(o)
        hv.append(MM.instance_hv(np.array(pts), *MM.ideal_nadir(norm[data.keys[i]])))
    out["baseline_hv_rescored"] = float(np.mean(hv)); out["baseline_rescore_max_abs_diff"] = worst
    print(f"baseline HV from saved schedules: {np.mean(hv):.4f} (max objective difference {worst:.2e})", flush=True)
    # 2. the reconstructed command reproduces the control's own logs (first 100 updates, no-delay rule)
    c1, _ = control()
    n = 100
    with tempfile.TemporaryDirectory() as d:
        cmd = train_cmd(1, c1, waiting=False, model_suffix="bv3_wtchk_s111", max_updates=n, chunk_updates=n)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "3"})
        assert r.returncode == 0, r.stderr[-3000:]
        new_r = json.loads((Path(d) / "train_log/SD2/reward_10x5+mix+bv3_wtchk_s111.json").read_text())
        new_v = json.loads((Path(d) / "train_log/SD2/valiquality_10x5+mix+bv3_wtchk_s111.json").read_text())
        resume = Path(d) / "trained_network/SD2/10x5+mix+bv3_wtchk_s111.resume.pth"
        import freeze_plateau_diagnosis_v3 as F
        ref_hash = F.torch_load(resume)["identity"]["cl_reference_sha256"]
    old_r = json.loads((CLB / "reward.json").read_text())[:n]
    old_v = json.loads((CLB / "valiquality.json").read_text())
    old_v = [e for e in old_v if e.get("update", e.get("i_update", 0)) <= n] if old_v and isinstance(old_v[0], dict) else old_v
    same_loss = [a["cl"]["loss"] for a in new_r] == [b["cl"]["loss"] for b in old_r]
    same_hv = [e["hv"] for e in new_v] == [e["hv"] for e in old_v[:len(new_v)]]
    start_hash = json.loads((REPO / "configs/plateau_diagnosis_v3.json").read_text())["start_network_sha256"]["fs"]
    out["reconstruction"] = {"updates": n, "losses_identical": same_loss, "validation_hv_identical": same_hv,
                             "start_network_sha256": ref_hash, "control_start_network_sha256": start_hash,
                             "start_identical": ref_hash == start_hash, "seconds": time.time() - t0}
    print(json.dumps(out["reconstruction"], indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify.json").write_text(json.dumps(out, indent=1) + "\n")
    assert same_loss and same_hv and ref_hash == start_hash, "reconstruction does not reproduce the control"


def train():
    c1, c2 = control()
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "3", "WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1"}
    log = open(OUT / "train.log", "a")
    tn = DAN / "trained_network/SD2"
    st = DAN / "train_log/SD2"

    def status(name):
        f = st / f"10x5+mix+{name}.status.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def run(cmd, name):
        while status(name).get("status") not in ("FINISHED", "CONVERGED"):
            log.write(f"[{time.strftime('%F %T')}] {name}: chunk (status {status(name).get('status')}, update "
                      f"{status(name).get('update')})\n"); log.flush()
            r = subprocess.run(cmd, cwd=DAN, env=env, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                raise SystemExit(f"{name}: train.py failed (rc {r.returncode}); see {OUT / 'train.log'}")
        s = status(name)
        log.write(f"[{time.strftime('%F %T')}] {name}: {s['status']} at update {s['update']} "
                  f"({s['train_seconds'] / 3600:.2f} h)\n"); log.flush()
    t0 = time.time()
    run(train_cmd(1, c1, model_suffix=P1), P1)
    resume = tn / f"10x5+mix+{P1}.resume.pth"
    run(train_cmd(2, c2, model_suffix=P2) + ["--resume_from", str(resume)], P2)
    s1, s2 = status(P1), status(P2)
    rec = {"phase1": s1, "phase2": s2, "phase1_resume_sha256": sha(resume),
           "final_resume_sha256": sha(tn / f"10x5+mix+{P2}.resume.pth"),
           "train_hours": (s1["train_seconds"] + s2["train_seconds"]) / 3600, "wall_hours": (time.time() - t0) / 3600}
    (OUT / "train.json").write_text(json.dumps(rec, indent=1) + "\n")


def evaluate():
    """Final state of phase 2 wrapped as a bundle (as the control's u4550), then best-of-128, seed 51, waiting rule."""
    os.environ["DECODING_DEVICE"] = "cuda"
    sys.argv = [sys.argv[0]]; sys.path.insert(0, str(REPO / "tools"))
    import numpy as np, torch
    import freeze_plateau_diagnosis_v3 as F
    from checkpointing import make_checkpoint_bundle
    tn = DAN / "trained_network/SD2"
    fin = F.torch_load(tn / f"10x5+mix+{P2}.resume.pth")
    run = json.loads((DAN / "train_log/SD2" / f"10x5+mix+{P2}.run.json").read_text())
    upd = json.loads((DAN / "train_log/SD2" / f"10x5+mix+{P2}.status.json").read_text())["update"]
    ck = REPO / "checkpoints/waiting_training_v3"; ck.mkdir(parents=True, exist_ok=True)
    out = ck / f"10x5+mix+{P2}_final_u{upd}.pth"
    torch.save(make_checkpoint_bundle(fin["policy"], config=run["config"], data_fingerprints={}, update=upd,
                                      validation_metrics={}, selection_metric=0.0, repo_root=REPO), out)
    (ck / "SHA256SUMS.json").write_text(json.dumps({out.name: sha(out)}, indent=1) + "\n")
    import waiting_decoder_v3 as WD                                   # WaitingEnv = --dispatch_rule semi_active
    import decoding_v3 as D, benchmark_v3 as B, mo_metrics as MM, nondelay_cpsat_v3 as N
    WD.use_waiting()
    data = B.load_split("evaluation", "10x5")
    net = D.load_net(out)
    I = 20
    idx = [i for i in range(I) for _ in range(len(D.PREFS))]
    t0 = time.time()
    o, m, s, c = D.run_method(net, data, idx, np.tile(D.PREFS, (I, 1)), "sample128")["sample128"]
    for r, i in enumerate(idx):
        ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i], m[r], s[r])
        assert max(abs(ind["makespan"] - o[r, 0]), abs(ind["carbon"] - o[r, 1]), abs(ind["tardiness"] - o[r, 2])) < 1e-6
    o, m, s = o.reshape(I, -1, 3), m.reshape(I, len(D.PREFS), -1), s.reshape(I, len(D.PREFS), -1)
    np.savez_compressed(OUT / "fs_waiting__sample128.npz", obj=o, machine=m, start=s, seconds=time.time() - t0)
    norm = json.loads((REPO / "results/baselines_v3/normalisation_points.json").read_text())["instances"]
    hv = float(np.mean([MM.instance_hv(o[i, N.SUBSET], *MM.ideal_nadir(norm[data.keys[i]])) for i in range(I)]))
    base = json.loads((OUT / "verify.json").read_text())["baseline_hv_rescored"]
    tr = json.loads((OUT / "train.json").read_text())
    rec = {"hv_waiting": hv, "hv_nodelay": base, "difference": hv - base, "train_hours": tr["train_hours"],
           "phase1_update": tr["phase1"]["update"], "phase2_update": tr["phase2"]["update"],
           "phase2_status": tr["phase2"]["status"], "checkpoint": str(out.relative_to(REPO)), "checkpoint_sha256": sha(out)}
    (OUT / "result.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    {"verify": verify, "train": train, "evaluate": evaluate}[sys.argv[1]]()
