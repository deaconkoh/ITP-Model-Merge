"""v2 re-run of the three-objective warm-start comparison (pre-registered in results/v2_rerun_seed111.md).

Arms (goal mct, preference 1/3 each, Step 4 PPO settings and training generator):
  V2-M     v2, init = converted makespan specialist  m_s<seed>_obsv2           updates 0 -> 3000
  V2-S     v2, scratch (fresh seeded network)                                  updates 0 -> 3000
  V1-CONT  v1, init = Step 4 C1  s4t_spec_m_k125s060_s<seed>@u1000             updates 1000 -> 3000
  V2-T     v2, init = v2 tardiness specialist  t_k125s060_s<seed>_v2ft         updates 0 -> 3000
Warm starts load the whole checkpoint (critic included); no warm-up.

Training uses Step 4's own train.Trainer (environments, sampling, policy_old rollouts, PPO.update) with a copy of the
update loop of Trainer.train that runs in resumable chunks and omits the validation pass (it only selects a "best"
checkpoint, unused here, and draws no random numbers). Every checkpoint holds policy, policy_old, the Adam state and
all RNG states (python, numpy, torch CPU, torch CUDA); resuming happens only from checkpoints whose update is a
multiple of reset_env_timestep (the environment is resampled there), so a resumed run equals an uninterrupted one.

Subcommands (all accept --arms, --seeds, --smoke, --out-root):
  train --until U     continue each (arm, seed) to update U (one subprocess per unit); no-op if already there
  eval                evaluate every snapshot not yet evaluated, on the 49 HELD-OUT instances
  plan                print the chunk end points of one arm (for the shell driver)
  status --t0 EPOCH   write the human-readable STATUS file
Checkpoints: daniel/trained_network/SD2/v2rr/ (git-ignored); results: results/v2_rerun_seed111/ (defaults).
--smoke: 2 updates per arm in 2 chunks, reset_env_timestep 1, evaluation on 3 instances, everything under
--out-root (required), never inside the repository.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
PY = sys.executable
TAG = "k125s060"
WC, WT = "0.002384", "5.815362"
ARMS = ["V2-M", "V2-S", "V1-CONT", "V2-T"]
SNAPS = [0, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]
CHUNK_ENDS = [500, 1000, 1500, 2000, 2500, 3000]
RESET_EVERY = 20


def arm_spec(arm, seed):
    """(obs version, init checkpoint path or '', first update)."""
    ck = DAN / "trained_network/SD2"
    return {
        "V2-M": ("v2", str(ck / f"10x5+carbon+priority+m_s{seed}_obsv2.pth"), 0),
        "V2-S": ("v2", "", 0),
        "V1-CONT": ("v1", str(ck / f"10x5+carbon+priority+s4t_spec_m_{TAG}_s{seed}@u1000.pth"), 1000),
        "V2-T": ("v2", str(ck / f"10x5+carbon+priority+t_{TAG}_s{seed}_v2ft.pth"), 0),
        # loop-fidelity checks only (never part of the study): 40 updates from a warm start
        "FID-V1": ("v1", str(ck / f"10x5+carbon+priority+s4t_spec_m_{TAG}_s{seed}@u1000.pth"), 0),
        "FID-V2": ("v2", str(ck / f"10x5+carbon+priority+m_s{seed}_obsv2.pth"), 0),
    }[arm]


def plan(arm, smoke):
    start = arm_spec(arm, 0)[2]
    if smoke:
        return [start, start + 1, start + 2], [start + 1, start + 2]
    if arm.startswith("FID"):
        return [0, 20, 40], [20, 40]
    return [u for u in SNAPS if u >= start], [u for u in CHUNK_ENDS if u > start]


class Paths:
    def __init__(self, a):
        if a.smoke and not a.out_root:
            raise SystemExit("--smoke requires --out-root (outputs never go inside the repository)")
        root = Path(a.out_root).resolve() if a.out_root else None
        self.ck = (root / "ckpt") if root else DAN / "trained_network/SD2/v2rr"
        self.res = (root / "results") if root else REPO / (a.results_dir or "results/v2_rerun_seed111")
        self.work = (root / "work") if root else DAN / "train_log/v2rr_work"
        self.prov, self.eval = self.res / "provenance", self.res / "eval"
        for p in (self.ck, self.prov, self.eval, self.work):
            p.mkdir(parents=True, exist_ok=True)

    def ckpt(self, arm, seed, u):
        return self.ck / f"v2rr_{arm}_s{seed}@u{u}.pth"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshots_on_disk(P, arm, seed):
    return sorted(int(f.stem.split("@u")[1]) for f in P.ck.glob(f"v2rr_{arm}_s{seed}@u*.pth"))


# ----------------------------------------------------------------------------------------------- training (one unit)
def _rng_state():
    import torch
    st = np.random.get_state()
    return {"python": random.getstate(), "numpy": (st[0], torch.from_numpy(st[1].astype(np.int64)), int(st[2]),
                                                   int(st[3]), float(st[4])),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def _set_rng_state(s):
    import torch
    random.setstate(s["python"])
    n = s["numpy"]; np.random.set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(s["torch_cpu"]); torch.cuda.set_rng_state_all(s["torch_cuda"])


def train_one(a):
    """Runs in its own process: sys.argv is rewritten for DANIEL's params module before train is imported."""
    arm, seed, until, smoke = a.arms[0], a.seeds[0], a.until, a.smoke
    P = Paths(a)
    for f in P.ck.glob(f".partial_v2rr_{arm}_s{seed}@u*.pth"):          # left by a kill during a save
        f.unlink()
    obs, init, start = arm_spec(arm, seed)
    snaps, _ = plan(arm, smoke)
    on_disk = [u for u in snapshots_on_disk(P, arm, seed) if u <= until and (smoke or u % RESET_EVERY == 0)]
    if on_disk and on_disk[-1] >= until:
        print(f"{arm} s{seed}: already at update {on_disk[-1]} >= {until}; nothing to do"); return 0
    resume = on_disk[-1] if on_disk else None
    argv = [str(DAN / "train.py"), "--config", str(REPO / "configs/canonical/mct.json"),
            "--train_data_path", str(DAN / "data/data_train/SD2/10x5+carbon+priority"),
            "--validation_data_path", str(DAN / "data/data_validation/SD2/10x5+carbon+priority"),
            "--test_data_path", str(DAN / "data/data_final_test/SD2/10x5+carbon+priority"),   # provenance only, never read
            "--n_j", "10", "--n_m", "5", "--data_suffix", "carbon+priority",
            "--carbon_reward_weight", WC, "--tardiness_reward_weight", WT,
            "--seed_train", str(seed), "--device", "cuda", "--model_suffix", f"v2rr_{arm}_{TAG}_s{seed}",
            "--init_from", init if resume is None else ""]
    if obs == "v2":
        argv += ["--obs_version", "v2", "--reward_version", "v2"]
    if smoke:
        argv += ["--reset_env_timestep", "1"]
    sys.argv = argv
    os.chdir(P.work)
    sys.path.insert(0, str(DAN))
    import torch
    import train as T
    from common_utils import setup_seed, sample_action
    from checkpointing import make_checkpoint_bundle, json_safe_config
    cfg = T.configs
    assert cfg.obs_version == obs and cfg.reward_version == obs and cfg.goal == "mct"
    trainer = T.Trainer(cfg)
    device = T.device

    def save(u):
        bundle = make_checkpoint_bundle(trainer.ppo.policy.state_dict(), config=json_safe_config(cfg),
                                        data_fingerprints=trainer.data_fingerprints, update=u, validation_metrics={},
                                        selection_metric=float("nan"), parents=trainer.parent_checkpoints,
                                        repo_root=REPO)
        bundle.update(policy_old_state_dict=trainer.ppo.policy_old.state_dict(),
                      optimizer_state_dict=trainer.ppo.optimizer.state_dict(), rng_state=_rng_state(),
                      study="v2_rerun", arm=arm, seed=seed)
        dst = P.ckpt(arm, seed, u); tmp = dst.with_name(".partial_" + dst.name)      # atomic; *.pth stays git-ignored
        torch.save(bundle, tmp); os.replace(tmp, dst)
        rec = P.prov / f"{arm}_s{seed}_checkpoints.json"
        d = json.loads(rec.read_text()) if rec.exists() else {}
        d[dst.name] = sha(dst); rec.write_text(json.dumps(d, indent=1, sort_keys=True))

    if resume is None:
        setup_seed(seed)                       # as Trainer.train() does before its loop
        cur = start
        save(cur)
    else:
        b = torch.load(P.ckpt(arm, seed, resume), map_location="cpu", weights_only=False)
        trainer.ppo.policy.load_state_dict(b["state_dict"]); trainer.ppo.policy_old.load_state_dict(b["policy_old_state_dict"])
        trainer.ppo.optimizer.load_state_dict(b["optimizer_state_dict"]); _set_rng_state(b["rng_state"])
        trainer.parent_checkpoints = b["parents"]
        cur = resume
    crash = os.environ.get("V2RR_CRASH_ONCE", "")            # smoke-test fault injection only: "<arm>:<update>"
    marker = P.prov / f"crash_injected_{arm}_s{seed}"
    t0 = time.time()
    print(f"{arm} s{seed}: {'resume from' if resume is not None else 'start at'} update {cur}, train to {until}", flush=True)
    for i_update in range(cur, until):
        if crash == f"{arm}:{i_update}" and not marker.exists():
            marker.write_text("injected\n"); raise RuntimeError(f"injected crash at {arm} update {i_update}")
        if os.environ.get("V2RR_CRASH_ALWAYS", "") == f"{arm}:{i_update}":
            raise RuntimeError(f"injected persistent crash at {arm} update {i_update}")
        # ---- copy of the body of train.Trainer.train (validation pass omitted) -------------------------------
        if i_update % trainer.reset_env_timestep == 0:
            dataset_job_length, dataset_op_pt, dataset_op_priority, dataset_op_carbon = trainer.sample_training_instances()
            due = [trainer.train_due[i] for i in trainer.last_sample_idxs] if trainer.train_due is not None else None
            refs = None
            if trainer.train_refs is not None:
                pairs = [trainer.train_refs["pairs"][i] for i in trainer.last_sample_idxs]
                refs = {"goal": trainer.train_refs["goal"], "tbar": trainer.train_refs["tbar"],
                        "mk": [p[0] for p in pairs], "cb": [p[1] for p in pairs]}
            state = trainer.env.set_initial_data(dataset_job_length, dataset_op_pt, dataset_op_priority,
                                                 dataset_op_carbon, due_date_list=due, reward_refs=refs)
        else:
            state = trainer.env.reset()
        while True:
            trainer.memory.push(state)
            with torch.no_grad():
                pi_envs, vals_envs = trainer.ppo.policy_old(
                    fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                    fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
            action_envs, action_logprob_envs = sample_action(pi_envs)
            state, reward, done = trainer.env.step(actions=action_envs.cpu().numpy())
            reward = torch.from_numpy(reward).to(device)
            trainer.memory.done_seq.append(torch.from_numpy(done).to(device))
            trainer.memory.reward_seq.append(reward)
            trainer.memory.action_seq.append(action_envs)
            trainer.memory.log_probs.append(action_logprob_envs)
            trainer.memory.val_seq.append(vals_envs.squeeze(1))
            if done.all():
                break
        trainer.ppo.update(trainer.memory)
        trainer.memory.clear_memory()
        # ------------------------------------------------------------------------------------------------------
        if (i_update + 1) in snaps or (i_update + 1) == until:
            save(i_update + 1)
    wall = time.time() - t0
    with open(P.prov / f"{arm}_s{seed}_chunks.jsonl", "a") as fh:
        fh.write(json.dumps({"from": cur, "to": until, "wall_seconds": round(wall, 2),
                             "seconds_per_update": round(wall / max(1, until - cur), 4),
                             "finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(f"{arm} s{seed}: reached update {until} in {wall:.0f} s ({wall / max(1, until - cur):.3f} s/update)", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate(a):
    P = Paths(a)
    sys.argv = sys.argv[:1]
    sys.path.insert(0, str(REPO / "tools"))
    import v2_common as V
    import ceiling_scorer as SC
    from model.main_model import DANIEL
    from common_utils import load_checkpoint_state_dict
    held = SC.HELD[:3] if a.smoke else SC.HELD
    for f in P.eval.glob(".partial_*"):
        f.unlink()
    data = V.P.load_data("data_train_vali"); due = SC.due_dates()
    for arm in a.arms:
        for seed in a.seeds:
            obs = arm_spec(arm, seed)[0]
            snaps, _ = plan(arm, a.smoke)
            ref = SC.reference(seed)
            for u in snaps:
                f, ck = P.eval / f"{arm}_s{seed}@u{u}.npz", P.ckpt(arm, seed, u)
                if f.exists() or not ck.exists():
                    continue
                cfg = V.v2_config() if obs == "v2" else V.P.configs
                net = DANIEL(cfg)
                net.load_state_dict(load_checkpoint_state_dict(ck, map_location=V.P.configs.device,
                                                               expected_schema=cfg.feature_schema))
                net.eval()
                o = V.run(net, data, due, held, obs=obs)
                score = np.array([SC.score(seed, i, o["mk"][k], o["cb"][k], o["tard"][k], ref=ref)
                                  for k, i in enumerate(held)])
                tmp = f.with_name(".partial_" + f.name)                      # atomic: a kill never leaves a half file
                np.savez_compressed(tmp, held=held, score=score, **o); os.replace(tmp, f)
                print(f"eval {arm} s{seed} @u{u}: score {score.mean():.4f}", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- status
def status(a):
    P = Paths(a)
    lines = [f"v2 re-run STATUS  (updated {time.strftime('%Y-%m-%d %H:%M:%S')})",
             f"elapsed since start: {(time.time() - a.t0) / 3600:.2f} h" if a.t0 else "", ""]
    rates, remaining = [], 0
    rows = []
    for arm in a.arms:
        for seed in a.seeds:
            snaps, _ = plan(arm, a.smoke)
            disk = snapshots_on_disk(P, arm, seed)
            reached = disk[-1] if disk else None
            ev = sorted(int(f.stem.split("@u")[1]) for f in P.eval.glob(f"{arm}_s{seed}@u*.npz"))
            last = f"{float(np.load(P.eval / f'{arm}_s{seed}@u{ev[-1]}.npz')['score'].mean()):.4f} @u{ev[-1]}" if ev else "-"
            ch = P.prov / f"{arm}_s{seed}_chunks.jsonl"
            if ch.exists():
                rates += [json.loads(l)["seconds_per_update"] for l in ch.read_text().splitlines() if l.strip()]
            state = ("FAILED" if (P.prov / f"{arm}_s{seed}.FAILED").exists() else
                     "done" if (P.prov / f"{arm}_s{seed}.done").exists() else
                     "in progress" if disk else "pending")
            if state not in ("done",):
                remaining += snaps[-1] - (reached if reached is not None else snaps[0])
            rows.append(f"  {arm:<8} seed {seed}: {state:<11} update reached {reached if reached is not None else '-':>5} / "
                        f"{snaps[-1]}   last evaluated score {last}")
    rate = float(np.median(rates)) if rates else None
    lines += rows + ["",
                     f"median training speed: {rate:.3f} s/update" if rate else "median training speed: (no chunk finished yet)",
                     f"estimated remaining: {remaining * rate / 3600:.2f} h training (+ evaluation)" if rate else ""]
    fails = sorted(p.name for p in P.prov.glob("*.FAILED"))
    lines += ["", "failures: " + (", ".join(fails) if fails else "none")]
    if a.status_file:
        Path(a.status_file).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "_train_one", "eval", "plan", "status"])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[111])
    ap.add_argument("--until", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--results-dir", default="")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--status-file", default="")
    a = ap.parse_args()
    if a.cmd == "plan":
        print(" ".join(map(str, plan(a.arms[0], a.smoke)[1]))); return 0
    if a.cmd == "_train_one":
        return train_one(a)
    if a.cmd == "eval":
        return evaluate(a)
    if a.cmd == "status":
        return status(a)
    rc = 0
    for arm in a.arms:                           # one subprocess per unit: DANIEL's config is per process
        for seed in a.seeds:
            until = a.until if a.until is not None else plan(arm, a.smoke)[0][-1]
            cmd = [PY, __file__, "_train_one", "--arms", arm, "--seeds", str(seed), "--until", str(until)]
            cmd += (["--smoke"] if a.smoke else []) + (["--out-root", a.out_root] if a.out_root else []) \
                + (["--results-dir", a.results_dir] if a.results_dir else [])
            rc = subprocess.run(cmd).returncode or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
