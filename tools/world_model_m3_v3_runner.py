"""World-model M3 runner (docs/world_model_m3_protocol_v3.md).

    python tools/world_model_m3_v3_runner.py --part a [--smoke]          # Part A: zero-training fronts, report, commit
    python tools/world_model_m3_v3_runner.py --part b [--jobs 3] [--smoke]

Part B trains the merge of each Part-B set (World-model base, Random shared base, Original specialists) with
train.py --merge_mix in parallel pinned slots, chunked and resumable; after each run: logs to
results/world_model_m3_v3/runs/<name>/, start (update 0) and best checkpoints to checkpoints/world_model_m3_v3/
(SHA256SUMS.json), evaluation on the evaluation split (66 preferences), commit. Then the report. Called by
scripts/run_world_model_m3_v3.sh (which does the pre-flight). Commits locally; never pushes.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_world_model_m3_v3 as F                                # noqa: E402

PY = sys.executable
TN, TL = REPO / "daniel" / "trained_network" / "SD2", REPO / "daniel" / "train_log" / "SD2"
SLOTS = ["0-3", "4-7", "8-11"]
CO_AUTHOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


class Runner:
    def __init__(self, a):
        self.cfg = F.load()
        self.smoke = a.smoke
        sfx = "_smoke" if a.smoke else ""
        self.res, self.ckpt = REPO / "results" / f"world_model_m3_v3{sfx}", REPO / "checkpoints" / f"world_model_m3_v3{sfx}"
        self.md = REPO / "results" / f"world_model_m3_v3{sfx}.md"
        self.tagA, self.tagB = ("wm3As", "wm3s") if a.smoke else ("wm3A", "wm3")
        self.jobs = a.jobs
        self.prefix = "world-model M3 v3" + (" SMOKE" if a.smoke else "")
        self.res.mkdir(parents=True, exist_ok=True)
        self.git_lock, self.log_lock = threading.Lock(), threading.Lock()
        self.logf = open(self.res / "runner.log", "a")
        rule, sm = self.cfg["part_b"]["rule"], (self.cfg["smoke"] if a.smoke else {})
        self.cap, self.chunk = sm.get("cap", rule["cap"]), sm.get("chunk", rule["chunk"])
        self.validate = sm.get("validate_timestep", self.cfg["part_b"]["contrastive"]["validate_timestep"])
        self.val_first = sm.get("mo_val_first", 0)
        self.eval_split, self.eval_first = sm.get("eval_split", "evaluation"), sm.get("eval_first", 0)

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self.log_lock:
            print(line, flush=True); self.logf.write(line + "\n"); self.logf.flush()

    def commit(self, msg, *paths):
        with self.git_lock:
            for p in paths:
                if Path(p).exists():
                    subprocess.run(["git", "add", "-A", "--", str(p)], cwd=REPO, check=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode:
                subprocess.run(["git", "commit", "-q", "-m", msg, "-m", CO_AUTHOR], cwd=REPO, check=True)

    def run(self, cmd, slot, timeout, out):
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        with open(out, "a") as f:
            try:
                return subprocess.run(["taskset", "-c", SLOTS[slot], *cmd], cwd=REPO / "daniel", stdout=f,
                                      stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                return 124

    def analyze(self):
        rc = subprocess.run([PY, str(REPO / "tools" / "analyze_world_model_m3_v3.py"), *(["--smoke"] if self.smoke else [])],
                            cwd=REPO, stdout=self.logf, stderr=subprocess.STDOUT).returncode
        self.log(f"report {'written' if rc == 0 else 'FAILED (see runner.log)'}")
        return rc

    # ------------------------------------------------------------------------------------------------ Part A
    def part_a(self):
        t0 = time.time()
        self.log("Part A: zero-training merge fronts of the four sets (validation and evaluation)")
        cmd = [PY, str(REPO / "tools" / "world_model_m3_part_a.py"), "--out", str(self.res / "part_a"), "--tag", self.tagA,
               "--first", str(self.cfg["smoke"]["part_a_first"] if self.smoke else 0)]
        rc = subprocess.run(["taskset", "-c", "0-3", *cmd], cwd=REPO, stdout=self.logf, stderr=subprocess.STDOUT,
                            env=dict(os.environ, OMP_NUM_THREADS="4")).returncode
        if rc:
            self.log(f"Part A rollouts FAILED (rc {rc})"); return 1
        (self.res / "part_a" / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                    "seconds": time.time() - t0}) + "\n")
        rc = self.analyze()
        self.commit(f"{self.prefix}: Part A (merging without training) and report", self.res, self.md)
        return rc

    # ------------------------------------------------------------------------------------------------ Part B
    def name(self, s):
        return f"10x5+mix+bv3_{self.tagB}_{s}_s111"

    def args(self, s):
        c, r = self.cfg["part_b"]["contrastive"], self.cfg["part_b"]["rule"]
        sp = self.cfg["sets"][s]["specialists"]
        return ["train.py", "--config", "../configs/benchmark_v3/mct.json", "--n_j", "10", "--n_m", "5",
                "--model_suffix", f"bv3_{self.tagB}_{s}_s111", "--seed_train", "111", "--v3_train_source", "generator",
                "--preference_dim", "3", "--pref_dirichlet", str(c["pref_dirichlet"]),
                "--mo_val_grid_step", str(c["mo_val_grid_step"]), "--mo_val_first", str(self.val_first),
                "--mo_val_norm", str(REPO / self.cfg["part_a"]["val_norm"]),
                "--finetune_objective", "contrastive", "--cl_k", str(c["cl_k"]), "--cl_beta", str(c["cl_beta"]),
                "--cl_margin", str(c["cl_margin"]), "--cl_grad_clip", str(c["cl_grad_clip"]), "--lr", str(c["lr"]),
                "--minibatch_size", str(c["minibatch_size"]), "--num_envs", str(c["num_envs"]),
                "--reset_env_timestep", str(c["reset_env_timestep"]), "--validate_timestep", str(self.validate),
                "--converge_window", str(r["converge_window"]), "--converge_tol", str(r["converge_tol"]),
                "--min_updates", str(r["min_updates"]), "--max_updates", str(self.cap),
                "--chunk_updates", str(self.chunk),
                "--merge_mix", ",".join(str(REPO / sp[o]) for o in "mct"), "--resume", "True", "--device", "cuda"]

    def unit(self, s, slot):
        name = self.name(s)
        rdir = self.res / "runs" / name
        if (rdir / "done.json").exists():
            self.log(f"skip {name} (done)"); return True
        rdir.mkdir(parents=True, exist_ok=True)
        live, out = TL / f"{name}.status.json", TL / f"{name}.stdout.log"
        TL.mkdir(parents=True, exist_ok=True)
        fails, meta_f = 0, rdir / "process.json"
        while True:
            st = json.loads(live.read_text())["status"] if live.exists() else "NONE"
            if st in ("CONVERGED", "NOT_CONVERGED"):
                break
            extra = ["--budget_checkpoints", "0"] if st == "NONE" else []        # the start mix: first chunk only
            self.log(f"train {name}: chunk (status {st}, slot {slot})")
            t0 = time.time()
            rc = self.run([PY, *self.args(s), *extra], slot,
                          int(max(self.chunk, 1) * 4 * 3 + 1800) if self.chunk else 3600, out)
            m = json.loads(meta_f.read_text()) if meta_f.exists() else {"process_seconds": 0.0, "chunk_rc": []}
            m["process_seconds"] += time.time() - t0; m["chunk_rc"].append(rc)
            meta_f.write_text(json.dumps(m, indent=1))
            if rc:
                fails += 1
                self.log(f"train {name}: chunk FAILED (rc {rc}), attempt {fails}")
                if fails >= 2:
                    (rdir / "FAILED").write_text(f"training failed twice (rc {rc}); see {out}\n")
                    self.commit(f"{self.prefix}: {name} FAILED", rdir); return False
            else:
                fails = 0
        for src, dst in ((f"valiquality_{name}.json", "valiquality.json"), (f"reward_{name}.json", "reward.json"),
                         (f"{name}.run.json", "run.json"), (f"{name}.status.json", "status.json")):
            shutil.copy(TL / src, rdir / dst)
        self.ckpt.mkdir(parents=True, exist_ok=True)
        shutil.copy(TN / f"{name}@u0.pth", self.ckpt / f"{name}_start.pth")
        shutil.copy(TN / f"{name}.pth", self.ckpt / f"{name}.pth")
        s_ = json.loads(live.read_text())
        self.log(f"{name}: {s_['status']} at update {s_['update']}, best validation HV {-s_['best_value']:.4f} @ "
                 f"{s_['best_update']}, {s_['train_seconds'] / 3600:.2f} h")
        edir = self.res / "eval"
        for attempt in (1, 2):
            rc = self.run([PY, "evaluate_mo_v3.py", "--model", str(self.ckpt / f"{name}.pth"), "--split", self.eval_split,
                           "--size", "10x5", "--first", str(self.eval_first), "--out", str(edir), "--device", "cuda"],
                          slot, 3600, out)
            if rc == 0:
                break
            self.log(f"eval {name} FAILED (attempt {attempt})")
        else:
            (rdir / "FAILED").write_text("evaluation failed twice\n"); self.commit(f"{self.prefix}: {name} eval FAILED", rdir)
            return False
        (rdir / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        self.write_sums()
        self.commit(f"{self.prefix}: {name} ({s_['status']})", rdir, edir, self.ckpt)
        return True

    def write_sums(self):
        with self.git_lock:
            if not self.ckpt.exists():
                return
            sums = {str(p.relative_to(self.ckpt)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(self.ckpt.rglob("*.pth"))}
            (self.ckpt / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1) + "\n")

    def part_b(self):
        units = [s for s in ("wm", "ctrl", "orig") if self.cfg["sets"][s]["part_b"]]
        self.log(f"Part B: {len(units)} merge fine-tuning runs, {self.jobs} parallel slots, cap {self.cap}")
        free = list(range(self.jobs)); lock = threading.Lock(); ok = []

        def work(s):
            with lock:
                slot = free.pop(0)
            try:
                ok.append(self.unit(s, slot))
            finally:
                with lock:
                    free.append(slot)
        with ThreadPoolExecutor(self.jobs) as ex:
            list(ex.map(work, units))
        self.write_sums()
        rc = self.analyze()
        self.log(f"runs ok {sum(ok)}/{len(units)}")
        self.commit(f"{self.prefix}: Part B report", self.res, self.ckpt, self.md)
        return 0 if all(ok) and rc == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    r = Runner(a)
    sys.exit(r.part_a() if a.part == "a" else r.part_b())


if __name__ == "__main__":
    main()
