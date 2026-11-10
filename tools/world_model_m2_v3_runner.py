"""World-model M2 runner (docs/world_model_m2_protocol_v3.md): builds the bases, trains the 3 x 2 single-objective
contrastive specialists per seed (random start vs world-model start) in parallel slots, saves start and best
checkpoints with SHA-256, evaluates each on the evaluation split, renders the report, commits locally.

    python tools/world_model_m2_v3_runner.py [--seeds 111[,222,333]] [--jobs 3] [--smoke]

Resumable: re-running skips finished runs and continues unfinished ones from their last chunk. Called by
scripts/run_world_model_m2_v3.sh (which does the pre-flight).
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_world_model_m2_v3 as F                                # noqa: E402

PY = sys.executable
TN, TL = REPO / "daniel" / "trained_network" / "SD2", REPO / "daniel" / "train_log" / "SD2"
SLOTS = ["0-3", "4-7", "8-11"]
ARMS = {"ctrl": "random start", "wm": "world-model start"}
OBJ_ORDER = ["t", "c", "m"]                                         # slowest-converging first (the PPO specialists)


class Runner:
    def __init__(self, a):
        self.cfg = F.load()
        self.smoke = a.smoke
        sfx = "_smoke" if a.smoke else ""
        self.res, self.ckpt = REPO / "results" / f"world_model_m2_v3{sfx}", REPO / "checkpoints" / f"world_model_m2_v3{sfx}"
        self.tag = "wm2s" if a.smoke else "wm2"
        self.seeds = [int(s) for s in a.seeds.split(",")]
        bad = set(self.seeds) - set(self.cfg["seeds_available"])
        if bad:
            raise SystemExit(f"seeds {sorted(bad)} are not in the frozen settings")
        self.jobs = a.jobs
        self.prefix = "world-model M2 v3" + (" SMOKE" if a.smoke else "")
        self.res.mkdir(parents=True, exist_ok=True)
        self.git_lock, self.log_lock = threading.Lock(), threading.Lock()
        self.logf = open(self.res / "runner.log", "a")
        rule = self.cfg["rule"]
        sm = self.cfg["smoke"] if a.smoke else {}
        self.cap, self.chunk = sm.get("cap", rule["cap"]), sm.get("chunk", rule["chunk"])
        self.validate = sm.get("validate_timestep", self.cfg["contrastive"]["validate_timestep"])
        ev = self.cfg["evaluation"]
        self.eval_split, self.eval_first = sm.get("eval_split", ev["split"]), sm.get("eval_first", 0)
        self.sample = sm.get("sample", ev["sample"])

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
                subprocess.run(["git", "commit", "-q", "-m", msg, "-m",
                                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], cwd=REPO, check=True)

    def run(self, cmd, slot, timeout, out):
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        with open(out, "a") as f:
            try:
                return subprocess.run(["taskset", "-c", SLOTS[slot], *cmd], cwd=REPO / "daniel", stdout=f,
                                      stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                return 124

    # ------------------------------------------------------------------ units
    def bases(self):
        out = self.ckpt / "bases"
        need = [s for s in self.seeds if not (out / f"world_model_base_s{s}.pth").exists()]
        if need:
            self.log(f"bases for seeds {need}")
            subprocess.run([PY, str(REPO / "tools" / "world_model_m2_bases.py"), "--seeds", ",".join(map(str, need)),
                            "--out", str(out)], cwd=REPO, check=True, stdout=self.logf, stderr=subprocess.STDOUT)
            self.commit(f"{self.prefix}: starting bases (seeds {need})", out)
        return out

    def name(self, arm, obj, seed):
        return f"10x5+mix+bv3_{self.tag}_{arm}_{obj}_s{seed}"

    def args(self, arm, obj, seed, base):
        c = self.cfg["contrastive"]; r = self.cfg["rule"]
        return ["train.py", "--config", f"../configs/benchmark_v3/{obj}.json", "--n_j", "10", "--n_m", "5",
                "--model_suffix", f"bv3_{self.tag}_{arm}_{obj}_s{seed}", "--seed_train", str(seed),
                "--v3_train_source", "generator", "--finetune_objective", "contrastive",
                "--cl_k", str(c["cl_k"]), "--cl_beta", str(c["cl_beta"]), "--cl_margin", str(c["cl_margin"]),
                "--cl_grad_clip", str(c["cl_grad_clip"]), "--lr", str(c["lr"]), "--minibatch_size", str(c["minibatch_size"]),
                "--num_envs", str(c["num_envs"]), "--reset_env_timestep", str(c["reset_env_timestep"]),
                "--validate_timestep", str(self.validate), "--converge_window", str(r["converge_window"]),
                "--converge_tol", str(r["converge_tol"]), "--min_updates", str(r["min_updates"]),
                "--max_updates", str(self.cap), "--chunk_updates", str(self.chunk),
                "--init_from", str(base), "--init_mode", "exact", "--resume", "True", "--device", "cuda"]

    def unit(self, arm, obj, seed, slot, bases):
        name = self.name(arm, obj, seed)
        rdir = self.res / "runs" / name
        if (rdir / "done.json").exists():
            self.log(f"skip {name} (done)"); return True
        rdir.mkdir(parents=True, exist_ok=True)
        base = bases / f"{'world_model_base' if arm == 'wm' else 'shared_random_base'}_s{seed}.pth"
        live, out = TL / f"{name}.status.json", TL / f"{name}.stdout.log"
        TL.mkdir(parents=True, exist_ok=True)
        fails, meta_f = 0, rdir / "process.json"
        while True:
            st = json.loads(live.read_text())["status"] if live.exists() else "NONE"
            if st in ("CONVERGED", "NOT_CONVERGED"):
                break
            extra = ["--budget_checkpoints", "0"] if st == "NONE" else []        # the start network: first chunk only
            self.log(f"train {name}: chunk (status {st}, slot {slot})")
            t0 = time.time()
            rc = self.run([PY, *self.args(arm, obj, seed, base), *extra], slot,
                          int(max(self.chunk, 1) * 2.5 * 3 + 900) if self.chunk else 3600, out)
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
        s = json.loads(live.read_text())
        self.log(f"{name}: {s['status']} at update {s['update']}, best {s['best_value']:.5g} @ {s['best_update']}, "
                 f"{s['train_seconds'] / 3600:.2f} h")
        edir = self.res / "eval"
        for attempt in (1, 2):
            rc = self.run([PY, "evaluate_v3.py", "--models", str(self.ckpt / f"{name}.pth"), "--split", self.eval_split,
                           "--size", "10x5", "--first", str(self.eval_first), "--sample", str(self.sample),
                           "--out", str(edir), "--device", "cuda"], slot, 3600, out)
            if rc == 0:
                break
            self.log(f"eval {name} FAILED (attempt {attempt})")
        else:
            (rdir / "FAILED").write_text("evaluation failed twice\n"); self.commit(f"{self.prefix}: {name} eval FAILED", rdir)
            return False
        (rdir / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        self.write_sums()
        self.commit(f"{self.prefix}: {name} ({s['status']})", rdir, edir, self.ckpt)
        return True

    def write_sums(self):
        with self.git_lock:
            sums = {str(p.relative_to(self.ckpt)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(self.ckpt.rglob("*.pth"))}
            (self.ckpt / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1) + "\n")

    def main(self):
        bases = self.bases()
        units = [(arm, o, s) for s in self.seeds for o in OBJ_ORDER for arm in ARMS]
        self.log(f"{len(units)} runs, {self.jobs} parallel slots, cap {self.cap}")
        free = list(range(self.jobs)); lock = threading.Lock(); ok = []

        def work(u):
            with lock:
                slot = free.pop(0)
            try:
                ok.append(self.unit(*u, slot, bases))
            finally:
                with lock:
                    free.append(slot)
        with ThreadPoolExecutor(self.jobs) as ex:
            list(ex.map(work, units))
        self.write_sums()
        rc = subprocess.run([PY, str(REPO / "tools" / "analyze_world_model_m2_v3.py"), *(["--smoke"] if self.smoke else []),
                             "--seeds", ",".join(map(str, self.seeds))], cwd=REPO, stdout=self.logf,
                            stderr=subprocess.STDOUT).returncode
        self.log(f"report {'written' if rc == 0 else 'FAILED (see runner.log)'}; runs ok {sum(ok)}/{len(units)}")
        self.commit(f"{self.prefix}: report", self.res, self.ckpt,
                    REPO / "results" / f"world_model_m2_v3{'_smoke' if self.smoke else ''}.md")
        return 0 if all(ok) and rc == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="111")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    sys.exit(Runner(ap.parse_args()).main())


if __name__ == "__main__":
    main()
