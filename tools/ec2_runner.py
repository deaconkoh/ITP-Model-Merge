"""Warm-start run orchestrator (docs/warmstart_protocol_v3.md). Started by scripts/ec2_run_all.sh in the background.

Units, in the pre-registered order: calibration -> validation normalisation points (a CPU lane, alongside training)
-> per size, per seed (111 first), per arm (B, C-M, BB-M, D, E, C-C, C-T, BB-C, BB-T[, ES]): preparation (distillation,
merge, shared base + fine-tunes) -> chunked PPO training with a hang guard -> evaluation -> local commit; then E0 per
size and seed. Resume: finished units are skipped; training continues from its resume state (bit-exact). A unit that
fails twice is marked FAILED (its dependents too) and the run continues; a re-run retries it.

    python tools/ec2_runner.py --sizes 10x5 --arms all --seeds "111 222 333" --jobs 1 [--smoke] [--with-shared-base]

Contrastive arms (docs/contrastive_protocol_v3.md): --arms CL-B,CL-D,CL-E --seeds 111 runs them in their own mode
(results/contrastive_v3, run tag "cl", frozen budgets and validation normalisation from configs/contrastive_v3.json);
they cannot be mixed with the warm-start arms in one call.

Merge-category arms (docs/merge_categories_protocol_v3.md): --arms SB-AVG,TA,TIES --seeds 111 trains a shared base and
M/C/T fine-tunes from it (u_shared_base, unchanged), builds the shared-base average, Task Arithmetic and TIES merges
(lambda chosen on the validation grid), evaluates each zero-training merge and fine-tunes each merge with the contrastive
objective (results/merge_categories_v3, run tag "mc", frozen inputs in configs/merge_categories_v3.json). Starting
networks and best checkpoints are committed to checkpoints/merge_categories_v3/ with SHA-256 hashes.

Converged contrastive arms (docs/contrastive_converged_protocol_v3.md): --arms CC-B,CC-D,CC-E,CC-SBAVG,CC-TA,CC-TIES
--seeds 111 runs every contrastive arm under the PPO stopping rule with a cap of 8,000 updates: an arm whose logged history
never met the rule continues from its saved final state (after an exactness check), the others rerun from their starting
networks (results/contrastive_converged_v3, run tag "cc", frozen inputs in configs/contrastive_converged_v3.json).

Merge-failure test (docs/merge_failure_test_protocol_v3.md): --arms MF-SP,MF-NA,MF-PR --seeds 111 measures six starting
networks at update 0 (no training; merge_diagnostics.py) and fine-tunes three one-change arms from the Model Soup start
under the converged runs' stopping rule (results/merge_failure_test_v3, run tag "mf", frozen inputs in
configs/merge_failure_test_v3.json); start and best checkpoints in checkpoints/merge_failure_test_v3/ with SHA-256.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "warmstart_v3"           # results/warmstart_v3_smoke under --smoke (set in main)
TAG = "ws"                                         # run-name tag; "wss" under --smoke, so smoke never collides
TN = REPO / "daniel" / "trained_network" / "SD2"
TL = REPO / "daniel" / "train_log" / "SD2"
SPEC = REPO / "checkpoints" / "specialists_v3"
PY = sys.executable
STATUS_FILE = REPO / "logs" / "warmstart_v3_STATUS.txt"
PREFIX = "warm-start v3"                           # commit-message prefix; "contrastive v3" for the CL arms

ARM_ORDER = ["B", "CM", "BBM", "D", "E", "CC", "CT", "BBC", "BBT"]
ARM_LABEL = {"B": "B (scratch)", "CM": "C-M", "CC": "C-C", "CT": "C-T", "BBM": "BB-M", "BBC": "BB-C", "BBT": "BB-T",
             "D": "D (distillation)", "E": "E (naive merge)", "ES": "E-shared", "E0": "E0 (zero-training merge)"}
SPU_LOCAL = {"10x5": 0.85, "20x10": 4.5}           # measured MO-DANIEL s/update (pinned, local); scaled by calibration
CL_ORDER = ["CLB", "CLD", "CLE"]
CL_LABEL = {"CLB": "CL-B (contrastive, scratch)", "CLD": "CL-D (contrastive from D's update-0 network)",
            "CLE": "CL-E (contrastive from the naive merge)"}
SPU_LOCAL_CL = {"10x5": 2.0}                       # contrastive s/update (K = 8): 1.7-1.8 s measured, 3 parallel smoke jobs
MC_ORDER = ["SBAVG", "TA", "TIES"]
MC_LABEL = {"SBAVG": "SB-AVG (shared-base average, contrastive)", "TA": "TA (Task Arithmetic, contrastive)",
            "TIES": "TIES (TIES-Merging, contrastive)"}
CKPT = REPO / "checkpoints" / "merge_categories_v3"     # checkpoints/merge_categories_v3_smoke under --smoke
CC_ORDER = ["CCD", "CCB", "CCE", "CCTA", "CCTIES", "CCSBAVG"]   # longest expected first (three jobs share the GPU)
CC_LABEL = {"CCB": "From scratch (contrastive, to convergence)", "CCD": "Distillation (contrastive, to convergence)",
            "CCE": "Naive weight average (contrastive, to convergence)",
            "CCSBAVG": "Model Soup (shared base) (contrastive, to convergence)",
            "CCTA": "Task Arithmetic (contrastive, to convergence)", "CCTIES": "TIES (contrastive, to convergence)"}
MF_ORDER = ["MFSP", "MFNA", "MFPR"]
MF_LABEL = {"MFSP": "Shrink and perturb (from Model Soup)", "MFNA": "No anchor (from Model Soup)",
            "MFPR": "Random preference input (from Model Soup)"}
MF_DIAG = ["SOUP", "DIST", "RAND", "MFSP", "MFNA", "MFPR"]


def settings(smoke):
    s = dict(validate=50, window=1000, tol=0.005, min_updates=1000, cap=8000, chunk={"10x5": 500, "20x10": 200},
             val_grid=0.25, val_first=0, eval_split="evaluation", eval_first=0, eval_grid=0.1, sample=16,
             sample_first={"10x5": 0, "20x10": 20},          # 20x10: best-of-16 on 100 instances costs ~50 min per run
             norm_first=0, norm_limit=60, norm_lex=20, distill=[], base_updates=500, ft_updates=1000, calib_updates=200)
    if smoke:
        s.update(validate=10, window=20, min_updates=20, cap=40, chunk={"10x5": 20, "20x10": 20}, val_grid=0.5,
                 val_first=3, eval_split="development", eval_first=3, eval_grid=0.25, sample=2, norm_first=3,
                 norm_limit=2, norm_lex=1, distill=["--instances", "6", "--epochs", "2", "--val_first", "3"],
                 base_updates=20, ft_updates=20, calib_updates=20)
    return s


class Runner:
    def __init__(self, a):
        self.a = a
        self.s = settings(a.smoke)
        self.sizes = a.sizes.split()
        self.seeds = [int(x) for x in a.seeds.split()]
        arms = ARM_ORDER if a.arms == "all" else [x.replace("-", "") for x in a.arms.replace(",", " ").split()]
        self.cc = any(x in CC_ORDER for x in arms)
        self.mc = any(x in MC_ORDER for x in arms)
        self.mf = any(x in MF_ORDER for x in arms)
        self.cl = self.cc or self.mc or self.mf or any(x in CL_ORDER for x in arms)
        if self.mf:
            self.init_failure(a, arms)
            return
        if self.cc:
            self.init_converged(a, arms)
            return
        if self.mc:
            self.init_merge_categories(a, arms)
            return
        if self.cl:
            self.init_contrastive(a, arms)
            return
        self.e0 = a.arms == "all" or "E0" in arms
        arms = [x for x in arms if x != "E0"]
        if a.with_shared_base and "ES" not in arms:
            arms = arms + ["ES"]
        bad = set(arms) - set(ARM_ORDER) - {"ES"}
        if bad:
            raise SystemExit(f"unknown arms {sorted(bad)}")
        self.arms = [x for x in ARM_ORDER + ["ES"] if x in arms]
        self.lock = threading.Lock()
        self.git_lock = threading.Lock()
        self.children = set()
        self.stop = False
        self.state = {}                                     # unit -> PENDING/RUNNING/DONE/FAILED
        self.started = time.time()
        self.log_path = Path(a.log)
        ncpu = os.cpu_count() or 4
        self.lane_cpus = max(2, ncpu // (a.jobs + 1)) if ncpu >= 4 else 1
        train_cpus = max(1, ncpu - self.lane_cpus)
        per = max(1, train_cpus // a.jobs)
        self.slots = [f"{i * per}-{i * per + per - 1}" for i in range(a.jobs)]
        self.slot_threads = per
        self.lane = f"{ncpu - self.lane_cpus}-{ncpu - 1}"
        self.free_slots = list(range(a.jobs))
        RES.mkdir(parents=True, exist_ok=True); TN.mkdir(parents=True, exist_ok=True); TL.mkdir(parents=True, exist_ok=True)

    def init_contrastive(self, a, arms):
        """CL mode: only CL arms; budgets, seed and validation normalisation from the verified frozen manifest."""
        bad = set(arms) - set(CL_ORDER)
        if bad:
            raise SystemExit(f"contrastive arms cannot be mixed with {sorted(bad)} in one call")
        if a.with_shared_base:
            raise SystemExit("--with-shared-base does not apply to the contrastive arms")
        sys.path.insert(0, str(REPO / "tools"))
        from freeze_contrastive_v3 import load
        self.manifest = load()                              # fails loudly on a hash mismatch
        if self.sizes != [self.manifest["size"]] or (not a.smoke and self.seeds != [self.manifest["seed"]]):
            raise SystemExit(f"contrastive arms are frozen for size {self.manifest['size']} and seed "
                             f"{self.manifest['seed']} (configs/contrastive_v3.json)")
        self.arms, self.e0 = [x for x in CL_ORDER if x in arms], False
        self.cl_common(a)

    def init_merge_categories(self, a, arms):
        """Merge-category mode: SB-AVG, TA and TIES, contrastive fine-tuning; inputs from the verified frozen manifest."""
        bad = set(arms) - set(MC_ORDER)
        if bad:
            raise SystemExit(f"merge-category arms cannot be mixed with {sorted(bad)} in one call")
        if a.with_shared_base:
            raise SystemExit("--with-shared-base does not apply to the merge-category arms (they build their own)")
        sys.path.insert(0, str(REPO / "tools"))
        from freeze_merge_categories_v3 import load
        m = load()                                          # fails loudly on a hash mismatch
        if self.sizes != [m["size"]] or (not a.smoke and self.seeds != [m["seed"]]):
            raise SystemExit(f"merge-category arms are frozen for size {m['size']} and seed {m['seed']} "
                             "(configs/merge_categories_v3.json)")
        sb = m["method"]["shared_base"]
        if not a.smoke and (self.s["base_updates"], self.s["ft_updates"]) != (sb["base_updates"], sb["ft_updates"]):
            raise SystemExit("shared-base budgets differ from configs/merge_categories_v3.json")
        self.mcm = m
        self.manifest = {"size": m["size"], "seed": m["seed"], "constants": m["contrastive"]["constants"],
                         "validation_norm": m["validation_norm"],
                         "arms": {x: {"updates": m["budget"]["updates"]} for x in MC_ORDER}}
        self.lams = [0.5, 1.0] if a.smoke else m["method"]["lambda_grid"]
        self.arms, self.e0 = [x for x in MC_ORDER if x in arms], False
        self.ckpt_lock = threading.Lock()
        self.cl_common(a)

    def init_converged(self, a, arms):
        """Converged mode: the six contrastive arms under the PPO stopping rule; inputs from the verified manifest."""
        bad = set(arms) - set(CC_ORDER)
        if bad:
            raise SystemExit(f"converged contrastive arms cannot be mixed with {sorted(bad)} in one call")
        if a.with_shared_base:
            raise SystemExit("--with-shared-base does not apply to the converged contrastive arms")
        sys.path.insert(0, str(REPO / "tools"))
        from freeze_contrastive_converged_v3 import load
        m = load()                                          # fails loudly on a hash mismatch
        if self.sizes != [m["size"]] or self.seeds != [m["seed"]]:
            raise SystemExit(f"converged arms are frozen for size {m['size']} and seed {m['seed']} "
                             "(configs/contrastive_converged_v3.json)")
        self.ccm, self.cc_mode = m, {}
        self.manifest = {"size": m["size"], "seed": m["seed"], "constants": m["contrastive"]["constants"],
                         "validation_norm": m["validation_norm"], "arms": {x: {"updates": m["rule"]["cap"]} for x in CC_ORDER}}
        self.arms, self.e0 = [x for x in CC_ORDER if x in arms], False
        self.ckpt_lock = threading.Lock()
        self.cl_common(a)

    def init_failure(self, a, arms):
        """Merge-failure mode: update-0 diagnostics and three one-change arms from the Model Soup start, under the
        converged runs' stopping rule; inputs from the verified frozen manifests."""
        bad = set(arms) - set(MF_ORDER)
        if bad:
            raise SystemExit(f"merge-failure arms cannot be mixed with {sorted(bad)} in one call")
        if a.with_shared_base:
            raise SystemExit("--with-shared-base does not apply to the merge-failure arms")
        sys.path.insert(0, str(REPO / "tools"))
        from freeze_merge_failure_test_v3 import load
        from freeze_contrastive_converged_v3 import load as load_cc
        self.mfm, self.ccm, self.cc_mode = load(), load_cc(), {}   # both fail loudly on a hash mismatch
        m = self.mfm
        if self.sizes != [m["size"]] or self.seeds != [m["seed"]]:
            raise SystemExit(f"merge-failure arms are frozen for size {m['size']} and seed {m['seed']} "
                             "(configs/merge_failure_test_v3.json)")
        self.manifest = {"size": m["size"], "seed": m["seed"], "constants": m["contrastive"]["constants"],
                         "validation_norm": m["validation_norm"], "arms": {x: {"updates": m["rule"]["cap"]} for x in MF_ORDER}}
        self.arms, self.e0 = [x for x in MF_ORDER if x in arms], False
        self.ckpt_lock = threading.Lock()
        self.cl_common(a)

    def cl_common(self, a):
        self.s.update(cl_validate=self.manifest["constants"]["validate_timestep"])
        if a.smoke:
            self.s.update(cl_validate=5)
        self.lock, self.git_lock = threading.Lock(), threading.Lock()
        self.children, self.stop, self.state, self.started = set(), False, {}, time.time()
        self.log_path = Path(a.log)
        ncpu = os.cpu_count() or 4
        self.lane_cpus = max(2, ncpu // (a.jobs + 1)) if ncpu >= 4 else 1
        per = max(1, (ncpu - self.lane_cpus) // a.jobs)
        self.slots = [f"{i * per}-{i * per + per - 1}" for i in range(a.jobs)]
        self.slot_threads = per
        self.lane = f"{ncpu - self.lane_cpus}-{ncpu - 1}"
        self.free_slots = list(range(a.jobs))
        RES.mkdir(parents=True, exist_ok=True); TN.mkdir(parents=True, exist_ok=True); TL.mkdir(parents=True, exist_ok=True)

    def cap(self, arm):
        if not self.cl:
            return self.s["cap"]
        if self.cc and self.a.smoke:                        # 5 updates beyond the start of the (continued) run
            src = self.ccm["arms"][arm]
            return (src["fixed_budget"] if self.cc_mode.get(arm, src["mode"]) == "continue" else 0) + 5
        return 5 if self.a.smoke else self.manifest["arms"][arm]["updates"]

    def chunk(self, size):
        if self.cl and self.a.smoke:
            return 0                                        # one process for the 5-update smoke runs
        return self.s["chunk"][size]

    # ------------------------------------------------------------------ helpers
    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self.lock:
            print(line, flush=True)

    def run(self, cmd, timeout, cpus, threads, cwd=REPO, out=None):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
        full = ["taskset", "-c", cpus] + cmd
        with open(out or os.devnull, "a") as fh:
            p = subprocess.Popen(full, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            with self.lock:
                self.children.add(p)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                rc = 124
            finally:
                with self.lock:
                    self.children.discard(p)
        return rc

    def commit(self, msg, *paths):
        with self.git_lock:
            for p in paths:
                if Path(p).exists():
                    subprocess.run(["git", "add", "-A", "--", str(p)], cwd=REPO, capture_output=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO, capture_output=True).returncode == 1:
                subprocess.run(["git", "commit", "-q", "-m", msg, "-m",
                                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], cwd=REPO, capture_output=True)
            with open(self.log_path, "a") as fh:
                fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] commit: {msg}\n")

    def spu(self, size):
        base = SPU_LOCAL_CL[size] if self.cl else SPU_LOCAL[size]
        f = RES / f"calibration_{size}.json"
        if f.exists():
            c = json.loads(f.read_text())
            return base * max(1.0, c["factor"]) * max(1, self.a.jobs)
        return base * 2 * max(1, self.a.jobs)

    def host(self):
        def q(cmd):
            try:
                return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
            except Exception:
                return "unknown"
        return {"gpu": q(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
                "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
                "logical_cpus": os.cpu_count(), "jobs": self.a.jobs, "python": sys.version.split()[0],
                "torch": q([PY, "-c", "import torch; print(torch.__version__, torch.version.cuda)"]),
                "ortools": q([PY, "-c", "import ortools; print(ortools.__version__)"])}

    # ------------------------------------------------------------------ names and paths
    def norm_file(self, size, split="validation"):
        if self.cl and split == "validation":
            return REPO / self.manifest["validation_norm"]["path"]    # frozen: the PPO arms' validation points
        tag = "" if split == "validation" else f"_{split}"
        return RES / "norm" / f"validation_normalisation_points{tag}_{size}.json"

    @staticmethod
    def name(arm, size, seed):
        return f"{size}+mix+bv3_{TAG}_{arm}_s{seed}"

    def spec(self, size, o, seed):
        return SPEC / f"{size}+mix+bv3_spec_{o}_s{seed}.pth"

    def eval_norm(self, size):
        if self.s["eval_split"] == "evaluation":
            return [str(REPO / "results" / "baselines_v3" / "normalisation_points.json")]
        return [str(self.norm_file(size, self.s["eval_split"]))]

    # ------------------------------------------------------------------ units
    def u_calibration(self, size, slot):
        f = RES / f"calibration_{size}.json"
        if f.exists():
            return True
        name = f"{size}+mix+bv3_{TAG}calib_m_s111"
        nj, nm = size.split("x")
        for p in (TN / f"{name}.pth", TN / f"{name}.resume.pth", TL / f"{name}.status.json"):
            p.unlink(missing_ok=True)
        n = self.s["calib_updates"]
        rc = self.run([PY, "train.py", "--config", "../configs/benchmark_v3/m.json", "--n_j", nj, "--n_m", nm,
                       "--model_suffix", f"bv3_{TAG}calib_m_s111", "--seed_train", "111", "--v3_train_source", "generator",
                       "--validate_timestep", "50" if not self.a.smoke else "10", "--converge_window", "1000",
                       "--min_updates", "1000", "--max_updates", str(n), "--device", "cuda"],
                      timeout=n * SPU_LOCAL[size] * 20 + 900, cpus=self.slots[slot], threads=self.slot_threads,
                      cwd=REPO / "daniel", out=TL / f"{name}.stdout.log")
        if rc != 0:
            return False
        st = json.loads((TL / f"{name}.status.json").read_text())
        ec2 = st["train_seconds"] / st["update"]
        local = []
        for s in (111, 222, 333):
            ff = REPO / "results" / "baselines_v3" / "runs" / f"{size}+mix+bv3_spec_m_s{s}" / "status.json"
            if ff.exists():
                x = json.loads(ff.read_text()); local.append(x["train_seconds"] / x["update"])
        loc = sum(local) / len(local)
        rec = {"size": size, "updates": st["update"], "ec2_seconds_per_update": ec2,
               "local_seconds_per_update": loc, "local_source": "results/baselines_v3/runs/*_spec_m_s*/status.json",
               "factor": ec2 / loc, "host": self.host(), "slot_cpus": self.slots[slot],
               "note": "DANIEL-M, same flags as the specialists; factor converts local specialist training time to "
                       "this machine (full-lifecycle cost view)"}
        f.write_text(json.dumps(rec, indent=1))
        self.commit(f"{PREFIX}: calibration {size}", f)
        self.log(f"calibration {size}: {ec2:.3f} s/update here vs {loc:.3f} locally (factor {ec2 / loc:.2f})")
        return True

    def u_norm(self, size, split="validation"):
        f = self.norm_file(size, split)
        if f.exists():
            return True
        f.parent.mkdir(parents=True, exist_ok=True)
        n = self.s["norm_first"] or {"10x5": 18, "20x10": 13}[size]
        tmo = n * (2 * self.s["norm_limit"] + 6 * self.s["norm_lex"]) * 3 // 2 + 1200
        rc = self.run([PY, "tools/mo_v3_normalisation.py", "--size", size, "--split", split,
                       "--first", str(self.s["norm_first"]), "--limit", str(self.s["norm_limit"]),
                       "--lex_limit", str(self.s["norm_lex"]), "--workers", str(self.lane_cpus),
                       "--out_dir", str(f.parent), "--specialist_dir", str(SPEC)],
                      timeout=tmo, cpus=self.lane, threads=self.lane_cpus, out=self.log_path)
        ok = rc == 0 and f.exists()
        self.commit(f"{PREFIX}: frozen {split} normalisation points {size}", RES / "norm")
        return ok

    def u_distill(self, size, seed, slot):
        out = TN / f"{size}+mix+bv3_{TAG}D_student_s{seed}.pth"
        rec = RES / "distill" / f"{out.stem}.distill.json"
        if out.exists() and rec.exists():
            return True
        rc = self.run([PY, "distill_v3.py", "--size", size, "--seed", str(seed), "--spec_dir", str(SPEC),
                       "--out", str(out), "--device", "cuda"] + self.s["distill"],
                      timeout=2400, cpus=self.slots[slot], threads=self.slot_threads, cwd=REPO / "daniel",
                      out=TL / f"{out.stem}.stdout.log")
        if rc != 0 or not out.exists():
            return False
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(out.with_suffix(".distill.json").read_text())
        self.commit(f"{PREFIX}: distillation {size} s{seed}", rec)
        return True

    def u_merge(self, size, seed, slot, shared=False):
        out = TN / f"{size}+mix+bv3_{TAG}{'ES' if shared else 'E'}_merged_s{seed}.pth"
        if out.exists():
            return True
        src = ([str(TN / f"{size}+mix+bv3_{TAG}ft_{o}_s{seed}.pth") for o in "mct"] if shared else
               [str(self.spec(size, o, seed)) for o in "mct"])
        rc = self.run([PY, "tools/merge_specialists_v3.py", "--sources", *src, "--out", str(out)]
                      + (["--shared_base"] if shared else []), timeout=600, cpus=self.slots[slot],
                      threads=self.slot_threads, out=self.log_path)
        return rc == 0 and out.exists()

    def train_chunks(self, name, args, size, slot, cap, chunk, rdir):
        live = TL / f"{name}.status.json"
        fails = 0
        tmo = int(chunk * self.spu(size) * 3 + 600) if not os.environ.get("WS_CHUNK_TIMEOUT") else int(os.environ["WS_CHUNK_TIMEOUT"])
        while not self.stop:
            st = json.loads(live.read_text())["status"] if live.exists() else "NONE"
            if st in ("CONVERGED", "NOT_CONVERGED", "FINISHED"):
                return True
            self.log(f"train {name}: chunk of {chunk} (status {st}, slot {slot}, timeout {tmo}s)")
            t0 = time.time()
            rc = self.run([PY, "train.py", *args, "--max_updates", str(cap), "--chunk_updates", str(chunk),
                           "--resume", "True", "--device", "cuda"], timeout=tmo, cpus=self.slots[slot],
                          threads=self.slot_threads, cwd=REPO / "daniel", out=TL / f"{name}.stdout.log")
            meta = rdir / "run_meta.json"
            m = json.loads(meta.read_text()) if meta.exists() else {}
            m["process_seconds"] = m.get("process_seconds", 0.0) + time.time() - t0
            m.setdefault("chunk_rc", []).append(rc)
            meta.write_text(json.dumps(m, indent=1))
            if rc != 0:
                fails += 1
                self.log(f"train {name}: chunk FAILED (rc {rc}{', hang guard timeout' if rc == 124 else ''}), attempt {fails}")
                if fails >= 2:
                    return False
            else:
                fails = 0
        return False

    def mo_args(self, arm, size, seed):
        if self.mf:
            return self.mf_args(arm, size, seed)
        if self.cc:
            return self.cc_args(arm, size, seed, self.cc_mode[arm])
        if self.cl:
            return self.cl_args(arm, size, seed)
        nj, nm = size.split("x")
        s = self.s
        a = ["--config", "../configs/benchmark_v3/mct.json", "--n_j", nj, "--n_m", nm, "--preference_dim", "3",
             "--log_ppo_stats", "True", "--mo_val_norm", str(self.norm_file(size)), "--mo_val_grid_step", str(s["val_grid"]),
             "--mo_val_first", str(s["val_first"]), "--model_suffix", f"bv3_{TAG}_{arm}_s{seed}", "--seed_train", str(seed),
             "--v3_train_source", "generator", "--validate_timestep", str(s["validate"]),
             "--converge_window", str(s["window"]), "--converge_tol", str(s["tol"]), "--min_updates", str(s["min_updates"])]
        init = {"CM": ("full", self.spec(size, "m", seed)), "CC": ("full", self.spec(size, "c", seed)),
                "CT": ("full", self.spec(size, "t", seed)), "BBM": ("backbone", self.spec(size, "m", seed)),
                "BBC": ("backbone", self.spec(size, "c", seed)), "BBT": ("backbone", self.spec(size, "t", seed)),
                "D": ("no_critic", TN / f"{size}+mix+bv3_{TAG}D_student_s{seed}.pth"),
                "E": ("full", TN / f"{size}+mix+bv3_{TAG}E_merged_s{seed}.pth"),
                "ES": ("full", TN / f"{size}+mix+bv3_{TAG}ES_merged_s{seed}.pth")}.get(arm)
        if init:
            a += ["--init_from", str(init[1]), "--init_mode", init[0]]
        return a

    def cl_args(self, arm, size, seed):
        nj, nm = size.split("x")
        s, c = self.s, self.manifest["constants"]
        a = ["--config", "../configs/benchmark_v3/mct.json", "--n_j", nj, "--n_m", nm, "--preference_dim", "3",
             "--mo_val_norm", str(self.norm_file(size)), "--mo_val_grid_step", str(c["mo_val_grid_step"]),
             "--mo_val_first", str(s["val_first"]), "--model_suffix", f"bv3_{TAG}_{arm}_s{seed}", "--seed_train", str(seed),
             "--v3_train_source", "generator", "--validate_timestep", str(s["cl_validate"]), "--converge_window", "0",
             "--finetune_objective", "contrastive", "--cl_k", str(c["cl_k"]), "--cl_beta", str(c["cl_beta"]),
             "--cl_margin", str(c["cl_margin"]), "--cl_grad_clip", str(c["cl_grad_clip"]), "--lr", str(c["lr"]),
             "--minibatch_size", str(c["minibatch_size"]), "--num_envs", str(c["num_envs"]),
             "--reset_env_timestep", str(c["reset_env_timestep"]), "--pref_dirichlet", str(c["pref_dirichlet"])]
        init = {"CLD": ("no_critic", TN / f"{size}+mix+bv3_{TAG}D_student_s{seed}.pth"),
                "CLE": ("full", TN / f"{size}+mix+bv3_{TAG}E_merged_s{seed}.pth"),
                **{x: ("full", self.start(x, size, seed)) for x in MC_ORDER}}.get(arm)
        if init:
            a += ["--init_from", str(init[1]), "--init_mode", init[0]]
        return a

    def u_shared_base(self, size, seed, slot):
        """E-shared preparation: equal-weight unconditioned base, then M/C/T fine-tunes from it (fixed budgets)."""
        nj, nm = size.split("x")
        rdir = RES / "shared_base" / f"{size}_s{seed}"; rdir.mkdir(parents=True, exist_ok=True)
        steps = [("base", "mct", f"bv3_{TAG}base_s{seed}", self.s["base_updates"], [])]
        steps += [(f"ft_{o}", o, f"bv3_{TAG}ft_{o}_s{seed}", self.s["ft_updates"],
                   ["--init_from", str(TN / f"{size}+mix+bv3_{TAG}base_s{seed}.pth")]) for o in "mct"]
        for tag, goal, suffix, n, extra in steps:
            name = f"{size}+mix+{suffix}"
            if (rdir / f"{tag}.status.json").exists():
                continue
            ok = self.train_chunks(name, ["--config", f"../configs/benchmark_v3/{goal}.json", "--n_j", nj, "--n_m", nm,
                                          "--model_suffix", suffix, "--seed_train", str(seed), "--v3_train_source",
                                          "generator", "--validate_timestep", str(self.s["validate"]), *extra],
                                   size, slot, n, n, rdir)
            if not ok:
                return False
            for f, dst in ((TL / f"{name}.status.json", f"{tag}.status.json"),
                           (TL / f"valiquality_{name}.json", f"{tag}.valiquality.json")):
                (rdir / dst).write_text(f.read_text())
        self.commit(f"{PREFIX}: shared base and fine-tunes {size} s{seed}", rdir)
        return self.u_merge(size, seed, slot, shared=True)

    def u_train(self, arm, size, seed, slot):
        name = self.name(arm, size, seed)
        rdir = RES / "runs" / name
        edir = RES / "eval" / f"{self.s['eval_split']}_{size}"
        if (rdir / "status.json").exists() and (edir / f"{name}.json").exists():
            return True
        rdir.mkdir(parents=True, exist_ok=True); (rdir / "FAILED").unlink(missing_ok=True)
        if self.cc and not self.cc_prepare(arm, size, seed, slot, rdir):
            return False
        m = json.loads((rdir / "run_meta.json").read_text()) if (rdir / "run_meta.json").exists() else {}
        m.update(arm=arm, label={**ARM_LABEL, **CL_LABEL, **MC_LABEL, **CC_LABEL, **MF_LABEL}[arm], size=size, seed=seed, slot_cpus=self.slots[slot],
                 omp_threads=self.slot_threads, host=self.host())
        (rdir / "run_meta.json").write_text(json.dumps(m, indent=1))
        if not self.train_chunks(name, self.mo_args(arm, size, seed), size, slot, self.cap(arm), self.chunk(size), rdir):
            if not self.stop:
                (rdir / "FAILED").write_text(f"training failed twice at {time.strftime('%F %T')}\n")
                self.commit(f"{PREFIX}: {name} FAILED", rdir)
            return False
        for f, dst in ((TL / f"valiquality_{name}.json", "valiquality.json"), (TL / f"reward_{name}.json", "reward.json"),
                       (TL / f"{name}.run.json", "run.json"), (TL / f"{name}.status.json", "status.json")):
            (rdir / dst).write_text(f.read_text())
        if (TL / f"{name}.extension.json").exists():
            (rdir / "extension.json").write_text((TL / f"{name}.extension.json").read_text())
        st = json.loads((rdir / "status.json").read_text())
        self.log(f"{name}: {st['status']} at update {st['update']}, best val HV {-st['best_value']:.4f} "
                 f"@ {st['best_update']}, train {st['train_seconds'] / 3600:.2f} h")
        if not self.evaluate(name, size, slot):
            (rdir / "FAILED").write_text(f"evaluation failed twice at {time.strftime('%F %T')}\n")
            self.commit(f"{PREFIX}: {name} evaluation FAILED", rdir)
            return False
        if self.mf:                                         # the run's contrastive reference (= its start network)
            self.reference_record(name, rdir)
        if self.mc or self.cc or self.mf:
            self.save_ckpt(TN / f"{name}.pth")              # the best-validation checkpoint
        self.commit(f"{PREFIX}: {name} ({st['status']})", rdir, edir, STATUS_FILE,
                    *([CKPT] if self.mc or self.cc or self.mf else []))
        return True

    # ------------------------------------------------------------------ converged contrastive arms
    def cc_args(self, arm, size, seed, mode, suffix=None):
        """The CL arms' flags plus the stopping rule; continue: the source run's identity strings and --resume_from;
        rerun: the source's starting network."""
        src, rule = self.ccm["arms"][arm], self.ccm["rule"]
        a = self.cl_args("-", size, seed)
        def put(flag, value):
            a[a.index(flag) + 1] = str(value)
        put("--model_suffix", suffix or f"bv3_{TAG}_{arm}_s{seed}")
        put("--converge_window", rule["converge_window"])
        a += ["--converge_tol", str(rule["converge_tol"]), "--min_updates", str(rule["min_updates"])]
        si = src["source_init"]
        if mode == "continue":                              # identity strings exactly as in the source run
            put("--mo_val_norm", si["mo_val_norm"]); put("--mo_val_first", si["mo_val_first"])
            put("--validate_timestep", self.manifest["constants"]["validate_timestep"])
            if si["init_from"]:
                a += ["--init_from", si["init_from"], "--init_mode", si["init_mode"]]
            a += ["--resume_from", src["resume_from"]]
        else:
            start = self.rerun_start(arm)
            if start:
                a += ["--init_from", str(start), "--init_mode", src["rerun_start"]["init_mode"]]
        return a

    def rerun_start(self, arm):
        rs = self.ccm["arms"][arm]["rerun_start"]
        if not rs.get("path"):
            return None                                     # random initialisation, seed 111
        for cand in (REPO / rs["path"], Path(self.ccm["arms"][arm]["source_init"]["init_from"] or "/nonexistent")):
            if cand.exists() and hashlib.sha256(cand.read_bytes()).hexdigest() == rs["sha256"]:
                return cand
        raise RuntimeError(f"starting network of {arm} not found with its frozen SHA-256")

    def cc_prepare(self, arm, size, seed, slot, rdir):
        """Decide continue/rerun once per arm (mode.json): a continued arm must first pass the exactness check in a
        0-update probe (rebuilt instance batch, last validation HV reproduced); if it cannot (exit code 3), it reruns
        from its starting network. The source files used are copied to the committed checkpoint folder."""
        f = rdir / "mode.json"
        if f.exists():
            self.cc_mode[arm] = json.loads(f.read_text())["mode"]
            return True
        src = self.ccm["arms"][arm]
        rec = {"planned": src["mode"], "reason": src["reason"], "mode": src["mode"]}
        if src["mode"] == "continue":
            probe = f"bv3_{TAG}_{arm}chk_s{seed}"; pname = f"{size}+mix+{probe}"
            rc = self.run([PY, "train.py", *self.cc_args(arm, size, seed, "continue", suffix=probe), "--resume", "True",
                           "--max_updates", str(src["fixed_budget"]), "--chunk_updates", "0", "--device", "cuda"],
                          timeout=1800, cpus=self.slots[slot], threads=self.slot_threads, cwd=REPO / "daniel",
                          out=TL / f"{pname}.stdout.log")
            ext = TL / f"{pname}.extension.json"
            if ext.exists():
                (rdir / "extension_check.json").write_text(ext.read_text())
            for p in list(TN.glob(f"{pname}*")) + list(TL.glob(f"*{pname}*")):
                p.unlink(missing_ok=True)
            if rc == 3:
                rec.update(mode="rerun", fallback="the saved final state could not be resumed exactly (exit code 3); "
                                                  "rerun from the starting network")
                self.log(f"{arm}: not exactly resumable; rerunning from its starting network")
            elif rc != 0:
                self.log(f"{arm}: extension check FAILED (rc {rc})")
                return False
            else:
                self.save_ckpt(Path(src["resume_from"]), sub="sources")
                self.log(f"{arm}: extension check passed; continuing from update {src['fixed_budget']}")
        if rec["mode"] == "rerun" and self.rerun_start(arm):
            self.save_ckpt(self.rerun_start(arm), sub="sources")
        f.write_text(json.dumps(rec, indent=1) + "\n")
        self.cc_mode[arm] = rec["mode"]
        return True

    def evaluate(self, model, size, slot):
        """daniel/evaluate_mo_v3.py on the evaluation split (model: a run name or a .pth path), retried once."""
        name = Path(model).stem
        for attempt in (1, 2):
            rc = self.run([PY, "evaluate_mo_v3.py", "--model", str(model), "--split", self.s["eval_split"], "--size", size,
                           "--first", str(self.s["eval_first"]), "--grid_step", str(self.s["eval_grid"]),
                           "--norm", *self.eval_norm(size), "--out", str(RES / "eval"), "--sample", str(self.s["sample"]),
                           "--sample_first", str(self.s["sample_first"][size]),
                           "--device", "cuda"], timeout=7200, cpus=self.slots[slot], threads=self.slot_threads,
                          cwd=REPO / "daniel", out=TL / f"{name}.stdout.log")
            if rc == 0:
                return True
            self.log(f"eval {name} FAILED (attempt {attempt})")
        return False

    # ------------------------------------------------------------------ merge categories
    def start(self, arm, size, seed):
        """The unconditioned starting network of a merge-category arm (written by u_lambda)."""
        return TN / f"{size}+mix+bv3_{TAG}{arm}_start_s{seed}.pth"

    def save_ckpt(self, src, sub=""):
        """Copy a checkpoint to the committed folder and record its SHA-256 in SHA256SUMS.json."""
        with self.ckpt_lock:
            (CKPT / sub).mkdir(parents=True, exist_ok=True)
            dst = CKPT / sub / Path(src).name
            shutil.copy2(src, dst)
            f = CKPT / "SHA256SUMS.json"
            rec = json.loads(f.read_text()) if f.exists() else {"description": "SHA-256 of every committed checkpoint of "
                                                                f"the {PREFIX} study", "files": {}}
            rec["files"][str(dst.relative_to(CKPT))] = hashlib.sha256(dst.read_bytes()).hexdigest()
            rec["files"] = dict(sorted(rec["files"].items()))
            f.write_text(json.dumps(rec, indent=1) + "\n")

    def probe(self, label, init, size, seed, slot):
        """A 0-update contrastive run from `init` (--init_mode full): validation at update 0 only (the CL arms' validation
        grid and frozen points) and the update-0 network saved as <name>.pth. Returns the validation HV or None."""
        name = self.name(label, size, seed)
        for p in (TN / f"{name}.pth", TL / f"{name}.status.json", TL / f"valiquality_{name}.json"):
            p.unlink(missing_ok=True)
        rc = self.run([PY, "train.py", *self.cl_args(label, size, seed), "--init_from", str(init), "--init_mode", "full",
                       "--max_updates", "0", "--device", "cuda"], timeout=1800, cpus=self.slots[slot],
                      threads=self.slot_threads, cwd=REPO / "daniel", out=TL / f"{name}.stdout.log")
        v = TL / f"valiquality_{name}.json"
        if rc != 0 or not v.exists() or not (TN / f"{name}.pth").exists():
            self.log(f"probe {name} FAILED (rc {rc})")
            return None
        rec = json.loads(v.read_text())
        zdir = RES / "zero" / name; zdir.mkdir(parents=True, exist_ok=True)
        (zdir / "valiquality.json").write_text(v.read_text())
        (zdir / "run.json").write_text((TL / f"{name}.run.json").read_text())
        return float(rec[0]["hv"])

    def u_lambda(self, size, seed, slot):
        """TA and TIES candidates for every lambda of the frozen grid, scored by update-0 validation HV only; the best
        (tie -> smaller lambda) becomes the arm's start. SB-AVG's start is the shared-base average; E's merge is probed
        for its zero-training value. Writes lambda_selection.json and commits every starting network."""
        out = RES / "lambda_selection.json"
        if out.exists() and all(self.start(x, size, seed).exists() for x in MC_ORDER):
            return True
        pre = f"{size}+mix+bv3_{TAG}"
        base = TN / f"{pre}base_s{seed}.pth"
        fts = [TN / f"{pre}ft_{o}_s{seed}.pth" for o in "mct"]
        rec = {"size": size, "seed": seed, "lambda_grid": self.lams, "ties_keep": self.mcm["method"]["ties_keep"],
               "rule": self.mcm["method"]["lambda_rule"], "base": base.name, "specialists": [f.name for f in fts],
               "validation_norm": self.manifest["validation_norm"], "candidates": {}, "selected": {}}
        for arm, method in (("TA", "ta"), ("TIES", "ties")):
            cands = []
            for lam in self.lams:
                tag = f"{arm}l{int(round(lam * 100)):03d}"
                pth = TN / f"{pre}{tag}_merged_s{seed}.pth"
                rc = self.run([PY, "tools/merge_task_vectors_v3.py", "--method", method, "--base", str(base), "--sources",
                               *map(str, fts), "--lam", str(lam), "--keep", str(rec["ties_keep"]), "--out", str(pth)],
                              timeout=600, cpus=self.slots[slot], threads=self.slot_threads, out=self.log_path)
                hv = self.probe(f"p{tag}", pth, size, seed, slot) if rc == 0 and pth.exists() else None
                if hv is None:
                    return False
                cands.append({"lambda": lam, "merged": pth.name, "validation_hv_update0": hv})
                self.save_ckpt(pth)
                self.log(f"lambda {arm} {lam}: update-0 validation HV {hv:.4f}")
            best = sorted(cands, key=lambda c: (-c["validation_hv_update0"], c["lambda"]))[0]
            rec["candidates"][arm] = cands
            rec["selected"][arm] = best
            shutil.copy2(TN / best["merged"], self.start(arm, size, seed))
        shutil.copy2(TN / f"{pre}ES_merged_s{seed}.pth", self.start("SBAVG", size, seed))
        rec["selected"]["SBAVG"] = {"lambda": None, "merged": f"{pre}ES_merged_s{seed}.pth"}
        for f in [base, *fts, TN / f"{pre}ES_merged_s{seed}.pth", TN / f"{pre}E_merged_s{seed}.pth",
                  *(self.start(x, size, seed) for x in MC_ORDER)]:
            self.save_ckpt(f)
        rec["start_sha256"] = {x: hashlib.sha256(self.start(x, size, seed).read_bytes()).hexdigest() for x in MC_ORDER}
        out.write_text(json.dumps(rec, indent=1) + "\n")
        self.commit(f"{PREFIX}: lambda selection and starting networks {size} s{seed}", out, CKPT, RES / "zero")
        return True

    def u_zeval(self, arm, size, seed, slot):
        """Zero-training value of a merge (SBAVG, TA, TIES: the arm's start; E: arm E's naive merge): a 0-update probe,
        then the same evaluation as the fine-tuned arms."""
        name = self.name(f"Z{arm}", size, seed)
        edir = RES / "eval" / f"{self.s['eval_split']}_{size}"
        if (edir / f"{name}.json").exists():
            return True
        init = TN / f"{size}+mix+bv3_{TAG}E_merged_s{seed}.pth" if arm == "E" else self.start(arm, size, seed)
        hv = self.probe(f"Z{arm}", init, size, seed, slot)
        if hv is None or not self.evaluate(TN / f"{name}.pth", size, slot):
            return False
        self.log(f"zero-training {arm}: update-0 validation HV {hv:.4f}")
        self.commit(f"{PREFIX}: zero-training {arm} {size} s{seed}", edir, RES / "zero")
        return True

    def u_e0(self, size, seed, slot):
        edir = RES / "eval" / f"{self.s['eval_split']}_{size}"
        name = self.name("E0", size, seed)
        if (edir / f"{name}.json").exists():
            return True
        rc = self.run([PY, "tools/e0_merge_front_v3.py", "--size", size, "--seed", str(seed), "--split",
                       self.s["eval_split"], "--first", str(self.s["eval_first"]), "--grid_step", str(self.s["eval_grid"]),
                       "--spec_dir", str(SPEC), "--norm", *self.eval_norm(size), "--out", str(RES / "eval"), "--tag", TAG,
                       "--device", "cuda"], timeout=7200, cpus=self.slots[slot], threads=self.slot_threads,
                      out=self.log_path)
        if rc == 0:
            self.commit(f"{PREFIX}: E0 {size} s{seed}", edir)
        return rc == 0

    # ------------------------------------------------------------------ merge-failure test
    def mf_args(self, arm, size, seed, suffix=None):
        """The converged Model Soup run's flags (a rerun from its start under the rule) plus the arm's one change."""
        return self.cc_args("CCSBAVG", size, seed, "rerun", suffix=suffix or f"bv3_{TAG}_{arm}_s{seed}") + \
            list(self.mfm["arms"][arm]["flags"])

    def reference_record(self, name, rdir):
        """rdir/reference.json: the identity's cl_reference_sha256 from the run's resume state (unpickled in daniel/)."""
        f = TN / f"{name}.resume.pth"
        r = subprocess.run([PY, "-c", "import sys, json; sys.argv = ['r', '--device', 'cpu']; import params, torch, "
                            f"benchmark_v3; i = torch.load({str(f)!r}, map_location='cpu', weights_only=False)['identity']; "
                            "print(json.dumps({k: i.get(k) for k in ('cl_reference_sha256', 'cl_anchor', 'init_shrink', "
                            "'init_perturb', 'init_pref_columns')}))"], cwd=REPO / "daniel", capture_output=True,
                           text=True, timeout=600)
        if r.returncode == 0:
            (rdir / "reference.json").write_text(json.dumps(json.loads(r.stdout.strip().splitlines()[-1]), indent=1) + "\n")
        else:
            self.log(f"{name}: could not read the reference hash ({r.stderr.strip()[-200:]})")

    def diag_args(self, subj, size, seed, suffix):
        if subj in MF_ORDER:
            return self.mf_args(subj, size, seed, suffix)
        if subj == "SOUP":
            return self.cc_args("CCSBAVG", size, seed, "rerun", suffix=suffix)
        a = self.cl_args("-", size, seed)
        a[a.index("--model_suffix") + 1] = suffix
        if subj == "DIST":
            d = self.mfm["diagnostics"]["distillation_start"]
            a += ["--init_from", d["path"], "--init_mode", d["init_mode"]]
        return a                                            # RAND: the fresh seed-111 network

    def u_diag(self, size, seed, slot):
        """Part A: update-0 diagnostics of every subject (no training); each update-0 network is saved and committed
        (starts/), the arms' under the arm's run name."""
        ddir = RES / "diagnostics"; ddir.mkdir(parents=True, exist_ok=True)
        for subj in MF_DIAG:
            out = ddir / f"{subj}.json"
            if out.exists():
                continue
            suffix = f"bv3_{TAG}_A{subj}_s{seed}"; name = f"{size}+mix+{suffix}"
            for p in (TN / f"{name}@u0.pth", TL / f"{name}.run.json"):
                p.unlink(missing_ok=True)
            rc = self.run([PY, "merge_diagnostics.py", "--diag_out", str(out), "--diag_label",
                           self.mfm["diagnostics"]["subjects"][subj]["name"], "--diag_seed", str(seed), "--diag_save_start",
                           "1", "--", *self.diag_args(subj, size, seed, suffix), "--max_updates", "0", "--device", "cuda"],
                          timeout=3600, cpus=self.slots[slot], threads=self.slot_threads, cwd=REPO / "daniel",
                          out=TL / f"{name}.stdout.log")
            if rc != 0 or not out.exists():
                self.log(f"diagnostics {subj} FAILED (rc {rc})")
                out.unlink(missing_ok=True)
                return False
            start = TN / f"{name}@u0.pth"
            if subj in MF_ORDER:
                dst = TN / f"{self.name(subj, size, seed)}_start.pth"
                shutil.copy2(start, dst); start = dst
            self.save_ckpt(start, sub="starts")
            d = json.loads(out.read_text())
            self.log(f"diagnostics {subj}: update-0 validation HV {d['validation']['hv_update0']:.4f}, entropy "
                     f"{d['sampled']['entropy']:.3f}, distinct {d['sampled']['distinct_schedules_of_k']:.2f}/"
                     f"{d['sampled']['k']}, grad norm {d['gradient']['total']:.4f}")
        self.commit(f"{PREFIX}: update-0 diagnostics {size} s{seed}", ddir, CKPT)
        return True

    # ------------------------------------------------------------------ plan and scheduling
    def plan(self):
        """Ordered units: (key, kind, args, deps, uses_slot)."""
        U = []
        for size in self.sizes:
            U.append((f"calib:{size}", "calib", (size,), [], True))
        for size in self.sizes:
            if self.a.smoke:
                U.append((f"normdev:{size}", "normdev", (size,), [], False))
        if self.mc:
            return self.plan_mc(U)
        if self.mf:
            for size in self.sizes:
                dev = [f"normdev:{size}"] if self.a.smoke else []
                for seed in self.seeds:
                    U.append((f"diag:{size}:{seed}", "diag", (size, seed), dev, True))      # short; first
                    for arm in self.arms:
                        U.append((f"train:{arm}:{size}:{seed}", "train", (arm, size, seed), dev, True))
            return U
        if self.cc:
            for size in self.sizes:
                for seed in self.seeds:
                    for arm in self.arms:
                        U.append((f"train:{arm}:{size}:{seed}", "train", (arm, size, seed),
                                  [f"normdev:{size}"] if self.a.smoke else [], True))
            return U
            if not self.cl:                                     # CL: frozen validation points (manifest)
                U.append((f"norm:{size}", "norm", (size,), [], False))
        for size in self.sizes:
            for seed in self.seeds:
                for arm in self.arms:
                    deps = ([] if self.cl else [f"norm:{size}"]) + ([f"normdev:{size}"] if self.a.smoke else [])
                    if arm in ("D", "CLD"):
                        U.append((f"distill:{size}:{seed}", "distill", (size, seed), [], True))
                        deps.append(f"distill:{size}:{seed}")
                    if arm in ("E", "CLE"):
                        U.append((f"merge:{size}:{seed}", "merge", (size, seed), [], True))
                        deps.append(f"merge:{size}:{seed}")
                    if arm == "ES":
                        U.append((f"shared:{size}:{seed}", "shared", (size, seed), [f"norm:{size}"], True))
                        deps.append(f"shared:{size}:{seed}")
                    U.append((f"train:{arm}:{size}:{seed}", "train", (arm, size, seed), deps, True))
            for seed in self.seeds if self.e0 else []:
                deps = [f"normdev:{size}"] if self.a.smoke else []
                U.append((f"e0:{size}:{seed}", "e0", (size, seed), deps, True))
        return U

    def plan_mc(self, U):
        """Merge categories: shared base + fine-tunes + SB-AVG merge, E's merge, lambda selection, then the contrastive
        arms; the zero-training evaluations fill free slots after the arms."""
        for size in self.sizes:
            dev = [f"normdev:{size}"] if self.a.smoke else []
            for seed in self.seeds:
                lam = f"lambda:{size}:{seed}"
                U.append((f"shared:{size}:{seed}", "shared", (size, seed), [], True))
                U.append((f"merge:{size}:{seed}", "merge", (size, seed), [], True))
                U.append((lam, "lambda", (size, seed), [f"shared:{size}:{seed}", f"merge:{size}:{seed}"], True))
                for arm in self.arms:
                    U.append((f"train:{arm}:{size}:{seed}", "train", (arm, size, seed), [lam] + dev, True))
                for arm in self.arms + ["E"]:
                    U.append((f"zeval:{arm}:{size}:{seed}", "zeval", (arm, size, seed), [lam] + dev, True))
        return U

    def execute(self, kind, args, slot):
        fn = {"calib": lambda: self.u_calibration(args[0], slot), "norm": lambda: self.u_norm(args[0]),
              "normdev": lambda: self.u_norm(args[0], "development"),
              "distill": lambda: self.u_distill(*args, slot), "merge": lambda: self.u_merge(*args, slot),
              "shared": lambda: self.u_shared_base(*args, slot), "train": lambda: self.u_train(*args, slot),
              "e0": lambda: self.u_e0(*args, slot), "lambda": lambda: self.u_lambda(*args, slot),
              "zeval": lambda: self.u_zeval(*args, slot), "diag": lambda: self.u_diag(*args, slot)}[kind]
        try:
            return fn()
        except Exception as e:                                      # a broken unit must not stop the run
            self.log(f"unit {kind}{args} raised {type(e).__name__}: {e}")
            return False

    def status(self, units):
        L = [f"{PREFIX} STATUS   updated {time.strftime('%Y-%m-%d %H:%M:%S')}   elapsed "
             f"{(time.time() - self.started) / 3600:.1f} h   jobs {self.a.jobs}   sizes {' '.join(self.sizes)}   "
             f"seeds {' '.join(map(str, self.seeds))}{'   [SMOKE]' if self.a.smoke else ''}", f"log: {self.log_path}", ""]
        counts = {}
        for key, *_ in units:
            counts[self.state.get(key, "PENDING")] = counts.get(self.state.get(key, "PENDING"), 0) + 1
        L.append("units: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        trains = [u for u in units if u[1] == "train"]
        done = sum(self.state.get(u[0]) == "DONE" for u in trains)
        L.append(f"training runs: {done}/{len(trains)} done, {sum(self.state.get(u[0]) == 'FAILED' for u in trains)} failed")
        lo = hi = 0.0
        for key, kind, args, *_ in trains:
            if self.state.get(key) in ("DONE", "FAILED"):
                continue
            arm, size, seed = args
            live = TL / f"{self.name(arm, size, seed)}.status.json"
            u = json.loads(live.read_text())["update"] if live.exists() else 0
            spu = self.spu(size)
            lo += max(0, min(2000 if not self.cl else self.cap(arm), self.cap(arm)) - u) * spu / self.a.jobs
            hi += max(0, self.cap(arm) - u) * spu / self.a.jobs
            if self.state.get(key) == "RUNNING" and live.exists():
                ls = json.loads(live.read_text())
                lat = ls.get("latest_value")
                L.append(f"  running {self.name(arm, size, seed)}: update {u}/{self.cap(arm)}, latest val HV "
                         f"{-lat if lat is not None else float('nan'):.4f}, best {-ls['best_value']:.4f} @ "
                         f"{ls['best_update']}, {ls['status']}, train {ls['train_seconds'] / 3600:.2f} h")
        for key, *_ in units:
            if self.state.get(key) == "RUNNING" and not key.startswith("train"):
                L.append(f"  running {key}")
        L.append(f"est. remaining training (rough, jobs {self.a.jobs}): {lo / 3600:.1f} h (stops near "
                 f"{min(2000, self.s['cap'])} updates) "
                 f"to {hi / 3600:.1f} h (cap)")
        failed = [k for k, v in self.state.items() if v == "FAILED"]
        if failed:
            L.append("FAILED: " + ", ".join(failed))
        STATUS_FILE.parent.mkdir(exist_ok=True)
        tmp = STATUS_FILE.with_suffix(".tmp"); tmp.write_text("\n".join(L) + "\n"); tmp.replace(STATUS_FILE)

    def main(self):
        units = self.plan()
        (RES / "plan.json").write_text(json.dumps({"units": [u[0] for u in units], "jobs": self.a.jobs,
                                                   "slots": self.slots, "lane": self.lane, "host": self.host(),
                                                   "settings": self.s, "arms": self.arms, "sizes": self.sizes,
                                                   "seeds": self.seeds, "smoke": self.a.smoke}, indent=1))
        self.log(f"plan: {len(units)} units, jobs {self.a.jobs}, training slots {self.slots} ({self.slot_threads} threads "
                 f"each), CPU lane {self.lane}")
        threads = {}
        lane_busy = False

        def worker(key, kind, args, slot):
            nonlocal lane_busy
            ok = self.execute(kind, args, slot)
            with self.lock:
                self.state[key] = "DONE" if ok else "FAILED"
                if slot is None:
                    lane_busy = False
                else:
                    self.free_slots.append(slot); self.free_slots.sort()
            self.log(f"unit {key}: {'DONE' if ok else 'FAILED'}")

        last_status = 0.0
        calib = [u[0] for u in units if u[1] == "calib"]
        while not self.stop:
            with self.lock:
                calib_phase = any(self.state.get(k) not in ("DONE", "FAILED") for k in calib)
                running = sum(v == "RUNNING" for v in self.state.values())
                for key, kind, args, deps, uses_slot in units:
                    if key in self.state:
                        continue
                    if calib_phase and (kind != "calib" or running):
                        continue                              # calibration runs alone, before anything else
                    if any(self.state.get(d) == "FAILED" for d in deps):
                        self.state[key] = "FAILED"; continue
                    if not all(self.state.get(d) == "DONE" for d in deps):
                        continue
                    if kind == "e0" and any(u[1] == "train" and u[2][1] == args[0] and
                                            self.state.get(u[0]) not in ("DONE", "FAILED") for u in units):
                        continue                              # E0 after its size's training runs (done or failed)
                    if uses_slot:
                        if not self.free_slots:
                            continue
                        slot = self.free_slots.pop(0)
                    else:
                        if lane_busy:
                            continue
                        lane_busy = True; slot = None
                    self.state[key] = "RUNNING"; running += 1
                    t = threading.Thread(target=worker, args=(key, kind, args, slot), daemon=True)
                    threads[key] = t; t.start()
                    if calib_phase:
                        break
                pending = [u for u in units if self.state.get(u[0]) in (None, "RUNNING")]
            if time.time() - last_status > 30:
                self.status(units); last_status = time.time()
            if not pending:
                break
            time.sleep(2)
        for t in threads.values():
            t.join()
        self.status(units)
        failed = [k for k, v in self.state.items() if v == "FAILED"]
        self.log(f"all units finished; FAILED: {failed if failed else 'none'}")
        return 0

    def shutdown(self, *_):
        self.stop = True
        self.log("INTERRUPTED: stopping all children; re-run the same command to resume")
        with self.lock:
            for p in list(self.children):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(3)
        with self.lock:
            for p in list(self.children):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        os._exit(130)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", default="10x5")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--seeds", default="111 222 333")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--with_shared_base", action="store_true")
    ap.add_argument("--log", default=str(REPO / "logs" / "warmstart_v3_runner.log"))
    a = ap.parse_args()
    global RES, TAG, STATUS_FILE, PREFIX, CKPT
    if any(x.replace("-", "") in MF_ORDER for x in a.arms.replace(",", " ").split()):
        study = "merge_failure_test_v3"
        RES, TAG = REPO / "results" / (study + "_smoke" if a.smoke else study), "mfs" if a.smoke else "mf"
        STATUS_FILE, PREFIX = REPO / "logs" / f"{study}_STATUS.txt", "merge failure test v3"
        CKPT = REPO / "checkpoints" / (study + "_smoke" if a.smoke else study)
    elif any(x.replace("-", "") in CC_ORDER for x in a.arms.replace(",", " ").split()):
        study = "contrastive_converged_v3"
        RES, TAG = REPO / "results" / (study + "_smoke" if a.smoke else study), "ccs" if a.smoke else "cc"
        STATUS_FILE, PREFIX = REPO / "logs" / f"{study}_STATUS.txt", "contrastive converged v3"
        CKPT = REPO / "checkpoints" / (study + "_smoke" if a.smoke else study)
    elif any(x.replace("-", "") in MC_ORDER for x in a.arms.replace(",", " ").split()):
        RES, TAG = REPO / "results" / ("merge_categories_v3_smoke" if a.smoke else "merge_categories_v3"), "mcs" if a.smoke else "mc"
        STATUS_FILE, PREFIX = REPO / "logs" / "merge_categories_v3_STATUS.txt", "merge categories v3"
        CKPT = REPO / "checkpoints" / ("merge_categories_v3_smoke" if a.smoke else "merge_categories_v3")
    elif any(x.replace("-", "") in CL_ORDER for x in a.arms.replace(",", " ").split()):
        RES, TAG = REPO / "results" / ("contrastive_v3_smoke" if a.smoke else "contrastive_v3"), "cls" if a.smoke else "cl"
        STATUS_FILE, PREFIX = REPO / "logs" / "contrastive_v3_STATUS.txt", "contrastive v3"
    elif a.smoke:
        RES, TAG = REPO / "results" / "warmstart_v3_smoke", "wss"
    r = Runner(a)
    signal.signal(signal.SIGTERM, r.shutdown); signal.signal(signal.SIGINT, r.shutdown)
    sys.exit(r.main())


if __name__ == "__main__":
    main()
