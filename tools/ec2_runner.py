"""Warm-start run orchestrator (docs/warmstart_protocol_v3.md). Started by scripts/ec2_run_all.sh in the background.

Units, in the pre-registered order: calibration -> validation normalisation points (a CPU lane, alongside training)
-> per size, per seed (111 first), per arm (B, C-M, BB-M, D, E, C-C, C-T, BB-C, BB-T[, ES]): preparation (distillation,
merge, shared base + fine-tunes) -> chunked PPO training with a hang guard -> evaluation -> local commit; then E0 per
size and seed. Resume: finished units are skipped; training continues from its resume state (bit-exact). A unit that
fails twice is marked FAILED (its dependents too) and the run continues; a re-run retries it.

    python tools/ec2_runner.py --sizes 10x5 --arms all --seeds "111 222 333" --jobs 1 [--smoke] [--with-shared-base]
"""
from __future__ import annotations
import argparse, json, os, re, signal, subprocess, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "warmstart_v3"
TN = REPO / "daniel" / "trained_network" / "SD2"
TL = REPO / "daniel" / "train_log" / "SD2"
SPEC = REPO / "checkpoints" / "specialists_v3"
PY = sys.executable
STATUS_FILE = REPO / "logs" / "warmstart_v3_STATUS.txt"

ARM_ORDER = ["B", "CM", "BBM", "D", "E", "CC", "CT", "BBC", "BBT"]
ARM_LABEL = {"B": "B (scratch)", "CM": "C-M", "CC": "C-C", "CT": "C-T", "BBM": "BB-M", "BBC": "BB-C", "BBT": "BB-T",
             "D": "D (distillation)", "E": "E (naive merge)", "ES": "E-shared", "E0": "E0 (zero-training merge)"}
SPU_LOCAL = {"10x5": 0.85, "20x10": 4.5}           # measured MO-DANIEL s/update (pinned, local); scaled by calibration


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
        arms = ARM_ORDER if a.arms == "all" else [x.replace("-", "") for x in a.arms.split()]
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
        f = RES / f"calibration_{size}.json"
        if f.exists():
            c = json.loads(f.read_text())
            return SPU_LOCAL[size] * max(1.0, c["factor"]) * max(1, self.a.jobs)
        return SPU_LOCAL[size] * 2 * max(1, self.a.jobs)

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
        tag = "" if split == "validation" else f"_{split}"
        return RES / "norm" / f"validation_normalisation_points{tag}_{size}.json"

    @staticmethod
    def name(arm, size, seed):
        return f"{size}+mix+bv3_ws_{arm}_s{seed}"

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
        name = f"{size}+mix+bv3_calib_m_s111"
        nj, nm = size.split("x")
        for p in (TN / f"{name}.pth", TN / f"{name}.resume.pth", TL / f"{name}.status.json"):
            p.unlink(missing_ok=True)
        n = self.s["calib_updates"]
        rc = self.run([PY, "train.py", "--config", "../configs/benchmark_v3/m.json", "--n_j", nj, "--n_m", nm,
                       "--model_suffix", "bv3_calib_m_s111", "--seed_train", "111", "--v3_train_source", "generator",
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
        self.commit(f"warm-start v3: calibration {size}", f)
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
        self.commit(f"warm-start v3: frozen {split} normalisation points {size}", RES / "norm")
        return ok

    def u_distill(self, size, seed, slot):
        out = TN / f"{size}+mix+bv3_wsD_student_s{seed}.pth"
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
        self.commit(f"warm-start v3: distillation {size} s{seed}", rec)
        return True

    def u_merge(self, size, seed, slot, shared=False):
        out = TN / f"{size}+mix+bv3_ws{'ES' if shared else 'E'}_merged_s{seed}.pth"
        if out.exists():
            return True
        src = ([str(TN / f"{size}+mix+bv3_wsft_{o}_s{seed}.pth") for o in "mct"] if shared else
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
        nj, nm = size.split("x")
        s = self.s
        a = ["--config", "../configs/benchmark_v3/mct.json", "--n_j", nj, "--n_m", nm, "--preference_dim", "3",
             "--log_ppo_stats", "True", "--mo_val_norm", str(self.norm_file(size)), "--mo_val_grid_step", str(s["val_grid"]),
             "--mo_val_first", str(s["val_first"]), "--model_suffix", f"bv3_ws_{arm}_s{seed}", "--seed_train", str(seed),
             "--v3_train_source", "generator", "--validate_timestep", str(s["validate"]),
             "--converge_window", str(s["window"]), "--converge_tol", str(s["tol"]), "--min_updates", str(s["min_updates"])]
        init = {"CM": ("full", self.spec(size, "m", seed)), "CC": ("full", self.spec(size, "c", seed)),
                "CT": ("full", self.spec(size, "t", seed)), "BBM": ("backbone", self.spec(size, "m", seed)),
                "BBC": ("backbone", self.spec(size, "c", seed)), "BBT": ("backbone", self.spec(size, "t", seed)),
                "D": ("no_critic", TN / f"{size}+mix+bv3_wsD_student_s{seed}.pth"),
                "E": ("full", TN / f"{size}+mix+bv3_wsE_merged_s{seed}.pth"),
                "ES": ("full", TN / f"{size}+mix+bv3_wsES_merged_s{seed}.pth")}.get(arm)
        if init:
            a += ["--init_from", str(init[1]), "--init_mode", init[0]]
        return a

    def u_shared_base(self, size, seed, slot):
        """E-shared preparation: equal-weight unconditioned base, then M/C/T fine-tunes from it (fixed budgets)."""
        nj, nm = size.split("x")
        rdir = RES / "shared_base" / f"{size}_s{seed}"; rdir.mkdir(parents=True, exist_ok=True)
        steps = [("base", "mct", f"bv3_wsbase_s{seed}", self.s["base_updates"], [])]
        steps += [(f"ft_{o}", o, f"bv3_wsft_{o}_s{seed}", self.s["ft_updates"],
                   ["--init_from", str(TN / f"{size}+mix+bv3_wsbase_s{seed}.pth")]) for o in "mct"]
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
        self.commit(f"warm-start v3: shared base and fine-tunes {size} s{seed}", rdir)
        return self.u_merge(size, seed, slot, shared=True)

    def u_train(self, arm, size, seed, slot):
        name = self.name(arm, size, seed)
        rdir = RES / "runs" / name
        edir = RES / "eval" / f"{self.s['eval_split']}_{size}"
        if (rdir / "status.json").exists() and (edir / f"{name}.json").exists():
            return True
        rdir.mkdir(parents=True, exist_ok=True); (rdir / "FAILED").unlink(missing_ok=True)
        m = json.loads((rdir / "run_meta.json").read_text()) if (rdir / "run_meta.json").exists() else {}
        m.update(arm=arm, label=ARM_LABEL[arm], size=size, seed=seed, slot_cpus=self.slots[slot],
                 omp_threads=self.slot_threads, host=self.host())
        (rdir / "run_meta.json").write_text(json.dumps(m, indent=1))
        if not self.train_chunks(name, self.mo_args(arm, size, seed), size, slot, self.s["cap"], self.s["chunk"][size], rdir):
            if not self.stop:
                (rdir / "FAILED").write_text(f"training failed twice at {time.strftime('%F %T')}\n")
                self.commit(f"warm-start v3: {name} FAILED", rdir)
            return False
        for f, dst in ((TL / f"valiquality_{name}.json", "valiquality.json"), (TL / f"reward_{name}.json", "reward.json"),
                       (TL / f"{name}.run.json", "run.json"), (TL / f"{name}.status.json", "status.json")):
            (rdir / dst).write_text(f.read_text())
        st = json.loads((rdir / "status.json").read_text())
        self.log(f"{name}: {st['status']} at update {st['update']}, best val HV {-st['best_value']:.4f} "
                 f"@ {st['best_update']}, train {st['train_seconds'] / 3600:.2f} h")
        for attempt in (1, 2):
            rc = self.run([PY, "evaluate_mo_v3.py", "--model", name, "--split", self.s["eval_split"], "--size", size,
                           "--first", str(self.s["eval_first"]), "--grid_step", str(self.s["eval_grid"]),
                           "--norm", *self.eval_norm(size), "--out", str(RES / "eval"), "--sample", str(self.s["sample"]),
                           "--sample_first", str(self.s["sample_first"][size]),
                           "--device", "cuda"], timeout=7200, cpus=self.slots[slot], threads=self.slot_threads,
                          cwd=REPO / "daniel", out=TL / f"{name}.stdout.log")
            if rc == 0:
                break
            self.log(f"eval {name} FAILED (attempt {attempt})")
        if rc != 0:
            (rdir / "FAILED").write_text(f"evaluation failed twice at {time.strftime('%F %T')}\n")
            self.commit(f"warm-start v3: {name} evaluation FAILED", rdir)
            return False
        self.commit(f"warm-start v3: {name} ({st['status']})", rdir, edir, STATUS_FILE)
        return True

    def u_e0(self, size, seed, slot):
        edir = RES / "eval" / f"{self.s['eval_split']}_{size}"
        name = self.name("E0", size, seed)
        if (edir / f"{name}.json").exists():
            return True
        rc = self.run([PY, "tools/e0_merge_front_v3.py", "--size", size, "--seed", str(seed), "--split",
                       self.s["eval_split"], "--first", str(self.s["eval_first"]), "--grid_step", str(self.s["eval_grid"]),
                       "--spec_dir", str(SPEC), "--norm", *self.eval_norm(size), "--out", str(RES / "eval"),
                       "--device", "cuda"], timeout=7200, cpus=self.slots[slot], threads=self.slot_threads,
                      out=self.log_path)
        if rc == 0:
            self.commit(f"warm-start v3: E0 {size} s{seed}", edir)
        return rc == 0

    # ------------------------------------------------------------------ plan and scheduling
    def plan(self):
        """Ordered units: (key, kind, args, deps, uses_slot)."""
        U = []
        for size in self.sizes:
            U.append((f"calib:{size}", "calib", (size,), [], True))
        for size in self.sizes:
            if self.a.smoke:
                U.append((f"normdev:{size}", "normdev", (size,), [], False))
            U.append((f"norm:{size}", "norm", (size,), [], False))
        for size in self.sizes:
            for seed in self.seeds:
                for arm in self.arms:
                    deps = [f"norm:{size}"] + ([f"normdev:{size}"] if self.a.smoke else [])
                    if arm == "D":
                        U.append((f"distill:{size}:{seed}", "distill", (size, seed), [], True))
                        deps.append(f"distill:{size}:{seed}")
                    if arm == "E":
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

    def execute(self, kind, args, slot):
        fn = {"calib": lambda: self.u_calibration(args[0], slot), "norm": lambda: self.u_norm(args[0]),
              "normdev": lambda: self.u_norm(args[0], "development"),
              "distill": lambda: self.u_distill(*args, slot), "merge": lambda: self.u_merge(*args, slot),
              "shared": lambda: self.u_shared_base(*args, slot), "train": lambda: self.u_train(*args, slot),
              "e0": lambda: self.u_e0(*args, slot)}[kind]
        try:
            return fn()
        except Exception as e:                                      # a broken unit must not stop the run
            self.log(f"unit {kind}{args} raised {type(e).__name__}: {e}")
            return False

    def status(self, units):
        L = [f"warm-start v3 STATUS   updated {time.strftime('%Y-%m-%d %H:%M:%S')}   elapsed "
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
            lo += max(0, min(2000, self.s["cap"]) - u) * spu / self.a.jobs
            hi += max(0, self.s["cap"] - u) * spu / self.a.jobs
            if self.state.get(key) == "RUNNING" and live.exists():
                ls = json.loads(live.read_text())
                lat = ls.get("latest_value")
                L.append(f"  running {self.name(arm, size, seed)}: update {u}/{self.s['cap']}, latest val HV "
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
    r = Runner(a)
    signal.signal(signal.SIGTERM, r.shutdown); signal.signal(signal.SIGINT, r.shutdown)
    sys.exit(r.main())


if __name__ == "__main__":
    main()
