"""Specialist-reuse runner (docs/specialist_reuse_protocol_v3.md), 10x5, seed 111, waiting allowed.

Stages, in dependency order, each resumable (re-run the same command):
  1 base          random unconditioned DANIEL after setup_seed(111)
  2 pool A        specialists M, C, T (single-objective contrastive from copies of the base) + the 6 From-scratch trials
  3 checks        Rewarded Soups and TIES on the real specialists: a preference change changes the merged weights;
                  each exact corner reproduces its specialist (weights and outputs); stops on failure
  4 distill       Distillation stage 1 (distill_v3.py, once)
  5 screen        TIES merge screen: validation HV of the merged start for every (keep, lambda)
  6 pool B        the 6 trials of Distillation, Rewarded Soups and TIES
  7 finals        the selected setting of each method, trained fresh with the stopping rule
  8 evaluation    greedy and best-of-128 (seed 51) on 100 evaluation instances x 67 preferences, one method at a time
  9 report        results/specialist_reuse_v3.md

    python tools/specialist_reuse_v3_runner.py [--jobs 3] [--smoke]

Called by scripts/run_specialist_reuse_v3.sh (pre-flight). Every checkpoint is copied to checkpoints/<name>/ with
SHA256SUMS.json; logs and results go to results/<name>/. Commits locally; never pushes.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_specialist_reuse_v3 as F                              # noqa: E402

PY = sys.executable
DAN = REPO / "daniel"
TN, TL = DAN / "trained_network" / "SD2", DAN / "train_log" / "SD2"
SLOTS = ["0-3", "4-7", "8-11"]
CO_AUTHOR = ("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
             "Claude-Session: https://claude.ai/code/session_01Rcu6f737di34xSocjiJkEU")
ENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", WANDB_MODE="offline", PYTHONUNBUFFERED="1")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Runner:
    def __init__(self, a):
        self.cfg = F.load()
        self.smoke, self.jobs = a.smoke, a.jobs
        sfx = "_smoke" if a.smoke else ""
        self.name = self.cfg["name"] + sfx
        self.res, self.ckpt = REPO / "results" / self.name, REPO / "checkpoints" / self.name
        self.md = REPO / "results" / f"{self.name}.md"
        self.tag = "srs" if a.smoke else "sr"
        self.prefix = "specialist reuse v3" + (" SMOKE" if a.smoke else "")
        for d in (self.res, self.ckpt):
            d.mkdir(parents=True, exist_ok=True)
        self.git_lock, self.log_lock, self.state_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.logf = open(self.res / "runner.log", "a")
        self.state_f = self.res / "state.json"
        self.state = json.loads(self.state_f.read_text()) if self.state_f.exists() else {"units": {}}
        sm = self.cfg["smoke"] if a.smoke else {}
        self.sm = sm
        st = self.cfg["stopping"]
        self.updates_trial = sm.get("updates", self.cfg["tuning"]["updates_per_trial"])
        self.rule = ({"converge_window": 0, "converge_tol": st["converge_tol"], "min_updates": 0,
                      "max_updates": sm["updates"], "chunk": 0} if a.smoke else st)   # smoke: one invocation
        self.validate = sm.get("validate_timestep", self.cfg["recipe"]["constants"]["validate_timestep"])
        self.trials = self.cfg["tuning"]["trials"][: sm.get("trials", len(self.cfg["tuning"]["trials"]))]
        self.base = self.ckpt / "base_s111.pth"
        (self.res / "settings.json").write_text(json.dumps({**self.cfg, "smoke_run": a.smoke, "jobs": a.jobs,
                                                             "slots": SLOTS[: a.jobs]}, indent=1) + "\n")

    # ------------------------------------------------------------------ plumbing
    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self.log_lock:
            print(line, flush=True); self.logf.write(line + "\n"); self.logf.flush()

    def save_state(self):
        with self.state_lock:
            tmp = self.state_f.with_suffix(".tmp"); tmp.write_text(json.dumps(self.state, indent=1)); tmp.replace(self.state_f)

    def commit(self, msg, *paths):
        with self.git_lock:
            for p in paths:
                if Path(p).exists():
                    subprocess.run(["git", "add", "-A", "--", str(p)], cwd=REPO, check=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode:
                subprocess.run(["git", "commit", "-q", "-m", msg, "-m", CO_AUTHOR], cwd=REPO, check=True)

    def run(self, cmd, slot, timeout, out, cwd=None):
        with open(out, "a") as f:
            try:
                return subprocess.run(["taskset", "-c", SLOTS[slot], *cmd], cwd=cwd or DAN, stdout=f,
                                      stderr=subprocess.STDOUT, env=ENV, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                return -9

    def write_sums(self):
        sums = {str(p.relative_to(self.ckpt)): sha(p) for p in sorted(self.ckpt.rglob("*.pth"))}
        (self.ckpt / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1) + "\n")

    def model(self, name):
        return f"10x5+mix+bv3_{self.tag}_{name}_s111"

    # ------------------------------------------------------------------ command lines
    def common(self, lr=None, beta=None):
        c = self.cfg["recipe"]["constants"]
        return ["--n_j", "10", "--n_m", "5", "--seed_train", "111", "--v3_train_source", "generator",
                "--finetune_objective", "contrastive", "--cl_k", str(c["cl_k"]),
                "--cl_beta", str(beta if beta is not None else c["cl_beta"]), "--cl_margin", str(c["cl_margin"]),
                "--cl_grad_clip", str(c["cl_grad_clip"]), "--lr", str(lr if lr is not None else c["lr"]),
                "--minibatch_size", str(c["minibatch_size"]), "--num_envs", str(c["num_envs"]),
                "--reset_env_timestep", str(c["reset_env_timestep"]), "--validate_timestep", str(self.validate),
                "--resume", "True", "--device", "cuda", "--dispatch_rule", "semi_active"]

    def stop_args(self, full):
        r = self.rule
        if full:
            return ["--converge_window", str(r["converge_window"]), "--converge_tol", str(r["converge_tol"]),
                    "--min_updates", str(r["min_updates"]), "--max_updates", str(r["max_updates"]),
                    "--chunk_updates", str(r["chunk"])]
        return ["--converge_window", "0", "--max_updates", str(self.updates_trial),
                "--chunk_updates", str(0 if self.smoke else min(500, self.updates_trial))]

    def specialist_args(self, o):
        return ["train.py", "--config", str(REPO / self.cfg["specialists"]["configs"][o]["path"]),
                "--model_suffix", f"bv3_{self.tag}_spec_{o}_s111", *self.common(), *self.stop_args(True),
                "--init_from", str(self.base), "--init_mode", "exact"]

    def specs(self):
        return [str(self.ckpt / "specialists" / f"{self.model(f'spec_{o}')}.pth") for o in "mct"]

    def method_args(self, method, suffix, lr=None, beta=None, full=False, ties=None, max_updates=None):
        a = ["train.py", "--config", str(REPO / self.cfg["mo_config"]["path"]), "--goal", "mct", "--obs_version", "v3",
             "--reward_version", "v3", "--preference_dim", "3", "--mo_val_norm", str(REPO / self.cfg["validation"]["norm"]),
             "--model_suffix", f"bv3_{self.tag}_{suffix}_s111", *self.common(lr, beta)]
        if self.smoke:
            a += ["--mo_val_first", str(self.sm["mo_val_first"])]
        a += self.stop_args(full) if max_updates is None else ["--converge_window", "0", "--max_updates", str(max_updates)]
        if method == "distill":
            a += ["--init_from", str(self.ckpt / "distill" / "student_s111.pth"), "--init_mode", "no_critic"]
        elif method in ("rs", "ties"):
            a += ["--merge_mix", ",".join(self.specs())]
            if method == "ties":
                keep, lam = ties if ties else self.state["ties_selected"]
                a += ["--merge_rule", "ties", "--merge_base", str(self.base), "--ties_keep", str(keep),
                      "--ties_lambda", str(lam)]
        return a

    # ------------------------------------------------------------------ one training run, in chunks, resumable
    def train_unit(self, key, cmd, name, slot):
        live, out = TL / f"{name}.status.json", self.res / "logs" / f"{name}.stdout.log"
        out.parent.mkdir(parents=True, exist_ok=True)
        u = self.state["units"].setdefault(key, {"name": name})
        u.setdefault("start", time.time()); u["slot"] = SLOTS[slot]
        while True:
            st = json.loads(live.read_text())["status"] if live.exists() else "NONE"
            if st in ("FINISHED", "CONVERGED"):
                break
            self.log(f"train {name}: chunk (status {st}, slot {slot})")
            rc = self.run([PY, *cmd], slot, 6 * 3600, out)
            if rc != 0:
                self.log(f"{name}: train.py failed (rc {rc}); see {out}")
                return False
        s = json.loads(live.read_text())
        rdir = self.res / "runs" / name; rdir.mkdir(parents=True, exist_ok=True)
        for src, dst in ((f"reward_{name}.json", "reward.json"), (f"valiquality_{name}.json", "valiquality.json"),
                         (f"{name}.run.json", "run.json"), (f"{name}.status.json", "status.json")):
            if (TL / src).exists():
                shutil.copy2(TL / src, rdir / dst)
        u.update(end=u.get("end") or time.time(), status=s["status"], update=s["update"], best=s["best_value"],
                 best_update=s["best_update"], train_seconds=s["train_seconds"],
                 capped=s["status"] == "FINISHED" and s["update"] >= s["max_updates"])
        self.save_state()
        self.log(f"{name}: {s['status']} at update {s['update']} (best {s['best_value']:.4f} @ {s['best_update']}, "
                 f"{s['train_seconds'] / 3600:.2f} h)")
        return True

    def pool(self, units):
        """units: [(key, cmd, name, after)] run on `jobs` slots; after(name) runs when a unit has finished."""
        free, lock, ok = list(range(self.jobs)), threading.Lock(), []

        def work(u):
            key, cmd, name, after = u
            with lock:
                slot = free.pop(0)
            try:
                r = self.train_unit(key, cmd, name, slot)
                if r and after:
                    after(name)
                ok.append(r)
            finally:
                with lock:
                    free.append(slot)
        with ThreadPoolExecutor(self.jobs) as ex:
            list(ex.map(work, units))
        return all(ok)

    # ------------------------------------------------------------------ stages
    def stage_base(self):
        if self.base.exists():
            return
        code = ("import sys; sys.path.insert(0, 'tools'); import world_model_m2_bases as W; "
                f"print(W.save_bundle(W.random_base_sd(111), '{self.base}'))")
        subprocess.run([PY, "-c", code], cwd=REPO, check=True, env=ENV, stdout=self.logf, stderr=subprocess.STDOUT)
        self.write_sums()
        self.log(f"base: {self.base.name} {sha(self.base)[:12]}")
        self.commit(f"{self.prefix}: shared random base", self.ckpt, self.res)

    def keep_specialist(self, name):
        d = self.ckpt / "specialists"; d.mkdir(exist_ok=True)
        shutil.copy2(TN / f"{name}.pth", d / f"{name}.pth")
        self.write_sums()

    def trial_units(self, method):
        out = []
        for k, t in enumerate(self.trials):
            suffix = f"{method}_t{k}"
            out.append((f"trial:{method}:{k}", self.method_args(method, suffix, t["lr"], t["cl_beta"]),
                        self.model(suffix), None))
        return out

    def stage_pool_a(self):
        units = [(f"spec:{o}", self.specialist_args(o), self.model(f"spec_{o}"), self.keep_specialist) for o in "tcm"]
        units += self.trial_units("scratch")
        ok = self.pool(units)
        self.commit(f"{self.prefix}: specialists and From-scratch trials", self.ckpt, self.res)
        return ok

    def stage_checks(self):
        f = self.res / "merge_checks.json"
        if not f.exists():
            screen = self.cfg["smoke"]["screen"] if self.smoke else self.cfg["methods"]["ties"]["screen"]
            r = subprocess.run([PY, str(REPO / "tools" / "specialist_reuse_v3_checks.py"), "--specs", ",".join(self.specs()),
                                "--base", str(self.base), "--keep", ",".join(map(str, screen["keep"])),
                                "--lam", ",".join(map(str, screen["lambda"])), "--out", str(f)],
                               cwd=REPO, env=ENV, stdout=self.logf, stderr=subprocess.STDOUT)
            if r.returncode != 0 or not f.exists():
                raise SystemExit("merge checks failed; nothing further started (see runner.log)")
        rec = json.loads(f.read_text())
        if not rec["all_pass"]:
            raise SystemExit("merge checks failed; nothing further started")
        self.log("merge checks passed: preference changes the merged weights; corners reproduce the specialists")
        self.commit(f"{self.prefix}: merge checks", self.res)

    def distill_cmd(self, out):
        d = self.cfg["methods"]["distill"]["stage1"]
        sm = self.sm.get("distill", {})
        return [PY, "distill_v3.py", "--size", "10x5", "--seed", "111", "--specs", ",".join(self.specs()),
                "--dispatch_rule", "semi_active", "--out", str(out), "--instances", str(sm.get("instances", d["instances"])),
                "--epochs", str(sm.get("epochs", d["epochs"])), "--batch", str(d["batch"]), "--lr", str(d["lr"]),
                "--wall_cap", str(sm.get("wall_cap", d["wall_cap"])), "--val_first", str(sm.get("val_first", 0))]

    def stage_distill(self):
        out = self.ckpt / "distill" / "student_s111.pth"
        if out.exists():
            return
        cmd = self.distill_cmd(out)
        t0 = time.time()
        if self.run(cmd, 0, 4 * 3600, self.res / "distill.log") != 0:
            raise SystemExit("distillation stage 1 failed")
        self.state["distill_stage1"] = {"wall_seconds": time.time() - t0,
                                        **{k: v for k, v in json.loads(out.with_suffix(".distill.json").read_text()).items()
                                           if k in ("collection_seconds", "training_seconds", "total_seconds", "states",
                                                    "epochs_run", "wall_cap_hit")}}
        self.save_state(); self.write_sums()
        self.log(f"distillation stage 1: {self.state['distill_stage1']['total_seconds']:.0f} s")
        self.commit(f"{self.prefix}: distillation stage 1", self.ckpt, self.res)

    def stage_screen(self):
        screen = self.cfg["smoke"]["screen"] if self.smoke else self.cfg["methods"]["ties"]["screen"]
        res = self.state.setdefault("ties_screen", {})
        for keep in screen["keep"]:
            for lam in screen["lambda"]:
                key = f"{keep}|{lam}"
                if key in res:
                    continue
                suffix = f"ties_screen_k{keep}_l{lam}".replace(".", "p")
                name = self.model(suffix)
                t0 = time.time()
                rc = self.run([PY, *self.method_args("ties", suffix, ties=(keep, lam), max_updates=0)], 0, 3600,
                              self.res / "logs" / f"{name}.stdout.log")
                v = json.loads((TL / f"valiquality_{name}.json").read_text()) if rc == 0 else None
                if not v:
                    raise SystemExit(f"TIES screen {key} failed")
                res[key] = {"keep": keep, "lambda": lam, "hv": v[0]["hv"], "seconds": time.time() - t0}
                self.save_state()
                self.log(f"TIES screen keep {keep} lambda {lam}: validation HV {v[0]['hv']:.4f}")
        best = max(res.values(), key=lambda r: (r["hv"], -abs(r["lambda"] - 1.0), -abs(r["keep"] - 0.2)))
        self.state["ties_selected"] = [best["keep"], best["lambda"]]
        self.save_state()
        self.log(f"TIES merge setting selected: keep {best['keep']}, lambda {best['lambda']}")
        self.commit(f"{self.prefix}: TIES screen", self.res)

    def stage_pool_b(self):
        units = self.trial_units("distill") + self.trial_units("rs") + self.trial_units("ties")
        ok = self.pool(units)
        self.commit(f"{self.prefix}: Distillation, Rewarded Soups and TIES trials", self.res)
        return ok

    def select(self, method):
        rows = []
        for k, t in enumerate(self.trials):
            u = self.state["units"][f"trial:{method}:{k}"]
            rows.append((-u["best"], t))
        rows.sort(key=lambda r: (-r[0], abs(__import__("math").log(r[1]["lr"] / 3e-4)), abs(r[1]["cl_beta"] - 0.1)))
        return rows[0][1]

    def stage_finals(self):
        sel = {m: self.select(m) for m in F.METHODS}
        self.state["selected"] = sel; self.save_state()
        for m, t in sel.items():
            self.log(f"selected for {m}: lr {t['lr']}, beta {t['cl_beta']}")

        def keep(name):
            d = self.ckpt / "finals"; d.mkdir(exist_ok=True)
            shutil.copy2(TN / f"{name}.pth", d / f"{name}.pth"); self.write_sums()
        units = [(f"final:{m}", self.method_args(m, m, t["lr"], t["cl_beta"], full=True), self.model(m), keep)
                 for m, t in sel.items()]
        ok = self.pool(units)
        self.commit(f"{self.prefix}: full runs", self.ckpt, self.res)
        return ok

    def stage_eval(self):
        for m in F.METHODS:
            out = self.res / "eval" / f"{m}.npz"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [PY, str(REPO / "tools" / "specialist_reuse_v3_eval.py"), "--model",
                   str(self.ckpt / "finals" / f"{self.model(m)}.pth"), "--out", str(out)]
            if self.smoke:
                cmd += ["--first", str(self.sm["eval_first"]), "--K", str(self.sm["sample_K"])]
            if subprocess.run(["taskset", "-c", "0-11", *cmd], cwd=REPO, env=ENV, stdout=self.logf,
                              stderr=subprocess.STDOUT).returncode != 0:
                raise SystemExit(f"evaluation of {m} failed")
            self.log(f"evaluated {m}")
            self.commit(f"{self.prefix}: evaluation {m}", self.res)

    def report(self):
        r = subprocess.run([PY, str(REPO / "tools" / "analyze_specialist_reuse_v3.py"), *(["--smoke"] if self.smoke else [])],
                           cwd=REPO, stdout=self.logf, stderr=subprocess.STDOUT)
        self.commit(f"{self.prefix}: report", self.res, self.md)
        return r.returncode == 0

    def main(self):
        t0 = time.time()
        self.log(f"start ({self.jobs} slots){' [SMOKE]' if self.smoke else ''}")
        self.stage_base()
        if not self.stage_pool_a():
            raise SystemExit("pool A had failures")
        self.stage_checks()
        self.stage_distill()
        self.stage_screen()
        if not self.stage_pool_b():
            raise SystemExit("pool B had failures")
        if not self.stage_finals():
            raise SystemExit("full runs had failures")
        self.stage_eval()
        ok = self.report()
        self.state["wall_hours"] = (time.time() - t0) / 3600; self.save_state()
        self.commit(f"{self.prefix}: run state", self.res)
        self.log(f"done ({(time.time() - t0) / 3600:.2f} h this invocation)")
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sys.exit(Runner(a).main())


if __name__ == "__main__":
    main()
