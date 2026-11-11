"""Plateau diagnosis runner (docs/plateau_diagnosis_protocol_v3.md).

    python tools/plateau_diagnosis_v3_runner.py --part a [--jobs 3] [--smoke]   # no training
    python tools/plateau_diagnosis_v3_runner.py --part b [--jobs 3] [--smoke]   # four 1,500-update runs

Part A: the checkpoints along training of the five converged runs (copied, unwrapped from the saved final states, or
re-created start networks checked against the logged start hash) go to checkpoints/plateau_diagnosis_v3/part_a/ with
SHA256SUMS.json; each is evaluated greedy and best-of-16 on the evaluation split; weight drift from the start; report.
Part B: from the World-model base mix start, one change per arm, 1,500 updates fixed; snapshots every 250 updates and
the best checkpoint to checkpoints/plateau_diagnosis_v3/part_b/ (SHA256SUMS.json); final and best evaluated greedy and
best-of-16; report. Called by scripts/run_plateau_diagnosis_v3.sh (pre-flight). Commits locally; never pushes.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_plateau_diagnosis_v3 as F                             # noqa: E402

PY = sys.executable
TN, TL = REPO / "daniel" / "trained_network" / "SD2", REPO / "daniel" / "train_log" / "SD2"
SLOTS = ["0-3", "4-7", "8-11"]
CO_AUTHOR = ("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
             "Claude-Session: https://claude.ai/code/session_01Rcu6f737di34xSocjiJkEU")
ENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
# run.json config keys that define a contrastive run's start network (for the --max_updates 0 rebuild)
REBUILD_KEYS = ("n_j", "n_m", "preference_dim", "mo_val_norm", "mo_val_grid_step", "mo_val_first", "seed_train",
                "v3_train_source", "validate_timestep", "finetune_objective", "cl_k", "cl_beta", "cl_margin",
                "cl_grad_clip", "lr", "minibatch_size", "num_envs", "reset_env_timestep", "pref_dirichlet")


def state_hash(sd):
    """The hash train.py logs for the frozen start network (cl_reference_sha256)."""
    h = hashlib.sha256()
    for k, v in sorted(sd.items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Runner:
    def __init__(self, a):
        self.cfg = F.load()
        self.smoke, self.jobs = a.smoke, a.jobs
        sfx = "_smoke" if a.smoke else ""
        self.res, self.ckpt = REPO / "results" / f"plateau_diagnosis_v3{sfx}", REPO / "checkpoints" / f"plateau_diagnosis_v3{sfx}"
        self.md = REPO / "results" / f"plateau_diagnosis_v3{sfx}.md"
        self.prefix = "plateau diagnosis v3" + (" SMOKE" if a.smoke else "")
        self.res.mkdir(parents=True, exist_ok=True)
        self.git_lock, self.log_lock = threading.Lock(), threading.Lock()
        self.logf = open(self.res / "runner.log", "a")
        sm = self.cfg["smoke"] if a.smoke else {}
        ev = self.cfg["part_a"]["evaluation"]
        self.eval_split, self.eval_first = sm.get("eval_split", ev["split"]), sm.get("eval_first", 0)
        self.eval_norm = REPO / sm.get("norm", ev["norm"])

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

    def run(self, cmd, slot, timeout, out, cwd=None):
        with open(out, "a") as f:
            try:
                return subprocess.run(["taskset", "-c", SLOTS[slot], *cmd], cwd=cwd or REPO / "daniel", stdout=f,
                                      stderr=subprocess.STDOUT, env=ENV, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                return 124

    def write_sums(self, d):
        with self.git_lock:
            sums = {str(p.relative_to(d)): file_sha(p) for p in sorted(d.rglob("*.pth"))}
            (d / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1) + "\n")

    def evaluate(self, model, edir, slot, out):
        """Greedy and best-of-16 on the evaluation split (66 preferences); skipped if already done for this file."""
        ev = self.cfg["part_a"]["evaluation"]
        rec = edir / f"{self.eval_split}_10x5" / f"{Path(model).stem}.json"
        if rec.exists() and json.loads(rec.read_text()).get("checkpoint_sha256") == file_sha(model) \
                and "sample" in json.loads(rec.read_text()):
            return 0
        for _ in (1, 2):
            rc = self.run([PY, "evaluate_mo_v3.py", "--model", str(model), "--split", self.eval_split, "--size", "10x5",
                           "--first", str(self.eval_first), "--grid_step", str(ev["grid_step"]),
                           "--sample", str(ev["sample"]), "--sample_seed", str(ev["sample_seed"]),
                           "--norm", str(self.eval_norm), "--out", str(edir), "--device", "cuda"], slot, 7200, out)
            if rc == 0:
                return 0
        return rc

    # ------------------------------------------------------------------------------------------------ Part A
    def rebuild_start(self, run_json, out_path, device="cuda"):
        """Re-create a contrastive run's start network: train.py with the run's own settings and --max_updates 0
        (no update), in a scratch directory; returns the bundle path."""
        c = json.loads((REPO / run_json).read_text())["config"]
        args = ["--config", str(REPO / "configs/benchmark_v3/mct.json"), "--model_suffix", "bv3_pdrebuild_s111",
                "--converge_window", "0", "--max_updates", "0", "--budget_checkpoints", "0", "--device", device]
        for k in REBUILD_KEYS:
            args += [f"--{k}", str(c[k])]
        if c.get("init_from"):
            args += ["--init_from", c["init_from"], "--init_mode", c["init_mode"]]
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([PY, str(REPO / "daniel" / "train.py"), *args], cwd=d, capture_output=True, text=True,
                               env=ENV, timeout=900)
            if r.returncode:
                raise RuntimeError(f"start rebuild failed: {r.stdout[-1500:]}{r.stderr[-1500:]}")
            shutil.copy(Path(d) / "trained_network" / "SD2" / "10x5+mix+bv3_pdrebuild_s111@u0.pth", out_path)

    def prepare(self):
        """The Part A checkpoints, each checked: update number, and (start networks) the logged start hash."""
        import torch
        sys.path.insert(0, str(REPO / "daniel"))
        from checkpointing import make_checkpoint_bundle
        out = self.ckpt / "part_a"; out.mkdir(parents=True, exist_ok=True)
        starts, man = self.cfg["start_network_sha256"], []
        for c in self.cfg["part_a"]["checkpoints"]:
            dst = out / f"{c['name']}.pth"
            src = REPO / c["source"]
            if file_sha(src) != c["source_sha256"]:
                raise SystemExit(f"{c['source']} changed since the freeze")
            if not dst.exists():
                if c["kind"] == "copy":
                    shutil.copy(src, dst)
                elif c["kind"] == "rebuild":
                    self.rebuild_start(c["source"], dst)
                else:                                                # the policy of a saved final state
                    r = F.torch_load(src)
                    if r["next_update"] != c["update"]:
                        raise SystemExit(f"{c['source']}: next_update {r['next_update']} != {c['update']}")
                    cfg = json.loads((REPO / self.cfg["runs"][c["run"]]["dir"] / "run.json").read_text())["config"]
                    v = [e for e in r["validation_log"] if e["update"] == c["update"]][-1]
                    b = make_checkpoint_bundle(r["policy"], config=cfg, data_fingerprints={}, update=c["update"],
                                               validation_metrics={k: x for k, x in v.items()
                                                                   if isinstance(x, (int, float))},
                                               selection_metric=v["selection_metric"], repo_root=REPO)
                    b["provenance"] = {"final_state_of": c["source"], "sha256": c["source_sha256"]}
                    torch.save(b, dst)
            b = torch.load(dst, map_location="cpu", weights_only=False)
            if b["update"] != c["update"]:
                raise SystemExit(f"{dst.name}: update {b['update']} != {c['update']}")
            if c["update"] == 0 and state_hash(b["state_dict"]) != starts[c["run"]]:
                raise SystemExit(f"{dst.name}: start network does not match the run's logged start hash")
            man.append({**c, "file": str(dst.relative_to(REPO)), "sha256": file_sha(dst)})
        self.write_sums(out)
        (self.res / "part_a").mkdir(parents=True, exist_ok=True)
        (self.res / "part_a" / "checkpoints.json").write_text(json.dumps(man, indent=1) + "\n")
        self.drift(man)
        self.log(f"Part A: {len(man)} checkpoints ready (start networks match the logged hashes)")
        return man

    def drift(self, man):
        """||theta - theta_0|| / ||theta_0|| of trunk, actor and both, per checkpoint (merge mixes: all three networks)."""
        import re
        import torch
        sds = {(c["run"], c["update"]): torch.load(REPO / c["file"], map_location="cpu", weights_only=False)["state_dict"]
               for c in man}
        rows = []
        for (run, u), sd in sorted(sds.items()):
            ref = sds[(run, 0)]
            row = {"run": run, "update": u}
            for name, pre in (("trunk", ("feature_exact.",)), ("actor", ("actor.",)), ("policy", ("feature_exact.", "actor."))):
                ks = [k for k in ref if re.sub(r"^nets\.\d+\.", "", k).startswith(pre)]
                d = sum(float(((sd[k].double() - ref[k].double()) ** 2).sum()) for k in ks) ** .5
                n0 = sum(float((ref[k].double() ** 2).sum()) for k in ks) ** .5
                row[f"{name}_drift"], row[f"{name}_drift_abs"] = d / n0, d
            rows.append(row)
        (self.res / "part_a" / "drift.json").write_text(json.dumps(rows, indent=1) + "\n")

    def part_a(self):
        t0 = time.time()
        man = self.prepare()
        self.commit(f"{self.prefix}: Part A checkpoints along training (SHA-256)", self.ckpt / "part_a",
                    self.res / "part_a")
        edir, out = self.res / "part_a" / "eval", self.res / "part_a" / "eval.log"
        edir.mkdir(parents=True, exist_ok=True)
        free, lock, bad = list(range(self.jobs)), threading.Lock(), []

        def work(c):
            with lock:
                slot = free.pop(0)
            try:
                t = time.time()
                rc = self.evaluate(REPO / c["file"], edir, slot, out)
                self.log(f"eval {c['name']}: {'ok' if rc == 0 else f'FAILED (rc {rc})'} ({time.time() - t:.0f} s)")
                if rc:
                    bad.append(c["name"])
            finally:
                with lock:
                    free.append(slot)
        with ThreadPoolExecutor(self.jobs) as ex:
            list(ex.map(work, man))
        (self.res / "part_a" / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                    "seconds": time.time() - t0, "failed": bad}) + "\n")
        rc = self.analyze()
        self.commit(f"{self.prefix}: Part A (no training) and report", self.res, self.md)
        return 1 if bad or rc else 0

    def analyze(self):
        rc = subprocess.run([PY, str(REPO / "tools" / "analyze_plateau_diagnosis_v3.py"), *(["--smoke"] if self.smoke else [])],
                            cwd=REPO, stdout=self.logf, stderr=subprocess.STDOUT).returncode
        self.log(f"report {'written' if rc == 0 else 'FAILED (see runner.log)'}")
        return rc

    # ------------------------------------------------------------------------------------------------ Part B
    def name(self, arm):
        return f"10x5+mix+bv3_pd{'s' if self.smoke else ''}_{arm}_s111"

    def args(self, arm):
        pb, sm = self.cfg["part_b"], (self.cfg["smoke"] if self.smoke else {})
        c = {**pb["base"]["contrastive"], **pb["arms"][arm]["change"]}
        sp = pb["base"]["specialists"]
        upd = sm.get("updates", pb["updates"])
        bc = sm.get("budget_checkpoints", pb["budget_checkpoints"])
        m3 = json.loads((REPO / pb["base"]["source"]).read_text())
        return ["train.py", "--config", "../configs/benchmark_v3/mct.json", "--n_j", "10", "--n_m", "5",
                "--model_suffix", self.name(arm).split("+mix+")[1], "--seed_train", "111",
                "--v3_train_source", "generator", "--preference_dim", "3", "--pref_dirichlet", str(c["pref_dirichlet"]),
                "--mo_val_grid_step", str(c["mo_val_grid_step"]), "--mo_val_first", str(sm.get("mo_val_first", 0)),
                "--mo_val_norm", str(REPO / m3["part_a"]["val_norm"]),
                "--finetune_objective", "contrastive", "--cl_k", str(c["cl_k"]), "--cl_beta", str(c["cl_beta"]),
                "--cl_margin", str(c["cl_margin"]), "--cl_grad_clip", str(c["cl_grad_clip"]), "--lr", str(c["lr"]),
                "--minibatch_size", str(c["minibatch_size"]), "--num_envs", str(c["num_envs"]),
                "--reset_env_timestep", str(c["reset_env_timestep"]),
                "--validate_timestep", str(sm.get("validate_timestep", pb["validate_timestep"])),
                "--converge_window", "0", "--max_updates", str(upd),
                "--chunk_updates", str(0 if self.smoke else pb["chunk"]),
                "--budget_checkpoints", ",".join(map(str, bc)), "--cl_log_drift", "True",
                "--merge_mix", ",".join(str(REPO / sp[o]) for o in "mct"), "--resume", "True", "--device", "cuda"]

    def unit(self, arm, slot):
        name = self.name(arm)
        rdir = self.res / "part_b" / "runs" / name
        if (rdir / "done.json").exists():
            self.log(f"skip {name} (done)"); return True
        rdir.mkdir(parents=True, exist_ok=True)
        live, out = TL / f"{name}.status.json", TL / f"{name}.stdout.log"
        TL.mkdir(parents=True, exist_ok=True)
        fails, meta_f = 0, rdir / "process.json"
        chunk = 0 if self.smoke else self.cfg["part_b"]["chunk"]
        while True:
            st = json.loads(live.read_text())["status"] if live.exists() else "NONE"
            if st in ("FINISHED", "CONVERGED", "NOT_CONVERGED"):
                break
            self.log(f"train {name}: chunk (status {st}, slot {slot})")
            t0 = time.time()
            rc = self.run([PY, *self.args(arm)], slot, int(chunk * 15 + 1800) if chunk else 3600, out)
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
        cdir = self.ckpt / "part_b"; cdir.mkdir(parents=True, exist_ok=True)
        bc = (self.cfg["smoke"] if self.smoke else self.cfg["part_b"])["budget_checkpoints"]
        for u in bc:
            shutil.copy(TN / f"{name}@u{u}.pth", cdir / f"{name}@u{u}.pth")
        shutil.copy(TN / f"{name}.pth", cdir / f"{name}_best.pth")
        s_ = json.loads(live.read_text())
        self.log(f"{name}: {s_['status']} at update {s_['update']}, best validation HV {-s_['best_value']:.4f} @ "
                 f"{s_['best_update']}, {s_['train_seconds'] / 3600:.2f} h")
        edir = self.res / "part_b" / "eval"
        for model in (cdir / f"{name}@u{bc[-1]}.pth", cdir / f"{name}_best.pth"):
            if self.evaluate(model, edir, slot, out):
                (rdir / "FAILED").write_text("evaluation failed twice\n")
                self.commit(f"{self.prefix}: {name} eval FAILED", rdir); return False
        (rdir / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        self.write_sums(cdir)
        self.commit(f"{self.prefix}: {name} ({s_['status']})", rdir, edir, cdir)
        return True

    def part_b(self):
        order = self.cfg["part_b"]["order"]
        self.log(f"Part B: {len(order)} runs, {self.jobs} parallel slots")
        free, lock, ok = list(range(self.jobs)), threading.Lock(), []

        def work(arm):
            with lock:
                slot = free.pop(0)
            try:
                ok.append(self.unit(arm, slot))
            finally:
                with lock:
                    free.append(slot)
        with ThreadPoolExecutor(self.jobs) as ex:
            list(ex.map(work, order))
        rc = self.analyze()
        self.log(f"runs ok {sum(ok)}/{len(order)}")
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
