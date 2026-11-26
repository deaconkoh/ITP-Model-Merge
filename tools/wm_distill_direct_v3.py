"""Experiments 2 and 3 (10x5, seed 111, waiting allowed): Distillation from, and direct multi-objective training on,
the immediate physical (C) and joint latent + physical (D) world-model bases. Exploratory, single seed.

Experiment 2, Distillation (C-Distillation, D-Distillation). Teachers: the existing makespan, carbon and tardiness
specialists of results|checkpoints/rs_four_bases_v3 (trained policies, not world-model heads). Stage 1 and the
subsequent multi-objective fine-tuning are the specialist-reuse baseline's own commands (tools/specialist_reuse_v3_runner
.Runner.distill_cmd / method_args("distill"), baseline SELECTED lr 3e-4, beta 0.1, not re-tuned): same student
(fresh preference-conditioned DANIEL, torch seed 111 + 7001; NOT initialised from any world-model encoder), same
teacher-data instances (GeneratorSource seed 111 + 7001, 400 per teacher), budget, objective, stopping rule, validation
and checkpoint selection. Only the teachers differ.
References: A = results/specialist_reuse_v3 (final:distill), B = results/specialist_reuse_wm_td_v3 (final:distill).

Experiment 3, Direct MO (C-Direct, D-Direct). The From-scratch DANIEL command of the baseline (selected lr 1e-3, beta
0.1) + --init_from <start> --init_mode exact, where <start> is the baseline's own freshly initialised network
(checkpoints/daniel_wm_direct_v3/scratch_init_s111.pth, hash-checked against the baseline From-scratch run) with ONLY
feature_exact replaced by the C / D epoch-60 encoder (taken from the shared bases, which carry exactly those encoders).
No specialists, no merging, no world-model losses. References: A = results/specialist_reuse_v3 (final:scratch),
B = results/daniel_wm_direct_v3.

Evaluation: tools/specialist_reuse_v3_eval.py (100 evaluation instances x 66 preferences + equal weights, best-of-128
seed 51), for the distilled students before fine-tuning (C, D and the A, B references) and for every final.

    python tools/wm_distill_direct_v3.py [--summarise]   # -> results|checkpoints/wm_distill_direct_v3
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_specialist_reuse_v3 as F                              # noqa: E402
import specialist_reuse_v3_runner as R                              # noqa: E402

NAME = "wm_distill_direct_v3"
RES, CK = REPO / "results" / NAME, REPO / "checkpoints" / NAME
A_RES, A_CK = REPO / "results" / "specialist_reuse_v3", REPO / "checkpoints" / "specialist_reuse_v3"
TD_RES, TD_CK = REPO / "results" / "specialist_reuse_wm_td_v3", REPO / "checkpoints" / "specialist_reuse_wm_td_v3"
WD_RES, WD_CK = REPO / "results" / "daniel_wm_direct_v3", REPO / "checkpoints" / "daniel_wm_direct_v3"
RS4_RES, RS4_CK = REPO / "results" / "rs_four_bases_v3", REPO / "checkpoints" / "rs_four_bases_v3"
DS_CK = REPO / "checkpoints" / "tardiness_wm_downstream_v3"
NEW = {"C": {"tag": "srip", "base": DS_CK / "start_dsC.pth",
             "specs": [RS4_CK / "C/specialists/10x5+mix+bv3_srip_spec_m_s111.pth",
                       RS4_CK / "C/specialists/10x5+mix+bv3_srip_spec_c_s111.pth",
                       RS4_CK / "C/specialists/10x5+mix+bv3_wmtp_dsC_s111.pth"]},
       "D": {"tag": "srjp", "base": DS_CK / "start_dsD.pth",
             "specs": [RS4_CK / f"D/specialists/10x5+mix+bv3_srjp_spec_{o}_s111.pth" for o in "mct"]}}
_load = F.load
sha = R.sha


def jload(p):
    return json.loads(Path(p).read_text())


def net_sha(sd):
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(sd.items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Arm(R.Runner):
    def __init__(self, a, key, method):
        F.load = lambda check=True: {**_load(check), "name": f"{NAME}/{key}_{method}"}
        super().__init__(a)
        F.load = _load
        self.key, self.method = key, method
        self.tag = NEW[key]["tag"]
        self.prefix = f"{NAME} {key} {method}"
        self.bstate = jload(A_RES / "state.json")
        self.start = self.ckpt / "start_s111.pth"

    def commit(self, msg, *paths):
        pass

    def specs(self):
        return [str(p) for p in NEW[self.key]["specs"]]

    def suffix(self):
        return "distill" if self.method == "distill" else "direct"

    def cmd(self):
        if self.method == "distill":
            sel = self.bstate["selected"]["distill"]
            return self.method_args("distill", "distill", sel["lr"], sel["cl_beta"], full=True)
        sel = self.bstate["selected"]["scratch"]
        return [*self.method_args("scratch", "direct", sel["lr"], sel["cl_beta"], full=True),
                "--init_from", str(self.start), "--init_mode", "exact"]

    def command_check(self):
        """New command vs the baseline's own command rebuilt by the same runner (tag sr, baseline files)."""
        new = self.cmd()
        tag, ck = self.tag, self.ckpt
        self.tag, self.ckpt = "sr", A_CK
        if self.method == "distill":
            sel = self.bstate["selected"]["distill"]
            old = self.method_args("distill", "distill", sel["lr"], sel["cl_beta"], full=True)
        else:
            sel = self.bstate["selected"]["scratch"]
            old = self.method_args("scratch", "scratch", sel["lr"], sel["cl_beta"], full=True)
        self.tag, self.ckpt = tag, ck
        n = min(len(old), len(new))
        flags = sorted({old[i - 1] for i in range(n) if old[i] != new[i]})
        allowed = {"--model_suffix", "--init_from"}
        ok = set(flags) <= allowed and (len(new) == len(old) if self.method == "distill" else
                                        new[len(old):][0::2] == ["--init_from", "--init_mode"] and new[len(old) + 3] == "exact")
        return {"differing_flags": flags, "added": new[len(old):], "matches": ok}

    # ------------------------------------------------------------------ Experiment 3 start
    def build_start(self):
        import torch
        rec = self.res / "start_check.json"
        if self.start.exists() and rec.exists():
            return jload(rec)
        scratch = torch.load(WD_CK / "scratch_init_s111.pth", map_location="cpu", weights_only=False)
        sd = scratch["state_dict"]
        base = torch.load(NEW[self.key]["base"], map_location="cpu", weights_only=False)["state_dict"]
        fe = {k for k in sd if k.startswith("feature_exact.")}
        assert fe == {k for k in base if k.startswith("feature_exact.")} and all(sd[k].shape == base[k].shape for k in fe)
        new = dict(scratch); new["state_dict"] = {k: (base[k].clone() if k in fe else v.clone()) for k, v in sd.items()}
        new["update"] = 0
        torch.save(new, self.start)
        a = new["state_dict"]
        wd = jload(WD_RES / "start_check.json")
        enc_src = {"C": "checkpoints/world_model_immediate_pilot_v3/B/epoch60/online_encoder.pth",
                   "D": "checkpoints/world_model_immediate_pilot_v3/E/epoch60/online_encoder.pth"}[self.key]
        ip = torch.load(REPO / enc_src, map_location="cpu", weights_only=False)
        chk = {"scratch_init_sha_matches_wm_direct_sums": jload(WD_CK / "SHA256SUMS.json")["scratch_init_s111.pth"]
               == sha(WD_CK / "scratch_init_s111.pth"),
               "scratch_init_hash_equals_baseline_scratch_reference": net_sha(sd) == wd["baseline_scratch_reference_hash"],
               "encoder_equals_epoch60_world_model_encoder": all(torch.equal(a[k], ip[k[len("feature_exact."):]]) for k in fe),
               "encoder_differs_from_scratch": any(not torch.equal(a[k], sd[k]) for k in fe),
               "non_encoder_identical_to_scratch_init": all(torch.equal(a[k], sd[k]) for k in sd if k not in fe),
               "encoder_source": enc_src, "start_sha256": sha(self.start)}
        assert all(v for v in chk.values() if isinstance(v, bool)), chk
        rec.write_text(json.dumps(chk, indent=1) + "\n")
        self.write_sums()
        return chk

    def teacher_check(self):
        n = jload(self.ckpt / "distill" / "student_s111.distill.json")
        o = jload(A_CK / "distill" / "student_s111.distill.json")
        keys = set(n) | set(o)
        return {"record_keys_equal": sorted(k for k in keys if n.get(k) == o.get(k)),
                "record_keys_differing": sorted(k for k in keys if n.get(k) != o.get(k)),
                "teachers": self.specs()}

    def record_check(self):
        name = self.model(self.suffix())
        n = jload(self.res / "runs" / name / "run.json")
        ref = "10x5+mix+bv3_sr_distill_s111" if self.method == "distill" else "10x5+mix+bv3_sr_scratch_s111"
        o = jload(A_RES / "runs" / ref / "run.json")
        keys = set(n["config"]) | set(o["config"])
        diff = sorted(k for k in keys if n["config"].get(k) != o["config"].get(k))
        fp = n["data_fingerprints"] == o["data_fingerprints"]
        return {"differing_config_keys": diff, "data_fingerprints_equal": fp,
                "matches": set(diff) <= {"model_suffix", "init_from"} and fp}

    def unit(self):
        def keep(name):
            d = self.ckpt / "finals"; d.mkdir(exist_ok=True)
            shutil.copy2(R.TN / f"{name}.pth", d / f"{name}.pth"); self.write_sums()
        key = f"final:{self.suffix()}"
        return (self, key, self.cmd(), self.model(self.suffix()), keep)


def pool(units, slots):
    free, lock, ok = list(range(slots)), threading.Lock(), []

    def work(u):
        r, key, cmd, name, after = u
        with lock:
            slot = free.pop(0)
        try:
            res = r.train_unit(key, cmd, name, slot)
            if res and after:
                after(name)
            ok.append(res)
        finally:
            with lock:
                free.append(slot)
    with ThreadPoolExecutor(slots) as ex:
        list(ex.map(work, units))
    return all(ok)


def evaluate(models):
    f = RES / "eval_seconds.json"
    st = jload(f) if f.exists() else {}
    for key, model in models.items():
        out = RES / "eval" / f"{key}.npz"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(RES / "eval.log", "a") as fh:
            rc = subprocess.run(["taskset", "-c", "0-11", R.PY, str(REPO / "tools" / "specialist_reuse_v3_eval.py"),
                                 "--model", str(model), "--out", str(out)], cwd=REPO, env=R.ENV, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit(f"evaluation {key} failed")
        st[key] = time.time() - t0
        f.write_text(json.dumps(st, indent=1) + "\n")
        print(f"evaluated {key}", flush=True)


def main_run(a):
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    arms = {(k, m): Arm(a, k, m) for k in NEW for m in ("direct", "distill")}
    checks = {}
    for (k, m), r in arms.items():
        c = {"command": r.command_check()}
        assert c["command"]["matches"], c
        if m == "direct":
            c["start"] = r.build_start()
        checks[f"{k}_{m}"] = c
    (RES / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    print("checks passed", flush=True)
    t1 = time.time()
    for k in NEW:                                                   # stage 1 (one at a time, ~90 s each)
        r = arms[(k, "distill")]
        r.stage_distill()
        checks[f"{k}_distill"]["teacher_inputs"] = r.teacher_check()
    (RES / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    t2 = time.time()
    units = [arms[(k, "direct")].unit() for k in NEW] + [arms[(k, "distill")].unit() for k in NEW]   # slowest first
    if not pool(units, 3):
        raise SystemExit("training had failures")
    t3 = time.time()
    models = {f"{k}_student": arms[(k, "distill")].ckpt / "distill" / "student_s111.pth" for k in NEW}
    models.update({"A_student": A_CK / "distill" / "student_s111.pth", "B_student": TD_CK / "distill" / "student_s111.pth"})
    for (k, m), r in arms.items():
        models[f"{k}_{m}"] = r.ckpt / "finals" / f"{r.model(r.suffix())}.pth"
    evaluate(models)
    t4 = time.time()
    for (k, m), r in arms.items():
        checks[f"{k}_{m}"]["record"] = r.record_check()
    (RES / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    assert all(c["record"]["matches"] for c in checks.values()), checks
    w = {"stage1_s": t2 - t1, "training_s": t3 - t2, "evaluation_s": t4 - t3, "total_s": t4 - t0,
         "concurrency": "C/D Direct and C/D Distillation fine-tuning: 4 runs on 3 CPU slots (4 cores each) and one GPU"}
    (RES / "wall.json").write_text(json.dumps(w, indent=1) + "\n")
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check_only", action="store_true")
    a = ap.parse_args()
    a.smoke, a.jobs = False, 3
    if a.check_only:
        RES.mkdir(parents=True, exist_ok=True)
        for k in NEW:
            for m in ("direct", "distill"):
                r = Arm(a, k, m)
                print(k, m, json.dumps(r.command_check()))
        return
    main_run(a)


if __name__ == "__main__":
    main()
