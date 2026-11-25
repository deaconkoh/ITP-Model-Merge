"""Rewarded Soups from four shared-base initialisations (10x5, seed 111, waiting allowed). Exploratory, single seed.

  A random                    results|checkpoints/specialist_reuse_v3 (reused, read-only)
  B outcome prediction        results|checkpoints/specialist_reuse_wm_v3 (reused, read-only)
  C immediate physical        new: shared base checkpoints/tardiness_wm_downstream_v3/start_dsC.pth
  D joint latent + physical   new: shared base checkpoints/tardiness_wm_downstream_v3/start_dsD.pth

C and D follow the specialist-reuse protocol exactly: every command is built by the baseline runner
(tools/specialist_reuse_v3_runner.Runner) - specialist recipe and stopping rule (0.5% over 1,000 updates, minimum
1,000, cap 8,000), merge checks, Rewarded Soups with the baseline's SELECTED setting (lr 1e-3, beta 0.1; not re-tuned),
and the evaluation script (100 evaluation instances x 66 preferences, best-of-128, seed 51). Only the run tag, the
output folders and the shared base change. The C/D bases are the downstream tardiness experiment's start bundles
(the random base with the encoder replaced by the immediate pilot's epoch-60 encoders); they are read in place so that
the tardiness runs already trained from them stay valid continuations.

Tardiness specialists (audit, before any training):
  C  the downstream run 10x5+mix+bv3_wmtp_dsC_s111 is the complete specialist: its saved run configuration equals the
     specialist-reuse command except the stopping flags, and the reuse rule replayed on its validation log stops at
     update 2,000 exactly (no improvement over updates 1,000-2,000), which is where the run ended. Adopted as is.
  D  the downstream run has not met the rule by update 2,000; it is continued bit-for-bit from its saved resume state
     (weights, optimiser, RNG, instance batch, logs) under the reuse rule (train.py --resume_from), as a new run
     name. The resumed policy must reproduce the last logged validation value exactly or nothing is trained.
Makespan and carbon specialists of C and D are trained fresh. The merged start (update 0 of Rewarded Soups) is
snapshotted (--budget_checkpoints 0; no effect on training) and evaluated, as eval_curve_v3 did for A and B.

    python tools/rs_four_bases_v3.py [--summarise]   # -> results|checkpoints/rs_four_bases_v3, results/rs_four_bases_v3.md
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "daniel"))
import freeze_specialist_reuse_v3 as F                              # noqa: E402
import specialist_reuse_v3_runner as R                              # noqa: E402

NAME = "rs_four_bases_v3"
RES, CK = REPO / "results" / NAME, REPO / "checkpoints" / NAME
MD = REPO / "results" / f"{NAME}.md"
A_RES, A_CK = REPO / "results" / "specialist_reuse_v3", REPO / "checkpoints" / "specialist_reuse_v3"
B_RES, B_CK = REPO / "results" / "specialist_reuse_wm_v3", REPO / "checkpoints" / "specialist_reuse_wm_v3"
DS_RES, DS_CK = REPO / "results" / "tardiness_wm_downstream_v3", REPO / "checkpoints" / "tardiness_wm_downstream_v3"
EC_RES = REPO / "results" / "eval_curve_v3"
IP_CK = REPO / "checkpoints" / "world_model_immediate_pilot_v3"
TN, TL = R.TN, R.TL
sha = R.sha
_load = F.load
LABELS = {"A": "A random", "B": "B outcome prediction", "C": "C immediate physical", "D": "D joint latent + physical"}
NEW = {"C": {"tag": "srip", "ds": "dsC", "pilot": "B", "t_mode": "adopt"},
       "D": {"tag": "srjp", "ds": "dsD", "pilot": "E", "t_mode": "extend"}}
STOP_KEYS = {"converge_window", "converge_tol", "min_updates", "max_updates", "budget_checkpoints"}


def ds_name(k):
    return f"10x5+mix+bv3_wmtp_{NEW[k]['ds']}_s111"


class Cfg(R.Runner):
    """One new configuration (C or D), built by the baseline runner."""

    def __init__(self, a, key):
        F.load = lambda check=True: {**_load(check), "name": f"{NAME}/{key}"}
        super().__init__(a)
        F.load = _load
        self.key = key
        self.tag = NEW[key]["tag"]
        self.prefix = f"{NAME} {key}"
        self.base = DS_CK / f"start_{NEW[key]['ds']}.pth"           # read-only, in place (see module docstring)

    def commit(self, msg, *paths):
        pass                                                         # committed once at the end, by hand

    def t_name(self):
        return ds_name(self.key) if NEW[self.key]["t_mode"] == "adopt" else self.model("spec_t")

    def specs(self):
        d = self.ckpt / "specialists"
        return [str(d / f"{self.model('spec_m')}.pth"), str(d / f"{self.model('spec_c')}.pth"),
                str(d / f"{self.t_name()}.pth")]

    def units(self):
        u = [(self, f"spec:{o}", self.specialist_args(o), self.model(f"spec_{o}"), self.keep_specialist) for o in "mc"]
        if NEW[self.key]["t_mode"] == "extend":
            cmd = self.specialist_args("t") + ["--resume_from", str(TN / f"{ds_name(self.key)}.resume.pth")]
            u.append((self, "spec:t", cmd, self.model("spec_t"), self.keep_specialist))
        return u

    def adopt_t(self):
        """C: the downstream tardiness run is the complete specialist (stops at 2,000 under the reuse rule)."""
        name = ds_name(self.key)
        src, dst = TN / f"{name}.pth", self.ckpt / "specialists" / f"{name}.pth"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst); self.write_sums()
        assert sha(dst) == sha(DS_CK / NEW[self.key]["ds"] / "selected.pth")
        st = json.loads((TL / f"{name}.status.json").read_text())
        rdir = self.res / "runs" / name; rdir.mkdir(parents=True, exist_ok=True)
        for f, d in ((f"valiquality_{name}.json", "valiquality.json"), (f"{name}.run.json", "run.json"),
                     (f"{name}.status.json", "status.json"), (f"reward_{name}.json", "reward.json")):
            shutil.copy2(TL / f, rdir / d)
        self.state["units"]["spec:t"] = {"name": name, "status": "CONVERGED (reuse rule replayed; run ended at 2000)",
                                         "update": st["update"], "best": st["best_value"],
                                         "best_update": st["best_update"], "train_seconds": st["train_seconds"],
                                         "capped": False, "source": "tardiness_wm_downstream_v3 (adopted)"}
        self.save_state()

    def rs_unit(self):
        sel = json.loads((A_RES / "state.json").read_text())["selected"]["rs"]
        self.state["selected"] = {"rs": sel, "source": "specialist_reuse_v3 state.json (not re-tuned)"}; self.save_state()

        def keep(name):
            d = self.ckpt / "finals"; d.mkdir(exist_ok=True)
            shutil.copy2(TN / f"{name}.pth", d / f"{name}.pth")
            shutil.copy2(TN / f"{name}@u0.pth", d / f"{name}@u0.pth"); self.write_sums()
        cmd = self.method_args("rs", "rs", sel["lr"], sel["cl_beta"], full=True) + ["--budget_checkpoints", "0"]
        return (self, "final:rs", cmd, self.model("rs"), keep)


# ---------------------------------------------------------------------------------------------- audit
def audit():
    import torch
    sys.path.insert(0, str(REPO / "daniel"))
    from convergence import is_converged
    out = {}
    base = torch.load(A_CK / "base_s111.pth", map_location="cpu", weights_only=False)["state_dict"]
    fe = {k for k in base if k.startswith("feature_exact.")}
    for k, c in NEW.items():
        s = torch.load(DS_CK / f"start_{c['ds']}.pth", map_location="cpu", weights_only=False)["state_dict"]
        out[f"{k}_base"] = {
            "sha256": sha(DS_CK / f"start_{c['ds']}.pth"),
            "sha_matches_downstream_sums": json.loads((DS_CK / "SHA256SUMS.json").read_text())[f"start_{c['ds']}.pth"]
            == sha(DS_CK / f"start_{c['ds']}.pth"),
            "same_keys_as_random_base": set(s) == set(base),
            "identical_outside_encoder_to_random_base": all(torch.equal(s[x], base[x]) for x in base if x not in fe),
            "encoder_differs_from_random_base": any(not torch.equal(s[x], base[x]) for x in fe),
            "encoder_source": str((IP_CK / c["pilot"] / "epoch60" / "online_encoder.pth").relative_to(REPO)),
            "encoder_equals_epoch60_checkpoint": json.loads((DS_RES / "start_check.json").read_text())["start"][
                f"{c['ds']}_encoder_equals_epoch60_online_encoder"]}
    # the downstream tardiness runs are prefixes of the specialist-reuse tardiness runs (A, B: exact reproductions)
    L = lambda f: json.loads(Path(f).read_text())                                    # noqa: E731
    for k, old, ds in (("A", A_RES / "runs/10x5+mix+bv3_sr_spec_t_s111/valiquality.json", "dsA"),
                       ("B", B_RES / "runs/10x5+mix+bv3_srwm_spec_t_s111/valiquality.json", "dsB")):
        vo, vn = L(old), L(DS_RES / f"{ds}_valiquality.json")
        out[f"{k}_downstream_tardiness_equals_reuse_specialist_prefix"] = all(
            vo[i]["selection_metric"] == vn[i]["selection_metric"] for i in range(len(vn)))
    ref = L(A_RES / "runs/10x5+mix+bv3_sr_spec_t_s111/run.json")["config"]
    rule = F.load()["stopping"]
    for k in NEW:
        cfgk = L(DS_RES / f"{NEW[k]['ds']}_run.json")["config"]
        diff = sorted(x for x in set(ref) | set(cfgk) if ref.get(x) != cfgk.get(x))
        v = L(DS_RES / f"{NEW[k]['ds']}_valiquality.json")
        stop = None
        for i in range(len(v)):
            conv, _ = is_converged([(e["update"], e["selection_metric"]) for e in v[:i + 1]], rule["converge_window"],
                                   rule["converge_tol"], rule["min_updates"])
            if conv:
                stop = v[i]["update"]; break
        rs_ = L(TL / f"{ds_name(k)}.status.json")
        out[f"{k}_tardiness"] = {"run": ds_name(k), "config_keys_differing_from_reuse_tardiness_specialist": diff,
                                 "only_start_name_and_stopping_differ": set(diff) <= {"model_suffix", "init_from"} | STOP_KEYS,
                                 "updates_run": rs_["update"], "status": rs_["status"],
                                 "reuse_rule_stop_update_replayed": stop,
                                 "decision": ("adopt: the reuse rule stops exactly where the run ended" if stop == rs_["update"]
                                              else "extend: continue from the saved resume state under the reuse rule"
                                              if stop is None else "truncate"),
                                 "resume_state_sha256": sha(TN / f"{ds_name(k)}.resume.pth")}
    out["C_tardiness"]["ok"] = out["C_tardiness"]["decision"].startswith("adopt") and NEW["C"]["t_mode"] == "adopt"
    out["D_tardiness"]["ok"] = out["D_tardiness"]["decision"].startswith("extend") and NEW["D"]["t_mode"] == "extend"
    # A and B are reusable as they are: same runner and protocol (B's own protocol check), and merged-start evaluations
    bst = L(B_RES / "state.json")
    out["B_protocol_check_vs_A"] = {k: v["matches"] for k, v in bst["protocol_check"].items()}
    ec = L(EC_RES / "summary.json")["runs"]
    out["merged_start_evaluations_reused"] = {
        r: {"u0_best_of_128_hv": ec[r]["points"][0]["best_of_128_hv"],
            "rerun_reproduced_original": ec[r]["reproduction"]["validation_curve_identical"]}
        for r in ("rs_random", "rs_world_model")}
    for k, res, ck, tag in (("A", A_RES, A_CK, "sr"), ("B", B_RES, B_CK, "srwm")):
        import numpy as np
        e = np.load(res / "eval" / "rs.npz")
        out[f"{k}_final_eval_matches_checkpoint"] = str(e["model_sha256"]) == sha(ck / "finals" / f"10x5+mix+bv3_{tag}_rs_s111.pth")
    flat = [v for v in out.values() if isinstance(v, bool)]
    for v in out.values():
        if isinstance(v, dict):
            flat += [x for kk, x in v.items() if isinstance(x, bool)]
            flat += [x for x in v.values() if isinstance(x, dict) for x in x.values() if isinstance(x, bool)]
    out["all_pass"] = all(flat)
    (RES / "audit.json").write_text(json.dumps(out, indent=1) + "\n")
    assert out["all_pass"], json.dumps(out, indent=1)
    return out


# ---------------------------------------------------------------------------------------------- orchestration
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


def evaluate(cfgs):
    st_f = RES / "eval_state.json"
    st = json.loads(st_f.read_text()) if st_f.exists() else {}
    for k, r in cfgs.items():
        for which, model in (("final", r.ckpt / "finals" / f"{r.model('rs')}.pth"),
                             ("merged_start", r.ckpt / "finals" / f"{r.model('rs')}@u0.pth")):
            out = RES / "eval" / f"{k}_{which}.npz"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(RES / "eval.log", "a") as f:
                rc = subprocess.run(["taskset", "-c", "0-11", R.PY, str(REPO / "tools" / "specialist_reuse_v3_eval.py"),
                                     "--model", str(model), "--out", str(out)], cwd=REPO, env=R.ENV, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise SystemExit(f"evaluation {k} {which} failed")
            st[f"{k}_{which}"] = time.time() - t0
            st_f.write_text(json.dumps(st, indent=1) + "\n")


def protocol_check(cfgs):
    L = lambda f: json.loads(Path(f).read_text())                                    # noqa: E731
    out = {}
    for k, r in cfgs.items():
        rows = {}
        for o in "mct":
            name = r.t_name() if o == "t" else r.model(f"spec_{o}")
            n = L(r.res / "runs" / name / "run.json")["config"]
            b = L(A_RES / "runs" / f"10x5+mix+bv3_sr_spec_{o}_s111" / "run.json")["config"]
            diff = sorted(x for x in set(n) | set(b) if n.get(x) != b.get(x))
            allowed = {"model_suffix", "init_from"} | ({"resume_from"} if o == "t" else set()) | \
                      (STOP_KEYS if (o == "t" and NEW[k]["t_mode"] == "adopt") else set())
            rows[f"spec_{o}"] = {"differing_config_keys": diff, "matches": set(diff) <= allowed}
        n = L(r.res / "runs" / r.model("rs") / "run.json")["config"]
        b = L(A_RES / "runs" / "10x5+mix+bv3_sr_rs_s111" / "run.json")["config"]
        diff = sorted(x for x in set(n) | set(b) if n.get(x) != b.get(x))
        rows["rs"] = {"differing_config_keys": diff, "matches": set(diff) <= {"model_suffix", "merge_mix", "budget_checkpoints"}}
        if NEW[k]["t_mode"] == "extend":
            ext = L(TL / f"{r.model('spec_t')}.extension.json")
            shutil.copy2(TL / f"{r.model('spec_t')}.extension.json", r.res / "runs" / r.model("spec_t") / "extension.json")
            rows["spec_t_extension"] = {"exact": ext["exact"], "next_update": ext["next_update"],
                                        "changed": ext["changed"],
                                        "source_resume_state_unchanged": sha(TN / f"{ds_name(k)}.resume.pth")
                                        == json.loads((RES / "audit.json").read_text())[f"{k}_tardiness"]["resume_state_sha256"]}
        out[k] = rows
    (RES / "protocol_check.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main_run(a):
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    au = audit()
    print("audit passed", flush=True)
    cfgs = {k: Cfg(a, k) for k in NEW}
    for k, r in cfgs.items():
        if NEW[k]["t_mode"] == "adopt":
            r.adopt_t()
    units = [u for r in cfgs.values() for u in r.units()]
    order = {"spec:t": 0, "spec:m": 1, "spec:c": 2}                          # slowest first
    units.sort(key=lambda u: order[u[1]])
    t1 = time.time()
    if not pool(units, 3):
        raise SystemExit("specialist training had failures")
    t2 = time.time()
    for r in cfgs.values():
        r.stage_checks()
    if not pool([r.rs_unit() for r in cfgs.values()], 2):
        raise SystemExit("Rewarded Soups run failed")
    t3 = time.time()
    evaluate(cfgs)
    t4 = time.time()
    protocol_check(cfgs)
    wall = json.loads((RES / "wall.json").read_text()) if (RES / "wall.json").exists() else {}
    wall.setdefault("invocations", []).append({"specialists_s": t2 - t1, "rewarded_soups_s": t3 - t2,
                                               "evaluation_s": t4 - t3, "total_s": t4 - t0,
                                               "audit_pass": au["all_pass"]})
    (RES / "wall.json").write_text(json.dumps(wall, indent=1) + "\n")
    summarise()


# ---------------------------------------------------------------------------------------------- summary
def metrics(npz):
    import numpy as np
    import analyze_specialist_reuse_v3 as AN
    cfg = _load()
    norm = json.loads((REPO / cfg["evaluation"]["norm"]).read_text())["instances"]
    e = np.load(npz); keys = list(e["keys"])
    hv, igd, n = AN.metrics(e["sample128_obj"], keys, norm, REPO / cfg["evaluation"]["igd_front"])
    ghv, gigd, _ = AN.metrics(e["greedy_obj"], keys, norm, REPO / cfg["evaluation"]["igd_front"])
    return {"best_of_128_hv": hv, "best_of_128_igd_plus": igd, "igd_instances": n, "greedy_hv": ghv,
            "greedy_igd_plus": gigd, "instances": len(keys), "eval_seconds": float(e["greedy_seconds"]) + float(e["sample128_seconds"]),
            "model_sha256": str(e["model_sha256"])}


def summarise():
    L = lambda f: json.loads(Path(f).read_text())                                    # noqa: E731
    ds = L(DS_RES / "summary.json")["rows"]
    pre = {"A": 0.0, "B": ds["dsB"]["pretraining_hours"], "C": ds["dsC"]["pretraining_hours"],
           "D": ds["dsD"]["pretraining_hours"]}
    src = {"A": (A_RES, EC_RES / "eval" / "rs_random_u0.npz"), "B": (B_RES, EC_RES / "eval" / "rs_world_model_u0.npz"),
           "C": (RES / "C", RES / "eval" / "C_merged_start.npz"), "D": (RES / "D", RES / "eval" / "D_merged_start.npz")}
    final = {"A": A_RES / "eval" / "rs.npz", "B": B_RES / "eval" / "rs.npz",
             "C": RES / "eval" / "C_final.npz", "D": RES / "eval" / "D_final.npz"}
    a_state = L(A_RES / "state.json")
    rows = {}
    for k in "ABCD":
        st = L(src[k][0] / "state.json")["units"]
        spec = {o: st[f"spec:{o}"] for o in "mct"}
        h = {o: spec[o]["train_seconds"] / 3600 for o in "mct"}
        rs = st["final:rs"]
        fin, start = metrics(final[k]), metrics(src[k][1])
        rows[k] = {
            "label": LABELS[k],
            "best_of_128_hv_before_fine_tuning": start["best_of_128_hv"],
            "best_of_128_hv_after_fine_tuning": fin["best_of_128_hv"],
            "best_of_128_igd_plus_after": fin["best_of_128_igd_plus"],
            "best_of_128_igd_plus_before": start["best_of_128_igd_plus"],
            "greedy_hv_after": fin["greedy_hv"], "igd_instances": fin["igd_instances"],
            "pretraining_hours": pre[k],
            "specialist_hours": h, "specialist_hours_total": sum(h.values()),
            "specialist_hours_longest": max(h.values()),
            "specialists": {o: {"updates": spec[o]["update"], "status": spec[o]["status"], "best": spec[o]["best"],
                                "best_update": spec[o]["best_update"]} for o in "mct"},
            "rs_hours": rs["train_seconds"] / 3600, "rs_updates": rs["update"], "rs_best_update": rs["best_update"],
            "rs_status": rs["status"], "rs_best_validation_hv": -rs["best"],
            "total_hours": pre[k] + sum(h.values()) + rs["train_seconds"] / 3600,
            "critical_path_hours": pre[k] + max(h.values()) + rs["train_seconds"] / 3600,
            "evaluation_hours": (fin["eval_seconds"] + start["eval_seconds"]) / 3600,
            "final_model_sha256": fin["model_sha256"]}
    tuning = {"rs_trials_hours": sum(u["train_seconds"] for kk, u in a_state["units"].items()
                                     if kk.startswith("trial:rs:")) / 3600,
              "trials": sum(1 for kk in a_state["units"] if kk.startswith("trial:rs:")),
              "note": "Rewarded Soups lr / beta chosen once on the random base (specialist_reuse_v3); reused unchanged by B, C, D"}
    out = {"rows": rows, "tuning": tuning,
           "protocol_check": L(RES / "protocol_check.json") if (RES / "protocol_check.json").exists() else None,
           "audit_pass": L(RES / "audit.json")["all_pass"],
           "wall": L(RES / "wall.json") if (RES / "wall.json").exists() else None}
    (RES / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    plot(rows)
    report(out)
    print(json.dumps(out, indent=1))


INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}          # categorical slots 1-4, fixed order
PARTS = (("pretraining_hours", "world-model pretraining", "#184f95"),
         ("specialist_hours_total", "specialist training (sum of 3)", "#3987e5"),
         ("rs_hours", "Rewarded Soups fine-tuning", "#9ec5f4"))


def _style(ax):
    ax.grid(color=GRID, lw=.8, axis="y"); ax.set_axisbelow(True); ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    ks = list("ABCD"); labs = [LABELS[k].replace(" ", "\n", 1) for k in ks]
    foot = "Single seed (111), 10x5, waiting allowed. 100 evaluation instances x 66 preferences, best-of-128 (seed 51)."
    # 1 final HV
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURF)
    hv = [rows[k]["best_of_128_hv_after_fine_tuning"] for k in ks]
    h0 = [rows[k]["best_of_128_hv_before_fine_tuning"] for k in ks]
    ax.bar(range(4), hv, color=[COL[k] for k in ks], width=.62, edgecolor=SURF, lw=2)
    ax.scatter(range(4), h0, marker="_", s=900, color=INK, lw=2, zorder=3, label="merged start, before fine-tuning")
    for i, (v, v0) in enumerate(zip(hv, h0)):
        ax.text(i, v + .002, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + .33, v0, f"{v0:.4f}", ha="left", va="center", fontsize=8, color=MUTED)
    lo = min(h0 + hv)
    ax.set_ylim(lo - .02, max(hv) + .015)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs)
    ax.set_ylabel("best-of-128 HV (evaluation split)")
    ax.set_title("Rewarded Soups: final best-of-128 HV by shared base", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5); _style(ax)
    fig.text(.01, .01, foot + " Axis does not start at 0.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(RES / "final_hv.png", dpi=150, facecolor=SURF); plt.close(fig)
    # 2 stacked time
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURF)
    bottom = [0.0] * 4
    for key, lab, col in PARTS:
        v = [rows[k][key] for k in ks]
        ax.bar(range(4), v, bottom=bottom, color=col, width=.62, edgecolor=SURF, lw=2, label=lab)
        bottom = [b + x for b, x in zip(bottom, v)]
    for i, b in enumerate(bottom):
        ax.text(i, b + .05, f"{b:.2f} h", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylabel("summed training hours")
    ax.set_ylim(0, max(bottom) * 1.45)
    ax.set_title("Total training time (summed durations, pretraining counted once)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3); _style(ax)
    fig.text(.01, .01, "Sums of per-run training seconds; runs shared 12 CPU cores and one GPU with 2-4 concurrent jobs. "
             "Evaluation and tuning excluded.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(RES / "training_time.png", dpi=150, facecolor=SURF); plt.close(fig)
    # 3 scatter
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURF)
    for k in ks:
        x, y = rows[k]["total_hours"], rows[k]["best_of_128_hv_after_fine_tuning"]
        ax.scatter([x], [y], s=90, color=COL[k], edgecolor=SURF, lw=2, zorder=3, label=LABELS[k])
        ax.annotate(LABELS[k], (x, y), xytext=(7, 5), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("total training hours incl. pretraining (summed)"); ax.set_ylabel("final best-of-128 HV")
    ax.set_title("Final HV vs total training cost", loc="left", fontsize=11)
    ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    xs = [rows[k]["total_hours"] for k in ks]
    ax.set_xlim(min(xs) - .8, max(xs) + 1.6)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(.01, .01, foot + " Upper left is better.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(RES / "hv_vs_time.png", dpi=150, facecolor=SURF); plt.close(fig)


def report(out):
    rows = out["rows"]
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"                       # noqa: E731
    L = []
    L += ["# Rewarded Soups from four shared-base initialisations (10x5, waiting allowed, seed 111)", "",
          "Exploratory single-seed comparison. Same specialist recipe, stopping rule, merge, Rewarded Soups setting "
          "(lr 1e-3, beta 0.1, selected once on the random base and not re-tuned), validation and evaluation for all four; "
          "only the shared base's encoder differs. Reserved final-test sets not used.", "",
          "| config | shared-base encoder | source |", "|---|---|---|",
          "| A random | untrained DANIEL (seed 111) | `results/specialist_reuse_v3` (reused) |",
          "| B outcome prediction | final outcomes under heuristic continuation rules (action pilot, pair features, abs+diff) | `results/specialist_reuse_wm_v3` (reused) |",
          "| C immediate physical | immediate physical consequences only (immediate pilot B, epoch 60) | new |",
          "| D joint latent + physical | latent next-state + immediate physical (immediate pilot E, epoch 60) | new |", "",
          "## Audit and reuse (before training; `rs_four_bases_v3/audit.json`)", ""]
    au = json.loads((RES / "audit.json").read_text())
    L += [f"- C and D shared bases equal the random base outside the encoder, and their encoders differ from it: "
          f"{au['C_base']['identical_outside_encoder_to_random_base'] and au['D_base']['identical_outside_encoder_to_random_base']}.",
          f"- The downstream tardiness runs of A and B reproduce the first 2,000 updates of A's and B's specialist-reuse "
          f"tardiness specialists exactly: {au['A_downstream_tardiness_equals_reuse_specialist_prefix']}, "
          f"{au['B_downstream_tardiness_equals_reuse_specialist_prefix']}. So the downstream runs are prefixes of what the "
          "reuse protocol would train.",
          f"- C tardiness: config differs from the reuse specialist only in {au['C_tardiness']['config_keys_differing_from_reuse_tardiness_specialist']}; "
          f"the reuse rule replayed on its validation log stops at update {au['C_tardiness']['reuse_rule_stop_update_replayed']}, "
          "where the run ended, so the run is the complete specialist (selected checkpoint: its best, update "
          f"{rows['C']['specialists']['t']['best_update']}). Not retrained.",
          f"- D tardiness: the rule is not met by update 2,000, so the run was continued from its saved resume state "
          "(weights, optimiser, RNG states, instance batch, logs) under the reuse rule, as a new run name.",
          "- A and B Rewarded Soups results, and their merged-start (update 0) best-of-128 evaluations from "
          "`results/eval_curve_v3`, are reused unchanged (re-runs there reproduced the originals exactly).", ""]
    pc = out["protocol_check"] or {}
    if pc:
        L += ["Protocol check (saved run configs vs A's): " + "; ".join(
            f"{k} {n}: {v['differing_config_keys']}" for k, r in pc.items() for n, v in r.items() if "differing_config_keys" in v)
              + ". All within the allowed keys: " + str(all(v.get("matches", True) for r in pc.values() for v in r.values()))
              + "."]
        if "spec_t_extension" in pc.get("D", {}):
            e = pc["D"]["spec_t_extension"]
            L += [f"D tardiness extension reproduced the logged validation value at update {e['next_update']} exactly: "
                  f"{e['exact']}; source resume state unchanged: {e['source_resume_state_unchanged']}."]
        L += [""]
    L += ["## Comparison", "",
          "| config | best-of-128 HV before fine-tuning | best-of-128 HV after | best-of-128 IGD+ | pretraining h | makespan spec. h | carbon spec. h | tardiness spec. h | specialists total h | RS fine-tuning h | total h incl. pretraining |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in "ABCD":
        r = rows[k]; h = r["specialist_hours"]
        L += [f"| {r['label']} | {f(r['best_of_128_hv_before_fine_tuning'])} | **{f(r['best_of_128_hv_after_fine_tuning'])}** | "
              f"{f(r['best_of_128_igd_plus_after'])} | {f(r['pretraining_hours'], 2)} | {f(h['m'], 2)} | {f(h['c'], 2)} | "
              f"{f(h['t'], 2)} | {f(r['specialist_hours_total'], 2)} | {f(r['rs_hours'], 2)} | {f(r['total_hours'], 2)} |"]
    L += ["", f"IGD+ on the CP-SAT reference subset ({rows['A']['igd_instances']} of the 100 instances); lower is better. "
          "HV: frozen normalisation, reference 1.1, 66 grid preferences.", "",
          "| config | specialist updates m / c / t (status) | RS updates (selected) | RS best validation HV | greedy HV after | IGD+ before | critical path h (specialists in parallel) | evaluation h |",
          "|---|---|---|---|---|---|---|---|"]
    for k in "ABCD":
        r = rows[k]; s = r["specialists"]
        L += [f"| {r['label']} | " + " / ".join(f"{s[o]['updates']}" for o in "mct") +
              f" ({', '.join(str(s[o]['status']).split(' ')[0] for o in 'mct')}) | {r['rs_updates']} ({r['rs_best_update']}) | "
              f"{f(r['rs_best_validation_hv'])} | {f(r['greedy_hv_after'])} | {f(r['best_of_128_igd_plus_before'])} | "
              f"{f(r['critical_path_hours'], 2)} | {f(r['evaluation_hours'], 2)} |"]
    t = out["tuning"]
    L += ["", f"Hyperparameter tuning (reported separately, not in the totals): {t['trials']} Rewarded Soups trials, "
          f"{t['rs_trials_hours']:.2f} h, run once on the random base; B, C and D reuse the selected setting.", "",
          "![final HV](rs_four_bases_v3/final_hv.png)", "", "![training time](rs_four_bases_v3/training_time.png)", "",
          "![HV vs time](rs_four_bases_v3/hv_vs_time.png)", "",
          "## Timing: summed durations vs elapsed wall clock, and contention", "",
          "- Every time above is a **summed training duration** (train.py's in-loop seconds, validation included), "
          "pretraining counted once per configuration. The critical-path column assumes the three specialists run in parallel.",
          "- Contention differed between runs. A's specialists ran 3 at a time on 4-core slots with its From-scratch trials; "
          "B's specialists ran 3 at a time; C's tardiness specialist (and D's first 2,000 tardiness updates) ran in the "
          "downstream experiment with 4 concurrent runs on 3-core slots; C's and D's makespan / carbon specialists and D's "
          "tardiness extension ran 3 at a time on 4-core slots; C's and D's Rewarded Soups ran 2 at a time. All shared one "
          "GPU. Durations are therefore indicative to perhaps +-20%, not exact.",
          "- Pretraining times are those recorded by the world-model experiments (B: data regeneration + encoder training; "
          "C, D: episodes and branching + physical-target extraction + 60 epochs)."]
    if out.get("wall"):
        w = out["wall"]["invocations"][-1]
        L += [f"- Elapsed wall clock of this invocation (C and D together): specialists {w['specialists_s'] / 3600:.2f} h, "
              f"Rewarded Soups {w['rewarded_soups_s'] / 3600:.2f} h, evaluation {w['evaluation_s'] / 3600:.2f} h, "
              f"total {w['total_s'] / 3600:.2f} h."]
    L += ["", "## Interpretation", "", "INTERPRETATION_PLACEHOLDER", "",
          "## Limitations", "",
          "- One seed (111) per configuration: the seed-to-seed spread of Rewarded Soups HV is not measured, so differences "
          "of the order of 0.01 HV are not evidence of superiority on their own.",
          "- B's encoder differs from C and D in data, targets and pretraining budget; C and D are matched to each other.",
          "- The Rewarded Soups setting was tuned on the random base only; the pretrained bases might prefer other settings.",
          "- Wall-clock times come from runs under different contention (above).",
          "- 10x5 only; no generalisation or larger instances.", ""]
    MD.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    a.smoke = False
    RES.mkdir(parents=True, exist_ok=True)
    if a.audit:
        print(json.dumps(audit(), indent=1)); return
    if a.summarise:
        summarise(); return
    main_run(a)


if __name__ == "__main__":
    main()
