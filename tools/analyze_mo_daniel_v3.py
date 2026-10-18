"""Analysis of the MO-DANIEL v3 runs (docs/evaluation_protocol_v3.md, Addendum A, A6) over whatever has completed.

Reads results/mo_daniel_v3/{runs,eval}, the specialists' greedy evaluation schedules and the CP-SAT results in
results/baselines_v3, and the frozen normalisation points; writes results/mo_daniel_v3.md and figures in
results/mo_daniel_v3/figures/.

    python tools/analyze_mo_daniel_v3.py [--eval_split evaluation] [--norm results/baselines_v3/normalisation_points.json]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "mo_daniel_v3"
BRES = REPO / "results" / "baselines_v3"
FIG = RES / "figures"
sys.argv, _ARGV = sys.argv[:1], sys.argv
sys.path.insert(0, str(REPO / "daniel")); sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "objective_validation"))
import benchmark_v3 as B                                    # noqa: E402
import mo_metrics as MM                                     # noqa: E402
import common as OV                                         # noqa: E402
import schedules as S                                       # noqa: E402
from analyze_baselines_v3 import bootstrap_ci, cpsat_schedules, load_cpsat   # noqa: E402
from mo_daniel_v3_status import degraded                    # noqa: E402
sys.argv = _ARGV

NAME_RE = re.compile(r"^(\d+x\d+)\+mix\+bv3_mo_s(\d+)$")
SEEDS = (111, 222, 333)
OBJ = MM.OBJECTIVES
RULES = (("MWKR+ECT", "MWKR", "ECT"), ("EDD+ECT", "EDD", "ECT"), ("min energy (MWKR+MINE)", "MWKR", "MINE"))
PREF_LABELS = ("w=(1,0,0)", "w=(0,1,0)", "w=(0,0,1)", "w=(1/3,1/3,1/3)")
PPO_KEYS = ("entropy", "approx_kl", "clip_fraction", "policy_loss", "value_loss", "explained_variance", "grad_norm")


def ms(v, fmt="{:.3f}"):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    if len(v) == 1:
        return fmt.format(v[0]) + " (1 seed)"
    return f"{fmt.format(np.mean(v))} ± {fmt.format(np.std(v, ddof=1))}"


def ci(x, fmt="{:.3f}"):
    if len(x) < 2:
        return "-"
    lo, hi = bootstrap_ci(x)
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def load_runs(split):
    runs = []
    for d in sorted((RES / "runs").glob("*")) if (RES / "runs").exists() else []:
        m = NAME_RE.match(d.name)
        if not m:
            continue
        r = {"name": d.name, "size": m.group(1), "seed": int(m.group(2)), "failed": (d / "FAILED").exists()}
        for k in ("status", "valiquality", "reward", "run_meta"):
            f = d / f"{k}.json"
            r[k] = json.loads(f.read_text()) if f.exists() else None
        e = RES / "eval" / f"{split}_{r['size']}"
        r["eval"] = dict(np.load(e / f"{d.name}.npz")) if (e / f"{d.name}.npz").exists() else None
        r["eval_json"] = json.loads((e / f"{d.name}.json").read_text()) if (e / f"{d.name}.json").exists() else None
        runs.append(r)
    return runs


def specialist_points(split, size, seed, keys):
    """[n, 3, 3] greedy objectives of DANIEL-M/C/T of one seed on `keys`, or None if any is missing."""
    out = []
    for o in "mct":
        f = BRES / "eval" / f"{split}_{size}" / f"{size}+mix+bv3_spec_{o}_s{seed}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        pos = {str(k): j for j, k in enumerate(z["keys"])}
        if not all(k in pos for k in keys):
            return None
        out.append(np.stack([[z[f_][pos[k]] for f_ in OBJ] for k in keys]))
    return np.stack(out, 1)


def rule_points(data, keys):
    pts = []
    for key in keys:
        i = data.keys.index(key)
        jl = np.asarray(data.job_length[i])
        inst = OV.Instance(key, jl, np.asarray(data.op_pt[i]), np.repeat(np.arange(len(jl)), jl),
                           np.concatenate([[0], np.cumsum(jl)[:-1]]))
        row = []
        for _, jr, mr in RULES:
            m, s = S.build(inst, jr, mr, energy=data.carbon[i], due=data.due[i])
            r = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i], m, s)
            row.append([r[k] for k in OBJ])
        pts.append(row)
    return np.asarray(pts)                                          # [n, 3 rules, 3]


def per_instance(points, keys, norm, fn):
    return np.array([fn(points[j], *MM.ideal_nadir(norm[k])) for j, k in enumerate(keys)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_split", default="evaluation")
    ap.add_argument("--norm", nargs="+", default=[str(BRES / "normalisation_points.json")])
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    norm = {}
    for f in a.norm:
        norm.update(json.loads(Path(f).read_text())["instances"])
    runs = load_runs(a.eval_split)
    L = ["# MO-DANIEL on benchmark v3: preference-conditioned DANIEL trained from random initialisation", "",
         "Protocol: `docs/evaluation_protocol_v3.md`, Addendum A (pre-registered). Generated by "
         "`tools/analyze_mo_daniel_v3.py` over the runs completed so far. Evaluation split: "
         f"`{a.eval_split}`. Lower objectives are better; higher hypervolume (HV) is better, lower IGD+ is better.", "",
         "## Plain-English summary", "", "_(placeholder: to be written after the runs complete)_", "",
         "## Completion", ""]
    for r in runs:
        st = (r["status"] or {}).get("status", "-")
        deg, drop = degraded(r["valiquality"]) if r["valiquality"] else (False, None)
        meta = r["run_meta"] or {}
        L.append(f"* {r['name']}: {'FAILED' if r['failed'] else st}"
                 + (f", best validation HV {-r['status']['best_value']:.4f} at update {r['status']['best_update']}, "
                    f"stopped at {r['status']['update']}, training {r['status']['train_seconds'] / 3600:.2f} h"
                    if r["status"] else "")
                 + (f", **DEGRADED after best** (last-3 mean {100 * drop:.1f}% below best)" if deg else
                    (f", drop after best {100 * drop:.1f}%" if drop is not None else ""))
                 + (f", evaluated" if r["eval"] is not None else ", not evaluated"))
    if not runs:
        L.append("* no runs yet")
    L.append("")
    for size in B.SIZES:
        sr = [r for r in runs if r["size"] == size and r["eval"] is not None]
        if not sr:
            continue
        keys = [str(k) for k in sr[0]["eval"]["keys"]]
        data = B.load_split(a.eval_split, size)
        b2 = load_cpsat("b2", a.eval_split, size); b1 = load_cpsat("b1", a.eval_split, size)
        sub = [k for k in keys if k in b2]
        grid_n = int(sr[0]["eval_json"]["grid_size"])
        L += [f"## {size}", "", f"### Main table, {size} ({a.eval_split} split, {len(keys)} instances; B2 subset: "
              f"{len(sub)} instances)", "",
              "Per-instance HV of the non-dominated set in normalised objective space (frozen ideal/nadir, reference "
              "1.1), averaged over instances. Seeds: mean ± std (ddof 1) of the per-seed instance means. CI: bootstrap "
              "95% over instances of the seed-averaged per-instance value (10,000 resamples).", "",
              "| Method | Points per instance | HV (all) | HV 95% CI | HV (B2 subset) | IGD+ (B2 subset) | IGD+ 95% CI |",
              "|---|---|---|---|---|---|---|"]

        def row(label, npts, per_seed_hv, per_seed_igd, deterministic=False):
            """per_seed_*: list over seeds of per-instance arrays aligned with keys / sub."""
            f = (lambda v: f"{np.mean(v):.3f}" if v else "-") if deterministic else ms
            hv_all = [h.mean() for h in per_seed_hv]
            hv_sub = [h[[keys.index(k) for k in sub]].mean() for h in per_seed_hv] if sub else []
            igd = [g.mean() for g in per_seed_igd] if sub else []
            L.append(f"| {label} | {npts} | {f(hv_all)} | {ci(np.mean(per_seed_hv, 0))} | {f(hv_sub)} | "
                     f"{f(igd)} | {ci(np.mean(per_seed_igd, 0)) if sub else '-'} |")

        def igd_of(points_by_key):
            if not sub:
                return np.array([])
            out = []
            for k in sub:
                front = np.array([[p[o] for o in OBJ] for p in b2[k]["front"]])
                ideal, nadir = MM.ideal_nadir(norm[k])
                out.append(MM.instance_igd_plus(points_by_key[k], front, ideal, nadir))
            return np.array(out)

        mo_hv, mo_igd = [], []
        for r in sorted(sr, key=lambda r: r["seed"]):
            pts = r["eval"]["objectives"][:, :grid_n]
            mo_hv.append(np.asarray(r["eval"]["hv"]))
            mo_igd.append(igd_of({k: pts[j] for j, k in enumerate(keys)}))
        row(f"**MO-DANIEL** ({len(sr)} seed{'s' if len(sr) > 1 else ''}, greedy, {grid_n} preferences)",
            f"≤ {grid_n}", mo_hv, mo_igd)
        spec = {s: specialist_points(a.eval_split, size, s, keys) for s in SEEDS}
        spec = {s: v for s, v in spec.items() if v is not None}
        if spec:
            row(f"Specialists M+C+T, combined non-dominated set ({len(spec)} seeds)", "3",
                [per_instance(v, keys, norm, MM.instance_hv) for v in spec.values()],
                [igd_of({k: v[j] for j, k in enumerate(keys)}) for v in spec.values()])
            for oi, lab in enumerate(("DANIEL-M", "DANIEL-C", "DANIEL-T")):
                row(f"{lab} (point)", "1", [per_instance(v[:, oi:oi + 1], keys, norm, MM.instance_hv)
                                            for v in spec.values()],
                    [igd_of({k: v[j, oi:oi + 1] for j, k in enumerate(keys)}) for v in spec.values()])
        rp = rule_points(data, keys)
        for ri, (lab, _, _) in enumerate(RULES):
            row(f"{lab} (rule, point)", "1", [per_instance(rp[:, ri:ri + 1], keys, norm, MM.instance_hv)],
                [igd_of({k: rp[j, ri:ri + 1] for j, k in enumerate(keys)})], deterministic=True)
        row("Rules combined (3 points)", "3", [per_instance(rp, keys, norm, MM.instance_hv)],
            [igd_of({k: rp[j] for j, k in enumerate(keys)})], deterministic=True)
        if b1:
            b1_pts = {k: np.array([[b1[k]["points"][p]["final"]["exact"][o] for o in OBJ]
                                   for p in ("mk", "carbon", "tard")]) for k in keys if k in b1}
            if len(b1_pts) == len(keys):
                row("CP-SAT B1 extreme points (3 points)", "3",
                    [per_instance(np.stack([b1_pts[k] for k in keys]), keys, norm, MM.instance_hv)],
                    [igd_of(b1_pts)], deterministic=True)
        if sub:
            hv_b2 = np.array([MM.instance_hv(np.array([[p[o] for o in OBJ] for p in b2[k]["front"]]),
                                             *MM.ideal_nadir(norm[k])) for k in sub])
            L.append(f"| CP-SAT B2 reference front (subset only{'; best-known' if size == '20x10' else ''}) | "
                     f"{np.mean([len(b2[k]['front']) for k in sub]):.1f} (mean) | - | - | {hv_b2.mean():.3f} | "
                     f"0 (by definition) | - |")
        L += ["", f"### Objectives at the corner and equal-weight preferences, {size} (greedy; mean ± std over seeds)", "",
              "| Method | Makespan | Carbon (kg CO2e) | Total tardiness |", "|---|---|---|---|"]
        for lab in PREF_LABELS:
            vals = [r["eval_json"]["objectives_at"][lab] for r in sr]
            L.append(f"| MO-DANIEL {lab} | " + " | ".join(ms([v[o] for v in vals], "{:.1f}") for o in OBJ) + " |")
        for oi, lab in enumerate(("DANIEL-M", "DANIEL-C", "DANIEL-T")):
            if spec:
                L.append(f"| {lab} (specialist) | " + " | ".join(
                    ms([v[:, oi, j].mean() for v in spec.values()], "{:.1f}") for j in range(3)) + " |")
        for ri, (lab, _, _) in enumerate(RULES):
            L.append(f"| {lab} (rule) | " + " | ".join(f"{rp[:, ri, j].mean():.1f}" for j in range(3)) + " |")
        L.append("")
        # convergence curves
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for r in [r for r in runs if r["size"] == size and r["valiquality"]]:
            v = r["valiquality"]
            ax[0].plot([e["update"] for e in v], [e["hv"] for e in v], label=f"s{r['seed']}")
            ax[1].plot([e["train_seconds"] / 3600 for e in v], [e["hv"] for e in v], label=f"s{r['seed']}")
        ax[0].set_xlabel("PPO updates"); ax[1].set_xlabel("training wall-clock (h)")
        for x in ax:
            x.set_ylabel("validation HV (15 preferences)"); x.grid(alpha=.3); x.legend()
        fig.suptitle(f"MO-DANIEL {size}: validation hypervolume"); fig.tight_layout()
        fig.savefig(FIG / f"convergence_{size}.png", dpi=110); plt.close(fig)
        L += [f"![convergence {size}](mo_daniel_v3/figures/convergence_{size}.png)", ""]
        # PPO statistics
        fig, axs = plt.subplots(2, 4, figsize=(15, 6.5))
        for r in [r for r in runs if r["size"] == size and r["reward"]]:
            rw = r["reward"]
            u = np.array([e["update"] for e in rw])
            series = {"training return": [e["mean_return"] for e in rw]}
            for k in PPO_KEYS:
                series[k] = [e.get("ppo", {}).get(k, np.nan) for e in rw]
            for x, (k, y) in zip(axs.flat, series.items()):
                y = np.asarray(y, dtype=float)
                w = max(1, min(50, len(y) // 5))
                sm = np.convolve(np.nan_to_num(y), np.ones(w) / w, mode="valid")
                x.plot(u[w - 1:], sm, label=f"s{r['seed']}", lw=1)
                x.set_title(k + (f" (rolling mean {w})" if w > 1 else ""), fontsize=9); x.grid(alpha=.3)
        axs.flat[0].legend(fontsize=8)
        fig.suptitle(f"MO-DANIEL {size}: PPO statistics per update"); fig.tight_layout()
        fig.savefig(FIG / f"ppo_stats_{size}.png", dpi=110); plt.close(fig)
        L += [f"![PPO statistics {size}](mo_daniel_v3/figures/ppo_stats_{size}.png)", ""]
        # example fronts
        ex = (sub or keys)[:3]
        r0 = sorted(sr, key=lambda r: r["seed"])[0]
        fig = plt.figure(figsize=(15, 4.6))
        for p, k in enumerate(ex):
            x = fig.add_subplot(1, len(ex), p + 1, projection="3d")
            j = keys.index(k)
            mo = r0["eval"]["objectives"][j, :grid_n]
            mo = mo[MM.nondominated(mo)]
            x.scatter(*mo.T, s=18, label=f"MO-DANIEL s{r0['seed']} (non-dominated)")
            if k in b2:
                fr = np.array([[q[o] for o in OBJ] for q in b2[k]["front"]])
                x.scatter(*fr.T, s=18, marker="^", label="CP-SAT B2 front")
            if spec:
                s0 = spec[sorted(spec)[0]][j]
                x.scatter(*s0.T, s=50, marker="*", label=f"specialists M/C/T s{sorted(spec)[0]}")
            x.set_xlabel("makespan"); x.set_ylabel("carbon"); x.set_zlabel("tardiness")
            x.set_title(Path(k).stem, fontsize=9)
        fig.axes[0].legend(fontsize=7, loc="upper left")
        fig.tight_layout(); fig.savefig(FIG / f"fronts_{size}.png", dpi=110); plt.close(fig)
        L += [f"![fronts {size}](mo_daniel_v3/figures/fronts_{size}.png)", ""]
    L += ["## Known limitation", "",
          "DANIEL's action mask only admits non-delay schedules (results/diagnose_v3.md, Test 2): no machine is held "
          "idle for a better one. The carbon optimum and the CP-SAT/EDD tardiness schedules are not reachable, so part "
          "of every gap to CP-SAT is a gap of the action space, kept as published.", ""]
    (REPO / "results" / "mo_daniel_v3.md").write_text("\n".join(L) + "\n")
    print("wrote results/mo_daniel_v3.md")


if __name__ == "__main__":
    main()
