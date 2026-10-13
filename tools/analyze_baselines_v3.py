"""Analysis of the benchmark v3 baselines (docs/evaluation_protocol_v3.md, section 5) over whatever has completed.

Reads results/baselines_v3/{runs,eval,cpsat} and the v3 manifest; writes results/baselines_v3.md and figures in
results/baselines_v3/figures/. When Parts A and B are complete (default plan) it also freezes the later-stage
normalisation points in results/baselines_v3/normalisation_points.json (never overwritten).

    python tools/analyze_baselines_v3.py [--eval_split evaluation] [--cpsat_split evaluation]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "baselines_v3"
FIG = RES / "figures"
sys.argv, _ARGV = sys.argv[:1], sys.argv
sys.path.insert(0, str(REPO / "daniel"))
sys.path.insert(0, str(REPO / "tools" / "objective_validation"))
import benchmark_v3 as B                                    # noqa: E402
import common as OV                                         # noqa: E402
import schedules as S                                       # noqa: E402
sys.argv = _ARGV

OBJ = {"m": ("DANIEL-M", "makespan"), "c": ("DANIEL-C", "carbon"), "t": ("DANIEL-T", "tardiness")}
FIELDS = ("makespan", "carbon", "tardiness", "tardy_share")
DEFAULT_SEEDS = (111, 222, 333)
FULL_B2 = {"10x5": 20, "20x10": 10}
NAME_RE = re.compile(r"^(\d+x\d+)\+mix\+bv3_spec_([mct])_s(\d+)$")


def ms(values, fmt="{:.1f}"):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return "-"
    if len(v) == 1:
        return fmt.format(v[0]) + " (1 seed)"
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}"


def bootstrap_ci(x, n=10_000, seed=20261013):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------------------------------------------------ loading

def load_runs(eval_split):
    runs = []
    for d in sorted((RES / "runs").glob("*")) if (RES / "runs").exists() else []:
        m = NAME_RE.match(d.name)
        if not m:
            continue
        size, obj, seed = m.group(1), m.group(2), int(m.group(3))
        r = {"name": d.name, "size": size, "obj": obj, "seed": seed, "failed": (d / "FAILED").exists()}
        for k in ("status", "valiquality", "run_meta"):
            f = d / f"{k}.json"
            r[k] = json.loads(f.read_text()) if f.exists() else None
        e = RES / "eval" / f"{eval_split}_{size}"
        r["greedy"] = dict(np.load(e / f"{d.name}.npz")) if (e / f"{d.name}.npz").exists() else None
        samp = sorted(e.glob(f"{d.name}_sample*.npz")) if e.exists() else []
        r["sample"] = dict(np.load(samp[0])) if samp else None
        r["sample_name"] = samp[0].stem.split("_")[-1] if samp else None
        runs.append(r)
    return runs


def load_cpsat(part, split, size):
    d = RES / "cpsat" / part / f"{split}_{size}"
    return {json.loads(f.read_text())["key"]: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))} \
        if d.exists() else {}


def cpsat_schedules(b1, b2):
    """All exact objective triples of every CP-SAT schedule of an instance."""
    out = []
    if b1:
        for p in b1["points"].values():
            prim = p.get("primary", {})
            if "exact" in prim:
                out.append(prim["exact"])
            out += [s["exact"] for s in p.get("lex_stages", []) if "exact" in s]
            if p.get("final"):
                out.append(p["final"]["exact"])
    if b2:
        out += [r["exact"] for r in b2.get("points", []) if "exact" in r]
    return out


def best_known(b1, b2):
    sch = cpsat_schedules(b1, b2)
    if not sch:
        return None
    best = {k: min(s[k] for s in sch) for k in ("makespan", "carbon", "tardiness")}
    best["optimal"] = {"makespan": b1["points"]["mk"]["primary"]["status"] == "OPTIMAL",
                       "tardiness": b1["points"]["tard"]["primary"]["status"] == "OPTIMAL",
                       "carbon": True} if b1 else {}
    return best


def rule_baseline(data, keys):
    out = {k: [] for k in FIELDS}
    for key in keys:
        i = data.keys.index(key)
        jl, pt = data.job_length[i], data.op_pt[i]
        inst = OV.Instance(key, jl, pt, np.repeat(np.arange(len(jl)), jl), np.concatenate([[0], np.cumsum(jl)[:-1]]))
        m, s = S.build(inst, *B.REFERENCE_RULE)
        r = B.schedule_objectives(jl, pt, data.carbon[i], data.due[i], m, s)
        assert abs(r["makespan"] - data.mk_ref[i]) < 1e-9 and abs(r["carbon"] - data.cb_ref[i]) < 1e-9
        last = np.cumsum(jl) - 1
        for k, v in (("makespan", r["makespan"]), ("carbon", r["carbon"]), ("tardiness", r["tardiness"]),
                     ("tardy_share", float((r["completion"][last] > data.due[i]).mean()))):
            out[k].append(v)
    return {k: np.asarray(v) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------------------- analysis

def gaps(run_eval, obj, keys, bk, t_ref):
    field = OBJ[obj][1]
    g, opt = [], []
    for k, v in zip(run_eval["keys"], run_eval[field]):
        b = bk.get(str(k))
        if b is None:
            continue
        g.append((v - b[field]) / (t_ref if field == "tardiness" else b[field]))
        opt.append(b["optimal"].get(field, False))
    return (float(np.mean(g)) if g else None), (sum(opt), len(opt))


def time_to(valiquality, best, pct):
    thr = best * (1 + pct) if best >= 0 else best * (1 - pct)
    for e in valiquality:
        if e["selection_metric"] <= thr:
            return e["update"], e["train_seconds"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_split", default="evaluation")
    ap.add_argument("--cpsat_split", default="evaluation")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    man = B.load_manifest()
    runs = load_runs(a.eval_split)
    L = ["# Benchmark v3 baselines: single-objective specialists and CP-SAT reference bounds", "",
         "Protocol: `docs/evaluation_protocol_v3.md` (pre-registered). Generated by `tools/analyze_baselines_v3.py` over "
         f"the units completed so far. Evaluation split: `{a.eval_split}`; CP-SAT split: `{a.cpsat_split}`.", "",
         "## Plain-English summary", "", "_(placeholder: to be written after the runs complete)_", ""]
    notconv = [r["name"] for r in runs if r["status"] and r["status"]["status"] == "NOT_CONVERGED"]
    failed = [r["name"] for r in runs if r["failed"]]
    cp_failed = sorted(p.name for p in (RES / "cpsat").glob("FAILED_*")) if (RES / "cpsat").exists() else []
    L += ["## Completion", "",
          f"* runs with an evaluation: {sum(r['greedy'] is not None for r in runs)}; "
          f"NOT CONVERGED (hit the cap): {', '.join(notconv) or 'none'}; FAILED: {', '.join(failed) or 'none'}",
          f"* CP-SAT batches FAILED: {', '.join(cp_failed) or 'none'}", ""]
    normal_ready = True
    for size in B.SIZES:
        data = B.load_split(a.eval_split, size, man)
        sr = [r for r in runs if r["size"] == size]
        b1 = load_cpsat("b1", a.cpsat_split, size); b2 = load_cpsat("b2", a.cpsat_split, size)
        keys = None
        for r in sr:
            if r["greedy"] is not None:
                keys = [str(k) for k in r["greedy"]["keys"]]; break
        if keys is None:
            keys = list(b1) if b1 else []
        n_full = len(data)
        if not sr and not b1:
            normal_ready = False
            continue
        L += [f"## {size}", ""]
        bk = {k: best_known(b1.get(k), b2.get(k)) for k in keys}
        bk = {k: v for k, v in bk.items() if v is not None}
        rule = rule_baseline(data, keys) if keys else None

        # main table
        L += [f"### Main table, {size} (evaluation split, greedy; mean ± std over seeds; {len(keys)} instances)", "",
              "| Model | Training objective | Makespan | Carbon (kg CO2e) | Tardiness | Tardy-job share | Training time (h) "
              "| Updates to convergence | Converged | Gap to bound on own objective |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for obj, (label, field) in OBJ.items():
            rr = [r for r in sr if r["obj"] == obj and r["greedy"] is not None]
            if not rr:
                L.append(f"| {label} | {field} | - | - | - | - | - | - | - | - |"); continue
            vals = {k: [float(np.mean(r["greedy"][k])) for r in rr] for k in FIELDS}
            tt = [r["status"]["train_seconds"] / 3600 for r in rr]
            up = [r["status"]["update"] for r in rr]
            nconv = sum(r["status"]["status"] == "CONVERGED" for r in rr)
            gp = [gaps(r["greedy"], obj, keys, bk, data.t_ref) for r in rr]
            gvals = [x[0] for x in gp if x[0] is not None]
            opt = gp[0][1] if gp else (0, 0)
            glabel = ("gap to bound" if opt[1] and opt[0] == opt[1] else
                      f"gap to best-known ({opt[0]}/{opt[1]} proven optimal)") if opt[1] else "no CP-SAT yet"
            gtxt = (ms([100 * x for x in gvals], "{:.1f}% of Tref" if field == "tardiness" else "{:.1f}%")
                    + f" [{glabel}]") if gvals else "-"
            L.append(f"| {label} | {field} | {ms(vals['makespan'])} | {ms(vals['carbon'])} | {ms(vals['tardiness'])} | "
                     f"{ms(vals['tardy_share'], '{:.3f}')} | {ms(tt, '{:.2f}')} | {ms(up, '{:.0f}')} | "
                     f"{'Y' if nconv == len(rr) else 'N'} ({nconv}/{len(rr)}) | {gtxt} |")
        for pkey, label in (("mk", "CP-SAT min makespan (B1, lexicographic)"),
                            ("tard", "CP-SAT min tardiness (B1, lexicographic)"),
                            ("carbon", "CP-SAT min carbon (B1, analytic + lexicographic)")):
            fin = [b1[k]["points"][pkey]["final"]["exact"] for k in keys if k in b1 and b1[k]["points"][pkey].get("final")]
            if fin:
                L.append(f"| {label} | {pkey} | {np.mean([f['makespan'] for f in fin]):.1f} | "
                         f"{np.mean([f['carbon'] for f in fin]):.1f} | {np.mean([f['tardiness'] for f in fin]):.1f} | "
                         f"{np.mean([f['tardy_share'] for f in fin]):.3f} | - | - | - | reference ({len(fin)} instances) |")
        if rule is not None:
            L.append(f"| MWKR + ECT (rule, context) | - | {rule['makespan'].mean():.1f} | {rule['carbon'].mean():.1f} | "
                     f"{rule['tardiness'].mean():.1f} | {rule['tardy_share'].mean():.3f} | - | - | - | - |")
        L += ["", "Tardiness gaps are (model - best-known) / Tref (Tref = "
              f"{data.t_ref:.1f}); makespan and carbon gaps are relative to the best-known CP-SAT value.", ""]

        # CI table
        L += [f"### Bootstrap 95% CIs over instances, {size} (seed-averaged per-instance values, 10,000 resamples)", "",
              "| Model | Makespan | Carbon | Tardiness | Tardy-job share |", "|---|---|---|---|---|"]
        for obj, (label, _) in OBJ.items():
            rr = [r for r in sr if r["obj"] == obj and r["greedy"] is not None]
            if not rr:
                continue
            cells = []
            for k in FIELDS:
                per = np.mean([r["greedy"][k] for r in rr], axis=0)
                lo, hi = bootstrap_ci(per); f = "{:.3f}" if k == "tardy_share" else "{:.1f}"
                cells.append(f"{f.format(per.mean())} [{f.format(lo)}, {f.format(hi)}]")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")

        # sampling table
        rows = []
        for obj, (label, _) in OBJ.items():
            rr = [r for r in sr if r["obj"] == obj and r["sample"] is not None]
            if rr:
                vals = {k: [float(np.mean(r["sample"][k])) for r in rr] for k in FIELDS}
                rows.append(f"| {label} | {ms(vals['makespan'])} | {ms(vals['carbon'])} | {ms(vals['tardiness'])} | "
                            f"{ms(vals['tardy_share'], '{:.3f}')} |")
        if rows:
            L += [f"### Secondary: sampling ({sr[0]['sample_name'] or 'best of N'}, selected by own objective), {size}", "",
                  "| Model | Makespan | Carbon | Tardiness | Tardy-job share |", "|---|---|---|---|---|"] + rows + [""]

        # time to 5/2/1 %
        L += [f"### Time to reach 5% / 2% / 1% of the final validation value, {size} (mean ± std over seeds)", "",
              "| Model | 5%: updates | 5%: hours | 2%: updates | 2%: hours | 1%: updates | 1%: hours | best checkpoint update |",
              "|---|---|---|---|---|---|---|---|"]
        for obj, (label, _) in OBJ.items():
            rr = [r for r in sr if r["obj"] == obj and r["valiquality"] and r["status"]
                  and r["status"]["status"] in ("CONVERGED", "NOT_CONVERGED")]
            if not rr:
                continue
            cells = []
            for pct in (0.05, 0.02, 0.01):
                t = [time_to(r["valiquality"], r["status"]["best_value"], pct) for r in rr]
                cells += [ms([x[0] for x in t], "{:.0f}"), ms([x[1] / 3600 if x[1] is not None else None for x in t], "{:.2f}")]
            L.append(f"| {label} | " + " | ".join(cells) + f" | {ms([r['status']['best_update'] for r in rr], '{:.0f}')} |")
        L.append("")

        # convergence curves
        done = [r for r in sr if r["valiquality"]]
        if done:
            for xkey, xlabel, tag in (("update", "PPO updates", "updates"), ("train_seconds", "training time (h)", "time")):
                fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
                for ax, (obj, (label, field)) in zip(axes, OBJ.items()):
                    for r in [r for r in done if r["obj"] == obj]:
                        x = np.array([e[xkey] for e in r["valiquality"]], dtype=float)
                        x = x / 3600 if xkey == "train_seconds" else x
                        ax.plot(x, [e["selection_metric"] for e in r["valiquality"]], label=f"s{r['seed']}", lw=1)
                    ax.set_title(f"{label} ({size})"); ax.set_xlabel(xlabel); ax.set_ylabel(f"validation {field} / ref")
                    ax.legend(fontsize=7)
                fig.tight_layout(); fig.savefig(FIG / f"convergence_{size}_{tag}.png", dpi=120); plt.close(fig)
                L += [f"![convergence {size} vs {tag}](baselines_v3/figures/convergence_{size}_{tag}.png)", ""]

        # B1
        if b1:
            L += [f"### CP-SAT B1 bounds, {size} ({len(b1)}/{n_full} instances)", "",
                  "| Point | Primary status counts | Mean primary wall (s) | Makespan | Carbon | Tardiness | Tardy-job share |",
                  "|---|---|---|---|---|---|---|"]
            for pkey, label in (("mk", "min makespan"), ("tard", "min tardiness"), ("carbon", "min carbon (analytic)")):
                prim = [v["points"][pkey]["primary"] for v in b1.values()]
                counts = {}
                for p in prim:
                    counts[p["status"]] = counts.get(p["status"], 0) + 1
                fin = [v["points"][pkey]["final"]["exact"] for v in b1.values() if v["points"][pkey].get("final")]
                L.append(f"| {label} | {', '.join(f'{k} {c}' for k, c in sorted(counts.items()))} | "
                         f"{np.mean([p.get('wall_s', 0.0) for p in prim]):.1f} | "
                         + " | ".join(f"{np.mean([f[k] for f in fin]):.{3 if k == 'tardy_share' else 1}f}" for k in FIELDS)
                         + " |")
            L += ["", "Objectives at each extreme point are after the lexicographic stages (exact re-scoring).", ""]
        if len(b1) < n_full:
            normal_ready = False

        # B2
        if b2:
            fronts = [len(v.get("front", [])) for v in b2.values()]
            st = {}
            for v in b2.values():
                for p in v.get("points", []):
                    st[p["status"]] = st.get(p["status"], 0) + 1
            tot = sum(st.values())
            label = "best-known" if size == "20x10" else "reference"
            L += [f"### CP-SAT B2 {label} front, {size} ({len(b2)} instances)", "",
                  f"* points on the front per instance: mean {np.mean(fronts):.1f}, min {min(fronts)}, max {max(fronts)}",
                  f"* grid statuses: {', '.join(f'{k} {c}' for k, c in sorted(st.items()))}; share proven optimal "
                  f"{st.get('OPTIMAL', 0) / max(tot, 1):.1%}", ""]
            fig = plt.figure(figsize=(15, 4.5))
            for j, key in enumerate(list(b2)[:3]):
                ax = fig.add_subplot(1, 3, j + 1, projection="3d")
                fr = b2[key].get("front", [])
                ax.scatter([p["makespan"] for p in fr], [p["carbon"] for p in fr], [p["tardiness"] for p in fr],
                           c="k", s=18, label="CP-SAT front")
                for obj, (lab, _) in OBJ.items():
                    pts = [(r["greedy"]["makespan"][i], r["greedy"]["carbon"][i], r["greedy"]["tardiness"][i])
                           for r in sr if r["obj"] == obj and r["greedy"] is not None
                           for i in np.flatnonzero(r["greedy"]["keys"] == key)]
                    if pts:
                        p = np.array(pts); ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=26, marker="^", label=lab)
                if rule is not None and key in keys:
                    i = keys.index(key)
                    ax.scatter([rule["makespan"][i]], [rule["carbon"][i]], [rule["tardiness"][i]], s=30, marker="s",
                               label="MWKR+ECT")
                ax.set_xlabel("makespan"); ax.set_ylabel("carbon"); ax.set_zlabel("tardiness")
                ax.set_title(Path(key).stem, fontsize=9); ax.legend(fontsize=6)
            fig.tight_layout(); fig.savefig(FIG / f"front_examples_{size}.png", dpi=120); plt.close(fig)
            L += [f"![fronts {size}](baselines_v3/figures/front_examples_{size}.png)", ""]
        if len(b2) < FULL_B2[size]:
            normal_ready = False
        for obj in OBJ:
            if sum(1 for r in sr if r["obj"] == obj and r["greedy"] is not None and r["seed"] in DEFAULT_SEEDS) < 3:
                normal_ready = False

    out_norm = RES / "normalisation_points.json"
    if normal_ready and a.eval_split == "evaluation" and not out_norm.exists():
        L += ["Normalisation points frozen: `results/baselines_v3/normalisation_points.json`.", ""]
        freeze_normalisation(runs, man, a.cpsat_split, out_norm)
    else:
        L += ["Normalisation points: " + ("frozen earlier (not overwritten)." if out_norm.exists() else
                                          "pending (Parts A and B incomplete)."), ""]
    (REPO / "results" / "baselines_v3.md").write_text("\n".join(L) + "\n")
    print("wrote results/baselines_v3.md")


def freeze_normalisation(runs, man, cpsat_split, path):
    """Protocol section 6: per instance ideal (CP-SAT best-known) and nadir (worst over the reference/baseline set)."""
    out = {"definition": "docs/evaluation_protocol_v3.md section 6", "instances": {}}
    for size in B.SIZES:
        data = B.load_split("evaluation", size, man)
        b1 = load_cpsat("b1", cpsat_split, size); b2 = load_cpsat("b2", cpsat_split, size)
        rule = rule_baseline(data, data.keys)
        for i, key in enumerate(data.keys):
            sch = cpsat_schedules(b1.get(key), b2.get(key))
            front = b2.get(key, {}).get("front", [])
            pts = [s for s in sch if s] + front
            pts += [{"makespan": r["greedy"]["makespan"][j], "carbon": r["greedy"]["carbon"][j],
                     "tardiness": r["greedy"]["tardiness"][j]}
                    for r in runs if r["size"] == size and r["greedy"] is not None
                    for j in np.flatnonzero(r["greedy"]["keys"] == key)]
            pts.append({k: float(rule[k][i]) for k in ("makespan", "carbon", "tardiness")})
            ideal = {k: float(min(s[k] for s in sch)) for k in ("makespan", "carbon", "tardiness")}
            nadir = {k: float(max(p[k] for p in pts)) for k in ("makespan", "carbon", "tardiness")}
            out["instances"][key] = {"ideal": ideal, "nadir": nadir}
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
