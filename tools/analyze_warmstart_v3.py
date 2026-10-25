"""Analysis of the warm-start study (docs/warmstart_protocol_v3.md) over whatever has completed; writes
results/warmstart_v3.md and figures in results/warmstart_v3/figures/ (smoke runs: --results
results/warmstart_v3_smoke --out results/warmstart_v3_smoke.md).

    python tools/analyze_warmstart_v3.py [--eval_split evaluation] [--norm <points.json> ...]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "warmstart_v3"
BRES = REPO / "results" / "baselines_v3"
FIG = RES / "figures"
TAG = "ws"
sys.argv, _ARGV = sys.argv[:1], sys.argv
sys.path.insert(0, str(REPO / "daniel")); sys.path.insert(0, str(REPO / "tools"))
import benchmark_v3 as B                                    # noqa: E402
import mo_metrics as MM                                     # noqa: E402
from analyze_baselines_v3 import load_cpsat                 # noqa: E402
from analyze_mo_daniel_v3 import specialist_points, rule_points, RULES   # noqa: E402
from mo_daniel_v3_status import degraded                    # noqa: E402
sys.argv = _ARGV

ARMS = ["B", "CM", "BBM", "D", "E", "CC", "CT", "BBC", "BBT", "ES"]
LABEL = {"B": "B (scratch, WI-DAN)", "CM": "C-M", "CC": "C-C", "CT": "C-T", "BBM": "BB-M", "BBC": "BB-C", "BBT": "BB-T",
         "D": "D (vertex distillation)",
         "E": "E (naive merge of independently trained specialists (no shared base); expected to be weak)",
         "ES": "E-shared", "E0": "E0 (zero-training merge front)"}
SHORT = {**{k: k for k in ARMS}, "CM": "C-M", "CC": "C-C", "CT": "C-T", "BBM": "BB-M", "BBC": "BB-C", "BBT": "BB-T",
         "E0": "E0"}
USES = {"B": "", "CM": "m", "CC": "c", "CT": "t", "BBM": "m", "BBC": "c", "BBT": "t", "D": "mct", "E": "mct", "ES": ""}
NAME_RE = re.compile(r"^(\d+x\d+)\+mix\+bv3_wss?_([A-Z0-9]+)_s(\d+)$")
OBJ = MM.OBJECTIVES
BUDGETS = (500, 1000, 2000)
CORNER = 0.6                                                    # corner region of x: lambda_x >= 0.6 (15 of 66)
RNG_SEED = 20261013


def ms(v, fmt="{:.3f}"):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    return fmt.format(v[0]) + " (1 seed)" if len(v) == 1 else f"{fmt.format(np.mean(v))} ± {fmt.format(np.std(v, ddof=1))}"


def paired(a, b, n=10_000):
    """Mean of a - b over instances and its bootstrap 95% CI (percentile); a, b aligned per instance."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return None
    rng = np.random.default_rng(RNG_SEED)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt_ci(t, f="{:+.3f}"):
    return "-" if t is None else f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]"


# ------------------------------------------------------------------ loading
def load(split):
    runs = {}
    for d in sorted((RES / "runs").glob("*")) if (RES / "runs").exists() else []:
        m = NAME_RE.match(d.name)
        if not m:
            continue
        r = {"name": d.name, "size": m.group(1), "arm": m.group(2), "seed": int(m.group(3)), "failed": (d / "FAILED").exists()}
        for k in ("status", "valiquality", "reward", "run_meta"):
            f = d / f"{k}.json"
            r[k] = json.loads(f.read_text()) if f.exists() else None
        runs[d.name] = r
    ev = {}
    for f in sorted((RES / "eval").glob(f"{split}_*/*.json")) if (RES / "eval").exists() else []:
        m = NAME_RE.match(f.stem)
        if not m:
            continue
        z = f.with_suffix(".npz")
        ev[f.stem] = {"size": m.group(1), "arm": m.group(2), "seed": int(m.group(3)), "json": json.loads(f.read_text()),
                      "npz": dict(np.load(z)) if z.exists() else None}
    return runs, ev


def val_series(r):
    v = r["valiquality"] or []
    return ([e["update"] for e in v], [e["hv"] for e in v], [e["train_seconds"] for e in v],
            {e["update"]: e.get("hv_per_instance") for e in v})


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_split", default="evaluation")
    ap.add_argument("--norm", nargs="+", default=[str(BRES / "normalisation_points.json")])
    ap.add_argument("--results", default=str(RES), help="results directory (results/warmstart_v3_smoke for smoke runs)")
    ap.add_argument("--out", default=str(REPO / "results" / "warmstart_v3.md"))
    a = ap.parse_args()
    global RES, FIG, TAG
    RES = Path(a.results); FIG = RES / "figures"
    TAG = "wss" if RES.name.endswith("_smoke") else "ws"
    FIG.mkdir(parents=True, exist_ok=True)
    norm = {}
    for f in a.norm:
        norm.update(json.loads(Path(f).read_text())["instances"])
    runs, ev = load(a.eval_split)
    proto = (REPO / "docs" / "warmstart_protocol_v3.md").read_text()
    L = ["# Warm-start study on benchmark v3: results", "",
         "Generated by `tools/analyze_warmstart_v3.py` over the units completed so far. Evaluation split: "
         f"`{a.eval_split}`. Higher hypervolume (HV) is better; lower IGD+, regret and objectives are better.", "",
         "## Plain-English summary", "", "_(placeholder: to be written after the runs complete)_", ""]
    verdict_lines = []
    for size in B.SIZES:
        sruns = [r for r in runs.values() if r["size"] == size]
        sev = {k: v for k, v in ev.items() if v["size"] == size}
        if not sruns and not sev:
            continue
        L += [f"## {size}", ""]
        calib = RES / f"calibration_{size}.json"
        factor = json.loads(calib.read_text())["factor"] if calib.exists() else None
        plan = json.loads((RES / "plan.json").read_text()) if (RES / "plan.json").exists() else {}
        L.append(f"Hardware and parallelism: {plan.get('host', {}).get('gpu', '?')}, {plan.get('host', {}).get('logical_cpus', '?')} "
                 f"logical CPUs, --jobs {plan.get('jobs', '?')} for every arm. Calibration factor (this machine / local "
                 f"specialist training, DANIEL-M): {factor:.2f}" if factor else "Calibration: not available.")
        L.append("")
        any_ev = next(iter(sev.values()), None)
        keys = [str(k) for k in any_ev["npz"]["keys"]] if any_ev and any_ev["npz"] is not None else []
        if keys and any(k not in norm for k in keys):
            L += [f"_Normalisation points missing for {size}; evaluation tables skipped._", ""]
            keys = []
        grid = MM.simplex_grid(_grid_step(int(any_ev["json"]["grid_size"])) if any_ev else 0.1)
        G = len(grid)
        b2 = load_cpsat("b2", a.eval_split, size); b1 = load_cpsat("b1", a.eval_split, size)
        sub = [k for k in keys if k in b2]
        arms_here = [x for x in ARMS + ["E0"] if any(v["arm"] == x for v in sev.values()) or
                     any(r["arm"] == x for r in sruns)]

        # per-arm, per-seed, per-instance evaluation quantities
        def ev_of(arm):
            return sorted([v for v in sev.values() if v["arm"] == arm and v["npz"] is not None], key=lambda v: v["seed"])

        def inst_mean(arrs):
            return np.nanmean(np.stack(arrs), 0) if arrs else None

        hv_inst = {x: inst_mean([np.asarray(v["npz"]["hv"]) for v in ev_of(x)]) for x in arms_here}
        igd_inst = {}
        for x in arms_here:
            per = [[v["json"]["igd_plus_per_instance"].get(k, np.nan) for k in sub] for v in ev_of(x)]
            igd_inst[x] = inst_mean([np.asarray(p, float) for p in per]) if per and sub else None

        # ---------------- main quality table
        if keys:
            L += [f"### Quality, {size} (greedy, {G} preferences; {len(keys)} instances; IGD+ on the {len(sub)}-instance "
                  "B2 subset)", "",
                  "ΔHV and ΔIGD+: arm − B, mean over instances of the seed-averaged per-instance difference, with a "
                  "paired bootstrap 95% CI over instances (10,000 resamples).", "",
                  "| Arm | Seeds | HV | ΔHV vs B [95% CI] | IGD+ | ΔIGD+ vs B [95% CI] | Unique non-dominated | "
                  "Converged | Degraded |", "|---|---|---|---|---|---|---|---|---|"]
            for x in arms_here:
                e = ev_of(x)
                if not e:
                    continue
                rs = [r for r in sruns if r["arm"] == x]
                conv = sum((r["status"] or {}).get("status") == "CONVERGED" for r in rs)
                degr = sum(degraded(r["valiquality"])[0] for r in rs if r["valiquality"])
                dhv = paired(hv_inst[x], hv_inst["B"]) if x != "B" and hv_inst.get("B") is not None else None
                digd = (paired(igd_inst[x], igd_inst["B"]) if x != "B" and igd_inst.get(x) is not None
                        and igd_inst.get("B") is not None else None)
                L.append(f"| {LABEL[x]} | {len(e)} | {ms([np.mean(v['npz']['hv']) for v in e])} | {fmt_ci(dhv)} | "
                         f"{ms([v['json']['igd_plus_mean'] for v in e]) if sub else '-'} | {fmt_ci(digd)} | "
                         f"{ms([v['json']['nd_size_mean'] for v in e], '{:.1f}')} | "
                         f"{'-' if x == 'E0' else f'{conv}/{len(rs)}'} | {'-' if x == 'E0' else f'{degr}/{len(rs)}'} |")
            refs = reference_points(a.eval_split, size, keys, b1, b2)
            for lab, pts in refs.items():
                hv = [MM.instance_hv(pts[k], *MM.ideal_nadir(norm[k])) for k in keys if k in pts]
                if hv:
                    L.append(f"| {lab} (reference) | - | {np.mean(hv):.3f} | - | - | - | - | - | - |")
            if sub:
                hvb2 = [MM.instance_hv(np.array([[p[o] for o in OBJ] for p in b2[k]["front"]]), *MM.ideal_nadir(norm[k]))
                        for k in sub]
                L.append(f"| CP-SAT B2 reference front (subset only; HV on the subset) | - | {np.mean(hvb2):.3f} | - | "
                         f"0 (by definition) | - | - | - | - |")
            L.append("")
            # secondary: sampling
            L += [f"### Secondary: best-of-K sampling per preference, {size}", "",
                  "| Arm | K | Instances | HV | IGD+ |", "|---|---|---|---|---|"]
            for x in arms_here:
                sm = [v["json"].get("sample") for v in ev_of(x) if v["json"].get("sample")]
                if sm:
                    L.append(f"| {LABEL[x]} | {sm[0]['k']} | {sm[0].get('n', '-')} | {ms([s['hv_mean'] for s in sm])} | "
                             f"{ms([s['igd_plus_mean'] for s in sm]) if sub else '-'} |")
            L.append("")
            # objectives at corners and equal weights
            L += [f"### Objectives at the corner and equal-weight preferences, {size} (greedy; mean ± std over seeds)", "",
                  "| Arm | MK at w=(1,0,0) | C at w=(0,1,0) | T at w=(0,0,1) | MK / C / T at w=(1/3,1/3,1/3) |",
                  "|---|---|---|---|---|"]
            for x in arms_here:
                e = ev_of(x)
                if not e:
                    continue
                at = [v["json"]["objectives_at"] for v in e]
                eq = " / ".join(ms([q["w=(1/3,1/3,1/3)"][o] for q in at], "{:.1f}") for o in OBJ)
                L.append(f"| {LABEL[x]} | {ms([q['w=(1,0,0)']['makespan'] for q in at], '{:.1f}')} | "
                         f"{ms([q['w=(0,1,0)']['carbon'] for q in at], '{:.1f}')} | "
                         f"{ms([q['w=(0,0,1)']['tardiness'] for q in at], '{:.1f}')} | {eq} |")
            L.append("")

        # ---------------- efficiency
        b_runs = [r for r in sruns if r["arm"] == "B" and r["valiquality"]]
        b_target = np.mean([max(val_series(r)[1]) for r in b_runs]) if b_runs else None
        L += [f"### Efficiency, {size} (validation hypervolume, 15 preferences)", "",
              (f"B's final validation HV (mean over its seeds of the best value): {b_target:.4f}; targets 90% = "
               f"{0.9 * b_target:.4f}, 95% = {0.95 * b_target:.4f}." if b_target else "B not finished: no targets yet."), "",
              "| Arm | AUC [0, 2000] | HV@500 | HV@1000 | HV@2000 | Updates to 90% of B | h to 90% | Updates to 95% of B | "
              "h to 95% | ΔHV@500 vs B (validation instances) |", "|---|---|---|---|---|---|---|---|---|---|"]
        effic = {}
        for x in [y for y in ARMS if any(r["arm"] == y for r in sruns)]:
            rs = sorted([r for r in sruns if r["arm"] == x and r["valiquality"]], key=lambda r: r["seed"])
            if not rs:
                continue
            row = {"auc": [], "b": {u: [] for u in BUDGETS}, "t90": [], "h90": [], "t95": [], "h95": [], "v500": []}
            for r in rs:
                u, hv, sec, per = val_series(r)
                row["auc"].append(MM.auc(u, hv, 2000))
                for bud in BUDGETS:
                    row["b"][bud].append(MM.curve_at(u, hv, bud) if u and u[-1] >= bud else None)
                for pct, ku, kh in ((0.9, "t90", "h90"), (0.95, "t95", "h95")):
                    t = MM.time_to(u, hv, sec, pct * b_target) if b_target else (None, None)
                    row[ku].append(t[0]); row[kh].append(None if t[1] is None else t[1] / 3600)
                row["v500"].append(per.get(500))
            effic[x] = row
            reach = lambda v: f"{ms([q for q in v if q is not None], '{:.0f}')} ({sum(q is not None for q in v)}/{len(v)} reached)"
            reach_h = lambda v: ms([q for q in v if q is not None], "{:.2f}")
            d500 = None
            if x != "B" and "B" in effic and all(p is not None for p in row["v500"] + effic["B"]["v500"]):
                d500 = paired(np.mean(row["v500"], 0), np.mean(effic["B"]["v500"], 0))
            L.append(f"| {LABEL[x]} | {ms(row['auc'])} | " + " | ".join(ms(row['b'][bud]) for bud in BUDGETS) +
                     f" | {reach(row['t90'])} | {reach_h(row['h90'])} | {reach(row['t95'])} | {reach_h(row['h95'])} | "
                     f"{fmt_ci(d500)} |")
        L.append("")

        # ---------------- cost
        L += [f"### Cost, {size}", "",
              "Marginal: the arm's own PPO training (train_seconds), plus D's imitation (state collection + training) and "
              "E-shared's base and fine-tunes. Full lifecycle: marginal plus the training of the specialists the arm uses "
              "(local train_seconds × the calibration factor).", "",
              "| Arm | Marginal (h) | of which imitation / shared base (h) | Full lifecycle (h) | Specialists used |",
              "|---|---|---|---|---|"]
        for x in [y for y in ARMS if any(r["arm"] == y for r in sruns)]:
            marg, extra, life = [], [], []
            for r in sorted([r for r in sruns if r["arm"] == x and r["status"]], key=lambda r: r["seed"]):
                m = r["status"]["train_seconds"] / 3600
                e = 0.0
                if x == "D":
                    f = RES / "distill" / f"{size}+mix+bv3_{TAG}D_student_s{r['seed']}.distill.json"
                    e = json.loads(f.read_text())["total_seconds"] / 3600 if f.exists() else float("nan")
                if x == "ES":
                    d = RES / "shared_base" / f"{size}_s{r['seed']}"
                    e = sum(json.loads(f.read_text())["train_seconds"] for f in d.glob("*.status.json")) / 3600
                spec_h = 0.0
                for o in USES[x]:
                    f = BRES / "runs" / f"{size}+mix+bv3_spec_{o}_s{r['seed']}" / "status.json"
                    spec_h += json.loads(f.read_text())["train_seconds"] / 3600 if f.exists() else float("nan")
                marg.append(m + e); extra.append(e); life.append(m + e + spec_h * (factor or float("nan")))
            L.append(f"| {LABEL[x]} | {ms(marg, '{:.2f}')} | {ms(extra, '{:.2f}')} | {ms(life, '{:.2f}')} | "
                     f"{', '.join('DANIEL-' + o.upper() for o in USES[x]) or '-'} |")
        L.append("")

        # ---------------- regret
        reg = regret_table(sev, keys, norm, grid, a.eval_split, size, b1, b2) if keys else None
        if reg:
            R, bias_inst = reg
            L += [f"### Preference regret, {size}", "",
                  "R(λ) = λ·f̂(λ) − min over every arm's and reference's points of λ·f, in the frozen normalised space; "
                  f"mean over instances and seeds. Corner region of x: λ_x ≥ {CORNER}.", "",
                  "| Arm | Mean R | R near m-corner | R near c-corner | R near t-corner |", "|---|---|---|---|---|"]
            for x in [y for y in ARMS + ["E0"] if y in R]:
                r = R[x]
                L.append(f"| {LABEL[x]} | {r['all']:.4f} | {r['m']:.4f} | {r['c']:.4f} | {r['t']:.4f} |")
            L.append("")
            plot_regret(R, grid, size)
            L += [f"![regret {size}]({RES.name}/figures/regret_{size}.png)", ""]
        else:
            bias_inst = {}

        # ---------------- figures
        plot_curves(sruns, size)
        plot_ppo(sruns, size)
        L += [f"![learning curves {size}]({RES.name}/figures/curves_{size}.png)", "",
              f"![PPO statistics {size}]({RES.name}/figures/ppo_{size}.png)", ""]
        if keys:
            plot_e0(sev, keys, size, b2, a.eval_split)
            L += [f"![E0 front {size}]({RES.name}/figures/e0_{size}.png)", ""]

        # ---------------- verdicts
        verdict_lines += [f"### {size}", ""] + verdicts(hv_inst, effic, bias_inst, sev, keys, norm, a.eval_split, size)

    L += ["## Hypothesis verdicts (pre-registered rules, docs/warmstart_protocol_v3.md)", ""] + \
         (verdict_lines or ["_No completed runs yet._"]) + [""]
    L += ["## Known limitation", "", "DANIEL's action mask only admits non-delay schedules (results/diagnose_v3.md, "
          "Test 2), kept as published; part of every gap to CP-SAT is a gap of the action space.", "",
          "## Pre-registration (unchanged copy of docs/warmstart_protocol_v3.md)", "", proto]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


def _grid_step(grid_size):
    """Grid step from the number of simplex grid points: (n+1)(n+2)/2 = grid_size."""
    n = int(round((-3 + np.sqrt(1 + 8 * grid_size)) / 2))
    return 1.0 / n


def reference_points(split, size, keys, b1, b2):
    """Raw objective points per instance of the references: specialists (all seeds, combined), rules, CP-SAT B1."""
    out = {}
    sp = [specialist_points(split, size, s, keys) for s in (111, 222, 333)]
    sp = [x for x in sp if x is not None]
    if sp:
        out["Specialists M+C+T, all seeds combined"] = {k: np.concatenate([x[j] for x in sp]) for j, k in enumerate(keys)}
    data = B.load_split(split, size)
    rp = rule_points(data, keys)
    out["Rules MWKR+ECT, EDD+ECT, min energy combined"] = {k: rp[j] for j, k in enumerate(keys)}
    if b1 and all(k in b1 for k in keys):
        out["CP-SAT B1 extreme points"] = {k: np.array([[b1[k]["points"][p]["final"]["exact"][o] for o in OBJ]
                                                        for p in ("mk", "carbon", "tard")]) for k in keys}
    return out


def regret_table(sev, keys, norm, grid, split, size, b1, b2):
    G = len(grid)
    arm_pts = {}
    for v in sev.values():
        arm_pts.setdefault(v["arm"], []).append(np.asarray(v["npz"]["objectives"])[:, :G])   # [I, G, 3] per seed
    refs = reference_points(split, size, keys, b1, b2)
    R, bias_inst = {}, {}
    per_arm = {x: np.zeros((len(keys), G)) for x in arm_pts}
    for i, k in enumerate(keys):
        ideal, nadir = MM.ideal_nadir(norm[k])
        cand = [p[i] for ps in arm_pts.values() for p in ps] + [r[k] for r in refs.values() if k in r]
        if k in b2:
            cand.append(np.array([[q[o] for o in OBJ] for q in b2[k]["front"]]))
        Z, keep = MM.normalise(np.concatenate(cand), ideal, nadir)
        lam = grid[:, keep]
        best = (Z @ lam.T).min(0)                                         # [G]
        for x, ps in arm_pts.items():
            vals = []
            for p in ps:
                f, _ = MM.normalise(p[i], ideal, nadir)
                vals.append((f * lam).sum(1) - best)
            per_arm[x][i] = np.mean(vals, 0)
    for x, r in per_arm.items():
        R[x] = {"all": float(r.mean()), "grid": r.mean(0)}
        for c, o in enumerate("mct"):
            R[x][o] = float(r[:, grid[:, c] >= CORNER - 1e-9].mean())
        bias_inst[x] = {o: r[:, grid[:, c] >= CORNER - 1e-9].mean(1) - r[:, grid[:, c] < CORNER - 1e-9].mean(1)
                        for c, o in enumerate("mct")}
    return R, bias_inst


def verdicts(hv_inst, effic, bias_inst, sev, keys, norm, split, size):
    out = []
    if hv_inst.get("B") is None or "B" not in effic:
        return ["UNCLEAR for every hypothesis: arm B has no complete results yet.", ""]
    b500 = effic["B"]["v500"]

    def early(x):
        if x not in effic or any(p is None for p in effic[x]["v500"] + b500):
            return None
        return paired(np.mean(effic[x]["v500"], 0), np.mean(b500, 0))

    def faster(x):
        if x not in effic:
            return None
        t, tb = effic[x]["t95"], effic["B"]["t95"]
        return all(a is not None and (b is None or a < b) for a, b in zip(t, tb)) if len(t) == len(tb) else None

    def per_x(tag, rule):
        res = []
        for o in "MCT":
            x = f"{tag}{o}"
            res.append((x, *rule(x, o.lower())))
        return res

    def c_rule(x, o):
        e = early(x); bias = paired(bias_inst[x][o], bias_inst["B"][o]) if x in bias_inst and "B" in bias_inst else None
        if e is None or bias is None:
            return "UNCLEAR", "insufficient data"
        a1, a2 = e[1] > 0, bias[2] < 0
        v = "SUPPORTED" if a1 and a2 else ("NOT SUPPORTED" if not a1 and not a2 else "UNCLEAR")
        return v, f"ΔHV@500 {fmt_ci(e)}; corner-bias difference vs B {fmt_ci(bias, '{:+.4f}')}"

    def bb_rule(x, _o=None):
        if hv_inst.get(x) is None:
            return "UNCLEAR", "insufficient data"
        d = paired(hv_inst[x], hv_inst["B"]); f = faster(x)
        if d is None:
            return "UNCLEAR", "insufficient data"
        if d[1] > 0 or f:
            v = "SUPPORTED"
        elif d[2] <= 0 and not f:
            v = "NOT SUPPORTED"
        else:
            v = "UNCLEAR"
        return v, f"final ΔHV {fmt_ci(d)}; faster to 95% of B on every seed: {f}"

    def overall(rows):
        vs = [r[1] for r in rows]
        if vs.count("SUPPORTED") >= 2:
            return "SUPPORTED"
        if vs.count("NOT SUPPORTED") >= 2:
            return "NOT SUPPORTED"
        return "UNCLEAR"

    rows = per_x("C", c_rule)
    out.append(f"* **C (whole-specialist copy helps early, biased toward its corner): {overall(rows)}** — " +
               "; ".join(f"{SHORT[x]}: {v} ({why})" for x, v, why in rows))
    rows = per_x("BB", bb_rule)
    out.append(f"* **BB (scheduling representation transfers): {overall(rows)}** — " +
               "; ".join(f"{SHORT[x]}: {v} ({why})" for x, v, why in rows))
    v, why = bb_rule("D")
    out.append(f"* **D (specialist behaviour transfers): {v}** — {why}")
    e500 = early("E"); dE = paired(hv_inst["E"], hv_inst["B"]) if hv_inst.get("E") is not None else None
    e0 = specialist_vs_e0(sev, keys, norm, split, size)
    if e500 is None or dE is None or e0 is None:
        out.append("* **E (merging unrelated specialists fails): UNCLEAR** — insufficient data")
    else:
        notbetter = e500[1] <= 0 and dE[1] <= 0
        e0_below = e0[2] < 0
        v = ("SUPPORTED" if notbetter and e0_below else
             "NOT SUPPORTED" if (e500[1] > 0 or dE[1] > 0 or e0[1] > 0) else "UNCLEAR")
        out.append(f"* **E (merging unrelated specialists fails): {v}** — E vs B ΔHV@500 {fmt_ci(e500)}, final "
                   f"{fmt_ci(dE)}; E0 − specialists' combined set HV {fmt_ci(e0)}")
    return out + [""]


def specialist_vs_e0(sev, keys, norm, split, size):
    diffs = []
    for v in sev.values():
        if v["arm"] != "E0":
            continue
        sp = specialist_points(split, size, v["seed"], keys)
        if sp is None:
            continue
        hs = np.array([MM.instance_hv(sp[j], *MM.ideal_nadir(norm[k])) for j, k in enumerate(keys)])
        diffs.append(np.asarray(v["npz"]["hv"]) - hs)
    if not diffs:
        return None
    return paired(np.mean(diffs, 0), np.zeros(len(keys)))


def plot_curves(sruns, size):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    cmap = plt.get_cmap("tab10")
    for c, x in enumerate([y for y in ARMS if any(r["arm"] == y and r["valiquality"] for r in sruns)]):
        rs = [r for r in sruns if r["arm"] == x and r["valiquality"]]
        allu = sorted({u for r in rs for u in val_series(r)[0]})
        Y = np.array([[MM.curve_at(*val_series(r)[:2], u) if u <= val_series(r)[0][-1] else np.nan for u in allu] for r in rs])
        H = np.array([[MM.curve_at(val_series(r)[0], val_series(r)[2], u) / 3600 if u <= val_series(r)[0][-1] else np.nan
                       for u in allu] for r in rs])
        m = np.nanmean(Y, 0)
        ax[0].plot(allu, m, color=cmap(c % 10), label=SHORT[x])
        ax[0].fill_between(allu, np.nanmin(Y, 0), np.nanmax(Y, 0), color=cmap(c % 10), alpha=.15)
        ax[1].plot(np.nanmean(H, 0), m, color=cmap(c % 10), label=SHORT[x])
    ax[0].set_xlabel("PPO updates"); ax[1].set_xlabel("training wall-clock (h)")
    for x in ax:
        x.set_ylabel("validation HV (mean over seeds; band = min-max)"); x.grid(alpha=.3)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{size}: validation hypervolume"); fig.tight_layout()
    fig.savefig(FIG / f"curves_{size}.png", dpi=110); plt.close(fig)


def plot_ppo(sruns, size):
    keys = ("entropy", "approx_kl", "clip_fraction", "value_loss", "explained_variance", "grad_norm")
    fig, axs = plt.subplots(2, 3, figsize=(14, 6.5))
    cmap = plt.get_cmap("tab10")
    for c, x in enumerate([y for y in ARMS if any(r["arm"] == y and r["reward"] for r in sruns)]):
        rs = [r for r in sruns if r["arm"] == x and r["reward"]]
        n = min(len(r["reward"]) for r in rs)
        for ax, k in zip(axs.flat, keys):
            Y = np.array([[e.get("ppo", {}).get(k, np.nan) for e in r["reward"][:n]] for r in rs], float)
            y = np.nanmean(Y, 0); w = max(1, min(50, n // 5))
            sm = np.convolve(np.nan_to_num(y), np.ones(w) / w, mode="valid")
            ax.plot(np.arange(w - 1, n), sm, color=cmap(c % 10), lw=1, label=SHORT[x])
            ax.set_title(f"{k} (rolling mean {w})", fontsize=9); ax.grid(alpha=.3)
    axs.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{size}: PPO statistics per update (mean over seeds)"); fig.tight_layout()
    fig.savefig(FIG / f"ppo_{size}.png", dpi=110); plt.close(fig)


def plot_regret(R, grid, size):
    xs = grid[:, 1] + grid[:, 2] / 2; ys = grid[:, 2] * np.sqrt(3) / 2         # simplex: m at (0,0), c at (1,0), t on top
    arms = [y for y in ARMS + ["E0"] if y in R]
    vmax = max(float(np.max(R[x]["grid"])) for x in arms) or 1.0
    fig, axs = plt.subplots(1, len(arms), figsize=(2.6 * len(arms), 2.8), squeeze=False)
    for ax, x in zip(axs[0], arms):
        sc = ax.scatter(xs, ys, c=R[x]["grid"], cmap="viridis", vmin=0, vmax=vmax, s=28)
        ax.set_title(SHORT[x], fontsize=9); ax.set_aspect("equal"); ax.axis("off")
        ax.text(-0.05, -0.08, "m", fontsize=8); ax.text(1.0, -0.08, "c", fontsize=8); ax.text(0.48, 0.9, "t", fontsize=8)
    fig.colorbar(sc, ax=axs[0].tolist(), shrink=0.8, label="mean regret")
    fig.suptitle(f"{size}: preference regret across the simplex")
    fig.savefig(FIG / f"regret_{size}.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_e0(sev, keys, size, b2, split):
    ex = ([k for k in keys if k in b2] or keys)[:3]
    e0 = sorted([v for v in sev.values() if v["arm"] == "E0" and v["npz"] is not None], key=lambda v: v["seed"])
    if not e0:
        return
    sp = specialist_points(split, size, e0[0]["seed"], keys)
    G = int(e0[0]["json"]["grid_size"])
    fig = plt.figure(figsize=(15, 4.6))
    for p, k in enumerate(ex):
        ax = fig.add_subplot(1, len(ex), p + 1, projection="3d")
        j = keys.index(k)
        pts = np.asarray(e0[0]["npz"]["objectives"])[j, :G]
        nd = pts[MM.nondominated(pts)]
        ax.scatter(*nd.T, s=16, label=f"E0 s{e0[0]['seed']} (non-dominated)")
        if k in b2:
            fr = np.array([[q[o] for o in OBJ] for q in b2[k]["front"]])
            ax.scatter(*fr.T, s=16, marker="^", label="CP-SAT B2 front")
        if sp is not None:
            ax.scatter(*sp[j].T, s=50, marker="*", label=f"specialists s{e0[0]['seed']}")
        ax.set_xlabel("makespan"); ax.set_ylabel("carbon"); ax.set_zlabel("tardiness"); ax.set_title(Path(k).stem, fontsize=9)
    fig.axes[0].legend(fontsize=7, loc="upper left"); fig.tight_layout()
    fig.savefig(FIG / f"e0_{size}.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
