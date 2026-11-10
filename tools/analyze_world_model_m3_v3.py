"""Renders results/world_model_m3_v3.md (docs/world_model_m3_protocol_v3.md): preference-weighted merging of four
specialist sets. Part A (no training): fronts, corner vs in-between quality, ridge test, the Mergeable criterion.
Part B (merge fine-tuning), once its runs exist: training and evaluation next to From scratch and Distillation, the
Better and Faster criteria, costs. Writes results/world_model_m3_v3/criteria.json and figures.

    python tools/analyze_world_model_m3_v3.py [--smoke]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import analyze_warmstart_v3 as AW                                   # noqa: E402  (reference points, CP-SAT loaders)
import freeze_world_model_m3_v3 as F                                # noqa: E402
import benchmark_v3 as B                                            # noqa: E402
import mo_metrics as MM                                             # noqa: E402

PAIR_NAME = {"mc": "Makespan-Carbon", "mt": "Makespan-Tardiness", "ct": "Carbon-Tardiness"}
OBJ_NAME = {"m": "makespan", "c": "carbon", "t": "tardiness"}
SETS = ["wm", "ctrl", "orig", "old"]


def f3(x, f="{:.3f}"):
    return "-" if x is None else f.format(x)


# ------------------------------------------------------------------------------------------------------- Part A data
def load_front(res, split, s, tag, sfx=""):
    p = res / "part_a" / f"{split}_10x5" / f"10x5+mix+bv3_{tag}_{s}_s111{sfx}"
    if not p.with_suffix(".json").exists():
        return None
    return {"json": json.loads(p.with_suffix(".json").read_text()), "npz": np.load(p.with_suffix(".npz"))}


def corner_mask(grid, thr):
    return grid.max(1) >= thr - 1e-9


def regret_pool(split, keys, cfg, fronts):
    """Per instance key: raw candidate points [n, 3] (frozen pool, docs/world_model_m3_protocol_v3.md)."""
    b1, b2 = AW.load_cpsat("b1", split, "10x5"), AW.load_cpsat("b2", split, "10x5")
    refs = AW.reference_points(split, "10x5", keys, b1, b2)
    G = len(MM.simplex_grid(cfg["part_a"]["grid_step"]))
    extra = []
    if split == "evaluation":
        for name in ("From scratch", "Distillation"):
            z = np.load(REPO / cfg["references"][name]["eval_npz"])
            pos = {str(k): j for j, k in enumerate(z["keys"])}
            extra.append({k: z["objectives"][pos[k], :G] for k in keys})
    pool = {}
    for i, k in enumerate(keys):
        c = [r[k] for r in refs.values() if k in r] + [e[k] for e in extra] + [f[i, :G] for f in fronts]
        if k in b2:
            c.append(np.array([[q[o] for o in MM.OBJECTIVES] for q in b2[k]["front"]]))
        pool[k] = np.concatenate(c)
    return pool


def regret(points, keys, pool, norm, grid, own=True):
    """[n, G] regret of every preference's point: lambda . f - min over the pool (plus own points) of lambda . z."""
    out = np.zeros((len(keys), len(grid)))
    for i, k in enumerate(keys):
        ideal, nadir = MM.ideal_nadir(norm[k])
        cand = np.concatenate([pool[k], points[i]]) if own else pool[k]
        Z, keep = MM.normalise(cand, ideal, nadir)
        lam = grid[:, keep]
        best = (Z @ lam.T).min(0)
        f, _ = MM.normalise(points[i], ideal, nadir)
        out[i] = (f * lam).sum(1) - best
    return out


def nd_share(points):
    """[n, G] 1 where the preference's schedule is not strictly dominated by another schedule of the same front."""
    out = np.zeros(points.shape[:2])
    for i, p in enumerate(points):
        for g in range(len(p)):
            out[i, g] = not np.any(np.all(p <= p[g], 1) & np.any(p < p[g], 1))
    return out


def ridge(points, keys, norm, grid, pair):
    """Mean normalised objectives along theta_t = t theta_a + (1-t) theta_b (simplex-edge grid points), the scalarised
    score s(t), the barrier and each objective's own dip."""
    a, b = "mct".index(pair[0]), "mct".index(pair[1])
    ts = np.round(np.arange(11) / 10, 1)
    gi = []
    for t in ts:
        w = np.zeros(3); w[a], w[b] = t, round(1 - t, 1)
        gi.append(int(np.flatnonzero(np.all(np.abs(grid - w) < 1e-9, 1))[0]))
    fa, fb = np.zeros((len(keys), 11)), np.zeros((len(keys), 11))
    for i, k in enumerate(keys):
        ideal, nadir = MM.ideal_nadir(norm[k])
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        z = np.where(nadir > ideal, (points[i, gi] - ideal) / span, 0.0)
        fa[i], fb[i] = z[:, a], z[:, b]
    ga, gb = fa.mean(0), fb.mean(0)
    s = ts * ga + (1 - ts) * gb
    line = lambda y: ts * y[-1] + (1 - ts) * y[0]
    return {"t": ts.tolist(), "score": s.tolist(), "f_a": ga.tolist(), "f_b": gb.tolist(),
            "barrier": float(np.max(s - line(s))), "dip_a": float(np.max(ga - line(ga))),
            "dip_b": float(np.max(gb - line(gb))), "grid_index": gi}


# ------------------------------------------------------------------------------------------------------- Part B data
def load_run(res, name):
    d = res / "runs" / name
    if not (d / "done.json").exists():
        return None
    val = json.loads((d / "valiquality.json").read_text())
    st = json.loads((d / "status.json").read_text())
    ev = res / "eval" / "evaluation_10x5"
    if not (ev / f"{name}.json").exists():
        ev = next((p for p in (res / "eval").glob("*_10x5") if (p / f"{name}.json").exists()), ev)
    return {"val": val, "status": st, "eval": json.loads((ev / f"{name}.json").read_text()),
            "npz": np.load(ev / f"{name}.npz"), "proc": json.loads((d / "process.json").read_text())["process_seconds"]
            if (d / "process.json").exists() else None}


def first_reach_hv(val, target):
    for e in val:
        if e["hv"] >= target - 1e-12:
            return e["update"], e["train_seconds"]
    return None


def cc_reference(cfg, name):
    r = cfg["references"][name]
    val = json.loads((REPO / r["valiquality"]).read_text())
    st = json.loads((REPO / r["status"]).read_text())
    ev = json.loads((REPO / r["eval_json"]).read_text())
    best = max(val, key=lambda e: e["hv"])
    return {"val0": val[0]["hv"], "best": best["hv"], "best_update": best["update"], "stop": st["update"],
            "status": st["status"], "eval": ev, "npz": np.load(REPO / r["eval_npz"]),
            "hours": st["train_seconds"] / 3600}


def spec_hours(cfg, s):
    st = cfg["sets"][s]
    h = [json.loads((REPO / st["cost_status"][o]).read_text())["train_seconds"] / 3600 for o in "mct"]
    base = (json.loads((REPO / st["shared_base_status"]).read_text())["train_seconds"] / 3600
            if "shared_base_status" in st else 0.0)
    return sum(h) + base, max(h) + base, base


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = F.load()
    sfx = "_smoke" if a.smoke else ""
    res = REPO / "results" / f"world_model_m3_v3{sfx}"
    out_md = REPO / "results" / f"world_model_m3_v3{sfx}.md"
    fig = res / "figures"; fig.mkdir(parents=True, exist_ok=True)
    tagA = "wm3As" if a.smoke else "wm3A"
    tagB = "wm3s" if a.smoke else "wm3"
    pa = cfg["part_a"]
    grid = MM.simplex_grid(pa["grid_step"]); G = len(grid)
    corner = corner_mask(grid, pa["corner"])
    label = {s: cfg["sets"][s]["label"] for s in SETS}
    crit = {}
    L = ["# World-model M3: preference-weighted merging (10x5, seed 111)", "",
         "Protocol: `docs/world_model_m3_protocol_v3.md`. Generated by `tools/analyze_world_model_m3_v3.py`.",
         "- **Single seed (111):** numbers only, no significance claims.",
         "- **Direction:** higher HV is better; lower IGD+, regret and barrier are better.",
         "- **Data:** the reserved final-test sets were never loaded.", ""]
    if a.smoke:
        L += ["**SMOKE RUN: tiny budgets on a few instances. These numbers mean nothing.**", ""]
    m = cfg["merge"]
    L += ["## Merge method (identical for every set)", "",
          f"- **Merge:** {m['definition']}.",
          f"- **Trunk and actor:** {m['actor_and_trunk']}.",
          f"- **Critic:** {m['critic']}.",
          f"- **Preference columns:** {m['preference_columns']}.", "",
          "**Specialist sets** (seed 111; makespan, carbon and tardiness specialists each):", ""]
    for s in SETS:
        st = cfg["sets"][s]
        L.append(f"- **{label[s]}:** `{st['specialists']['m'].replace('_m_', '_{m,c,t}_')}`"
                 + ("" if st["part_b"] else " (Part A only)"))
    L.append("")

    # ---------------------------------------------------------------- Part A
    A = {split: {s: load_front(res, split, s, tagA) for s in SETS} for split in ("validation", "evaluation")}
    V15 = {s: load_front(res, "validation", s, tagA, "_g025") for s in SETS}
    have_a = all(A[sp][s] is not None for sp in A for s in SETS)
    partA = {}
    if have_a:
        for split in ("validation", "evaluation"):
            keys = [str(k) for k in A[split]["wm"]["npz"]["keys"]]
            norm = json.loads((REPO / (pa["eval_norm"] if split == "evaluation" else pa["val_norm"])).read_text())["instances"]
            pts = {s: np.asarray(A[split][s]["npz"]["objectives"])[:, :G] for s in SETS}
            pool = regret_pool(split, keys, cfg, list(pts.values()))
            for s in SETS:
                R = regret(pts[s], keys, pool, norm, grid, own=False)
                N = nd_share(pts[s])
                partA.setdefault(s, {})[split] = {
                    "hv": A[split][s]["json"]["hv_mean"], "igd": A[split][s]["json"]["igd_plus_mean"],
                    "igd_n": A[split][s]["json"]["igd_plus_n"], "nd": A[split][s]["json"]["nd_size_mean"],
                    "regret_all": float(R.mean()), "regret_corner": float(R[:, corner].mean()),
                    "regret_between": float(R[:, ~corner].mean()),
                    "nd_corner": float(N[:, corner].mean()), "nd_between": float(N[:, ~corner].mean()),
                    "regret_grid": R.mean(0).tolist(), "n": len(keys)}
                if split == "validation":
                    partA[s]["ridge"] = {p: ridge(np.asarray(A[split][s]["npz"]["objectives"]), keys, norm, grid, p)
                                         for p in pa["pairs"]}
                    partA[s]["val15"] = V15[s]["json"]["hv_mean"] if V15[s] else None
            if split == "evaluation":
                eval_pool, eval_keys, eval_norm = pool, keys, norm
        e0 = cfg["references"]["e0"]
        z_e0 = np.load(REPO / e0["111"]["npz"])
        n_ev = partA["orig"]["evaluation"]["n"]
        same = np.array_equal(np.asarray(A["evaluation"]["orig"]["npz"]["objectives"]), z_e0["objectives"][:n_ev])
        e0_mean = float(np.mean([e0[s]["hv"] for s in ("111", "222", "333")]))
        L += ["## Part A: merging without training", "",
              f"**Rollouts:** greedy, one merged network per preference: the 66-point grid (step {pa['grid_step']}) plus "
              f"equal weights.",
              f"- **Evaluation split:** {partA['wm']['evaluation']['n']} instances. IGD+ uses the CP-SAT B2 subset "
              f"({partA['wm']['evaluation']['igd_n']} instances).",
              f"- **Validation split:** {partA['wm']['validation']['n']} instances, with the frozen validation "
              "normalisation. There are no B2 fronts on validation, so validation IGD+ is n/a.",
              "- **'Validation HV, 15 preferences'** is measured on the training validation grid (step 0.25). It is "
              "directly comparable with the best validation HVs from training: From scratch 0.542, Distillation 0.540.",
              ""]
        cc = {n: cc_reference(cfg, n) for n in ("From scratch", "Distillation")}
        L += ["### Fronts", "",
              "| Merge | Validation HV, 15 preferences | Validation HV, 66 | Validation unique non-dominated | "
              "Evaluation HV | Evaluation IGD+ (B2) | Evaluation unique non-dominated |",
              "|---|---|---|---|---|---|---|"]
        for s in SETS:
            v, e = partA[s]["validation"], partA[s]["evaluation"]
            L.append(f"| {label[s]} | {f3(partA[s]['val15'])} | {f3(v['hv'])} | {f3(v['nd'], '{:.1f}')} | {f3(e['hv'])} | "
                     f"{f3(e['igd'])} | {f3(e['nd'], '{:.1f}')} |")
        for n, r in cc.items():
            L.append(f"| *{n}, converged contrastive (reference)* | {f3(r['best'])} (best) | - | - | "
                     f"{f3(r['eval']['hv_mean'])} | {f3(r['eval'].get('igd_plus_mean'))} | "
                     f"{f3(r['eval'].get('nd_size_mean'), '{:.1f}')} |")
        L += ["", "**Original specialists vs the earlier zero-training interpolation (E0):**",
              f"- **Reproduction:** the Original specialists mix gives evaluation HV **{partA['orig']['evaluation']['hv']:.4f}**. "
              f"Its objectives are {'bit-identical to' if same else 'NOT identical to'} the committed E0 seed-111 front "
              f"(`{e0['111']['npz']}`), whose HV is {e0['111']['hv']:.4f}.",
              f"- **Why it differs from 0.555:** 0.555 is the **mean over seeds 111, 222 and 333** "
              f"({e0['111']['hv']:.3f} / {e0['222']['hv']:.3f} / {e0['333']['hv']:.3f}, mean {e0_mean:.3f}). M3 uses seed "
              "111 only, so the right comparison is seed 111's 0.571. The method, grid, normalisation and code path "
              "are the same.", ""]
        L += ["### Per-preference quality: corner vs in-between", "",
              f"- **Corner:** any weight ≥ {pa['corner']} ({int(corner.sum())} of {G} preferences). "
              f"**In-between:** the other {int((~corner).sum())}.",
              "- **Mean regret:** λ·f − min over the pool of λ·z, in normalised space, averaged over instances and "
              "preferences.",
              f"  - **Pool, per split and instance:** {pa['regret_pool']}.",
              f"- **Share non-dominated:** {pa['nondominated_share']}.", "",
              "| Merge | Split | Mean regret, corner | Mean regret, in-between | Mean regret, all | "
              "Share non-dominated, corner | Share non-dominated, in-between |", "|---|---|---|---|---|---|---|"]
        for split in ("evaluation", "validation"):
            for s in SETS:
                q = partA[s][split]
                L.append(f"| {label[s]} | {split} | {q['regret_corner']:.4f} | {q['regret_between']:.4f} | "
                         f"{q['regret_all']:.4f} | {q['nd_corner']:.2f} | {q['nd_between']:.2f} |")
        L += ["", "### Ridge test (validation)", "",
              "- **Path:** θ_t = t·θ_a + (1−t)·θ_b for t = 0, 0.1, …, 1. These are the simplex-edge points of the 66-point "
              "grid, the same rollouts as above.",
              "- **Score:** s(t) = t·f_a + (1−t)·f_b, where f is the normalised objective (validation ideal/nadir) "
              "averaged over instances. Lower is better.",
              "- **Barrier:** max_t [s(t) − straight line between s(0) and s(1)].",
              "  - **> 0:** there is a ridge (the path is worse than the line).",
              "  - **≤ 0:** no ridge.",
              "- **Dip a / dip b:** the same measure for each objective on its own.", "",
              "| Merge | " + " | ".join(f"{PAIR_NAME[p]}: barrier (dip a / dip b)" for p in pa["pairs"]) + " |",
              "|---|" + "---|" * len(pa["pairs"])]
        for s in SETS:
            L.append(f"| {label[s]} | " + " | ".join(
                f"{partA[s]['ridge'][p]['barrier']:+.4f} ({partA[s]['ridge'][p]['dip_a']:+.4f} / "
                f"{partA[s]['ridge'][p]['dip_b']:+.4f})" for p in pa["pairs"]) + " |")
        L += ["", "![ridge test](world_model_m3_v3{}/figures/ridge.png)".format(sfx), ""]
        plot_ridge(partA, label, pa["pairs"], fig / "ridge.png")
        # Mergeable
        c = cfg["criteria"]["mergeable"]
        rw, rc = partA["wm"][c["regret_split"]]["regret_between"], partA["ctrl"][c["regret_split"]]["regret_between"]
        ok_a = rw <= c["regret_ratio"] * rc
        smaller = [p for p in pa["pairs"] if partA["wm"]["ridge"][p]["barrier"] < partA["ctrl"]["ridge"][p]["barrier"]]
        ok_b = len(smaller) >= c["barrier_min_pairs"]
        crit["mergeable"] = {"regret_between_wm": rw, "regret_between_ctrl": rc, "ratio": rw / rc if rc else None,
                             "a": ok_a, "pairs_smaller_barrier": smaller, "b": ok_b, "verdict": ok_a and ok_b}
        partA_summary = {s: {k: v for k, v in partA[s].items()} for s in SETS}
    else:
        L += ["## Part A: merging without training", "", "_Not run yet._", ""]
        partA_summary = None

    # ---------------------------------------------------------------- Part B
    runs = {s: load_run(res, f"10x5+mix+bv3_{tagB}_{s}_s111") for s in SETS if cfg["sets"][s]["part_b"]}
    L += ["## Part B: merge fine-tuning", "",
          "Sets 1–3 only. Three trainable specialist networks; for each training sample with preference w, the "
          "policy is the mix for that w, and gradients update all three networks.",
          "- **Settings:** the multi-objective contrastive settings, unchanged: Dirichlet(1,1,1) preferences per "
          "environment, K 8, β 0.1, margin 0.01, lr 3e-4, clip 1.0 on the joint gradient norm, 20 instances "
          "resampled every 20 updates.",
          "- **Reference:** the frozen starting mix for the same w.",
          f"- **Stopping rule:** {cfg['part_b']['rule']['statement']}.", ""]
    have_b = any(r is not None for r in runs.values())
    partB = {}
    if have_b:
        ccr = {n: cc_reference(cfg, n) for n in ("From scratch", "Distillation")}
        L += ["| Arm | HV at update 0 | Best validation HV | Update of best | Stop update | Evaluation HV | IGD+ | "
              "Unique non-dominated | Mean regret |", "|---|---|---|---|---|---|---|---|---|"]
        for s, r in runs.items():
            if r is None:
                L.append(f"| {label[s]} | not run | | | | | | | |"); continue
            val, st = r["val"], r["status"]
            best = max(val, key=lambda e: e["hv"])
            pts = np.asarray(r["npz"]["objectives"])[:, :G]
            keys = [str(k) for k in r["npz"]["keys"]]
            reg = (float(regret(pts, keys, eval_pool, eval_norm, grid).mean())
                   if have_a and keys == eval_keys[:len(keys)] and len(keys) == len(eval_keys) else None)
            stop = f"{st['update']} ({'converged' if st['status'] == 'CONVERGED' else 'capped'})"
            partB[s] = {"val0": val[0]["hv"], "best": best["hv"], "best_update": best["update"], "stop": st["update"],
                        "status": st["status"], "eval_hv": r["eval"]["hv_mean"], "igd": r["eval"].get("igd_plus_mean"),
                        "nd": r["eval"].get("nd_size_mean"), "regret": reg, "hours": st["train_seconds"] / 3600,
                        "reach": first_reach_hv(val, cfg["criteria"]["faster"]["target_hv"])}
            q = partB[s]
            L.append(f"| {label[s]} | {q['val0']:.3f} | {q['best']:.3f} | {q['best_update']} | {stop} | {q['eval_hv']:.3f} | "
                     f"{f3(q['igd'])} | {f3(q['nd'], '{:.1f}')} | {f3(q['regret'], '{:.4f}')} |")
        for n, r in ccr.items():
            pts = np.asarray(r["npz"]["objectives"])[:, :G]
            reg = float(regret(pts, eval_keys, eval_pool, eval_norm, grid).mean()) if have_a else None
            L.append(f"| *{n}, converged (reference)* | {r['val0']:.3f} | {r['best']:.3f} | {r['best_update']} | "
                     f"{r['stop']} ({'converged' if r['status'] == 'CONVERGED' else r['status']}) | "
                     f"{r['eval']['hv_mean']:.3f} | {f3(r['eval'].get('igd_plus_mean'))} | "
                     f"{f3(r['eval'].get('nd_size_mean'), '{:.1f}')} | {f3(reg, '{:.4f}')} |")
        L += ["", "Mean regret uses the frozen Part A evaluation pool, plus the arm's own points.", "",
              "![validation HV](world_model_m3_v3{}/figures/partb_validation.png)".format(sfx), ""]
        plot_partb(runs, label, cfg["criteria"]["faster"]["target_hv"], fig / "partb_validation.png")
        # Better / Faster
        if runs.get("wm") and runs.get("ctrl"):
            cb, cf = cfg["criteria"]["better"], cfg["criteria"]["faster"]
            hw, hc = partB["wm"]["eval_hv"], partB["ctrl"]["eval_hv"]
            crit["better"] = {"eval_hv_wm": hw, "eval_hv_ctrl": hc, "ratio": hw / hc, "verdict": hw >= (1 + cb["margin"]) * hc}
            rw_, rc_ = partB["wm"]["reach"], partB["ctrl"]["reach"]
            if rw_ is None:
                fv, ratio = "no (world-model mix not reached)", None
            elif rc_ is None:
                fv, ratio = "yes (control not reached)", None
            else:
                ratio = rw_[0] / rc_[0] if rc_[0] else (0.0 if rw_[0] == 0 else float("inf"))
                fv = "yes" if rw_[0] <= cf["ratio"] * rc_[0] else "no"
            crit["faster"] = {"updates_wm": rw_[0] if rw_ else None, "updates_ctrl": rc_[0] if rc_ else None,
                              "ratio": ratio, "verdict": fv}
        # costs
        L += ["### Cost", "",
              "- **Merge fine-tuning:** this Part B run.",
              "- **+ specialists:** the three specialists' training. For the Old shared-base specialists, their shared "
              "PPO base as well.",
              "- **+ world model:** M1 pretraining, 1.27 h (0.22 h data + 1.05 h GPU). It applies to the world-model set "
              "only and is paid once for all objectives.",
              "- **Parallel:** wall-clock if the three specialists train in parallel (the longest one, plus a shared base "
              "where there is one).", "",
              "| Arm | Merge fine-tuning (h) | + specialist training (h) | + world-model pretraining (h) | "
              "Same, specialists in parallel (h) |", "|---|---|---|---|---|"]
        for s, q in partB.items():
            tot, par, _ = spec_hours(cfg, s)
            wmh = cfg["world_model_cost_h"]["total"] if s == "wm" else 0.0
            L.append(f"| {label[s]} | {q['hours']:.2f} | {q['hours'] + tot:.2f} | {q['hours'] + tot + wmh:.2f} | "
                     f"{q['hours'] + par + wmh:.2f} |")
        for n, r in ccr.items():
            L.append(f"| *{n}, converged (reference)* | {r['hours']:.2f} | - | - | - |")
        L += ["", "Runs shared one GPU in up to 3 parallel slots, so hours are approximate; updates are the primary "
              "measure.", ""]
    else:
        L += ["_Not run yet._ Command: `bash scripts/run_world_model_m3_v3.sh --jobs 3`.", ""]

    # ---------------------------------------------------------------- verdicts
    L += ["## Verdicts (H2: world-model base vs random shared base)", ""]
    if "mergeable" in crit:
        c = crit["mergeable"]
        L += [f"- **Mergeable: {'YES' if c['verdict'] else 'NO'}.**",
              f"  - **(a)** In-between mean regret (evaluation): {c['regret_between_wm']:.4f} vs "
              f"{c['regret_between_ctrl']:.4f}, ratio {c['ratio']:.3f}. Needs ≤ 0.90 → {'yes' if c['a'] else 'no'}.",
              f"  - **(b)** Smaller barrier on {len(c['pairs_smaller_barrier'])} of 3 pairs "
              f"({', '.join(PAIR_NAME[p] for p in c['pairs_smaller_barrier']) or 'none'}). Needs ≥ 2 → "
              f"{'yes' if c['b'] else 'no'}."]
    else:
        L.append("- **Mergeable:** pending (Part A not run).")
    if "better" in crit:
        c = crit["better"]
        L.append(f"- **Better after merging: {'YES' if c['verdict'] else 'NO'}.** Evaluation HV {c['eval_hv_wm']:.3f} vs "
                 f"{c['eval_hv_ctrl']:.3f}, ratio {c['ratio']:.3f}. Needs ≥ 1.05.")
        c = crit["faster"]
        L.append(f"- **Faster after merging: {c['verdict'].upper()}.** Updates to reach validation HV 0.542: "
                 f"world-model mix {c['updates_wm'] if c['updates_wm'] is not None else 'not reached'}, "
                 f"random shared base mix {c['updates_ctrl'] if c['updates_ctrl'] is not None else 'not reached'}. "
                 "Needs ≤ 70%.")
    else:
        L += ["- **Better after merging:** pending (Part B not run).", "- **Faster after merging:** pending (Part B not run)."]
    L += ["", "Single seed: numbers only.", ""]
    out_md.write_text("\n".join(L))
    (res / "criteria.json").write_text(json.dumps({"criteria": crit, "part_a": partA_summary, "part_b": partB},
                                                  indent=1, default=lambda o: None) + "\n")
    print(f"wrote {out_md.relative_to(REPO)}")


def plot_ridge(partA, label, pairs, path):
    fig, ax = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.4), sharey=False)
    for j, p in enumerate(pairs):
        for s in SETS:
            r = partA[s]["ridge"][p]
            ax[j].plot(r["t"], r["score"], marker="o", ms=3, label=label[s])
        ax[j].set_title(f"{PAIR_NAME[p]} (t = weight on {OBJ_NAME[p[0]]})", fontsize=9)
        ax[j].set_xlabel("t"); ax[j].set_ylabel("s(t), normalised (lower is better)")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_partb(runs, label, target, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for s, r in runs.items():
        if r is not None:
            ax.plot([e["update"] for e in r["val"]], [e["hv"] for e in r["val"]], label=label[s])
    ax.axhline(target, ls="--", c="grey", lw=0.8, label=f"From scratch best ({target})")
    ax.set_xlabel("update"); ax.set_ylabel("validation HV (15 preferences)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
