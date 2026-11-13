"""Report of the specialist-reuse experiment (docs/specialist_reuse_protocol_v3.md): results/specialist_reuse_v3.md.

    python tools/analyze_specialist_reuse_v3.py [--smoke]
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel")); sys.path.insert(0, str(REPO / "tools"))
import mo_metrics as MM                                             # noqa: E402
import freeze_specialist_reuse_v3 as F                              # noqa: E402

G = 66


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def hours(s):
    return None if s is None else s / 3600


def time_to_target(valiq, target):
    for e in valiq:
        if e["hv"] >= target - 1e-12:
            return e["update"], e.get("train_seconds")
    return None, None


def metrics(obj, keys, norm, b2):
    hv, igd = [], []
    for i, k in enumerate(keys):
        ideal, nadir = MM.ideal_nadir(norm[k])
        hv.append(MM.instance_hv(obj[i, :G], ideal, nadir))
        fr = b2 / f"{Path(k).stem}.json"
        if fr.exists():
            front = np.array([[p[o] for o in MM.OBJECTIVES] for p in json.loads(fr.read_text())["front"]])
            igd.append(MM.instance_igd_plus(obj[i, :G], front, ideal, nadir))
    return float(np.mean(hv)), (float(np.mean(igd)) if igd else None), len(igd)


def render(smoke=False):
    cfg = F.load()
    name = cfg["name"] + ("_smoke" if smoke else "")
    res = REPO / "results" / name
    st = json.loads((res / "state.json").read_text())
    U = st["units"]
    norm = json.loads((REPO / cfg["evaluation"]["norm"]).read_text())["instances"]
    b2 = REPO / cfg["evaluation"]["igd_front"]
    tag = "srs" if smoke else "sr"
    model = lambda s: f"10x5+mix+bv3_{tag}_{s}_s111"                 # noqa: E731
    valiq = lambda n: json.loads((res / "runs" / n / "valiquality.json").read_text())   # noqa: E731
    rows = {}
    for m in F.METHODS:
        e = np.load(res / "eval" / f"{m}.npz")
        keys = list(e["keys"])
        ks = [k for k in e.files if k.startswith("sample") and k.endswith("_obj")]
        sk = ks[0][: -len("_obj")]
        g = metrics(e["greedy_obj"], keys, norm, b2); s = metrics(e[f"{sk}_obj"], keys, norm, b2)
        rows[m] = {"greedy_hv": g[0], "greedy_igd": g[1], "igd_n": g[2], "sample_hv": s[0], "sample_igd": s[1],
                   "greedy_spp": float(e["greedy_seconds_per_pair"]), "sample_spp": float(e[f"{sk}_seconds_per_pair"]),
                   "n": len(keys), "sample_name": sk}
    spec = [U[f"spec:{o}"] for o in "mct"]
    spec_s = sum(u["train_seconds"] for u in spec)
    spec_wall = max(u["end"] for u in spec) - min(u["start"] for u in spec)
    d1 = st.get("distill_stage1", {}).get("total_seconds", 0.0)
    target = -U["final:scratch"]["best"]
    for m in F.METHODS:
        u = U[f"final:{m}"]
        upd, sec = time_to_target(valiq(model(m)), target)
        add = u["train_seconds"] + (d1 if m == "distill" else 0.0)
        rows[m].update(final_update=u["update"], status=u["status"], capped=u["capped"], best_val=-u["best"],
                       ttt_update=upd, ttt_hours=hours(sec + (d1 if (m == "distill" and sec is not None) else 0.0))
                       if sec is not None else None,
                       spec_h=None if m == "scratch" else hours(spec_s), add_h=hours(add),
                       total_h=hours(add + (0.0 if m == "scratch" else spec_s)))
    sc = rows["scratch"]
    L = []; w = L.append
    w(f"# Specialist reuse vs from scratch, waiting allowed (10x5, seed 111){' — SMOKE TEST' if smoke else ''}")
    w("")
    w("Protocol: `docs/specialist_reuse_protocol_v3.md`; frozen settings `configs/specialist_reuse_v3.json`. Waiting "
      "option `--dispatch_rule semi_active` in specialist training, distillation, multi-objective training, validation, "
      "the TIES screen and evaluation. Single seed: numbers only.")
    w("")
    w(f"Evaluation: {sc['n']} evaluation instances x 66 preferences; HV with the frozen normalisation, reference point "
      f"1.1, per-instance mean; IGD+ on the {sc['igd_n']} instances with a CP-SAT B2 front; best-of-128 = "
      f"`{sc['sample_name']}` (seed 51).")
    w("")
    w("| Method | Greedy HV | Δ vs scratch | Best-of-128 HV | Δ vs scratch | IGD+ greedy / best-of-128 | Updates to scratch's best validation HV | Time to it (h) | Specialist training (h) | Additional training (h) | Total training (h) | Inference s/pair greedy / best-of-128 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, lab in F.METHODS.items():
        r = rows[m]
        ttt = "not reached" if r["ttt_update"] is None else f"{r['ttt_update']}"
        w(f"| {lab}{' (capped)' if r['capped'] else ''} | {f(r['greedy_hv'])} | {r['greedy_hv'] - sc['greedy_hv']:+.3f} | "
          f"{f(r['sample_hv'])} | {r['sample_hv'] - sc['sample_hv']:+.3f} | {f(r['greedy_igd'])} / {f(r['sample_igd'])} | "
          f"{ttt} | {f(r['ttt_hours'], 2) if r['ttt_update'] is not None else 'not reached'} | {f(r['spec_h'], 2) if r['spec_h'] is not None else '-'} | "
          f"{f(r['add_h'], 2)} | {f(r['total_h'], 2)} | {f(r['greedy_spp'], 4)} / {f(r['sample_spp'], 4)} |")
    w("")
    w(f"- **Target** for time to target: From scratch's best validation HV, {f(target, 4)} (greedy, 18 validation "
      "instances x 15 preferences, every 50 updates).")
    w("- **Updates and time to target** count the method's own multi-objective training. Distillation's time also "
      "includes its stage 1.")
    w(f"- **Specialist training** is the sum of the three specialists, {f(hours(spec_s), 2)} h. They ran in parallel, "
      f"{f(hours(spec_wall), 2)} h wall-clock.")
    w("- **Cost of a run when the specialists already exist:** the Additional column, which is Distillation stage 1 "
      "plus fine-tuning, or merge fine-tuning.")
    w("- **Concurrency:** runs shared 3 slots, which inflates wall-clock times. Inference was timed with the GPU alone "
      "and the same batching for every method.")
    w("")
    w("Full runs: " + "; ".join(f"{F.METHODS[m]} {rows[m]['status']} at update {rows[m]['final_update']} "
                                f"(best validation HV {f(rows[m]['best_val'], 4)})" for m in F.METHODS) + ".")
    w("")
    # tuning
    w("## Tuning (validation; equal budget per method)")
    w("")
    w(f"Each method: {len([k for k in U if k.startswith('trial:scratch')])} trials x "
      f"{cfg['smoke']['updates'] if smoke else cfg['tuning']['updates_per_trial']} updates over lr x beta. "
      "Score: best validation HV in the trial. Ties go to the setting closest to the recipe.")
    w("")
    w("| Method | lr | beta | Best validation HV | Trial time (h) | Selected |")
    w("|---|---|---|---|---|---|")
    cost = {}
    for m in F.METHODS:
        sel = st["selected"][m]
        tot = 0.0
        for k in sorted(x for x in U if x.startswith(f"trial:{m}:")):
            u = U[k]; t = cfg["tuning"]["trials"][int(k.split(":")[2])]
            tot += u["train_seconds"]
            w(f"| {F.METHODS[m]} | {t['lr']:g} | {t['cl_beta']:g} | {f(-u['best'], 4)} | {f(hours(u['train_seconds']), 2)} | "
              f"{'**yes**' if t == sel else ''} |")
        cost[m] = tot
    scr = st.get("ties_screen", {})
    cost["ties"] += sum(r["seconds"] for r in scr.values())
    w("")
    w("TIES merge screen (merged start, no training; validation HV at update 0):")
    w("")
    w("| keep | lambda | Validation HV | Selected |")
    w("|---|---|---|---|")
    for r in scr.values():
        w(f"| {r['keep']} | {r['lambda']} | {f(r['hv'], 4)} | {'**yes**' if [r['keep'], r['lambda']] == st['ties_selected'] else ''} |")
    w("")
    w("Tuning cost (h): " + "; ".join(f"{F.METHODS[m]} {f(hours(c), 2)}" for m, c in cost.items()) +
      f". TIES includes its screen. Distillation stage 1 ({f(hours(d1), 2)} h) is shared by the trials and the full "
      "run, and is counted in the Additional column.")
    w("")
    # checks
    ch = json.loads((res / "merge_checks.json").read_text())
    w("## Checks before the multi-objective stages (real specialists)")
    w("")
    w("| Merge | Preference change changes the weights (max difference) | Corners reproduce the specialist: weights / max probability difference (M, C, T) |")
    w("|---|---|---|")
    for k, v in ch["checks"].items():
        cs = ", ".join(f"{'yes' if c['weights_equal'] else 'NO'} / {c['max_prob_difference']:.1e}" for c in v["corners"].values())
        w(f"| {k} | {'yes' if v['preference_changes_weights'] else 'NO'} ({v['max_weight_difference_w1_w2']:.3g}) | {cs} |")
    w("")
    w(f"Corner outputs were compared on {ch['states_compared']} validation states visited under the waiting rule.")
    w("")
    w("## Specialists")
    w("")
    for o in "mct":
        u = U[f"spec:{o}"]
        w(f"- {o}: {u['status']}{' (capped)' if u['capped'] else ''} at update {u['update']}, best own-objective "
          f"validation score {f(u['best'], 4)} @ {u['best_update']}, {f(hours(u['train_seconds']), 2)} h")
    md = REPO / "results" / f"{name}.md"
    md.write_text("\n".join(L) + "\n")
    (res / "summary.json").write_text(json.dumps({"rows": rows, "tuning_cost_seconds": cost, "target": target}, indent=1) + "\n")
    print(f"written: {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    render(ap.parse_args().smoke)
