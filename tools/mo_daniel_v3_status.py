"""Helpers for scripts/run_mo_daniel_v3.sh: per-run metadata and the human-readable status file
logs/mo_daniel_v3_STATUS.txt.

    python tools/mo_daniel_v3_status.py meta --name N --key value ...   (merge into results/mo_daniel_v3/runs/<N>/run_meta.json)
    python tools/mo_daniel_v3_status.py status --plan <plan.json>
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "mo_daniel_v3"
TRAIN_LOG = REPO / "daniel" / "train_log" / "SD2"
DEGRADE_TOL = 0.05          # Addendum A: DEGRADED if the mean of the last 3 validation HVs is > 5% below the best


def degraded(valiquality) -> tuple[bool, float | None]:
    hv = [e["hv"] for e in valiquality if "hv" in e]
    if len(hv) < 4 or max(hv) <= 0:
        return False, None
    drop = (max(hv) - sum(hv[-3:]) / 3) / max(hv)
    return drop > DEGRADE_TOL, drop


def meta(name, pairs):
    f = RES / "runs" / name / "run_meta.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(f.read_text()) if f.exists() else {}
    for k, v in pairs:
        if k.startswith("add_"):
            rec[k[4:]] = rec.get(k[4:], 0.0) + float(v)
        elif k.startswith("append_"):
            rec.setdefault(k[7:], []).append(v)
        else:
            rec[k] = v
    f.write_text(json.dumps(rec, indent=1))


def _h(s):
    return f"{s / 3600:.1f} h"


def status(plan_file):
    plan = json.loads(Path(plan_file).read_text())
    st = Path(plan["state_dir"])
    now = time.time()
    L = [f"MO-DANIEL v3 STATUS   updated {time.strftime('%Y-%m-%d %H:%M:%S')}   started "
         f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(plan['started']))}   elapsed {_h(now - plan['started'])}",
         f"log: {plan['log']}", ""]
    L.append("Validation normalisation points (Addendum A2):")
    for size in plan["sizes"]:
        f = RES / f"validation_normalisation_points{plan['norm_tag']}_{size}.json"
        cdir = RES / "cpsat" / "b1" / f"{plan['norm_split']}_{size}"
        n_done = len(list(cdir.glob("*.json"))) if cdir.exists() else 0
        L.append(f"  {size}: " + ("frozen" if f.exists() else f"pending (CP-SAT B1 {n_done}/{plan['norm_n'][size]})"))
    L.append("")
    done = conv = notc = failed = degr = 0
    lo = hi = 0.0
    cur = (st / "current").read_text().strip() if (st / "current").exists() else ""
    cur_line = "  current: none"
    fails = []
    for r in plan["runs"]:
        name, spu = r["name"], plan["sec_per_update"][r["size"]]
        rdir = RES / "runs" / name
        evald = RES / "eval" / f"{plan['eval_split']}_{r['size']}" / f"{name}.json"
        if (rdir / "FAILED").exists():
            failed += 1; fails.append(f"{name}: {(rdir / 'FAILED').read_text().strip()}"); continue
        if (rdir / "status.json").exists() and evald.exists():
            done += 1
            s = json.loads((rdir / "status.json").read_text())["status"]
            conv += s == "CONVERGED"; notc += s == "NOT_CONVERGED"
            degr += degraded(json.loads((rdir / "valiquality.json").read_text()))[0]
            continue
        live = TRAIN_LOG / f"{name}.status.json"
        u = 0
        if live.exists():
            ls = json.loads(live.read_text()); u = ls["update"]
            if u > 0 and ls["train_seconds"] > 0:
                spu = ls["train_seconds"] / u
            if name == cur:
                lat = ls["latest_value"]
                cur_line = (f"  current: {name}   update {u}/{plan['cap']}   latest val HV "
                            f"{(-lat if lat is not None else float('nan')):.4f}   best HV {-ls['best_value']:.4f} @ "
                            f"{ls['best_update']}   rel-impr(window) {ls['rel_improvement_window']}   {ls['status']}   "
                            f"train time {_h(ls['train_seconds'])}")
        elif name == cur:
            cur_line = f"  current: {name} (starting)"
        hi += max(0, plan["cap"] - u) * spu
        lo += max(0, plan["min_stop"] - u) * spu
    L += [f"MO-DANIEL runs: {done}/{len(plan['runs'])} done (converged {conv}, NOT converged {notc}, degraded after "
          f"best {degr}), failed {failed}", cur_line,
          f"  est. remaining training: {_h(lo)} (every run stops at ~{plan['min_stop']} updates) to {_h(hi)} "
          f"(every run to the cap {plan['cap']})"]
    L += [f"  FAILED {f}" for f in fails]
    out = REPO / "logs" / "mo_daniel_v3_STATUS.txt"
    tmp = out.with_suffix(".tmp"); tmp.write_text("\n".join(L) + "\n"); tmp.replace(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["meta", "status"])
    ap.add_argument("--name"); ap.add_argument("--plan")
    a, rest = ap.parse_known_args()
    if a.cmd == "meta":
        meta(a.name, [(rest[i].lstrip("-"), rest[i + 1]) for i in range(0, len(rest), 2)])
    else:
        status(a.plan)


if __name__ == "__main__":
    main()
