"""Helpers for scripts/run_baselines_v3.sh: manifest verification, per-run metadata, CP-SAT batch completeness and
the human-readable status file logs/baselines_v3_STATUS.txt.

    python tools/baselines_v3_status.py verify
    python tools/baselines_v3_status.py meta --name N --key value ...      (merge into runs/<N>/run_meta.json)
    python tools/baselines_v3_status.py cpsat_done --part b1 --size 10x5 --split evaluation --start 0 --count 10
    python tools/baselines_v3_status.py status --plan <plan.json>
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "baselines_v3"
TRAIN_LOG = REPO / "daniel" / "train_log" / "SD2"


def _bv3():
    argv, sys.argv = sys.argv, sys.argv[:1]
    sys.path.insert(0, str(REPO / "daniel"))
    import benchmark_v3 as B
    sys.argv = argv
    return B


def verify():
    B = _bv3()
    man = B.load_manifest()
    for split in B.SPLITS:
        for size in B.SIZES:
            B.load_split(split, size, man)                      # every file hash-checked
    spec = (REPO / "configs" / "problem_spec_v3.yaml").read_text()
    if f"sha256: {B.file_sha256(B.MANIFEST)}" not in spec or "status: FROZEN" not in spec:
        raise SystemExit("frozen spec does not record this manifest")
    print(f"v3 manifest verified ({man['n_instances']} instances, sha256 {B.file_sha256(B.MANIFEST)[:12]}...)")


def meta(name, pairs):
    f = RES / "runs" / name / "run_meta.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(f.read_text()) if f.exists() else {}
    for k, v in pairs:
        if k.startswith("add_"):                                # numeric accumulators
            rec[k[4:]] = rec.get(k[4:], 0.0) + float(v)
        elif k.startswith("append_"):
            rec.setdefault(k[7:], []).append(v)
        else:
            rec[k] = v
    f.write_text(json.dumps(rec, indent=1))


def cpsat_names(part, size, split, start, count, first=None):
    B = _bv3()
    data = B.load_split(split, size)
    n = len(data) if part == "b1" else min(first if first is not None else len(data), len(data))
    return [Path(k).stem for k in data.keys[start:min(n, start + count)]]


def cpsat_done(part, size, split, start, count, first=None) -> bool:
    d = RES / "cpsat" / part / f"{split}_{size}"
    return all((d / f"{n}.json").exists() for n in cpsat_names(part, size, split, start, count, first))


def _hours(s):
    return f"{s / 3600:.1f} h"


def status(plan_file):
    plan = json.loads(Path(plan_file).read_text())
    st = Path(plan["state_dir"])
    now = time.time()
    lines = [f"baselines v3 STATUS   updated {time.strftime('%Y-%m-%d %H:%M:%S')}   mode {plan['mode']}   "
             f"started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(plan['started']))}   "
             f"elapsed {_hours(now - plan['started'])}", f"log: {plan['log']}", ""]
    if plan["runs"]:
        done = conv = notc = failed = 0
        remaining_hi = remaining_lo = 0.0
        cur = (st / "current_A").read_text().strip() if (st / "current_A").exists() else ""
        fails = []
        cur_line = "  current: none"
        for r in plan["runs"]:
            name, spu = r["name"], plan["sec_per_update"][r["size"]]
            final = RES / "runs" / name / "status.json"
            evald = RES / "eval" / f"{plan['eval_split']}_{r['size']}" / f"{name}.json"
            if (RES / "runs" / name / "FAILED").exists():
                failed += 1; fails.append(f"{name}: {(RES / 'runs' / name / 'FAILED').read_text().strip()}")
                continue
            if final.exists() and evald.exists():
                done += 1
                s = json.loads(final.read_text())["status"]
                conv += s == "CONVERGED"; notc += s == "NOT_CONVERGED"
                continue
            live = TRAIN_LOG / f"{name}.status.json"
            u = 0
            if live.exists():
                ls = json.loads(live.read_text()); u = ls["update"]
                if ls["update"] > 0 and ls["train_seconds"] > 0:
                    spu = ls["train_seconds"] / ls["update"]
                if name == cur:
                    cur_line = (f"  current: {name}   update {u}/{plan['cap']}   latest val {ls['latest_value']}   "
                                f"best {ls['best_value']:.5g} @ {ls['best_update']}   rel-impr(window) "
                                f"{ls['rel_improvement_window']}   {ls['status']}   train time {_hours(ls['train_seconds'])}")
            elif name == cur:
                cur_line = f"  current: {name} (starting)"
            remaining_hi += max(0, plan["cap"] - u) * spu
            remaining_lo += max(0, plan["min_stop"] - u) * spu
        lines += [f"PART A  specialists: {done}/{len(plan['runs'])} done (converged {conv}, NOT converged {notc}), "
                  f"failed {failed}", cur_line,
                  f"  est. remaining training: {_hours(remaining_lo)} (every run stops at ~{plan['min_stop']} updates) "
                  f"to {_hours(remaining_hi)} (every run to the cap {plan['cap']})"]
        lines += [f"  FAILED {f}" for f in fails] + [""]
    if plan["cpsat"]:
        lines.append("PART B  CP-SAT:")
        rem = 0.0
        for u in plan["cpsat"]:
            d = RES / "cpsat" / u["part"] / f"{plan['cpsat_split']}_{u['size']}"
            js = sorted(d.glob("*.json")) if d.exists() else []
            walls = [json.loads(f.read_text()).get("wall_total_s", 0.0) for f in js]
            per = (sum(walls) / len(walls)) if walls else u["worst_s"]
            rem += max(0, u["n"] - len(js)) * per
            lines.append(f"  {u['part'].upper()} {u['size']}: {len(js)}/{u['n']} instances"
                         + (f"   mean {per:.0f} s/instance" if walls else ""))
        cur = (st / "current_B").read_text().strip() if (st / "current_B").exists() else "none"
        lines += [f"  current: {cur}", f"  est. remaining: {_hours(rem)}"]
        fails = sorted((RES / "cpsat").glob("FAILED_*")) if (RES / "cpsat").exists() else []
        lines += [f"  FAILED {f.name}: {f.read_text().strip()}" for f in fails] + [""]
    out = REPO / "logs" / "baselines_v3_STATUS.txt"
    tmp = out.with_suffix(".tmp"); tmp.write_text("\n".join(lines) + "\n"); tmp.replace(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "meta", "cpsat_done", "status"])
    ap.add_argument("--name"); ap.add_argument("--plan")
    ap.add_argument("--part"); ap.add_argument("--size"); ap.add_argument("--split", default="evaluation")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--first", type=int, default=None)
    a, rest = ap.parse_known_args()
    if a.cmd == "verify":
        verify()
    elif a.cmd == "meta":
        pairs = [(rest[i].lstrip("-"), rest[i + 1]) for i in range(0, len(rest), 2)]
        if a.size:
            pairs.append(("size", a.size))
        meta(a.name, pairs)
    elif a.cmd == "cpsat_done":
        sys.exit(0 if cpsat_done(a.part, a.size, a.split, a.start, a.count, a.first) else 1)
    else:
        status(a.plan)


if __name__ == "__main__":
    main()
