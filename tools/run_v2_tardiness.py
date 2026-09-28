"""Part C of results/v2_fixes.md: retrain the 10x5 tardiness specialist on v2, plus the V1-CONT control.

Arms (4 seeds each, the Step 5 tardiness-specialist protocol: configs/canonical/t.json, params.py defaults,
1000 updates, best-validation checkpoint):
  V2       --obs_version v2 --reward_version v2, init = the seed's converted tardiness specialist with the critic's
           final Linear (weight, bias) multiplied by U / Tbar_s (U = 99), so its values are on the v2 return scale;
  V1CONT   v1 as is, init = the seed's v1 tardiness specialist (1000 further updates; secondary control).
After each run the new specialist is evaluated greedily on the 49 HELD-OUT instances (per-job completions saved to
results/v2_fixes/raw/eval_<arm>_s<seed>.npz), a provenance record is written, and the unit is committed. Resume skips
units with a .done marker. Hang guard: training log silent for 420 s -> kill and stop.

    python tools/run_v2_tardiness.py [--arms V2 V1CONT] [--seeds ...]
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
CK = DAN / "trained_network/SD2"
OUT = REPO / "results/v2_fixes"
TAG, SEEDS, U = "k125s060", [111, 222, 333, 444], 99.0
PY = sys.executable


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_v2_init(seed):
    """Converted tardiness specialist with the critic rescaled to the v2 return scale."""
    import torch
    sys.path.insert(0, str(DAN))
    import reward_refs as RR
    src = CK / f"10x5+carbon+priority+t_{TAG}_s{seed}_obsv2.pth"
    dst = CK / f"10x5+carbon+priority+t_{TAG}_s{seed}_obsv2_v2init.pth"
    if dst.exists():
        return dst
    tbar = float(RR.load_manifest()["seeds"][str(seed)]["tbar"])
    ck = torch.load(src, map_location="cpu")
    sd = dict(ck["state_dict"])
    for k in ("critic.linears.2.weight", "critic.linears.2.bias"):
        sd[k] = sd[k] * (U / tbar)
    new = dict(ck, state_dict=sd)
    new["parents"] = [{"path": str(src), "sha256": sha(src),
                       "modification": f"critic final Linear x U/Tbar = {U}/{tbar} (v2 return scale)"}]
    torch.save(new, dst)
    log(f"v2 init for seed {seed}: critic scaled by {U / tbar:.6f}")
    return dst


def command(arm, seed):
    common = ["--config", "../configs/canonical/t.json",
              "--train_data_path", "./data/data_train/SD2/10x5+carbon+priority",
              "--validation_data_path", "./data/data_validation/SD2/10x5+carbon+priority",
              "--test_data_path", "./data/data_final_test/SD2/10x5+carbon+priority",     # provenance only, never read
              "--n_j", "10", "--n_m", "5", "--data_suffix", "carbon+priority", "--seed_train", str(seed),
              "--device", "cuda"]
    if arm == "V2":
        return [PY, "train.py", *common, "--obs_version", "v2", "--reward_version", "v2",
                "--init_from", str(make_v2_init(seed)), "--model_suffix", f"t_{TAG}_s{seed}_v2ft"], f"t_{TAG}_s{seed}_v2ft"
    return [PY, "train.py", *common, "--init_from", str(CK / f"10x5+carbon+priority+t_{TAG}_s{seed}.pth"),
            "--model_suffix", f"t_{TAG}_s{seed}_v1cont"], f"t_{TAG}_s{seed}_v1cont"


def evaluate(arm, seed, stem):
    sys.path.insert(0, str(REPO / "tools"))
    import numpy as np
    import v2_common as V
    import ceiling_scorer as SC
    data = V.P.load_data("data_train_vali"); due = SC.due_dates()
    obs = "v2" if arm == "V2" else "v1"
    o = V.run(V.load_policy(f"10x5+carbon+priority+{stem}", obs), data, due, SC.HELD, obs=obs)
    np.savez_compressed(OUT / "raw" / f"eval_{arm}_s{seed}.npz", held=SC.HELD, **o)
    return {k: float(np.mean(v)) for k, v in o.items() if k != "job_completion"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["V2", "V1CONT"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--hang-sec", type=int, default=420)
    a = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True); (OUT / "provenance").mkdir(exist_ok=True)
    for arm in a.arms:
        for s in a.seeds:
            done = OUT / "provenance" / f"{arm}_s{s}.done"
            if done.exists():
                log(f"skip {arm} s{s} (complete)"); continue
            cmd, stem = command(arm, s)
            lf = OUT / "provenance" / f"train_{arm}_s{s}.log"
            log(f"start {arm} s{s}")
            t0 = time.time()
            with open(lf, "w") as fh:
                proc = subprocess.Popen(cmd, cwd=DAN, stdout=fh, stderr=subprocess.STDOUT)
                last, changed = 0, time.time()
                while proc.poll() is None:
                    time.sleep(10)
                    sz = lf.stat().st_size
                    if sz != last:
                        last, changed = sz, time.time()
                    elif time.time() - changed > a.hang_sec:
                        proc.kill(); log(f"HANG: {arm} s{s} silent {a.hang_sec}s -- killed; stopping"); return 1
            if proc.returncode != 0:
                log(f"FAILED: {arm} s{s} exit {proc.returncode}; see {lf}"); return 1
            wall = time.time() - t0
            ck = CK / f"10x5+carbon+priority+{stem}.pth"
            ev = evaluate(arm, s, stem)
            import torch
            meta = torch.load(ck, map_location="cpu")
            prov = {"arm": arm, "seed": s, "checkpoint": ck.name, "checkpoint_sha256": sha(ck),
                    "best_validation_update": meta.get("update"), "selection_metric": meta.get("selection_metric"),
                    "wall_clock_seconds": round(wall, 1), "command": " ".join(cmd), "held_out_means": ev}
            (OUT / "provenance" / f"{arm}_s{s}.json").write_text(json.dumps(prov, indent=1))
            done.write_text("ok\n")
            log(f"done {arm} s{s}: {wall / 60:.1f} min, best update {prov['best_validation_update']}, HELD-OUT {ev}")
            subprocess.run(["git", "add", str(OUT / "provenance"), str(OUT / "raw")], cwd=REPO, check=True)
            subprocess.run(["git", "commit", "-q", "-m", f"v2 Part C: {arm} seed {s} (train + HELD-OUT evaluation)\n\n"
                            "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                            "Claude-Session: https://claude.ai/code/session_01Rcu6f737di34xSocjiJkEU"], cwd=REPO, check=True)
    log("PARTC_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
