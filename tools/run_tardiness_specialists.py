"""STEP 5: train the tardiness specialists -- 4 seeds x 2 sizes, canonical protocol.

Identical to the makespan/carbon/priority specialist protocol (train.py defaults, same splits,
same seeds) except for the goal (t) and the due dates, which train.py READS from the frozen,
hash-checked manifest. Checkpoint names carry the due-date tag: <size>+carbon+priority+t_<tag>_s<seed>.

Safety nets:
  * resume: a run counts as complete ONLY if this driver wrote its .done marker after a clean
    exit. train.py saves the .pth on every validation improvement, so a .pth alone proves nothing;
    an incomplete run is restarted from scratch.
  * hang guard: if the training log stops growing for --hang-sec (420 s = 7 min), the run is
    killed and the driver STOPS rather than continuing.
  * GPU health check (nvidia-smi responsive, < 85 C) before every run.
  * after each seed: a provenance record (run manifest, checkpoint SHA-256, due-date manifest
    hash, timings) is written under results/provenance/ and committed.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
CKPT = DAN / "trained_network/SD2"
PROV = REPO / "results/provenance/tardiness_specialists"
MANIFEST = DAN / "data/due_dates/due_date_manifest.json"
SEEDS = [111, 222, 333, 444]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gpu_ok(max_temp=85):
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi TIMED OUT"
    if r.returncode != 0:
        return False, f"nvidia-smi exit {r.returncode}"
    t, u, m = [x.strip() for x in r.stdout.strip().split(",")]
    return int(t) < max_temp, f"{t} C, {u}% util, {m} MiB"


def train_one(size, seed, tag, args, logdir):
    name = f"{size}+carbon+priority+t_{tag}_s{seed}"
    done = logdir / f"{name}.done"
    if done.exists():
        log(f"skip {name} (complete)"); return "skip", name
    n_j, n_m = size.split("x")
    cmd = [sys.executable, "train.py", "--config", "../configs/canonical/t.json",
           "--train_data_path", f"./data/data_train/SD2/{size}+carbon+priority",
           "--validation_data_path", f"./data/data_validation/SD2/{size}+carbon+priority",
           "--test_data_path", f"./data/data_final_test/SD2/{size}+carbon+priority",
           "--n_j", n_j, "--n_m", n_m, "--data_suffix", "carbon+priority",
           "--seed_train", str(seed), "--model_suffix", f"t_{tag}_s{seed}", "--device", "cuda",
           "--due_date_manifest", str(Path(args.manifest).resolve())]
    lf = logdir / f"train_{name}.log"
    ok, g = gpu_ok()
    log(f"GPU before {name}: {g}")
    if not ok:
        return "gpu", name
    t0 = time.time()
    with open(lf, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=DAN, stdout=fh, stderr=subprocess.STDOUT)
        last_size, last_change = 0, time.time()
        while proc.poll() is None:
            time.sleep(10)
            sz = lf.stat().st_size
            if sz != last_size:
                last_size, last_change = sz, time.time()
            elif time.time() - last_change > args.hang_sec:
                proc.kill()
                log(f"HANG: {name} log silent for {args.hang_sec}s -- killed; GPU: {gpu_ok()[1]}")
                return "hang", name
    el = time.time() - t0
    if proc.returncode != 0 or not (CKPT / f"{name}.pth").exists():
        log(f"FAILED {name}: exit {proc.returncode}; tail:\n" + "".join(lf.read_text().splitlines(True)[-8:]))
        return "fail", name
    run_json = DAN / "train_log/SD2" / f"{name}.run.json"
    rec = {"model_name": name, "checkpoint_sha256": sha(CKPT / f"{name}.pth"),
           "due_date_manifest": str(Path(args.manifest).resolve()),
           "due_date_manifest_sha256": sha(args.manifest), "dd_tag": tag, "seed": seed, "size": size,
           "wall_clock_seconds": round(el, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "command": " ".join(cmd),
           "run_manifest": json.loads(run_json.read_text()) if run_json.exists() else None}
    PROV.mkdir(parents=True, exist_ok=True)
    (PROV / f"{name}.json").write_text(json.dumps(rec, indent=2))
    done.write_text(json.dumps({"exit": 0, "seconds": round(el, 1)}))
    subprocess.run(["git", "add", str(PROV / f"{name}.json")], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-q", "-m",
                    f"Train tardiness specialist {name}\n\n"
                    "Provenance record: run manifest, checkpoint SHA-256, due-date manifest hash.\n\n"
                    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                    "Claude-Session: https://claude.ai/code/session_01Rcu6f737di34xSocjiJkEU"],
                   cwd=REPO)
    log(f"DONE {name} in {el/60:.1f} min; committed provenance")
    return "ok", name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["10x5", "20x10"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--hang-sec", type=int, default=420)
    ap.add_argument("--gap-sec", type=int, default=60)
    ap.add_argument("--log-dir", default=str(Path.home() / ".claude/jobs/programme"))
    ap.add_argument("--manifest", default=str(MANIFEST),
                    help="due-date manifest (a sensitivity manifest must carry its own k/s tag)")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    logdir = Path(args.log_dir); logdir.mkdir(parents=True, exist_ok=True)
    first = True
    for size in args.sizes:
        tag = manifest["dd_tag_by_size"][size]
        for seed in args.seeds:
            if not first:
                time.sleep(args.gap_sec)
            first = False
            status, name = train_one(size, seed, tag, args, logdir)
            if status not in ("ok", "skip"):
                log(f"STOPPED at {name}: {status}"); return 1
    log("TARDINESS_SPECIALISTS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
