"""SHA-256 manifest of every committed input the warm-start run needs (docs/warmstart_protocol_v3.md, Inputs).

    python tools/ec2_inputs.py freeze      # write checkpoints/INPUTS.sha256.json (locally, before hand-over)
    python tools/ec2_inputs.py verify      # pre-flight: every file present and unchanged (exit 1 otherwise)

Covered: the selected specialist checkpoints (checkpoints/specialists_v3, each also checked against the
checkpoint_sha256 its evaluation recorded), the Step 4 v1 checkpoints of tests/test_v1_reproduction.py, the frozen spec,
the v3 data manifest (its instance files are hash-checked by the loader), the frozen evaluation normalisation points,
the CP-SAT B1/B2 results, and the specialists' evaluation schedules and run records used by the analysis.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "checkpoints" / "INPUTS.sha256.json"
PATTERNS = (
    "checkpoints/specialists_v3/*.pth",
    "checkpoints/step4_v1/*.pth",
    "configs/problem_spec_v3.yaml",
    "daniel/data/benchmark_v3/manifest_benchmark_v3.json",
    "results/baselines_v3/normalisation_points.json",
    "results/baselines_v3/cpsat/b1/evaluation_*/*.json",
    "results/baselines_v3/cpsat/b2/evaluation_*/*.json",
    "results/baselines_v3/eval/evaluation_*/*+mix+bv3_spec_*.npz",
    "results/baselines_v3/eval/evaluation_*/*+mix+bv3_spec_*.json",
    "results/baselines_v3/runs/*/status.json",
    "results/baselines_v3/runs/*/run_meta.json",
)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files():
    out = []
    for pat in PATTERNS:
        hit = sorted(REPO.glob(pat))
        if not hit:
            raise SystemExit(f"no input matches {pat}")
        out += hit
    return out


def freeze():
    rec = {"description": __doc__.split("\n")[0], "files": {f.relative_to(REPO).as_posix(): sha(f) for f in files()}}
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"wrote {OUT.relative_to(REPO)} ({len(rec['files'])} files)")


def verify() -> int:
    rec = json.loads(OUT.read_text())["files"]
    bad = []
    for rel, h in rec.items():
        p = REPO / rel
        if not p.exists():
            bad.append(f"missing {rel}")
        elif sha(p) != h:
            bad.append(f"hash mismatch {rel}")
    for p in sorted((REPO / "checkpoints" / "specialists_v3").glob("*.pth")):
        size = p.name.split("+")[0]
        ev = REPO / "results" / "baselines_v3" / "eval" / f"evaluation_{size}" / f"{p.stem}.json"
        if not ev.exists() or json.loads(ev.read_text())["checkpoint_sha256"] != sha(p):
            bad.append(f"{p.name} is not the checkpoint whose evaluation is reported")
    if bad:
        print("INPUT VERIFICATION FAILED:\n  " + "\n  ".join(bad[:20]))
        return 1
    print(f"inputs verified: {len(rec)} files, specialist checkpoints match their reported evaluations")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "freeze":
        freeze()
    else:
        sys.exit(verify())
