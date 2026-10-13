"""Build and freeze benchmark v3 (daniel/benchmark_v3.py, docs/benchmark_v3.md). Run ONCE; refuses to overwrite.

Instance tree daniel/data/benchmark_v3/<split>/SD2/<size>+mix/ (plain SD2 format, no carbon/priority columns):
  train        <- data_train/SD2/<size>+mix               (copies with LF line endings; the canonical grouped split)
  validation   <- data_validation/SD2/<size>+mix
  development  <- data_development/SD2/<size>+mix
  evaluation   <- data_train_vali/SD2/<size>+carbon+priority, annotations stripped (the 100-instance pool every
                  earlier evaluation used; at 10x5 it contains the 49 HELD-OUT instances, recorded as a subset)
The reserved final-test sets are NOT read and NOT part of the manifest.

Every copy is checked to have the same structural hash as its source. Splits are checked disjoint (exact and
structural hashes) and disjoint from the objective-validation DEV instances. Per instance the manifest stores the
file SHA-256, structural SHA-256, source, P_m, v_j, k_j, W_j, d_j and the reference-heuristic schedule's objectives;
per size the tardiness reference Tref (mean reference tardiness over the TRAIN pool).

    python tools/benchmark_v3/build.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.argv = sys.argv[:1]                                  # daniel/params.py parses argv on import
sys.path.insert(0, str(REPO / "daniel"))
sys.path.insert(0, str(REPO / "tools" / "objective_validation"))
import benchmark_v3 as B                                 # noqa: E402
from data_manifest import assert_disjoint_splits, structural_fjsp_sha256      # noqa: E402
from data_utils import sorted_instance_files, text_to_matrix_with_priority_carbon   # noqa: E402
import common as OV                                      # noqa: E402  (objective-validation Instance)
import schedules as S                                    # noqa: E402  (validated independent builder)

DATA = REPO / "daniel" / "data"
SOURCES = {"train": ("data_train", "+mix"), "validation": ("data_validation", "+mix"),
           "development": ("data_development", "+mix"), "evaluation": ("data_train_vali", "+carbon+priority")}
HELDOUT_SPLIT = REPO / "results" / "blend_zero_training" / "split.json"


def to_plain(src: Path, dst: Path):
    """Strip the retired priority and carbon columns; keep the header line verbatim."""
    lines = src.read_text().splitlines()
    job_length, op_pt, _, _ = text_to_matrix_with_priority_carbon(lines)
    out, o = [lines[0]], 0
    for n in job_length:
        row = [str(int(n))]
        for _ in range(int(n)):
            elig = np.flatnonzero(op_pt[o])
            row.append(str(len(elig)))
            for m in elig:
                row += [str(m + 1), str(int(op_pt[o, m]))]
            o += 1
        out.append(" ".join(row))
    dst.write_text("\n".join(out) + "\n")


def reference(job_length, op_pt, carbon, due) -> dict:
    jl = np.asarray(job_length)
    inst = OV.Instance("", jl, np.asarray(op_pt), np.repeat(np.arange(len(jl)), jl),
                       np.concatenate([[0], np.cumsum(jl)[:-1]]))
    machine, start = S.build(inst, *B.REFERENCE_RULE)
    S.validate(inst, machine, start)
    r = B.schedule_objectives(job_length, op_pt, carbon, due, machine, start)
    return {"rule": "/".join(B.REFERENCE_RULE), "makespan": r["makespan"], "carbon": r["carbon"],
            "tardiness": r["tardiness"]}


def main():
    if B.ROOT.exists() or B.MANIFEST.exists():
        raise SystemExit(f"{B.ROOT} already exists; benchmark v3 is frozen (delete it deliberately to rebuild)")
    t0 = time.time()
    rows, dirs = {}, {}
    for size in B.SIZES:
        for split, (src_split, suffix) in SOURCES.items():
            src_dir = DATA / src_split / "SD2" / f"{size}{suffix}"
            if "final_test" in str(src_dir):
                raise AssertionError("final-test data must not be read")
            dst_dir = B.ROOT / split / "SD2" / f"{size}+mix"
            dst_dir.mkdir(parents=True)
            for src in sorted_instance_files(str(src_dir)):
                src = Path(src); dst = dst_dir / src.name
                if suffix == "+mix":
                    # LF line endings: the working copies are CRLF but git stores LF, and .gitattributes marks this
                    # tree -text so the bytes (and the manifest's SHA-256) survive a fresh checkout on any platform
                    dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
                    same = structural_fjsp_sha256(dst) == structural_fjsp_sha256(src)
                else:
                    # compare matrices: data_manifest.structural_fjsp_sha256 tries the PLAIN layout first, and an
                    # annotated row can happen to parse as plain (it does for data_train_vali 10x5+mix_050.fjs)
                    to_plain(src, dst)
                    jl, pt = B.read_plain_instance(dst)
                    jl0, pt0, _, _ = text_to_matrix_with_priority_carbon(src.read_text().splitlines())
                    same = np.array_equal(jl, jl0) and np.array_equal(pt, pt0)
                if not same:
                    raise AssertionError(f"structure changed while copying {src}")
                rows[B.instance_key(dst)] = {"split": split, "size": size,
                                             "source": src.relative_to(REPO).as_posix()}
            dirs[(split, size)] = dst_dir
            if [Path(f).name for f in sorted_instance_files(str(dst_dir))] != \
                    [Path(f).name for f in sorted_instance_files(str(src_dir))]:
                raise AssertionError(f"loader order differs from the source for {split} {size}")

    # disjointness: the four splits per size, exact and structural hashes; and against the DEV instances
    disjoint = {}
    for size in B.SIZES:
        assert_disjoint_splits(**{s: dirs[(s, size)] for s in SOURCES})
        dev = {structural_fjsp_sha256(p) for p in (DATA / "objective_dev").rglob(f"{size}+mix/*.fjs")}
        ours = {structural_fjsp_sha256(B.DATA_ROOT / k) for k, r in rows.items() if r["size"] == size}
        if dev & ours:
            raise AssertionError(f"{size}: overlap with the objective-validation DEV instances")
        disjoint[size] = {"splits_pairwise": "disjoint (exact and structural SHA-256)",
                          "vs_objective_dev": f"0 overlaps against {len(dev)} DEV instances"}

    instances = {}
    for key, row in rows.items():
        path = B.DATA_ROOT / key
        p = B.generate_parameters(path, row["size"])
        job_length, op_pt = B.read_plain_instance(path)
        carbon = B.carbon_matrix(op_pt, p["power_kw"])
        instances[key] = {**row, "sha256": B.file_sha256(path), "structural_sha256": p["structural_sha256"],
                          "power_kw": p["power_kw"].tolist(), "kdraw_v": p["kdraw_v"].tolist(),
                          "k_j": p["k_j"].tolist(), "work_content": p["work_content"].tolist(),
                          "due_dates": p["due_dates"].tolist(),
                          "reference": reference(job_length, op_pt, carbon, p["due_dates"]),
                          "carbon_min": float(np.where(op_pt > 0, carbon, np.inf).min(1).sum())}

    sizes = {}
    for size in B.SIZES:
        k, s = B.DUE[size]
        tr = [r["reference"]["tardiness"] for r in instances.values() if r["size"] == size and r["split"] == "train"]
        counts = {sp: sum(1 for r in instances.values() if r["size"] == size and r["split"] == sp) for sp in SOURCES}
        sizes[size] = {"n_jobs": B.SIZES[size][0], "n_machines": B.SIZES[size][1], "k": k, "s": s,
                       "t_ref": float(np.mean(tr)), "t_ref_definition": "mean reference tardiness over the train pool",
                       "counts": counts}
        if not sizes[size]["t_ref"] > 0:
            raise AssertionError(f"{size}: reference tardiness is zero on the whole train pool")

    split_doc = json.loads(HELDOUT_SPLIT.read_text())
    ev = [Path(k).name for k, r in instances.items() if r["split"] == "evaluation" and r["size"] == "10x5"]
    assert set(split_doc["held_out"]) | set(split_doc["fit"]) == set(ev)
    order = [Path(f).name for f in sorted_instance_files(str(dirs[("evaluation", "10x5")]))]
    assert [order[i] for i in split_doc["held_idx"]] == split_doc["held_out"]
    subsets = {"evaluation/10x5/heldout_49": {"source": HELDOUT_SPLIT.relative_to(REPO).as_posix(),
                                              "source_sha256": B.file_sha256(HELDOUT_SPLIT),
                                              "files": split_doc["held_out"], "loader_idx": split_doc["held_idx"]},
               "evaluation/10x5/fit_51": {"source": HELDOUT_SPLIT.relative_to(REPO).as_posix(),
                                          "files": split_doc["fit"], "loader_idx": split_doc["fit_idx"]}}

    code = {p: B.file_sha256(REPO / p) for p in ("daniel/benchmark_v3.py", "tools/benchmark_v3/build.py",
                                                  "tools/objective_validation/schedules.py")}
    manifest = {
        "version": B.VERSION, "tag": B.TAG, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec": "configs/problem_spec_v3.yaml",
        "definitions": {
            "time_unit": "1 SD2 time unit = 1 minute", "time_unit_hours": B.TIME_UNIT_HOURS,
            "carbon": "C = gamma * sum over chosen (o, m) of P_m * p_om * time_unit_hours (kg CO2e)",
            "gamma_kg_per_kwh": B.GAMMA, "power_kw_range": list(B.POWER_KW),
            "due_dates": "d_j = k_j * W_j, k_j = k * (1 + s * (v_j - 0.5)), W_j = sum of mean eligible pt; r_j = 0",
            "rng": "numpy default_rng(int(sha256('<structural_sha256>|<purpose>')[:16], 16)); purposes power, kdraw",
            "reference_rule": "/".join(B.REFERENCE_RULE),
            "score": "w_m * MK / MKref_i + w_c * C / Cref_i + w_t * T / Tref_size",
            "observation_scales": {"time": B.PT_SCALE, "carbon": B.CARBON_SCALE, "power": B.POWER_SCALE},
        },
        "sizes": sizes, "subsets": subsets, "disjointness": disjoint,
        "final_test": "NOT INCLUDED: the reserved final-test sets were not read (docs/benchmark_v3.md)",
        "code_sha256": code, "n_instances": len(instances), "instances": instances,
    }
    B.MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"benchmark v3 frozen: {len(instances)} instances, manifest {B.MANIFEST} sha256 {B.file_sha256(B.MANIFEST)}"
          f" ({time.time() - t0:.0f} s)")
    for size, v in sizes.items():
        print(size, v)


if __name__ == "__main__":
    main()
