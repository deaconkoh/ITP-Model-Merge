"""Generate the DEV-CAL and DEV-CHECK instances with the existing SD2 generator and dedicated seeds, draw and store
every per-instance parameter, and check disjointness from every existing instance pool.

  instances   daniel/data/objective_dev/{cal,check}/SD2/<size>+mix/<size>+mix_NNN.fjs  (standard SD2 format)
  manifest    daniel/data/objective_dev/manifest_objective_dev.json: per instance file SHA-256, structural SHA-256,
              P_m (kW), v_j (k_j draws), old i.i.d. carbon (comparison only), plus the generation settings
  report      results/objective_validation/disjointness.json

data_final_test is never opened (its disjointness is argued from its generator seed). HELD-OUT files are hashed only.

    python tools/objective_validation/generate.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                      # noqa: E402

_argv = sys.argv[:]; sys.argv = [sys.argv[0]]           # daniel/params parses argv on import
sys.path.insert(0, str(C.DAN))
from data_utils import SD2_instance_generator, matrix_to_text   # noqa: E402
from data_manifest import structural_fjsp_sha256                # noqa: E402
sys.argv = _argv


def generate():
    if C.MANIFEST.exists():
        raise SystemExit(f"{C.MANIFEST} exists; DEV instances are frozen")
    recs = {}
    for (split, size), seed in C.SEEDS.items():
        n_j, n_m = C.SIZES[size]
        cfg = SimpleNamespace(n_j=n_j, n_m=n_m, op_per_job=0, low=1, high=99, data_suffix="mix",
                              op_per_mch_min=1, op_per_mch_max=n_m)
        d = C.instance_dir(split, size); d.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)                              # as generate_data_to_files
        for idx in range(C.COUNTS[split]):
            jl, pt, opm = SD2_instance_generator(cfg)
            f = d / f"{size}+mix_{idx + 1:03d}.fjs"
            f.write_text("\n".join(matrix_to_text(jl, pt, opm)) + "\n")
            key = C.instance_key(f)
            elig = pt > 0
            old = C.keyed_rng(key, "oldcarbon").integers(1, 101, size=pt.shape) * elig
            recs[key] = {"split": split, "size": size, "sha256": C.file_sha256(f),
                         "structural_sha256": structural_fjsp_sha256(f),
                         "power_kw": [round(float(x), 6) for x in C.keyed_rng(key, "power").uniform(*C.POWER_KW, n_m)],
                         "kdraw_v": [round(float(x), 6) for x in C.keyed_rng(key, "kdraw").uniform(0, 1, n_j)],
                         "old_carbon": old.astype(int).tolist()}
        print(f"{split} {size}: {C.COUNTS[split]} instances (seed {seed})", flush=True)
    man = {"purpose": "objective-generation validation DEV instances (results/objective_validation.md)",
           "generator": "daniel/data_utils.SD2_instance_generator, op_per_job=n_m, pt U{1..99}, eligible machines "
                        "U{1..n_m} (mix); np.random.seed(seed) once per (split, size), instances in sequence",
           "seeds": {f"{s}_{z}": v for (s, z), v in C.SEEDS.items()},
           "time_unit_assumption": "1 SD2 time unit = 1 minute",
           "power": "P_m ~ U(4, 15) kW per machine, numpy default_rng(sha256(key|power)), rounded to 1e-6",
           "kdraw": "v_j ~ U(0,1) per job, default_rng(sha256(key|kdraw)); k_j = k*(1 + s*(v_j - 0.5))",
           "old_carbon": "i.i.d. integer U{1..100} per eligible (operation, machine), default_rng(sha256(key|oldcarbon)); "
                         "comparison only",
           "gamma_kg_per_kwh": C.GAMMA, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "instances": recs}
    C.MANIFEST.write_text(json.dumps(man))
    return man


def disjointness(man):
    dev = {r["structural_sha256"]: k for k, r in man["instances"].items()}
    by_split = {}
    for k, r in man["instances"].items():
        by_split.setdefault(r["split"], set()).add(r["structural_sha256"])
    rep = {"dev_internal_duplicates": len(man["instances"]) - len(dev),
           "cal_vs_check_overlap": len(by_split["cal"] & by_split["check"]), "pools": {}}
    for split in ("data_train", "data_validation", "data_train_vali", "data_development", "SD2"):
        root = C.DAN / "data" / split
        files = [f for f in root.rglob("*.fjs") if ("10x5" in f.parent.name or "20x10" in f.parent.name)
                 and "objective_dev" not in str(f) and "final_test" not in str(f)]
        hit = sum(structural_fjsp_sha256(f) in dev for f in files)
        rep["pools"][split] = {"files_hashed": len(files), "overlaps": hit}
    recorded = 0; hit = 0
    for mf in sorted((C.DAN / "data/manifests").glob("*.json")):
        m = json.loads(mf.read_text())
        for entries in m.get("datasets", {}).values():
            for e in entries:
                recorded += 1; hit += e.get("structural_sha256") in dev
    rep["pools"]["recorded split-manifest hashes (incl. the manifest's final_test split)"] = {"hashes": recorded, "overlaps": hit}
    rep["data_final_test"] = ("never opened; generated by generate_data_to_files with seed_final_test_datagen = 400 "
                              "(daniel/params.py:77), whereas DEV uses seeds 20261013-20261016")
    (C.OUT).mkdir(parents=True, exist_ok=True)
    (C.OUT / "disjointness.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    disjointness(generate())
