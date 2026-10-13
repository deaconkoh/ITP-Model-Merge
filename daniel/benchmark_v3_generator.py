"""Fresh SD2 training instances for benchmark v3: the original DANIEL training scheme (docs/evaluation_protocol_v3.md,
training-data decision). Opt-in with train.py --v3_train_source generator.

Instances come from data_utils.SD2_instance_generator, drawn from the global numpy RNG that train.py seeds with
--seed_train, exactly as in DANIEL (a fresh batch of --num_envs instances every --reset_env_timestep updates). Their
v3 parameters come from the SAME frozen code as the manifest: P_m and v_j from benchmark_v3 (keyed on the structural
hash, which is computed here from the matrices in the same canonical form as data_manifest.structural_fjsp_sha256),
due dates and carbon by the benchmark_v3 definitions, and the score references by the reference heuristic of the
manifest builder (MWKR/ECT, tools/objective_validation/schedules.py); Tref is the size's frozen value.

Disjointness: every fresh instance's structural hash is checked against the v3 validation, development and evaluation
pools (hashes recorded in the manifest) and against the reserved final-test stream regenerated IN MEMORY from its
generator seed (seed_final_test_datagen, 100 instances); the final-test files are never opened. A colliding instance
is discarded and redrawn; the count is reported.
"""
from __future__ import annotations
import hashlib, json, random, sys
from pathlib import Path
import numpy as np

import benchmark_v3 as B
from data_utils import SD2_instance_generator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "objective_validation"))
import common as _OV                                        # noqa: E402  (Instance container only)
import schedules as _S                                      # noqa: E402  (the reference heuristic of the manifest)

FORBIDDEN_SPLITS = ("validation", "development", "evaluation")
FINAL_TEST_COUNT = 100


def structural_sha256(job_length, op_pt) -> str:
    """data_manifest.structural_fjsp_sha256 of the plain file of (job_length, op_pt), from the matrices."""
    jobs, o = [], 0
    for n in job_length:
        ops = []
        for _ in range(int(n)):
            elig = [(int(m) + 1, int(op_pt[o, m])) for m in np.flatnonzero(op_pt[o])]
            ops.append((len(elig), elig)); o += 1
        jobs.append(ops)
    return hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def reference(job_length, op_pt, carbon, due) -> tuple[float, float]:
    jl = np.asarray(job_length)
    inst = _OV.Instance("", jl, np.asarray(op_pt), np.repeat(np.arange(len(jl)), jl),
                        np.concatenate([[0], np.cumsum(jl)[:-1]]))
    machine, start = _S.build(inst, *B.REFERENCE_RULE)
    r = B.schedule_objectives(job_length, op_pt, carbon, due, machine, start)
    return r["makespan"], r["carbon"]


def final_test_stream_hashes(config, seed: int, count: int = FINAL_TEST_COUNT) -> set:
    """Structural hashes of the reserved final-test pool as generated (data_utils.generate_data_to_files with its seed),
    regenerated in memory. The global RNG states are restored afterwards."""
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(seed); random.seed(seed)
        out = set()
        for _ in range(count):
            jl, pt, _ = SD2_instance_generator(config=config)
            out.add(structural_sha256(jl, pt))
        return out
    finally:
        np.random.set_state(np_state); random.setstate(py_state)


class GeneratorSource:
    def __init__(self, config, size: str, manifest: dict):
        if size not in B.SIZES or (config.n_j, config.n_m) != B.SIZES[size]:
            raise ValueError(f"generator size {size} does not match --n_j {config.n_j} --n_m {config.n_m}")
        if config.data_suffix != "mix" or (config.low, config.high) != (1, int(B.PT_SCALE)) or config.op_per_job != 0:
            raise ValueError("benchmark v3 instances are SD2 'mix' with processing times 1..99 and n_m operations per job")
        self.config, self.size = config, size
        self.t_ref = float(manifest["sizes"][size]["t_ref"])
        self.forbidden = {r["structural_sha256"] for r in manifest["instances"].values()
                          if r["size"] == size and r["split"] in FORBIDDEN_SPLITS}
        self.n_pool_hashes = len(self.forbidden)
        self.forbidden |= final_test_stream_hashes(config, config.seed_final_test_datagen)
        self.generated = 0
        self.collisions = 0

    def sample(self, n: int) -> B.BenchmarkData:
        out = B.BenchmarkData("<SD2 generator>", self.size, t_ref=self.t_ref)
        while len(out) < n:
            jl, pt, _ = SD2_instance_generator(config=self.config)
            self.generated += 1
            key = structural_sha256(jl, pt)
            if key in self.forbidden:
                self.collisions += 1
                continue
            v = B.kdraw(key, len(jl)); power = B.machine_power(key, pt.shape[1])
            carbon = B.carbon_matrix(pt, power); due = B.due_dates(jl, pt, v, self.size)
            mk_ref, cb_ref = reference(jl, pt, carbon, due)
            out.keys.append(key); out.job_length.append(jl); out.op_pt.append(pt); out.power.append(power)
            out.carbon.append(carbon); out.due.append(due); out.mk_ref.append(mk_ref); out.cb_ref.append(cb_ref)
        return out

    def provenance(self) -> dict:
        return {"source": "SD2 generator (data_utils.SD2_instance_generator, global numpy RNG seeded by --seed_train)",
                "generated": self.generated, "collisions_discarded": self.collisions,
                "forbidden_hashes": {"v3 validation/development/evaluation": self.n_pool_hashes,
                                     "final-test stream regenerated in memory (seed "
                                     f"{self.config.seed_final_test_datagen}, {FINAL_TEST_COUNT} instances)":
                                         len(self.forbidden) - self.n_pool_hashes}}
