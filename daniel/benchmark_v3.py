"""SD2 multi-objective benchmark v3: frozen objective definitions, hashed parameter manifest, verified loading and an
independent schedule scorer (configs/problem_spec_v3.yaml, docs/benchmark_v3.md).

OBJECTIVES (raw units; 1 SD2 time unit = 1 minute)

  makespan        max_j C_j
  carbon          C = gamma * sum over the chosen (o, m) of P_m * p_om / 60          kg CO2e (processing energy only)
                  P_m ~ U(4, 15) kW per machine, gamma = 0.540 kg CO2e / kWh (CEA-FJSP, Wang et al. 2022)
  total tardiness T = sum_j max(0, C_j - d_j),  d_j = k_j * W_j (TWK),  r_j = 0
                  W_j = sum over job j's operations of the mean processing time over eligible machines
                  k_j = k * (1 + s * (v_j - 0.5)),  v_j ~ U(0, 1)  ->  k_j ~ U[k(1 - s/2), k(1 + s/2)]
                  (k, s) = (1.3, 0.8) at 10x5 and (0.9, 0.2) at 20x10 (results/objective_validation.md)

RANDOM PARAMETERS are drawn from numpy default_rng(int(sha256('<structural sha256>|<purpose>')[:16], 16)) with
purpose 'power' (P_m, n_m draws) or 'kdraw' (v_j, n_j draws). Keying on the instance's STRUCTURAL hash (jobs,
eligibility, processing times; data_manifest.structural_fjsp_sha256) makes every parameter a function of the instance
content alone: renaming or moving a file cannot change it.

NORMALISED SCORE (reward v3 and evaluation), preference weights w (goal g: 1/|g| on each objective named by g)
  score_i = w_m * MK_i / MKref_i + w_c * C_i / Cref_i + w_t * T_i / Tref_size
  MKref_i, Cref_i  makespan and carbon of the fixed reference heuristic (MWKR job rule, ECT machine rule, active-style
                   dispatching; tools/objective_validation/schedules.py) on instance i;
  Tref_size        mean tardiness of the same heuristic over the size's TRAIN pool (tardiness can be 0 per instance).

FREEZE AND LOAD: tools/benchmark_v3/build.py writes the instance tree and the manifest ONCE (P_m, v_j, k_j, W_j, d_j,
references, SHA-256 of every file). Training and evaluation READ the manifest; a hash mismatch or a missing instance
raises. The reserved final-test sets are not part of the manifest (docs/benchmark_v3.md).
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from data_utils import sorted_instance_files, text_to_matrix
from data_manifest import structural_fjsp_sha256, _parse_job_line

VERSION = "sd2-mo-v3"
TAG = "bv3"                                    # must appear in every v3 checkpoint's --model_suffix
DATA_ROOT = Path(__file__).resolve().parent / "data"
ROOT = DATA_ROOT / "benchmark_v3"
MANIFEST = ROOT / "manifest_benchmark_v3.json"
SPLITS = ("train", "validation", "development", "evaluation")
SIZES = {"10x5": (10, 5), "20x10": (20, 10)}

TIME_UNIT_HOURS = 1.0 / 60.0                   # 1 SD2 time unit = 1 minute
POWER_KW = (4.0, 15.0)
GAMMA = 0.540
DUE = {"10x5": (1.3, 0.8), "20x10": (0.9, 0.2)}           # (k, s) per size
REFERENCE_RULE = ("MWKR", "ECT")

# Fixed observation scales (no batch dependence): the SD2 generator maximum processing time, the largest possible
# carbon of one operation and the largest machine power.
PT_SCALE = 99.0
CARBON_SCALE = GAMMA * POWER_KW[1] * PT_SCALE * TIME_UNIT_HOURS      # 13.365 kg CO2e
POWER_SCALE = POWER_KW[1]


def keyed_rng(key: str, purpose: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{key}|{purpose}".encode()).hexdigest()[:16], 16))


def machine_power(key: str, n_m: int) -> np.ndarray:
    return keyed_rng(key, "power").uniform(POWER_KW[0], POWER_KW[1], n_m)


def kdraw(key: str, n_j: int) -> np.ndarray:
    return keyed_rng(key, "kdraw").uniform(0.0, 1.0, n_j)


def work_content(job_length, op_pt) -> np.ndarray:
    """W_j: sum over the job's operations of the mean processing time over eligible machines (raw units)."""
    op_pt = np.asarray(op_pt, dtype=np.float64)
    op_mean = np.array([row[row > 0].mean() for row in op_pt])
    bounds = np.concatenate([[0], np.cumsum(job_length)]).astype(int)
    return np.array([op_mean[bounds[j]:bounds[j + 1]].sum() for j in range(len(job_length))])


def due_multipliers(v, size: str) -> np.ndarray:
    k, s = DUE[size]
    return k * (1.0 + s * (np.asarray(v, dtype=np.float64) - 0.5))


def due_dates(job_length, op_pt, v, size: str) -> np.ndarray:
    return due_multipliers(v, size) * work_content(job_length, op_pt)


def carbon_matrix(op_pt, power) -> np.ndarray:
    """carbon[o, m] = gamma * P_m * p_om * (hours per time unit), kg CO2e; 0 where the pair is ineligible."""
    op_pt = np.asarray(op_pt, dtype=np.float64)
    return GAMMA * np.asarray(power, dtype=np.float64)[None, :] * op_pt * TIME_UNIT_HOURS


def size_of(path) -> str:
    return Path(path).parent.name.split("+")[0] if Path(path).suffix == ".fjs" else Path(path).name.split("+")[0]


def instance_key(path) -> str:
    return Path(path).resolve().relative_to(DATA_ROOT.resolve()).as_posix()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_plain_instance(path):
    """(job_length, op_pt) of a PLAIN SD2 file. The retired '+carbon+priority' format is rejected."""
    lines = Path(path).read_text().splitlines()
    for line in lines[1:]:
        if line.strip():
            _parse_job_line([int(float(t)) for t in line.split()], extended=False)   # raises on a non-plain row
    job_length, op_pt = text_to_matrix(lines)
    return job_length, op_pt


def generate_parameters(path, size: str) -> dict:
    """Every generated parameter of one instance (for the freeze and the reproducibility test ONLY)."""
    job_length, op_pt = read_plain_instance(path)
    n_j, n_m = SIZES[size]
    if len(job_length) != n_j or op_pt.shape[1] != n_m:
        raise ValueError(f"{path}: expected {size}, got {len(job_length)} jobs x {op_pt.shape[1]} machines")
    key = structural_fjsp_sha256(path)
    v = kdraw(key, n_j)
    return {"structural_sha256": key, "power_kw": machine_power(key, n_m), "kdraw_v": v,
            "k_j": due_multipliers(v, size), "work_content": work_content(job_length, op_pt),
            "due_dates": due_dates(job_length, op_pt, v, size)}


# ----------------------------------------------------------------------------------------------- independent scorer

def schedule_objectives(job_length, op_pt, carbon, due, machine, start) -> dict:
    """Raw objectives of a COMPLETE schedule (machine and start of every operation), with a feasibility check that
    does not use the environment: eligible machine, start >= 0, job precedence, no overlap on any machine."""
    op_pt = np.asarray(op_pt, dtype=np.float64)
    machine = np.asarray(machine, dtype=int); start = np.asarray(start, dtype=np.float64)
    n = op_pt.shape[0]
    if machine.shape != (n,) or start.shape != (n,):
        raise ValueError("one machine and one start time per operation are required")
    p = op_pt[np.arange(n), machine]
    if np.any(p <= 0):
        raise ValueError("operation scheduled on an ineligible machine")
    if np.any(start < 0):
        raise ValueError("negative start time")
    comp = start + p
    first = np.concatenate([[0], np.cumsum(job_length)[:-1]]).astype(int)
    last = first + np.asarray(job_length, dtype=int) - 1
    for f, l in zip(first, last):
        if np.any(start[f + 1:l + 1] < comp[f:l] - 1e-9):
            raise ValueError("job precedence violated")
    for m in range(op_pt.shape[1]):
        on = np.flatnonzero(machine == m)
        order = on[np.argsort(start[on], kind="stable")]
        if np.any(start[order[1:]] < comp[order[:-1]] - 1e-9):
            raise ValueError(f"overlap on machine {m}")
    cj = comp[last]
    return {"makespan": float(comp.max()), "carbon": float(np.asarray(carbon)[np.arange(n), machine].sum()),
            "tardiness": float(np.maximum(0.0, cj - np.asarray(due)).sum()), "completion": comp}


def goal_weights(goal: str) -> tuple[float, float, float]:
    """Preference weights (w_m, w_c, w_t) of a goal built from m, c, t: 1/|goal| on each named objective."""
    if not goal or set(goal) - set("mct") or len(set(goal)) != len(goal):
        raise ValueError(f"benchmark v3 goals are built from m, c, t; got {goal!r}")
    return tuple((1.0 / len(goal)) if x in goal else 0.0 for x in "mct")


def score(weights, makespan, carbon, tardiness, mk_ref, cb_ref, t_ref):
    w_m, w_c, w_t = weights
    return (w_m * np.asarray(makespan) / mk_ref + w_c * np.asarray(carbon) / cb_ref
            + w_t * np.asarray(tardiness) / t_ref)


# ---------------------------------------------------------------------------------------------------- frozen loading

@dataclass
class BenchmarkData:
    """One split x size of benchmark v3, in loader order (data_utils.sorted_instance_files)."""
    directory: str
    size: str
    keys: list = field(default_factory=list)
    job_length: list = field(default_factory=list)
    op_pt: list = field(default_factory=list)
    power: list = field(default_factory=list)
    carbon: list = field(default_factory=list)
    due: list = field(default_factory=list)
    mk_ref: list = field(default_factory=list)
    cb_ref: list = field(default_factory=list)
    t_ref: float = float("nan")

    def __len__(self):
        return len(self.keys)

    def env_kwargs(self, idx, goal: str) -> dict:
        """Arguments of FJSPEnvForSameOpNums.set_initial_data for the instances `idx` (repeats allowed)."""
        idx = list(idx)
        return dict(job_length_list=[self.job_length[i] for i in idx], op_pt_list=[self.op_pt[i] for i in idx],
                    carbon_list=[self.carbon[i] for i in idx], due_date_list=[self.due[i] for i in idx],
                    machine_power_list=[self.power[i] for i in idx],
                    reward_refs={"goal": goal, "tbar": self.t_ref, "mk": [self.mk_ref[i] for i in idx],
                                 "cb": [self.cb_ref[i] for i in idx]})


def load_manifest(path=None) -> dict:
    path = Path(path or MANIFEST)
    if not path.exists():
        raise FileNotFoundError(f"benchmark v3 manifest not found: {path} (tools/benchmark_v3/build.py)")
    man = json.loads(path.read_text())
    if man.get("version") != VERSION:
        raise ValueError(f"{path} is not a {VERSION} manifest")
    return man


def load_directory(directory, manifest: dict | None = None) -> BenchmarkData:
    """Every instance of one v3 directory with its frozen parameters, hash-checked. Carbon is computed from the frozen
    machine powers by the definition above; due dates and references are read, never re-derived."""
    manifest = manifest or load_manifest()
    files = sorted_instance_files(str(directory))
    if not files:
        raise FileNotFoundError(f"no instances in {directory}")
    size = size_of(directory)
    if size not in manifest["sizes"]:
        raise KeyError(f"size {size} is not in the benchmark v3 manifest")
    out = BenchmarkData(str(directory), size, t_ref=float(manifest["sizes"][size]["t_ref"]))
    for f in files:
        key = instance_key(f)
        rec = manifest["instances"].get(key)
        if rec is None:
            raise KeyError(f"instance {key} is not in the benchmark v3 manifest")
        h = file_sha256(f)
        if h != rec["sha256"]:
            raise ValueError(f"SOURCE HASH MISMATCH for {key}: manifest {rec['sha256'][:12]}..., file {h[:12]}...")
        job_length, op_pt = read_plain_instance(f)
        power = np.asarray(rec["power_kw"], dtype=np.float64)
        out.keys.append(key); out.job_length.append(job_length); out.op_pt.append(op_pt); out.power.append(power)
        out.carbon.append(carbon_matrix(op_pt, power))
        out.due.append(np.asarray(rec["due_dates"], dtype=np.float64))
        out.mk_ref.append(float(rec["reference"]["makespan"])); out.cb_ref.append(float(rec["reference"]["carbon"]))
    return out


def load_split(split: str, size: str, manifest: dict | None = None) -> BenchmarkData:
    if split not in SPLITS:
        raise ValueError(f"unknown benchmark v3 split {split!r}; expected one of {SPLITS}")
    return load_directory(ROOT / split / "SD2" / f"{size}+mix", manifest)
