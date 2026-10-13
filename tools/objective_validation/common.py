"""Shared definitions for the objective-generation validation (results/objective_validation.md).

Everything here is new code; no existing training / environment / evaluation path is touched. The reserved
final-test sets are never opened; the 49 HELD-OUT instances are never used (only hashed for the disjointness check).
"""
from __future__ import annotations
import hashlib, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DAN = REPO / "daniel"
DEV_ROOT = DAN / "data/objective_dev"
OUT = REPO / "results/objective_validation"
MANIFEST = DEV_ROOT / "manifest_objective_dev.json"

SIZES = {"10x5": (10, 5), "20x10": (20, 10)}
SEEDS = {("cal", "10x5"): 20261013, ("cal", "20x10"): 20261014,
         ("check", "10x5"): 20261015, ("check", "20x10"): 20261016}
COUNTS = {"cal": 100, "check": 50}

TIME_UNIT_MINUTES = 1.0                       # assumption: 1 SD2 time unit = 1 minute
POWER_KW = (4.0, 15.0)                        # P_m ~ U(4, 15) kW per machine (CEA-FJSP)
GAMMA = 0.540                                 # kg CO2e per kWh (CEA-FJSP)
K_GRID = [round(0.6 + 0.1 * i, 1) for i in range(15)]      # 0.6 .. 2.0
S_GRID = [0.2, 0.4, 0.6, 0.8]


def keyed_rng(key: str, purpose: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{key}|{purpose}".encode()).hexdigest()[:16], 16))


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instance_dir(split: str, size: str) -> Path:
    return DEV_ROOT / split / "SD2" / f"{size}+mix"


def instance_key(path) -> str:
    return Path(path).resolve().relative_to(DAN / "data").as_posix()


@dataclass
class Instance:
    key: str
    job_length: np.ndarray          # [J]
    pt: np.ndarray                  # [N, M] raw processing times, 0 = ineligible
    job_of: np.ndarray              # [N] job index of each operation
    first: np.ndarray               # [J] first operation index of each job

    @property
    def n_j(self):
        return len(self.job_length)

    @property
    def n_m(self):
        return self.pt.shape[1]

    @property
    def n_ops(self):
        return self.pt.shape[0]

    def work_content(self) -> np.ndarray:
        """TWK W_j = sum over the job's operations of the MEAN processing time over eligible machines."""
        mean = np.array([row[row > 0].mean() for row in self.pt])
        return np.array([mean[self.job_of == j].sum() for j in range(self.n_j)])


def parse_fjs(path) -> Instance:
    """Standard SD2 text format (own parser)."""
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    n_j, n_m = (int(float(x)) for x in lines[0].split()[:2])
    rows, jl = [], []
    for j in range(n_j):
        v = [int(x) for x in lines[j + 1].split()]
        n_ops, k = v[0], 1
        jl.append(n_ops)
        for _ in range(n_ops):
            cnt = v[k]; row = np.zeros(n_m, dtype=np.int64)
            for q in range(cnt):
                row[v[k + 1 + 2 * q] - 1] = v[k + 2 + 2 * q]
            k += 1 + 2 * cnt
            rows.append(row)
    jl = np.array(jl)
    job_of = np.repeat(np.arange(n_j), jl)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    return Instance(instance_key(path), jl, np.array(rows), job_of, first)


def structural_hash(job_length, pt) -> str:
    """Same canonical form as daniel/data_manifest.structural_fjsp_sha256 (jobs, eligibility, times; annotations
    ignored), computed from matrices."""
    import json
    jobs, o = [], 0
    for n in job_length:
        ops = []
        for _ in range(int(n)):
            elig = [(int(m) + 1, int(pt[o, m])) for m in np.flatnonzero(pt[o])]
            ops.append((len(elig), elig)); o += 1
        jobs.append(ops)
    # json.dumps turns tuples into lists, matching the reference implementation
    return hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def instances(split: str, size: str) -> list[Instance]:
    return [parse_fjs(f) for f in sorted(instance_dir(split, size).glob("*.fjs"))]


def load_manifest() -> dict:
    import json
    return json.loads(MANIFEST.read_text())


def params_for(inst: Instance, man: dict) -> dict:
    """Stored per-instance parameters (power, v_j, old carbon), verified against the file hash."""
    rec = man["instances"][inst.key]
    return {"power": np.array(rec["power_kw"]), "v": np.array(rec["kdraw_v"]),
            "oldcarbon": np.array(rec["old_carbon"], dtype=float)}


def energy_carbon(inst: Instance, power: np.ndarray) -> np.ndarray:
    """carbon_om = gamma * P_m * p_om * (time unit in hours); 0 where ineligible. kg CO2e."""
    return GAMMA * power[None, :] * inst.pt * (TIME_UNIT_MINUTES / 60.0)


def due_dates(inst: Instance, v: np.ndarray, k: float, s: float) -> np.ndarray:
    """d_j = k_j * W_j with k_j = k * (1 + s * (v_j - 0.5)), i.e. k_j ~ U[k(1-s/2), k(1+s/2)]."""
    return k * (1.0 + s * (v - 0.5)) * inst.work_content()
