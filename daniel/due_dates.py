"""Per-job due dates for the total-tardiness objective: generation, freeze, and verified loading.

GENERATION (TWK rule with the range-of-due-dates convention)

    d_j = k_j * P_j ,   k_j = k * u_j ,   u_j ~ U[1 - s/2, 1 + s/2] ,   r_j = 0

  P_j  work content: sum over job j's operations of the mean processing time over ELIGIBLE
       machines (raw time units).
  s    dispersion = 0.6, the midpoint of the relative-range-of-due-dates values
       RDD in {0.2, 0.4, 0.6, 0.8, 1.0} used to generate the OR-Library weighted-tardiness
       benchmarks: due dates ~ U[P(1-TF-RDD/2), P(1-TF+RDD/2)], Crauwels, Potts & Van Wassenhove
       (1998), INFORMS Journal on Computing 10, 341-350
       (http://people.brunel.ac.uk/~mastjjb/jeb/orlib/wtinfo.html). There the range is RDD times
       the processing-time scale P; here it is s times each job's own due-date scale k*P_j.
  k    tightness, selected by a pre-stated rule against TRAINED-policy completion times.
  r_j  release times are zero: SD2 instances are static, every job available at time zero.

  u_j is drawn from an RNG seeded by the instance's path relative to the data root, so the
  draws do not depend on k and a k sweep rescales identical relative tightnesses.

FREEZE AND LOAD

After k is chosen the due dates are written ONCE to a manifest covering every split at both
sizes, including the reserved final-test sets, together with a SHA-256 of each source instance
file. From then on training and evaluation READ the manifest; nothing re-derives due dates.
A source-hash mismatch, a missing instance or a missing size raises immediately.
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np

from data_utils import sorted_instance_files, text_to_matrix_with_priority_carbon

DATA_ROOT = Path(__file__).resolve().parent / "data"
MANIFEST = DATA_ROOT / "due_dates" / "due_date_manifest.json"
S_DISPERSION = 0.6
SPLITS = ("data_train", "data_validation", "data_train_vali", "data_final_test")
FORMULA = "d_j = k * u_j * P_j, u_j ~ U[1 - s/2, 1 + s/2], P_j = sum_ops mean eligible pt, r_j = 0"
SOURCE = ("Crauwels, Potts & Van Wassenhove (1998), 'Local search heuristics for the single "
          "machine total weighted tardiness scheduling problem', INFORMS Journal on Computing 10, "
          "341-350; OR-Library wtinfo: RDD in {0.2,0.4,0.6,0.8,1.0}, s = midpoint 0.6")


def dd_tag(k: float, s: float) -> str:
    """Encodes k and s in checkpoint and evaluation names, e.g. k100s060."""
    return f"k{round(k * 100):03d}s{round(s * 100):03d}"


def instance_key(path) -> str:
    return Path(path).resolve().relative_to(DATA_ROOT.resolve()).as_posix()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def work_content(job_length, op_pt) -> np.ndarray:
    op_pt = np.asarray(op_pt, dtype=np.float64)
    op_mean = np.array([row[row > 0].mean() for row in op_pt])
    bounds = np.concatenate([[0], np.cumsum(job_length)]).astype(int)
    return np.array([op_mean[bounds[j]:bounds[j + 1]].sum() for j in range(len(job_length))])


def relative_draws(key: str, n_jobs: int, s: float) -> np.ndarray:
    seed = int(hashlib.sha256(f"{key}|s={s:.4f}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).uniform(1 - s / 2, 1 + s / 2, size=n_jobs)


def derive(path, k: float, s: float = S_DISPERSION) -> np.ndarray:
    """Derive due dates for one instance file. For the k selection and the freeze ONLY."""
    with open(path) as fh:
        job_length, op_pt, _, _ = text_to_matrix_with_priority_carbon(fh.readlines())
    P = work_content(job_length, op_pt)
    return k * relative_draws(instance_key(path), len(P), s) * P


def size_of(directory) -> str:
    return Path(directory).name.split("+")[0]


def build_manifest(k_by_size: dict, s: float = S_DISPERSION, rule: str = "", notes: str = ""):
    inst = {}
    for size, k in k_by_size.items():
        for split in SPLITS:
            d = DATA_ROOT / split / "SD2" / f"{size}+carbon+priority"
            files = sorted_instance_files(str(d))
            if not files:
                raise FileNotFoundError(f"no instances in {d}")
            for f in files:
                inst[instance_key(f)] = {"sha256": file_sha256(f),
                                         "due_dates": [round(float(x), 6) for x in derive(f, k, s)]}
    return {
        "k_by_size": {sz: float(k) for sz, k in k_by_size.items()},
        "s": s,
        "dd_tag_by_size": {sz: dd_tag(k, s) for sz, k in k_by_size.items()},
        "formula": FORMULA,
        "release_times": "r_j = 0 for every job (static instances)",
        "dispersion_source": SOURCE,
        "k_selection_rule": rule,
        "notes": notes,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_instances": len(inst),
        "instances": inst,
    }


def load_manifest(path=MANIFEST) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"due-date manifest not found: {path} -- due dates are never "
                                "re-derived at train/eval time; freeze the manifest first")
    return json.loads(path.read_text())


def due_dates_for_directory(directory, manifest=None, path=MANIFEST) -> list[np.ndarray]:
    """Frozen due dates for every instance in `directory`, in loader order. Verifies hashes."""
    manifest = manifest or load_manifest(path)
    out = []
    files = sorted_instance_files(str(directory))
    if not files:
        raise FileNotFoundError(f"no instances in {directory}")
    for f in files:
        key = instance_key(f)
        entry = manifest["instances"].get(key)
        if entry is None:
            raise KeyError(f"instance {key} is not in the due-date manifest")
        h = file_sha256(f)
        if h != entry["sha256"]:
            raise ValueError(f"SOURCE HASH MISMATCH for {key}: manifest {entry['sha256'][:12]}..., "
                             f"file {h[:12]}... -- the instance changed after due dates were frozen")
        out.append(np.asarray(entry["due_dates"], dtype=np.float64))
    return out


def manifest_tag(directory, manifest=None, path=MANIFEST) -> str:
    manifest = manifest or load_manifest(path)
    size = size_of(directory)
    if size not in manifest["dd_tag_by_size"]:
        raise KeyError(f"size {size} not in the due-date manifest")
    return manifest["dd_tag_by_size"][size]
