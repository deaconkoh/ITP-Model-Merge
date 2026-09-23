"""STEP 2: select the due-date tightness k against TRAINED-policy job completion times.

Fixed BEFORE looking at the numbers:
  * s = 0.6 (midpoint of the published RDD range; see daniel/due_dates.py for the citation).
  * k grid: 0.50 to 2.00 in steps of 0.05.
  * Reference schedules: the four trained makespan specialists (seeds 111-444), greedy decoding,
    on the 100-instance train_vali pool of each size. Statistics are averaged over the seeds;
    the per-seed range is reported alongside.
  * RULE: the SMALLEST k whose seed-mean tardy fraction is in [0.2, 0.6] AND which leaves at most
    10% of instances with zero tardiness -- at BOTH sizes. If no single k qualifies at both sizes,
    per-size k chosen by the same rule applied to each size separately (pre-approved).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel"))
from data_utils import sorted_instance_files          # noqa: E402
from due_dates import derive, S_DISPERSION, dd_tag    # noqa: E402

SEEDS = [111, 222, 333, 444]
KS = [round(0.50 + 0.05 * i, 2) for i in range(31)]
LO, HI, ZMAX = 0.2, 0.6, 0.10
JC = REPO / "daniel/test_results/SD2/job_completion"


def stats_for(size):
    files = sorted_instance_files(str(REPO / f"daniel/data/data_train_vali/SD2/{size}+carbon+priority"))
    C = {s: np.load(JC / f"{size}+carbon+priority+m_s{s}__{size}+carbon+priority.npy") for s in SEEDS}
    unit = [derive(f, 1.0, S_DISPERSION) for f in files]          # due dates at k=1; scale by k
    out = {}
    for k in KS:
        tf, zero, med = [], [], []
        for s in SEEDS:
            T = np.array([np.maximum(0, C[s][i] - k * unit[i]).sum() for i in range(len(files))])
            F = np.array([(C[s][i] > k * unit[i] + 1e-9).mean() for i in range(len(files))])
            tf.append(F.mean()); zero.append((T <= 1e-9).mean()); med.append(np.median(T))
        out[k] = (np.mean(tf), min(tf), max(tf), np.mean(zero), np.mean(med))
    return out


def main():
    res = {size: stats_for(size) for size in ("10x5", "20x10")}
    print(f"s = {S_DISPERSION}; trained makespan specialists, 4 seeds, 100 instances per size\n")
    print(f"  {'k':>5}  |{'10x5 tardy (seed range)':>28}{'zeroT':>7}{'med T':>8}  |"
          f"{'20x10 tardy (seed range)':>28}{'zeroT':>7}{'med T':>8}")
    ok = {}
    for k in KS:
        row = []
        for size in ("10x5", "20x10"):
            m, lo, hi, z, med = res[size][k]
            q = LO <= m <= HI and z <= ZMAX
            ok.setdefault(size, []).append((k, q))
            row.append(f"{m:>10.3f} [{lo:.3f},{hi:.3f}]{'*' if q else ' '}{z:>6.2f}{med:>8.0f}")
        print(f"  {k:>5.2f}  |{row[0]}  |{row[1]}")
    print("  * = qualifies at that size")
    both = [k for (k, a), (_, b) in zip(ok["10x5"], ok["20x10"]) if a and b]
    print()
    if both:
        k = min(both)
        print(f"SINGLE k QUALIFIES AT BOTH SIZES: k = {k}  -> tag {dd_tag(k, S_DISPERSION)}")
        print(f"CHOSEN k_by_size = {{'10x5': {k}, '20x10': {k}}}")
    else:
        per = {sz: min([k for k, q in ok[sz] if q], default=None) for sz in ok}
        print("NO SINGLE k QUALIFIES AT BOTH SIZES -> per-size k by the same rule (pre-approved)")
        print(f"CHOSEN k_by_size = {per}")


if __name__ == "__main__":
    main()
