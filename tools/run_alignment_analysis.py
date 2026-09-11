"""Step 1: decompose specialist incompatibility into symmetry vs objective divergence.

  --mode selftest : align a known permutation of a checkpoint back to it (must recover ~0)
  --mode control  : SAME objective, DIFFERENT seed  (high symmetry share expected)
  --mode cross    : CROSS objective, same and different seed

Symmetry share = 1 - d_after^2 / d_before^2, i.e. the fraction of squared parameter
distance that permutation alignment removes. Reported overall and per layer block.
Run from the repository root.
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "daniel"))

from permutation_symmetry import apply_perm, random_perm, layer_groups, sq_dist  # noqa: E402
from align_specialists import align  # noqa: E402

SEEDS = [111, 222, 333, 444]
GROUP_ORDER = ["op_attn_L0", "op_attn_L1", "mch_attn_L0", "mch_attn_L1", "actor_mlp", "critic_mlp"]


def load_sd(size, tag):
    from checkpointing import unwrap_checkpoint
    p = REPO / "daniel/trained_network/SD2" / f"{size}+carbon+priority+{tag}.pth"
    sd, _ = unwrap_checkpoint(torch.load(p, map_location="cpu", weights_only=False))
    return {k: v.double() for k, v in sd.items()}


def report_pair(A, B, label, iters=30):
    keys = list(A.keys())
    g = layer_groups(A)
    P, hist = align(A, B, iters=iters)
    Bp = apply_perm(B, P)

    d0 = sq_dist(A, B, keys); d1 = sq_dist(A, Bp, keys)
    share = 1 - d1 / d0
    rows = []
    for name in GROUP_ORDER:
        k = g[name]
        a0 = sq_dist(A, B, k); a1 = sq_dist(A, Bp, k)
        rows.append((name, a0, a1, 1 - a1 / a0 if a0 > 0 else float("nan")))
    return dict(label=label, d0=d0, d1=d1, share=share, per_layer=rows, iters=len(hist))


def print_table(res):
    print(f"\n  {res['label']}")
    print(f"    overall: ||A-B||={np.sqrt(res['d0']):8.4f} -> ||A-PB||={np.sqrt(res['d1']):8.4f}"
          f"   SYMMETRY SHARE = {100*res['share']:5.1f}%   ({res['iters']} iters)")
    print(f"    {'block':<14}{'d^2 before':>12}{'d^2 after':>12}{'share':>9}")
    for name, a0, a1, sh in res["per_layer"]:
        print(f"    {name:<14}{a0:12.3f}{a1:12.3f}{100*sh:8.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "control", "cross"], required=True)
    ap.add_argument("--sizes", nargs="+", default=["10x5", "20x10"])
    args = ap.parse_args()

    if args.mode == "selftest":
        print("=" * 78)
        print("ALIGNER SELF-TEST: B is an exact permutation of A, so alignment must recover ~0")
        print("=" * 78)
        rng = np.random.default_rng(3)
        for size in args.sizes:
            A = load_sd(size, "c_s111")
            B = apply_perm(A, random_perm(rng))
            r = report_pair(A, B, f"{size}: c_s111  vs  a random permutation of itself")
            print_table(r)
            print(f"    -> recovery {'OK' if r['share'] > 0.999 else 'INCOMPLETE'} "
                  f"(residual {np.sqrt(r['d1']):.2e})")
        return

    if args.mode == "control":
        print("=" * 78)
        print("CONTROL: SAME objective, DIFFERENT seed (functionally near-equivalent nets)")
        print("=" * 78)
        for size in args.sizes:
            for obj in ["c", "m"]:
                shares = []
                for s1, s2 in itertools.combinations(SEEDS, 2):
                    A = load_sd(size, f"{obj}_s{s1}"); B = load_sd(size, f"{obj}_s{s2}")
                    r = report_pair(A, B, f"{size} {obj}_s{s1} vs {obj}_s{s2}")
                    print_table(r); shares.append(r["share"])
                print(f"\n  >>> {size} objective '{obj}': mean symmetry share "
                      f"{100*np.mean(shares):.1f}%  (sd {100*np.std(shares, ddof=1):.1f}, n={len(shares)})")
        return

    print("=" * 78)
    print("CROSS-OBJECTIVE: carbon vs makespan")
    print("=" * 78)
    for size in args.sizes:
        same, diff = [], []
        for s1 in SEEDS:
            for s2 in SEEDS:
                A = load_sd(size, f"c_s{s1}"); B = load_sd(size, f"m_s{s2}")
                r = report_pair(A, B, f"{size} c_s{s1} vs m_s{s2}"
                                      f"{'  [SAME seed: shared init]' if s1 == s2 else ''}")
                print_table(r)
                (same if s1 == s2 else diff).append(r["share"])
        print(f"\n  >>> {size} cross-objective SAME seed : mean {100*np.mean(same):.1f}% "
              f"(sd {100*np.std(same, ddof=1):.1f}, n={len(same)})")
        print(f"  >>> {size} cross-objective DIFF seed : mean {100*np.mean(diff):.1f}% "
              f"(sd {100*np.std(diff, ddof=1):.1f}, n={len(diff)})")


if __name__ == "__main__":
    main()
