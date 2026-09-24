"""STEP A: select each seed's merge-init composition on the VALIDATION split, not on the pool the
arms are scored on.

Rule (fixed before looking): for each seed, the grid composition (66 points, step 0.1; the centroid
is the reference, not a candidate) with the lowest mean scalarised objective at (1/3,1/3,1/3) on the
10x5 validation split. Scalarisation as everywhere else in the tardiness programme: makespan and
carbon per instance relative to that seed's centroid composition ON THE SAME SPLIT, tardiness
relative to the centroid's pool mean. No rule switching if the pick looks unlucky.

Candidate set. The literal rule (all 66 grid points) picked a VERTEX -- a single specialist, not a
merge -- at seeds 333 (100/0/0 = makespan specialist) and 444 (0/0/100 = tardiness specialist). Arm
(a) is "fine-tune from a MERGE"; a vertex would make it the same checkpoint and seed as arm (c) or
(e), so the merge-vs-specialist comparison would be zero by construction. The arms therefore use the
MERGE-ONLY candidate set: the 63 grid points with at least two specialists at non-zero weight. This
is a definitional restriction (a merge must combine specialists), not a rule switch after an unlucky
draw; the literal picks are reported alongside, because "on validation, no merge beat the best single
specialist" is itself evidence about composition.

Writes results/step4/merge_init_selection.json (read by tools/run_arms_t.py).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_simplex import load_all          # noqa: E402
from preference3 import Preference, scalarise  # noqa: E402
from simplex_merge import name_for             # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TAG, C = "k125s060", (333, 333, 334)
SEEDS = [111, 222, 333, 444]


def objectives(data, s, pref):
    ref = data[C][s]
    return {p: scalarise(data[p][s], ref, pref, True) for p in data}


def f(p):
    return "/".join(str(x // 10) for x in p)


def pick_table(vali, pool, cands, pref, label):
    print(f"--- candidate set: {label} ({len(cands)} compositions)")
    print(f"  {'seed':<6}{'VALIDATION pick':<18}{'pool optimum':<16}{'same?':<7}"
          f"{'pool gap':>10}{'pick rank on pool':>19}{'indistinguishable on vali':>28}")
    sel, gaps = {}, []
    for s in SEEDS:
        ov, op = objectives(vali, s, pref), objectives(pool, s, pref)
        mv = {p: float(ov[p].mean()) for p in cands}
        mp = {p: float(op[p].mean()) for p in cands}
        pick, best = min(cands, key=mv.get), min(cands, key=mp.get)
        gap = 100 * (mp[pick] - mp[best]) / mp[best]
        rank = sorted(cands, key=mp.get).index(pick) + 1
        tied = sum(1 for p in cands if p != pick and stats.ttest_rel(ov[p], ov[pick]).pvalue > 0.05)
        gaps.append(gap)
        print(f"  {s:<6}{f(pick):<18}{f(best):<16}{'yes' if pick == best else 'NO':<7}"
              f"{gap:>+9.2f}%{rank:>12} of {len(cands)}{tied:>20} of {len(cands) - 1}")
        sel[str(s)] = {"composition_permille": list(pick), "checkpoint": name_for("10x5", pick, s, "t", TAG),
                       "validation_objective": mv[pick], "pool_objective": mp[pick],
                       "pool_optimum_permille": list(best), "pool_optimum_objective": mp[best],
                       "pool_gap_percent": gap, "pick_rank_on_pool": rank,
                       "validation_ties_p_gt_0.05": tied}
    print(f"  mean evaluation-pool gap from choosing on validation: {np.mean(gaps):+.2f}%\n")
    return sel


def main():
    pref = Preference(1 / 3, 1 / 3, 1 / 3, third="t")
    vali = load_all("10x5", f"vali-{TAG}", "t", TAG)
    pool = load_all("10x5", f"trainvali-{TAG}", "t", TAG)
    grid = [p for p in sorted(pool) if p != C]
    assert len(grid) == 66 and all(p in vali for p in grid) and C in vali, "validation evaluations incomplete"
    n_vali = next(iter(vali[C].values())).shape[0]
    print(f"validation split: {n_vali} instances per seed; evaluation pool: 100\n")
    literal = pick_table(vali, pool, grid, pref, "LITERAL rule, all 66 grid points")
    merges = [p for p in grid if sum(x > 0 for x in p) >= 2]
    sel = pick_table(vali, pool, merges, pref, "MERGE-ONLY, >= 2 specialists (used for arm a)")
    out = REPO / "results/step4/merge_init_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rule": __doc__.split("Rule (fixed before looking): ")[1].split("\n\nWrites")[0],
                               "preference": "1/3,1/3,1/3", "due_date_tag": TAG,
                               "n_validation_instances": n_vali, "candidate_set_used": "merge-only (63)",
                               "literal_rule_picks": literal, "per_seed": sel, **sel}, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
