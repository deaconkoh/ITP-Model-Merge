"""Part B3 of results/v2_fixes.md: every converted checkpoint must behave exactly like its v1 source.

For each (checkpoint, seed): v1 checkpoint in the v1 env and v2 checkpoint in the v2 env, lock-step on the 49 HELD-OUT
instances. Pass = identical greedy action at every step, max |action-probability difference| < 1e-5 and per-instance
|Step 4 score difference| < 1e-6. The v1 side is also checked against the Step 4 result files (exact).
Writes results/v2_fixes/checkpoint_verification.json; exit 1 on any failure (STOP 1).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import v2_common as V                                  # noqa: E402
import ceiling_scorer as SC                            # noqa: E402

OUT = REPO / "results/v2_fixes"
S4 = REPO / "daniel/test_results/SD2/trainvali-k125s060_10x5+carbon+priority"
KINDS = {"makespan specialist": "m_s{s}", "carbon specialist": "c_s{s}", "tardiness specialist": "t_k125s060_s{s}",
         "C1@1000": "s4t_spec_m_k125s060_s{s}@u1000"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = V.P.load_data("data_train_vali"); due = SC.due_dates()
    rows, ok_all = [], True
    for kind, pat in KINDS.items():
        for s in SC.SEEDS:
            stem = pat.format(s=s); name = f"10x5+carbon+priority+{stem}"
            p1, p2 = V.load_policy(name, "v1"), V.load_policy(name + "_obsv2", "v2")
            o1, o2, n_diff, n_steps, max_dp = V.lockstep(p1, p2, data, due, SC.HELD)
            ref = SC.reference(s)
            sc = lambda o: np.array([SC.score(s, i, o["mk"][k], o["cb"][k], o["tard"][k], ref=ref)
                                     for k, i in enumerate(SC.HELD)])
            d_sc = float(np.abs(sc(o1) - sc(o2)).max())
            step4 = np.load(S4 / f"Result_DANIELG+{name}_trainvali-k125s060_10x5+carbon+priority.npy")[:, [0, 1, 3]][SC.HELD]
            v1_exact = bool(np.array_equal(np.stack([o1["mk"], o1["cb"], o1["tard"]], 1), step4))
            ok = n_diff == 0 and max_dp < 1e-5 and d_sc < 1e-6 and v1_exact
            ok_all &= ok
            rows.append(dict(kind=kind, seed=s, checkpoint=name + "_obsv2", steps=n_steps, differing_actions=n_diff,
                             max_abs_prob_diff=max_dp, max_abs_score_diff=d_sc, v1_side_equals_step4=v1_exact,
                             pooled_score_v1=float(sc(o1).mean()), pooled_score_v2=float(sc(o2).mean()), passed=ok))
            print(f"{kind:<22} s{s}: steps {n_steps}, differing actions {n_diff}, max|dp| {max_dp:.2e}, "
                  f"max|dscore| {d_sc:.2e}, v1==Step4 {v1_exact} -> {'PASS' if ok else 'FAIL'}", flush=True)
    (OUT / "checkpoint_verification.json").write_text(json.dumps({"pass": ok_all, "rows": rows}, indent=1))
    print("B3", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
