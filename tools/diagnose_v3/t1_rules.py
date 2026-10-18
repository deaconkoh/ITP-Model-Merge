"""Test 1: dispatching-rule baselines on the evaluation split, both sizes (all three objectives each).

Rules (tools/objective_validation/schedules.build, unrestricted by DANIEL's action mask):
  MWKR+ECT  the manifest reference rule (context)
  EDD+ECT   earliest due date job, earliest-completion machine
  MWKR+MINE / EDD+MINE  minimum-energy machine (ties: earliest completion)
plus the per-operation minimum-energy assignment lower bound (= the CP-SAT carbon optimum).
Every schedule is re-scored by benchmark_v3.schedule_objectives.

    python tools/diagnose_v3/t1_rules.py
"""
import numpy as np
import dv3
from dv3 import B, S

RULES = [("MWKR", "ECT"), ("EDD", "ECT"), ("MWKR", "MINE"), ("EDD", "MINE")]


def main():
    out = {}
    for size in dv3.SIZES:
        data = B.load_split("evaluation", size)
        rec = {}
        for jr, mr in RULES:
            mk, cb, td, share = [], [], [], []
            for i in range(len(data)):
                inst = dv3.ov_instance(data, i)
                m, s = S.build(inst, jr, mr, energy=data.carbon[i], due=data.due[i])
                r = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i], m, s)
                cj = r["completion"][inst.first + inst.job_length - 1]
                mk.append(r["makespan"]); cb.append(r["carbon"]); td.append(r["tardiness"])
                share.append(float(np.mean(cj > data.due[i])))
            rec[f"{jr}+{mr}"] = {**dv3.ratios(data, range(len(data)), mk, cb, td), "tardy_share": float(np.mean(share))}
        lb = [np.where(data.op_pt[i] > 0, data.carbon[i], np.inf).min(1).sum() for i in range(len(data))]
        rec["min-energy assignment (carbon LB)"] = {"carbon": float(np.mean(lb)),
                                                    "carbon_ratio": float(np.mean(np.array(lb) / data.cb_ref))}
        out[size] = rec
        for k, v in rec.items():
            print(size, f"{k:34s}", {a: round(b, 3) for a, b in v.items()})
    dv3.save_json("t1_rules.json", out)


if __name__ == "__main__":
    main()
