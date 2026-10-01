"""Audit diagnostic: train/eval processing-time-scale mismatch (results/code_audit.md). EVALUATION ONLY.

Training always builds one environment from 20 instances of the train pool, whose batch maximum processing time is 99
in effectively every batch, so every training instance is scaled by 99. Canonical evaluation scales each HELD-OUT
instance by its OWN maximum (96-99). For the HELD-OUT instances whose maximum is below 99, this script evaluates each
checkpoint both ways: canonical (alone / same-max batch) and "as trained" (batched with one max-99 companion, which
sets the scale to 99 exactly as in training; the companion's result is discarded). No repo code is changed.

    python tools/audit/scale_mismatch.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/audit"))
import audit_common as A                                # noqa: E402

CK = [("C1@1000 s111", "10x5+carbon+priority+s4t_spec_m_k125s060_s111@u1000", "v1", 111),
      ("C1@1000 s222", "10x5+carbon+priority+s4t_spec_m_k125s060_s222@u1000", "v1", 222),
      ("C1@1000 s333", "10x5+carbon+priority+s4t_spec_m_k125s060_s333@u1000", "v1", 333),
      ("C1@1000 s444", "10x5+carbon+priority+s4t_spec_m_k125s060_s444@u1000", "v1", 444),
      ("Step4 scratch@1000 s111", "10x5+carbon+priority+s4t_scratch_k125s060_s111@u1000", "v1", 111),
      ("V2-M@1000 s111", "v2rr/v2rr_V2-M_s111@u1000", "v2", 111),
      ("V2-M@3000 s111", "v2rr/v2rr_V2-M_s111@u3000", "v2", 111),
      ("V2-S@3000 s111", "v2rr/v2rr_V2-S_s111@u3000", "v2", 111),
      ("V2-T@3000 s111", "v2rr/v2rr_V2-T_s111@u3000", "v2", 111)]


def main():
    data, due = A.pool()
    ub = {i: float(data[1][i][data[1][i] > 0].max()) for i in A.HELD}
    low = [i for i in A.HELD if ub[i] < 99]
    comp = next(i for i in A.HELD if ub[i] == 99)
    out = {"n_heldout": len(A.HELD), "n_low_max": len(low), "low_max_values": {str(i): ub[i] for i in low}, "rows": []}
    for label, name, obs, seed in CK:
        pol = A.V.load_policy(name, obs)
        canon = A.V.run(pol, data, due, A.HELD, obs=obs)                         # canonical (grouped by max)
        sc_c = np.array([A.SC.score(seed, i, canon["mk"][k], canon["cb"][k], canon["tard"][k]) for k, i in enumerate(A.HELD)])
        sc_t = sc_c.copy()
        for i in low:
            o = A.rollout(pol, data, due, np.array([i, comp]), obs=obs)          # scale 99, as in training
            sc_t[list(A.HELD).index(i)] = A.SC.score(seed, i, o["mk"][0], o["cb"][0], o["tard"][0])
        k = [list(A.HELD).index(i) for i in low]
        row = dict(checkpoint=label, heldout_canonical=float(sc_c.mean()), heldout_as_trained=float(sc_t.mean()),
                   low_canonical=float(sc_c[k].mean()), low_as_trained=float(sc_t[k].mean()),
                   low_changed=int((sc_c[k] != sc_t[k]).sum()))
        out["rows"].append(row)
        print(f"{label:<26} HELD-OUT canonical {row['heldout_canonical']:.4f}  as-trained {row['heldout_as_trained']:.4f}"
              f"  | {len(low)} low-max: {row['low_canonical']:.4f} -> {row['low_as_trained']:.4f} ({row['low_changed']} changed)", flush=True)
    A.write_result("scale_mismatch", out)


if __name__ == "__main__":
    main()
