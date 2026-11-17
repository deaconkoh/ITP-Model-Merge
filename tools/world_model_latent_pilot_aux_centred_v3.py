"""Latent next-state pilot + CENTRED machine-availability auxiliary loss: the one change from
world_model_latent_pilot_aux_v3 is the auxiliary label, (free time - mean free time over the machines of the same next
state) / 15.44 (the same fixed scale). Everything else (data, split, seed-111 start, predictor, weight 1.0, 30 epochs,
evaluation, availability-difference probe, specialist load check) is the aux_v3 code unchanged.

    python tools/world_model_latent_pilot_aux_centred_v3.py   # -> results|checkpoints/world_model_latent_pilot_aux_centred_v3
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import world_model_latent_pilot_aux_v3 as A                         # noqa: E402

NAME = "world_model_latent_pilot_aux_centred_v3"
ABS_RES = A.L.RES                                                    # the absolute-availability variant
A.LABEL = "centred"
A.L.RES = REPO / "results" / NAME
A.L.CK = REPO / "checkpoints" / NAME
A.AP.OUT = A.L.RES / "availability_difference_probe.json"


def main():
    A.main()
    s = json.loads((A.L.RES / "settings.json").read_text())
    assert abs(s["aux"]["scale"] - 15.444444444444446) < 1e-9 and s["aux"]["weight"] == 1.0
    rows = {}
    for name, res in (("original", A.BASE_RES), ("absolute_aux", ABS_RES), ("centred_aux", A.L.RES)):
        h = json.loads((res / "heldout.json").read_text())
        p = json.loads((res / "availability_difference_probe.json").read_text())
        rows[name] = {"probe_sign_accuracy_nonzero": p["readouts"]["encoded"]["sign_accuracy_nonzero"],
                      "probe_mae_nonzero": p["readouts"]["encoded"]["mae_nonzero"],
                      "probe_pearson_nonzero": p["readouts"]["encoded"]["pearson_nonzero"],
                      "probe_nonzero_comparisons": p["heldout_nonzero_difference"],
                      "raw_sign_accuracy_nonzero": p["readouts"]["raw"]["sign_accuracy_nonzero"],
                      "raw_mae_nonzero": p["readouts"]["raw"]["mae_nonzero"],
                      "zero_mae_nonzero": p["baseline_zero"]["mae_nonzero"],
                      "same_op_discrimination": h["discrimination"]["same_operation_other_machine"]["accuracy"],
                      "same_op_pairs": h["discrimination"]["same_operation_other_machine"]["unordered_pairs"],
                      "median_pred_branch_distance_same_op": h["discrimination"]["same_operation_other_machine"]["median_pred_branch_distance"],
                      "median_true_branch_distance_same_op": h["discrimination"]["same_operation_other_machine"]["median_true_branch_distance"],
                      "discrimination_overall": h["discrimination"]["overall"]["accuracy"],
                      "change_explained_changed_tokens": h["errors"]["changed_tokens"]["change_explained"],
                      "responds_reliably": h["decision"]["responds_reliably"]}
    (A.L.RES / "comparison.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
