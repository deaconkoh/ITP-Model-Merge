"""Can the frozen target encoder preserve the change in machine availability between two machine choices for the
same operation? One diagnostic on the latent next-state pilot's saved data and frozen checkpoints.

Pairs: same state, same job, different machine (the pilot's evaluation rule, including its exclusion of pairs whose
next-state encodings are identical), on the training factories (fit) and the held-out factories (report). Every pair
gives one comparison per machine m (the same machine aligned across the two branches):
    target  = next available time of m in branch i - in branch j            (simulator, p1_mch_free_time)
    encoded = target-encoder token of m in branch i's next state - in branch j's
    raw     = raw next-state machine features of m in branch i - in branch j  (the encoder's own input, n_fea_m)
Two ordinary least-squares linear readouts with an intercept, the same rows and procedure. No action, machine id or
target value as an extra input. Only these two readouts are fitted; the encoder and all checkpoints are read only.

    python tools/world_model_latent_pilot_availability_probe_v3.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.argv = sys.argv[:1] + ["none"]
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import world_model_latent_pilot_v3 as L                             # noqa: E402
from model.main_model import DualAttentionNetwork                  # noqa: E402

OUT = L.RES / "availability_difference_probe.json"
TOL = 1e-9


def main():
    t0 = time.time()
    cfg, dev = L.setup()
    Dt = L.load_data()
    J, M = L.B.SIZES[L.W.SIZE]
    N = Dt["s_fea_j"].shape[1]
    tgt = DualAttentionNetwork(L.configs).to(dev)
    tgt.load_state_dict(torch.load(L.CK / "target_encoder.pth", map_location=dev)); tgt.eval()
    st, a = Dt["tr_state"], Dt["tr_action"]
    out = {}
    split = dict(zip(("train", "heldout"), L.split_rows(Dt)))
    for name, rows in split.items():
        Z = []
        with torch.no_grad():
            for b0 in range(0, len(rows), 512):
                po, pm, _ = L.targets(tgt, Dt, rows[b0:b0 + 512], dev)
                Z.append(torch.cat([po, pm], 1).cpu().numpy())
        Z = np.concatenate(Z)
        pos = {g: k for k, g in enumerate(rows)}
        Xe, Xr, Y, chosen = [], [], [], []
        for s in np.unique(st[rows]):
            k = rows[st[rows] == s]
            for x in range(len(k)):
                for y in range(x + 1, len(k)):
                    i, j = k[x], k[y]
                    if a[i] // M != a[j] // M:
                        continue
                    zi, zj = Z[pos[i]], Z[pos[j]]
                    if np.sqrt(((zi - zj) ** 2).mean()) <= L.C["eval_tol"]:
                        continue
                    for m in range(M):
                        Xe.append(zi[N + m] - zj[N + m]); Xr.append(Dt["n_fea_m"][i, m] - Dt["n_fea_m"][j, m])
                        Y.append(Dt["p1_mch_free_time"][i, m] - Dt["p1_mch_free_time"][j, m])
                        chosen.append(m in (a[i] % M, a[j] % M))
        out[name] = {"Xe": np.array(Xe), "Xr": np.array(Xr), "Y": np.array(Y), "chosen": np.array(chosen)}
    tr, ho = out["train"], out["heldout"]
    Y = ho["Y"]; nz = np.abs(Y) > TOL
    res = {"pairs": {k: int(len(v["Y"]) // M) for k, v in out.items()},
           "machine_comparisons": {k: int(len(v["Y"])) for k, v in out.items()},
           "heldout_zero_difference": int((~nz).sum()), "heldout_nonzero_difference": int(nz.sum()),
           "heldout_nonzero_on_the_two_chosen_machines": int((nz & ho["chosen"]).sum()),
           "heldout_nonzero_on_other_machines": int((nz & ~ho["chosen"]).sum()),
           "train_nonzero_difference": int((np.abs(tr["Y"]) > TOL).sum()),
           "baseline_zero": {"mae_all": float(np.abs(Y).mean()), "mae_nonzero": float(np.abs(Y[nz]).mean()),
                             "mae_zero": 0.0}, "readouts": {}}
    for key, label in (("Xe", "encoded"), ("Xr", "raw")):
        A = np.c_[tr[key], np.ones(len(tr[key]))]
        w, *_ = np.linalg.lstsq(A, tr["Y"], rcond=None)
        P = np.c_[ho[key], np.ones(len(ho[key]))] @ w
        Ptr = A @ w
        res["readouts"][label] = {
            "inputs": int(tr[key].shape[1]),
            "train_r2": float(1 - ((Ptr - tr["Y"]) ** 2).sum() / ((tr["Y"] - tr["Y"].mean()) ** 2).sum()),
            "mae_all": float(np.abs(P - Y).mean()), "mae_nonzero": float(np.abs(P - Y)[nz].mean()),
            "mae_zero": float(np.abs(P - Y)[~nz].mean()),
            "pearson_all": float(np.corrcoef(P, Y)[0, 1]), "pearson_nonzero": float(np.corrcoef(P[nz], Y[nz])[0, 1]),
            "sign_accuracy_nonzero": float((np.sign(P[nz]) == np.sign(Y[nz])).mean())}
    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
