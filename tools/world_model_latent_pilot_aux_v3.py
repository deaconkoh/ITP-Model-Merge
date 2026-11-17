"""Latent next-state pilot + machine-availability auxiliary loss (one controlled change; 10x5, waiting allowed).

Identical to tools/world_model_latent_pilot_v3.py (same saved collection, factory split, seed-111 start trunk,
predictor, EMA target, variance guard, 30 epochs, batch order) except one added loss: the ACTUAL simulator next state
is passed through the ONLINE encoder, and a shared linear head (8 -> 1, pretraining only) predicts each machine's next
available time from its 8-value machine token alone. Label: the simulator's mch_free_time divided by a fixed scale
(the largest next free time among the training transitions, recorded before training). Loss: MSE, weight AUX_WEIGHT,
backpropagated into the online encoder together with the latent prediction loss.

Then, on the same held-out factories: the pilot's evaluation (unchanged code), the availability-difference probe
(fresh linear readout fitted on the training factories) and the same-operation discrimination test. The original
pilot's data, checkpoints and results are only read; outputs go to *_aux_v3 folders.

    python tools/world_model_latent_pilot_aux_v3.py
"""
from __future__ import annotations
import copy, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.argv = sys.argv[:1] + ["none"]
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import world_model_latent_pilot_v3 as L                             # noqa: E402
import world_model_latent_pilot_availability_probe_v3 as AP         # noqa: E402
from common_utils import setup_seed                                 # noqa: E402
from model.main_model import DANIEL, DualAttentionNetwork          # noqa: E402

BASE_RES, BASE_CK = L.RES, L.CK
NAME = "world_model_latent_pilot_aux_v3"
L.RES = REPO / "results" / NAME
L.CK = REPO / "checkpoints" / NAME
AP.OUT = L.RES / "availability_difference_probe.json"
AUX_WEIGHT = 1.0
C = L.C


def settings(Dt, tr_rows):
    f = L.RES / "settings.json"
    if f.exists():
        return json.loads(f.read_text())
    base = json.loads((BASE_RES / "settings.json").read_text())
    s = {"baseline": {"results": str(BASE_RES.relative_to(REPO)), "checkpoints": str(BASE_CK.relative_to(REPO)),
                      "collection_sha256": L.sha(L.DATA / "collection.npz")},
         "constants": C, "constants_identical_to_baseline": base["constants"] == C,
         "aux": {"weight": AUX_WEIGHT, "head": "shared Linear(8, 1), pretraining only",
                 "input": "online-encoder machine token of the actual simulator next state",
                 "label": "p1_mch_free_time / scale",
                 "scale": float(np.abs(Dt["p1_mch_free_time"][tr_rows]).max()),
                 "scale_rule": "largest next free time among the training transitions"},
         "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    assert s["constants_identical_to_baseline"]
    L.RES.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(s, indent=1) + "\n")
    return s


def train(s):
    cfg, dev = L.setup()
    Dt = L.load_data()
    tr_rows, _ = L.split_rows(Dt)
    base_meta = json.loads((BASE_RES / "train_log.json").read_text())
    assert base_meta["train_transitions"] == len(tr_rows)
    setup_seed(C["train_seed"])
    trunk = DualAttentionNetwork(L.configs).to(dev)
    sd = torch.load(L.START, map_location=dev, weights_only=False)["state_dict"]
    trunk.load_state_dict({k[len("feature_exact."):]: v for k, v in sd.items() if k.startswith("feature_exact.")})
    tgt = copy.deepcopy(trunk)
    for p in tgt.parameters():
        p.requires_grad_(False)
    d = L.configs.layer_fea_output_dim[-1]
    pred = L.Predictor(d, Dt["s_fea_pairs"].shape[-1], C["pred_hidden"], C["pred_heads"]).to(dev)
    head = torch.nn.Linear(d, 1).to(dev)                             # created after the predictor: its init unchanged
    opt = torch.optim.Adam(list(trunk.parameters()) + list(pred.parameters()) + list(head.parameters()), lr=C["lr"])
    rng = np.random.default_rng(C["train_seed"])
    mon = np.sort(rng.choice(tr_rows, size=min(1024, len(tr_rows)), replace=False))
    st0 = L.collapse_stats(tgt, Dt, mon, dev)
    floor = C["var_floor_share_of_initial_std"] * st0["mean_std"]
    scale = s["aux"]["scale"]
    hist = [{"epoch": 0, **st0}]
    L.log(f"aux train: {len(tr_rows)} transitions, aux weight {AUX_WEIGHT}, scale {scale:.4f}, start {st0}")
    t0 = time.time()
    for ep in range(1, C["epochs"] + 1):
        perm = rng.permutation(tr_rows)
        tot = {"pred": 0.0, "aux": 0.0, "var": 0.0, "n": 0}
        trunk.train(); pred.train()
        for b0 in range(0, len(perm), C["batch"]):
            tr = np.sort(perm[b0:b0 + C["batch"]])
            (po, pm, pg), (h_op, h_mch, _) = L.predict(trunk, pred, Dt, tr, Dt["tr_action"][tr], dev)
            yo, ym, yg = L.targets(tgt, Dt, tr, dev)
            lp = ((po - yo) ** 2).mean() + ((pm - ym) ** 2).mean() + ((pg - yg) ** 2).mean()
            _, n_mch, _ = L.encode(trunk, L.tensors(Dt, "n_", tr, dev))          # online encoder, actual next state
            lab = torch.as_tensor(Dt["p1_mch_free_time"][tr] / scale, dtype=torch.float32, device=dev)
            la = ((head(n_mch).squeeze(-1) - lab) ** 2).mean()
            X = torch.cat([h_op, h_mch], 1).reshape(-1, h_op.shape[-1])
            lv = torch.relu(floor - X.std(0)).mean()
            loss = lp + AUX_WEIGHT * la + C["var_weight"] * lv
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p, q in zip(tgt.parameters(), trunk.parameters()):
                    p.mul_(C["ema"]).add_(q.detach(), alpha=1 - C["ema"])
            n = len(tr)
            tot["pred"] += float(lp) * n; tot["aux"] += float(la) * n; tot["var"] += float(lv) * n; tot["n"] += n
        trunk.eval()
        st = L.collapse_stats(tgt, Dt, mon, dev)
        collapsed = st["effective_rank"] < C["collapse_rank_share"] * st0["effective_rank"] or \
            st["mean_std"] < C["collapse_std_share"] * st0["mean_std"]
        hist.append({"epoch": ep, "pred_loss": tot["pred"] / tot["n"], "aux_loss": tot["aux"] / tot["n"],
                     "var_loss": tot["var"] / tot["n"], **st, "collapsed": bool(collapsed), "seconds": time.time() - t0})
        L.log(f"epoch {ep}: pred {hist[-1]['pred_loss']:.5f} aux {hist[-1]['aux_loss']:.5f} var {hist[-1]['var_loss']:.5f} "
              f"rank {st['effective_rank']:.2f} std {st['mean_std']:.4f}{' COLLAPSED' if collapsed else ''}")
    L.CK.mkdir(parents=True, exist_ok=True)
    meta = {"collection_sha256": L.sha(L.DATA / "collection.npz"), "settings": C, "aux": s["aux"]}
    bundle = L.W.trunk_bundle(trunk.state_dict(), cfg, meta, C["epochs"], hist[-1]["pred_loss"])
    torch.save(bundle, L.CK / "encoder_bundle.pth")
    torch.save(tgt.state_dict(), L.CK / "target_encoder.pth"); torch.save(pred.state_dict(), L.CK / "predictor.pth")
    torch.save(trunk.state_dict(), L.CK / "online_encoder.pth"); torch.save(head.state_dict(), L.CK / "aux_head.pth")
    (L.CK / "SHA256SUMS.json").write_text(json.dumps({p.name: L.sha(p) for p in sorted(L.CK.glob("*.pth"))}, indent=1) + "\n")
    (L.RES / "train_log.json").write_text(json.dumps({"history": hist, "variance_floor": floor, "aux_weight": AUX_WEIGHT,
                                                      "aux_scale": scale, "train_transitions": int(len(tr_rows)),
                                                      "train_seconds": time.time() - t0}, indent=1) + "\n")
    L.log(f"aux train done: {time.time() - t0:.0f} s")


def load_check():
    """The exported encoder loads into a DANIEL specialist (the tardiness pilot's random start bundle): feature_exact
    replaced, strict load, everything outside the encoder unchanged."""
    spec = torch.load(L.SPEC_START, map_location="cpu", weights_only=False)
    enc = torch.load(L.CK / "encoder_bundle.pth", map_location="cpu", weights_only=False)["state_dict"]
    fe = {k for k in spec["state_dict"] if k.startswith("feature_exact.")}
    assert fe == {k for k in enc if k.startswith("feature_exact.")}
    sd = {k: (enc[k] if k in fe else v) for k, v in spec["state_dict"].items()}
    net = DANIEL(L.configs)
    missing = net.load_state_dict(sd, strict=True)
    return {"strict_load_into_specialist_daniel": True, "missing_or_unexpected": [str(missing)],
            "shapes_equal": all(enc[k].shape == spec["state_dict"][k].shape for k in fe),
            "encoder_tensors": len(fe)}


def main():
    cfg, dev = L.setup()
    Dt = L.load_data()
    tr_rows, _ = L.split_rows(Dt)
    s = settings(Dt, tr_rows)
    if not (L.CK / "predictor.pth").exists():
        train(s)
    chk = load_check()
    (L.RES / "load_check.json").write_text(json.dumps(chk, indent=1) + "\n")
    L.log(f"load check: {chk}")
    L.evaluate()                                                     # writes heldout.json, examples.json, curves.png
    AP.main()                                                        # writes availability_difference_probe.json
    # before / after
    b_h, a_h = (json.loads((r / "heldout.json").read_text()) for r in (BASE_RES, L.RES))
    b_p, a_p = (json.loads((r / "availability_difference_probe.json").read_text()) for r in (BASE_RES, L.RES))
    row = lambda h, p: {"availability_sign_accuracy_nonzero": p["readouts"]["encoded"]["sign_accuracy_nonzero"],   # noqa: E731
                        "availability_mae_nonzero": p["readouts"]["encoded"]["mae_nonzero"],
                        "availability_pearson_nonzero": p["readouts"]["encoded"]["pearson_nonzero"],
                        "zero_baseline_mae_nonzero": p["baseline_zero"]["mae_nonzero"],
                        "raw_feature_sign_accuracy_nonzero": p["readouts"]["raw"]["sign_accuracy_nonzero"],
                        "same_op_discrimination": h["discrimination"]["same_operation_other_machine"]["accuracy"],
                        "same_op_pairs": h["discrimination"]["same_operation_other_machine"]["unordered_pairs"],
                        "median_true_branch_distance_same_op": h["discrimination"]["same_operation_other_machine"]["median_true_branch_distance"],
                        "median_pred_branch_distance_same_op": h["discrimination"]["same_operation_other_machine"]["median_pred_branch_distance"],
                        "discrimination_overall": h["discrimination"]["overall"]["accuracy"],
                        "change_explained_changed_tokens": h["errors"]["changed_tokens"]["change_explained"],
                        "changed_machines_change_explained": h["errors"]["changed_machines"]["change_explained"],
                        "shuffle_worse_share": h["shuffle_worse_share_of_states"],
                        "responds_reliably": h["decision"]["responds_reliably"]}
    cmp = {"baseline": row(b_h, b_p), "aux": row(a_h, a_p)}
    (L.RES / "before_after.json").write_text(json.dumps(cmp, indent=1) + "\n")
    print(json.dumps(cmp, indent=1))


if __name__ == "__main__":
    main()
