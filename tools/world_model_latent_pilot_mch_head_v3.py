"""Does a dedicated machine prediction head use the availability information in the centred-target encoder?

Frozen: the online and EMA target encoders of checkpoints/world_model_latent_pilot_aux_centred_v3 (read only).
Two predictor arms on the same saved transitions, training factories only:
  control      the existing Predictor (one shared output layer for operation and machine tokens), retrained
  mch_head     the same backbone and the same operation and global outputs; machine tokens get a dedicated output
               head (Linear-ReLU-Linear, h -> h -> d), shared across machines
Matched: seed 111, shared-layer initialisation (the machine head is created after the shared layers), data order
(the pilot's rng stream), Adam lr 1e-4, batch 256, 30 epochs, loss = op + machine + global MSE to the frozen target
encoding of the next state (the variance guard acts only on the encoder, which is frozen, so it is omitted in both).
Held-out factories: error on changed machine tokens, same-operation discrimination (the pilot's rule), branch
distances predicted vs true.

    python tools/world_model_latent_pilot_mch_head_v3.py  # -> results|checkpoints/world_model_latent_pilot_mch_head_v3
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.argv = sys.argv[:1] + ["none"]
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
import world_model_latent_pilot_v3 as L                             # noqa: E402
from common_utils import setup_seed                                 # noqa: E402
from model.main_model import DualAttentionNetwork                  # noqa: E402

SRC = REPO / "checkpoints" / "world_model_latent_pilot_aux_centred_v3"
SRC_RES = REPO / "results" / "world_model_latent_pilot_aux_centred_v3"
NAME = "world_model_latent_pilot_mch_head_v3"
RES, CK = REPO / "results" / NAME, REPO / "checkpoints" / NAME
C = L.C


class MchHeadPredictor(L.Predictor):
    def __init__(self, d, n_pair, h, heads):
        super().__init__(d, n_pair, h, heads)
        self.out_m = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d))        # created last

    def forward(self, h_op, h_mch, g, a_op, a_m, pair, flags_op, flags_m):
        B_, N, d = h_op.shape
        M = h_mch.shape[1]
        ctx = torch.cat([a_op, a_m, g, pair], -1)
        tok = torch.cat([h_op, h_mch], 1)
        typ = torch.zeros(B_, N + M, 2, device=tok.device); typ[:, :N, 0] = 1; typ[:, N:, 1] = 1
        fl = torch.cat([flags_op, flags_m], 1)
        x = self.att(self.inp(torch.cat([tok, typ, fl, ctx[:, None].expand(-1, N + M, -1)], -1)))
        return h_op + self.out(x[:, :N]), h_mch + self.out_m(x[:, N:]), g + self.glob(torch.cat([g, ctx], -1))


def frozen(path, dev):
    net = DualAttentionNetwork(L.configs).to(dev)
    net.load_state_dict(torch.load(path, map_location=dev)); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def train_arm(arm, Dt, trunk, tgt, dev, log):
    tr_rows, _ = L.split_rows(Dt)
    setup_seed(C["train_seed"])
    d = L.configs.layer_fea_output_dim[-1]
    cls = L.Predictor if arm == "control" else MchHeadPredictor
    pred = cls(d, Dt["s_fea_pairs"].shape[-1], C["pred_hidden"], C["pred_heads"]).to(dev)
    init = {k: v.clone() for k, v in pred.state_dict().items()}
    opt = torch.optim.Adam(pred.parameters(), lr=C["lr"])
    rng = np.random.default_rng(C["train_seed"])
    rng.choice(tr_rows, size=min(1024, len(tr_rows)), replace=False)           # the pilot's monitor draw (rng stream)
    hist, t0 = [], time.time()
    for ep in range(1, C["epochs"] + 1):
        perm = rng.permutation(tr_rows)
        tot, n = 0.0, 0
        pred.train()
        for b0 in range(0, len(perm), C["batch"]):
            tr = np.sort(perm[b0:b0 + C["batch"]])
            (po, pm, pg), _ = L.predict(trunk, pred, Dt, tr, Dt["tr_action"][tr], dev)
            yo, ym, yg = L.targets(tgt, Dt, tr, dev)
            loss = ((po - yo) ** 2).mean() + ((pm - ym) ** 2).mean() + ((pg - yg) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(tr); n += len(tr)
        hist.append({"epoch": ep, "loss": tot / n, "seconds": time.time() - t0})
        log(f"{arm} epoch {ep}: loss {tot / n:.5f}")
    pred.eval()
    torch.save(pred.state_dict(), CK / f"predictor_{arm}.pth")
    return pred, init, hist, time.time() - t0


def evaluate(pred, Dt, trunk, tgt, dev):
    _, ho = L.split_rows(Dt)
    J, M = L.B.SIZES[L.W.SIZE]
    N = Dt["s_fea_j"].shape[1]
    Z1, ZH = [], []
    with torch.no_grad():
        for b0 in range(0, len(ho), 512):
            tr = ho[b0:b0 + 512]
            (po, pm, _), _ = L.predict(trunk, pred, Dt, tr, Dt["tr_action"][tr], dev)
            yo, ym, _ = L.targets(tgt, Dt, tr, dev)
            zo, zm, _ = L.targets(tgt, Dt, tr, dev, pre="s_")
            Z1.append(torch.cat([yo, ym], 1).cpu().numpy()); ZH.append(torch.cat([po, pm], 1).cpu().numpy())
            if b0 == 0:
                Z0 = []
            Z0.append(torch.cat([zo, zm], 1).cpu().numpy())
    Z1, ZH, Z0 = np.concatenate(Z1), np.concatenate(ZH), np.concatenate(Z0)
    ch_m = (np.abs(Dt["p1_mch_free_time"] - Dt["p0_mch_free_time"]) > 1e-9) | \
        (Dt["p1_mch_working_flag"] != Dt["p0_mch_working_flag"])
    ch_m = ch_m[ho]
    em = ((ZH[:, N:] - Z1[:, N:]) ** 2).mean(-1); ec = ((Z0[:, N:] - Z1[:, N:]) ** 2).mean(-1)
    R = {"changed_machine_tokens": int(ch_m.sum()),
         "changed_machine_mse": float(em[ch_m].mean()), "changed_machine_mse_copy": float(ec[ch_m].mean()),
         "all_tokens_mse": float(((ZH - Z1) ** 2).mean())}
    R["changed_machine_change_explained"] = 1 - R["changed_machine_mse"] / R["changed_machine_mse_copy"]
    st_ho = Dt["tr_state"][ho]; a = Dt["tr_action"][ho]
    ok, dt, dp, dtm, dpm = [], [], [], [], []
    for s in np.unique(st_ho):
        k = np.flatnonzero(st_ho == s)
        for x in range(len(k)):
            for y in range(x + 1, len(k)):
                i, j = k[x], k[y]
                if np.sqrt(((Z1[i] - Z1[j]) ** 2).mean()) <= C["eval_tol"] or a[i] // M != a[j] // M:
                    continue
                ok += [((ZH[i] - Z1[i]) ** 2).mean() < ((ZH[i] - Z1[j]) ** 2).mean(),
                       ((ZH[j] - Z1[j]) ** 2).mean() < ((ZH[j] - Z1[i]) ** 2).mean()]
                dt.append(np.sqrt(((Z1[i] - Z1[j]) ** 2).mean())); dp.append(np.sqrt(((ZH[i] - ZH[j]) ** 2).mean()))
                dtm.append(np.sqrt(((Z1[i, N:] - Z1[j, N:]) ** 2).mean()))
                dpm.append(np.sqrt(((ZH[i, N:] - ZH[j, N:]) ** 2).mean()))
    dt, dp, dtm, dpm = map(np.array, (dt, dp, dtm, dpm))
    R.update({"same_op_pairs": len(dt), "same_op_discrimination": float(np.mean(ok)),
              "branch_distance_all_tokens": {"median_true": float(np.median(dt)), "median_pred": float(np.median(dp)),
                                             "median_ratio_pred_to_true": float(np.median(dp / dt))},
              "branch_distance_machine_tokens": {"median_true": float(np.median(dtm)), "median_pred": float(np.median(dpm)),
                                                 "median_ratio_pred_to_true": float(np.median(dpm / np.maximum(dtm, 1e-12)))}})
    return R


def main():
    cfg, dev = L.setup()
    RES.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True)

    def log(m):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(RES / "run.log", "a") as f:
            f.write(line + "\n")
    Dt = L.load_data()
    src_sha = {p: L.sha(SRC / p) for p in ("online_encoder.pth", "target_encoder.pth")}
    assert all(json.loads((SRC / "SHA256SUMS.json").read_text())[p] == h for p, h in src_sha.items())
    trunk, tgt = frozen(SRC / "online_encoder.pth", dev), frozen(SRC / "target_encoder.pth", dev)
    before = {k: v.clone() for k, v in list(trunk.state_dict().items()) + [("t." + k, v) for k, v in tgt.state_dict().items()]}
    (RES / "settings.json").write_text(json.dumps({"encoders": {"path": str(SRC.relative_to(REPO)), "sha256": src_sha,
                                                                "frozen": True},
                                                   "constants": C, "historical_reference_same_op": 0.5171484566389025,
                                                   "mch_head": "Linear(64,64)-ReLU-Linear(64,8), shared across machines",
                                                   "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1) + "\n")
    out, inits = {}, {}
    for arm in ("control", "mch_head"):
        pred, init, hist, secs = train_arm(arm, Dt, trunk, tgt, dev, log)
        inits[arm] = init
        out[arm] = {"parameters": int(sum(p.numel() for p in pred.parameters())), "train_seconds": secs,
                    "final_loss": hist[-1]["loss"], "history": hist, **evaluate(pred, Dt, trunk, tgt, dev)}
        log(f"{arm}: {json.dumps({k: v for k, v in out[arm].items() if k != 'history'})}")
    shared = [k for k in inits["control"]]
    out["matched_shared_init"] = all(torch.equal(inits["control"][k], inits["mch_head"][k]) for k in shared)
    after = {k: v for k, v in list(trunk.state_dict().items()) + [("t." + k, v) for k, v in tgt.state_dict().items()]}
    out["encoders_unchanged"] = all(torch.equal(before[k], after[k]) for k in before)
    out["historical_reference_same_op"] = 0.5171484566389025
    (RES / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    (CK / "SHA256SUMS.json").write_text(json.dumps({p.name: L.sha(p) for p in sorted(CK.glob("*.pth"))}, indent=1) + "\n")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "history"} if isinstance(v, dict) else v)
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
