"""Frozen-encoder comparison: does the revised world-model encoder (pilot: pair features + action-difference loss)
carry anything useful beyond the job-machine pair facts that DANIEL's actor already receives?

Two frozen encoders, otherwise identical:
  pretrained  the trunk of checkpoints/world_model_m1_v3_action_pilot/pilot_pairs_abs+diff.pth
  random      the same architecture at the pilot's own initialisation (M1's seed-111 start trunk,
              checkpoints/world_model_m1_v3/world_model_trunk_start.pth)
Each gets a fresh, identical head (seed 111) that also receives the candidate pair's DANIEL actor features; the same
pilot training states (hold-out rows excluded), the same loss (absolute physical outcomes + action-difference term,
weight 1) and the same budget (20 epochs). Targets stay physical: per-job completion, per-machine busy time and energy;
makespan, carbon and tardiness are derived for scoring only. Evaluation: the held-out validation instances, the 3
familiar continuation rules, all pairs and same-operation pairs (two machines for the same operation, finished by the
same rule); exact ties excluded, predicted ties count 0.5. Saved checkpoints are only read; heads are written to a new
folder. The reserved final-test sets are never loaded.

    python tools/world_model_m1_encoder_probe.py   # -> results/world_model_m1_v3/encoder_probe.json
"""
from __future__ import annotations
import hashlib, itertools, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.argv = sys.argv[:1]
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import world_model_m1_action_pilot as P                             # noqa: E402
from common_utils import setup_seed                                 # noqa: E402

W, B = P.W, P.B
OUT = REPO / "results" / "world_model_m1_v3" / "encoder_probe.json"
CK = REPO / "checkpoints" / "world_model_m1_v3_encoder_probe"
EPOCHS = 20


def score(V, pj, pe):
    valid = V["rec_cand"] >= 0
    M = B.SIZES[W.SIZE][1]
    job = np.where(valid, V["rec_cand"] // M, -1)
    out = {}
    for r, rule in enumerate(W.CONT):
        T = P.objectives(V["tj"][:, :, r], V["tm"][:, :, r, :, 1], V["due"][:, None, :])
        Q = P.objectives(pj[:, :, r], pe[:, :, r], V["due"][:, None, :])
        for k in T:
            dt, dp, sj = [], [], []
            for i, j in itertools.combinations(range(valid.shape[1]), 2):
                m = valid[:, i] & valid[:, j]
                dt.append(T[k][m, i] - T[k][m, j]); dp.append(Q[k][m, i] - Q[k][m, j]); sj.append(job[m, i] == job[m, j])
            dt, dp, sj = np.concatenate(dt), np.concatenate(dp), np.concatenate(sj)
            use = np.abs(dt) > 1e-9
            ok = np.where(np.abs(dp) < 1e-9, 0.5, (np.sign(dp) == np.sign(dt)).astype(float))
            out[f"{rule}|{k}"] = {
                "accuracy": float(ok[use].mean()), "same_op_accuracy": float(ok[use & sj].mean()),
                "mae_difference": float(np.abs(dp - dt).mean()), "zero_baseline": float(np.abs(dt).mean()),
                "same_op_mae_difference": float(np.abs(dp - dt)[sj].mean()),
                "same_op_zero_baseline": float(np.abs(dt)[sj].mean()),
                "mae_absolute": float(np.abs(Q[k] - T[k])[valid].mean()),
                "pairs": int(use.sum()), "same_op_pairs": int((use & sj).sum())}
    for k in ("makespan", "carbon", "tardiness"):
        out[f"mean|{k}"] = {m: float(np.mean([out[f'{r}|{k}'][m] for r in W.CONT])) for m in out[f"{W.CONT[0]}|{k}"]}
    return out


def main():
    cfg = W.settings(False); W.set_features(cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; W.configs.device = dev
    V = W.load_split(False, "val")
    pv = dict(np.load(P.PDATA / "val_pairs.npz"))
    assert np.array_equal(pv["rec_cand"], V["rec_cand"])
    D = P.load_pilot()
    rows = np.flatnonzero(~D["holdout"].astype(bool))
    n_pair = D["fea_pairs"].shape[-1]
    src_pre = P.CK / "pilot_pairs_abs+diff.pth"
    src_rnd = W.paths(False)["ckpt"] / "world_model_trunk_start.pth"
    pre = torch.load(src_pre, map_location=dev, weights_only=False)
    trunk_pre = {k[len("trunk."):]: v for k, v in pre.items() if k.startswith("trunk.")}
    st = torch.load(src_rnd, map_location=dev, weights_only=False)
    trunk_rnd = {k[len("feature_exact."):]: v for k, v in st["state_dict"].items() if k.startswith("feature_exact.")}
    R = {"sources": {"pretrained": {"path": str(src_pre.relative_to(REPO)), "sha256": hashlib.sha256(src_pre.read_bytes()).hexdigest()},
                     "random": {"path": str(src_rnd.relative_to(REPO)), "sha256": hashlib.sha256(src_rnd.read_bytes()).hexdigest()}},
         "train_states": int(len(rows)), "epochs": EPOCHS, "loss": "absolute + action difference (weight 1)",
         "simple_rule_same_op": json.loads((REPO / "results/world_model_m1_v3/build_audit.json").read_text())["pair_feature_ranking"]}
    R["simple_rule_same_op"] = {k: v["lower_feature_is_better_share"] for k, v in R["simple_rule_same_op"].items()}
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)   # noqa: E731
    CK.mkdir(parents=True, exist_ok=True)
    heads0 = None
    for name, trunk in (("pretrained", trunk_pre), ("random", trunk_rnd)):
        setup_seed(111)
        model = P.PairWorldModel(W.configs, n_pair).to(dev)
        h0 = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("trunk.")}
        if heads0 is None:
            heads0 = h0
        assert all(torch.equal(heads0[k], h0[k]) for k in h0), "heads not identically initialised"
        model.trunk.load_state_dict(trunk)
        for p in model.trunk.parameters():
            p.requires_grad_(False)
        frozen = {k: v.clone() for k, v in model.trunk.state_dict().items()}
        params = [p for n, p in model.named_parameters() if not n.startswith("trunk.")]
        info = P.train(model, D, rows, cfg, dev, EPOCHS, True, True, params, f"{name} encoder", log)
        assert all(torch.equal(frozen[k], v) for k, v in model.trunk.state_dict().items()), "a frozen encoder changed"
        model.eval()
        pj, pm = [], []
        with torch.no_grad():
            for b0 in range(0, len(V["instance"]), 256):
                sel = np.arange(b0, min(len(V["instance"]), b0 + 256))
                s, job, mch, _ = W.batch_tensors(V, sel, dev)
                a, b = model(s, job, mch, P.pair_tensor(pv, sel, dev))
                pj.append(a.cpu().numpy()); pm.append(b.cpu().numpy())
        pj = np.concatenate(pj) * cfg["t_norm"]; pe = np.concatenate(pm)[..., 1] * cfg["e_norm"]
        R[name] = {**info, "eval": score(V, pj, pe)}
        torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("trunk.")}, CK / f"{name}_head.pth")
        OUT.write_text(json.dumps(R, indent=1) + "\n")
    (CK / "SHA256SUMS.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in sorted(CK.glob("*.pth"))}, indent=1) + "\n")
    OUT.write_text(json.dumps(R, indent=1) + "\n")
    print(f"written {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
