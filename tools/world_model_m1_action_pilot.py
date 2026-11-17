"""Can the M1 world model learn how choosing one action over another changes the outcome? Two small experiments,
following the M1 build audit (results/world_model_m1_v3/build_audit.json). The original M1 checkpoint, data and
results are only read; every new artifact goes to new paths. The reserved final-test sets are never loaded.

Experiment 1 (frozen encoder): the M1 trunk frozen; two identical fresh heads (seed 111) trained 2 epochs each on the
  M1 training shards (hold-out rows excluded, as in pretraining):
    abs        the original absolute-outcome loss (world_model.loss)
    abs+diff   the same + an action-difference term: for every pair of valid actions of the same state and rule, the
               Huber error of the predicted difference of all 20 physical outputs (job completion, machine busy,
               machine energy; normalised as the targets), averaged like the absolute term; weight 1 (fixed a priori)
Experiment 2 (pilot): the first 20 training shards and the validation shard regenerated with DANIEL's actor pair
  features (fea_pairs, 10 per pair) captured (states and targets checked identical to the saved shards); encoder and
  head trained from M1's own start (seed 111), the same subset, initialisation and budget, 4 variants:
    {original inputs, + pair features of the candidate action} x {abs, abs+diff}
Evaluation (both): the saved validation states, the 3 familiar rules, makespan / carbon / tardiness derived from the
  predicted physical outcomes; pair accuracy on non-tied pairs (predicted tie = 0.5), error on the difference between
  actions vs the zero-difference baseline, absolute error; same-job pairs vs the processing-time / energy rule.

    python tools/world_model_m1_action_pilot.py all     # -> results/world_model_m1_v3/action_pilot.json
"""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
ACTION = sys.argv[1] if len(sys.argv) > 1 else "all"
sys.argv = sys.argv[:1]
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
import world_model_m1_v3 as W                                       # noqa: E402
import world_model as WM                                            # noqa: E402
from common_utils import setup_seed                                 # noqa: E402

B = W.B
OUT = REPO / "results" / "world_model_m1_v3" / "action_pilot.json"
PDATA = REPO / "data" / "world_model_m1_v3_pilot"
CK = REPO / "checkpoints" / "world_model_m1_v3_action_pilot"
PILOT_SHARDS, PILOT_EPOCHS, FROZEN_EPOCHS, DIFF_WEIGHT, SEED = 20, 20, 2, 1.0, 111
import os                                                          # noqa: E402
SMOKE = os.environ.get("AP_SMOKE") == "1"
if SMOKE:                                                          # quick check of the code path, separate outputs
    PILOT_SHARDS, PILOT_EPOCHS, FROZEN_EPOCHS = 2, 1, 1
    OUT = OUT.with_name("action_pilot_smoke.json"); PDATA = PDATA.with_name(PDATA.name + "_smoke")
    CK = CK.with_name(CK.name + "_smoke")
TOL = {"makespan": 5.0, "carbon": 0.5, "tardiness": 5.0}


# ------------------------------------------------------------------------------------------------ models

class PairWorldModel(WM.WorldModel):
    """WorldModel whose action readout also gets the candidate pair's DANIEL actor features (fea_pairs[j, m])."""

    def __init__(self, config, n_pair, hidden=64):
        super().__init__(config, hidden)
        d = self.d
        self.n_pair = n_pair
        self.job_head = WM.mlp(d + 4 * d + n_pair + WM.N_RULES, hidden, 1)
        self.mch_head = WM.mlp(d + 4 * d + n_pair + WM.N_RULES, hidden, 2)

    def forward(self, s, job, mch, pf):
        a, h_job, h_mch = self.embed(s, job, mch)
        a = torch.cat([a, pf], -1)
        B_, A, _ = a.shape
        J, M = h_job.shape[1], h_mch.shape[1]
        R = WM.N_RULES
        c = torch.eye(R, device=a.device)
        aa = a[:, :, None, None, :]; cc = c[None, None, :, None, :]
        xj = torch.cat([h_job[:, None, None].expand(B_, A, R, J, self.d), aa.expand(B_, A, R, J, a.shape[-1]),
                        cc.expand(B_, A, R, J, R)], -1)
        xm = torch.cat([h_mch[:, None, None].expand(B_, A, R, M, self.d), aa.expand(B_, A, R, M, a.shape[-1]),
                        cc.expand(B_, A, R, M, R)], -1)
        return self.job_head(xj).squeeze(-1), self.mch_head(xm)


def diff_loss(pred, target, valid):
    """Huber error of the predicted difference between two valid actions of the same state and rule, over the 20
    physical outputs (averaged like world_model.loss)."""
    pj, pm = pred; tj, tm = target
    A = pj.shape[1]
    tot, n = 0.0, 0.0
    n_out = tj.shape[-1] + tm.shape[-2] * tm.shape[-1]
    for i, j in itertools.combinations(range(A), 2):
        m = (valid[:, i] & valid[:, j]).float()                                        # [B]
        hj = nn.functional.huber_loss(pj[:, i] - pj[:, j], tj[:, i] - tj[:, j], reduction="none", delta=1.0).sum(-1)
        hm = nn.functional.huber_loss(pm[:, i] - pm[:, j], tm[:, i] - tm[:, j], reduction="none", delta=1.0).sum((-1, -2))
        per = (hj + hm) / n_out                                                         # [B, R]
        tot = tot + (per * m[:, None]).sum(); n = n + m.sum() * per.shape[1]
    return tot / max(float(n), 1.0)


def pair_tensor(D, sel, dev):
    """[B, A, P] DANIEL pair features of each candidate action (0 for padding)."""
    c = D["rec_cand"][sel]; M = B.SIZES[W.SIZE][1]
    j, m = np.where(c >= 0, c // M, 0), np.where(c >= 0, c % M, 0)
    fp = D["fea_pairs"][sel]
    out = fp[np.arange(len(sel))[:, None], j, m] * (c >= 0)[..., None]
    return torch.as_tensor(out, dtype=torch.float32, device=dev)


def train(model, D, rows, cfg, dev, epochs, use_diff, use_pairs, params, tag, log):
    tj, tm = W.normalise(cfg, D["tj"], D["tm"])
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    g = torch.Generator().manual_seed(SEED)
    bs = cfg["batch_states"]
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        model.train()
        if not any(p.requires_grad for p in model.trunk.parameters()):
            model.trunk.eval()                                        # frozen encoder: no dropout, fixed features
        order = rows[torch.randperm(len(rows), generator=g).numpy()]
        tot = n = 0
        for b0 in range(0, len(order), bs):
            sel = order[b0:b0 + bs]
            s, job, mch, val = W.batch_tensors(D, sel, dev)
            tgt = (torch.as_tensor(np.nan_to_num(tj[sel]), device=dev), torch.as_tensor(np.nan_to_num(tm[sel]), device=dev))
            pred = model(s, job, mch, pair_tensor(D, sel, dev)) if use_pairs else model(s, job, mch)
            L = WM.loss(pred, tgt, val)
            if use_diff:
                L = L + DIFF_WEIGHT * diff_loss(pred, tgt, val)
            opt.zero_grad(); L.backward(); opt.step()
            tot += L.item() * len(sel); n += len(sel)
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "seconds": time.time() - t0})
        log(f"{tag}: epoch {ep + 1}/{epochs} loss {tot / n:.6f} ({time.time() - t0:.0f} s)")
    return {"history": hist, "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------ evaluation

def objectives(jc, en, due):
    return {"makespan": jc.max(-1), "carbon": en.sum(-1), "tardiness": np.maximum(0.0, jc - due).sum(-1)}


def evaluate(model, V, cfg, dev, use_pairs, pf_val=None):
    model.eval()
    pj, pm = [], []
    with torch.no_grad():
        for b0 in range(0, len(V["instance"]), 256):
            sel = np.arange(b0, min(len(V["instance"]), b0 + 256))
            s, job, mch, _ = W.batch_tensors(V, sel, dev)
            out = model(s, job, mch, pair_tensor(pf_val, sel, dev)) if use_pairs else model(s, job, mch)
            pj.append(out[0].cpu().numpy()); pm.append(out[1].cpu().numpy())
    pj = np.concatenate(pj) * cfg["t_norm"]; pe = np.concatenate(pm)[..., 1] * cfg["e_norm"]
    return score(V, pj, pe)


def score(V, pj, pe):
    valid = V["rec_cand"] >= 0
    M = B.SIZES[W.SIZE][1]
    job = np.where(valid, V["rec_cand"] // M, -1)
    res = {}
    for r, rule in enumerate(W.CONT):
        T = objectives(V["tj"][:, :, r], V["tm"][:, :, r, :, 1], V["due"][:, None, :])
        P = objectives(pj[:, :, r], pe[:, :, r], V["due"][:, None, :])
        for k in T:
            dt, dp, sj = [], [], []
            for i, j in itertools.combinations(range(valid.shape[1]), 2):
                m = valid[:, i] & valid[:, j]
                dt.append(T[k][m, i] - T[k][m, j]); dp.append(P[k][m, i] - P[k][m, j]); sj.append(job[m, i] == job[m, j])
            dt, dp, sj = np.concatenate(dt), np.concatenate(dp), np.concatenate(sj)
            use = np.abs(dt) > 1e-9
            ok = np.where(np.abs(dp) < 1e-9, 0.5, (np.sign(dp) == np.sign(dt)).astype(float))
            res[f"{rule}|{k}"] = {"accuracy": float(ok[use].mean()), "same_job_accuracy": float(ok[use & sj].mean()),
                                  "mae_difference": float(np.abs(dp - dt).mean()),
                                  "mae_difference_zero_baseline": float(np.abs(dt).mean()),
                                  "mae_absolute": float(np.abs(P[k] - T[k])[valid].mean())}
    keys = list(res)
    res["mean"] = {m: float(np.mean([res[k][m] for k in keys])) for m in res[keys[0]]}
    for k in ("makespan", "carbon", "tardiness"):
        res[f"mean|{k}"] = {m: float(np.mean([res[f'{r}|{k}'][m] for r in W.CONT])) for m in res[keys[0]]}
    return res


# ------------------------------------------------------------------------------------------------ data for the pilot

def enrich(cfg, log):
    """Regenerates the first PILOT_SHARDS training shards and the validation shard with fea_pairs captured; checks
    states and targets identical to the saved shards. Writes to data/world_model_m1_v3_pilot/ only."""
    PDATA.mkdir(parents=True, exist_ok=True)
    W.STATE_KEYS = W.STATE_KEYS + ("fea_pairs",)
    W.DTYPES["fea_pairs"] = np.float32
    from benchmark_v3_generator import GeneratorSource
    saved = sorted((W.paths(False)["data"] / "train").glob("shard_*.npz"))
    checks = {}
    if not (PDATA / f"train_{PILOT_SHARDS - 1:04d}.npz").exists():
        setup_seed(cfg["train_seed"])
        gen = GeneratorSource(W.configs, W.SIZE, B.load_manifest())
        rng = np.random.default_rng(cfg["train_seed"])
        t0 = time.time()
        for sh in range(PILOT_SHARDS):
            data = gen.sample(cfg["shard_instances"])
            n = len(data); idx = np.arange(n)
            beh = rng.integers(0, len(W.BEHAVIOURS), n)
            N = data.op_pt[0].shape[0]
            steps = [np.sort(rng.choice(N - 1, size=cfg["states_per_episode"], replace=False)) for _ in idx]
            states, rec, ep_actions = W.behaviour_episodes(data, idx, beh, steps, cfg["n_candidates"], rng)
            tj, tm, _, _ = W.continuations(data, idx, ep_actions, rec, W.CONT, False)
            ref = np.load(saved[sh])
            same = all(np.array_equal(states[k], ref[k]) for k in W.STATE_KEYS if k != "fea_pairs") and \
                np.array_equal(rec["cand"], ref["rec_cand"]) and np.allclose(tj.astype(np.float32), ref["tj"], equal_nan=True)
            if not same:
                raise SystemExit(f"regenerated shard {sh} differs from the saved one; pilot data not trusted")
            g_inst = sh * cfg["shard_instances"] + rec["episode"]
            np.savez(PDATA / f"train_{sh:04d}.npz", **states, **{f"rec_{k}": v for k, v in rec.items()},
                     holdout=(g_inst % 10) == 9, tj=tj.astype(np.float32), tm=tm.astype(np.float32))
        checks["train_regeneration_seconds"] = time.time() - t0
        log(f"pilot data: {PILOT_SHARDS} training shards regenerated identically with pair features "
            f"({checks['train_regeneration_seconds']:.0f} s)")
    if not (PDATA / "val_pairs.npz").exists():
        V = W.load_split(False, "val")
        # reproduce_states returned states via behaviour_episodes with fea_pairs captured
        data = B.load_split("validation", W.SIZE)
        n = min(len(data), cfg["val_instances"])
        rng = np.random.default_rng(cfg["val_seed"])
        idx = np.repeat(np.arange(n), len(W.BEHAVIOURS)); beh = np.tile(np.arange(len(W.BEHAVIOURS)), n)
        steps = [np.arange(0, data.op_pt[0].shape[0] - 1, cfg["val_step_stride"]) for _ in idx]
        st, rc, _ = W.behaviour_episodes(data, idx, beh, steps, cfg["n_candidates"], rng)
        if not (np.array_equal(rc["cand"], V["rec_cand"]) and all(np.array_equal(st[k], V[k]) for k in W.STATE_KEYS
                                                                     if k != "fea_pairs")):
            raise SystemExit("validation states not reproduced; pilot not trusted")
        np.savez(PDATA / "val_pairs.npz", fea_pairs=st["fea_pairs"], rec_cand=rc["cand"])
        log("pilot data: validation pair features captured (states identical to the saved shard)")
    checks["identical_to_saved"] = True
    return checks


def load_pilot():
    files = sorted(PDATA.glob("train_*.npz"))[:PILOT_SHARDS]
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


# ------------------------------------------------------------------------------------------------ main

def main():
    cfg = W.settings(False); W.set_features(cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; W.configs.device = dev
    logf = open(OUT.with_suffix(".log"), "a")

    def log(msg):                                                   # its own log; the M1 run.log is not touched
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    CK.mkdir(parents=True, exist_ok=True)
    V = W.load_split(False, "val")
    # reference rows: zero-difference baseline and the simple pair rule on same-job pairs (from the audit)
    audit = json.loads((REPO / "results/world_model_m1_v3/build_audit.json").read_text())
    R["simple_rule_same_job"] = {k: v["lower_feature_is_better_share"] for k, v in audit["pair_feature_ranking"].items()}
    R["original_m1"] = score(V, *orig_predictions(cfg, dev, V))
    # ---- experiment 1: frozen encoder
    if "frozen" not in R:
        t0 = time.time()
        if SMOKE:
            fs = sorted((W.paths(False)["data"] / "train").glob("shard_*.npz"))[:2]
            parts = [np.load(f) for f in fs]
            D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
        else:
            D = W.load_split(False, "train")
        D.pop("factory", None)
        rows = np.flatnonzero(~D["holdout"].astype(bool))
        if SMOKE:
            rows = rows[:3000]
        log(f"frozen test: {len(rows)} training states loaded ({time.time() - t0:.0f} s)")
        R["frozen"] = {}
        m1 = W.load_world_model(False, dev)
        for arm, use_diff in (("abs", False), ("abs+diff", True)):
            setup_seed(SEED)
            model = WM.WorldModel(W.configs).to(dev)
            model.trunk.load_state_dict(m1.trunk.state_dict())
            for p in model.trunk.parameters():
                p.requires_grad_(False)
            heads = [p for n, p in model.named_parameters() if not n.startswith("trunk.")]
            info = train(model, D, rows, cfg, dev, FROZEN_EPOCHS, use_diff, False, heads, f"frozen {arm}", log)
            assert all(torch.equal(a, b) for a, b in zip(model.trunk.state_dict().values(), m1.trunk.state_dict().values()))
            R["frozen"][arm] = {**info, "eval": evaluate(model, V, cfg, dev, False)}
            torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("trunk.")}, CK / f"frozen_{arm}_heads.pth")
            OUT.write_text(json.dumps(R, indent=1) + "\n")
        R["frozen"]["load_seconds"] = time.time() - t0
        del D
    # ---- experiment 2: the pilot
    R["pilot_data"] = enrich(cfg, log)
    if "pilot" not in R:
        P = load_pilot()
        rows = np.flatnonzero(~P["holdout"].astype(bool))
        pv = dict(np.load(PDATA / "val_pairs.npz"))
        R["pilot"] = {"train_states": int(len(rows)), "shards": PILOT_SHARDS, "epochs": PILOT_EPOCHS}
        start = torch.load(W.paths(False)["ckpt"] / "world_model_trunk_start.pth", map_location=dev, weights_only=False)
        trunk0 = {k[len("feature_exact."):]: v for k, v in start["state_dict"].items() if k.startswith("feature_exact.")}
        n_pair = P["fea_pairs"].shape[-1]
        for inputs in ("original", "pairs"):
            for arm, use_diff in (("abs", False), ("abs+diff", True)):
                tag = f"pilot {inputs} {arm}"
                setup_seed(SEED)
                model = (PairWorldModel(W.configs, n_pair) if inputs == "pairs" else WM.WorldModel(W.configs)).to(dev)
                model.trunk.load_state_dict(trunk0)                     # M1's own start trunk (seed 111)
                info = train(model, P, rows, cfg, dev, PILOT_EPOCHS, use_diff, inputs == "pairs",
                             list(model.parameters()), tag, log)
                R["pilot"][f"{inputs}|{arm}"] = {**info, "eval": evaluate(model, V, cfg, dev, inputs == "pairs", pv)}
                torch.save(model.state_dict(), CK / f"pilot_{inputs}_{arm}.pth")
                OUT.write_text(json.dumps(R, indent=1) + "\n")
    import hashlib
    (CK / "SHA256SUMS.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in sorted(CK.glob("*.pth"))}, indent=1) + "\n")
    OUT.write_text(json.dumps(R, indent=1) + "\n")
    log("action pilot: done")


def orig_predictions(cfg, dev, V):
    m = W.load_world_model(False, dev)
    pj, pm = [], []
    with torch.no_grad():
        for b0 in range(0, len(V["instance"]), 256):
            sel = np.arange(b0, min(len(V["instance"]), b0 + 256))
            s, job, mch, _ = W.batch_tensors(V, sel, dev)
            a, b = m(s, job, mch)
            pj.append(a.cpu().numpy()); pm.append(b.cpu().numpy())
    return np.concatenate(pj) * cfg["t_norm"], np.concatenate(pm)[..., 1] * cfg["e_norm"]


if __name__ == "__main__":
    main()
