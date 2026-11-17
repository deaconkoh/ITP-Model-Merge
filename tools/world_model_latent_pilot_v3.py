"""Action-conditioned latent next-state pilot (10x5, waiting allowed: --dispatch_rule semi_active).

Pretext task, objective-agnostic: from a factory state and one discrete operation-machine assignment, predict the
ENCODED next state (one simulator step later). The encoder is DANIEL's trunk (DualAttentionNetwork, the specialists'
feature_exact keys), so it can later be loaded into a specialist with --init_mode exact.

  collect   fresh generator factories (seed 3111; the generator excludes the validation/evaluation pools and the
            final-test stream), split by factory (i % 5 == 4 held out). Five behaviour sources on every factory:
            MWKR+ECT, MWKR+MINE, EDD+ECT (lockstep decoder), RANDOM legal moves, and the waiting-enabled DANIEL policy
            bv3_wt_W2 (sampled, one simplex preference per episode; the preference only drives behaviour). At selected
            states, 2-4 distinct legal assignments are each simulated ONE step from a copy of the same state.
  train     online encoder (seed-111 random start trunk) + predictor -> stop-gradient EMA target encoder of the next
            state; variance hinge as collapse guard; collapse metrics logged every epoch. No rewards or preferences.
  evaluate  held-out factories: predictor vs copy-forward vs action shuffle, on changed / unchanged tokens; pairwise
            discrimination between branches of the same state; examples.

    python tools/world_model_latent_pilot_v3.py all      # LP_SMOKE=1 for the smoke run (separate folders)
"""
from __future__ import annotations
import copy, hashlib, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
ACTION = sys.argv[1] if len(sys.argv) > 1 else "all"
sys.argv = sys.argv[:1]
os.environ.setdefault("DECODING_DEVICE", "cpu")
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
import world_model_m1_v3 as W                                       # noqa: E402
import decoding_v3 as D                                             # noqa: E402
import world_model as WM                                            # noqa: E402
from common_utils import setup_seed                                 # noqa: E402
from model.main_model import DANIEL, DualAttentionNetwork          # noqa: E402

B, configs = W.B, W.configs
SMOKE = os.environ.get("LP_SMOKE") == "1"
SFX = "_smoke" if SMOKE else ""
DATA = REPO / "data" / f"world_model_latent_pilot_v3{SFX}"
RES = REPO / "results" / f"world_model_latent_pilot_v3{SFX}"
CK = REPO / "checkpoints" / f"world_model_latent_pilot_v3{SFX}"
START = REPO / "checkpoints" / "world_model_m1_v3" / "world_model_trunk_start.pth"
POLICY = REPO / "checkpoints" / "waiting_training_v3" / "10x5+mix+bv3_wt_W2_s111_final_u5300.pth"
SPEC_START = REPO / "checkpoints" / "world_model_m1_v3_tardiness_pilot" / "start_rnd.pth"
SOURCES = ["MWKR+ECT", "MWKR+MINE", "EDD+ECT", "RANDOM", "DANIEL"]
C = {"instances": 10 if SMOKE else 200, "seed": 3111, "holdout_mod": 5, "holdout_rem": 4, "states_per_episode": 8,
     "max_branches": 4, "epochs": 2 if SMOKE else 30, "lr": 1e-4, "batch": 256, "train_seed": 111, "ema": 0.996,
     "var_weight": 1.0, "var_floor_share_of_initial_std": 0.5, "pred_hidden": 64, "pred_heads": 4,
     "collapse_rank_share": 0.25, "collapse_std_share": 0.10, "eval_tol": 1e-6}
RULE = {"change_explained_min": 0.2, "shuffle_worse_share_min": 0.9, "discrimination_min": 0.9,
        "discrimination_per_type_min": 0.8, "per_type_min_pairs": 50}
STATE = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs")
PHYS = ("op_ct_lb", "op_scheduled_flag", "mch_free_time", "mch_working_flag")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(msg):
    RES.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RES / "run.log", "a") as f:
        f.write(line + "\n")


def setup():
    cfg = W.settings(False); W.set_features(cfg)
    configs.dispatch_rule = "semi_active"
    dev = "cuda" if torch.cuda.is_available() else "cpu"; configs.device = dev
    return cfg, dev


def settings():
    f = RES / "settings.json"
    if f.exists():
        return json.loads(f.read_text())
    s = {"constants": C, "decision_rule": RULE, "sources": SOURCES, "dispatch_rule": "semi_active",
         "decision_rule_text": "responds reliably to the action iff, on held-out factories: change explained on changed "
                               "tokens >= 0.2; shuffled-action error > true-action error (changed tokens) in >= 90% of "
                               "states; pairwise discrimination >= 0.9 overall and >= 0.8 in each pair type with >= 50 "
                               "pairs",
         "inputs": {p.name: {"path": str(p.relative_to(REPO)), "sha256": sha(p)} for p in (START, POLICY, SPEC_START)},
         "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(s, indent=1) + "\n")
    return s


# ------------------------------------------------------------------------------------------------ collection

def load_policy(dev):
    b = torch.load(POLICY, map_location=dev, weights_only=False)
    cf = copy.copy(configs); cf.preference_dim = 3; cf.goal = "mct"
    net = DANIEL(cf).to(dev); net.load_state_dict(b["state_dict"]); return net.eval()


def snapshot(env, rows, keys):
    return {k: np.asarray(getattr(env, k))[rows].copy() for k in keys}


def choose_branches(env, e, a0, rng):
    """2-4 distinct legal actions for row e: behaviour, same op other machine, immediate + later-starting, random."""
    M = env.number_of_machines
    legal = ~env.dynamic_pair_mask[e].astype(bool)                     # [J, M]
    P = env.pair_free_time[e]
    if legal.sum() < 2:
        return None
    earliest = P[legal].min()
    imm = legal & (P <= earliest + 1e-9); late = legal & (P > earliest + 1e-9)
    chosen = [int(a0)]
    j0, m0 = divmod(int(a0), M)
    other_m = [m for m in np.flatnonzero(legal[j0]) if m != m0]
    if other_m:
        chosen.append(int(j0 * M + rng.choice(other_m)))
    for cls in (imm, late):
        flat = [a for a in np.flatnonzero(cls.reshape(-1)) if a not in chosen]
        if cls.reshape(-1)[chosen].any() or not flat:
            continue
        if len(chosen) < C["max_branches"]:
            chosen.append(int(rng.choice(flat)))
    rest = [a for a in np.flatnonzero(legal.reshape(-1)) if a not in chosen]
    while len(chosen) < C["max_branches"] and rest:
        a = int(rng.choice(rest)); chosen.append(a); rest.remove(a)
    start = P.reshape(-1)[chosen]
    return chosen, (start > earliest + 1e-9).astype(np.uint8)


def run_source(src, data, idx, rng, net, dev):
    E = len(idx)
    env = W.T.FJSPEnvForSameOpNums(*B.SIZES[W.SIZE]); env.set_initial_data(**data.env_kwargs(idx, "mct"))
    assert env.dispatch_rule == "semi_active"
    prefs = D.PREFS[rng.integers(0, len(D.PREFS), E)] if src == "DANIEL" else np.full((E, 3), 1 / 3)
    env.set_preference(prefs)
    dec = None
    if src in ("MWKR+ECT", "MWKR+MINE", "EDD+ECT"):
        dec = W.Decoder(data, idx); dec.reset()
    N = env.number_of_ops
    want = np.zeros((E, N), bool)
    for e in range(E):
        want[e, np.sort(rng.choice(N - 1, size=C["states_per_episode"], replace=False))] = True
    S = {k: [] for k in STATE + ("job_first_op_id", "job_last_op_id")}; T = {k: [] for k in STATE}; PH0 = {k: [] for k in PHYS}; PH1 = {k: [] for k in PHYS}
    rec = {"state_episode": [], "state_step": [], "tr_state": [], "tr_action": [], "tr_later": [], "tr_behaviour": [],
           "beh_later": []}
    actions = np.zeros((E, N), int)
    gen = torch.Generator(device=dev).manual_seed(int(rng.integers(1 << 31)))
    for t in range(N):
        legal = ~env.dynamic_pair_mask.astype(bool)
        if dec is not None:
            ok, nst = dec.allowed()
            assert not (ok & ~legal).any(), "a decoder-allowed action is illegal in the semi-active environment"
            a = dec.rule(src, ok, nst)
        elif src == "RANDOM":
            a = np.where(legal.reshape(E, -1), rng.random((E, legal[0].size)), -1.0).argmax(1)
        else:
            st_ = env.state
            with torch.no_grad():
                pi, _ = net(**{k: getattr(st_, f"{k}_tensor").to(dev) for k in
                               ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask",
                                "fea_pairs")}, pref=st_.pref_tensor.to(dev))
            a = torch.multinomial(pi, 1, generator=gen).squeeze(1).cpu().numpy()
        assert legal.reshape(E, -1)[np.arange(E), a].all()
        P = env.pair_free_time.reshape(E, -1)
        earliest = np.where(legal.reshape(E, -1), P, np.inf).min(1)
        rec["beh_later"].append(P[np.arange(E), a] > earliest + 1e-9)
        rows, acts = [], []
        for e in np.flatnonzero(want[:, t]):
            br = choose_branches(env, e, a[e], rng)
            if br is None:
                continue
            ch, later = br
            sid = len(rec["state_episode"])
            rec["state_episode"].append(e); rec["state_step"].append(t)
            for k in STATE:
                S[k].append(np.asarray(getattr(env, k)[e]).copy())
            S["job_first_op_id"].append(env.job_first_op_id[e].copy()); S["job_last_op_id"].append(env.job_last_op_id[e].copy())
            for i, (x, l) in enumerate(zip(ch, later)):
                rows.append(e); acts.append(x)
                rec["tr_state"].append(sid); rec["tr_action"].append(x); rec["tr_later"].append(l)
                rec["tr_behaviour"].append(i == 0)
        if rows:
            before = snapshot(env, np.arange(E), PHYS + ("fea_j",))
            br_env = D.select(env, rows)
            ph0 = snapshot(br_env, np.arange(len(rows)), PHYS)
            br_env.step(np.asarray(acts))
            for k in STATE:
                T[k] += list(np.asarray(getattr(br_env, k)).copy())
            for k in PHYS:
                PH0[k] += list(ph0[k]); PH1[k] += list(np.asarray(getattr(br_env, k)).copy())
            after = snapshot(env, np.arange(E), PHYS + ("fea_j",))
            assert all(np.array_equal(before[k], after[k]) for k in before), "a branch step changed the main episode"
        actions[:, t] = a
        if dec is not None:
            dec.step(a)
        env.step(a)
    out = {**{f"s_{k}": np.stack(v) for k, v in S.items()}, **{f"n_{k}": np.stack(v) for k, v in T.items()},
           **{f"p0_{k}": np.stack(v) for k, v in PH0.items()}, **{f"p1_{k}": np.stack(v) for k, v in PH1.items()},
           **{k: np.asarray(v) for k, v in rec.items() if k != "beh_later"},
           "beh_later": np.stack(rec["beh_later"], 1), "actions": actions, "prefs": prefs}
    return out


def replay_check(data, idx, out, n, rng):
    """The saved next state equals a fresh semi-active environment replaying the episode prefix + the branch action."""
    bad = 0
    for k in rng.choice(len(out["tr_state"]), size=n, replace=False):
        s = out["tr_state"][k]; e = out["state_episode"][s]; t = out["state_step"][s]
        env = W.T.FJSPEnvForSameOpNums(*B.SIZES[W.SIZE]); env.set_initial_data(**data.env_kwargs([idx[e]], "mct"))
        for u in range(t):
            env.step(np.array([out["actions"][e, u]]))
        if not all(np.array_equal(np.asarray(getattr(env, q))[0], out[f"s_{q}"][s]) for q in STATE):
            bad += 1; continue
        env.step(np.array([out["tr_action"][k]]))
        bad += not all(np.array_equal(np.asarray(getattr(env, q))[0], out[f"n_{q}"][k]) for q in STATE)
    return {"checked": int(n), "mismatches": int(bad)}


def collect():
    cfg, dev = setup()
    if (DATA / "collection.npz").exists():
        return
    t0 = time.time()
    from benchmark_v3_generator import GeneratorSource
    setup_seed(C["seed"])
    gen = GeneratorSource(configs, W.SIZE, B.load_manifest())
    data = gen.sample(C["instances"])
    n = len(data); idx = np.arange(n)
    hold = (idx % C["holdout_mod"]) == C["holdout_rem"]
    net = load_policy(dev)
    rng = np.random.default_rng(C["seed"])
    parts, cov, checks = [], {}, {}
    off_s = 0
    for si, src in enumerate(SOURCES):
        ts = time.time()
        out = run_source(src, data, idx, rng, net, dev)
        checks[src] = replay_check(data, idx, out, 5, rng)
        if checks[src]["mismatches"]:
            raise SystemExit(f"replay check failed for {src}: {checks[src]}")
        st_inst = idx[out["state_episode"]]; tr_inst = st_inst[out["tr_state"]]
        M = B.SIZES[W.SIZE][1]
        # pair types within each state
        for split, mask_s in (("train", ~hold[st_inst]), ("heldout", hold[st_inst])):
            sids = np.flatnonzero(mask_s); trm = np.isin(out["tr_state"], sids)
            a = out["tr_action"][trm]; s_of = out["tr_state"][trm]; beh = out["tr_behaviour"][trm]
            same_op = np.array([((a[(s_of == s_of[i])] // M == a[i] // M).sum() > 1) for i in range(len(a))])
            em = np.isin(np.arange(n), np.unique(out["state_episode"][sids]))
            cov.setdefault(src, {})[split] = {
                "factories": int(em.sum()), "states": int(len(sids)), "transitions": int(trm.sum()),
                "branches_per_state": float(trm.sum() / max(1, len(sids))),
                "share_transitions_in_same_op_group": float(same_op.mean()),
                "share_later_starting": float(out["tr_later"][trm].mean()),
                "share_states_with_both_start_types": float(np.mean([len(set(out["tr_later"][out["tr_state"] == s])) == 2
                                                                    for s in sids])),
                "behaviour_actions_later_starting": float(out["beh_later"][em].mean())}
        out["state_episode_inst"] = st_inst; out["source"] = np.full(len(st_inst), si)
        out["tr_state"] = out["tr_state"] + off_s; off_s += len(st_inst)
        parts.append(out)
        log(f"collect {src}: {len(st_inst)} states, {len(out['tr_state'])} transitions, {time.time() - ts:.0f} s")
    keys = [k for k in parts[0] if k not in ("actions", "prefs", "beh_later", "state_episode", "state_step")]
    allp = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    allp["state_holdout"] = hold[allp["state_episode_inst"]]
    allp["factory"] = np.asarray([data.keys[i] for i in idx])
    assert not set(allp["state_episode_inst"][allp["state_holdout"]]) & set(allp["state_episode_inst"][~allp["state_holdout"]])
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / "collection.npz", **allp)
    secs = time.time() - t0
    (RES / "coverage.json").write_text(json.dumps(
        {"factories": {"total": int(n), "train": int((~hold).sum()), "heldout": int(hold.sum())},
         "per_source": cov, "replay_checks": checks, "branch_main_episode_untouched": True,
         "totals": {"states": int(len(allp["state_holdout"])), "transitions": int(len(allp["tr_state"]))},
         "collection_seconds": secs, "generator": gen.provenance()}, indent=1, default=str) + "\n")
    log(f"collection: {len(allp['state_holdout'])} states, {len(allp['tr_state'])} transitions, {secs:.0f} s")


# ------------------------------------------------------------------------------------------------ model

class Predictor(nn.Module):
    """Per-token residual predictor of the next state's tokens (operations, machines) and globals."""

    def __init__(self, d, n_pair, h, heads):
        super().__init__()
        ctx = 4 * d + n_pair
        self.inp = nn.Sequential(nn.Linear(d + 2 + 3 + ctx, h), nn.ReLU(), nn.Linear(h, h))
        self.att = nn.TransformerEncoderLayer(h, heads, 2 * h, dropout=0.0, batch_first=True)
        self.out = nn.Linear(h, d)
        self.glob = WM.mlp(2 * d + ctx, h, 2 * d)

    def forward(self, h_op, h_mch, g, a_op, a_m, pair, flags_op, flags_m):
        B_, N, d = h_op.shape
        M = h_mch.shape[1]
        ctx = torch.cat([a_op, a_m, g, pair], -1)                                       # [B, 4d + n_pair]
        tok = torch.cat([h_op, h_mch], 1)
        typ = torch.zeros(B_, N + M, 2, device=tok.device); typ[:, :N, 0] = 1; typ[:, N:, 1] = 1
        fl = torch.cat([flags_op, flags_m], 1)
        x = self.inp(torch.cat([tok, typ, fl, ctx[:, None].expand(-1, N + M, -1)], -1))
        y = tok + self.out(self.att(x))
        return y[:, :N], y[:, N:], g + self.glob(torch.cat([g, ctx], -1))


def encode(trunk, s):
    h_op, h_mch, g_op, g_mch = WM.trunk_forward(trunk, s)
    return h_op, h_mch, torch.cat([g_op, g_mch], -1)


def tensors(Dt, pre, rows, dev):
    return {"fea_j": torch.as_tensor(Dt[f"{pre}fea_j"][rows], dtype=torch.float32, device=dev),
            "op_mask": torch.as_tensor(Dt[f"{pre}op_mask"][rows], dtype=torch.float32, device=dev),
            "candidate": torch.as_tensor(Dt[f"{pre}candidate"][rows], dtype=torch.int64, device=dev),
            "fea_m": torch.as_tensor(Dt[f"{pre}fea_m"][rows], dtype=torch.float32, device=dev),
            "mch_mask": torch.as_tensor(Dt[f"{pre}mch_mask"][rows], dtype=torch.float32, device=dev),
            "comp_idx": torch.as_tensor(Dt[f"{pre}comp_idx"][rows], dtype=torch.float32, device=dev)}


def predict(trunk, pred, Dt, tr, actions, dev):
    """Prediction for transitions tr with the given actions (flat j*M+m)."""
    M = B.SIZES[W.SIZE][1]
    st = Dt["tr_state"][tr]
    s = tensors(Dt, "s_", st, dev)
    h_op, h_mch, g = encode(trunk, s)
    job = torch.as_tensor(actions // M, device=dev); mch = torch.as_tensor(actions % M, device=dev)
    ar = torch.arange(len(tr), device=dev)
    op = s["candidate"][ar, job]
    a_op = h_op[ar, op]; a_m = h_mch[ar, mch]
    pair = torch.as_tensor(Dt["s_fea_pairs"][st, actions // M, actions % M], dtype=torch.float32, device=dev)
    N = h_op.shape[1]
    J = s["candidate"].shape[1]
    fo = torch.zeros(len(tr), N, 3, device=dev)
    fo[ar, op, 0] = 1
    first = torch.as_tensor(Dt["s_job_first_op_id"][st, actions // M], device=dev)
    last = torch.as_tensor(Dt["s_job_last_op_id"][st, actions // M], device=dev)
    opi = torch.arange(N, device=dev)[None]
    fo[..., 1] = ((opi >= first[:, None]) & (opi <= last[:, None])).float()
    fm = torch.zeros(len(tr), M, 3, device=dev); fm[ar, mch, 2] = 1
    return pred(h_op, h_mch, g, a_op, a_m, pair, fo, fm), (h_op, h_mch, g)


def targets(tgt, Dt, tr, dev, pre="n_"):
    with torch.no_grad():
        rows = tr if pre == "n_" else Dt["tr_state"][tr]
        return encode(tgt, tensors(Dt, pre, rows, dev))


def collapse_stats(tgt, Dt, tr, dev):
    with torch.no_grad():
        h_op, h_mch, _ = targets(tgt, Dt, tr, dev)
        X = torch.cat([h_op, h_mch], 1).reshape(-1, h_op.shape[-1]).double()
        std = X.std(0)
        ev = torch.linalg.eigvalsh(torch.cov(X.T)).clamp(min=0)
        p = ev / ev.sum()
        rank = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
        return {"mean_std": float(std.mean()), "min_std": float(std.min()), "effective_rank": rank}


def load_data():
    return dict(np.load(DATA / "collection.npz", allow_pickle=False))


def split_rows(Dt):
    hold_tr = Dt["state_holdout"][Dt["tr_state"]]
    return np.flatnonzero(~hold_tr), np.flatnonzero(hold_tr)


def train():
    cfg, dev = setup()
    if (CK / "predictor.pth").exists():
        return
    Dt = load_data()
    tr_rows, _ = split_rows(Dt)
    setup_seed(C["train_seed"])
    trunk = DualAttentionNetwork(configs).to(dev)
    sd = torch.load(START, map_location=dev, weights_only=False)["state_dict"]
    trunk.load_state_dict({k[len("feature_exact."):]: v for k, v in sd.items() if k.startswith("feature_exact.")})
    tgt = copy.deepcopy(trunk)
    for p in tgt.parameters():
        p.requires_grad_(False)
    d = configs.layer_fea_output_dim[-1]
    pred = Predictor(d, Dt["s_fea_pairs"].shape[-1], C["pred_hidden"], C["pred_heads"]).to(dev)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(pred.parameters()), lr=C["lr"])
    rng = np.random.default_rng(C["train_seed"])
    mon = np.sort(rng.choice(tr_rows, size=min(1024, len(tr_rows)), replace=False))
    st0 = collapse_stats(tgt, Dt, mon, dev)
    floor = C["var_floor_share_of_initial_std"] * st0["mean_std"]
    hist = [{"epoch": 0, **st0}]
    log(f"train: {len(tr_rows)} transitions, start {st0}, variance floor {floor:.4f}")
    t0 = time.time()
    for ep in range(1, C["epochs"] + 1):
        perm = rng.permutation(tr_rows)
        tot = {"pred": 0.0, "var": 0.0, "n": 0}
        trunk.train(); pred.train()
        for b0 in range(0, len(perm), C["batch"]):
            tr = np.sort(perm[b0:b0 + C["batch"]])
            (po, pm, pg), (h_op, h_mch, _) = predict(trunk, pred, Dt, tr, Dt["tr_action"][tr], dev)
            yo, ym, yg = targets(tgt, Dt, tr, dev)
            lp = ((po - yo) ** 2).mean() + ((pm - ym) ** 2).mean() + ((pg - yg) ** 2).mean()
            X = torch.cat([h_op, h_mch], 1).reshape(-1, h_op.shape[-1])
            lv = torch.relu(floor - X.std(0)).mean()
            loss = lp + C["var_weight"] * lv
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p, q in zip(tgt.parameters(), trunk.parameters()):
                    p.mul_(C["ema"]).add_(q.detach(), alpha=1 - C["ema"])
            tot["pred"] += float(lp) * len(tr); tot["var"] += float(lv) * len(tr); tot["n"] += len(tr)
        trunk.eval()
        st = collapse_stats(tgt, Dt, mon, dev)
        collapsed = st["effective_rank"] < C["collapse_rank_share"] * st0["effective_rank"] or \
            st["mean_std"] < C["collapse_std_share"] * st0["mean_std"]
        hist.append({"epoch": ep, "pred_loss": tot["pred"] / tot["n"], "var_loss": tot["var"] / tot["n"], **st,
                     "collapsed": bool(collapsed), "seconds": time.time() - t0})
        log(f"epoch {ep}: pred {hist[-1]['pred_loss']:.5f} var {hist[-1]['var_loss']:.5f} "
            f"rank {st['effective_rank']:.2f} std {st['mean_std']:.4f}{' COLLAPSED' if collapsed else ''}")
    CK.mkdir(parents=True, exist_ok=True)
    meta = {"collection_sha256": sha(DATA / "collection.npz"), "settings": C}
    bundle = W.trunk_bundle(trunk.state_dict(), cfg, meta, C["epochs"], hist[-1]["pred_loss"])
    spec = torch.load(SPEC_START, map_location="cpu", weights_only=False)["state_dict"]
    fe = {k: v.shape for k, v in spec.items() if k.startswith("feature_exact.")}
    mine = {k: v.shape for k, v in bundle["state_dict"].items() if k.startswith("feature_exact.")}
    assert fe == mine, "encoder keys or shapes differ from the specialist start bundle"
    torch.save(bundle, CK / "encoder_bundle.pth")
    torch.save(tgt.state_dict(), CK / "target_encoder.pth"); torch.save(pred.state_dict(), CK / "predictor.pth")
    torch.save(trunk.state_dict(), CK / "online_encoder.pth")
    (CK / "SHA256SUMS.json").write_text(json.dumps({p.name: sha(p) for p in sorted(CK.glob("*.pth"))}, indent=1) + "\n")
    (RES / "train_log.json").write_text(json.dumps({"history": hist, "variance_floor": floor,
                                                    "train_transitions": int(len(tr_rows)),
                                                    "train_seconds": time.time() - t0,
                                                    "encoder_matches_specialist_keys": True}, indent=1) + "\n")
    log(f"train done: {time.time() - t0:.0f} s")


# ------------------------------------------------------------------------------------------------ evaluation

def evaluate():
    cfg, dev = setup()
    Dt = load_data()
    _, ho = split_rows(Dt)
    J, M = B.SIZES[W.SIZE]
    t0 = time.time()
    trunk = DualAttentionNetwork(configs).to(dev); trunk.load_state_dict(torch.load(CK / "online_encoder.pth", map_location=dev))
    tgt = DualAttentionNetwork(configs).to(dev); tgt.load_state_dict(torch.load(CK / "target_encoder.pth", map_location=dev))
    d = configs.layer_fea_output_dim[-1]
    pred = Predictor(d, Dt["s_fea_pairs"].shape[-1], C["pred_hidden"], C["pred_heads"]).to(dev)
    pred.load_state_dict(torch.load(CK / "predictor.pth", map_location=dev))
    trunk.eval(); tgt.eval(); pred.eval()
    # shuffled action: the next branch of the same state (cyclic, a derangement)
    st = Dt["tr_state"]
    shuf = np.empty(len(st), int)
    for s in np.unique(st[ho]):
        k = np.flatnonzero(st == s)
        shuf[k] = Dt["tr_action"][np.roll(k, -1)]
    # changed tokens (simulator's physical arrays)
    ch_op = (np.abs(Dt["p1_op_ct_lb"] - Dt["p0_op_ct_lb"]) > 1e-9) | (Dt["p1_op_scheduled_flag"] != Dt["p0_op_scheduled_flag"])
    ch_m = (np.abs(Dt["p1_mch_free_time"] - Dt["p0_mch_free_time"]) > 1e-9) | (Dt["p1_mch_working_flag"] != Dt["p0_mch_working_flag"])
    E = {k: [] for k in ("model", "copy", "shuffle")}
    Z1, ZH = [], []
    with torch.no_grad():
        for b0 in range(0, len(ho), 512):
            tr = ho[b0:b0 + 512]
            (po, pm, _), _ = predict(trunk, pred, Dt, tr, Dt["tr_action"][tr], dev)
            (so, sm, _), _ = predict(trunk, pred, Dt, tr, shuf[tr], dev)
            yo, ym, _ = targets(tgt, Dt, tr, dev)
            zo, zm, _ = targets(tgt, Dt, tr, dev, pre="s_")
            y = torch.cat([yo, ym], 1)
            for k, p in (("model", torch.cat([po, pm], 1)), ("copy", torch.cat([zo, zm], 1)),
                         ("shuffle", torch.cat([so, sm], 1))):
                E[k].append(((p - y) ** 2).mean(-1).cpu().numpy())                       # [b, N+M] per-token MSE
            Z1.append(y.cpu().numpy()); ZH.append(torch.cat([po, pm], 1).cpu().numpy())
    E = {k: np.concatenate(v) for k, v in E.items()}
    Z1, ZH = np.concatenate(Z1), np.concatenate(ZH)
    N = Dt["s_fea_j"].shape[1]
    chg = np.concatenate([ch_op[ho], ch_m[ho]], 1)
    src = Dt["source"][st[ho]]

    def summ(sel_rows, tok):
        r = {}
        for k in E:
            r[f"mse_{k}"] = float(E[k][sel_rows][tok[sel_rows]].mean())
        r["change_explained"] = 1 - r["mse_model"] / r["mse_copy"]
        return r

    allr = np.arange(len(ho))
    ones = np.ones_like(chg)
    groups = {"all_tokens": ones, "changed_tokens": chg, "unchanged_tokens": ~chg,
              "changed_operations": np.concatenate([ch_op[ho], np.zeros_like(ch_m[ho])], 1),
              "changed_machines": np.concatenate([np.zeros_like(ch_op[ho]), ch_m[ho]], 1)}
    R = {"heldout_transitions": int(len(ho)), "heldout_states": int(len(np.unique(st[ho]))),
         "changed_tokens_per_transition": {"operations": float(ch_op[ho].sum(1).mean()),
                                           "machines": float(ch_m[ho].sum(1).mean())},
         "errors": {g: summ(allr, t) for g, t in groups.items()}}
    R["by_source"] = {SOURCES[i]: {g: summ(np.flatnonzero(src == i), t) for g, t in
                                   (("all_tokens", ones), ("changed_tokens", chg))} for i in range(len(SOURCES))}
    # shuffle: per state, changed-token error with shuffled vs true actions
    st_ho = st[ho]
    worse = []
    for s in np.unique(st_ho):
        k = np.flatnonzero(st_ho == s)
        m_true = np.mean([E["model"][i][chg[i]].mean() for i in k])
        m_sh = np.mean([E["shuffle"][i][chg[i]].mean() for i in k])
        worse.append(m_sh > m_true)
    R["shuffle_worse_share_of_states"] = float(np.mean(worse))
    # pairwise discrimination between branches of the same state
    a = Dt["tr_action"][ho]; later = Dt["tr_later"][ho]
    pos = {g: i for i, g in enumerate(ho)}
    pairs = {"same_operation_other_machine": [], "immediate_vs_later": [], "other": []}
    dtrue, dpred = {k: [] for k in pairs}, {k: [] for k in pairs}
    for s in np.unique(st_ho):
        k = np.flatnonzero(st_ho == s)
        for x in range(len(k)):
            for y in range(x + 1, len(k)):
                i, j = k[x], k[y]
                dt = np.sqrt(((Z1[i] - Z1[j]) ** 2).mean())
                if dt <= C["eval_tol"]:
                    continue
                typ = ("same_operation_other_machine" if a[i] // M == a[j] // M else
                       "immediate_vs_later" if later[i] != later[j] else "other")
                ok_i = ((ZH[i] - Z1[i]) ** 2).mean() < ((ZH[i] - Z1[j]) ** 2).mean()
                ok_j = ((ZH[j] - Z1[j]) ** 2).mean() < ((ZH[j] - Z1[i]) ** 2).mean()
                pairs[typ] += [ok_i, ok_j]
                dtrue[typ].append(dt); dpred[typ].append(np.sqrt(((ZH[i] - ZH[j]) ** 2).mean()))

    def spearman(x, y):
        if len(x) < 3:
            return None
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    allp = sum(pairs.values(), [])
    R["discrimination"] = {"overall": {"accuracy": float(np.mean(allp)), "ordered_pairs": len(allp),
                                       "spearman_pred_vs_true_branch_distance":
                                           spearman(sum(dtrue.values(), []), sum(dpred.values(), []))},
                           **{k: {"accuracy": float(np.mean(v)) if v else None, "ordered_pairs": len(v),
                                  "unordered_pairs": len(v) // 2,
                                  "spearman_pred_vs_true_branch_distance": spearman(dtrue[k], dpred[k]),
                                  "median_true_branch_distance": float(np.median(dtrue[k])) if dtrue[k] else None,
                                  "median_pred_branch_distance": float(np.median(dpred[k])) if dpred[k] else None}
                              for k, v in pairs.items()}}
    # decision
    disc_types_ok = all(v["accuracy"] >= RULE["discrimination_per_type_min"]
                        for k, v in R["discrimination"].items() if k != "overall"
                        and v["unordered_pairs"] >= RULE["per_type_min_pairs"])
    R["decision"] = {"change_explained_changed_tokens": R["errors"]["changed_tokens"]["change_explained"],
                     "shuffle_worse_share": R["shuffle_worse_share_of_states"],
                     "discrimination_overall": R["discrimination"]["overall"]["accuracy"],
                     "discrimination_each_type_ok": disc_types_ok}
    R["decision"]["responds_reliably"] = bool(
        R["decision"]["change_explained_changed_tokens"] >= RULE["change_explained_min"]
        and R["decision"]["shuffle_worse_share"] >= RULE["shuffle_worse_share_min"]
        and R["decision"]["discrimination_overall"] >= RULE["discrimination_min"] and disc_types_ok)
    # examples: one held-out state per pair type (first with 3+ branches where possible), deterministic
    ex = []
    used = set()
    for want in ("same_operation_other_machine", "immediate_vs_later", "other", "any"):
        for s in np.unique(st_ho):
            k = np.flatnonzero(st_ho == s)
            if s in used or len(k) < 2:
                continue
            types = {("same_operation_other_machine" if a[i] // M == a[j] // M else
                      "immediate_vs_later" if later[i] != later[j] else "other")
                     for x, i in enumerate(k) for j in k[x + 1:]}
            if want != "any" and want not in types:
                continue
            used.add(s)
            dm = [[float(np.sqrt(((ZH[i] - Z1[j]) ** 2).mean())) for j in k] for i in k]
            tm = [[float(np.sqrt(((Z1[i] - Z1[j]) ** 2).mean())) for j in k] for i in k]
            ex.append({"state": int(s), "source": SOURCES[int(Dt["source"][s])], "for_type": want,
                       "branches": [{"job": int(a[i] // M), "machine": int(a[i] % M),
                                     "start": "later" if later[i] else "immediate",
                                     "changed_operations": int(ch_op[ho][i].sum()),
                                     "changed_machines": int(ch_m[ho][i].sum()),
                                     "mse_model_changed": float(E["model"][i][chg[i]].mean()),
                                     "mse_copy_changed": float(E["copy"][i][chg[i]].mean())} for i in k],
                       "true_distance_between_next_states": tm,
                       "pred_to_true_distance": dm,
                       "diagonal_smallest_in_every_row": bool(all(np.argmin(r) == x for x, r in enumerate(dm)))})
            break
    (RES / "examples.json").write_text(json.dumps(ex, indent=1) + "\n")
    R["eval_seconds"] = time.time() - t0
    (RES / "heldout.json").write_text(json.dumps(R, indent=1) + "\n")
    plot()
    log(f"evaluate: {json.dumps(R['decision'])}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = json.loads((RES / "train_log.json").read_text())["history"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    ax[0].plot([e["epoch"] for e in h[1:]], [e["pred_loss"] for e in h[1:]], color="#1f77b4", marker="o", ms=3)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("next-state prediction loss (MSE)"); ax[0].set_yscale("log")
    ax[1].plot([e["epoch"] for e in h], [e["effective_rank"] for e in h], color="#d62728", marker="o", ms=3)
    ax[1].axhline(C["collapse_rank_share"] * h[0]["effective_rank"], color="#555555", ls="--", lw=.9,
                  label="collapse threshold")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("target-token effective rank"); ax[1].legend(frameon=False)
    for a_ in ax:
        a_.grid(alpha=.3)
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)
    fig.suptitle("Latent next-state pilot (10x5, waiting allowed)", fontsize=11)
    fig.tight_layout(); fig.savefig(RES / "curves.png", dpi=140, facecolor="white"); plt.close(fig)


def main():
    settings()
    if ACTION in ("collect", "all"):
        collect()
    if ACTION in ("train", "all"):
        train()
    if ACTION in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
