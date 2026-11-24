"""Joint latent-state + measurable-outcome world-model pilot (10x5, waiting allowed: semi-active simulator).

    S_t -> DANIEL encoder -> action-conditioned latent predictor -> predicted latent of S_t+1
        -> outcome head (+ continuation rule one-hot) -> final job completion, machine busy time, machine energy

Configurations (same DANIEL encoder, same seed-111 random start trunk, same data, splits and budgets):
  A  random untrained encoder (frozen); for latent metrics a predictor is trained on it (target = itself, frozen)
  B  direct outcome prediction (PairWorldModel: encoder + action readout + pair features), abs + diff Huber
  C  latent next-state prediction only (EMA target encoder, stop-gradient, variance hinge)
  D  joint: C + an outcome head that reads ONLY the predicted next latent; both losses train predictor and encoder

Stages (resumable; JP_SMOKE=1 runs a tiny version into *_smoke folders):
  collect   fresh generator factories (seed 4111), split by factory 70/15/15 (train / validation / test); five behaviour
            sources; at sampled states up to 5 distinct legal branch actions, each simulated one step; final outcomes
            of every branch under 3 continuation rules (W.Decoder, never waits) from S_t+1
  lrcheck   B, C, D at 1e-4, 3e-4, 1e-3 for a short budget; choice per config on the validation factories
  train     B, C, D (and A's latent predictor) at the chosen lr, same epoch cap; validation monitoring every few epochs
  probe     frozen-encoder outcome probes (A, B, C, D, raw features): identical fresh PairWorldModel heads
  evaluate  test factories: latent prediction, outcome prediction, action ranking, availability readouts, collapse
  report    results/world_model_joint_pilot_v3.md

    python tools/world_model_joint_pilot_v3.py all
"""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, os, resource, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
_ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
import world_model_latent_pilot_v3 as L                             # noqa: E402
import world_model_m1_action_pilot as AP                            # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
from common_utils import setup_seed                                 # noqa: E402
from model.main_model import DualAttentionNetwork                  # noqa: E402

W, B, WM, D, configs = L.W, L.B, L.WM, L.D, L.configs
SMOKE = os.environ.get("JP_SMOKE") == "1"
SFX = "_smoke" if SMOKE else ""
NAME = "world_model_joint_pilot_v3"
DATA, RES, CK = (REPO / "data" / f"{NAME}{SFX}", REPO / "results" / f"{NAME}{SFX}", REPO / "checkpoints" / f"{NAME}{SFX}")
SOURCES, RULES = L.SOURCES, W.CONT
C = {"instances": 20 if SMOKE else 1350, "seed": 4111, "split_mod": 20, "train_below": 14, "val_below": 17,
     "states_per_episode": 8, "max_branches": 5, "chunk_instances": 270,
     "lrs": [3e-4] if SMOKE else [1e-4, 3e-4, 1e-3], "lr_epochs": 1 if SMOKE else 10, "epochs": 2 if SMOKE else 60,
     "eval_every": 1 if SMOKE else 5, "batch_states": 56, "train_seed": 111, "ema": 0.996, "var_weight": 1.0,
     "var_floor_share_of_initial_std": 0.5, "pred_hidden": 64, "pred_heads": 4, "head_hidden": 64, "diff_weight": 1.0,
     "norm_transitions": 1024, "grad_ratio_band": [1 / 3, 3.0], "grad_monitor_batches": 5,
     "probe_epochs": 1 if SMOKE else 10, "monitor_val_states": 150 if SMOKE else 2000,
     "avail_fit_transitions": 20000, "collapse_rank_share": 0.25, "collapse_std_share": 0.10,
     "converge_window_epochs": 1 if SMOKE else 20, "converge_rel_tol": 0.01, "eval_tol": 1e-6}
RULE = {"probe_margin_over_B_and_C": 0.03, "latent_tolerance_vs_C": 0.02}
RULE_TEXT = ("D justifies specialist training only if, on the TEST factories: (1) in the frozen-encoder probes D's mean "
             "ranking accuracy AND mean same-operation accuracy each beat both B and C by >= 0.03, and D's mean "
             "difference MAE is below the zero-difference baseline; (2) D's latent change explained (changed tokens) and "
             "overall discrimination are within 0.02 of C's or better; (3) D's availability readouts (absolute R^2 and "
             "same-operation difference sign accuracy) are at least C's; (4) D did not collapse.")
ENC_IN = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx")
F32 = ("fea_j", "fea_m", "fea_pairs")
U8 = ("op_mask", "mch_mask", "comp_idx", "dynamic_pair_mask")
LOGF = None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(msg):
    global LOGF
    RES.mkdir(parents=True, exist_ok=True)
    LOGF = LOGF or open(RES / "run.log", "a")
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def settings():
    f = RES / "settings.json"
    if f.exists():
        return json.loads(f.read_text())
    s = {"constants": C, "decision_rule": RULE, "decision_rule_text": RULE_TEXT, "sources": SOURCES,
         "continuation_rules": RULES, "dispatch_rule": "semi_active", "seed": 111,
         "inputs": {p.name: {"path": str(p.relative_to(REPO)), "sha256": sha(p)} for p in (L.START, L.POLICY, L.SPEC_START)},
         "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    jdump(f, s)
    (RES / "settings.sha256").write_text(sha(f) + "\n")
    return s


# ================================================================================================ 1. collection

def choose_branches(env, e, a0, rng):
    """Up to 5 distinct legal actions: behaviour; same operation, other machine; other operation, same machine;
    an immediate and a waiting (later-starting) assignment; random fill."""
    M = env.number_of_machines
    legal = ~env.dynamic_pair_mask[e].astype(bool)
    P = env.pair_free_time[e]
    if legal.sum() < 2:
        return None
    earliest = P[legal].min()
    imm = legal & (P <= earliest + 1e-9); late = legal & ~imm
    chosen = [int(a0)]
    j0, m0 = divmod(int(a0), M)

    def add(cands):
        c = [int(a) for a in cands if int(a) not in chosen]
        if c and len(chosen) < C["max_branches"]:
            chosen.append(int(rng.choice(c)))
    add([j0 * M + m for m in np.flatnonzero(legal[j0]) if m != m0])
    add([j * M + m0 for j in np.flatnonzero(legal[:, m0]) if j != j0])
    for cls in (imm, late):
        if not cls.reshape(-1)[chosen].any():
            add(np.flatnonzero(cls.reshape(-1)))
    rest = [int(a) for a in np.flatnonzero(legal.reshape(-1)) if int(a) not in chosen]
    while len(chosen) < C["max_branches"] and rest:
        a = int(rng.choice(rest)); chosen.append(a); rest.remove(a)
    start = P.reshape(-1)[chosen]
    return chosen, (start > earliest + 1e-9).astype(np.uint8)


def decoder_check(data, idx, out, n, rng):
    """The continuation decoder, replayed through prefix + branch action, sits exactly at the stored S_t+1 (machine
    free times and scheduled operations), including waiting actions."""
    bad = 0
    for k in rng.choice(len(out["tr_state"]), size=min(n, len(out["tr_state"])), replace=False):
        s = out["tr_state"][k]; e = out["state_episode"][s]; t = out["state_step"][s]
        d = W.Decoder(data, [idx[e]]); d.reset()
        for u in range(t):
            d.step(np.array([out["actions"][e, u]]))
        d.step(np.array([out["tr_action"][k]]))
        ok = np.allclose(d.F[0], out["p1_mch_free_time"][k], atol=1e-9) and \
            np.array_equal(d.sched[0], out["p1_op_scheduled_flag"][k].astype(bool))
        bad += not ok
    return {"checked": int(min(n, len(out["tr_state"]))), "mismatches": int(bad)}


def compact(out):
    for k in list(out):
        base = k[2:] if k[:2] in ("s_", "n_") else None
        if base in F32:
            out[k] = out[k].astype(np.float32)
        elif base in U8:
            assert np.isin(out[k], (0, 1)).all(), k
            out[k] = out[k].astype(np.uint8)
        elif base in ("candidate", "job_first_op_id", "job_last_op_id"):
            out[k] = out[k].astype(np.int16)
        elif k.startswith(("p0_", "p1_")):
            out[k] = out[k].astype(np.float32)
    for k in ("n_fea_pairs", "n_dynamic_pair_mask"):              # never model inputs for the next state
        out.pop(k, None)
    return out


def collect():
    cfg, dev = L.setup()
    if (DATA / "collection.npz").exists():
        return
    settings()
    L.choose_branches = choose_branches
    L.C["max_branches"] = C["max_branches"]; L.C["states_per_episode"] = C["states_per_episode"]
    from benchmark_v3_generator import GeneratorSource
    setup_seed(C["seed"])
    gen = GeneratorSource(configs, W.SIZE, B.load_manifest())
    t0 = time.time()
    data = gen.sample(C["instances"])
    n = len(data)
    rem = np.arange(n) % C["split_mod"]
    split = np.where(rem < C["train_below"], 0, np.where(rem < C["val_below"], 1, 2))        # 0 train, 1 val, 2 test
    net = L.load_policy(dev)
    rng = np.random.default_rng(C["seed"])
    J, M = B.SIZES[W.SIZE]
    parts, checks = [], {}
    t_episodes = t_cont = 0.0
    ep_off = 0
    for si, src in enumerate(SOURCES):
        for c0 in range(0, n, C["chunk_instances"]):
            idx = np.arange(c0, min(n, c0 + C["chunk_instances"]))
            ts = time.time()
            out = L.run_source(src, data, idx, rng, net, dev)
            rc = L.replay_check(data, idx, out, 5, rng); dc = decoder_check(data, idx, out, 5, rng)
            if rc["mismatches"] or dc["mismatches"]:
                raise SystemExit(f"{src}: replay {rc} decoder {dc}")
            checks.setdefault(src, []).append({"replay": rc, "decoder_matches_environment": dc})
            S = len(out["state_episode"])
            cand = np.full((S, C["max_branches"]), -1, np.int64); slot = np.zeros(len(out["tr_state"]), np.int64)
            fill = np.zeros(S, int)
            for k, s in enumerate(out["tr_state"]):
                cand[s, fill[s]] = out["tr_action"][k]; slot[k] = fill[s]; fill[s] += 1
            t_episodes += time.time() - ts
            ts = time.time()
            tj, tm, _, _ = W.continuations(data, idx, out["actions"], {"episode": out["state_episode"],
                                                                       "step": out["state_step"], "cand": cand},
                                           RULES, False)
            t_cont += time.time() - ts
            st = out["tr_state"]
            out["tj"] = tj[st, slot].astype(np.float32); out["tm"] = tm[st, slot].astype(np.float32)
            assert np.isfinite(out["tj"]).all() and np.isfinite(out["tm"]).all()
            inst = idx[out["state_episode"]]
            ep = out["tr_state"]
            o, m_ = np.divmod(out["tr_action"], M)
            op = out["s_candidate"][ep, o]
            out["tr_pt"] = np.array([data.op_pt[inst[s]][q, mm] for s, q, mm in zip(ep, op, m_)], np.float32)
            out["tr_carbon"] = np.array([data.carbon[inst[s]][q, mm] for s, q, mm in zip(ep, op, m_)], np.float32)
            out["tr_slot"] = slot; out["cand"] = cand
            out["state_inst"] = inst; out["state_source"] = np.full(S, si)
            N = out["actions"].shape[1]
            out["state_progress"] = np.minimum(2, (out["state_step"] * 3) // max(1, N - 1))
            out["state_due"] = np.stack([np.asarray(data.due[i], np.float32) for i in inst])
            out["state_ep"] = out["state_episode"] + ep_off
            out["ep_actions"] = out["actions"].astype(np.int16); out["ep_inst"] = idx; out["ep_source"] = np.full(len(idx), si)
            out["beh_later_share"] = np.array([out["beh_later"].mean()])
            ep_off += len(idx)
            for k in ("actions", "prefs", "beh_later", "state_episode"):
                out.pop(k, None)
            parts.append(compact(out))
            log(f"collect {src} instances {c0}-{idx[-1]}: {S} states, {len(st)} transitions")
    # concatenate with state offsets
    allp, s_off = {}, 0
    for p in parts:
        p["tr_state"] = p["tr_state"] + s_off; s_off += len(p["state_inst"])
    for k in parts[0]:
        allp[k] = np.concatenate([p[k] for p in parts])
    S = len(allp["state_inst"])
    tix = np.full((S, C["max_branches"]), -1, np.int64)
    tix[allp["tr_state"], allp["tr_slot"]] = np.arange(len(allp["tr_state"]))
    allp["tix"] = tix
    allp["state_split"] = split[allp["state_inst"]]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not set(allp["state_inst"][allp["state_split"] == a]) & set(allp["state_inst"][allp["state_split"] == b])
    allp["factory"] = np.asarray([data.keys[i] for i in range(n)])
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / "collection.npz", **allp)
    jdump(RES / "coverage.json", coverage(allp, split, checks, t_episodes, t_cont, time.time() - t0, gen))
    log(f"collection done: {S} states, {len(allp['tr_state'])} transitions, "
        f"{(DATA / 'collection.npz').stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f} s")


def pair_types(cand, M):
    """Per valid (state, slot): has a sibling with the same operation / with another operation on the same machine."""
    valid = cand >= 0
    job, mch = cand // M, cand % M
    same_op = np.zeros_like(valid); same_m = np.zeros_like(valid)
    A = cand.shape[1]
    for i in range(A):
        for j in range(A):
            if i == j:
                continue
            both = valid[:, i] & valid[:, j]
            same_op[:, i] |= both & (job[:, i] == job[:, j])
            same_m[:, i] |= both & (mch[:, i] == mch[:, j]) & (job[:, i] != job[:, j])
    return same_op, same_m


def coverage(Dt, split, checks, t_ep, t_cont, total, gen):
    M = B.SIZES[W.SIZE][1]
    so, sm = pair_types(Dt["cand"], M)
    valid = Dt["cand"] >= 0
    later = np.zeros_like(valid); later[Dt["tr_state"], Dt["tr_slot"]] = Dt["tr_later"].astype(bool)
    out = {"factories": {n_: int((split == i).sum()) for i, n_ in enumerate(("train", "validation", "test"))},
           "per_split_source": {}}
    for i, sp in enumerate(("train", "validation", "test")):
        for si, src in enumerate(SOURCES):
            m = (Dt["state_split"] == i) & (Dt["state_source"] == si)
            v = valid[m]
            out["per_split_source"][f"{sp}|{src}"] = {
                "states": int(m.sum()), "transitions": int(v.sum()), "branches_per_state": float(v.sum() / max(1, m.sum())),
                "share_in_same_operation_group": float(so[m][v].mean()) if v.any() else None,
                "share_in_other_operation_same_machine_group": float(sm[m][v].mean()) if v.any() else None,
                "share_waiting": float(later[m][v].mean()) if v.any() else None,
                "share_states_with_immediate_and_waiting": float(np.mean([later[s][valid[s]].any() and
                                                                          (~later[s][valid[s]]).any()
                                                                          for s in np.flatnonzero(m)])) if m.any() else None,
                "progress_share_early_mid_late": [float((Dt["state_progress"][m] == q).mean()) for q in range(3)]}
    return {**out, "totals": {"states": int(len(Dt["state_inst"])), "transitions": int(len(Dt["tr_state"]))},
            "behaviour_waiting_share_per_chunk": Dt["beh_later_share"].tolist(),
            "checks": checks, "branch_main_episode_untouched": True,
            "seconds": {"episodes_and_branching": t_ep, "continuation_rollouts": t_cont, "total_with_instance_generation": total},
            "bytes": int((DATA / "collection.npz").stat().st_size), "peak_rss_gb": peak_rss_gb(),
            "generator": gen.provenance()}


# ================================================================================================ 2. models

def load_data():
    return dict(np.load(DATA / "collection.npz", allow_pickle=False))


def prepare(Dt, cfg):
    """State-level normalised outcome targets [S, A, R, J] / [S, A, R, M, 2] (padding slots 0)."""
    tj, tm = W.normalise(cfg, Dt["tj"], Dt["tm"])
    S, A = Dt["cand"].shape
    TJ = np.zeros((S, A) + tj.shape[1:], np.float32); TM = np.zeros((S, A) + tm.shape[1:], np.float32)
    TJ[Dt["tr_state"], Dt["tr_slot"]] = tj; TM[Dt["tr_state"], Dt["tr_slot"]] = tm
    Dt["TJ"], Dt["TM"] = TJ, TM
    rj = np.zeros((S, A) + Dt["tj"].shape[1:], np.float32); rm = np.zeros((S, A) + Dt["tm"].shape[1:], np.float32)
    rj[Dt["tr_state"], Dt["tr_slot"]] = Dt["tj"]; rm[Dt["tr_state"], Dt["tr_slot"]] = Dt["tm"]
    Dt["RJ"], Dt["RM"] = rj, rm
    return Dt


def splits(Dt):
    return {n_: np.flatnonzero(Dt["state_split"] == i) for i, n_ in enumerate(("train", "validation", "test"))}


def start_trunk_sd():
    sd = torch.load(L.START, map_location="cpu", weights_only=False)["state_dict"]
    return {k[len("feature_exact."):]: v for k, v in sd.items() if k.startswith("feature_exact.")}


class OutcomeHead(nn.Module):
    """Reads ONLY the predicted next latent (operation, machine, global tokens) + the continuation rule. Job membership
    (static op -> job map of the instance) only selects which predicted operation tokens are pooled for each job."""

    def __init__(self, d, h):
        super().__init__()
        R = len(RULES)
        self.job = WM.mlp(4 * d + R, h, 1)
        self.mch = WM.mlp(3 * d + R, h, 2)

    def forward(self, po, pm, pg, first, last):
        T_, N, d = po.shape
        J, M, R = first.shape[1], pm.shape[1], len(RULES)
        opi = torch.arange(N, device=po.device)
        mem = (opi[None, None, :] >= first[:, :, None]) & (opi[None, None, :] <= last[:, :, None])      # [T, J, N]
        mf = mem.float()
        mean = (mf @ po) / mf.sum(-1, keepdim=True).clamp(min=1)
        mx = po[:, None].masked_fill(~mem[..., None], -1e9).max(2).values
        c = torch.eye(R, device=po.device)
        xj = torch.cat([mean, mx, pg[:, None].expand(-1, J, -1)], -1)
        xj = torch.cat([xj[:, None].expand(-1, R, -1, -1), c[None, :, None].expand(T_, R, J, R)], -1)
        xm = torch.cat([pm, pg[:, None].expand(-1, M, -1)], -1)
        xm = torch.cat([xm[:, None].expand(-1, R, -1, -1), c[None, :, None].expand(T_, R, M, R)], -1)
        return self.job(xj).squeeze(-1), self.mch(xm)                                              # [T,R,J], [T,R,M,2]


class RawPairModel(nn.Module):
    """The PairWorldModel heads on raw scheduling features (no encoder): operation, machine and pair features."""

    def __init__(self, n_op, n_m, n_pair, h):
        super().__init__()
        R = len(RULES)
        a = 2 * n_op + 2 * n_m + n_pair
        self.job_head = WM.mlp(n_op + a + R, h, 1)
        self.mch_head = WM.mlp(n_m + a + R, h, 2)

    def forward(self, s, job, mch, pf):
        h_op, h_mch = s["fea_j"], s["fea_m"]
        Bn, N, do = h_op.shape; M, dm = h_mch.shape[1:]
        cand = s["candidate"]; J = cand.shape[1]; A = job.shape[1]; R = len(RULES)
        h_job = torch.gather(h_op, 1, cand.unsqueeze(-1).expand(-1, -1, do))
        a_op = torch.gather(h_job, 1, job.clamp(min=0).unsqueeze(-1).expand(-1, -1, do))
        a_m = torch.gather(h_mch, 1, mch.clamp(min=0).unsqueeze(-1).expand(-1, -1, dm))
        a = torch.cat([a_op, a_m, h_op.mean(1)[:, None].expand(-1, A, -1), h_mch.mean(1)[:, None].expand(-1, A, -1), pf], -1)
        c = torch.eye(R, device=a.device)
        aa, cc = a[:, :, None, None, :], c[None, None, :, None, :]
        xj = torch.cat([h_job[:, None, None].expand(Bn, A, R, J, do), aa.expand(Bn, A, R, J, a.shape[-1]),
                        cc.expand(Bn, A, R, J, R)], -1)
        xm = torch.cat([h_mch[:, None, None].expand(Bn, A, R, M, dm), aa.expand(Bn, A, R, M, a.shape[-1]),
                        cc.expand(Bn, A, R, M, R)], -1)
        return self.job_head(xj).squeeze(-1), self.mch_head(xm)


def batch(Dt, sel, dev):
    s = L.tensors(Dt, "s_", sel, dev)
    cand = Dt["cand"][sel]
    valid = cand >= 0
    srow, slot = np.nonzero(valid)
    return s, cand, valid, srow, slot, Dt["tix"][sel][valid]


def pair_feats(Dt, sel, cand, dev):
    M = B.SIZES[W.SIZE][1]
    j, m = np.where(cand >= 0, cand // M, 0), np.where(cand >= 0, cand % M, 0)
    fp = Dt["s_fea_pairs"][sel[:, None], j, m] * (cand >= 0)[..., None]
    return torch.as_tensor(fp, dtype=torch.float32, device=dev)


def latent_pred(trunk, pred, Dt, sel, s, srow, acts, dev):
    """The latent pilot's predictor input (L.predict), with the state encoded once per state."""
    M = B.SIZES[W.SIZE][1]
    h_op, h_mch, g = L.encode(trunk, s)
    sr = torch.as_tensor(srow, device=dev)
    Ho, Hm, G = h_op[sr], h_mch[sr], g[sr]
    job_np, mch_np = acts // M, acts % M
    job = torch.as_tensor(job_np, device=dev); mch = torch.as_tensor(mch_np, device=dev)
    ar = torch.arange(len(acts), device=dev)
    op = s["candidate"][sr, job]
    a_op, a_m = Ho[ar, op], Hm[ar, mch]
    st = sel[srow]
    pair = torch.as_tensor(Dt["s_fea_pairs"][st, job_np, mch_np], dtype=torch.float32, device=dev)
    N = Ho.shape[1]
    fo = torch.zeros(len(acts), N, 3, device=dev); fo[ar, op, 0] = 1
    first = torch.as_tensor(Dt["s_job_first_op_id"][st, job_np].astype(np.int64), device=dev)
    last = torch.as_tensor(Dt["s_job_last_op_id"][st, job_np].astype(np.int64), device=dev)
    opi = torch.arange(N, device=dev)[None]
    fo[..., 1] = ((opi >= first[:, None]) & (opi <= last[:, None])).float()
    fm = torch.zeros(len(acts), M, 3, device=dev); fm[ar, mch, 2] = 1
    return pred(Ho, Hm, G, a_op, a_m, pair, fo, fm), (h_op, h_mch, g)


def head_out(head, Dt, sel, srow, slot, po, pm, pg, b, dev):
    st = sel[srow]
    first = torch.as_tensor(Dt["s_job_first_op_id"][st].astype(np.int64), device=dev)
    last = torch.as_tensor(Dt["s_job_last_op_id"][st].astype(np.int64), device=dev)
    oj, om = head(po, pm, pg, first, last)
    r, c = torch.as_tensor(srow, device=dev), torch.as_tensor(slot, device=dev)
    A = C["max_branches"]
    pj = oj.new_zeros((b, A) + oj.shape[1:]); pj[r, c] = oj
    pmm = om.new_zeros((b, A) + om.shape[1:]); pmm[r, c] = om
    return pj, pmm


def out_targets(Dt, sel, dev):
    return torch.as_tensor(Dt["TJ"][sel], device=dev), torch.as_tensor(Dt["TM"][sel], device=dev)


def out_loss(pred_, tgt, valid):
    return WM.loss(pred_, tgt, valid) + C["diff_weight"] * AP.diff_loss(pred_, tgt, valid)


def lat_loss(p, y):
    return sum(((a - b) ** 2).mean() for a, b in zip(p, y))


class Model:
    """kind: B, C, D, A_latent, or probe:<encoder> / probe:raw."""

    def __init__(self, kind, dev, n_pair, trunk_sd=None):
        self.kind, self.dev = kind, dev
        d = configs.layer_fea_output_dim[-1]
        setup_seed(C["train_seed"])
        self.trunk = DualAttentionNetwork(configs).to(dev)
        self.trunk.load_state_dict(trunk_sd or start_trunk_sd())
        self.tgt = self.pred = self.head = self.out = None
        if kind == "B" or kind.startswith("probe:"):
            setup_seed(C["train_seed"])
            if kind == "probe:raw":
                self.out = RawPairModel(configs.fea_j_input_dim, configs.fea_m_input_dim, n_pair, C["head_hidden"]).to(dev)
            else:
                self.out = AP.PairWorldModel(configs, n_pair, C["head_hidden"]).to(dev)
                self.out.trunk.load_state_dict(trunk_sd or start_trunk_sd())
                self.trunk = self.out.trunk
        if kind in ("C", "D", "A_latent"):
            self.tgt = copy.deepcopy(self.trunk)
            for p in self.tgt.parameters():
                p.requires_grad_(False)
            setup_seed(C["train_seed"])
            self.pred = L.Predictor(d, n_pair, C["pred_hidden"], C["pred_heads"]).to(dev)
            if kind == "D":
                self.head = OutcomeHead(d, C["head_hidden"]).to(dev)
        if kind == "A_latent" or kind.startswith("probe:"):
            for p in self.trunk.parameters():
                p.requires_grad_(False)
            if kind == "A_latent":
                self.tgt = self.trunk                              # frozen random encoder, target = itself
        self.lam_out, self.L0 = 1.0, None

    def modules(self):
        return [m for m in (self.trunk, self.pred, self.head, self.out) if m is not None]

    def params(self):
        ps = []
        for m in self.modules():
            ps += [p for p in m.parameters() if p.requires_grad and all(p is not q for q in ps)]
        return ps

    def train_mode(self, on):
        for m in self.modules():
            m.train(on)
        if self.kind == "A_latent" or self.kind.startswith("probe:"):
            self.trunk.eval()

    def latent(self, Dt, sel, s, srow, slot, acts):
        return latent_pred(self.trunk, self.pred, Dt, sel, s, srow, acts, self.dev)

    def outcome(self, Dt, sel, s, cand, srow, slot, lat=None):
        """Normalised outcome predictions [b, A, R, J] / [b, A, R, M, 2]."""
        M = B.SIZES[W.SIZE][1]
        if self.kind == "D":
            (po, pm, pg), _ = lat if lat is not None else self.latent(Dt, sel, s, srow, slot, cand[srow, slot])
            return head_out(self.head, Dt, sel, srow, slot, po, pm, pg, len(sel), self.dev)
        job = torch.as_tensor(np.where(cand >= 0, cand // M, -1), device=self.dev)
        mch = torch.as_tensor(np.where(cand >= 0, cand % M, -1), device=self.dev)
        return self.out(s, job, mch, pair_feats(Dt, sel, cand, self.dev))

    def losses(self, Dt, sel):
        """Raw loss terms of one state batch: dict of latent / variance / outcome terms (unnormalised)."""
        s, cand, valid, srow, slot, tr = batch(Dt, sel, self.dev)
        out = {}
        lat = None
        if self.pred is not None:
            lat = self.latent(Dt, sel, s, srow, slot, cand[srow, slot])
            (po, pm, pg), (h_op, h_mch, _) = lat
            y = L.targets(self.tgt, Dt, tr, self.dev)
            out["lat"] = lat_loss((po, pm, pg), y)
            if self.kind != "A_latent":
                X = torch.cat([h_op, h_mch], 1).reshape(-1, h_op.shape[-1])
                out["var"] = torch.relu(self.floor - X.std(0)).mean()
        if self.kind in ("B", "D") or self.kind.startswith("probe:"):
            pj = self.outcome(Dt, sel, s, cand, srow, slot, lat)
            out["out"] = out_loss(pj, out_targets(Dt, sel, self.dev), torch.as_tensor(valid, device=self.dev))
        return out

    def total(self, t):
        if self.kind == "D":
            return t["lat"] / self.L0["lat"] + C["var_weight"] * t["var"] + self.lam_out * t["out"] / self.L0["out"]
        if self.kind == "C":
            return t["lat"] + C["var_weight"] * t["var"]
        if self.kind == "A_latent":
            return t["lat"]
        return t["out"]

    def ema(self):
        if self.kind in ("C", "D"):
            with torch.no_grad():
                for p, q in zip(self.tgt.parameters(), self.trunk.parameters()):
                    p.mul_(C["ema"]).add_(q.detach(), alpha=1 - C["ema"])


def grad_stats(m, Dt, batches):
    """D: gradient norms of the normalised latent and outcome terms on the encoder, ratio and cosine."""
    tp = [p for p in m.trunk.parameters() if p.requires_grad]
    rows = []
    for sel in batches:
        t = m.losses(Dt, sel)
        g1 = torch.autograd.grad(t["lat"] / m.L0["lat"], tp, retain_graph=True, allow_unused=True)
        g2 = torch.autograd.grad(m.lam_out * t["out"] / m.L0["out"], tp, allow_unused=True)
        v1 = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(g1, tp)])
        v2 = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(g2, tp)])
        rows.append({"latent_grad_norm": float(v1.norm()), "outcome_grad_norm": float(v2.norm()),
                     "ratio_latent_to_outcome": float(v1.norm() / v2.norm().clamp(min=1e-12)),
                     "cosine": float(torch.nn.functional.cosine_similarity(v1, v2, 0))})
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


# ================================================================================================ 3. evaluation pieces

def latent_eval(m, Dt, states, full=False):
    M = B.SIZES[W.SIZE][1]
    m.train_mode(False)
    N = Dt["s_fea_j"].shape[1]
    E = {"model": [], "copy": [], "shuffle": []}
    Z1, ZH, TR = [], [], []
    with torch.no_grad():
        for b0 in range(0, len(states), 128):
            sel = states[b0:b0 + 128]
            s, cand, valid, srow, slot, tr = batch(Dt, sel, m.dev)
            acts = cand[srow, slot]
            nxt = np.empty(len(srow), int)                                  # next branch of the same state (cyclic)
            for r in np.unique(srow):
                k = np.flatnonzero(srow == r); nxt[k] = np.roll(k, -1)
            (po, pm, _), _ = m.latent(Dt, sel, s, srow, slot, acts)
            (so, sm, _), _ = m.latent(Dt, sel, s, srow, slot, acts[nxt])
            yo, ym, _ = L.targets(m.tgt, Dt, tr, m.dev)
            zo, zm, _ = L.targets(m.tgt, Dt, tr, m.dev, pre="s_")
            y = torch.cat([yo, ym], 1)
            for k, p in (("model", torch.cat([po, pm], 1)), ("copy", torch.cat([zo, zm], 1)), ("shuffle", torch.cat([so, sm], 1))):
                E[k].append(((p - y) ** 2).mean(-1).cpu().numpy())
            Z1.append(y.reshape(len(tr), -1).cpu().numpy()); ZH.append(torch.cat([po, pm], 1).reshape(len(tr), -1).cpu().numpy())
            TR.append(tr)
    E = {k: np.concatenate(v) for k, v in E.items()}
    Z1, ZH, TR = np.concatenate(Z1), np.concatenate(ZH), np.concatenate(TR)
    ch_op = (np.abs(Dt["p1_op_ct_lb"][TR] - Dt["p0_op_ct_lb"][TR]) > 1e-9) | (Dt["p1_op_scheduled_flag"][TR] != Dt["p0_op_scheduled_flag"][TR])
    ch_m = (np.abs(Dt["p1_mch_free_time"][TR] - Dt["p0_mch_free_time"][TR]) > 1e-9) | (Dt["p1_mch_working_flag"][TR] != Dt["p0_mch_working_flag"][TR])
    chg = np.concatenate([ch_op, ch_m], 1)

    def summ(rows, tok):
        r = {f"mse_{k}": float(E[k][rows][tok[rows]].mean()) for k in E}
        r["change_explained"] = 1 - r["mse_model"] / r["mse_copy"]
        return r
    allr = np.arange(len(TR))
    R = {"transitions": int(len(TR)), "changed_tokens": summ(allr, chg)}
    st = Dt["tr_state"][TR]
    # pairwise discrimination by pair type
    a = Dt["tr_action"][TR]; later = Dt["tr_later"][TR]
    types = ("same_operation_other_machine", "other_operation_same_machine", "immediate_vs_waiting", "other")
    ok = {t: [] for t in types}; dtr = {t: [] for t in types}; dpr = {t: [] for t in types}
    order = np.argsort(st, kind="stable")
    bounds = np.flatnonzero(np.diff(st[order])) + 1
    for k in np.split(order, bounds):
        for x in range(len(k)):
            for y_ in range(x + 1, len(k)):
                i, j = k[x], k[y_]
                dt = np.sqrt(((Z1[i] - Z1[j]) ** 2).mean())
                if dt <= C["eval_tol"]:
                    continue
                t = ("same_operation_other_machine" if a[i] // M == a[j] // M else
                     "other_operation_same_machine" if a[i] % M == a[j] % M else
                     "immediate_vs_waiting" if later[i] != later[j] else "other")
                ok[t] += [((ZH[i] - Z1[i]) ** 2).mean() < ((ZH[i] - Z1[j]) ** 2).mean(),
                          ((ZH[j] - Z1[j]) ** 2).mean() < ((ZH[j] - Z1[i]) ** 2).mean()]
                dtr[t].append(dt); dpr[t].append(np.sqrt(((ZH[i] - ZH[j]) ** 2).mean()))
    allok = sum(ok.values(), [])
    R["discrimination"] = {"overall": float(np.mean(allok)) if allok else None,
                           **{t: float(np.mean(v)) if v else None for t, v in ok.items()},
                           "pairs": {t: len(v) // 2 for t, v in ok.items()}}
    if not full:
        return R

    def spear(x, y):
        if len(x) < 3:
            return None
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    R["discrimination"]["spearman_pred_vs_true_branch_distance"] = {t: spear(dtr[t], dpr[t]) for t in types}
    ones = np.ones_like(chg)
    R["errors"] = {g: summ(allr, t) for g, t in (("all_tokens", ones), ("changed_tokens", chg), ("unchanged_tokens", ~chg),
                                                 ("changed_operations", np.concatenate([ch_op, np.zeros_like(ch_m)], 1)),
                                                 ("changed_machines", np.concatenate([np.zeros_like(ch_op), ch_m], 1)))}
    worse = []
    for k in np.split(order, bounds):
        worse.append(np.mean([E["shuffle"][i][chg[i]].mean() for i in k]) > np.mean([E["model"][i][chg[i]].mean() for i in k]))
    R["shuffle_worse_share_of_states"] = float(np.mean(worse))
    src = Dt["state_source"][st]; prog = Dt["state_progress"][st]
    R["by_source"] = {SOURCES[i]: summ(np.flatnonzero(src == i), chg) for i in range(len(SOURCES))}
    R["by_progress"] = {n_: summ(np.flatnonzero(prog == q), chg) for q, n_ in enumerate(("early", "mid", "late"))}
    return R


def outcome_predict(m, Dt, states, cfg):
    m.train_mode(False)
    PJ, PE = [], []
    with torch.no_grad():
        for b0 in range(0, len(states), 128):
            sel = states[b0:b0 + 128]
            s, cand, valid, srow, slot, tr = batch(Dt, sel, m.dev)
            pj, pm = m.outcome(Dt, sel, s, cand, srow, slot)
            PJ.append(pj.cpu().numpy() * cfg["t_norm"]); PE.append(pm[..., 1].cpu().numpy() * cfg["e_norm"])
    return np.concatenate(PJ), np.concatenate(PE)


def outcome_score(Dt, states, pj, pe):
    """AP.score extended: ranking accuracy of alternative actions, same operation / other machine, other operation /
    same machine; difference MAE vs the zero-difference baseline; absolute MAE. Per rule and objective."""
    M = B.SIZES[W.SIZE][1]
    cand = Dt["cand"][states]; valid = cand >= 0
    job, mch = cand // M, cand % M
    due = Dt["state_due"][states][:, None, :]
    res = {}
    for r, rule in enumerate(RULES):
        T = AP.objectives(Dt["RJ"][states][:, :, r], Dt["RM"][states][:, :, r, :, 1], due)
        P = AP.objectives(pj[:, :, r], pe[:, :, r], due)
        for k in T:
            dt, dp, sj, sm = [], [], [], []
            for i, j in itertools.combinations(range(cand.shape[1]), 2):
                mm = valid[:, i] & valid[:, j]
                dt.append(T[k][mm, i] - T[k][mm, j]); dp.append(P[k][mm, i] - P[k][mm, j])
                sj.append(job[mm, i] == job[mm, j]); sm.append((mch[mm, i] == mch[mm, j]) & (job[mm, i] != job[mm, j]))
            dt, dp, sj, sm = map(np.concatenate, (dt, dp, sj, sm))
            use = np.abs(dt) > 1e-9
            ok = np.where(np.abs(dp) < 1e-9, 0.5, (np.sign(dp) == np.sign(dt)).astype(float))
            res[f"{rule}|{k}"] = {"accuracy": float(ok[use].mean()), "same_operation_accuracy": float(ok[use & sj].mean()),
                                  "other_operation_same_machine_accuracy": float(ok[use & sm].mean()),
                                  "pairs": int(use.sum()), "same_operation_pairs": int((use & sj).sum()),
                                  "mae_difference": float(np.abs(dp - dt).mean()),
                                  "mae_difference_zero_baseline": float(np.abs(dt).mean()),
                                  "mae_absolute": float(np.abs(P[k] - T[k])[valid].mean())}
    keys = list(res)
    res["mean"] = {q: float(np.mean([res[k][q] for k in keys])) for q in res[keys[0]]}
    for k in ("makespan", "carbon", "tardiness"):
        res[f"mean|{k}"] = {q: float(np.mean([res[f'{r}|{k}'][q] for r in RULES])) for q in res[keys[0]]}
    return res


def rule_baseline(Dt, states):
    """Same operation, other machine: shorter processing time ranks better for makespan / tardiness; lower carbon
    for carbon (the simple pair rule of the earlier audit)."""
    pt = np.zeros(Dt["cand"].shape, np.float32); cb = np.zeros_like(pt)
    pt[Dt["tr_state"], Dt["tr_slot"]] = Dt["tr_pt"]; cb[Dt["tr_state"], Dt["tr_slot"]] = Dt["tr_carbon"]
    R = len(RULES)
    pj = np.repeat(pt[states][:, :, None, None], R, 2) * np.ones((1, 1, 1, B.SIZES[W.SIZE][0]))
    pe = np.repeat(cb[states][:, :, None, None], R, 2) * np.ones((1, 1, 1, B.SIZES[W.SIZE][1]))
    full = outcome_score(Dt, states, pj, pe)
    return {k: {"same_operation_accuracy": v["same_operation_accuracy"]} for k, v in full.items() if k.startswith("mean")}


def collapse(m, Dt, tr):
    net = m.tgt if m.tgt is not None else m.trunk
    return L.collapse_stats(net, Dt, tr, m.dev)


def monitor(m, Dt, vstates, cfg, train_terms, ep, t0):
    row = {"epoch": ep, "seconds": time.time() - t0, **{f"train_{k}": v for k, v in train_terms.items()}}
    vtr = Dt["tix"][vstates][Dt["cand"][vstates] >= 0]
    with torch.no_grad():
        m.train_mode(False)
        tot = {}
        for b0 in range(0, len(vstates), 128):
            t = m.losses(Dt, vstates[b0:b0 + 128])
            for k, v in t.items():
                tot[k] = tot.get(k, 0.0) + float(v) * len(vstates[b0:b0 + 128])
        row.update({f"val_{k}": v / len(vstates) for k, v in tot.items()})
        if m.L0 is not None:
            row["val_norm_mean"] = 0.5 * (row["val_lat"] / m.L0["lat"] + row["val_out"] / m.L0["out"])
    if m.pred is not None:
        le = latent_eval(m, Dt, vstates)
        row.update({"val_change_explained": le["changed_tokens"]["change_explained"],
                    "val_disc_overall": le["discrimination"]["overall"],
                    "val_disc_same_op": le["discrimination"]["same_operation_other_machine"]})
    if m.kind in ("B", "D") or m.kind.startswith("probe:"):
        sc = outcome_score(Dt, vstates, *outcome_predict(m, Dt, vstates, cfg))["mean"]
        row.update({"val_rank_acc": sc["accuracy"], "val_same_op_acc": sc["same_operation_accuracy"],
                    "val_diff_mae": sc["mae_difference"], "val_diff_mae_zero": sc["mae_difference_zero_baseline"]})
    row.update({f"collapse_{k}": v for k, v in collapse(m, Dt, vtr[:1024]).items()})
    m.train_mode(True)
    return row


def run(kind, lr, epochs, Dt, sp, cfg, dev, eval_every, trunk_sd=None, tag=None):
    tag = tag or kind
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m = Model(kind, dev, Dt["s_fea_pairs"].shape[-1], trunk_sd)
    opt = torch.optim.Adam(m.params(), lr=lr)
    rng = np.random.default_rng(C["train_seed"])
    trs = sp["train"]
    vstates = sp["validation"][:C["monitor_val_states"]]
    info = {"kind": kind, "lr": lr, "epochs": epochs, "parameters_trained": int(sum(p.numel() for p in m.params()))}
    # fixed normalisation batch (~1,024 transitions) and monitor batches
    nb = np.sort(rng.choice(trs, size=min(len(trs), int(C["norm_transitions"] / 4.3) + 1), replace=False))
    gm = [np.sort(rng.choice(trs, size=min(len(trs), C["batch_states"]), replace=False)) for _ in range(C["grad_monitor_batches"])]
    m.train_mode(True)
    if m.pred is not None and kind != "A_latent":
        st0 = L.collapse_stats(m.tgt, Dt, Dt["tix"][nb][Dt["cand"][nb] >= 0], dev)
        m.floor = C["var_floor_share_of_initial_std"] * st0["mean_std"]
        info["variance_floor"] = m.floor; info["collapse_start"] = st0
    if kind == "D":
        with torch.no_grad():
            t = m.losses(Dt, nb)
        m.L0 = {"lat": float(t["lat"]), "out": float(t["out"])}
        info["L0"] = m.L0
        # leakage and gradient-path checks (once, before training)
        sel = gm[0]
        s, cand, valid, srow, slot, tr = batch(Dt, sel, dev)
        with torch.no_grad():
            p1 = m.outcome(Dt, sel, s, cand, srow, slot)
            Dx = {k: (np.random.default_rng(0).permutation(v) if k.startswith("n_") else v) for k, v in Dt.items()}
            p2 = m.outcome(Dx, sel, s, cand, srow, slot)
        info["leakage_check_next_state_perturbation_changes_nothing"] = all(torch.equal(a, b) for a, b in zip(p1, p2))
        g = grad_stats(m, Dt, [sel])
        info["both_terms_reach_encoder"] = g["latent_grad_norm"] > 0 and g["outcome_grad_norm"] > 0
        (po, _, _), _ = m.latent(Dt, sel, s, srow, slot, cand[srow, slot])
        info["predicted_latent_requires_grad"] = bool(po.requires_grad)
        assert info["leakage_check_next_state_perturbation_changes_nothing"] and info["both_terms_reach_encoder"] \
            and info["predicted_latent_requires_grad"], info
    hist = [monitor(m, Dt, vstates, cfg, {}, 0, time.time())]
    grads = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = rng.permutation(trs)
        acc, n = {}, 0
        m.train_mode(True)
        for b0 in range(0, len(perm), C["batch_states"]):
            sel = np.sort(perm[b0:b0 + C["batch_states"]])
            t = m.losses(Dt, sel)
            loss = m.total(t)
            opt.zero_grad(); loss.backward(); opt.step(); m.ema()
            for k, v in t.items():
                acc[k] = acc.get(k, 0.0) + float(v) * len(sel)
            n += len(sel)
        terms = {k: v / n for k, v in acc.items()}
        if kind == "D":
            g = grad_stats(m, Dt, gm)
            g["epoch"] = ep; g["lam_out"] = m.lam_out
            if ep == 1 and "calibration" not in info:
                r = g["ratio_latent_to_outcome"]
                lo, hi = C["grad_ratio_band"]
                info["calibration"] = {"median_ratio_epoch1": r, "rescaled": not (lo <= r <= hi),
                                       "lam_out": m.lam_out * (r if not (lo <= r <= hi) else 1.0)}
                m.lam_out = info["calibration"]["lam_out"]
            grads.append(g)
        if ep % eval_every == 0 or ep == epochs:
            hist.append(monitor(m, Dt, vstates, cfg, terms, ep, t0))
            log(f"{tag} lr {lr:g} epoch {ep}: " + ", ".join(f"{k} {v:.4f}" for k, v in hist[-1].items()
                                                           if isinstance(v, float) and k.startswith(("val_", "train_"))))
    info["train_seconds"] = time.time() - t0
    info["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    info["peak_rss_gb"] = peak_rss_gb()
    if m.pred is not None and kind != "A_latent":
        f = hist[-1]
        info["collapsed"] = bool(f["collapse_effective_rank"] < C["collapse_rank_share"] * info["collapse_start"]["effective_rank"]
                                 or f["collapse_mean_std"] < C["collapse_std_share"] * info["collapse_start"]["mean_std"])
    return m, hist, grads, info


def converged(hist):
    """Every tracked validation metric improves by < converge_rel_tol (relative) over the last window."""
    keys = [k for k in hist[-1] if k.startswith("val_") and k != "val_diff_mae_zero"]
    end = hist[-1]["epoch"]; ref = [h for h in hist if h["epoch"] <= end - C["converge_window_epochs"]]
    if not ref:
        return {"verdict": "insufficient history"}
    ref = ref[-1]; last = [h for h in hist if h["epoch"] > ref["epoch"]]
    out = {}
    for k in keys:
        higher = k in ("val_change_explained", "val_disc_overall", "val_disc_same_op", "val_rank_acc", "val_same_op_acc")
        best = max(h[k] for h in last) if higher else min(h[k] for h in last)
        gain = (best - ref[k]) if higher else (ref[k] - best)
        out[k] = gain / max(abs(ref[k]), 1e-12)
    still = {k: v for k, v in out.items() if v >= C["converge_rel_tol"]}
    return {"relative_improvement_last_window": out, "still_improving": sorted(still),
            "verdict": "converged" if not still else "would benefit from more training"}


def save_model(m, kind, cfg):
    d = CK / kind; d.mkdir(parents=True, exist_ok=True)
    bundle = W.trunk_bundle(m.trunk.state_dict(), cfg, {"collection_sha256": sha(DATA / "collection.npz"), "kind": kind},
                            C["epochs"], 0.0)
    spec = torch.load(L.SPEC_START, map_location="cpu", weights_only=False)["state_dict"]
    fe = {k: v.shape for k, v in spec.items() if k.startswith("feature_exact.")}
    assert fe == {k: v.shape for k, v in bundle["state_dict"].items() if k.startswith("feature_exact.")}
    torch.save(bundle, d / "encoder_bundle.pth")
    torch.save(m.trunk.state_dict(), d / "online_encoder.pth")
    for name, mod in (("target_encoder", m.tgt if m.kind in ("C", "D") else None), ("predictor", m.pred),
                      ("outcome_head", m.head), ("outcome_model", m.out)):
        if mod is not None:
            torch.save(mod.state_dict(), d / f"{name}.pth")


def load_model(kind, dev, n_pair):
    m = Model(kind, dev, n_pair, torch.load(CK / kind / "online_encoder.pth", map_location="cpu")
              if kind != "A_latent" else None)
    d = CK / kind
    if kind in ("C", "D"):
        m.tgt.load_state_dict(torch.load(d / "target_encoder.pth", map_location=dev))
    if m.pred is not None:
        m.pred.load_state_dict(torch.load(d / "predictor.pth", map_location=dev))
    if m.head is not None:
        m.head.load_state_dict(torch.load(d / "outcome_head.pth", map_location=dev))
    if m.out is not None and (d / "outcome_model.pth").exists():
        m.out.load_state_dict(torch.load(d / "outcome_model.pth", map_location=dev))
    m.train_mode(False)
    return m


# ================================================================================================ stages

def data_ready():
    cfg, dev = L.setup()
    Dt = prepare(load_data(), cfg)
    return cfg, dev, Dt, splits(Dt)


def lrcheck():
    f = RES / "lrcheck.json"
    if f.exists():
        return json.loads(f.read_text())
    cfg, dev, Dt, sp = data_ready()
    out = {"runs": {}, "chosen": {}}
    for kind in ("B", "C", "D"):
        for lr in C["lrs"]:
            _, hist, _, info = run(kind, lr, C["lr_epochs"], Dt, sp, cfg, dev, C["lr_epochs"], tag=f"lrcheck {kind}")
            out["runs"][f"{kind}|{lr:g}"] = {"final": hist[-1], "train_seconds": info["train_seconds"],
                                             "peak_gpu_gb": info["peak_gpu_gb"]}
        key = {"B": ("val_out", min), "C": ("val_change_explained", max), "D": ("val_norm_mean", min)}[kind]
        lr = key[1](C["lrs"], key=lambda x: out["runs"][f"{kind}|{x:g}"]["final"][key[0]])
        out["chosen"][kind] = {"lr": lr, "criterion": key[0]}
        log(f"lrcheck {kind}: chose {lr:g} by {key[0]}")
    out["chosen"]["A_latent"] = {"lr": out["chosen"]["C"]["lr"], "criterion": "same as C"}
    out["chosen"]["probe"] = {"lr": out["chosen"]["B"]["lr"], "criterion": "same as B"}
    jdump(f, out)
    return out


def train():
    lrc = lrcheck()
    cfg, dev, Dt, sp = data_ready()
    for kind in ("B", "C", "D", "A_latent"):
        f = RES / f"train_{kind}.json"
        if f.exists():
            continue
        m, hist, grads, info = run(kind, lrc["chosen"][kind]["lr"], C["epochs"], Dt, sp, cfg, dev, C["eval_every"])
        save_model(m, kind, cfg)
        jdump(f, {"info": info, "history": hist, "gradients": grads, "convergence": converged(hist)})
        log(f"trained {kind}: {info['train_seconds']:.0f} s, convergence {converged(hist)['verdict']}")
    (CK / "SHA256SUMS.json").write_text(json.dumps({str(p.relative_to(CK)): sha(p) for p in sorted(CK.rglob("*.pth"))},
                                                   indent=1) + "\n")


def probe():
    lrc = lrcheck()
    cfg, dev, Dt, sp = data_ready()
    f = RES / "probes.json"
    P = json.loads(f.read_text()) if f.exists() else {}
    enc = {"A": None, "B": "B", "C": "C", "D": "D", "raw": None}
    for name, src in enc.items():
        if name in P:
            continue
        sd = torch.load(CK / src / "online_encoder.pth", map_location="cpu") if src else None
        kind = "probe:raw" if name == "raw" else f"probe:{name}"
        m, hist, _, info = run(kind, lrc["chosen"]["probe"]["lr"], C["probe_epochs"], Dt, sp, cfg, dev,
                               C["probe_epochs"], trunk_sd=sd, tag=f"probe {name}")
        sc = outcome_score(Dt, sp["test"], *outcome_predict(m, Dt, sp["test"], cfg))
        P[name] = {"test": sc, "info": info, "final_validation": hist[-1]}
        jdump(f, P)
    P["rule_baseline_same_operation"] = rule_baseline(Dt, sp["test"])
    jdump(f, P)


def avail_probe(enc, Dt, fit_tr, test_tr, dev, fit_pairs, test_pairs):
    """Linear OLS readouts of machine availability from machine representations of the ACTUAL next state."""
    def feats(rows):
        if enc == "raw":
            return Dt["n_fea_m"][rows].astype(np.float64)
        out = []
        with torch.no_grad():
            for b0 in range(0, len(rows), 1024):
                r = rows[b0:b0 + 1024]
                _, h_mch, _ = L.encode(enc, L.tensors(Dt, "n_", r, dev))
                out.append(h_mch.cpu().numpy().astype(np.float64))
        return np.concatenate(out)

    def ols(X, y):
        Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
        return np.linalg.lstsq(Xa, y, rcond=None)[0]

    def pred(w, X):
        return np.concatenate([X, np.ones((len(X), 1))], 1) @ w
    free = Dt["p1_mch_free_time"].astype(np.float64) * B.PT_SCALE                        # minutes
    yc = free - free.mean(1, keepdims=True)
    Xf, Xt = feats(fit_tr), feats(test_tr)
    f_ = Xf.shape[-1]
    w = ols(Xf.reshape(-1, f_), yc[fit_tr].reshape(-1))
    p = pred(w, Xt.reshape(-1, f_)); y = yc[test_tr].reshape(-1)
    res = {"absolute_centred_free_time": {"mae_minutes": float(np.abs(p - y).mean()), "r2": float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                                          "pearson": float(np.corrcoef(p, y)[0, 1])}}
    # same operation, other machine: per-machine differences between the two branches
    allr = np.unique(np.concatenate([fit_pairs.reshape(-1), test_pairs.reshape(-1)]))
    Fa = dict(zip(allr, feats(allr)))

    def diffs(pairs):
        X = np.stack([Fa[i] - Fa[j] for i, j in pairs]).reshape(-1, f_)
        Y = np.stack([free[i] - free[j] for i, j in pairs]).reshape(-1)
        return X, Y
    Xd, Yd = diffs(fit_pairs); w2 = ols(Xd, Yd)
    Xd2, Yd2 = diffs(test_pairs); pd_ = pred(w2, Xd2)
    nz = np.abs(Yd2) > 1e-9
    res["same_operation_branch_difference"] = {"sign_accuracy_nonzero": float((np.sign(pd_[nz]) == np.sign(Yd2[nz])).mean()),
                                               "mae_minutes_nonzero": float(np.abs(pd_[nz] - Yd2[nz]).mean()),
                                               "mae_zero_baseline_nonzero": float(np.abs(Yd2[nz]).mean()),
                                               "pearson_nonzero": float(np.corrcoef(pd_[nz], Yd2[nz])[0, 1]),
                                               "comparisons_nonzero": int(nz.sum())}
    return res


def same_op_pairs(Dt, states):
    M = B.SIZES[W.SIZE][1]
    out = []
    for s in states:
        c = Dt["cand"][s]; t = Dt["tix"][s]
        for i, j in itertools.combinations(np.flatnonzero(c >= 0), 2):
            if c[i] // M == c[j] // M:
                out.append((t[i], t[j]))
    return np.asarray(out, int).reshape(-1, 2)


def evaluate():
    f = RES / "evaluation.json"
    if f.exists():
        return json.loads(f.read_text())
    cfg, dev, Dt, sp = data_ready()
    n_pair = Dt["s_fea_pairs"].shape[-1]
    te = sp["test"]
    t0 = time.time()
    R = {"latent": {}, "outcome_native": {}, "availability": {}, "collapse": {}}
    for kind in ("A_latent", "C", "D"):
        m = load_model(kind, dev, n_pair)
        R["latent"][kind] = latent_eval(m, Dt, te, full=True)
        if kind != "A_latent":
            R["collapse"][kind] = collapse(m, Dt, Dt["tix"][te][Dt["cand"][te] >= 0][:4096])
        if kind == "D":
            R["outcome_native"]["D"] = outcome_score(Dt, te, *outcome_predict(m, Dt, te, cfg))
            R["outcome_native"]["D_by_progress"] = {n_: outcome_score(Dt, te[Dt["state_progress"][te] == q],
                                                                      *outcome_predict(m, Dt, te[Dt["state_progress"][te] == q], cfg))["mean"]
                                                    for q, n_ in enumerate(("early", "mid", "late"))}
        log(f"evaluated latent {kind}")
    mB = load_model("B", dev, n_pair)
    R["outcome_native"]["B"] = outcome_score(Dt, te, *outcome_predict(mB, Dt, te, cfg))
    R["outcome_native"]["B_by_progress"] = {n_: outcome_score(Dt, te[Dt["state_progress"][te] == q],
                                                              *outcome_predict(mB, Dt, te[Dt["state_progress"][te] == q], cfg))["mean"]
                                            for q, n_ in enumerate(("early", "mid", "late"))}
    rng = np.random.default_rng(C["train_seed"])
    tr_tr = Dt["tix"][sp["train"]][Dt["cand"][sp["train"]] >= 0]
    fit = np.sort(rng.choice(tr_tr, size=min(len(tr_tr), C["avail_fit_transitions"]), replace=False))
    tst = Dt["tix"][te][Dt["cand"][te] >= 0]
    fp = same_op_pairs(Dt, sp["train"]); fp = fp[np.sort(rng.choice(len(fp), size=min(len(fp), 20000), replace=False))]
    tp = same_op_pairs(Dt, te)
    for name in ("A", "B", "C", "D", "raw"):
        if name == "raw":
            enc = "raw"
        else:
            enc = DualAttentionNetwork(configs).to(dev)
            enc.load_state_dict(start_trunk_sd() if name == "A" else torch.load(CK / name / "online_encoder.pth", map_location=dev))
            enc.eval()
        R["availability"][name] = avail_probe(enc, Dt, fit, tst, dev, fp, tp)
    R["eval_seconds"] = time.time() - t0
    R["peak_rss_gb"] = peak_rss_gb()
    jdump(f, R)
    return R


def decide(R, P, T):
    m = lambda d: d["test"]["mean"]
    pB, pC, pD = m(P["B"]), m(P["C"]), m(P["D"])
    g = RULE["probe_margin_over_B_and_C"]
    c1 = {"rank_acc_D_minus_B": pD["accuracy"] - pB["accuracy"], "rank_acc_D_minus_C": pD["accuracy"] - pC["accuracy"],
          "same_op_D_minus_B": pD["same_operation_accuracy"] - pB["same_operation_accuracy"],
          "same_op_D_minus_C": pD["same_operation_accuracy"] - pC["same_operation_accuracy"],
          "D_diff_mae": pD["mae_difference"], "zero_baseline": pD["mae_difference_zero_baseline"]}
    c1["pass"] = all(c1[k] >= g for k in ("rank_acc_D_minus_B", "rank_acc_D_minus_C", "same_op_D_minus_B", "same_op_D_minus_C")) \
        and c1["D_diff_mae"] < c1["zero_baseline"]
    lC, lD = R["latent"]["C"], R["latent"]["D"]
    tol = RULE["latent_tolerance_vs_C"]
    c2 = {"change_explained_D_minus_C": lD["errors"]["changed_tokens"]["change_explained"] - lC["errors"]["changed_tokens"]["change_explained"],
          "discrimination_D_minus_C": lD["discrimination"]["overall"] - lC["discrimination"]["overall"]}
    c2["pass"] = c2["change_explained_D_minus_C"] >= -tol and c2["discrimination_D_minus_C"] >= -tol
    aC, aD = R["availability"]["C"], R["availability"]["D"]
    c3 = {"abs_r2_D": aD["absolute_centred_free_time"]["r2"], "abs_r2_C": aC["absolute_centred_free_time"]["r2"],
          "diff_sign_D": aD["same_operation_branch_difference"]["sign_accuracy_nonzero"],
          "diff_sign_C": aC["same_operation_branch_difference"]["sign_accuracy_nonzero"]}
    c3["pass"] = c3["abs_r2_D"] >= c3["abs_r2_C"] and c3["diff_sign_D"] >= c3["diff_sign_C"]
    c4 = {"collapsed": T["D"]["info"].get("collapsed")}
    c4["pass"] = not c4["collapsed"]
    return {"1_frozen_probe_ranking": c1, "2_latent_vs_C": c2, "3_availability_vs_C": c3, "4_no_collapse": c4,
            "justifies_specialist_training": all(c["pass"] for c in (c1, c2, c3, c4))}


def report():
    R = evaluate()
    P = json.loads((RES / "probes.json").read_text())
    T = {k: json.loads((RES / f"train_{k}.json").read_text()) for k in ("B", "C", "D", "A_latent")}
    lrc = json.loads((RES / "lrcheck.json").read_text())
    cov = json.loads((RES / "coverage.json").read_text())
    dec = decide(R, P, T)
    jdump(RES / "decision.json", dec)
    plot(T, lrc)
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    lines = ["# Joint latent-state + measurable-outcome world-model pilot (10x5, waiting allowed, seed 111)", "",
             "Single-seed pilot. Settings frozen in `results/world_model_joint_pilot_v3/settings.json` before collection "
             f"(sha256 {(RES / 'settings.sha256').read_text().strip()[:12]}…).", "",
             "## Architecture", "",
             "`S_t -> DANIEL encoder (DualAttentionNetwork, feature_exact keys) -> action-conditioned latent predictor -> "
             "predicted latent of S_t+1 -> outcome head (+ continuation-rule one-hot) -> 10 job completions, 5 machine busy "
             "times, 5 machine energies`.",
             "- Latent predictor: the latent pilot's per-token residual MLP + one self-attention layer; input = encoder tokens of "
             "S_t, the chosen operation/machine tokens, globals, the pair's 10 observed DANIEL pair features, per-token flags.",
             "- Target: EMA copy of the online encoder (tau 0.996), stop-gradient, applied to the actual S_t+1.",
             "- Outcome head (D): per job, mean+max pooling of the predicted tokens of that job's operations + predicted globals "
             "+ rule; per machine, the predicted machine token + predicted globals + rule. It never sees S_t, raw action "
             "features or S_t+1 (checked: perturbing every S_t+1 array leaves all predictions bit-identical; both loss terms "
             "give non-zero encoder gradients; the predicted latent is not detached).",
             f"- D loss: L_lat/L_lat0 + {C['var_weight']} x variance hinge + lambda_out x L_out/L_out0 (L0 = each term at "
             f"initialisation on a fixed ~1,024-transition batch). lambda_out calibrated once after epoch 1 if the encoder "
             f"gradient-norm ratio left [1/3, 3]: {json.dumps(T['D']['info'].get('calibration'))}.",
             "- B: the earlier PairWorldModel (encoder + action readout + pair features), abs + diff Huber, retrained here. "
             "C: latent only. A: the random start encoder (a predictor trained on it for latent metrics).",
             "- Multi-step readiness: the predictor maps tokens -> tokens and can be rolled; on imagined states the observed "
             "pair features would be unavailable (limitation; not exercised here).", "",
             "## Dataset", "",
             f"- {sum(cov['factories'].values())} generator factories ({cov['factories']['train']} train / {cov['factories']['validation']} validation / {cov['factories']['test']} test) (seed {C['seed']}); split by factory; semi-active simulator (waiting allowed).",
             f"- {cov['totals']['states']:,} states, {cov['totals']['transitions']:,} transitions; up to {C['max_branches']} branches per state; "
             f"final outcomes of every branch under {', '.join(RULES)} (decoder continuation, never waits).",
             f"- Generation: episodes+branching {cov['seconds']['episodes_and_branching']:.0f} s, continuation rollouts "
             f"{cov['seconds']['continuation_rollouts']:.0f} s; {cov['bytes'] / 1e9:.2f} GB; peak RAM {cov['peak_rss_gb']:.1f} GB.",
             "- Checks: replay of S_t+1, decoder = environment after prefix + action, branch independence (all passed).", "",
             "| split, source | states | branches/state | same-op group | other-op same-machine | waiting | early/mid/late |",
             "|---|---|---|---|---|---|---|"]
    for k, v in cov["per_split_source"].items():
        if k.startswith("train"):
            lines.append(f"| {k.replace('|', ', ')} | {v['states']} | {v['branches_per_state']:.2f} | {f3(v['share_in_same_operation_group'])} | "
                         f"{f3(v['share_in_other_operation_same_machine_group'])} | {f3(v['share_waiting'])} | "
                         f"{'/'.join(f'{x:.2f}' for x in v['progress_share_early_mid_late'])} |")
    lines += ["", "## Learning-rate check (validation factories)", "", "| config | " + " | ".join(f"lr {x:g}" for x in C["lrs"]) + " | chosen |",
              "|---|" + "---|" * len(C["lrs"]) + "---|"]
    for kind in ("B", "C", "D"):
        crit = lrc["chosen"][kind]["criterion"]
        lines.append(f"| {kind} ({crit}) | " + " | ".join(f"{lrc['runs'][f'{kind}|{x:g}']['final'][crit]:.4g}" for x in C["lrs"])
                     + f" | {lrc['chosen'][kind]['lr']:g} |")
    lines += ["", "## Training and convergence", "", "| config | lr | epochs | seconds | peak GPU GB | verdict | still improving |",
              "|---|---|---|---|---|---|---|"]
    for k in ("B", "C", "D", "A_latent"):
        i = T[k]["info"]; cv = T[k]["convergence"]
        lines.append(f"| {k} | {i['lr']:g} | {i['epochs']} | {i['train_seconds']:.0f} | {f3(i['peak_gpu_gb'])} | {cv['verdict']} | "
                     f"{', '.join(cv.get('still_improving', [])) or '-'} |")
    if T["D"]["gradients"]:
        g = T["D"]["gradients"]
        lines += ["", f"D gradient balance on the encoder (median over monitor batches): epoch 1 ratio latent/outcome "
                      f"{g[0]['ratio_latent_to_outcome']:.2f}, cosine {g[0]['cosine']:.2f}; final epoch ratio "
                      f"{g[-1]['ratio_latent_to_outcome']:.2f}, cosine {g[-1]['cosine']:.2f}."]
    lines += ["", "## Results on the test factories", "", "### Latent prediction (own target space; scale-free metrics)", "",
              "| config | change explained (changed tokens) | shuffle worse share | discrimination overall | same op / other machine | other op / same machine | immediate vs waiting |",
              "|---|---|---|---|---|---|---|"]
    for k in ("A_latent", "C", "D"):
        l = R["latent"][k]
        lines.append(f"| {k} | {f3(l['errors']['changed_tokens']['change_explained'])} | {f3(l['shuffle_worse_share_of_states'])} | "
                     f"{f3(l['discrimination']['overall'])} | {f3(l['discrimination']['same_operation_other_machine'])} | "
                     f"{f3(l['discrimination']['other_operation_same_machine'])} | {f3(l['discrimination']['immediate_vs_waiting'])} |")
    lines += ["", "Copy-current-latent baseline = change explained 0 by definition; action shuffle = the next branch's action.", "",
              "### Outcome prediction (mean over 3 rules and 3 objectives)", "",
              "| model | ranking accuracy | same op / other machine | other op / same machine | difference MAE | zero-difference MAE | absolute MAE |",
              "|---|---|---|---|---|---|---|"]
    rows = [("B native", R["outcome_native"]["B"]["mean"]), ("D native (from predicted latent)", R["outcome_native"]["D"]["mean"])]
    rows += [(f"probe on frozen {k}", P[k]["test"]["mean"]) for k in ("A", "B", "C", "D", "raw")]
    for name, v in rows:
        lines.append(f"| {name} | {f3(v['accuracy'])} | {f3(v['same_operation_accuracy'])} | {f3(v['other_operation_same_machine_accuracy'])} | "
                     f"{f3(v['mae_difference'])} | {f3(v['mae_difference_zero_baseline'])} | {f3(v['mae_absolute'])} |")
    rb = P["rule_baseline_same_operation"]
    lines += ["", f"Simple pair rule (shorter processing time / lower energy) on same-operation pairs: "
                  f"makespan {f3(rb['mean|makespan']['same_operation_accuracy'])}, carbon {f3(rb['mean|carbon']['same_operation_accuracy'])}, "
                  f"tardiness {f3(rb['mean|tardiness']['same_operation_accuracy'])}.", "",
              "Per objective (frozen probes, same-operation accuracy / ranking accuracy):", "",
              "| encoder | makespan | carbon | tardiness |", "|---|---|---|---|"]
    for k in ("A", "B", "C", "D", "raw"):
        lines.append(f"| {k} | " + " | ".join(f"{f3(P[k]['test'][f'mean|{o}']['same_operation_accuracy'])} / {f3(P[k]['test'][f'mean|{o}']['accuracy'])}"
                                              for o in ("makespan", "carbon", "tardiness")) + " |")
    lines += ["", "### Machine availability readouts (linear, from the representation of the actual next state)", "",
              "| representation | absolute R^2 | absolute MAE (min) | same-op difference sign accuracy | difference MAE (min) | zero-baseline MAE |",
              "|---|---|---|---|---|---|"]
    for k in ("A", "B", "C", "D", "raw"):
        a = R["availability"][k]
        lines.append(f"| {k} | {f3(a['absolute_centred_free_time']['r2'])} | {f3(a['absolute_centred_free_time']['mae_minutes'])} | "
                     f"{f3(a['same_operation_branch_difference']['sign_accuracy_nonzero'])} | {f3(a['same_operation_branch_difference']['mae_minutes_nonzero'])} | "
                     f"{f3(a['same_operation_branch_difference']['mae_zero_baseline_nonzero'])} |")
    lines += ["", "### Collapse diagnostics (target encoder on test next states)", "", "| config | effective rank (start) | effective rank (end) | mean std (start -> end) | collapsed |",
              "|---|---|---|---|---|"]
    for k in ("C", "D"):
        i = T[k]["info"]; c = R["collapse"][k]
        lines.append(f"| {k} | {i['collapse_start']['effective_rank']:.2f} | {c['effective_rank']:.2f} | "
                     f"{i['collapse_start']['mean_std']:.4f} -> {c['mean_std']:.4f} | {i.get('collapsed')} |")
    lines += ["", "## Pre-registered decision", "", RULE_TEXT, "", "```", json.dumps(dec, indent=1), "```", "",
              f"**Verdict: {'proceed to specialist training' if dec['justifies_specialist_training'] else 'does not yet justify specialist training'}.**", "",
              "## Limitations", "",
              "- One seed, pilot scale; differences of a few hundredths are within run-to-run noise not measured here.",
              "- Final outcomes depend on the continuation rule (three fixed non-waiting rules); they are not properties of S_t+1 alone.",
              "- Latent metrics live in each model's own target space; only scale-free metrics are compared across configs.",
              "- The predictor uses the observed pair features of S_t; multi-step rollouts on imagined states would need them predicted.",
              "- Resources: see train_*.json (peak GPU memory, RAM) and coverage.json."]
    (REPO / "results" / f"{NAME}{SFX}.md").write_text("\n".join(lines) + "\n")
    log(f"report written; decision {dec['justifies_specialist_training']}")
    print(json.dumps(dec, indent=1))


def plot(T, lrc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a", "A_latent": "#7f7f7f"}
    panels = [("val_change_explained", "latent change explained (validation)"), ("val_disc_same_op", "latent discrimination, same op"),
              ("val_rank_acc", "outcome ranking accuracy"), ("val_same_op_acc", "outcome same-op accuracy"),
              ("collapse_effective_rank", "effective rank (target encoder)"), ("val_diff_mae", "outcome difference MAE")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
    for ax, (k, t) in zip(axs.reshape(-1), panels):
        for kind, d in T.items():
            h = [x for x in d["history"] if k in x and x[k] is not None]
            if h:
                ax.plot([x["epoch"] for x in h], [x[k] for x in h], color=col[kind], marker="o", ms=3, lw=1.6, label=kind)
        if k == "val_diff_mae":
            h = [x for x in T["B"]["history"] if "val_diff_mae_zero" in x]
            if h:
                ax.axhline(h[-1]["val_diff_mae_zero"], color="#555", ls="--", lw=.9, label="zero-difference baseline")
        ax.set_title(t, fontsize=10); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "curves.png", dpi=130); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), facecolor="white")
    for ax, kind in zip(axs, ("B", "C", "D")):
        crit = lrc["chosen"][kind]["criterion"]
        ax.plot(C["lrs"], [lrc["runs"][f"{kind}|{x:g}"]["final"][crit] for x in C["lrs"]], marker="o", color=col[kind])
        ax.set_xscale("log"); ax.set_title(f"{kind}: {crit}", fontsize=10); ax.set_xlabel("learning rate"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RES / "lrcheck.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["collect", "lrcheck", "train", "probe", "evaluate", "report", "all"])
    a = ap.parse_args(_ARGS)
    t0 = time.time()
    stages = ["collect", "lrcheck", "train", "probe", "evaluate", "report"] if a.stage == "all" else [a.stage]
    for st in stages:
        log(f"stage {st}")
        {"collect": collect, "lrcheck": lrcheck, "train": train, "probe": probe, "evaluate": evaluate, "report": report}[st]()
    log(f"done ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
