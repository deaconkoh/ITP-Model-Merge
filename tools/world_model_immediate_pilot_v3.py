"""Immediate-physical-consequence joint world model (follow-up to the joint pilot; 10x5, waiting allowed, seed 111).

    S_t -> DANIEL encoder -> action-conditioned latent predictor -> predicted latent of S_t+1
        -> physical head -> the factory's physical state IMMEDIATELY after the one decision

The physical head reads only the predicted next tokens; no continuation rule, no final outcomes, no rewards, no
objective scores. Targets (minutes; times relative to S_t's decision time t_t; z-scored with training statistics):
  F  machine free time F_m - t_t (5)          B  machine cumulative processing time (5)
  R  job ready time R_j - t_t (10)            W  job remaining minimum-processing workload (10)
  T  decision-time advance t_t+1 - t_t (1)

Configurations (same DANIEL encoder, same seed-111 start trunk, same training data and validation split):
  A       random start encoder (frozen); latent metrics from the joint pilot's A_latent predictor
  A_phys  frozen random encoder; predictor + physical head trained (how much is head arithmetic?)
  B       encoder + predictor + physical head, physical loss only (no latent loss, no EMA target)
  C, D    the joint pilot's latent-only and final-outcome joint checkpoints (reused, not retrained)
  E       joint: latent loss (EMA target, stop-gradient, variance hinge) + immediate physical loss
  raw     raw-feature linear token maps instead of the encoder; predictor + physical head (no DANIEL encoder)

Data: the joint pilot's collection (train + validation factories; its test factories are not used, they informed
this redesign) plus 300 FRESH factories (seed 6111) for the final assessment. Physical targets are extracted by
replaying every state's episode prefix in the semi-active environment and stepping every branch action.

Stages (resumable; IP_SMOKE=1 runs a tiny version into *_smoke folders):
  collect   fresh factories (the joint pilot's routine) + physical-target extraction (old and fresh) with checks
  train     E, B (to convergence, epoch-60 snapshot = matched budget), A_phys, raw (60 epochs)
  probe     relational probes (frozen encoders) + the joint pilot's final-outcome probes (secondary)
  evaluate  fresh factories: physical prediction, sibling differences, latent, availability, collapse
  report    results/world_model_immediate_pilot_v3.md

    python tools/world_model_immediate_pilot_v3.py all
"""
from __future__ import annotations
import copy, hashlib, itertools, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
_ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
import world_model_joint_pilot_v3 as J                              # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
import torch.nn.functional as Fn                                    # noqa: E402
from common_utils import setup_seed                                 # noqa: E402
from model.main_model import DualAttentionNetwork                  # noqa: E402

L, AP, W, B, WM, D, configs = J.L, J.AP, J.W, J.B, J.WM, J.D, J.configs
SMOKE = os.environ.get("IP_SMOKE") == "1"
SFX = "_smoke" if SMOKE else ""
NAME = "world_model_immediate_pilot_v3"
DATA, RES, CK = (REPO / "data" / f"{NAME}{SFX}", REPO / "results" / f"{NAME}{SFX}", REPO / "checkpoints" / f"{NAME}{SFX}")
OLD_DATA = REPO / "data" / "world_model_joint_pilot_v3" / "collection.npz"
OLD_CK = REPO / "checkpoints" / "world_model_joint_pilot_v3"
OLD_RES = REPO / "results" / "world_model_joint_pilot_v3"
GROUPS = ("F", "B", "R", "W", "T")
GROUP_NAMES = {"F": "machine free time", "B": "machine cumulative processing", "R": "job ready time",
               "W": "job remaining workload", "T": "decision-time advance"}
GROUP_LEVEL = {"F": "machine", "B": "machine", "R": "job", "W": "job", "T": "global"}
C = {"old_instances": 1350, "old_seed": 4111, "fresh_instances": 8 if SMOKE else 300, "fresh_seed": 6111,
     "smoke_states_train": 300, "smoke_states_val": 80,
     "lr": 1e-3, "lr_fallback": 3e-4, "instability_val_rise": 0.5,
     "epochs_matched": 2 if SMOKE else 60, "epoch_cap": 3 if SMOKE else 150, "eval_every": 1 if SMOKE else 5,
     "batch_states": 56, "train_seed": 111, "ema": 0.996, "var_weight": 1.0, "var_floor_share_of_initial_std": 0.5,
     "pred_hidden": 64, "pred_heads": 4, "head_hidden": 64, "diff_weight": 1.0, "norm_transitions": 1024,
     "grad_ratio_band": [1 / 3, 3.0], "grad_monitor_batches": 5, "monitor_val_states": 80 if SMOKE else 2000,
     "converge_window_epochs": 1 if SMOKE else 20, "converge_rel_tol": 0.01,
     "collapse_rank_share": 0.25, "collapse_std_share": 0.10,
     "probe_states": 300 if SMOKE else 20000, "probe_epochs": 2 if SMOKE else 30, "probe_batch": 256, "probe_lr": 1e-3,
     "probe_hidden": 64, "avail_fit_transitions": 20000, "avail_fit_pairs": 20000,
     "final_probe_epochs": 1 if SMOKE else 10, "final_probe_lr": 1e-3, "identity_tol_minutes": 1e-6}
RULE = {"vs_C_latent_tolerance": 0.02, "vs_C_same_op_gain": 0.10, "vs_C_availability_sign_gain": 0.10,
        "vs_D_tolerance": 0.02, "vs_B_relational_gain": 0.02, "vs_B_delta_mae_ratio_max": 1.10,
        "delta_r2_min_groups": ["F", "R", "T"], "delta_r2_min": 0.5}
RULE_TEXT = (
    "E (immediate-physical joint world model) is recommended as the shared encoder for specialist / Rewarded Soups "
    "training only if, on the FRESH test factories at the matched 60-epoch budget, ALL hold: "
    "(1) vs C: latent change explained (changed tokens) >= C - 0.02, same-operation/other-machine discrimination >= "
    "C + 0.10, actual-state availability same-operation difference sign accuracy >= C + 0.10; "
    "(2) vs D: same-operation/other-machine discrimination, overall discrimination and availability sign accuracy each "
    ">= D - 0.02, and the relational-probe mean (pair earliest start R^2, best-machine accuracy, machine earliest "
    "start R^2) >= D's; "
    "(3) vs B: relational-probe mean >= B + 0.02 and E's mean changed-entity change MAE <= 1.10 x B's; "
    "(4) physical quality: E's sibling-difference MAE below the zero-difference baseline in every target group, and "
    "changed-entity change R^2 >= 0.5 for machine free time, job ready time and decision-time advance; "
    "(5) E not collapsed. Final-outcome probes are reported but are not part of the rule.")
LOGF = None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(msg):
    global LOGF
    RES.mkdir(parents=True, exist_ok=True)
    LOGF = LOGF or open(RES / "run.log", "a")
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()


J.log = log
jdump = J.jdump


def settings():
    f = RES / "settings.json"
    if f.exists():
        return json.loads(f.read_text())
    s = {"constants": C, "decision_rule": RULE, "decision_rule_text": RULE_TEXT, "dispatch_rule": "semi_active",
         "targets": {g: {"name": GROUP_NAMES[g], "level": GROUP_LEVEL[g]} for g in GROUPS},
         "target_frame": "minutes; F, R and T relative to S_t's decision time t_t; B, W absolute; z-scored per group "
                         "with training-split mean and std",
         "dropped_as_redundant": {
             "machine energy / carbon": "gamma * P_m * B_m / 60 (carbon is processing energy only; checked)",
             "job completion lower bound": "R_j + W_j (checked against op_ct_lb of the job's last operation)",
             "operation completion lower bounds": "R_j + cumulative minimum processing times (static)",
             "machine idle time": "F_m - B_m (definition)", "current makespan": "max_m F_m (definition)",
             "remaining operation count": "same information as W_j (which operations are scheduled)"},
         "splits": {"train": "joint pilot train factories", "validation": "joint pilot validation factories (selection, "
                    "monitoring)", "test": f"{C['fresh_instances']} fresh factories, seed {C['fresh_seed']}",
                    "unused": "joint pilot test factories (informed this redesign)"},
         "reused": {"collection": {"path": str(OLD_DATA.relative_to(REPO)), "bytes": OLD_DATA.stat().st_size},
                    "checkpoints": {k: str((OLD_CK / k).relative_to(REPO)) for k in ("A_latent", "B", "C", "D")}},
         "inputs": {p.name: {"path": str(p.relative_to(REPO)), "sha256": sha(p)} for p in (L.START, L.POLICY, L.SPEC_START)},
         "joint_pilot_settings_sha256": (OLD_RES / "settings.sha256").read_text().strip(),
         "seed": 111, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    jdump(f, s)
    (RES / "settings.sha256").write_text(sha(f) + "\n")
    return s


# ================================================================================================ 1. data

def generate(seed, n):
    from benchmark_v3_generator import GeneratorSource
    setup_seed(seed)
    gen = GeneratorSource(configs, W.SIZE, B.load_manifest())
    return gen.sample(n)


def smoke_states(Dt):
    tr = np.flatnonzero(Dt["state_split"] == 0)[:C["smoke_states_train"]]
    va = np.flatnonzero(Dt["state_split"] == 1)[:C["smoke_states_val"]]
    return np.sort(np.concatenate([tr, va]))


def extract(Dt, data, states):
    """Replay every episode that holds one of `states`, step every branch action at those states, and read the
    physical state before and after (minutes). Cross-checks against the stored arrays and the simulator identities."""
    Jn, M = B.SIZES[W.SIZE]
    S, T = len(Dt["state_inst"]), len(Dt["tr_state"])
    P = {f"ph{w}_{g}": np.full((T,) + ((M,) if g in "FB" else (Jn,) if g in "RW" else ()), np.nan)
         for w in (0, 1) for g in ("F", "B", "R", "W", "t")}
    st_pt = np.zeros((S, Jn, M)); st_minpt = np.zeros((S, Jn))
    chk = {"stored_mismatch": 0, "transitions": 0, "max_err": {}}

    def err(k, v):
        chk["max_err"][k] = max(chk["max_err"].get(k, 0.0), float(np.max(np.abs(v))) if np.size(v) else 0.0)
    ok_later = []
    eps = np.unique(Dt["state_ep"][states])
    want = set(states.tolist())
    by_ep = {}
    for s in states:
        by_ep.setdefault(int(Dt["state_ep"][s]), []).append(int(s))
    for c0 in range(0, len(eps), 1350):
        ep = eps[c0:c0 + 1350]; E = len(ep); inst = Dt["ep_inst"][ep]
        env = W.T.FJSPEnvForSameOpNums(Jn, M); env.set_initial_data(**data.env_kwargs(inst, "mct"))
        busy = np.zeros((E, M))
        at = {}
        for r, e in enumerate(ep):
            for s in by_ep[int(e)]:
                at.setdefault(int(Dt["state_step"][s]), []).append((r, s))
        first = env.job_first_op_id; last = env.job_last_op_id
        opi = np.arange(env.number_of_ops)
        member = (opi[None, None, :] >= first[:, :, None]) & (opi[None, None, :] <= last[:, :, None])      # [E, J, N]
        N = env.number_of_ops
        for t in range(N):
            if t in at:
                rows, acts, trs, srows = [], [], [], []
                for r, s in at[t]:
                    assert s in want
                    for slot in range(Dt["cand"].shape[1]):
                        if Dt["cand"][s, slot] >= 0:
                            rows.append(r); acts.append(int(Dt["cand"][s, slot])); trs.append(int(Dt["tix"][s, slot]))
                            srows.append(s)
                    # state-level: processing time of every candidate pair (minutes; 0 where ineligible), min pt
                    cop = env.candidate[r]
                    pt_raw = np.asarray(data.op_pt[inst[r]], np.float64)
                    st_pt[s] = pt_raw[np.minimum(cop, N - 1)] * (~np.asarray(env.dynamic_pair_mask[r]).astype(bool))
                    st_minpt[s] = np.asarray(env.op_min_pt[r])[np.minimum(cop, N - 1)] * B.PT_SCALE
                rows = np.asarray(rows); acts = np.asarray(acts); trs = np.asarray(trs)
                br = D.select(env, rows)
                mem = member[rows]

                def phys(en, bz):
                    sched = np.asarray(en.op_scheduled_flag)
                    minpt = np.asarray(en.op_min_pt) * B.PT_SCALE
                    work = np.einsum("tn,tjn->tj", minpt * (1 - sched), mem)
                    return {"F": np.asarray(en.mch_free_time) * B.PT_SCALE, "B": bz.copy(),
                            "R": np.asarray(en.candidate_free_time) * B.PT_SCALE, "W": work,
                            "t": np.asarray(en.next_schedule_time) * B.PT_SCALE}
                p0 = phys(br, busy[rows])
                ar = np.arange(len(rows))
                job, mch = acts // M, acts % M
                op = np.asarray(br.candidate)[ar, job]
                pt = np.array([data.op_pt[inst[r]][o, m] for r, o, m in zip(rows, op, mch)], np.float64)
                # stored "before" arrays
                for q in ("mch_free_time", "op_ct_lb", "op_scheduled_flag"):
                    chk["stored_mismatch"] += int(not np.array_equal(np.asarray(getattr(br, q)).astype(np.float32), Dt[f"p0_{q}"][trs]))
                minpt_op = np.asarray(br.op_min_pt)[ar, op] * B.PT_SCALE
                br.step(acts)
                b1 = busy[rows].copy(); b1[ar, mch] += pt
                p1 = phys(br, b1)
                for q in ("mch_free_time", "op_ct_lb", "op_scheduled_flag"):
                    chk["stored_mismatch"] += int(not np.array_equal(np.asarray(getattr(br, q)).astype(np.float32), Dt[f"p1_{q}"][trs]))
                # identities: start = max(R_j, F_m); completion = start + p; only the chosen machine / job change
                start = np.maximum(p0["R"][ar, job], p0["F"][ar, mch]); comp = start + pt
                fa = p0["F"].copy(); fa[ar, mch] = comp
                ra = p0["R"].copy(); ra[ar, job] = comp
                wa = p0["W"].copy(); wa[ar, job] -= minpt_op
                err("analytic_F", fa - p1["F"]); err("analytic_R", ra - p1["R"]); err("analytic_W", wa - p1["W"])
                err("true_free_time_vs_scaled", np.asarray(br.true_mch_free_time) - p1["F"])
                err("true_job_ready_vs_scaled", np.asarray(br.true_candidate_free_time) - p1["R"])
                lb_last = np.take_along_axis(np.asarray(br.op_ct_lb), np.asarray(br.job_last_op_id), 1) * B.PT_SCALE
                err("job_completion_lb_eq_R_plus_W", lb_last - (p1["R"] + p1["W"]))
                err("decision_time_never_decreases", np.minimum(p1["t"] - p0["t"], 0))
                ok_later += list(((start > p0["t"] + 1e-6) == Dt["tr_later"][trs].astype(bool)))
                for g in ("F", "B", "R", "W", "t"):
                    P[f"ph0_{g}"][trs] = p0[g]; P[f"ph1_{g}"][trs] = p1[g]
                chk["transitions"] += len(trs)
            a = Dt["ep_actions"][ep, t].astype(int)
            job, mch = a // M, a % M
            op = np.asarray(env.candidate)[np.arange(E), job]
            busy[np.arange(E), mch] += np.array([data.op_pt[inst[r]][o, m] for r, (o, m) in enumerate(zip(op, mch))])
            env.step(a)
    # carbon is processing energy only: carbon[o, m] = k_m * p_om for one constant per machine
    kerr = 0.0
    for i in np.unique(Dt["ep_inst"][eps]):
        pt_, cb = np.asarray(data.op_pt[i], np.float64), np.asarray(data.carbon[i], np.float64)
        el = pt_ > 0
        k = np.where(el, cb / np.where(el, pt_, 1), np.nan)
        kerr = max(kerr, float(np.nanmax(np.nanmax(k, 0) - np.nanmin(k, 0))))
    chk["max_err"]["carbon_per_minute_constant_per_machine"] = kerr
    chk["waiting_flag_equals_start_after_decision_time_share"] = float(np.mean(ok_later)) if ok_later else None
    out = {k: v.astype(np.float32) for k, v in P.items()}
    out["st_pair_pt"] = st_pt.astype(np.float32); out["st_cand_minpt"] = st_minpt.astype(np.float32)
    return out, chk


def assert_checks(chk, tag):
    bad = {k: v for k, v in chk["max_err"].items() if v > C["identity_tol_minutes"] * 1e3 and k != "carbon_per_minute_constant_per_machine"}
    if chk["stored_mismatch"] or bad or chk["max_err"]["carbon_per_minute_constant_per_machine"] > 1e-9:
        raise SystemExit(f"{tag}: extraction checks failed: {chk}")


def collect():
    cfg, dev = L.setup()
    settings()
    fresh = DATA / "fresh"
    if not (DATA / "phys_old.npz").exists():
        t0 = time.time()
        Dt = J.load_data()
        data = generate(C["old_seed"], C["old_instances"])
        assert list(data.keys) == list(Dt["factory"]), "regenerated joint-pilot factories differ from the stored ones"
        states = smoke_states(Dt) if SMOKE else np.flatnonzero(Dt["state_split"] < 2)
        P, chk = extract(Dt, data, states)
        assert_checks(chk, "old")
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez(DATA / "phys_old.npz", states=states, **P)
        jdump(RES / "extraction_old.json", {**chk, "seconds": time.time() - t0, "states": int(len(states))})
        log(f"extraction (joint-pilot train+validation states): {len(states)} states, {chk['transitions']} transitions, "
            f"{time.time() - t0:.0f} s; max identity errors {chk['max_err']}")
    if not (fresh / "collection.npz").exists():
        # the joint pilot's collection routine, constants overridden; every fresh factory is a test factory
        J.DATA, J.RES = fresh, RES / "fresh_collection"
        J.C.update({"instances": C["fresh_instances"], "seed": C["fresh_seed"], "train_below": 0, "val_below": 0})
        J.collect()
        J.DATA, J.RES = OLD_DATA.parent, OLD_RES
    if not (DATA / "phys_fresh.npz").exists():
        t0 = time.time()
        Df = dict(np.load(fresh / "collection.npz", allow_pickle=False))
        old_keys = set(np.load(OLD_DATA, allow_pickle=False)["factory"].tolist())
        dup = old_keys & set(Df["factory"].tolist())
        assert not dup, f"{len(dup)} fresh factories duplicate joint-pilot factories"
        data = generate(C["fresh_seed"], C["fresh_instances"])
        assert list(data.keys) == list(Df["factory"])
        states = np.arange(len(Df["state_inst"]))
        P, chk = extract(Df, data, states)
        assert_checks(chk, "fresh")
        np.savez(DATA / "phys_fresh.npz", states=states, **P)
        jdump(RES / "extraction_fresh.json", {**chk, "seconds": time.time() - t0, "states": int(len(states)),
                                              "duplicates_with_joint_pilot_factories": 0})
        log(f"extraction (fresh): {len(states)} states, {chk['transitions']} transitions, {time.time() - t0:.0f} s")
    if not (RES / "coverage.json").exists():
        Dt, Df = load_all()
        jdump(RES / "coverage.json", coverage(Dt, Df))


def coverage(Dt, Df):
    M = B.SIZES[W.SIZE][1]
    out = {}
    for name, D_, sts in (("train", Dt, Dt["sp"]["train"]), ("validation", Dt, Dt["sp"]["validation"]),
                          ("fresh test", Df, Df["sp"]["test"])):
        cand = D_["cand"][sts]; valid = cand >= 0
        so, sm = J.pair_types(cand, M)
        later = np.zeros_like(valid); tix = D_["tix"][sts]
        later[valid] = D_["tr_later"][tix[valid]].astype(bool)
        out[name] = {"factories": int(len(np.unique(D_["state_inst"][sts]))), "states": int(len(sts)),
                     "transitions": int(valid.sum()), "branches_per_state": float(valid.sum() / max(1, len(sts))),
                     "share_in_same_operation_group": float(so[valid].mean()),
                     "share_in_other_operation_same_machine_group": float(sm[valid].mean()),
                     "share_waiting": float(later[valid].mean()),
                     "progress_share_early_mid_late": [float((D_["state_progress"][sts] == q).mean()) for q in range(3)],
                     "per_source_states": {J.SOURCES[i]: int((D_["state_source"][sts] == i).sum()) for i in range(len(J.SOURCES))}}
    fc = RES / "fresh_collection" / "coverage.json"
    if fc.exists():
        f = json.loads(fc.read_text())
        out["fresh_collection_seconds"] = f["seconds"]; out["fresh_collection_checks"] = f["checks"]
        out["fresh_bytes"] = f["bytes"]
    return out


# ------------------------------------------------------------------------------------------------ targets

def attach(Dt, P, stats=None):
    """State-level padded targets: Y[g] normalised after-values [S, A, k], Ym[g] after (minutes, frame), Y0m[g] before."""
    S, A = Dt["cand"].shape
    t0 = P["ph0_t"].astype(np.float64)
    after = {"F": P["ph1_F"] - t0[:, None], "B": P["ph1_B"], "R": P["ph1_R"] - t0[:, None], "W": P["ph1_W"],
             "T": (P["ph1_t"] - t0)[:, None]}
    before = {"F": P["ph0_F"] - t0[:, None], "B": P["ph0_B"], "R": P["ph0_R"] - t0[:, None], "W": P["ph0_W"],
              "T": np.zeros_like(t0)[:, None]}
    if stats is None:
        tr = Dt["tix"][Dt["sp"]["train"]][Dt["cand"][Dt["sp"]["train"]] >= 0]
        stats = {g: (float(np.mean(after[g][tr])), float(np.std(after[g][tr]) + 1e-8)) for g in GROUPS}
    Dt["Y"], Dt["Ym"], Dt["Y0m"] = {}, {}, {}
    ok = Dt["cand"] >= 0
    srow, slot = np.nonzero(ok)
    trs = Dt["tix"][srow, slot]
    for g in GROUPS:
        k = after[g].shape[1]
        for name, src, norm in (("Y", after[g], True), ("Ym", after[g], False), ("Y0m", before[g], False)):
            a = np.zeros((S, A, k), np.float32)
            v = src[trs]
            a[srow, slot] = (v - stats[g][0]) / stats[g][1] if norm else v
            Dt[name][g] = a
    Dt["norm_stats"] = stats
    have = np.zeros(S, bool); have[Dt["tr_state"][np.isfinite(P["ph1_t"])]] = True
    Dt["have_phys"] = have
    st = np.full(S, -1); st[Dt["tr_state"]] = np.arange(len(Dt["tr_state"]))          # one transition per state
    Dt["st_R0"] = np.where(st[:, None] >= 0, (P["ph0_R"] - t0[:, None])[np.maximum(st, 0)], np.nan)
    Dt["st_F0"] = np.where(st[:, None] >= 0, (P["ph0_F"] - t0[:, None])[np.maximum(st, 0)], np.nan)
    Dt["st_pair_pt"] = P["st_pair_pt"]
    return stats


def load_all():
    Dt = J.load_data()
    Po = dict(np.load(DATA / "phys_old.npz"))
    keep = Po.pop("states")
    use = np.zeros(len(Dt["state_inst"]), bool); use[keep] = True
    Dt["sp"] = {"train": np.flatnonzero(use & (Dt["state_split"] == 0)), "validation": np.flatnonzero(use & (Dt["state_split"] == 1))}
    stats = attach(Dt, Po)
    assert Dt["have_phys"][Dt["sp"]["train"]].all() and Dt["have_phys"][Dt["sp"]["validation"]].all()
    Df = dict(np.load(DATA / "fresh" / "collection.npz", allow_pickle=False))
    Pf = dict(np.load(DATA / "phys_fresh.npz")); Pf.pop("states")
    Df["sp"] = {"test": np.arange(len(Df["state_inst"]))}
    attach(Df, Pf, stats)
    return Dt, Df


# ================================================================================================ 2. models

class PhysHead(nn.Module):
    """Reads ONLY predicted next tokens. Machine: token + globals -> (F, B); job: mean+max of its predicted operation
    tokens + globals -> (R, W); global: predicted globals -> T."""

    def __init__(self, d, h):
        super().__init__()
        self.mch = WM.mlp(3 * d, h, 2)
        self.job = WM.mlp(4 * d, h, 2)
        self.glob = WM.mlp(2 * d, h, 1)

    def forward(self, po, pm, pg, first, last):
        T_, N, d = po.shape
        Jn, M = first.shape[1], pm.shape[1]
        opi = torch.arange(N, device=po.device)
        mem = (opi[None, None, :] >= first[:, :, None]) & (opi[None, None, :] <= last[:, :, None])
        mf = mem.float()
        mean = (mf @ po) / mf.sum(-1, keepdim=True).clamp(min=1)
        mx = po[:, None].masked_fill(~mem[..., None], -1e9).max(2).values
        om = self.mch(torch.cat([pm, pg[:, None].expand(-1, M, -1)], -1))
        oj = self.job(torch.cat([mean, mx, pg[:, None].expand(-1, Jn, -1)], -1))
        return {"F": om[..., 0], "B": om[..., 1], "R": oj[..., 0], "W": oj[..., 1], "T": self.glob(pg)}


class RawEnc(nn.Module):
    """No DANIEL encoder: linear maps of the raw operation / machine features to the token width; globals = means."""

    def __init__(self, d):
        super().__init__()
        self.op = nn.Linear(configs.fea_j_input_dim, d)
        self.m = nn.Linear(configs.fea_m_input_dim, d)

    def forward(self, s):
        h_op, h_m = self.op(s["fea_j"]), self.m(s["fea_m"])
        return h_op, h_m, torch.cat([h_op.mean(1), h_m.mean(1)], -1)


def phys_loss(P, Y, valid):
    v = valid.float()
    A = valid.shape[1]
    pairs = (valid[:, :, None] & valid[:, None, :] & ~torch.eye(A, dtype=torch.bool, device=valid.device)).float()
    out = {}
    for g in GROUPS:
        p, y = P[g], Y[g]
        a = Fn.huber_loss(p, y, reduction="none", delta=1.0).mean(-1)
        la = (a * v).sum() / v.sum().clamp(min=1)
        d = Fn.huber_loss(p[:, :, None] - p[:, None, :], y[:, :, None] - y[:, None, :], reduction="none", delta=1.0).mean(-1)
        out[g] = la + C["diff_weight"] * (d * pairs).sum() / pairs.sum().clamp(min=1)
    return out


class PModel:
    """kind: E (joint), B (physical only), A_phys (frozen random encoder), raw (raw-feature token maps)."""

    def __init__(self, kind, dev, n_pair, trunk_sd=None):
        self.kind, self.dev = kind, dev
        d = configs.layer_fea_output_dim[-1]
        setup_seed(C["train_seed"])
        if kind == "raw":
            self.trunk = RawEnc(d).to(dev)
        else:
            self.trunk = DualAttentionNetwork(configs).to(dev)
            self.trunk.load_state_dict(trunk_sd or J.start_trunk_sd())
        self.tgt = None
        if kind == "E":
            self.tgt = copy.deepcopy(self.trunk)
            for p in self.tgt.parameters():
                p.requires_grad_(False)
        if kind == "A_phys":
            for p in self.trunk.parameters():
                p.requires_grad_(False)
        setup_seed(C["train_seed"])
        self.pred = L.Predictor(d, n_pair, C["pred_hidden"], C["pred_heads"]).to(dev)
        self.head = PhysHead(d, C["head_hidden"]).to(dev)
        self.lam, self.L0, self.floor = 1.0, None, None

    def modules(self):
        return [self.trunk, self.pred, self.head]

    def params(self):
        return [p for m in self.modules() for p in m.parameters() if p.requires_grad]

    def train_mode(self, on):
        for m in self.modules():
            m.train(on)
        if self.kind == "A_phys":
            self.trunk.eval()

    def encode(self, s):
        return self.trunk(s) if self.kind == "raw" else L.encode(self.trunk, s)

    def latent(self, Dt, sel, s, srow, slot, acts):
        """J.latent_pred with this model's encoder."""
        M = B.SIZES[W.SIZE][1]
        dev = self.dev
        h_op, h_mch, g = self.encode(s)
        sr = torch.as_tensor(srow, device=dev)
        Ho, Hm, G = h_op[sr], h_mch[sr], g[sr]
        job_np, mch_np = acts // M, acts % M
        job = torch.as_tensor(job_np, device=dev); mch = torch.as_tensor(mch_np, device=dev)
        ar = torch.arange(len(acts), device=dev)
        op = s["candidate"][sr, job]
        st = sel[srow]
        pair = torch.as_tensor(Dt["s_fea_pairs"][st, job_np, mch_np], dtype=torch.float32, device=dev)
        N = Ho.shape[1]
        fo = torch.zeros(len(acts), N, 3, device=dev); fo[ar, op, 0] = 1
        first = torch.as_tensor(Dt["s_job_first_op_id"][st, job_np].astype(np.int64), device=dev)
        last = torch.as_tensor(Dt["s_job_last_op_id"][st, job_np].astype(np.int64), device=dev)
        opi = torch.arange(N, device=dev)[None]
        fo[..., 1] = ((opi >= first[:, None]) & (opi <= last[:, None])).float()
        fm = torch.zeros(len(acts), M, 3, device=dev); fm[ar, mch, 2] = 1
        return self.pred(Ho, Hm, G, Ho[ar, op], Hm[ar, mch], pair, fo, fm), (h_op, h_mch, g)

    def physical(self, Dt, sel, s, cand, srow, slot, lat=None):
        (po, pm, pg), _ = lat if lat is not None else self.latent(Dt, sel, s, srow, slot, cand[srow, slot])
        st = sel[srow]
        first = torch.as_tensor(Dt["s_job_first_op_id"][st].astype(np.int64), device=self.dev)
        last = torch.as_tensor(Dt["s_job_last_op_id"][st].astype(np.int64), device=self.dev)
        o = self.head(po, pm, pg, first, last)
        r, c = torch.as_tensor(srow, device=self.dev), torch.as_tensor(slot, device=self.dev)
        out = {}
        for g, v in o.items():
            v = v if v.dim() == 2 else v[:, None]
            z = v.new_zeros((len(sel), Dt["cand"].shape[1], v.shape[-1])); z[r, c] = v
            out[g] = z
        return out

    def losses(self, Dt, sel):
        s, cand, valid, srow, slot, tr = J.batch(Dt, sel, self.dev)
        lat = self.latent(Dt, sel, s, srow, slot, cand[srow, slot])
        out = {}
        if self.kind == "E":
            (po, pm, pg), (h_op, h_mch, _) = lat
            out["lat"] = J.lat_loss((po, pm, pg), L.targets(self.tgt, Dt, tr, self.dev))
            X = torch.cat([h_op, h_mch], 1).reshape(-1, h_op.shape[-1])
            out["var"] = torch.relu(self.floor - X.std(0)).mean()
        P = self.physical(Dt, sel, s, cand, srow, slot, lat)
        Y = {g: torch.as_tensor(Dt["Y"][g][sel], device=self.dev) for g in GROUPS}
        per = phys_loss(P, Y, torch.as_tensor(valid, device=self.dev))
        out.update({f"phys_{g}": v for g, v in per.items()})
        out["phys"] = sum(per.values()) / len(GROUPS)
        return out

    def total(self, t):
        if self.kind == "E":
            return t["lat"] / self.L0["lat"] + C["var_weight"] * t["var"] + self.lam * t["phys"] / self.L0["phys"]
        return t["phys"]

    def ema(self):
        if self.kind == "E":
            with torch.no_grad():
                for p, q in zip(self.tgt.parameters(), self.trunk.parameters()):
                    p.mul_(C["ema"]).add_(q.detach(), alpha=1 - C["ema"])

    def state(self):
        out = {"online_encoder": self.trunk.state_dict(), "predictor": self.pred.state_dict(), "phys_head": self.head.state_dict()}
        if self.tgt is not None:
            out["target_encoder"] = self.tgt.state_dict()
        return out

    def load(self, st):
        self.trunk.load_state_dict(st["online_encoder"]); self.pred.load_state_dict(st["predictor"])
        self.head.load_state_dict(st["phys_head"])
        if self.tgt is not None:
            self.tgt.load_state_dict(st["target_encoder"])


def grad_stats(m, Dt, batches):
    tp = [p for p in m.trunk.parameters() if p.requires_grad]
    rows = []
    for sel in batches:
        t = m.losses(Dt, sel)
        g1 = torch.autograd.grad(t["lat"] / m.L0["lat"], tp, retain_graph=True, allow_unused=True)
        g2 = torch.autograd.grad(m.lam * t["phys"] / m.L0["phys"], tp, allow_unused=True)
        v1 = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(g1, tp)])
        v2 = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(g2, tp)])
        rows.append({"latent_grad_norm": float(v1.norm()), "physical_grad_norm": float(v2.norm()),
                     "ratio_latent_to_physical": float(v1.norm() / v2.norm().clamp(min=1e-12)),
                     "cosine": float(Fn.cosine_similarity(v1, v2, 0))})
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


# ================================================================================================ 3. evaluation pieces

def phys_predict(m, Dt, states):
    """Predicted after-values in minutes (frame of the targets) per transition, in J.batch order, + transition ids."""
    m.train_mode(False)
    out = {g: [] for g in GROUPS}; TR = []
    with torch.no_grad():
        for b0 in range(0, len(states), 128):
            sel = states[b0:b0 + 128]
            s, cand, valid, srow, slot, tr = J.batch(Dt, sel, m.dev)
            P = m.physical(Dt, sel, s, cand, srow, slot)
            for g in GROUPS:
                mu, sd = Dt["norm_stats"][g]
                out[g].append(P[g][torch.as_tensor(srow, device=m.dev), torch.as_tensor(slot, device=m.dev)].cpu().numpy() * sd + mu)
            TR.append(tr)
    return {g: np.concatenate(v) for g, v in out.items()}, np.concatenate(TR)


def r2(p, y):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    den = ((y - y.mean()) ** 2).sum()
    return float(1 - ((p - y) ** 2).sum() / den) if den > 0 else None


def sibling_pairs(Dt, TR):
    """Pairs (i, j) of positions in TR from the same state, with their type."""
    M = B.SIZES[W.SIZE][1]
    st = Dt["tr_state"][TR]
    order = np.argsort(st, kind="stable"); bounds = np.flatnonzero(np.diff(st[order])) + 1
    a = Dt["tr_action"][TR]; later = Dt["tr_later"][TR]
    I, Jx, typ = [], [], []
    for k in np.split(order, bounds):
        for x, y in itertools.combinations(k, 2):
            I.append(x); Jx.append(y)
            typ.append("same_operation_other_machine" if a[x] // M == a[y] // M else
                       "other_operation_same_machine" if a[x] % M == a[y] % M else
                       "immediate_vs_waiting" if later[x] != later[y] else "other")
    return np.asarray(I), np.asarray(Jx), np.asarray(typ)


def phys_metrics(Dt, TR, P, pairs=None):
    """Per group: after-value MAE / R^2 (and copy-current baseline); change on changed entities (chosen machine /
    job, clock) MAE / R^2; unchanged-entity error; sibling differences vs zero-difference baseline by pair type."""
    M = B.SIZES[W.SIZE][1]
    srow = Dt["tr_state"][TR]; slot = Dt["tr_slot"][TR]
    a = Dt["tr_action"][TR]; job, mch = a // M, a % M
    I, Jx, typ = pairs if pairs is not None else sibling_pairs(Dt, TR)
    res = {}
    for g in GROUPS:
        y = Dt["Ym"][g][srow, slot].astype(np.float64); y0 = Dt["Y0m"][g][srow, slot].astype(np.float64)
        p = P[g].astype(np.float64)
        k = y.shape[1]
        chg = np.zeros_like(y, bool)
        if GROUP_LEVEL[g] == "machine":
            chg[np.arange(len(y)), mch] = True
        elif GROUP_LEVEL[g] == "job":
            chg[np.arange(len(y)), job] = True
        else:
            chg[:] = True
        dt, dp = (y - y0)[chg], (p - y0)[chg]
        r = {"after_mae": float(np.abs(p - y).mean()), "after_r2": r2(p.reshape(-1), y.reshape(-1)),
             "copy_after_mae": float(np.abs(y0 - y).mean()), "copy_after_r2": r2(y0.reshape(-1), y.reshape(-1)),
             "change_mae_changed": float(np.abs(dp - dt).mean()), "change_r2_changed": r2(dp, dt),
             "copy_change_mae_changed": float(np.abs(dt).mean()),
             "unchanged_mae": float(np.abs(p - y0)[~chg].mean()) if (~chg).any() else None,
             "unchanged_true_change_max": float(np.abs(y - y0)[~chg].max()) if (~chg).any() else None}
        td = y[I] - y[Jx]; pd_ = p[I] - p[Jx]
        sib = {}
        for t in ("all", "same_operation_other_machine", "other_operation_same_machine", "immediate_vs_waiting", "other"):
            sel = np.ones(len(I), bool) if t == "all" else typ == t
            if not sel.any():
                sib[t] = None; continue
            tt, pp = td[sel].reshape(-1), pd_[sel].reshape(-1)
            nz = np.abs(tt) > 1e-6
            sib[t] = {"pairs": int(sel.sum()), "difference_mae": float(np.abs(pp - tt).mean()),
                      "zero_difference_mae": float(np.abs(tt).mean()),
                      "sign_accuracy_nonzero": float((np.sign(pp[nz]) == np.sign(tt[nz])).mean()) if nz.any() else None}
        r["siblings"] = sib
        res[g] = r
    res["mean_change_mae_changed"] = float(np.mean([res[g]["change_mae_changed"] for g in GROUPS]))
    return res


def val_phys(m, Dt, states):
    P, TR = phys_predict(m, Dt, states)
    R = phys_metrics(Dt, TR, P)
    return {"val_delta_mae_mean": R["mean_change_mae_changed"],
            **{f"val_delta_mae_{g}": R[g]["change_mae_changed"] for g in GROUPS},
            "val_sibling_diff_mae_F": R["F"]["siblings"]["all"]["difference_mae"]}


def monitor(m, Dt, vstates, terms, ep, t0):
    row = {"epoch": ep, "seconds": time.time() - t0, **{f"train_{k}": v for k, v in terms.items()}}
    with torch.no_grad():
        m.train_mode(False)
        tot = {}
        for b0 in range(0, len(vstates), 128):
            t = m.losses(Dt, vstates[b0:b0 + 128])
            for k, v in t.items():
                tot[k] = tot.get(k, 0.0) + float(v) * len(vstates[b0:b0 + 128])
        row.update({f"val_{k}": v / len(vstates) for k, v in tot.items()})
        if m.L0 is not None:
            row["val_total_norm"] = row["val_lat"] / m.L0["lat"] + row["val_phys"] / m.L0["phys"]
    if m.kind == "E":
        le = J.latent_eval(m, Dt, vstates)
        row.update({"val_change_explained": le["changed_tokens"]["change_explained"],
                    "val_disc_overall": le["discrimination"]["overall"],
                    "val_disc_same_op": le["discrimination"]["same_operation_other_machine"]})
    row.update(val_phys(m, Dt, vstates))
    if m.kind != "raw":
        vtr = Dt["tix"][vstates][Dt["cand"][vstates] >= 0]
        row.update({f"collapse_{k}": v for k, v in L.collapse_stats(m.tgt if m.tgt is not None else m.trunk, Dt, vtr[:1024], m.dev).items()})
    m.train_mode(True)
    return row


TRACK = {"E": ["val_lat", "val_phys", "val_change_explained", "val_disc_same_op", "val_delta_mae_mean"],
         "B": ["val_phys", "val_delta_mae_mean"], "A_phys": ["val_phys", "val_delta_mae_mean"],
         "raw": ["val_phys", "val_delta_mae_mean"]}
HIGHER = ("val_change_explained", "val_disc_overall", "val_disc_same_op")


def converged(hist, kind):
    end = hist[-1]["epoch"]; ref = [h for h in hist if h["epoch"] <= end - C["converge_window_epochs"]]
    if not ref:
        return {"verdict": "insufficient history"}
    ref = ref[-1]; last = [h for h in hist if h["epoch"] > ref["epoch"]]
    out = {}
    for k in TRACK[kind]:
        hi = k in HIGHER
        best = max(h[k] for h in last) if hi else min(h[k] for h in last)
        out[k] = ((best - ref[k]) if hi else (ref[k] - best)) / max(abs(ref[k]), 1e-12)
    still = sorted(k for k, v in out.items() if v >= C["converge_rel_tol"])
    return {"relative_improvement_last_window": out, "still_improving": still,
            "verdict": "converged" if not still else "would benefit from more training"}


def save(m, kind, tag, cfg, hist):
    d = CK / kind / tag; d.mkdir(parents=True, exist_ok=True)
    for k, v in m.state().items():
        torch.save(v, d / f"{k}.pth")
    if kind != "raw":
        bundle = W.trunk_bundle(m.trunk.state_dict(), cfg, {"kind": kind, "checkpoint": tag,
                                                             "phys_old_sha256": sha(DATA / "phys_old.npz")},
                                hist[-1]["epoch"], 0.0)
        spec = torch.load(L.SPEC_START, map_location="cpu", weights_only=False)["state_dict"]
        fe = {k: v.shape for k, v in spec.items() if k.startswith("feature_exact.")}
        assert fe == {k: v.shape for k, v in bundle["state_dict"].items() if k.startswith("feature_exact.")}
        torch.save(bundle, d / "encoder_bundle.pth")


def load(kind, tag, dev, n_pair):
    m = PModel(kind, dev, n_pair)
    d = CK / kind / tag
    m.load({k: torch.load(d / f"{k}.pth", map_location=dev) for k in ("online_encoder", "predictor", "phys_head", "target_encoder")
            if (d / f"{k}.pth").exists()})
    m.train_mode(False)
    return m


def run(kind, lr, Dt, cfg, dev, cap, matched):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m = PModel(kind, dev, Dt["s_fea_pairs"].shape[-1])
    opt = torch.optim.Adam(m.params(), lr=lr)
    rng = np.random.default_rng(C["train_seed"])
    trs = Dt["sp"]["train"]; vstates = Dt["sp"]["validation"][:C["monitor_val_states"]]
    info = {"kind": kind, "lr": lr, "epoch_cap": cap, "matched_epochs": matched,
            "parameters_trained": int(sum(p.numel() for p in m.params()))}
    nb = np.sort(rng.choice(trs, size=min(len(trs), int(C["norm_transitions"] / 4.3) + 1), replace=False))
    gm = [np.sort(rng.choice(trs, size=min(len(trs), C["batch_states"]), replace=False)) for _ in range(C["grad_monitor_batches"])]
    m.train_mode(True)
    if kind == "E":
        st0 = L.collapse_stats(m.tgt, Dt, Dt["tix"][nb][Dt["cand"][nb] >= 0], dev)
        m.floor = C["var_floor_share_of_initial_std"] * st0["mean_std"]
        info["variance_floor"] = m.floor; info["collapse_start"] = st0
        with torch.no_grad():
            t = m.losses(Dt, nb)
        m.L0 = {"lat": float(t["lat"]), "phys": float(t["phys"])}
        info["L0"] = m.L0
        sel = gm[0]
        s, cand, valid, srow, slot, tr = J.batch(Dt, sel, dev)
        with torch.no_grad():
            p1 = m.physical(Dt, sel, s, cand, srow, slot)
            Dx = {k: (np.random.default_rng(0).permutation(v) if k.startswith("n_") else v) for k, v in Dt.items()}
            Dx["Y"] = {g: np.random.default_rng(1).permutation(v) for g, v in Dt["Y"].items()}
            p2 = m.physical(Dx, sel, s, cand, srow, slot)
        info["leakage_check_next_state_and_target_perturbation_changes_nothing"] = all(torch.equal(p1[g], p2[g]) for g in GROUPS)
        g = grad_stats(m, Dt, [sel])
        info["both_terms_reach_encoder"] = g["latent_grad_norm"] > 0 and g["physical_grad_norm"] > 0
        (po, _, _), _ = m.latent(Dt, sel, s, srow, slot, cand[srow, slot])
        info["predicted_latent_requires_grad"] = bool(po.requires_grad)
        assert info["leakage_check_next_state_and_target_perturbation_changes_nothing"] and info["both_terms_reach_encoder"] \
            and info["predicted_latent_requires_grad"], info
    elif kind != "raw":
        info["collapse_start"] = L.collapse_stats(m.trunk, Dt, Dt["tix"][nb][Dt["cand"][nb] >= 0], dev)
    hist = [monitor(m, Dt, vstates, {}, 0, time.time())]
    grads = []
    t0 = time.time()
    snap = None
    for ep in range(1, cap + 1):
        perm = rng.permutation(trs)
        acc, n = {}, 0
        m.train_mode(True)
        for b0 in range(0, len(perm), C["batch_states"]):
            sel = np.sort(perm[b0:b0 + C["batch_states"]])
            t = m.losses(Dt, sel)
            loss = m.total(t)
            if not torch.isfinite(loss):
                return None, {"unstable": f"non-finite loss at epoch {ep}"}
            opt.zero_grad(); loss.backward(); opt.step(); m.ema()
            for k, v in t.items():
                acc[k] = acc.get(k, 0.0) + float(v) * len(sel)
            n += len(sel)
        terms = {k: v / n for k, v in acc.items()}
        if kind == "E":
            g = grad_stats(m, Dt, gm)
            g["epoch"] = ep; g["lam_phys"] = m.lam
            if ep == 1:
                r = g["ratio_latent_to_physical"]; lo, hi = C["grad_ratio_band"]
                info["calibration"] = {"median_ratio_epoch1": r, "rescaled": not (lo <= r <= hi),
                                       "lam_phys": m.lam * (r if not (lo <= r <= hi) else 1.0)}
                m.lam = info["calibration"]["lam_phys"]
            grads.append(g)
        if ep == 1 or ep % C["eval_every"] == 0 or ep == cap:
            hist.append(monitor(m, Dt, vstates, terms, ep, t0))
            log(f"{kind} lr {lr:g} epoch {ep}: " + ", ".join(f"{k} {v:.4f}" for k, v in hist[-1].items()
                                                         if isinstance(v, float) and k.startswith("val_")))
            key = "val_total_norm" if kind == "E" else "val_phys"
            if ep == min(5, cap) and not SMOKE and hist[-1][key] > (1 + C["instability_val_rise"]) * hist[1][key]:
                return None, {"unstable": f"validation {key} rose from {hist[1][key]:.4f} (epoch 1) to {hist[-1][key]:.4f} (epoch {ep})"}
        if ep == matched:
            snap = {"history": copy.deepcopy(hist), "train_seconds": time.time() - t0,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
                    "convergence": converged(hist, kind)}
            save(m, kind, "epoch60" if not SMOKE else f"epoch{matched}", J_cfg(), hist)
        if ep > matched and hist[-1]["epoch"] == ep and converged(hist, kind)["verdict"] == "converged":
            break
    info["train_seconds"] = time.time() - t0
    info["epochs_run"] = hist[-1]["epoch"]
    info["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    info["peak_rss_gb"] = J.peak_rss_gb()
    if hist[-1]["epoch"] > matched:
        save(m, kind, "final", J_cfg(), hist)
    for h in (hist, snap["history"]):
        f = h[-1]
        if kind in ("E", "B", "A_phys") and "collapse_effective_rank" in f:
            st0 = info["collapse_start"]
            h[-1]["collapsed"] = bool(f["collapse_effective_rank"] < C["collapse_rank_share"] * st0["effective_rank"]
                                      or f["collapse_mean_std"] < C["collapse_std_share"] * st0["mean_std"])
    return {"info": info, "matched": snap, "history": hist, "gradients": grads, "convergence": converged(hist, kind)}, None


_CFG = None


def J_cfg():
    return _CFG


def train():
    global _CFG
    cfg, dev = L.setup(); _CFG = cfg
    Dt, _ = load_all()
    for kind, cap in (("E", C["epoch_cap"]), ("B", C["epoch_cap"]), ("A_phys", C["epochs_matched"]), ("raw", C["epochs_matched"])):
        f = RES / f"train_{kind}.json"
        if f.exists():
            continue
        out, bad = run(kind, C["lr"], Dt, cfg, dev, cap, C["epochs_matched"])
        restart = None
        if bad:
            log(f"{kind}: unstable at lr {C['lr']:g} ({bad['unstable']}); restarting once at {C['lr_fallback']:g}")
            restart = {"from_lr": C["lr"], **bad}
            out, bad = run(kind, C["lr_fallback"], Dt, cfg, dev, cap, C["epochs_matched"])
            if bad:
                raise SystemExit(f"{kind} unstable at both learning rates: {bad}")
        out["info"]["restart"] = restart
        jdump(f, out)
        log(f"trained {kind}: {out['info']['epochs_run']} epochs, {out['info']['train_seconds']:.0f} s, "
            f"matched-budget convergence {out['matched']['convergence']['verdict']}, final {out['convergence']['verdict']}")
    (CK / "SHA256SUMS.json").write_text(json.dumps({str(p.relative_to(CK)): sha(p) for p in sorted(CK.rglob("*.pth"))},
                                                   indent=1) + "\n")


# ------------------------------------------------------------------------------------------------ probes

def encoders(dev):
    """Frozen encoders for probes / readouts: name -> DualAttentionNetwork or 'raw'."""
    mt = "epoch60" if not SMOKE else f"epoch{C['epochs_matched']}"
    src = {"A": None, "B_final_outcome (joint pilot)": OLD_CK / "B" / "online_encoder.pth",
           "C": OLD_CK / "C" / "online_encoder.pth", "D": OLD_CK / "D" / "online_encoder.pth",
           "B": CK / "B" / mt / "online_encoder.pth", "E": CK / "E" / mt / "online_encoder.pth"}
    for k in ("B", "E"):
        if (CK / k / "final" / "online_encoder.pth").exists():
            src[f"{k} (converged)"] = CK / k / "final" / "online_encoder.pth"
    out = {}
    for name, p in src.items():
        net = DualAttentionNetwork(configs).to(dev)
        net.load_state_dict(J.start_trunk_sd() if p is None else torch.load(p, map_location=dev))
        out[name] = net.eval()
    out["raw"] = "raw"
    return out


def state_feats(enc, Dt, states, dev):
    """Per state: candidate-operation tokens [S, J, d], machine tokens [S, M, d], globals [S, g]."""
    if enc == "raw":
        cj = np.take_along_axis(Dt["s_fea_j"][states], Dt["s_candidate"][states].astype(np.int64)[..., None], 1)
        fm = Dt["s_fea_m"][states]
        return cj, fm, np.concatenate([Dt["s_fea_j"][states].mean(1), fm.mean(1)], -1)
    A_, Bm, G = [], [], []
    with torch.no_grad():
        for b0 in range(0, len(states), 1024):
            r = states[b0:b0 + 1024]
            s = L.tensors(Dt, "s_", r, dev)
            h_op, h_m, g = L.encode(enc, s)
            A_.append(torch.gather(h_op, 1, s["candidate"].unsqueeze(-1).expand(-1, -1, h_op.shape[-1])).cpu().numpy())
            Bm.append(h_m.cpu().numpy()); G.append(g.cpu().numpy())
    return np.concatenate(A_), np.concatenate(Bm), np.concatenate(G)


def relational_targets(Dt, states):
    elig = ~Dt["s_dynamic_pair_mask"][states].astype(bool)                               # [S, J, M]
    R0, F0 = Dt["st_R0"][states], Dt["st_F0"][states]
    start = np.maximum(R0[:, :, None], F0[:, None, :])
    comp = start + Dt["st_pair_pt"][states]
    best = np.where(elig, comp, np.inf)
    job_ok = elig.sum(-1) >= 2
    mstart = np.where(elig, start, np.inf).min(1); m_ok = elig.any(1)
    return elig, start, best, job_ok, mstart, m_ok


class RelProbe(nn.Module):
    def __init__(self, dj, dm, dg, h):
        super().__init__()
        self.pair = WM.mlp(dj + dm + dg, h, 2)
        self.mch = WM.mlp(dm + dj + dg, h, 1)

    def forward(self, cj, hm, g):
        S, Jn, _ = cj.shape; M = hm.shape[1]
        x = torch.cat([cj[:, :, None].expand(-1, -1, M, -1), hm[:, None].expand(-1, Jn, -1, -1),
                       g[:, None, None].expand(-1, Jn, M, -1)], -1)
        o = self.pair(x)
        xm = torch.cat([hm, cj.mean(1, keepdim=True).expand(-1, M, -1), g[:, None].expand(-1, M, -1)], -1)
        return o[..., 0], o[..., 1], self.mch(xm).squeeze(-1)


def relational_probe(enc, Dt, tr_states, Df, te_states, dev):
    """(i) earliest start of every eligible candidate pair, (ii) earliest-completion machine per candidate job,
    (iii) each machine's earliest start over candidate jobs. Frozen features, identical small MLP probe."""
    Xtr = [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in state_feats(enc, Dt, tr_states, dev)]
    Xte = [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in state_feats(enc, Df, te_states, dev)]
    el, st, best, jok, ms, mok = relational_targets(Dt, tr_states)
    mu_s, sd_s = float(st[el].mean()), float(st[el].std() + 1e-8)
    mu_m, sd_m = float(ms[mok].mean()), float(ms[mok].std() + 1e-8)
    T = {k: torch.as_tensor(v, device=dev) for k, v in (("el", el), ("st", np.where(el, (st - mu_s) / sd_s, 0)),
                                                        ("best", np.where(np.isfinite(best), best, 1e9)), ("jok", jok),
                                                        ("ms", np.where(mok, (ms - mu_m) / sd_m, 0)), ("mok", mok))}
    setup_seed(C["train_seed"])
    net = RelProbe(Xtr[0].shape[-1], Xtr[1].shape[-1], Xtr[2].shape[-1], C["probe_hidden"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=C["probe_lr"])
    rng = np.random.default_rng(C["train_seed"])
    for ep in range(C["probe_epochs"]):
        perm = rng.permutation(len(tr_states))
        for b0 in range(0, len(tr_states), C["probe_batch"]):
            i = torch.as_tensor(perm[b0:b0 + C["probe_batch"]], device=dev)
            ps, sc, pm = net(Xtr[0][i], Xtr[1][i], Xtr[2][i])
            e = T["el"][i].float()
            l1 = (Fn.huber_loss(ps, T["st"][i].float(), reduction="none") * e).sum() / e.sum().clamp(min=1)
            tgt = T["best"][i].argmin(-1)
            logits = sc.masked_fill(~T["el"][i], -1e9)
            jk = T["jok"][i].float()
            l2 = (Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none") * jk.reshape(-1)).sum() / jk.sum().clamp(min=1)
            mk = T["mok"][i].float()
            l3 = (Fn.huber_loss(pm, T["ms"][i].float(), reduction="none") * mk).sum() / mk.sum().clamp(min=1)
            loss = l1 + l2 + l3
            opt.zero_grad(); loss.backward(); opt.step()
    el, st, best, jok, ms, mok = relational_targets(Df, te_states)
    with torch.no_grad():
        P = [], [], []
        for b0 in range(0, len(te_states), 2048):
            o = net(*(x[b0:b0 + 2048] for x in Xte))
            for a, b in zip(P, o):
                a.append(b.cpu().numpy())
    ps, sc, pm = (np.concatenate(x) for x in P)
    ps = ps * sd_s + mu_s; pm = pm * sd_m + mu_m
    pick = np.where(el, sc, -np.inf).argmax(-1)
    picked = np.take_along_axis(best, pick[..., None], -1)[..., 0]
    acc = float((picked[jok] <= best.min(-1)[jok] + 1e-6).mean())
    out = {"pair_earliest_start_r2": r2(ps[el], st[el]), "pair_earliest_start_mae_minutes": float(np.abs(ps[el] - st[el]).mean()),
           "best_machine_accuracy": acc,
           "best_machine_chance": float(np.mean(1.0 / el.sum(-1)[jok])),
           "machine_earliest_start_r2": r2(pm[mok], ms[mok]), "machine_earliest_start_mae_minutes": float(np.abs(pm[mok] - ms[mok]).mean()),
           "pairs": int(el.sum()), "jobs": int(jok.sum()), "machines": int(mok.sum())}
    out["mean"] = float(np.mean([out["pair_earliest_start_r2"], out["best_machine_accuracy"], out["machine_earliest_start_r2"]]))
    return out


def probe():
    global _CFG
    cfg, dev = L.setup(); _CFG = cfg
    Dt, Df = load_all()
    f = RES / "probes.json"
    P = json.loads(f.read_text()) if f.exists() else {"relational": {}, "final_outcome": {}}
    rng = np.random.default_rng(C["train_seed"])
    trs = np.sort(rng.choice(Dt["sp"]["train"], size=min(len(Dt["sp"]["train"]), C["probe_states"]), replace=False))
    te = Df["sp"]["test"]
    for name, enc in encoders(dev).items():
        if name in P["relational"]:
            continue
        t0 = time.time()
        P["relational"][name] = {**relational_probe(enc, Dt, trs, Df, te, dev), "seconds": time.time() - t0}
        jdump(f, P)
        log(f"relational probe {name}: {P['relational'][name]}")
    # secondary: the joint pilot's frozen-encoder final-outcome probes, trained on the joint pilot's train factories
    Do = J.prepare(Dt, cfg)
    Dfp = J.prepare(Df, cfg)
    sp = {"train": Dt["sp"]["train"], "validation": Dt["sp"]["validation"]}
    mt = "epoch60" if not SMOKE else f"epoch{C['epochs_matched']}"
    src = {"A": None, "B": CK / "B" / mt / "online_encoder.pth", "C": OLD_CK / "C" / "online_encoder.pth",
           "D": OLD_CK / "D" / "online_encoder.pth", "E": CK / "E" / mt / "online_encoder.pth", "raw": None}
    J.C["monitor_val_states"] = C["monitor_val_states"]
    for name, p in src.items():
        if name in P["final_outcome"]:
            continue
        sd = torch.load(p, map_location="cpu") if p else None
        m, hist, _, info = J.run("probe:raw" if name == "raw" else f"probe:{name}", C["final_probe_lr"], C["final_probe_epochs"],
                                 Do, sp, cfg, dev, C["final_probe_epochs"], trunk_sd=sd, tag=f"final-outcome probe {name}")
        sc = J.outcome_score(Dfp, te, *J.outcome_predict(m, Dfp, te, cfg))
        P["final_outcome"][name] = {"test": sc, "train_seconds": info["train_seconds"], "final_validation": hist[-1]}
        jdump(f, P)
        log(f"final-outcome probe {name}: {sc['mean']}")
    P["final_outcome_rule_baseline"] = J.rule_baseline(Dfp, te)
    jdump(f, P)


# ------------------------------------------------------------------------------------------------ evaluate

def ols_fit(X, y):
    return np.linalg.lstsq(np.concatenate([X, np.ones((len(X), 1))], 1), y, rcond=None)[0]


def ols_pred(w, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ w


def actual_mch(enc, Dt, tr, dev):
    if enc == "raw":
        return Dt["n_fea_m"][tr].astype(np.float64)
    out = []
    with torch.no_grad():
        for b0 in range(0, len(tr), 1024):
            out.append(L.encode(enc, L.tensors(Dt, "n_", tr[b0:b0 + 1024], dev))[1].cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def predicted_mch(m, Dt, tr):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(tr), 1024):
            t = tr[b0:b0 + 1024]
            sel, srow = np.unique(Dt["tr_state"][t], return_inverse=True)
            s = L.tensors(Dt, "s_", sel, m.dev)
            (_, pm, _), _ = m.latent(Dt, sel, s, srow, None, Dt["tr_action"][t])
            out.append(pm.cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def availability(feat, Dt, Df, fit_tr, te_tr, fit_pairs, te_pairs):
    """OLS readouts (J.avail_probe recipe), fitted on joint-pilot train transitions, scored on fresh transitions:
    centred machine free time of S_t+1, and the per-machine free-time difference of same-operation branch pairs."""
    def y(Dx, rows):
        f = Dx["p1_mch_free_time"][rows].astype(np.float64) * B.PT_SCALE
        return f - f.mean(1, keepdims=True), f
    rows_f = np.unique(np.concatenate([fit_tr, fit_pairs.reshape(-1)])); rows_t = np.unique(np.concatenate([te_tr, te_pairs.reshape(-1)]))
    Ff = dict(zip(rows_f, feat(Dt, rows_f))); Ft = dict(zip(rows_t, feat(Df, rows_t)))
    d = next(iter(Ff.values())).shape[-1]
    Xf = np.stack([Ff[i] for i in fit_tr]).reshape(-1, d); Xt = np.stack([Ft[i] for i in te_tr]).reshape(-1, d)
    w = ols_fit(Xf, y(Dt, fit_tr)[0].reshape(-1)); p = ols_pred(w, Xt); yt = y(Df, te_tr)[0].reshape(-1)
    res = {"absolute_centred_free_time": {"r2": r2(p, yt), "mae_minutes": float(np.abs(p - yt).mean()),
                                          "pearson": float(np.corrcoef(p, yt)[0, 1])}}
    Xd = np.stack([Ff[i] - Ff[j] for i, j in fit_pairs]).reshape(-1, d)
    Yd = (Dt["p1_mch_free_time"][fit_pairs[:, 0]] - Dt["p1_mch_free_time"][fit_pairs[:, 1]]).astype(np.float64).reshape(-1) * B.PT_SCALE
    w2 = ols_fit(Xd, Yd)
    Xd2 = np.stack([Ft[i] - Ft[j] for i, j in te_pairs]).reshape(-1, d)
    Yd2 = (Df["p1_mch_free_time"][te_pairs[:, 0]] - Df["p1_mch_free_time"][te_pairs[:, 1]]).astype(np.float64).reshape(-1) * B.PT_SCALE
    pd_ = ols_pred(w2, Xd2); nz = np.abs(Yd2) > 1e-9
    res["same_operation_branch_difference"] = {"sign_accuracy_nonzero": float((np.sign(pd_[nz]) == np.sign(Yd2[nz])).mean()),
                                               "mae_minutes_nonzero": float(np.abs(pd_[nz] - Yd2[nz]).mean()),
                                               "mae_zero_baseline_nonzero": float(np.abs(Yd2[nz]).mean()),
                                               "pearson_nonzero": float(np.corrcoef(pd_[nz], Yd2[nz])[0, 1]),
                                               "comparisons_nonzero": int(nz.sum())}
    return res


def evaluate():
    global _CFG
    f = RES / "evaluation.json"
    if f.exists():
        return json.loads(f.read_text())
    cfg, dev = L.setup(); _CFG = cfg
    Dt, Df = load_all()
    n_pair = Dt["s_fea_pairs"].shape[-1]
    te = Df["sp"]["test"]
    mt = "epoch60" if not SMOKE else f"epoch{C['epochs_matched']}"
    t0 = time.time()
    R = {"physical": {}, "latent": {}, "availability_actual": {}, "availability_predicted": {}, "collapse": {}}
    models = {"E": load("E", mt, dev, n_pair), "B": load("B", mt, dev, n_pair), "A_phys": load("A_phys", mt, dev, n_pair),
              "raw": load("raw", mt, dev, n_pair)}
    for k in ("E", "B"):
        if (CK / k / "final").exists():
            models[f"{k} (converged)"] = load(k, "final", dev, n_pair)
    pairs = None
    for name, m in models.items():
        P, TR = phys_predict(m, Df, te)
        pairs = pairs or sibling_pairs(Df, TR)
        R["physical"][name] = phys_metrics(Df, TR, P, pairs)
        R["physical"][name]["by_progress"] = {}
        for q, pn in enumerate(("early", "mid", "late")):
            sq = te[Df["state_progress"][te] == q]
            Pq, TRq = phys_predict(m, Df, sq)
            R["physical"][name]["by_progress"][pn] = phys_metrics(Df, TRq, Pq)["mean_change_mae_changed"]
        log(f"evaluated physical {name}: mean change MAE {R['physical'][name]['mean_change_mae_changed']:.3f}")
    # latent prediction: A (joint pilot's A_latent predictor), C, D, E (+ converged E)
    old = {k: J.load_model(k, dev, n_pair) for k in ("A_latent", "C", "D")}
    for name, m in [("A", old["A_latent"]), ("C", old["C"]), ("D", old["D"]), ("E", models["E"])] + \
            ([("E (converged)", models["E (converged)"])] if "E (converged)" in models else []):
        R["latent"][name] = J.latent_eval(m, Df, te, full=True)
        log(f"evaluated latent {name}")
    for name, m in (("C", old["C"]), ("D", old["D"]), ("E", models["E"])):
        R["collapse"][name] = J.collapse(m, Df, Df["tix"][te][Df["cand"][te] >= 0][:4096])
    # availability readouts
    rng = np.random.default_rng(C["train_seed"])
    tr_tr = Dt["tix"][Dt["sp"]["train"]][Dt["cand"][Dt["sp"]["train"]] >= 0]
    fit = np.sort(rng.choice(tr_tr, size=min(len(tr_tr), C["avail_fit_transitions"]), replace=False))
    tst = Df["tix"][te][Df["cand"][te] >= 0]
    fp = J.same_op_pairs(Dt, Dt["sp"]["train"]); fp = fp[np.sort(rng.choice(len(fp), size=min(len(fp), C["avail_fit_pairs"]), replace=False))]
    tp = J.same_op_pairs(Df, te)
    for name, enc in encoders(dev).items():
        R["availability_actual"][name] = availability(lambda Dx, rows, e=enc: actual_mch(e, Dx, rows, dev), Dt, Df, fit, tst, fp, tp)
    pm = {"A": old["A_latent"], "C": old["C"], "D": old["D"], **models}
    for name, m in pm.items():
        m.train_mode(False)
        R["availability_predicted"][name] = availability(lambda Dx, rows, mm=m: predicted_mch(mm, Dx, rows), Dt, Df, fit, tst, fp, tp)
    log("evaluated availability")
    R["eval_seconds"] = time.time() - t0
    R["peak_rss_gb"] = J.peak_rss_gb()
    jdump(f, R)
    return R


# ================================================================================================ decision + report

def decide(R, P, T):
    lat = lambda k: R["latent"][k]
    av = lambda k: R["availability_actual"][k]["same_operation_branch_difference"]["sign_accuracy_nonzero"]
    rel = lambda k: P["relational"][k]["mean"]
    ce = lambda k: lat(k)["errors"]["changed_tokens"]["change_explained"]
    so = lambda k: lat(k)["discrimination"]["same_operation_other_machine"]
    ov = lambda k: lat(k)["discrimination"]["overall"]
    c1 = {"change_explained_E_minus_C": ce("E") - ce("C"), "same_op_disc_E_minus_C": so("E") - so("C"),
          "availability_sign_E_minus_C": av("E") - av("C")}
    c1["pass"] = c1["change_explained_E_minus_C"] >= -RULE["vs_C_latent_tolerance"] and \
        c1["same_op_disc_E_minus_C"] >= RULE["vs_C_same_op_gain"] and c1["availability_sign_E_minus_C"] >= RULE["vs_C_availability_sign_gain"]
    c2 = {"same_op_disc_E_minus_D": so("E") - so("D"), "overall_disc_E_minus_D": ov("E") - ov("D"),
          "availability_sign_E_minus_D": av("E") - av("D"), "relational_mean_E": rel("E"), "relational_mean_D": rel("D")}
    tol = RULE["vs_D_tolerance"]
    c2["pass"] = all(c2[k] >= -tol for k in ("same_op_disc_E_minus_D", "overall_disc_E_minus_D", "availability_sign_E_minus_D")) \
        and c2["relational_mean_E"] >= c2["relational_mean_D"]
    pE, pB = R["physical"]["E"], R["physical"]["B"]
    c3 = {"relational_mean_E_minus_B": rel("E") - rel("B"), "change_mae_E": pE["mean_change_mae_changed"],
          "change_mae_B": pB["mean_change_mae_changed"]}
    c3["pass"] = c3["relational_mean_E_minus_B"] >= RULE["vs_B_relational_gain"] and \
        c3["change_mae_E"] <= RULE["vs_B_delta_mae_ratio_max"] * c3["change_mae_B"]
    c4 = {"sibling_diff_mae_vs_zero": {g: [pE[g]["siblings"]["all"]["difference_mae"], pE[g]["siblings"]["all"]["zero_difference_mae"]] for g in GROUPS},
          "change_r2": {g: pE[g]["change_r2_changed"] for g in RULE["delta_r2_min_groups"]}}
    c4["pass"] = all(a < b for a, b in c4["sibling_diff_mae_vs_zero"].values()) and \
        all(v is not None and v >= RULE["delta_r2_min"] for v in c4["change_r2"].values())
    c5 = {"collapsed": T["E"]["matched"]["history"][-1].get("collapsed")}
    c5["pass"] = not c5["collapsed"]
    return {"1_vs_C_transition_learning": c1, "2_vs_D_continuation_dependency": c2, "3_vs_B_value_of_latent": c3,
            "4_physical_quality": c4, "5_no_collapse": c5,
            "recommend_E_for_specialist_training": all(c["pass"] for c in (c1, c2, c3, c4, c5))}


def plot(T, TD):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"E": "#1baf7a", "B": "#2a78d6", "A_phys": "#8a8a85", "raw": "#eb6834"}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2), facecolor="#fcfcfb")
    ax = ax.ravel()
    panels = [("val_phys", "physical loss (validation)"), ("val_delta_mae_mean", "mean change MAE, changed entities (min)"),
              ("val_delta_mae_F", "machine free-time change MAE (min)"), ("val_change_explained", "latent change explained (E)"),
              ("val_disc_same_op", "latent discrimination, same op / other machine (E)")]
    for a, (k, title) in zip(ax, panels):
        for kind, t in T.items():
            h = [x for x in t["history"] if k in x]
            if h:
                a.plot([x["epoch"] for x in h], [x[k] for x in h], marker="o", ms=3, lw=1.6, color=col[kind], label=kind)
        a.axvline(C["epochs_matched"], color="#b5b4ad", lw=.8, ls="--")
        a.set_title(title, fontsize=10, loc="left"); a.set_xlabel("epoch"); a.grid(color="#e4e3de", lw=.7)
        a.legend(frameon=False, fontsize=8)
    a = ax[5]
    g = T["E"]["gradients"]
    a.plot([x["epoch"] for x in g], [x["ratio_latent_to_physical"] for x in g], color=col["E"], lw=1.6, label="E: latent / physical")
    gd = TD.get("gradients", [])
    a.plot([x["epoch"] for x in gd], [x["ratio_latent_to_outcome"] for x in gd], color="#6b6a64", lw=1.6, ls="--",
           label="D (joint pilot): latent / final outcome")
    a2 = a.twinx()
    a2.plot([x["epoch"] for x in g], [x["cosine"] for x in g], color=col["E"], lw=1, alpha=.5)
    a2.plot([x["epoch"] for x in gd], [x["cosine"] for x in gd], color="#6b6a64", lw=1, alpha=.5, ls="--")
    a2.set_ylabel("cosine (thin lines)", fontsize=8)
    a.set_yscale("log"); a.set_title("encoder gradient-norm ratio (thick) and cosine", fontsize=10, loc="left")
    a.set_xlabel("epoch"); a.grid(color="#e4e3de", lw=.7); a.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f"Validation factories (joint pilot validation split). Dashed vertical: matched budget "
                         f"({C['epochs_matched']} epochs, as C and D). Single seed (111).", fontsize=8, color="#6b6a64")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RES / "curves.png", dpi=140, facecolor="#fcfcfb"); plt.close(fig)


def report():
    R = evaluate()
    P = json.loads((RES / "probes.json").read_text())
    T = {k: json.loads((RES / f"train_{k}.json").read_text()) for k in ("E", "B", "A_phys", "raw")}
    TD = json.loads((OLD_RES / "train_D.json").read_text())
    cov = json.loads((RES / "coverage.json").read_text())
    xo, xf = json.loads((RES / "extraction_old.json").read_text()), json.loads((RES / "extraction_fresh.json").read_text())
    dec = decide(R, P, T)
    jdump(RES / "decision.json", dec)
    plot(T, TD)
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    f2 = lambda x: "-" if x is None else f"{x:.2f}"
    st = json.loads((RES / "settings.json").read_text())
    L_ = ["# Immediate-physical-consequence joint world model (10x5, waiting allowed, seed 111)", "",
          "Follow-up to `results/world_model_joint_pilot_v3.md` (whose decision rule and verdict are unchanged). Single seed. "
          f"Settings and the decision rule were frozen in `results/{NAME}/settings.json` before any training "
          f"(sha256 {(RES / 'settings.sha256').read_text().strip()[:12]}…).", "",
          "## Why", "",
          "The joint pilot's outcome head predicted final schedule outcomes under fixed continuation heuristics, so part of "
          "what it learned was the heuristics' behaviour. Here the world model only learns what the factory looks like one "
          "decision later; whether that is good is left to the specialists.", "",
          "## Architecture (E)", "",
          "`S_t -> DANIEL encoder -> action-conditioned latent predictor -> predicted S_t+1 tokens -> physical head`.",
          "- Physical head reads ONLY the predicted tokens: per machine (predicted machine token + predicted globals) -> free "
          "time, cumulative processing; per job (mean + max of its predicted operation tokens + predicted globals) -> ready "
          "time, remaining workload; globals -> decision-time advance. No continuation rule, no S_t, no raw action features.",
          "- Latent target: EMA copy of the online encoder (tau 0.996), stop-gradient, on the actual S_t+1; variance hinge.",
          f"- Loss: L_lat/L_lat0 + variance hinge + lambda_phys x L_phys/L_phys0; L_phys = mean over the 5 target groups of "
          f"Huber(after-value) + Huber(sibling difference). One-off calibration after epoch 1: "
          f"{json.dumps(T['E']['info'].get('calibration'))}.",
          f"- Checks before training: perturbing every S_t+1 array and every target leaves all physical predictions "
          f"bit-identical ({T['E']['info']['leakage_check_next_state_and_target_perturbation_changes_nothing']}); both terms reach "
          f"the encoder ({T['E']['info']['both_terms_reach_encoder']}); predicted latent not detached "
          f"({T['E']['info']['predicted_latent_requires_grad']}).",
          "- B: identical encoder + predictor + physical head with the physical loss only. A_phys: same, encoder frozen at "
          "the random start. raw: linear token maps of the raw operation/machine features instead of the encoder. C, D: "
          "the joint pilot's checkpoints (latent only; joint with final outcomes), not retrained.", "",
          "## Immediate physical targets", "",
          "From the semi-active simulator step: start = max(job ready, machine free); completion = start + p; the chosen "
          "machine's free time and the chosen job's ready time become the completion; operation completion lower bounds of "
          "the job shift; the decision time advances to the earliest pair start. Carbon is processing energy only "
          "(no time-of-use term).", "",
          "| group | level | target (minutes) |", "|---|---|---|"]
    for g in GROUPS:
        L_.append(f"| {g} | {GROUP_LEVEL[g]} | {GROUP_NAMES[g]}" + (" relative to t_t" if g in "FR" else "") + " |")
    L_ += ["", "Dropped as redundant (identities checked on every extracted transition):", ""]
    L_ += [f"- {k}: {v}" for k, v in st["dropped_as_redundant"].items()]
    L_ += ["", "Max identity / cross-check errors (minutes) over all extracted transitions: " +
           ", ".join(f"{k} {max(xo['max_err'][k], xf['max_err'][k]):.1e}" for k in xo["max_err"]) +
           f"; stored S_t/S_t+1 arrays reproduced exactly (mismatches: {xo['stored_mismatch'] + xf['stored_mismatch']}).",
           "The change of every target is therefore elementary simulator arithmetic given the raw physical state "
           "(exact analytic baseline: error 0); the question is what the ENCODER learns from predicting it through the latent.", "",
           "## Data and splits", "",
           "| split | factories | states | transitions | branches/state | same-op group | other-op same-machine | waiting | early/mid/late |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k in ("train", "validation", "fresh test"):
        v = cov[k]
        L_.append(f"| {k} | {v['factories']} | {v['states']:,} | {v['transitions']:,} | {v['branches_per_state']:.2f} | "
                  f"{f3(v['share_in_same_operation_group'])} | {f3(v['share_in_other_operation_same_machine_group'])} | "
                  f"{f3(v['share_waiting'])} | {'/'.join(f'{x:.2f}' for x in v['progress_share_early_mid_late'])} |")
    L_ += ["", f"- Train and validation: the joint pilot's collection (re-used; physical targets extracted by replay in "
               f"{xo['seconds']:.0f} s). The joint pilot's test factories are not used.",
           f"- Fresh test: {C['fresh_instances']} new generator factories (seed {C['fresh_seed']}), none duplicating any joint-pilot "
           f"factory; collected with the joint pilot's routine (replay and decoder checks passed); extraction {xf['seconds']:.0f} s.",
           "- Selection and monitoring on validation only; every number below is on the fresh test unless stated.", "",
           "## Training and convergence", "",
           "| config | lr | epochs run | seconds | peak GPU GB | at 60 epochs | at stop | still improving at stop |", "|---|---|---|---|---|---|---|---|"]
    for k in ("E", "B", "A_phys", "raw"):
        i = T[k]["info"]
        L_.append(f"| {k} | {i['lr']:g} | {i['epochs_run']} | {i['train_seconds']:.0f} | {f3(i['peak_gpu_gb'])} | "
                  f"{T[k]['matched']['convergence']['verdict']} | {T[k]['convergence']['verdict']} | "
                  f"{', '.join(T[k]['convergence'].get('still_improving', [])) or '-'} |")
    g = T["E"]["gradients"]; gd = TD["gradients"]
    pick = lambda gs, e: min(gs, key=lambda x: abs(x["epoch"] - e))
    L_ += ["", "Gradient balance on the encoder (median over 5 monitor batches; ratio = latent / measurable term):", "",
           "| model | epoch 1 ratio | epoch 1 cosine | epoch 30 ratio | epoch 60 ratio | epoch 60 cosine | last epoch ratio | last cosine |",
           "|---|---|---|---|---|---|---|---|",
           f"| E (immediate physical) | {f2(g[0]['ratio_latent_to_physical'])} | {f2(g[0]['cosine'])} | {f2(pick(g, 30)['ratio_latent_to_physical'])} | "
           f"{f2(pick(g, 60)['ratio_latent_to_physical'])} | {f2(pick(g, 60)['cosine'])} | {f2(g[-1]['ratio_latent_to_physical'])} | {f2(g[-1]['cosine'])} |",
           f"| D (final outcome, joint pilot) | {f2(gd[0]['ratio_latent_to_outcome'])} | {f2(gd[0]['cosine'])} | {f2(pick(gd, 30)['ratio_latent_to_outcome'])} | "
           f"{f2(pick(gd, 60)['ratio_latent_to_outcome'])} | {f2(pick(gd, 60)['cosine'])} | {f2(gd[-1]['ratio_latent_to_outcome'])} | {f2(gd[-1]['cosine'])} |",
           "", "Per-group validation physical loss of E at epoch 60: " +
           ", ".join(f"{g_} {T['E']['matched']['history'][-1][f'val_phys_{g_}']:.4f}" for g_ in GROUPS) + ".", "",
           "![curves](world_model_immediate_pilot_v3/curves.png)", "",
           "## Results (fresh test factories, matched 60-epoch budget)", "",
           "### Immediate physical prediction", "",
           "Change MAE / R^2 on the entities the action changes (chosen machine for F and B, chosen job for R and W, the clock "
           "for T); copy-current = predicting no change.", ""]
    names = [k for k in ("E", "B", "A_phys", "raw", "E (converged)", "B (converged)") if k in R["physical"]]
    L_ += ["| model | " + " | ".join(f"{g} change MAE / R^2" for g in GROUPS) + " | mean change MAE |", "|---|" + "---|" * (len(GROUPS) + 1)]
    for k in names:
        p = R["physical"][k]
        L_.append(f"| {k} | " + " | ".join(f"{f2(p[g]['change_mae_changed'])} / {f3(p[g]['change_r2_changed'])}" for g in GROUPS)
                  + f" | {f2(p['mean_change_mae_changed'])} |")
    p = R["physical"]["E"]
    L_.append("| copy current (no change) | " + " | ".join(f"{f2(p[g]['copy_change_mae_changed'])} / -" for g in GROUPS) +
              f" | {f2(np.mean([p[g]['copy_change_mae_changed'] for g in GROUPS]))} |")
    L_.append("| analytic simulator arithmetic | " + " | ".join("0 / 1" for _ in GROUPS) + " | 0 |")
    L_ += ["", "After-value MAE (minutes) / R^2 over all entities, and error on entities the action does not change:", "",
           "| model | " + " | ".join(f"{g} after MAE / R^2" for g in GROUPS) + " | unchanged-entity MAE (F, R) |", "|---|" + "---|" * (len(GROUPS) + 1)]
    for k in names:
        p = R["physical"][k]
        L_.append(f"| {k} | " + " | ".join(f"{f2(p[g]['after_mae'])} / {f3(p[g]['after_r2'])}" for g in GROUPS) +
                  f" | {f2(p['F']['unchanged_mae'])}, {f2(p['R']['unchanged_mae'])} |")
    p = R["physical"]["E"]
    L_.append("| copy current | " + " | ".join(f"{f2(p[g]['copy_after_mae'])} / {f3(p[g]['copy_after_r2'])}" for g in GROUPS) + " | 0, 0 |")
    L_ += ["", "### Differences between alternative actions from the same state", "",
           "Sign accuracy of the predicted per-entity difference (where the true difference is non-zero) and difference MAE "
           "vs the zero-difference baseline (minutes).", "",
           "| model | pair type | " + " | ".join(f"{g} sign acc / MAE (zero)" for g in GROUPS) + " |", "|---|---|" + "---|" * len(GROUPS)]
    for k in [x for x in ("E", "B", "A_phys", "raw") if x in R["physical"]]:
        for t in ("all", "same_operation_other_machine", "other_operation_same_machine", "immediate_vs_waiting"):
            cells = []
            for g in GROUPS:
                s = R["physical"][k][g]["siblings"][t]
                cells.append("-" if s is None else f"{f3(s['sign_accuracy_nonzero'])} / {f2(s['difference_mae'])} ({f2(s['zero_difference_mae'])})")
            L_.append(f"| {k} | {t.replace('_', ' ')} | " + " | ".join(cells) + " |")
    L_ += ["", "### Latent transition prediction (own target space; scale-free)", "",
           "| config | change explained (changed tokens) | shuffle worse share | discrimination overall | same op / other machine | other op / same machine | immediate vs waiting |",
           "|---|---|---|---|---|---|---|"]
    for k, l in R["latent"].items():
        L_.append(f"| {k} | {f3(l['errors']['changed_tokens']['change_explained'])} | {f3(l['shuffle_worse_share_of_states'])} | "
                  f"{f3(l['discrimination']['overall'])} | {f3(l['discrimination']['same_operation_other_machine'])} | "
                  f"{f3(l['discrimination']['other_operation_same_machine'])} | {f3(l['discrimination']['immediate_vs_waiting'])} |")
    L_ += ["", "Copy-current-latent baseline = change explained 0 by definition. A = the joint pilot's predictor on the frozen random encoder.", "",
           "### Machine availability readouts (linear OLS; fitted on train transitions, scored on fresh)", ""]
    for title, key in (("From the ACTUAL next-state embedding (encoder of S_t+1)", "availability_actual"),
                       ("From the PREDICTED next latent (predictor output from S_t and the action)", "availability_predicted")):
        L_ += [f"{title}:", "", "| representation | absolute R^2 | absolute MAE (min) | same-op difference sign accuracy | difference MAE (min) | zero-baseline MAE |",
               "|---|---|---|---|---|---|"]
        for k, a in R[key].items():
            L_.append(f"| {k} | {f3(a['absolute_centred_free_time']['r2'])} | {f2(a['absolute_centred_free_time']['mae_minutes'])} | "
                      f"{f3(a['same_operation_branch_difference']['sign_accuracy_nonzero'])} | {f2(a['same_operation_branch_difference']['mae_minutes_nonzero'])} | "
                      f"{f2(a['same_operation_branch_difference']['mae_zero_baseline_nonzero'])} |")
        L_.append("")
    L_ += ["### Arithmetic or relational understanding? (frozen encoders of S_t, identical small MLP probes)", "",
           "None of these targets is given by the action: each combines a job's readiness with a machine's availability "
           "across the state. (i) earliest start of every eligible candidate job-machine pair; (ii) the machine giving the "
           "earliest completion for each candidate job; (iii) each machine's earliest start over candidate jobs.", "",
           "| encoder | pair earliest start R^2 (MAE min) | best machine accuracy (chance) | machine earliest start R^2 (MAE min) | mean |",
           "|---|---|---|---|---|"]
    for k, v in P["relational"].items():
        L_.append(f"| {k} | {f3(v['pair_earliest_start_r2'])} ({f2(v['pair_earliest_start_mae_minutes'])}) | "
                  f"{f3(v['best_machine_accuracy'])} ({f3(v['best_machine_chance'])}) | {f3(v['machine_earliest_start_r2'])} "
                  f"({f2(v['machine_earliest_start_mae_minutes'])}) | {f3(v['mean'])} |")
    L_ += ["", "### Secondary: final-outcome frozen probes (the joint pilot's protocol; not part of the decision)", "",
           "| encoder | ranking accuracy | same op / other machine | difference MAE | zero-difference MAE |", "|---|---|---|---|---|"]
    for k, v in P["final_outcome"].items():
        m_ = v["test"]["mean"]
        L_.append(f"| {k} | {f3(m_['accuracy'])} | {f3(m_['same_operation_accuracy'])} | {f2(m_['mae_difference'])} | {f2(m_['mae_difference_zero_baseline'])} |")
    rb = P["final_outcome_rule_baseline"]
    L_ += ["", f"Simple pair rule on same-operation pairs: makespan {f3(rb['mean|makespan']['same_operation_accuracy'])}, "
               f"carbon {f3(rb['mean|carbon']['same_operation_accuracy'])}, tardiness {f3(rb['mean|tardiness']['same_operation_accuracy'])}.", "",
           "### Collapse diagnostics (target / online encoder on fresh next states)", "",
           "| config | effective rank | mean std | min std |", "|---|---|---|---|"]
    for k, c in R["collapse"].items():
        L_.append(f"| {k} | {c['effective_rank']:.2f} | {c['mean_std']:.4f} | {c['min_std']:.4f} |")
    L_ += [f"", f"E at the start: effective rank {T['E']['info']['collapse_start']['effective_rank']:.2f}, mean std "
                f"{T['E']['info']['collapse_start']['mean_std']:.4f}; flagged collapsed: {T['E']['matched']['history'][-1].get('collapsed')}.", "",
           "## Pre-registered decision", "", RULE_TEXT, "", "```", json.dumps(dec, indent=1), "```", "",
           f"**Verdict: {'E is recommended as the shared encoder for specialist training.' if dec['recommend_E_for_specialist_training'] else 'E is not (yet) recommended as the shared encoder for specialist training.'}**", "",
           "## Resources", "",
           f"- Physical-target extraction: {xo['seconds']:.0f} s (train+validation), {xf['seconds']:.0f} s (fresh); fresh collection "
           f"{cov.get('fresh_collection_seconds', {}).get('total_with_instance_generation', 0):.0f} s.",
           "- Training: " + "; ".join(f"{k} {T[k]['info']['train_seconds'] / 60:.0f} min ({T[k]['info']['epochs_run']} epochs), peak GPU "
                                      f"{f3(T[k]['info']['peak_gpu_gb'])} GB" for k in T) + ".",
           f"- Evaluation {R['eval_seconds'] / 60:.0f} min; peak RAM {R['peak_rss_gb']:.1f} GB.", "",
           "## Limitations", "",
           "- One seed, pilot scale; differences of a few hundredths are within run-to-run noise not measured here.",
           "- The immediate targets are deterministic arithmetic of the raw physical state; a good fit says the representation "
           "carries the needed quantities, not that it learned anything beyond them. The relational probes and the A_phys / raw "
           "controls are the evidence on that question.",
           "- DANIEL's observations are z-scored per step, so absolute times are only partly recoverable from S_t by any encoder.",
           "- C and D were trained for 60 epochs and not continued; the matched-budget comparison is at 60 epochs.",
           "- The predictor uses the observed pair features of S_t; multi-step rollouts on imagined states would need them predicted.", ""]
    (REPO / "results" / f"{NAME}{SFX}.md").write_text("\n".join(L_) + "\n")
    log(f"report written; recommend E: {dec['recommend_E_for_specialist_training']}")


def main():
    stages = _ARGS or ["all"]
    t0 = time.time()
    for st in (["collect", "train", "probe", "evaluate", "report"] if stages == ["all"] else stages):
        log(f"stage {st}")
        {"collect": collect, "train": train, "probe": probe, "evaluate": evaluate, "report": report}[st]()
    log(f"done ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
