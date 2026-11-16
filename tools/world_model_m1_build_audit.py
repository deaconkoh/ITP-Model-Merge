"""Audit of how the M1 world model was built: is the near-chance action ranking an implementation bug, missing action
information, or a loss that gives little incentive to learn action effects? Inference, small simulator replays and
analysis only, on saved artifacts (checkpoint and validation data). No training; the reserved final-test sets are
never loaded. Plan: ~/.claude/plans (M1 build audit).

1. Targets: the saved validation states are reproduced exactly by re-running the generation's behaviour episodes
   (frozen val_seed); 12 state-action pairs are then branched in the REAL environment (each branch its own row,
   prefix replayed, forced action, finished by the rule chosen by the generation's decoder in lockstep, masks
   asserted equal at every step) and the targets rebuilt from the environment's own arrays.
2. The action through the model: readout indices, action-embedding differences, head inputs, sensitivity, accuracy
   for same-job vs different-job pairs, and how much a missing pair feature (processing time, energy of the chosen
   pair) would rank by itself.
3. The loss: variance split and Huber loss of a mean predictor, a state-only oracle and the trained model.

    python tools/world_model_m1_build_audit.py        # -> results/world_model_m1_v3/build_audit.json
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.argv = sys.argv[:1]
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import world_model_m1_v3 as W                                       # noqa: E402
import world_model as WMmod                                         # noqa: E402

B = W.B
OUT = REPO / "results" / "world_model_m1_v3" / "build_audit.json"


def objectives(jc, en, due):
    return {"makespan": jc.max(-1), "carbon": en.sum(-1), "tardiness": np.maximum(0.0, jc - due).sum(-1)}


# ------------------------------------------------------------------------------------------------ 1. targets

def reproduce_states(cfg, V):
    data = B.load_split("validation", W.SIZE)
    n = min(len(data), cfg["val_instances"])
    rng = np.random.default_rng(cfg["val_seed"])
    idx = np.repeat(np.arange(n), len(W.BEHAVIOURS))
    beh = np.tile(np.arange(len(W.BEHAVIOURS)), n)
    N = data.op_pt[0].shape[0]
    steps = [np.arange(0, N - 1, cfg["val_step_stride"]) for _ in idx]
    states, rec, ep_actions = W.behaviour_episodes(data, idx, beh, steps, cfg["n_candidates"], rng)
    same = {k: bool(np.array_equal(states[k], V[k])) for k in W.STATE_KEYS}
    same["rec_cand"] = bool(np.array_equal(rec["cand"], V["rec_cand"]))
    same["rec_step"] = bool(np.array_equal(rec["step"], V["rec_step"]))
    same["instance"] = bool(np.array_equal(idx[rec["episode"]], V["instance"]))
    return data, idx, rec, ep_actions, same


def pick_pairs(V, rng, per_rule=4):
    """Per familiar rule: states where two legal candidates differ in recorded makespan by >= the upper quartile."""
    valid = V["rec_cand"] >= 0
    out = []
    for r in range(3):
        mk = V["tj"][:, :, r].max(-1)
        rows = []
        for i, j in itertools.combinations(range(valid.shape[1]), 2):
            m = valid[:, i] & valid[:, j]
            rows += [(s, i, j, abs(mk[s, i] - mk[s, j])) for s in np.flatnonzero(m)]
        q = np.quantile([x[3] for x in rows], 0.75)
        big = [x for x in rows if x[3] >= q]
        for k in rng.choice(len(big), per_rule, replace=False):
            s, i, j, d = big[k]
            out.append({"state": int(s), "a1": int(i), "a2": int(j), "rule": r, "recorded_makespan_diff": float(d)})
    return out


def replay(data, inst, prefix, k, action, rule):
    """Real-environment branch: prefix (k steps), the forced action, then `rule` (the generation decoder's choice,
    masks asserted equal every step). Returns rebuilt targets and the schedule."""
    env = W.T.FJSPEnvForSameOpNums(*B.SIZES[W.SIZE])
    env.set_initial_data(**data.env_kwargs([inst], W.configs.goal))
    d = W.Decoder(data, np.array([inst])); d.reset()
    N, M = d.N, d.M
    machine = np.full(N, -1)
    for t in range(N):
        ok, nst = d.allowed()
        if not np.array_equal(ok, ~env.dynamic_pair_mask.astype(bool)):
            raise AssertionError("decoder and environment masks disagree during a replay")
        a = prefix[t] if t < k else (action if t == k else d.rule(rule, ok, nst)[0])
        op = env.candidate[0, a // M]
        machine[op] = a % M
        d.step(np.array([a])); env.step(np.array([a]))
    jl = np.asarray(data.job_length[inst]); last = np.cumsum(jl) - 1
    pt_raw = np.asarray(data.op_pt[inst], dtype=float); cb = np.asarray(data.carbon[inst])
    comp = env.true_op_ct[0]
    p = pt_raw[np.arange(N), machine]
    busy = np.array([p[machine == m].sum() for m in range(M)])
    energy = np.array([cb[np.arange(N), machine][machine == m].sum() for m in range(M)])
    start = comp - p
    ind = B.schedule_objectives(jl, pt_raw, cb, np.asarray(data.due[inst]), machine, start)
    return comp[last], busy, energy, ind


# ------------------------------------------------------------------------------------------------ main

def main():
    cfg = W.settings(False); W.set_features(cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; W.configs.device = dev
    V = W.load_split(False, "val")
    R = {}
    # ---- 1. targets
    data, idx, rec, ep_actions, same = reproduce_states(cfg, V)
    R["state_reproduction_exact"] = same
    if not all(same.values()):
        OUT.write_text(json.dumps(R, indent=1) + "\n")
        raise SystemExit(f"saved validation states not reproduced: {same}")
    man = B.load_manifest()
    R["due_matches_manifest"] = bool(all(np.allclose(V["due"][s], data.due[V["instance"][s]]) for s in range(len(V["due"]))))
    pairs = pick_pairs(V, np.random.default_rng(7))
    M = B.SIZES[W.SIZE][1]
    checks = []
    for p in pairs:
        s = p["state"]; e = rec["episode"][s]; k = int(rec["step"][s]); inst = int(idx[e]); rule = W.CONT[p["rule"]]
        for slot in ("a1", "a2"):
            a = int(V["rec_cand"][s, p[slot]])
            jc, busy, energy, ind = replay(data, inst, ep_actions[e], k, a, rule)
            tj = V["tj"][s, p[slot], p["rule"]]; tm = V["tm"][s, p[slot], p["rule"]]
            o = objectives(tj, tm[:, 1], V["due"][s])
            checks.append({**p, "slot": slot, "action": a, "job": a // M, "machine": a % M,
                           "max_abs_job_completion": float(np.abs(jc - tj).max()),
                           "max_abs_machine_busy": float(np.abs(busy - tm[:, 0]).max()),
                           "max_abs_machine_energy": float(np.abs(energy - tm[:, 1]).max()),
                           "objectives_vs_scorer": float(max(abs(o["makespan"] - ind["makespan"]), abs(o["carbon"] - ind["carbon"]),
                                                             abs(o["tardiness"] - ind["tardiness"])))})
    R["replays"] = {"schedules": len(checks),
                    "max_abs_diff": {k: max(c[k] for c in checks) for k in ("max_abs_job_completion", "max_abs_machine_busy",
                                                                           "max_abs_machine_energy", "objectives_vs_scorer")},
                    "pairs": checks}
    # normalisation of the training shards (stored normalised? the pretraining divides at load time)
    tr0 = np.load(sorted((W.paths(False)["data"] / "train").glob("shard_*.npz"))[0])
    R["train_shard_units"] = {"tj_max": float(np.nanmax(tr0["tj"])), "tm_busy_max": float(np.nanmax(tr0["tm"][..., 0])),
                              "tm_energy_max": float(np.nanmax(tr0["tm"][..., 1])), "t_norm": cfg["t_norm"],
                              "e_norm": cfg["e_norm"], "note": "raw units in the shards; pretrain() divides by t_norm/e_norm"}
    # ---- 2. the action through the model
    model = W.load_world_model(False, dev)
    valid = V["rec_cand"] >= 0
    S, A = valid.shape
    with torch.no_grad():
        pj, pm, emb = [], [], []
        for b0 in range(0, S, 256):
            sel = np.arange(b0, min(S, b0 + 256))
            st, job, mch, _ = W.batch_tensors(V, sel, dev)
            a, h_job, h_mch = model.embed(st, job, mch)
            x, y = model(st, job, mch)
            pj.append(x.cpu().numpy()); pm.append(y.cpu().numpy()); emb.append(a.cpu().numpy())
            if b0 == 0:                                                # head-input structure on the first batch
                c = torch.eye(WMmod.N_RULES, device=dev)
                xj0 = torch.cat([h_job[:, None, None].expand(-1, A, 3, -1, -1),
                                 a[:, :, None, None].expand(-1, -1, 3, h_job.shape[1], -1),
                                 c[None, None, :, None, :].expand(len(sel), A, 3, h_job.shape[1], 3)], -1)
                d_ = h_job.shape[-1]
                R["head_input_check"] = {
                    "state_part_identical_across_actions": bool(torch.equal(xj0[:, 0, :, :, :d_], xj0[:, 1, :, :, :d_])),
                    "action_part_identical_across_rules": bool(torch.equal(xj0[:, :, 0, :, d_:5 * d_], xj0[:, :, 2, :, d_:5 * d_])),
                    "rule_part_only_varies_by_rule": bool(torch.equal(xj0[:, 0, :, :, 5 * d_:], xj0[:, 1, :, :, 5 * d_:]))}
    pj = np.concatenate(pj) * cfg["t_norm"]; pm = np.concatenate(pm); emb = np.concatenate(emb)
    pe = pm[..., 1] * cfg["e_norm"]
    # indexing: readout op = candidate[job]; that op is the operation the environment schedules for the action
    cand = V["candidate"].astype(int)
    job = np.where(valid, V["rec_cand"] // M, 0); mch = np.where(valid, V["rec_cand"] % M, 0)
    op_read = np.take_along_axis(cand, job, 1)
    ok_idx = []
    for c in checks:
        s = c["state"]; slot = 0 if c["slot"] == "a1" else 1
        ai = c["a1"] if slot == 0 else c["a2"]
        e = rec["episode"][s]
        env = W.T.FJSPEnvForSameOpNums(*B.SIZES[W.SIZE]); env.set_initial_data(**data.env_kwargs([int(idx[e])], W.configs.goal))
        for t in range(int(rec["step"][s])):
            env.step(np.array([ep_actions[e][t]]))
        ok_idx.append(int(env.candidate[0, c["job"]]) == int(op_read[s, ai]))
    R["readout_index_matches_environment"] = f"{sum(ok_idx)}/{len(ok_idx)}"
    # action-embedding differences and same-job pairs
    d = emb.shape[-1] // 4
    same_job, diffs_aop, diffs_a = [], [], []
    for i, j in itertools.combinations(range(A), 2):
        m = valid[:, i] & valid[:, j]
        same_job.append(job[m, i] == job[m, j])
        diffs_aop.append(np.abs(emb[m, i, :d] - emb[m, j, :d]).max(-1))
        diffs_a.append(np.linalg.norm(emb[m, i] - emb[m, j], axis=-1))
    same_job = np.concatenate(same_job); diffs_aop = np.concatenate(diffs_aop); diffs_a = np.concatenate(diffs_a)
    R["action_embedding"] = {"pairs": int(len(same_job)), "same_job_pairs": int(same_job.sum()),
                             "a_op_identical_when_same_job": bool((diffs_aop[same_job] == 0).all()),
                             "median_norm_diff_same_job": float(np.median(diffs_a[same_job])) if same_job.any() else None,
                             "median_norm_diff_other": float(np.median(diffs_a[~same_job])),
                             "median_norm_of_a": float(np.median(np.linalg.norm(emb[valid], axis=-1)))}
    # sensitivity, accuracy split, pair-feature ranking power
    inst = V["instance"]
    sens, split, pairfeat = {}, {}, {}
    for r, rule in enumerate(W.CONT):
        T = objectives(V["tj"][:, :, r], V["tm"][:, :, r, :, 1], V["due"][:, None, :])
        P = objectives(pj[:, :, r], pe[:, :, r], V["due"][:, None, :])
        for kname in T:
            dt, dp, sj, pt_better, en_better = [], [], [], [], []
            for i, j in itertools.combinations(range(A), 2):
                m = valid[:, i] & valid[:, j]
                dt.append(T[kname][m, i] - T[kname][m, j]); dp.append(P[kname][m, i] - P[kname][m, j])
                sj.append(job[m, i] == job[m, j])
                ss = np.flatnonzero(m)
                ops = op_read[ss, i]
                pti = np.array([data.op_pt[inst[s]][o, mch[s, i]] for s, o in zip(ss, ops)], float)
                ptj = np.array([data.op_pt[inst[s]][o, mch[s, j]] for s, o in zip(ss, op_read[ss, j])], float)
                eni = np.array([data.carbon[inst[s]][o, mch[s, i]] for s, o in zip(ss, ops)])
                enj = np.array([data.carbon[inst[s]][o, mch[s, j]] for s, o in zip(ss, op_read[ss, j])])
                pt_better.append(np.sign(pti - ptj)); en_better.append(np.sign(eni - enj))
            dt, dp, sj = np.concatenate(dt), np.concatenate(dp), np.concatenate(sj)
            ptb, enb = np.concatenate(pt_better), np.concatenate(en_better)
            use = np.abs(dt) > 1e-9
            okk = np.where(np.abs(dp) < 1e-9, 0.5, (np.sign(dp) == np.sign(dt)).astype(float))
            sens[f"{rule}|{kname}"] = {"sd_true_diff": float(dt[use].std()), "sd_pred_diff": float(dp[use].std()),
                                       "corr_pred_true_diff": float(np.corrcoef(dp[use], dt[use])[0, 1])}
            split[f"{rule}|{kname}"] = {"all_accuracy": float(okk[use].mean()), "same_job_accuracy": float(okk[use & sj].mean()) if (use & sj).any() else None,
                                        "same_job_pairs": int((use & sj).sum()),
                                        "different_job_accuracy": float(okk[use & ~sj].mean()),
                                        "different_job_pairs": int((use & ~sj).sum())}
            feat = ptb if kname != "carbon" else enb
            u2 = use & sj & (feat != 0)
            pairfeat[f"{rule}|{kname}"] = {
                "same_job_pairs_with_feature_difference": int(u2.sum()),
                "feature": "processing time on the chosen machine" if kname != "carbon" else "energy on the chosen machine",
                "lower_feature_is_better_share": float((np.sign(dt[u2]) == feat[u2]).mean()) if u2.any() else None}
    R["sensitivity"] = sens; R["accuracy_same_vs_different_job"] = split; R["pair_feature_ranking"] = pairfeat
    # reproduces the earlier diagnostic?
    diag = json.loads((REPO / "results/world_model_m1_v3/action_diagnostic.json").read_text())
    R["diagnostic_reproduced"] = {f"{rule}|{k}": [diag["results"][rule][k]["accuracy"], split[f"{rule}|{k}"]["all_accuracy"]]
                                  for rule in W.CONT for k in ("makespan", "carbon", "tardiness")}
    R["missing_information"] = {
        "fea_pairs (pair processing time and its 5 ratios, pair waiting time, pair energy, lateness)": "not in the readout; not stored in the dataset",
        "dynamic_pair_mask (legal pairs)": "not an input of the trunk or heads",
        "operation features": "z-scored per state (absolute scale removed; machine and global embeddings carry the rest)",
        "global context": "two mean-pooled embeddings"}
    # ---- 3. the loss
    tjn, tmn = W.normalise(cfg, V["tj"][:, :, :3], V["tm"][:, :, :3])
    pjn = pj / cfg["t_norm"]; pmn = np.stack([pm[..., 0], pm[..., 1]], -1)
    mean = json.loads((W.paths(False)["res"] / "mean_predictor.json").read_text())
    mj = np.asarray(mean["job"]) / cfg["t_norm"]; mm = np.asarray(mean["mch"]) / np.array([cfg["t_norm"], cfg["e_norm"]])
    R["loss"] = {"validation": loss_split(tjn, tmn, pjn, pmn, mj, mm, valid)}
    # training hold-out shards (read from disk), the same computation with the model's predictions
    files = sorted((W.paths(False)["data"] / "train").glob("shard_*.npz"))
    acc = []
    for f in files:
        z = np.load(f)
        if not z["holdout"].any():
            continue
        D = {k: z[k] for k in z.files}
        h = np.flatnonzero(D["holdout"])
        D = {k: v[h] if v.shape[:1] == D["holdout"].shape else v for k, v in D.items()}
        vj, vm = W.normalise(cfg, D["tj"], D["tm"])
        vv = D["rec_cand"] >= 0
        with torch.no_grad():
            qj, qm = [], []
            for b0 in range(0, len(h), 256):
                sel = np.arange(b0, min(len(h), b0 + 256))
                st, jb, mc, _ = W.batch_tensors(D, sel, dev)
                x, y = model(st, jb, mc)
                qj.append(x.cpu().numpy()); qm.append(y.cpu().numpy())
        acc.append(loss_split(vj, vm, np.concatenate(qj), np.concatenate(qm), mj, mm, vv))
        if len(acc) == 5:
            break
    R["loss"]["training_holdout_5_shards"] = {k: float(np.mean([a[k] for a in acc])) if not isinstance(acc[0][k], dict)
                                              else {kk: float(np.mean([a[k][kk] for a in acc])) for kk in acc[0][k]}
                                              for k in acc[0]}
    R["loss"]["note"] = ("Huber (delta 1) averaged over the 20 outputs, as world_model.loss; errors are << 1, so it is "
                         "0.5 x squared error. state-only oracle: each target replaced by its mean over the state's "
                         "valid actions (same rule), the lowest loss any predictor that ignores the action can reach.")
    OUT.write_text(json.dumps(R, indent=1, default=float) + "\n")
    print(json.dumps({k: R[k] for k in ("state_reproduction_exact", "due_matches_manifest", "readout_index_matches_environment",
                                        "head_input_check", "action_embedding")}, indent=1))
    print(json.dumps(R["replays"]["max_abs_diff"]), json.dumps(R["loss"], indent=1, default=float))


def loss_split(tj, tm, pj, pm, mj, mm, valid):
    """tj [S, A, R, J], tm [S, A, R, M, 2] normalised targets; pj, pm predictions; mj [R, J], mm [R, M, 2] means."""
    def hub(p, t):
        e = np.abs(p - t)
        return np.where(e <= 1, 0.5 * e ** 2, e - 0.5)
    w = valid[:, :, None]
    def oracle(t):
        m = np.where(valid[(...,) + (None,) * (t.ndim - 2)], t, np.nan)
        return np.broadcast_to(np.nanmean(m, axis=1, keepdims=True), t.shape)
    out = {}
    for name, pj_, pm_ in (("mean_predictor", np.broadcast_to(mj, tj.shape), np.broadcast_to(mm, tm.shape)),
                           ("state_only_oracle", oracle(tj), oracle(tm)), ("world_model", pj, pm)):
        hj = hub(pj_, tj).sum(-1); hm = hub(pm_, tm).sum((-1, -2))
        per = (hj + hm) / (tj.shape[-1] + tm.shape[-2] * tm.shape[-1])
        out[name] = float(per[np.broadcast_to(w, per.shape)].mean())
    groups = {}
    for gname, t in (("job completion", tj), ("machine busy", tm[..., 0]), ("machine energy", tm[..., 1])):
        m = np.where(valid[(...,) + (None,) * (t.ndim - 2)], t, np.nan)
        within = np.nanmean((m - np.nanmean(m, axis=1, keepdims=True)) ** 2)
        total = np.nanmean((m - np.nanmean(m)) ** 2)
        groups[gname] = within / total
    out["within_state_variance_share"] = groups
    out["share_of_mean_loss_removed_by_state_oracle"] = 1 - out["state_only_oracle"] / out["mean_predictor"]
    out["share_of_action_dependent_loss_removed_by_model"] = ((out["state_only_oracle"] - out["world_model"])
                                                            / out["state_only_oracle"])
    return out


if __name__ == "__main__":
    main()
