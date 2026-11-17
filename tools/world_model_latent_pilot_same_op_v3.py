"""Reanalysis of the latent next-state pilot's same-operation, different-machine pairs (52% discrimination).

Saved data, model and held-out factories only (results/world_model_latent_pilot_v3, data/world_model_latent_pilot_v3,
checkpoints/world_model_latent_pilot_v3). The instance matrices (processing times, carbon, machine power) are
regenerated in memory from the pilot's generator seed and checked against the saved structural factory hashes.

  1. Reproduce the original discrimination accuracy on same-operation pairs (the evaluate() code, unchanged).
  2. Classify every held-out pair from SIMULATOR facts only (no model predictions):
       interchangeable       swapping the two machine labels maps one branch's next observation onto the other's
                             (all observation arrays, machine axes permuted) AND every operation still to be scheduled
                             has the same eligibility, processing time and carbon on both machines, and the machines'
                             power is equal
       meaningfully different the operation's finish time or its energy differs between the two machines, or another
                             job's next operation eligible on either machine gets a different earliest start
       unclear               neither established from the saved data (reason recorded)
  3. If same-operation accuracy on meaningfully different pairs stays near chance: a frozen-encoder probe. A shared
     readout (MLP, token only) on the final target encoder's ACTUAL next-state machine tokens -> that machine's next
     available time, fitted on the training factories; on held-out pairs, does it recover the availability difference
     between the two branches? The same readout is also applied to the predictor's predicted machine tokens.

    python tools/world_model_latent_pilot_same_op_v3.py   # -> results/world_model_latent_pilot_v3/same_op_reanalysis.json
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
from common_utils import setup_seed                                 # noqa: E402
from model.main_model import DualAttentionNetwork                  # noqa: E402

OUT = L.RES / "same_op_reanalysis.json"
CHANCE_BAND = 0.05                  # "near chance": |accuracy - 0.5| <= 0.05
TOL_T, TOL_E = 1e-9, 1e-12


def regenerate():
    from benchmark_v3_generator import GeneratorSource
    setup_seed(L.C["seed"])
    gen = GeneratorSource(L.configs, L.W.SIZE, L.B.load_manifest())
    return gen.sample(L.C["instances"])


def swap_equal(Dt, i, j, m1, m2):
    """Next observation of branch i equals that of branch j with machines m1 and m2 swapped."""
    p = np.arange(Dt["n_fea_m"].shape[1]); p[[m1, m2]] = [m2, m1]
    t = 1e-6
    return (np.allclose(Dt["n_fea_j"][i], Dt["n_fea_j"][j], atol=t)
            and np.array_equal(Dt["n_candidate"][i], Dt["n_candidate"][j])
            and np.array_equal(Dt["n_op_mask"][i], Dt["n_op_mask"][j])
            and np.allclose(Dt["n_fea_m"][i], Dt["n_fea_m"][j][p], atol=t)
            and np.allclose(Dt["n_fea_pairs"][i], Dt["n_fea_pairs"][j][:, p], atol=t)
            and np.array_equal(Dt["n_dynamic_pair_mask"][i], Dt["n_dynamic_pair_mask"][j][:, p])
            and np.array_equal(Dt["n_mch_mask"][i], Dt["n_mch_mask"][j][p][:, p])
            and np.allclose(Dt["n_comp_idx"][i], Dt["n_comp_idx"][j][p][:, p], atol=t))


def main():
    t0 = time.time()
    cfg, dev = L.setup()
    Dt = L.load_data()
    data = regenerate()
    assert [data.keys[i] for i in range(len(data))] == list(Dt["factory"]), "regenerated factories differ"
    J, M = L.B.SIZES[L.W.SIZE]
    tr_rows, ho = L.split_rows(Dt)
    d = L.configs.layer_fea_output_dim[-1]
    trunk = DualAttentionNetwork(L.configs).to(dev); trunk.load_state_dict(torch.load(L.CK / "online_encoder.pth", map_location=dev))
    tgt = DualAttentionNetwork(L.configs).to(dev); tgt.load_state_dict(torch.load(L.CK / "target_encoder.pth", map_location=dev))
    pred = L.Predictor(d, Dt["s_fea_pairs"].shape[-1], L.C["pred_hidden"], L.C["pred_heads"]).to(dev)
    pred.load_state_dict(torch.load(L.CK / "predictor.pth", map_location=dev))
    trunk.eval(); tgt.eval(); pred.eval()

    def tokens(rows, predicted):
        out = []
        with torch.no_grad():
            for b0 in range(0, len(rows), 512):
                tr = rows[b0:b0 + 512]
                if predicted:
                    (po, pm, _), _ = L.predict(trunk, pred, Dt, tr, Dt["tr_action"][tr], dev)
                else:
                    po, pm, _ = L.targets(tgt, Dt, tr, dev)
                out.append(torch.cat([po, pm], 1).cpu().numpy())
        return np.concatenate(out)
    Z1, ZH = tokens(ho, False), tokens(ho, True)
    N = Dt["s_fea_j"].shape[1]

    # ---- 1. reproduce: the evaluate() pair loop, same-operation pairs
    st = Dt["tr_state"]; st_ho = st[ho]; a = Dt["tr_action"][ho]
    pairs = []
    for s in np.unique(st_ho):
        k = np.flatnonzero(st_ho == s)
        for x in range(len(k)):
            for y in range(x + 1, len(k)):
                i, j = k[x], k[y]
                if np.sqrt(((Z1[i] - Z1[j]) ** 2).mean()) <= L.C["eval_tol"] or a[i] // M != a[j] // M:
                    continue
                ok_i = ((ZH[i] - Z1[i]) ** 2).mean() < ((ZH[i] - Z1[j]) ** 2).mean()
                ok_j = ((ZH[j] - Z1[j]) ** 2).mean() < ((ZH[j] - Z1[i]) ** 2).mean()
                pairs.append((i, j, ok_i, ok_j))
    acc = float(np.mean([p[2] for p in pairs] + [p[3] for p in pairs]))
    saved = json.loads((L.RES / "heldout.json").read_text())["discrimination"]["same_operation_other_machine"]
    R = {"reproduction": {"accuracy": acc, "unordered_pairs": len(pairs), "saved_accuracy": saved["accuracy"],
                          "saved_unordered_pairs": saved["unordered_pairs"],
                          "identical": abs(acc - saved["accuracy"]) < 1e-12 and len(pairs) == saved["unordered_pairs"]}}
    print("reproduction", R["reproduction"], flush=True)

    # ---- 2. classify by simulator facts
    rows = []
    for i, j, ok_i, ok_j in pairs:
        gi, gj = ho[i], ho[j]
        s = st[gi]; inst = int(Dt["state_episode_inst"][s])
        job = int(a[i] // M); m1, m2 = int(a[i] % M), int(a[j] % M)
        op = int(Dt["s_candidate"][s, job])
        pt = np.asarray(data.op_pt[inst], dtype=float); cb = np.asarray(data.carbon[inst], dtype=float)
        pw = np.asarray(data.power[inst], dtype=float)
        fin_i, fin_j = Dt["p1_op_ct_lb"][gi, op], Dt["p1_op_ct_lb"][gj, op]
        d_fin = float(fin_i - fin_j)
        d_en = float(cb[op, m1] - cb[op, m2])
        free_i, free_j = Dt["p1_mch_free_time"][gi], Dt["p1_mch_free_time"][gj]
        # other jobs' next operations eligible on m1 or m2: earliest start max(job ready, machine free) per branch
        delayed = []
        mask = Dt["s_dynamic_pair_mask"][s]
        first = Dt["s_job_first_op_id"][s]
        for jj in range(J):
            if jj == job:
                continue
            o2 = int(Dt["s_candidate"][s, jj])
            if Dt["p0_op_scheduled_flag"][gi, o2]:
                continue                                                     # job finished
            ready = 0.0 if o2 == first[jj] else Dt["p0_op_ct_lb"][gi, o2 - 1]
            for m in (m1, m2):
                if mask[jj, m]:
                    continue
                si, sj = max(ready, free_i[m]), max(ready, free_j[m])
                if abs(si - sj) > TOL_T:
                    delayed.append({"job": jj, "machine": m, "start_branch_i": float(si), "start_branch_j": float(sj)})
        reasons = []
        if abs(d_fin) > TOL_T:
            reasons.append("finish_time")
        if abs(d_en) > TOL_E:
            reasons.append("energy")
        if delayed:
            reasons.append("other_job_start")
        # interchangeability: observation swap symmetry + remaining operations symmetric + equal power
        rem = Dt["p1_op_scheduled_flag"][gi] == 0
        same_future = (np.allclose(pt[rem, m1], pt[rem, m2]) and np.allclose(cb[rem, m1], cb[rem, m2]))
        obs_swap = swap_equal(Dt, gi, gj, m1, m2)
        if reasons:
            cls = "meaningfully_different"
        elif obs_swap and same_future and abs(pw[m1] - pw[m2]) < 1e-12:
            cls = "interchangeable"
        else:
            cls = "unclear"
        why = None if cls != "unclear" else ("no immediate difference; remaining operations differ on the two machines"
                                              if not same_future else "no immediate difference; observation not swap-symmetric"
                                              if not obs_swap else "no immediate difference; machine power differs")
        rows.append({"i": i, "j": j, "ok": [bool(ok_i), bool(ok_j)], "class": cls, "reasons": reasons, "why_unclear": why,
                     "d_finish": d_fin, "d_energy": d_en, "delayed": delayed, "inst": inst, "state": int(s),
                     "job": job, "op": op, "m1": m1, "m2": m2, "pt": [float(pt[op, m1]), float(pt[op, m2])],
                     "carbon": [float(cb[op, m1]), float(cb[op, m2])], "free_before": Dt["p0_mch_free_time"][gi, [m1, m2]].tolist(),
                     "free_after_i": free_i[[m1, m2]].tolist(), "free_after_j": free_j[[m1, m2]].tolist(),
                     "true_dist": float(np.sqrt(((Z1[i] - Z1[j]) ** 2).mean())),
                     "pred_dist": float(np.sqrt(((ZH[i] - ZH[j]) ** 2).mean()))})

    def grp(sel):
        if not sel:
            return {"unordered_pairs": 0, "accuracy": None}
        return {"unordered_pairs": len(sel), "accuracy": float(np.mean(sum((r["ok"] for r in sel), []))),
                "median_true_branch_distance": float(np.median([r["true_dist"] for r in sel])),
                "median_pred_branch_distance": float(np.median([r["pred_dist"] for r in sel]))}
    R["classes"] = {c: grp([r for r in rows if r["class"] == c]) for c in ("interchangeable", "meaningfully_different", "unclear")}
    R["unclear_reasons"] = {w: sum(r["why_unclear"] == w for r in rows) for w in {r["why_unclear"] for r in rows} if w}
    md = [r for r in rows if r["class"] == "meaningfully_different"]
    combos = {}
    for r in md:
        combos.setdefault("+".join(r["reasons"]), []).append(r)
    R["meaningful_by_what_changed"] = {k: grp(v) for k, v in sorted(combos.items(), key=lambda kv: -len(kv[1]))}
    R["meaningful_by_single_fact"] = {f: grp([r for r in md if f in r["reasons"]]) for f in ("finish_time", "energy", "other_job_start")}
    # by size of the finish-time difference (env time units; quartiles of |difference| among pairs where it differs)
    fd = [r for r in md if "finish_time" in r["reasons"]]
    if fd:
        q = np.quantile([abs(r["d_finish"]) for r in fd], [0.25, 0.5, 0.75])
        b = np.digitize([abs(r["d_finish"]) for r in fd], q)
        R["finish_time_difference_quartiles"] = {"edges": q.tolist(),
                                                 "accuracy_by_quartile": [grp([r for r, k in zip(fd, b) if k == x]) for x in range(4)]}
    ed = [r for r in md if "energy" in r["reasons"]]
    rel = np.array([abs(r["d_energy"]) / max(r["carbon"]) for r in ed])
    q = np.quantile(rel, [0.25, 0.5, 0.75])
    b = np.digitize(rel, q)
    R["energy_relative_difference_quartiles"] = {"edges": q.tolist(),
                                                 "accuracy_by_quartile": [grp([r for r, k in zip(ed, b) if k == x]) for x in range(4)]}
    near = R["classes"]["meaningfully_different"]["accuracy"] is not None and \
        abs(R["classes"]["meaningfully_different"]["accuracy"] - 0.5) <= CHANCE_BAND
    R["meaningful_near_chance"] = bool(near)
    # example: largest finish-time difference + delayed other job, both directions missed
    cands = sorted([r for r in md if not any(r["ok"]) and "finish_time" in r["reasons"] and r["delayed"]],
                   key=lambda r: -abs(r["d_finish"]))
    if cands:
        r = cands[0]
        R["example_missed"] = {k: r[k] for k in ("inst", "state", "job", "op", "m1", "m2", "pt", "carbon", "d_finish",
                                                "d_energy", "free_before", "free_after_i", "free_after_j", "delayed",
                                                "true_dist", "pred_dist", "ok")}
        R["example_missed"]["source"] = L.SOURCES[int(Dt["source"][r["state"]])]

    # ---- 3. frozen-encoder probe (only if near chance on meaningfully different pairs)
    if near:
        tp = time.time()
        Ztr = tokens(tr_rows, False)[:, N:]                                         # actual next-state machine tokens
        scale = float(np.abs(Dt["p1_mch_free_time"][tr_rows]).max())
        Ytr = Dt["p1_mch_free_time"][tr_rows] / scale
        setup_seed(111)
        net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                  torch.nn.Linear(64, 1)).to(dev)
        X = torch.as_tensor(Ztr.reshape(-1, d), dtype=torch.float32, device=dev)
        Y = torch.as_tensor(Ytr.reshape(-1, 1), dtype=torch.float32, device=dev)
        rng = np.random.default_rng(111)
        val = rng.random(len(tr_rows)) < 0.1                                         # by transition, train factories only
        vm = torch.as_tensor(np.repeat(val, M), device=dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        best, best_sd, bad, hist = np.inf, None, 0, []
        for ep in range(200):
            perm = torch.randperm(int((~vm).sum()), device=dev)
            Xt, Yt = X[~vm][perm], Y[~vm][perm]
            for b0 in range(0, len(Xt), 4096):
                loss = ((net(Xt[b0:b0 + 4096]) - Yt[b0:b0 + 4096]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = float(((net(X[vm]) - Y[vm]) ** 2).mean())
            hist.append(v)
            if v < best - 1e-7:
                best, best_sd, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 15:
                    break
        net.load_state_dict(best_sd); net.eval()

        def read(Zm):
            with torch.no_grad():
                return net(torch.as_tensor(Zm.reshape(-1, d), dtype=torch.float32, device=dev)).cpu().numpy().reshape(-1, M) * scale
        Yho = Dt["p1_mch_free_time"][ho]
        P_act, P_pred = read(Z1[:, N:]), read(ZH[:, N:])
        r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())   # noqa: E731
        probe = {"epochs": len(hist), "train_val_mse": best, "time_scale": scale,
                 "heldout_r2_free_time": {"actual_tokens": r2(P_act, Yho), "predicted_tokens": r2(P_pred, Yho)}}
        for name, sel in (("meaningfully_different", md), ("finish_time_differs", fd)):
            out = {}
            for src, P in (("actual_tokens", P_act), ("predicted_tokens", P_pred)):
                td, pdd = [], []
                for r in sel:
                    for m in (r["m1"], r["m2"]):
                        t_ = Yho[r["i"], m] - Yho[r["j"], m]
                        if abs(t_) > TOL_T:
                            td.append(t_); pdd.append(P[r["i"], m] - P[r["j"], m])
                td, pdd = np.array(td), np.array(pdd)
                out[src] = {"machine_pairs_with_availability_difference": int(len(td)),
                            "sign_accuracy": float((np.sign(pdd) == np.sign(td)).mean()) if len(td) else None,
                            "pearson": float(np.corrcoef(td, pdd)[0, 1]) if len(td) > 2 else None,
                            "median_abs_true_difference": float(np.median(np.abs(td))) if len(td) else None,
                            "median_abs_recovered_difference": float(np.median(np.abs(pdd))) if len(td) else None}
            probe[name] = out
        probe["seconds"] = time.time() - tp
        R["frozen_encoder_probe"] = probe
    R["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(R, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in R.items() if k not in ("example_missed",)}, indent=1, default=float))
    print("example", json.dumps(R.get("example_missed"), default=float))


if __name__ == "__main__":
    main()
