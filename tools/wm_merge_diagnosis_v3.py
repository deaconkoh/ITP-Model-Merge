"""Experiment 1: why the joint latent + physical shared base (D) merges worse than immediate physical alone (C).
Diagnostics on EXISTING checkpoints only (10x5, seed 111, waiting allowed); A (random) and B (outcome prediction)
are reference controls. Nothing here trains a world model or a specialist.

  probe     (pilot data; frozen encoders)  A representation drift (CKA, relative change) on the immediate pilot's fresh
            test factory states; B retention probes: relational (earliest start of every eligible pair, best machine,
            machine earliest start), machine-availability readout of S_t+1, current machine-free / job-ready readouts,
            immediate physical prediction with (i) the ORIGINAL pilot predictor + head on the specialist encoder (C, D)
            and (ii) a predictor + head refitted on the frozen encoder (identical budget for every encoder); D's original
            latent predictor on the specialist encoders.
  weights   A parameter drift (encoder / actor / critic, per layer) and C task vectors (cosine, sign agreement,
            magnitude) relative to each configuration's own shared base.
  interp    C pairwise weight interpolation: each configuration's merged start (Rewarded Soups, update 0) at
            preferences on the three edges of the simplex = theta(a) = (1-a) theta_i + a theta_j; greedy on the 18
            validation instances, scored with the preference-weighted normalised score of training (decoding_v3.score).
  matched   D merged starts with the tardiness specialist replaced by its snapshot at matched updates (500, 1000, 1500,
            2000; from the downstream tardiness runs, same trajectories) and the selected makespan / carbon specialists:
            validation HV for all, best-of-128 evaluation (100 x 66, seed 51) at 1000 and 2000.
  report    results/wm_merge_diagnosis_v3.md + figures

    python tools/wm_merge_diagnosis_v3.py {probe|weights|interp|matched|report|all}
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
NAME = "wm_merge_diagnosis_v3"
RES, CK = REPO / "results" / NAME, REPO / "checkpoints" / NAME
MD = REPO / "results" / f"{NAME}.md"
CKP = REPO / "checkpoints"
A_CK, B_CK = CKP / "specialist_reuse_v3", CKP / "specialist_reuse_wm_v3"
DS_CK, RS4_CK = CKP / "tardiness_wm_downstream_v3", CKP / "rs_four_bases_v3"
A_RES, B_RES, RS4_RES = (REPO / "results" / n for n in ("specialist_reuse_v3", "specialist_reuse_wm_v3", "rs_four_bases_v3"))
IP_CK = CKP / "world_model_immediate_pilot_v3"
OBJ = "mct"
ONAME = {"m": "makespan", "c": "carbon", "t": "tardiness"}
LABELS = {"A": "A random", "B": "B outcome prediction", "C": "C immediate physical", "D": "D joint latent + physical"}
CONF = {
    "A": {"base": A_CK / "base_s111.pth", "ds": "dsA", "runs": A_RES / "runs",
          "specs": {o: A_CK / "specialists" / f"10x5+mix+bv3_sr_spec_{o}_s111.pth" for o in OBJ},
          "merged0": CKP / "eval_curve_v3" / "snapshots" / "10x5+mix+bv3_ecv_rs_random_s111@u0.pth"},
    "B": {"base": B_CK / "base_s111.pth", "ds": "dsB", "runs": B_RES / "runs",
          "specs": {o: B_CK / "specialists" / f"10x5+mix+bv3_srwm_spec_{o}_s111.pth" for o in OBJ},
          "merged0": CKP / "eval_curve_v3" / "snapshots" / "10x5+mix+bv3_ecv_rs_world_model_s111@u0.pth"},
    "C": {"base": DS_CK / "start_dsC.pth", "ds": "dsC", "runs": RS4_RES / "C" / "runs",
          "specs": {"m": RS4_CK / "C/specialists/10x5+mix+bv3_srip_spec_m_s111.pth",
                    "c": RS4_CK / "C/specialists/10x5+mix+bv3_srip_spec_c_s111.pth",
                    "t": RS4_CK / "C/specialists/10x5+mix+bv3_wmtp_dsC_s111.pth"},
          "merged0": RS4_CK / "C/finals/10x5+mix+bv3_srip_rs_s111@u0.pth", "native": "B"},
    "D": {"base": DS_CK / "start_dsD.pth", "ds": "dsD", "runs": RS4_RES / "D" / "runs",
          "specs": {o: RS4_CK / f"D/specialists/10x5+mix+bv3_srjp_spec_{o}_s111.pth" for o in OBJ},
          "merged0": RS4_CK / "D/finals/10x5+mix+bv3_srjp_rs_s111@u0.pth", "native": "E"},
}
MATCHED = [500, 1000, 1500, 2000]
MATCHED_EVAL = [1000, 2000]
REFIT_EPOCHS = 20
INTERP_ALPHAS = [round(0.1 * k, 1) for k in range(11)]
PAIRS = [("m", "c"), ("m", "t"), ("c", "t")]


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(p).with_suffix(".tmp"); tmp.write_text(json.dumps(x, indent=1) + "\n"); tmp.replace(p)


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sd_of(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)["state_dict"]


def trunk(sd):
    return {k[len("feature_exact."):]: v for k, v in sd.items() if k.startswith("feature_exact.")}


# ============================================================================================== audit
def audit():
    import torch
    out = {}
    for k, c in CONF.items():
        b = sd_of(c["base"])
        row = {"base": str(c["base"].relative_to(REPO)), "base_sha256": sha(c["base"])}
        for o, p in c["specs"].items():
            bun = torch.load(p, map_location="cpu", weights_only=False)
            name = p.stem
            run = jload(c["runs"] / name / "run.json")
            row[o] = {"path": str(p.relative_to(REPO)), "update": bun["update"], "sha256": sha(p),
                      "initialised_from_base": Path(run["config"]["init_from"]).resolve() == c["base"].resolve(),
                      "same_keys_as_base": set(bun["state_dict"]) == set(b)}
        m0 = torch.load(c["merged0"], map_location="cpu", weights_only=False)
        row["merged_start"] = {"path": str(c["merged0"].relative_to(REPO)), "update": m0["update"],
                               "parents_are_the_specialists": [q["sha256"] for q in m0["parents"]] == [row[o]["sha256"] for o in OBJ]}
        sel = sd_of(DS_CK / c["ds"] / "selected.pth")
        row["tardiness_snapshots"] = sorted(int(x.stem[1:]) for x in (DS_CK / c["ds"]).glob("u*.pth"))
        if k in ("C", "D"):
            t = sd_of(c["specs"]["t"])
            sel_u = jload(REPO / "results/tardiness_wm_downstream_v3/summary.json")["rows"][c["ds"]]["best_update"]
            snap = sd_of(DS_CK / c["ds"] / f"u{sel_u if k == 'C' else 2000}.pth")
            row["tardiness_snapshots_on_same_trajectory"] = (
                all(torch.equal(t[x], snap[x]) for x in t) if k == "C" else
                jload(RS4_RES / "protocol_check.json")["D"]["spec_t_extension"]["exact"])
        row["downstream_start_equals_base"] = all(torch.equal(sd_of(DS_CK / f"start_{c['ds']}.pth")[x], b[x]) for x in b)
        del sel
        out[k] = row
    flat = []
    for r in out.values():
        for v in r.values():
            if isinstance(v, bool):
                flat.append(v)
            elif isinstance(v, dict):
                flat += [x for x in v.values() if isinstance(x, bool)]
    out["all_pass"] = all(flat)
    jdump(RES / "audit.json", out)
    assert out["all_pass"], json.dumps(out, indent=1)
    return out


# ============================================================================================== weights
def groups(k):
    if k.startswith("feature_exact."):
        p = k.split(".")
        return "encoder", ".".join(p[1:3]) if p[1].endswith("blocks") else p[1]
    p = k.split(".")
    return p[0], ".".join(p[:2])


def weights():
    import torch
    out = {"drift": {}, "task_vectors": {}}
    for k, c in CONF.items():
        b = sd_of(c["base"])
        tv = {}
        out["drift"][k] = {}
        for o in OBJ:
            s = sd_of(c["specs"][o])
            d = {x: (s[x].double() - b[x].double()) for x in b}
            tv[o] = d
            part, layer = {}, {}
            for x, v in d.items():
                g, lg = groups(x)
                for key, dct in ((g, part), (f"{g}:{lg}", layer)):
                    a = dct.setdefault(key, [0.0, 0.0, 0])
                    a[0] += float((v ** 2).sum()); a[1] += float((b[x].double() ** 2).sum()); a[2] += v.numel()
            f = lambda a: {"abs": a[0] ** .5, "base_norm": a[1] ** .5, "relative": a[0] ** .5 / max(a[1] ** .5, 1e-12),  # noqa: E731
                           "rms": (a[0] / a[2]) ** .5, "params": a[2]}
            out["drift"][k][o] = {"parts": {g: f(a) for g, a in part.items()}, "layers": {g: f(a) for g, a in layer.items()}}
        rows = {}
        for part in ("all", "encoder", "actor", "critic"):
            keep = [x for x in b if part == "all" or groups(x)[0] == part]
            vec = {o: torch.cat([tv[o][x].reshape(-1) for x in keep]) for o in OBJ}
            r = {"norm": {o: float(vec[o].norm()) for o in OBJ}}
            for i, j in PAIRS:
                u, v = vec[i], vec[j]
                nz = (u != 0) & (v != 0)
                r[f"{i}{j}"] = {"cosine": float(torch.nn.functional.cosine_similarity(u, v, 0)),
                                "sign_agreement": float((torch.sign(u[nz]) == torch.sign(v[nz])).double().mean()),
                                "norm_ratio": float(max(u.norm(), v.norm()) / max(min(u.norm(), v.norm()), 1e-12))}
            m = sum(vec[o] for o in OBJ) / 3
            r["mean_vector_norm_over_mean_norm"] = float(m.norm() / (sum(vec[o].norm() for o in OBJ) / 3))
            rows[part] = r
        out["task_vectors"][k] = rows
    jdump(RES / "weights.json", out)
    return out


# ============================================================================================== probes
def probe():
    import numpy as np
    import torch
    import world_model_immediate_pilot_v3 as P
    J, L, configs = P.J, P.L, P.configs
    from model.main_model import DualAttentionNetwork
    from common_utils import setup_seed
    f = RES / "probes.json"
    R = jload(f) if f.exists() else {}
    cfg, dev = L.setup(); P._CFG = cfg
    Dt, Df = P.load_all()
    n_pair = Dt["s_fea_pairs"].shape[-1]
    te = Df["sp"]["test"]
    rng = np.random.default_rng(111)
    trs = np.sort(rng.choice(Dt["sp"]["train"], size=min(len(Dt["sp"]["train"]), P.C["probe_states"]), replace=False))
    rng = np.random.default_rng(P.C["train_seed"])
    tr_tr = Dt["tix"][Dt["sp"]["train"]][Dt["cand"][Dt["sp"]["train"]] >= 0]
    fit = np.sort(rng.choice(tr_tr, size=min(len(tr_tr), P.C["avail_fit_transitions"]), replace=False))
    tst = Df["tix"][te][Df["cand"][te] >= 0]
    fp = J.same_op_pairs(Dt, Dt["sp"]["train"]); fp = fp[np.sort(rng.choice(len(fp), size=min(len(fp), P.C["avail_fit_pairs"]), replace=False))]
    tp = J.same_op_pairs(Df, te)
    pairs = None

    def enc_of(sd):
        net = DualAttentionNetwork(configs).to(dev)
        net.load_state_dict({k: v.to(dev) for k, v in trunk(sd).items()})
        return net.eval()

    def feats(enc, Dx, states):
        a, m, g = P.state_feats(enc, Dx, states, dev)
        return a, m, g

    def lin_cka(X, Y):
        X = X - X.mean(0); Y = Y - Y.mean(0)
        return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))

    def readouts(enc):
        """OLS readouts of the CURRENT state (fit on old train states, scored on fresh test states): machine free time
        (relative to t) from machine tokens, job ready time from the candidate-operation tokens."""
        out = {}
        sub = trs[:5000]
        for nm, key, tok in (("machine_free_time_now", "st_F0", 1), ("job_ready_time_now", "st_R0", 0)):
            Xa, Ya = [], []
            for Dx, st in ((Dt, sub), (Df, te)):
                F = feats(enc, Dx, st)[tok].astype(np.float64)
                y = Dx[key][st]
                ok = np.isfinite(y)
                Xa.append(F[ok]); Ya.append(y[ok])
            w = P.ols_fit(Xa[0], Ya[0]); p = P.ols_pred(w, Xa[1])
            out[nm] = {"r2": P.r2(p, Ya[1]), "mae_minutes": float(np.abs(p - Ya[1]).mean())}
        return out

    def phys_eval(m):
        nonlocal pairs
        Pp, TR = P.phys_predict(m, Df, te)
        pairs = pairs or P.sibling_pairs(Df, TR)
        pm = P.phys_metrics(Df, TR, Pp, pairs)
        return {"mean_change_mae_changed": pm["mean_change_mae_changed"],
                **{f"{g}_change_r2_changed": pm[g]["change_r2_changed"] for g in P.GROUPS},
                **{f"{g}_change_mae_changed": pm[g]["change_mae_changed"] for g in P.GROUPS},
                **{f"{g}_sibling_sign_accuracy": pm[g]["siblings"]["all"]["sign_accuracy_nonzero"] for g in P.GROUPS},
                **{f"{g}_sibling_difference_mae": pm[g]["siblings"]["all"]["difference_mae"] for g in P.GROUPS}}

    def refit(sd):
        """Predictor + physical head trained on the FROZEN encoder (kind A_phys = frozen trunk), REFIT_EPOCHS epochs,
        identical seed / batches / lr for every encoder."""
        m = P.PModel("A_phys", dev, n_pair, trunk_sd={k: v.to(dev) for k, v in trunk(sd).items()})
        opt = torch.optim.Adam(m.params(), lr=P.C["lr"])
        r = np.random.default_rng(P.C["train_seed"])
        t0 = time.time()
        for ep in range(REFIT_EPOCHS):
            perm = r.permutation(Dt["sp"]["train"])
            m.train_mode(True)
            for b0 in range(0, len(perm), P.C["batch_states"]):
                sel = np.sort(perm[b0:b0 + P.C["batch_states"]])
                loss = m.total(m.losses(Dt, sel))
                opt.zero_grad(); loss.backward(); opt.step()
        m.train_mode(False)
        return {**phys_eval(m), "seconds": time.time() - t0}

    # representation drift on fixed fresh test states (candidate-op tokens, machine tokens, globals)
    R.setdefault("representation", {})
    base_feats = {}
    for k, c in CONF.items():
        b = enc_of(sd_of(c["base"]))
        base_feats[k] = [x.reshape(-1, x.shape[-1]).astype(np.float64) for x in feats(b, Df, te)]
    for k, c in CONF.items():
        if k in R["representation"]:
            continue
        row = {}
        sf = {o: [x.reshape(-1, x.shape[-1]).astype(np.float64) for x in feats(enc_of(sd_of(c["specs"][o])), Df, te)]
              for o in OBJ}
        for o in OBJ:
            row[o] = {}
            for t, nm in enumerate(("operation_tokens", "machine_tokens", "globals")):
                X, Y = base_feats[k][t], sf[o][t]
                row[o][nm] = {"cka_to_base": lin_cka(X, Y),
                              "relative_change": float(np.linalg.norm(Y - X) / max(np.linalg.norm(X), 1e-12))}
        for i, j in PAIRS:
            row[f"{i}{j}"] = {nm: lin_cka(sf[i][t], sf[j][t]) for t, nm in enumerate(("operation_tokens", "machine_tokens", "globals"))}
        R["representation"][k] = row
        jdump(f, R); print(f"representation {k}", flush=True)

    # retention probes: base + each specialist encoder
    R.setdefault("retention", {})
    for k, c in CONF.items():
        for who, p in [("base", c["base"])] + [(o, c["specs"][o]) for o in OBJ]:
            key = f"{k}:{who}"
            if key in R["retention"]:
                continue
            t0 = time.time()
            sd = sd_of(p); enc = enc_of(sd)
            row = {"relational": P.relational_probe(enc, Dt, trs, Df, te, dev),
                   "availability_next_state": P.availability(lambda Dx, rows, e=enc: P.actual_mch(e, Dx, rows, dev),
                                                             Dt, Df, fit, tst, fp, tp),
                   "current_state_readouts": readouts(enc)}
            if c.get("native"):                                    # the ORIGINAL pilot predictor + head on this encoder
                m = P.load(c["native"], "epoch60", dev, n_pair)
                m.trunk.load_state_dict({kk: v.to(dev) for kk, v in trunk(sd).items()})
                m.train_mode(False)
                row["physical_original_head"] = phys_eval(m)
                if c["native"] == "E":
                    row["latent_original_predictor"] = J.latent_eval(m, Df, te, full=True)
            setup_seed(111)
            row["physical_refit_head"] = refit(sd)
            row["seconds"] = time.time() - t0
            R["retention"][key] = row
            jdump(f, R)
            print(f"retention {key}: refit MAE {row['physical_refit_head']['mean_change_mae_changed']:.2f} "
                  f"({row['seconds']:.0f} s)", flush=True)
    return R


# ============================================================================================== interpolation
def _decoder():
    os.environ["DECODING_DEVICE"] = "cuda"
    argv = sys.argv; sys.argv = sys.argv[:1]
    import decoding_v3 as D
    import benchmark_v3 as B
    sys.argv = argv
    D.configs.dispatch_rule = "semi_active"
    return D, B


def interp():
    import numpy as np
    D, B = _decoder()
    f = RES / "interpolation.json"
    R = jload(f) if f.exists() else {}
    data = B.load_split("validation", "10x5")
    I = len(data)
    prefs = []
    for i, j in PAIRS:
        for a in INTERP_ALPHAS:
            w = np.zeros(3); w[OBJ.index(i)] = 1 - a; w[OBJ.index(j)] = a
            prefs.append(w)
    prefs = np.asarray(prefs)
    for k, c in CONF.items():
        if k in R:
            continue
        net = D.load_net(c["merged0"])
        idx = [n for n in range(I) for _ in range(len(prefs))]
        P = np.tile(prefs, (I, 1))
        cost = D.Cost(len(idx))
        obj, _, _ = D.greedy(net, data, idx, P, cost)
        obj = obj.reshape(I, len(prefs), 3)
        mk = np.asarray(data.mk_ref)[:I, None]; cb = np.asarray(data.cb_ref)[:I, None]
        norm = np.stack([obj[..., 0] / mk, obj[..., 1] / cb, obj[..., 2] / data.t_ref], -1)     # [I, P, 3]
        row = {}
        for q, (i, j) in enumerate(PAIRS):
            sl = slice(q * len(INTERP_ALPHAS), (q + 1) * len(INTERP_ALPHAS))
            W = prefs[sl]; N = norm[:, sl]                                                  # [I, 11, 3]
            merged = (N * W[None]).sum(-1)                                                  # score at own preference
            ei, ej = N[:, 0], N[:, -1]                                                      # pure specialists
            si = (ei[:, None] * W[None]).sum(-1); sj = (ej[:, None] * W[None]).sum(-1)
            row[f"{i}{j}"] = {"alpha": INTERP_ALPHAS,
                              "merged_score": merged.mean(0).tolist(),
                              "specialist_i_score": si.mean(0).tolist(), "specialist_j_score": sj.mean(0).tolist(),
                              "best_endpoint_score": np.minimum(si, sj).mean(0).tolist(),
                              "gap_vs_best_endpoint": (merged - np.minimum(si, sj)).mean(0).tolist(),
                              "normalised_objectives": N.mean(0).tolist()}
            g = np.asarray(row[f"{i}{j}"]["gap_vs_best_endpoint"])[1:-1]
            row[f"{i}{j}"]["mean_interior_gap"] = float(g.mean()); row[f"{i}{j}"]["max_interior_gap"] = float(g.max())
        R[k] = row
        jdump(f, R)
        print(f"interpolation {k}: " + ", ".join(f"{p} {row[p]['mean_interior_gap']:+.4f}" for p in row), flush=True)
    return R


# ============================================================================================== matched merges
def matched():
    import torch
    import freeze_specialist_reuse_v3 as F
    import specialist_reuse_v3_runner as Rn
    import argparse
    _load = F.load
    F.load = lambda check=True: {**_load(check), "name": f"{NAME}/merges"}
    r = Rn.Runner(argparse.Namespace(smoke=False, jobs=1))
    F.load = _load
    r.tag = "wmdx"
    r.commit = lambda *a, **k: None
    sel = jload(A_RES / "state.json")["selected"]["rs"]
    f = RES / "matched.json"
    R = jload(f) if f.exists() else {"merges": {}, "eval_seconds": {}}
    (CK / "merges").mkdir(parents=True, exist_ok=True)
    for k, c in CONF.items():
        for u in MATCHED:
            key = f"{k}@t{u}"
            if key in R["merges"]:
                continue
            specs = [str(c["specs"]["m"]), str(c["specs"]["c"]), str(DS_CK / c["ds"] / f"u{u}.pth")]
            r.specs = lambda s=specs: s
            suffix = f"{k.lower()}t{u}"
            name = r.model(suffix)
            cmd = r.method_args("rs", suffix, sel["lr"], sel["cl_beta"], max_updates=0) + ["--budget_checkpoints", "0"]
            out = r.res / "logs" / f"{name}.stdout.log"; out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            if r.run([Rn.PY, *cmd], 0, 3600, out) != 0:
                raise SystemExit(f"merge {key} failed; see {out}")
            v = jload(Rn.TL / f"valiquality_{name}.json")
            dst = CK / "merges" / f"{name}@u0.pth"
            shutil.copy2(Rn.TN / f"{name}@u0.pth", dst)
            rdir = r.res / "runs" / name; rdir.mkdir(parents=True, exist_ok=True)
            for s_, d_ in ((f"valiquality_{name}.json", "valiquality.json"), (f"{name}.run.json", "run.json")):
                shutil.copy2(Rn.TL / s_, rdir / d_)
            m0 = torch.load(dst, map_location="cpu", weights_only=False)
            R["merges"][key] = {"config": k, "tardiness_update": u, "validation_hv": v[0]["hv"],
                                "makespan_update": torch.load(c["specs"]["m"], map_location="cpu", weights_only=False)["update"],
                                "carbon_update": torch.load(c["specs"]["c"], map_location="cpu", weights_only=False)["update"],
                                "parents": [q["sha256"] for q in m0["parents"]], "checkpoint": str(dst.relative_to(REPO)),
                                "seconds": time.time() - t0}
            jdump(f, R)
            print(f"merge {key}: validation HV {v[0]['hv']:.4f}", flush=True)
    sums = {str(p.relative_to(CK)): sha(p) for p in sorted(CK.rglob("*.pth"))}
    jdump(CK / "SHA256SUMS.json", sums)
    # best-of-128 evaluation at the evaluated budgets
    for k in CONF:
        for u in MATCHED_EVAL:
            key = f"{k}@t{u}"
            out = RES / "eval" / f"{k}_t{u}.npz"
            if out.exists():
                continue
            t0 = time.time()
            with open(RES / "eval.log", "a") as fh:
                rc = subprocess.run(["taskset", "-c", "0-11", Rn.PY, str(REPO / "tools" / "specialist_reuse_v3_eval.py"),
                                     "--model", str(REPO / R["merges"][key]["checkpoint"]), "--out", str(out)],
                                    cwd=REPO, env=Rn.ENV, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise SystemExit(f"evaluation {key} failed")
            R["eval_seconds"][key] = time.time() - t0
            jdump(f, R)
            print(f"evaluated {key}", flush=True)
    return R


def matched_metrics():
    import numpy as np
    import freeze_specialist_reuse_v3 as F
    import analyze_specialist_reuse_v3 as AN
    cfg = F.load()
    norm = jload(REPO / cfg["evaluation"]["norm"])["instances"]
    b2 = REPO / cfg["evaluation"]["igd_front"]
    out = {}
    for f in sorted((RES / "eval").glob("*.npz")):
        e = np.load(f); keys = list(e["keys"])
        hv, igd, n = AN.metrics(e["sample128_obj"], keys, norm, b2)
        out[f.stem] = {"best_of_128_hv": hv, "best_of_128_igd_plus": igd, "model_sha256": str(e["model_sha256"])}
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    RES.mkdir(parents=True, exist_ok=True)
    if cmd in ("audit", "all"):
        audit(); print("audit passed", flush=True)
    if cmd in ("weights", "all"):
        weights(); print("weights done", flush=True)
    if cmd in ("probe", "all"):
        subprocess.run([sys.executable, "-W", "ignore", __file__, "_probe"], check=True)
    if cmd == "_probe":
        probe()
    if cmd in ("interp", "all"):
        subprocess.run([sys.executable, "-W", "ignore", __file__, "_interp"], check=True)
    if cmd == "_interp":
        interp()
    if cmd in ("matched", "all"):
        matched()


if __name__ == "__main__":
    main()
