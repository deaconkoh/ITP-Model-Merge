"""Where does the M1 world model lose the ability to distinguish scheduling actions? Inference and analysis only, on
saved artifacts: the M1 checkpoint (trunk + original prediction heads) and the saved held-out VALIDATION data
(data/world_model_m1_v3/val; 18 validation instances, 4,410 states, up to 4 legal candidate actions per state, exact
outcomes under every continuation rule). No rollouts, no training. The reserved final-test sets are never loaded.

1. Full world model: for every state and each of the 3 continuation rules seen in pretraining, every pair of legal
   candidate actions; predicted outcomes -> makespan (max job completion), carbon (sum of machine energy), total
   tardiness (sum of max(0, completion - due)), the same definitions as the saved labels. Pairwise ranking accuracy
   excludes exact true ties; a predicted tie (|difference| < 1e-9) counts as half correct. Accuracy by size of the
   true difference (quartiles of the non-tied pairs).
2. Does the model use the action: pairing checks (every candidate legal under the saved mask; no true job completion
   below the environment's saved lower bound for that action); candidate actions shuffled within each state and
   rule with labels fixed; absolute error on the objectives vs error on the difference between two actions (and the
   error of predicting no difference at all).

    python tools/world_model_m1_action_diagnostic.py     # -> results/world_model_m1_v3/action_diagnostic.json
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.argv = sys.argv[:1] + ["probe"]                                 # world_model_m1_v3 parses nothing at import
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import world_model_m1_v3 as W                                       # noqa: E402

TOL = {"makespan": 5.0, "carbon": 0.5, "tardiness": 5.0}            # near-tie thresholds (min, kg, min)
OBJ = ("makespan", "carbon", "tardiness")


def objectives(jc, en, due):
    """jc [..., J] job completion, en [..., M] machine energy, due [S, J] broadcast -> dict of [...]."""
    return {"makespan": jc.max(-1), "carbon": en.sum(-1), "tardiness": np.maximum(0.0, jc - due).sum(-1)}


def predict(model, V, cfg, dev, perm=None):
    """Predicted job completion [S, A, R, J] and machine energy [S, A, R, M] (raw units). perm [S, A]: the candidate
    actions fed to the model, permuted within the state (labels untouched)."""
    pj, pm = [], []
    cand = V["rec_cand"] if perm is None else np.take_along_axis(V["rec_cand"], perm, 1)
    D = dict(V); D["rec_cand"] = cand
    with torch.no_grad():
        for b0 in range(0, len(V["instance"]), 256):
            sel = np.arange(b0, min(b0 + 256, len(V["instance"])))
            s, job, mch, _ = W.batch_tensors(D, sel, dev)
            a, b = model(s, job, mch)
            pj.append(a.cpu().numpy()); pm.append(b.cpu().numpy())
    return np.concatenate(pj) * cfg["t_norm"], np.concatenate(pm)[..., 1] * cfg["e_norm"]


def pairs(valid):
    A = valid.shape[1]
    for i, j in itertools.combinations(range(A), 2):
        m = valid[:, i] & valid[:, j]
        yield i, j, m


def rank_stats(P, Y, valid, tol):
    """P, Y [S, A]: predicted and true objective. Pairwise accuracy (exact true ties excluded; predicted ties half)."""
    dt, dp = [], []
    for i, j, m in pairs(valid):
        dt.append(Y[m, i] - Y[m, j]); dp.append(P[m, i] - P[m, j])
    dt, dp = np.concatenate(dt), np.concatenate(dp)
    tie = np.abs(dt) < 1e-9
    near = np.abs(dt) <= tol
    use = ~tie
    ok = np.where(np.abs(dp[use]) < 1e-9, 0.5, (np.sign(dp[use]) == np.sign(dt[use])).astype(float))
    q = np.quantile(np.abs(dt[use]), [0.25, 0.5, 0.75])
    bins = np.digitize(np.abs(dt[use]), q)
    by_bin = [float(ok[bins == b].mean()) for b in range(4)]
    return {"pairs": int(len(dt)), "exact_ties": int(tie.sum()), "near_ties": int(near.sum()),
            "accuracy": float(ok.mean()), "accuracy_excl_near_ties": float(ok[~near[use]].mean()),
            "accuracy_by_true_difference_quartile": by_bin, "quartile_edges": q.tolist(),
            "predicted_ties": int((np.abs(dp[use]) < 1e-9).sum()),
            "mae_difference": float(np.abs(dp - dt).mean()), "mae_difference_if_predict_zero": float(np.abs(dt).mean())}


def main():
    cfg = W.settings(False); W.set_features(cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; W.configs.device = dev
    V = W.load_split(False, "val")
    model = W.load_world_model(False, dev)
    valid = V["rec_cand"] >= 0
    S, A = valid.shape
    M = W.B.SIZES[W.SIZE][1]
    out = {"states": int(S), "valid_actions": int(valid.sum()), "rules": W.CONT, "checks": {}, "results": {}}
    # ---- pairing checks
    c = V["rec_cand"]; job, mch = np.where(c >= 0, c // M, 0), np.where(c >= 0, c % M, 0)
    legal = V["dynamic_pair_mask"][np.arange(S)[:, None], job, mch] == 0
    out["checks"]["candidates_legal_under_saved_mask"] = f"{int((legal & valid).sum())}/{int(valid.sum())}"
    lb = V["sj"][:, :, None, :]                                        # [S, A, 1, J] environment lower bound
    tj = V["tj"][:, :, :3]
    viol = (tj < lb - 1e-6) & valid[:, :, None, None]
    out["checks"]["true_completion_below_saved_lower_bound"] = int(viol.sum())
    # chosen-action specificity: the lower bound of the moved job differs across actions within a state
    out["checks"]["states_where_saved_lower_bounds_differ_across_actions"] = int(
        (np.nanmax(np.where(valid[:, :, None], V["sj"], np.nan), 1) - np.nanmin(np.where(valid[:, :, None], V["sj"], np.nan), 1)
         > 1e-9).any(1).sum())
    # ---- predictions: original and shuffled (within state, per rule)
    pj, pe = predict(model, V, cfg, dev)
    rng = np.random.default_rng(2026)
    shuf_j = np.empty_like(pj); shuf_e = np.empty_like(pe)
    for r in range(3):
        perm = np.tile(np.arange(A), (S, 1))
        for s in range(S):
            k = np.flatnonzero(valid[s])
            perm[s, k] = rng.permutation(k)
        a, b = predict(model, V, cfg, dev, perm)
        shuf_j[:, :, r], shuf_e[:, :, r] = a[:, :, r], b[:, :, r]
    due = V["due"][:, None, :]
    for r, rule in enumerate(W.CONT):
        T = objectives(V["tj"][:, :, r], V["tm"][:, :, r, :, 1], due)
        P = objectives(pj[:, :, r], pe[:, :, r], due)
        Q = objectives(shuf_j[:, :, r], shuf_e[:, :, r], due)
        res = {}
        for k in OBJ:
            st = rank_stats(P[k], T[k], valid, TOL[k])
            st["accuracy_shuffled"] = rank_stats(Q[k], T[k], valid, TOL[k])["accuracy"]
            st["mae_objective"] = float(np.abs(P[k] - T[k])[valid].mean())
            st["true_sd_between_states"] = float(np.nanstd(np.nanmean(np.where(valid, T[k], np.nan), 1)))
            res[k] = st
        out["results"][rule] = res
        print(rule, {k: round(v["accuracy"], 3) for k, v in res.items()},
              "shuffled", {k: round(v["accuracy_shuffled"], 3) for k, v in res.items()}, flush=True)
    out["tie_rule"] = ("exact true ties (|true difference| < 1e-9) are excluded from accuracy; a predicted tie counts as "
                       "0.5; near ties: |true difference| <= 5 min (makespan, tardiness) or 0.5 kg (carbon)")
    f = REPO / "results" / "world_model_m1_v3" / "action_diagnostic.json"
    f.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["checks"]), f"\nwritten: {f.relative_to(REPO)}")


if __name__ == "__main__":
    main()
