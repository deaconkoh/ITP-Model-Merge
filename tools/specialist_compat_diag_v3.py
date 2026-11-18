"""Two evaluation-only diagnostics on the saved specialists of the matched 10x5, seed-111, waiting-enabled Rewarded Soups
experiment: random shared base (results/specialist_reuse_v3) vs outcome-prediction shared base
(results/specialist_reuse_wm_v3). Validation split only (the instances every specialist was validated on); the
reserved final-test sets are never loaded. No training; checkpoints are only read.

1. Carbon headroom. Carbon is additive over the chosen (operation, machine) pairs (benchmark_v3.schedule_objectives),
   so sum_o min_m carbon[o, m] over eligible m is a lower bound for every schedule. A schedule that uses exactly those
   assignments is built in the simulator under semi_active and checked with the independent scorer; if its carbon
   equals the bound, the bound is the optimum. Both saved carbon specialists are rolled out greedily on the same
   instances (the training validation decoder) and compared with it.
2. Interpolation. For each specialist pair and each base, theta(a) = (1 - a) theta_A + a theta_B (every tensor of the
   saved selected checkpoints, i.e. before merge fine-tuning), a = 0, 0.1, ..., 1; greedy rollouts; cost = 0.5 X/Xref +
   0.5 Y/Yref with the frozen v3 score references (MKref_i, Cref_i per instance, Tref of the size), mean over instances.
   Barrier = max_a [cost(a) - ((1 - a) cost(0) + a cost(1))]; blends worse than both endpoints are flagged.
   Supplementary: the end-of-run weights (the last update of each run), the nearest-budget pair that exists.

    python tools/specialist_compat_diag_v3.py      # -> results/specialist_compat_diag_v3/
"""
from __future__ import annotations
import copy, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel")); sys.path.insert(0, str(REPO / "tools"))
sys.argv = [sys.argv[0], "--device", "cuda", "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct",
            "--preference_dim", "0", "--n_j", "10", "--n_m", "5", "--data_suffix", "mix", "--low", "1", "--high", "99",
            "--op_per_job", "0", "--dispatch_rule", "semi_active"]
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from params import configs                                          # noqa: E402
import benchmark_v3 as B                                            # noqa: E402
from common_utils import load_checkpoint_state_dict                # noqa: E402
from feature_schemas import SCHEMA_V3                               # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums              # noqa: E402
from model.main_model import DANIEL                                 # noqa: E402

DEV = configs.device
OUT = REPO / "results" / "specialist_compat_diag_v3"
BASES = {"random": ("results/specialist_reuse_v3", "checkpoints/specialist_reuse_v3/specialists", "sr"),
         "world_model": ("results/specialist_reuse_wm_v3", "checkpoints/specialist_reuse_wm_v3/specialists", "srwm")}
LABEL = {"random": "random shared base", "world_model": "outcome-prediction shared base"}
OBJ = {"m": "makespan", "c": "carbon", "t": "tardiness"}
PAIRS = [("m", "c"), ("m", "t"), ("c", "t")]
ALPHAS = np.round(np.arange(11) / 10, 1)
KEYS = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs")


def spec_path(base, o):
    _, ck, tag = BASES[base]
    return REPO / ck / f"10x5+mix+bv3_{tag}_spec_{o}_s111.pth"


def resume_path(base, o):
    return REPO / "daniel/trained_network/SD2" / f"10x5+mix+bv3_{BASES[base][2]}_spec_{o}_s111.resume.pth"


def net_from(sd):
    c = copy.copy(configs); c.preference_dim = 0; c.feature_schema = SCHEMA_V3
    net = DANIEL(c).to(DEV)
    net.load_state_dict(sd)
    return net.eval()


def load_sd(path):
    return {k: v.detach().cpu() for k, v in load_checkpoint_state_dict(path, map_location="cpu",
                                                                         expected_schema=SCHEMA_V3).items()}


def env_for(val, goal="mct"):
    assert configs.dispatch_rule == "semi_active"
    env = FJSPEnvForSameOpNums(10, 5)
    st = env.set_initial_data(**val.env_kwargs(range(len(val)), goal))
    assert env.dispatch_rule == "semi_active"
    return env, st


def finish(env, machine, val):
    """Objectives from the environment, cross-checked with the independent scorer on every instance."""
    n = env.number_of_ops
    pt = env.true_op_pt[env.env_idxs[:, None], np.arange(n)[None, :], machine]
    start = env.true_op_ct - pt
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
    obj = np.stack([env.current_makespan, env.total_carbon, tard], 1).astype(float)
    for i in range(len(val)):
        s = B.schedule_objectives(val.job_length[i], val.op_pt[i], val.carbon[i], val.due[i], machine[i], start[i])
        assert np.allclose([s["makespan"], s["carbon"], s["tardiness"]], obj[i], rtol=1e-6, atol=1e-6), (i, s, obj[i])
    return obj


@torch.no_grad()
def rollout(net, val, policy=None):
    env, st = env_for(val)
    E, M = env.number_of_envs, env.number_of_machines
    machine = np.full((E, env.number_of_ops), -1)
    while True:
        if policy is None:
            pi, _ = net(**{k: getattr(st, f"{k}_tensor") for k in KEYS})
            a = pi.argmax(1).cpu().numpy()
        else:
            a = policy(env)
        op = env.candidate[env.env_idxs, a // M]
        machine[env.env_idxs, op] = a % M
        st, _, done = env.step(a)
        if np.all(done):
            break
    assert (machine >= 0).all()
    return finish(env, machine, val)


def min_carbon_policy(val):
    """Pick any legal (candidate operation, its cheapest-carbon eligible machine) pair."""
    cmat = [np.where(np.asarray(val.op_pt[i]) > 0, np.asarray(val.carbon[i]), np.inf) for i in range(len(val))]
    best = np.stack([c.argmin(1) for c in cmat])                             # [E, N]

    def pol(env):
        M = env.number_of_machines
        mask = env.dynamic_pair_mask                                         # True = not allowed
        a = np.full(env.number_of_envs, -1)
        for e in range(env.number_of_envs):
            for j in range(env.number_of_jobs):
                op = env.candidate[e, j]
                if env.mask[e, j]:
                    continue
                m = best[e, op]
                if not mask[e, j, m]:
                    a[e] = j * M + m
                    break
        assert (a >= 0).all(), "no legal cheapest-carbon pair"
        return a
    return pol


# ------------------------------------------------------------------------------------------------ 1. carbon

def carbon_part(val):
    lb = np.array([float(np.where(np.asarray(val.op_pt[i]) > 0, np.asarray(val.carbon[i]), np.inf).min(1).sum())
                   for i in range(len(val))])
    obj_lb = rollout(None, val, policy=min_carbon_policy(val))
    attained = np.allclose(obj_lb[:, 1], lb, rtol=1e-9, atol=1e-9)
    cref = np.asarray(val.cb_ref)
    rows, agg = [], {}
    specs = {}
    for base in BASES:
        p = spec_path(base, "c")
        b = torch.load(p, map_location="cpu", weights_only=False)
        obj = rollout(net_from(load_sd(p)), val)
        saved = b["validation_metrics"]
        specs[base] = {"obj": obj, "update": b["update"],
                       "reproduces_saved_validation": bool(np.isclose(obj[:, 1].mean(), saved["carbon"], rtol=1e-6)
                                                           and np.isclose(obj[:, 0].mean(), saved["makespan"], rtol=1e-6)),
                       "saved_validation_carbon": saved["carbon"]}
    for i, k in enumerate(val.keys):
        r = {"instance": Path(k).name, "optimum": lb[i], "reference_heuristic": cref[i]}
        for base in BASES:
            c = specs[base]["obj"][i, 1]
            r[f"{base}_carbon"] = c; r[f"{base}_gap_pct"] = 100 * (c / lb[i] - 1)
        rows.append(r)
    for base in BASES:
        c = specs[base]["obj"][:, 1]
        agg[base] = {"selected_update": specs[base]["update"], "mean_carbon": float(c.mean()),
                     "mean_gap_pct": float(np.mean(100 * (c / lb - 1))),
                     "median_gap_pct": float(np.median(100 * (c / lb - 1))),
                     "max_gap_pct": float(np.max(100 * (c / lb - 1))),
                     "share_of_reference_to_optimum_gap_closed": float(np.mean((cref - c) / (cref - lb))),
                     "instances_at_optimum": int(np.sum(np.isclose(c, lb, rtol=1e-9))),
                     "mean_makespan": float(specs[base]["obj"][:, 0].mean()),
                     "reproduces_saved_validation": specs[base]["reproduces_saved_validation"],
                     "saved_validation_carbon": specs[base]["saved_validation_carbon"]}
    agg["optimum_mean"] = float(lb.mean()); agg["reference_heuristic_mean"] = float(cref.mean())
    agg["optimum_mean_score_c_over_cref"] = float(np.mean(lb / cref))
    agg["min_carbon_schedule"] = {"attains_bound_on_every_instance": bool(attained),
                                  "makespan_mean": float(obj_lb[:, 0].mean()),
                                  "checked_by_independent_scorer": True}
    # learning curves at matched updates (saved validation means; same instances)
    curves = {}
    for base, (res, _, tag) in BASES.items():
        v = json.loads((REPO / res / "runs" / f"10x5+mix+bv3_{tag}_spec_c_s111" / "valiquality.json").read_text())
        curves[base] = {e["update"]: e["carbon"] for e in v}
    common = sorted(set(curves["random"]) & set(curves["world_model"]))
    matched = [{"update": u, **{f"{b}_carbon": curves[b][u] for b in BASES},
                **{f"{b}_gap_of_mean_pct": 100 * (curves[b][u] / lb.mean() - 1) for b in BASES}} for u in common]
    probe = json.loads((REPO / "results/world_model_m1_v3/action_pilot.json").read_text())
    pc = {}
    for lab, key in (("original M1 encoder", ("original_m1", "mean|carbon")),
                     ("outcome-prediction pilot encoder (pairs, abs+diff)", ("pilot", "pairs|abs+diff", "eval", "mean|carbon"))):
        d = probe
        for kk in key:
            d = d[kk]
        pc[lab] = d
    pc["simple_rule_same_job_lower_energy_better_share"] = probe["simple_rule_same_job"]
    return {"rows": rows, "aggregate": agg, "matched_curve": matched, "probe": pc}, lb


# ------------------------------------------------------------------------------------------------ 2. interpolation

def weighted(obj, val, a, b):
    ref = {"m": np.asarray(val.mk_ref), "c": np.asarray(val.cb_ref), "t": np.full(len(val), val.t_ref)}
    col = {"m": 0, "c": 1, "t": 2}
    return 0.5 * obj[:, col[a]] / ref[a] + 0.5 * obj[:, col[b]] / ref[b]


def interp_part(val, which="selected"):
    out = {}
    for base in BASES:
        sds = {o: load_sd(spec_path(base, o)) if which == "selected" else
               {k: v.detach().cpu() for k, v in torch.load(resume_path(base, o), map_location="cpu",
                                                           weights_only=False)["policy"].items()} for o in "mct"}
        upd = {}
        for o in "mct":
            upd[o] = (torch.load(spec_path(base, o), map_location="cpu", weights_only=False)["update"] if which == "selected"
                      else torch.load(resume_path(base, o), map_location="cpu", weights_only=False)["next_update"])
        for a, b in PAIRS:
            A, Bsd = sds[a], sds[b]
            pts = []
            for al in ALPHAS:
                sd = {k: ((1 - al) * A[k].double() + al * Bsd[k].double()).to(A[k].dtype) if A[k].is_floating_point()
                      else A[k] for k in A}
                obj = rollout(net_from(sd), val)
                cost = weighted(obj, val, a, b)
                pts.append({"alpha": float(al), OBJ[a]: float(obj[:, "mct".index(a)].mean()),
                            OBJ[b]: float(obj[:, "mct".index(b)].mean()), "cost": float(cost.mean()),
                            "cost_per_instance": cost.tolist()})
            c = np.array([p["cost"] for p in pts]); line = (1 - ALPHAS) * c[0] + ALPHAS * c[-1]
            rise = c - line
            worse = [float(al) for al, x in zip(ALPHAS, c) if x > max(c[0], c[-1]) + 1e-12]
            k = int(np.argmax(rise))
            out[f"{base}|{a}{b}"] = {"updates": {a: upd[a], b: upd[b]}, "points": pts,
                                     "largest_rise_above_line": float(rise[k]), "at_alpha": float(ALPHAS[k]),
                                     "largest_rise_pct_of_line": float(100 * rise[k] / line[k]),
                                     "blends_worse_than_both_endpoints": worse,
                                     "endpoint_costs": [float(c[0]), float(c[-1])]}
            print(f"{which} {base} {a}{b}: rise {rise[k]:.4f} @ {ALPHAS[k]} worse-than-both {worse}", flush=True)
    return out


def plots(I, I_end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"random": "#7f7f7f", "world_model": "#1f77b4"}
    for a, b in PAIRS:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
        for base in BASES:
            r = I[f"{base}|{a}{b}"]; c = [p["cost"] for p in r["points"]]
            ax.plot(ALPHAS, c, color=col[base], marker="o", lw=1.8,
                    label=f"{LABEL[base]} (largest rise {r['largest_rise_above_line']:+.3f})")
            ax.plot([0, 1], [c[0], c[-1]], color=col[base], ls="--", lw=.9)
            for al in r["blends_worse_than_both_endpoints"]:
                ax.plot(al, c[int(round(al * 10))], "x", color="#d62728", ms=9, mew=2)
            xs = [p[OBJ[a]] for p in r["points"]]; ys = [p[OBJ[b]] for p in r["points"]]
            ax2.plot(xs, ys, color=col[base], marker="o", lw=1.5, label=LABEL[base])
            ax2.annotate(f"100% {OBJ[a]}", (xs[0], ys[0]), fontsize=8, color=col[base], xytext=(4, 4), textcoords="offset points")
            ax2.annotate(f"100% {OBJ[b]}", (xs[-1], ys[-1]), fontsize=8, color=col[base], xytext=(4, 4), textcoords="offset points")
        ax.plot([], [], "x", color="#d62728", ms=9, mew=2, label="worse than both endpoints")
        ax.set_xticks(ALPHAS); ax.set_xticklabels([f"{int(100 - 100 * x)}/{int(100 * x)}" for x in ALPHAS], fontsize=7)
        ax.set_xlabel(f"blend ({OBJ[a]} specialist % / {OBJ[b]} specialist %)")
        ax.set_ylabel(f"cost = 0.5 {OBJ[a]}/ref + 0.5 {OBJ[b]}/ref (lower is better)")
        ax.set_title("Weighted cost along the blend (dashed: straight line between endpoints)", fontsize=10)
        ax2.set_xlabel(f"mean {OBJ[a]}"); ax2.set_ylabel(f"mean {OBJ[b]}")
        ax2.set_title("Raw objectives along the blend (both lower is better)", fontsize=10)
        for x in (ax, ax2):
            x.grid(alpha=.3); x.legend(frameon=False, fontsize=8)
            for s in ("top", "right"):
                x.spines[s].set_visible(False)
        fig.suptitle(f"{OBJ[a].capitalize()}-{OBJ[b]} specialist interpolation, before merge fine-tuning "
                     f"(10x5, 18 validation instances, greedy, waiting allowed, seed 111)", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / f"interp_{a}{b}.png", dpi=140, facecolor="white"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    val = B.load_split("validation", "10x5")
    print(f"validation instances: {len(val)} ({val.directory})", flush=True)
    C, lb = carbon_part(val)
    (OUT / "carbon.json").write_text(json.dumps(C, indent=1) + "\n")
    print(json.dumps(C["aggregate"], indent=1), flush=True)
    I = interp_part(val, "selected")
    I_end = interp_part(val, "end_of_run")
    (OUT / "interpolation.json").write_text(json.dumps({"selected_checkpoints": I, "end_of_run_weights": I_end,
                                                        "cost": "0.5 X/Xref + 0.5 Y/Yref, frozen v3 score references",
                                                        "instances": len(val)}, indent=1) + "\n")
    plots(I, I_end)
    (OUT / "settings.json").write_text(json.dumps({
        "split": "validation 10x5", "instances": len(val), "decoder": "greedy", "dispatch_rule": configs.dispatch_rule,
        "checkpoints": {b: {o: str(spec_path(b, o).relative_to(REPO)) for o in "mct"} for b in BASES},
        "end_of_run": {b: {o: str(resume_path(b, o).relative_to(REPO)) for o in "mct"} for b in BASES},
        "final_test_loaded": False, "seconds": time.time() - t0}, indent=1) + "\n")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
