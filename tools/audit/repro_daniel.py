"""Code audit, Part 1: reproduce the published DANIEL SD2 10x5 baseline (results/code_audit.md). EVALUATION ONLY.

Two modes (one per process: DANIEL's config is module-global):
  legacy    the ORIGINAL DANIEL checkpoint (10x5+mix.pth from github.com/wrqccc/FJSP-DRL, passed by --ckpt) run in
            OUR environment/model with the legacy feature schema (10 operation, 8 pair features, makespan only),
            greedy, one instance at a time, on the paper-test instances that are not reserved (our train + validation
            splits of the paper's SD2 10x5 test set, 81 files) and on our HELD-OUT 49 (the paper's validation set).
            Writes per-instance makespans for comparison with the original code's own run.
  ours      OUR v1 makespan specialists m_s<seed> (carbon+priority instances, canonical schema), greedy and
            best-of-100 sampling (policy softmax, no temperature), on the same two instance sets.
Never loads data_final_test or the manifest's final_test split.

    python tools/audit/repro_daniel.py legacy --ckpt <clone>/trained_network/SD2/10x5+mix.pth --out <json>
    python tools/audit/repro_daniel.py ours --out <json>
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DAN = REPO / "daniel"
MODE = sys.argv[1]
ARGS = dict(zip(sys.argv[2::2], sys.argv[3::2]))
SPLIT = json.loads((REPO / "results/blend_zero_training/split.json").read_text())
HELD_NAMES = SPLIT["held_out"]
PAPER_DIRS = ["data_train", "data_validation"]          # 63 + 18 of the paper's 100 SD2 10x5 test instances


def files(split, suffix):
    d = DAN / f"data/{split}/SD2/10x5+{suffix}"
    return sorted(d.glob("*.fjs"))


if MODE == "legacy":
    sys.argv = [sys.argv[0], "--enable_priority", "False", "--enable_carbon", "False", "--carbon_feature", "False",
                "--fea_j_input_dim", "10", "--fea_pair_input_dim", "8", "--feature_schema", "legacy_f10_p8_v1",
                "--goal", "m", "--device", "cuda", "--test_data", "10x5+mix"]
    sys.path.insert(0, str(DAN))
    import torch
    from params import configs
    if ARGS.get("--emulate-original-mch-feature") == "1":
        # DIAGNOSTIC ONLY (this process): reproduce the published code's machine feature 3, which is computed as
        # np.max(op_pt, axis=1) although named mch_min_pt (FJSP-DRL fjsp_env_same_op_nums.py), and its
        # normalisation op_pt / (max - min_incl_zeros + 1e-8). Our repository code is not changed.
        import numpy as _np, numpy.ma as _ma
        import fjsp_env_same_op_nums as _E
        _E.machine_min_processing_time = lambda op_pt, rel: _np.max(op_pt, axis=1).data
        def _norm(raw):
            raw = _np.asarray(raw, dtype=_np.float64); rel = raw > 0
            lo, hi = float(_np.min(raw)), float(_np.max(raw))
            return (raw - lo) / (hi - lo + 1e-8), rel, ~rel, float(_np.min(raw[rel])), hi
        _E.normalize_processing_times = _norm
    from model.PPO import PPO_initialize
    from common_utils import load_checkpoint_state_dict, greedy_select_action
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from data_utils import text_to_matrix
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(load_checkpoint_state_dict(ARGS["--ckpt"], map_location="cuda",
                                                          expected_schema="legacy_f10_p8_v1"))
    ppo.policy.eval()
    out = {}
    sets = {"paper_test_81": [f for s in PAPER_DIRS for f in files(s, "mix")],
            "held_out_49": [DAN / "data/data_train_vali/SD2/10x5+mix" / n for n in HELD_NAMES]}
    for name, fl in sets.items():
        res = {}
        for f in fl:
            jl, pt = text_to_matrix(f.read_text().splitlines())
            env = FJSPEnvForSameOpNums(n_j=jl.shape[0], n_m=pt.shape[1])
            state = env.set_initial_data([jl], [pt])
            while True:
                with torch.no_grad():
                    pi, _ = ppo.policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                                       candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                                       mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                                       dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
                state, _, done = env.step(actions=greedy_select_action(pi).cpu().numpy())
                if done:
                    break
            res[f.name] = float(env.current_makespan[0])
        out[name] = res
        print(f"legacy {name}: n={len(res)} mean makespan {np.mean(list(res.values())):.3f}", flush=True)
    Path(ARGS["--out"]).write_text(json.dumps(out, indent=1))

elif MODE == "ours":
    sys.argv = sys.argv[:1]
    sys.path.insert(0, str(REPO / "tools"))
    import v2_common as V
    import torch
    from data_utils import load_priority_carbon_data_from_files, sorted_instance_files
    from due_dates import due_dates_for_directory, load_manifest
    man = load_manifest()
    out = {}
    for seed in (111, 222, 333, 444):
        pol = V.load_policy(f"10x5+carbon+priority+m_s{seed}", "v1")
        for name, dirs in (("paper_test_81", [DAN / f"data/{s}/SD2/10x5+carbon+priority" for s in PAPER_DIRS]),
                           ("held_out_49", [DAN / "data/data_train_vali/SD2/10x5+carbon+priority"])):
            g_all, s_all = {}, {}
            for d in dirs:
                data = load_priority_carbon_data_from_files(str(d)); due = due_dates_for_directory(d, man)
                names = [Path(f).name for f in sorted_instance_files(str(d))]
                idx = np.arange(len(names)) if name != "held_out_49" else np.array([names.index(n) for n in HELD_NAMES])
                g = V.run(pol, data, due, idx)["mk"]
                gen = torch.Generator(device=V.P.DEV); gen.manual_seed(20261002 + seed)
                smp = V.run(pol, data, due, np.repeat(idx, 100), gen=gen)["mk"].reshape(len(idx), 100).min(1)
                for k, i in enumerate(idx):
                    g_all[names[i]] = float(g[k]); s_all[names[i]] = float(smp[k])
            out.setdefault(name, {})[str(seed)] = {"greedy": g_all, "best_of_100": s_all}
            print(f"ours m_s{seed} {name}: greedy {np.mean(list(g_all.values())):.2f}  best-of-100 "
                  f"{np.mean(list(s_all.values())):.2f}", flush=True)
    Path(ARGS["--out"]).write_text(json.dumps(out, indent=1))
