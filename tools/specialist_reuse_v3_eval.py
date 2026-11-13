"""Evaluation of one specialist-reuse checkpoint (docs/specialist_reuse_protocol_v3.md), waiting rule (semi-active):
greedy and best-of-K sampling (K 128, seed 51, evaluate_mo_v3's layout: one instance per batch, generator seeded once)
on the evaluation split, 66 grid preferences + equal weights. One schedule per preference per decoder is saved
(machine, start, objectives) with the batched seconds per instance-preference pair; every schedule is re-scored by
the independent scorer.

    python tools/specialist_reuse_v3_eval.py --model final.pth --out eval/method.npz [--first N] [--K 128]
"""
from __future__ import annotations
import argparse, hashlib, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--K", type=int, default=128)
ap.add_argument("--seed", type=int, default=51)
a = ap.parse_args()
os.environ["DECODING_DEVICE"] = "cuda"
sys.argv = sys.argv[:1]
sys.path.insert(0, str(REPO / "tools"))
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import decoding_v3 as D                                             # noqa: E402
import benchmark_v3 as B                                            # noqa: E402
D.configs.dispatch_rule = "semi_active"                             # the waiting rule in every environment built below


def main():
    data = B.load_split("evaluation", "10x5")
    I = a.first or len(data)
    idx = [i for i in range(I) for _ in range(len(D.PREFS))]
    prefs = np.tile(D.PREFS, (I, 1))
    net = D.load_net(Path(a.model))
    env = D.make_env(data, [0], D.PREFS[:1])
    assert env.dispatch_rule == "semi_active"
    out = {}
    for dec in ("greedy", "sample"):
        torch.cuda.synchronize(); t0 = time.time()
        if dec == "greedy":
            c = D.Cost(len(idx)); o, m, s = D.greedy(net, data, idx, prefs, c)
        else:
            c = D.Cost(len(idx)); r = D.sample_best(net, data, idx, prefs, a.K, a.seed, c, keep=(a.K,)); o, m, s = r[a.K]
        torch.cuda.synchronize(); sec = time.time() - t0
        for k, i in enumerate(idx):
            ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i], m[k], s[k])
            assert max(abs(ind["makespan"] - o[k, 0]), abs(ind["carbon"] - o[k, 1]), abs(ind["tardiness"] - o[k, 2])) < 1e-6
        name = "greedy" if dec == "greedy" else f"sample{a.K}"
        out.update({f"{name}_obj": o.reshape(I, -1, 3), f"{name}_machine": m.reshape(I, len(D.PREFS), -1),
                    f"{name}_start": s.reshape(I, len(D.PREFS), -1), f"{name}_seconds": sec,
                    f"{name}_seconds_per_pair": sec / len(idx)})
        print(f"{name}: {sec:.0f} s ({sec / len(idx):.4f} s per pair)", flush=True)
    out["model_sha256"] = hashlib.sha256(Path(a.model).read_bytes()).hexdigest()
    out["keys"] = np.asarray(data.keys[:I])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)


if __name__ == "__main__":
    main()
