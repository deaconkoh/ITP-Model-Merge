"""Arm D, vertex distillation (docs/warmstart_protocol_v3.md): a fresh preference-conditioned DANIEL student learns
pi(.|s, e_x) = pi_x(.|s) for each specialist x in {M, C, T} of the same seed, with e_x the corner preference.

1. States: each specialist samples its own rollouts on fresh SD2-generator instances with v3 parameters
   (benchmark_v3_generator.GeneratorSource: validation/development/evaluation and final-test hashes excluded), global
   numpy stream seeded seed + 7001 (disjoint from the PPO arms' instance stream), torch sampling generator seeded the
   same way. Stored on the CPU with the teacher's masked action distribution.
2. Loss: KL(pi_x(.|s) || pi_student(.|s, e_x)) over the allowed actions; mean over states. Adam, lr 3e-4, batch 256,
   fixed epochs. The student's critic receives no gradient (it stays at its initialisation and is replaced by the PPO
   arm's fresh critic: train.py --init_mode no_critic).
3. Budget: fixed state count and epochs, plus a hard wall-clock cap (collection included; default 20 min). Both the
   collection and the training seconds are recorded as D's imitation cost.
4. Held-out check: greedy-action agreement with each teacher on the teachers' greedy validation trajectories.

    python distill_v3.py --size 10x5 --seed 111 --spec_dir ../checkpoints/specialists_v3 --out <student.pth>
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--size", required=True, choices=["10x5", "20x10"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--spec_dir", default=str(HERE.parent / "checkpoints" / "specialists_v3"))
ap.add_argument("--out", required=True)
ap.add_argument("--instances", type=int, default=0, help="instances per specialist (default 400 at 10x5, 100 at 20x10)")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wall_cap", type=float, default=1200.0, help="hard cap in seconds, collection included")
ap.add_argument("--val_first", type=int, default=0, help="held-out check on the first N validation instances (0 = all)")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
N_J, N_M = {"10x5": (10, 5), "20x10": (20, 10)}[args.size]
sys.argv = [sys.argv[0], "--device", args.device, "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct",
            "--preference_dim", "3", "--n_j", str(N_J), "--n_m", str(N_M), "--data_suffix", "mix", "--low", "1",
            "--high", "99", "--op_per_job", "0"]

import copy                                                 # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from params import configs                                  # noqa: E402
import benchmark_v3 as B                                    # noqa: E402
from benchmark_v3_generator import GeneratorSource          # noqa: E402
from checkpointing import make_checkpoint_bundle, json_safe_config, file_sha256   # noqa: E402
from common_utils import load_checkpoint_state_dict        # noqa: E402
from feature_schemas import SCHEMA_V3                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums      # noqa: E402
from model.main_model import DANIEL                         # noqa: E402

KEYS = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs")
DEV = configs.device
EYE = np.eye(3)


def teacher(o):
    c = copy.copy(configs); c.preference_dim = 0; c.feature_schema = SCHEMA_V3
    net = DANIEL(c).to(DEV)
    path = Path(args.spec_dir) / f"{args.size}+mix+bv3_spec_{o}_s{args.seed}.pth"
    net.load_state_dict(load_checkpoint_state_dict(path, map_location=DEV, expected_schema=SCHEMA_V3))
    net.eval()
    return net, path


def fwd(net, st, pref=None):
    return net(**{k: getattr(st, f"{k}_tensor") for k in KEYS}, pref=pref)


@torch.no_grad()
def collect(net, data, k, gen, greedy=False):
    """Rollouts of the teacher on all instances of `data`; returns stored states, teacher probabilities, corner index."""
    env = FJSPEnvForSameOpNums(N_J, N_M)
    st = env.set_initial_data(**data.env_kwargs(range(len(data)), "mct"))
    buf = {k_: [] for k_ in KEYS}; probs = []
    while True:
        pi, _ = fwd(net, st)
        for k_ in KEYS:
            buf[k_].append(getattr(st, f"{k_}_tensor").cpu())
        probs.append(pi.cpu())
        a = pi.argmax(1) if greedy else torch.multinomial(pi, 1, generator=gen).squeeze(1)
        st, _, done = env.step(a.cpu().numpy())
        if np.all(done):
            break
    out = {k_: torch.cat(v) for k_, v in buf.items()}
    out["teacher"] = torch.cat(probs)
    out["corner"] = torch.full((len(out["teacher"]),), k, dtype=torch.long)
    return out


def cat(parts):
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def kl_loss(student, batch):
    pref = torch.as_tensor(EYE[batch["corner"].numpy()], dtype=torch.float32, device=DEV)
    sc, _, _ = student.forward_logits(**{k: batch[k].to(DEV) for k in KEYS}, pref=pref)
    logq = F.log_softmax(sc, dim=1)
    p = batch["teacher"].to(DEV)
    pos = p > 0
    return torch.where(pos, p * (torch.log(p.clamp_min(1e-30)) - logq), torch.zeros_like(p)).sum(1).mean()


def main():
    t0 = time.time()
    n_inst = args.instances or (400 if args.size == "10x5" else 100)
    man = B.load_manifest()
    np.random.seed(args.seed + 7001)
    src = GeneratorSource(configs, args.size, man)
    gen = torch.Generator(device=DEV); gen.manual_seed(args.seed + 7001)
    teachers, parts, parents = {}, [], []
    for k, o in enumerate("mct"):
        net, path = teacher(o)
        teachers[o] = net
        parents.append({"path": str(path), "sha256": file_sha256(path), "role": f"teacher {o}"})
        for c0 in range(0, n_inst, 50):
            parts.append(collect(net, src.sample(min(50, n_inst - c0)), k, gen))
    data = cat(parts)
    collect_s = time.time() - t0
    n = len(data["teacher"])
    torch.manual_seed(args.seed + 7001)
    student = DANIEL(configs).to(DEV)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    val = B.load_split("validation", args.size)
    vidx = list(range(len(val) if not args.val_first else min(args.val_first, len(val))))
    vsub = B.BenchmarkData(val.directory, val.size, t_ref=val.t_ref)
    for f in ("keys", "job_length", "op_pt", "power", "carbon", "due", "mk_ref", "cb_ref"):
        setattr(vsub, f, [getattr(val, f)[i] for i in vidx])
    held = cat([collect(teachers[o], vsub, k, gen, greedy=True) for k, o in enumerate("mct")])
    hist, capped = [], False
    t1 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        tot = 0.0
        student.train()
        for s in range(0, n, args.batch):
            b = perm[s:s + args.batch]
            loss = kl_loss(student, {k: v[b] for k, v in data.items()})
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
            if time.time() - t0 > args.wall_cap:
                capped = True; break
        student.eval()
        with torch.no_grad():
            agree, hkl = [], []
            for k in range(3):
                m = held["corner"] == k
                hb = {kk: v[m] for kk, v in held.items()}
                pref = torch.as_tensor(np.tile(EYE[k], (int(m.sum()), 1)), dtype=torch.float32, device=DEV)
                sc, _, _ = student.forward_logits(**{kk: hb[kk].to(DEV) for kk in KEYS}, pref=pref)
                agree.append(float((sc.argmax(1).cpu() == hb["teacher"].argmax(1)).float().mean()))
                hkl.append(float(kl_loss(student, hb)))
        hist.append({"epoch": ep, "train_kl": tot / n, "heldout_kl": dict(zip("mct", hkl)),
                     "heldout_greedy_agreement": dict(zip("mct", agree)), "seconds": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if capped:
            break
    train_s = time.time() - t1
    rec = {"size": args.size, "seed": args.seed, "states": n, "instances_per_teacher": n_inst,
           "epochs_run": len(hist), "epochs_planned": args.epochs, "wall_cap_s": args.wall_cap, "wall_cap_hit": capped,
           "collection_seconds": collect_s, "training_seconds": train_s, "total_seconds": time.time() - t0,
           "generator": src.provenance(), "history": hist, "teachers": parents,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    last = hist[-1]
    torch.save(make_checkpoint_bundle(
        student.state_dict(), config=json_safe_config(configs), data_fingerprints={}, update=0,
        validation_metrics={f"heldout_agreement_{o}": last["heldout_greedy_agreement"][o] for o in "mct"},
        selection_metric=float("nan"), parents=parents, repo_root=HERE.parent), out)
    out.with_suffix(".distill.json").write_text(json.dumps(rec, indent=1))
    print(f"student saved: {out} ({n} states, {rec['total_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
