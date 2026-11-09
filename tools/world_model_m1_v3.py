"""World-model trunk, Milestone 1 (docs/world_model_m1_protocol_v3.md): data generation, pretraining, prediction sanity
(Question A) and frozen-trunk transfer probes (Question B). 10x5 only.

    python tools/world_model_m1_v3.py generate --split train      # generator stream, resumable shards
    python tools/world_model_m1_v3.py generate --split val        # the 18 v3 validation instances
    python tools/world_model_m1_v3.py pretrain
    python tools/world_model_m1_v3.py sanity
    python tools/world_model_m1_v3.py probe
    (add --smoke to every step: tiny sizes, separate outputs)

Every setting comes from the frozen configs/world_model_m1_v3.json (tools/freeze_world_model_m1_v3.py). Data are
written to data/world_model_m1_v3[_smoke]/ (gitignored); results to results/world_model_m1_v3[_smoke]/, checkpoints
to checkpoints/world_model_m1_v3[_smoke]/. Never touches the evaluation split or the reserved final-test sets (the
generator's disjointness check regenerates the final-test stream in memory, as for every specialist; no file is read).
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, random, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import tardiness_diagnosis_v3 as T                                  # noqa: E402  (params for v3, the no-waiting decoder)
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

B, configs = T.B, T.configs
from common_utils import setup_seed                                 # noqa: E402
from checkpointing import make_checkpoint_bundle, json_safe_config, unwrap_checkpoint, validate_checkpoint_schema  # noqa
from feature_schemas import SCHEMA_V3, SCHEMA_V3_LATE               # noqa: E402
from model.main_model import DANIEL, DualAttentionNetwork          # noqa: E402
import world_model as WM                                            # noqa: E402
import freeze_world_model_m1_v3 as F                                # noqa: E402

SIZE = "10x5"
CONT = ["MWKR+ECT", "MWKR+MINE", "EDD+ECT"]                         # continuation rules seen in pretraining
UNSEEN = "LST+ECT"                                                  # the transfer task's continuation
BEHAVIOURS = ["MWKR+ECT", "MWKR+MINE", "EDD+ECT", "RANDOM", "MIX"]
STATE_KEYS = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask")
# compact storage: features float32, 0/1 masks uint8 (checked to be exactly 0/1 when written)
DTYPES = {"fea_j": np.float32, "op_mask": np.uint8, "candidate": np.int16, "fea_m": np.float32, "mch_mask": np.uint8,
          "comp_idx": np.uint8, "dynamic_pair_mask": np.uint8}


def paths(smoke):
    s = "_smoke" if smoke else ""
    return {"data": REPO / "data" / f"world_model_m1_v3{s}", "res": REPO / "results" / f"world_model_m1_v3{s}",
            "ckpt": REPO / "checkpoints" / f"world_model_m1_v3{s}"}


def settings(smoke):
    cfg = F.load()
    return {**cfg, **cfg["smoke"]} if smoke else cfg


def set_features(cfg):
    on = cfg["feature_set"] == "lateness"
    configs.lateness_inputs = "on" if on else "off"
    configs.fea_j_input_dim = 17 if on else 12
    configs.feature_schema = SCHEMA_V3_LATE if on else SCHEMA_V3
    configs.n_j, configs.n_m = B.SIZES[SIZE]


# ------------------------------------------------------------------------------------- decoder with all the rules

class Decoder(T.NonDelay):
    """The verified no-waiting decoder plus machine energy, vectorised job sums, the MINE machine rule, targets and
    DANIEL-style shortcuts."""

    def __init__(self, data, idx):
        super().__init__(data, idx)
        self.carbon = np.stack([np.asarray(data.carbon[i], dtype=np.float64) for i in self.idx])        # kg [E, N, M]
        big = np.where(self.elig, self.raw, np.inf)
        self.min_raw = big.min(2); self.fast_m = big.argmin(2)
        cb = np.where(self.elig, self.carbon, np.inf)
        self.min_cb = cb.min(2); self.green_m = cb.argmin(2)

    def job_sum(self, per_op):
        cs = np.cumsum(per_op, 1)
        hi = np.take_along_axis(cs, self.last, 1)
        lo = np.where(self.first > 0, np.take_along_axis(cs, np.maximum(self.first - 1, 0), 1), 0.0)
        return hi - lo

    def rule(self, name, ok, nst, rng=None):
        if name == "RANDOM":
            return np.where(ok.reshape(self.E, -1), rng.random((self.E, self.J * self.M)), -1.0).argmax(1)
        job_rule, mch_rule = name.split("+")
        can = ok.any(2)
        if job_rule == "EDD":
            score = self.due / B.PT_SCALE
        elif job_rule == "MWKR":
            score = -self.job_sum(self.mean_pt * ~self.sched)
        elif job_rule == "LST":
            score = self.due / B.PT_SCALE - nst[:, None] - self.job_sum(self.min_pt * ~self.sched)
        else:
            raise ValueError(name)
        j = np.where(can, score, np.inf).argmin(1)
        okj = ok[self.ei, j]                                                          # [E, M]
        op = self.cand[self.ei, j]
        pt = self.pt[self.ei, op]
        if mch_rule == "ECT":
            m = np.where(okj, pt, np.inf).argmin(1)
        elif mch_rule == "MINE":                                      # least energy; ties: ECT, then lowest index
            en = np.where(okj, self.carbon[self.ei, op], np.inf)
            tie = okj & (en == en.min(1, keepdims=True))
            m = np.where(tie, pt, np.inf).argmin(1)
        else:
            raise ValueError(name)
        return j * self.M + m

    def outcome(self):
        """Exact targets of the finished schedule (raw): job completion [E, J], machine busy time and energy [E, M]."""
        oh = (self.machine[:, :, None] == np.arange(self.M)[None, None, :])
        busy = (oh * np.take_along_axis(self.raw, self.machine[:, :, None], 2)).sum(1)
        energy = (oh * np.take_along_axis(self.carbon, self.machine[:, :, None], 2)).sum(1)
        return self.rr.copy(), busy, energy

    def shortcut(self):
        """At the current (partial) state: DANIEL's earliest-possible job completion (last scheduled completion + the
        remaining minimum processing times, = the environment's op_ct_lb of the job's last operation x 99), and the
        analogous machine estimate: committed busy time / energy plus every unscheduled operation credited to its
        fastest (busy) or lowest-energy (energy) machine."""
        un = ~self.sched
        ms = np.arange(self.M)[None, None, :]
        job = self.rr + self.job_sum(self.min_raw * un)
        done = (self.machine[:, :, None] == ms) & self.sched[:, :, None]                       # [E, N, M]
        fast = (self.fast_m[:, :, None] == ms) & un[:, :, None]
        green = (self.green_m[:, :, None] == ms) & un[:, :, None]
        busy = (done * np.where(self.elig, self.raw, 0)).sum(1) + (fast * self.min_raw[:, :, None]).sum(1)
        energy = (done * np.where(self.elig, self.carbon, 0)).sum(1) + (green * self.min_cb[:, :, None]).sum(1)
        return job, busy, energy


# ------------------------------------------------------------------------------------------------ generation

def behaviour_episodes(data, idx, beh, steps, n_cand, rng):
    """Runs the real environment and the decoder in lockstep under per-episode behaviour policies; snapshots the
    environment's input tensors at the given steps. beh [E] behaviour index, steps: list of step arrays per episode.
    Returns per-state records and every episode's full action sequence."""
    E = len(idx)
    env = T.FJSPEnvForSameOpNums(*B.SIZES[SIZE]); env.set_initial_data(**data.env_kwargs(idx, configs.goal))
    d = Decoder(data, idx); d.reset()
    want = np.zeros((E, d.N), bool)
    for e, st in enumerate(steps):
        want[e, st] = True
    S = {k: [] for k in STATE_KEYS}
    rec = {"episode": [], "step": [], "behaviour": [], "cand": []}
    for t in range(d.N):
        ok, nst = d.allowed()
        if not np.array_equal(ok, ~env.dynamic_pair_mask.astype(bool)):
            raise AssertionError("decoder and environment masks disagree")
        acts = {b: d.rule(b, ok, nst, rng) for b in ("MWKR+ECT", "MWKR+MINE", "EDD+ECT", "RANDOM")}
        mix = np.where(rng.random(E) < 0.3, acts["RANDOM"], acts["MWKR+ECT"])
        table = np.stack([acts["MWKR+ECT"], acts["MWKR+MINE"], acts["EDD+ECT"], acts["RANDOM"], mix], 1)
        a = table[np.arange(E), beh]
        for e in np.flatnonzero(want[:, t]):
            for k in STATE_KEYS:
                x = np.asarray(getattr(env, k)[e])
                if DTYPES[k] is np.uint8 and not np.isin(x, (0, 1)).all():
                    raise AssertionError(f"{k} is not a 0/1 mask")
                S[k].append(x.astype(DTYPES[k]))
            others = np.setdiff1d(np.flatnonzero(ok[e].reshape(-1)), [a[e]])
            pick = rng.choice(others, size=min(n_cand - 1, len(others)), replace=False) if len(others) else []
            c = np.full(n_cand, -1); c[0] = a[e]; c[1:1 + len(pick)] = pick
            rec["episode"].append(e); rec["step"].append(t); rec["behaviour"].append(beh[e]); rec["cand"].append(c)
        d.step(a); env.step(a)
    return ({k: np.stack(v) for k, v in S.items()}, {k: np.asarray(v) for k, v in rec.items()}, d.actions.copy())


def continuations(data, idx, ep_actions, rec, rules, with_shortcut, chunk=6000, keep=None):
    """Finishes (state, candidate action) with every rule. Returns targets per (state, action, rule): jobs [S, A, R, J],
    machines [S, A, R, M, 2] (raw), and shortcuts after the action [S, A, J], [S, A, M, 2]."""
    S, A = rec["cand"].shape
    R = len(rules)
    rows = [(s, a, r) for s in range(S) for a in range(A) if rec["cand"][s, a] >= 0 for r in range(R)]
    N, J, M = ep_actions.shape[1], B.SIZES[SIZE][0], B.SIZES[SIZE][1]
    tj = np.full((S, A, R, J), np.nan); tm = np.full((S, A, R, M, 2), np.nan)
    sj = np.full((S, A, J), np.nan); sm = np.full((S, A, M, 2), np.nan)
    for c0 in range(0, len(rows), chunk):
        part = np.asarray(rows[c0:c0 + chunk])
        s, a, r = part[:, 0], part[:, 1], part[:, 2]
        ep = rec["episode"][s]
        d = Decoder(data, np.asarray(idx)[ep]); d.reset()
        k = rec["step"][s]; cand = rec["cand"][s, a]; prefix = ep_actions[ep]
        for t in range(N):
            ok, nst = d.allowed()
            act = np.zeros(d.E, int)
            for ri, name in enumerate(rules):
                sel = r == ri
                if sel.any():
                    act[sel] = d.rule(name, ok, nst)[sel]
            act = np.where(t < k, prefix[:, t], act)
            act = np.where(t == k, cand, act)
            d.step(act)
            if with_shortcut:
                hit = (t == k) & (r == 0)
                if hit.any():
                    j_, b_, e_ = d.shortcut()
                    sj[s[hit], a[hit]] = j_[hit]
                    sm[s[hit], a[hit], :, 0] = b_[hit]; sm[s[hit], a[hit], :, 1] = e_[hit]
        jc, busy, energy = d.outcome()
        tj[s, a, r] = jc; tm[s, a, r, :, 0] = busy; tm[s, a, r, :, 1] = energy
        if keep is not None:                    # tests: the decoded schedules, to replay through the environment
            keep.append({"rows": part, "instance": np.asarray(idx)[ep], "actions": d.actions.copy(),
                         "machine": d.machine.copy(), "start": d.start.copy()})
    return tj, tm, sj, sm


def save_shard(path, states, rec, extra):
    np.savez(path, **{k: v for k, v in states.items()}, **{f"rec_{k}": v for k, v in rec.items()}, **extra)


def generate(split, smoke):
    cfg = settings(smoke); set_features(cfg)
    P = paths(smoke); out = P["data"] / split; out.mkdir(parents=True, exist_ok=True)
    n_cand = cfg["n_candidates"]
    log = Logger(P["res"])
    t_start = time.time()
    if split == "val":
        data = B.load_split("validation", SIZE)
        n = min(len(data), cfg["val_instances"])
        rng = np.random.default_rng(cfg["val_seed"])
        idx = np.repeat(np.arange(n), len(BEHAVIOURS))
        beh = np.tile(np.arange(len(BEHAVIOURS)), n)
        N = data.op_pt[0].shape[0]
        steps = [np.arange(0, N - 1, cfg["val_step_stride"]) for _ in idx]
        states, rec, ep_actions = behaviour_episodes(data, idx, beh, steps, n_cand, rng)
        tj, tm, sj, sm = continuations(data, idx, ep_actions, rec, CONT + [UNSEEN], True)
        inst = idx[rec["episode"]]
        save_shard(out / "shard_0000.npz", states, rec,
                   {"instance": inst, "tj": tj, "tm": tm, "sj": sj, "sm": sm,
                    "due": np.stack([data.due[i] for i in inst]), "keys": np.asarray([data.keys[i] for i in inst])})
        meta = {"split": "validation", "instances": int(n), "states": int(len(inst)),
                "seconds": time.time() - t_start}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        log(f"val: {n} instances, {len(inst)} states, {meta['seconds']:.0f} s")
        return
    # ---- training stream: the SD2 generator (seeded), resumable by shard
    from benchmark_v3_generator import GeneratorSource
    state_file = out / "rng_state.pkl"
    n_shards = cfg["train_instances"] // cfg["shard_instances"]
    done = sorted(out.glob("shard_*.npz"))
    if state_file.exists() and done:
        st = pickle.loads(state_file.read_bytes())
        np.random.set_state(st["np"]); random.setstate(st["py"])
        gen = st["gen"]; rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]
        first = st["next_shard"]; elapsed = st["seconds"]
    else:
        setup_seed(cfg["train_seed"])
        gen = GeneratorSource(configs, SIZE, B.load_manifest())
        rng = np.random.default_rng(cfg["train_seed"]); first = 0; elapsed = 0.0
    for sh in range(first, n_shards):
        t0 = time.time()
        data = gen.sample(cfg["shard_instances"])
        n = len(data)
        idx = np.arange(n)
        beh = rng.integers(0, len(BEHAVIOURS), n)
        N = data.op_pt[0].shape[0]
        steps = [np.sort(rng.choice(N - 1, size=cfg["states_per_episode"], replace=False)) for _ in idx]
        states, rec, ep_actions = behaviour_episodes(data, idx, beh, steps, n_cand, rng)
        tj, tm, _, _ = continuations(data, idx, ep_actions, rec, CONT, False)
        g_inst = sh * cfg["shard_instances"] + rec["episode"]
        save_shard(out / f"shard_{sh:04d}.npz", states, rec,
                   {"instance": g_inst, "holdout": (g_inst % 10) == 9, "tj": tj.astype(np.float32),
                    "tm": tm.astype(np.float32)})
        elapsed += time.time() - t0
        state_file.write_bytes(pickle.dumps({"np": np.random.get_state(), "py": random.getstate(), "gen": gen,
                                             "rng": rng.bit_generator.state, "next_shard": sh + 1,
                                             "seconds": elapsed}))
        if sh % 10 == 0 or sh == n_shards - 1:
            log(f"train shard {sh + 1}/{n_shards}: {(sh + 1) * cfg['shard_instances']} instances, {elapsed / 60:.1f} min")
    files = sorted(out.glob("shard_*.npz"))
    meta = {"split": "generator", "instances": n_shards * cfg["shard_instances"],
            "states": int(sum(np.load(f)["instance"].shape[0] for f in files)),
            "bytes": int(sum(f.stat().st_size for f in files)), "seconds": elapsed,
            "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            "generator": gen.provenance()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"train: {meta['instances']} instances, {meta['states']} states, {meta['bytes'] / 1e9:.2f} GB, "
        f"{elapsed / 60:.1f} min")


class Logger:
    def __init__(self, res):
        res.mkdir(parents=True, exist_ok=True); self.f = open(res / "run.log", "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True); self.f.write(line + "\n"); self.f.flush()


# ------------------------------------------------------------------------------------------------ datasets

def load_split(smoke, split):
    """All shards of a split, filled into preallocated arrays (no concatenation copy: the PC has 11 GB of RAM)."""
    files = sorted((paths(smoke)["data"] / split).glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"no {split} shards; run generate first")
    sizes = [np.load(f)["instance"].shape[0] for f in files]
    first = np.load(files[0])
    out = {k: np.empty((sum(sizes),) + first[k].shape[1:], first[k].dtype) for k in first.files}
    o = 0
    for f, n in zip(files, sizes):
        z = np.load(f)
        for k in z.files:
            out[k][o:o + n] = z[k]
        o += n
    return out


def normalise(cfg, tj, tm):
    tj = (tj / cfg["t_norm"]).astype(np.float32)
    return tj, np.stack([tm[..., 0] / cfg["t_norm"], tm[..., 1] / cfg["e_norm"]], -1).astype(np.float32)


def batch_tensors(D, sel, dev):
    s = {"fea_j": torch.as_tensor(D["fea_j"][sel], dtype=torch.float32, device=dev),
         "op_mask": torch.as_tensor(D["op_mask"][sel], dtype=torch.float32, device=dev),
         "candidate": torch.as_tensor(D["candidate"][sel], dtype=torch.int64, device=dev),
         "fea_m": torch.as_tensor(D["fea_m"][sel], dtype=torch.float32, device=dev),
         "mch_mask": torch.as_tensor(D["mch_mask"][sel], dtype=torch.float32, device=dev),
         "comp_idx": torch.as_tensor(D["comp_idx"][sel], dtype=torch.float32, device=dev)}
    c = D["rec_cand"][sel]
    M = B.SIZES[SIZE][1]
    job = torch.as_tensor(np.where(c >= 0, c // M, -1), device=dev)
    mch = torch.as_tensor(np.where(c >= 0, c % M, -1), device=dev)
    return s, job, mch, torch.as_tensor(c >= 0, device=dev)


# ------------------------------------------------------------------------------------------------ pretraining

def trunk_bundle(trunk_sd, cfg, meta, epoch, value):
    """The trunk as a full DANIEL bundle (fresh seed-111 actor and critic), so train.py can load it
    (--init_mode backbone into a preference-conditioned DANIEL, or --init_mode exact into a specialist)."""
    setup_seed(cfg["train_seed"])
    net = DANIEL(configs)
    sd = net.state_dict()
    for k, v in trunk_sd.items():
        sd["feature_exact." + k] = v.detach().cpu().clone()
    return make_checkpoint_bundle({k: v.cpu() for k, v in sd.items()}, config=json_safe_config(configs),
                                  data_fingerprints={"world_model_data": meta}, update=epoch,
                                  validation_metrics={"holdout_loss": value}, selection_metric=value,
                                  repo_root=REPO)


def pretrain(smoke):
    cfg = settings(smoke); set_features(cfg)
    P = paths(smoke); log = Logger(P["res"]); P["ckpt"].mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    configs.device = dev
    D = load_split(smoke, "train")
    meta = json.loads((P["data"] / "train" / "meta.json").read_text())
    hold = D["holdout"].astype(bool)
    tr, ho = np.flatnonzero(~hold), np.flatnonzero(hold)
    tj, tm = normalise(cfg, D["tj"], D["tm"])
    # mean predictor (training part, per output and continuation), raw units
    valid = D["rec_cand"][tr] >= 0
    mean = {"job": np.nanmean(D["tj"][tr][valid], 0).tolist(), "mch": np.nanmean(D["tm"][tr][valid], 0).tolist()}
    (P["res"] / "mean_predictor.json").write_text(json.dumps(mean))
    setup_seed(cfg["train_seed"])
    model = WM.WorldModel(configs).to(dev)
    start = {k: v.detach().clone() for k, v in model.trunk.state_dict().items()}
    torch.save(trunk_bundle(start, cfg, meta["sha256"], 0, float("nan")), P["ckpt"] / "world_model_trunk_start.pth")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(cfg["train_seed"])
    bs = cfg["batch_states"]

    def run(rows, train):
        model.train(train); tot, n = 0.0, 0
        order = rows[torch.randperm(len(rows), generator=g).numpy()] if train else rows
        for b0 in range(0, len(order), bs):
            sel = order[b0:b0 + bs]
            s, job, mch, val = batch_tensors(D, sel, dev)
            tgt = (torch.as_tensor(np.nan_to_num(tj[sel]), dtype=torch.float32, device=dev),
                   torch.as_tensor(np.nan_to_num(tm[sel]), dtype=torch.float32, device=dev))
            with torch.set_grad_enabled(train):
                L = WM.loss(model(s, job, mch), tgt, val)
            if train:
                opt.zero_grad(); L.backward(); opt.step()
            tot += L.item() * len(sel); n += len(sel)
        return tot / max(n, 1)

    hist, best, best_ep, wait, t0 = [], float("inf"), 0, 0, time.time()
    best_sd = None
    for ep in range(1, cfg["max_epochs"] + 1):
        lt = run(tr, True)
        with torch.no_grad():
            lh = run(ho, False)
        hist.append({"epoch": ep, "train_loss": lt, "holdout_loss": lh, "seconds": time.time() - t0})
        log(f"pretrain epoch {ep}: train {lt:.5f}, hold-out {lh:.5f} ({(time.time() - t0) / 60:.1f} min)")
        if lh < best:
            best, best_ep, wait = lh, ep, 0
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_sd)
    trunk_sd = {k[len("trunk."):]: v for k, v in best_sd.items() if k.startswith("trunk.")}
    torch.save(trunk_bundle(trunk_sd, cfg, meta["sha256"], best_ep, best), P["ckpt"] / "world_model_trunk.pth")
    torch.save({"state_dict": {k: v for k, v in best_sd.items() if not k.startswith("trunk.")},
                "epoch": best_ep, "holdout_loss": best}, P["ckpt"] / "world_model_heads.pth")
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(P["ckpt"].glob("*.pth"))}
    (P["ckpt"] / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1) + "\n")
    out = {"epochs_run": len(hist), "best_epoch": best_ep, "best_holdout_loss": best, "seconds": time.time() - t0,
           "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
           "train_states": int(len(tr)), "holdout_states": int(len(ho)), "history": hist}
    (P["res"] / "pretrain.json").write_text(json.dumps(out, indent=1))
    log(f"pretrain: best epoch {best_ep} of {len(hist)}, hold-out loss {best:.5f}, {out['seconds'] / 60:.1f} min")


def load_world_model(smoke, dev):
    P = paths(smoke)
    model = WM.WorldModel(configs).to(dev)
    tb = torch.load(P["ckpt"] / "world_model_trunk.pth", map_location=dev, weights_only=False)
    model.trunk.load_state_dict({k[len("feature_exact."):]: v for k, v in tb["state_dict"].items()
                                 if k.startswith("feature_exact.")})
    hb = torch.load(P["ckpt"] / "world_model_heads.pth", map_location=dev, weights_only=False)
    model.load_state_dict({**model.state_dict(), **hb["state_dict"]})
    return model.eval()


# ------------------------------------------------------------------------------------------------ Question A

GROUPS = ("job completion (min)", "machine busy time (min)", "machine energy (kg CO2e)")


def sanity(smoke):
    cfg = settings(smoke); set_features(cfg)
    P = paths(smoke); log = Logger(P["res"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"; configs.device = dev
    V = load_split(smoke, "val")
    model = load_world_model(smoke, dev)
    R = len(CONT)
    pj, pm = [], []
    with torch.no_grad():
        for b0 in range(0, len(V["instance"]), 256):
            sel = np.arange(b0, min(b0 + 256, len(V["instance"])))
            s, job, mch, _ = batch_tensors(V, sel, dev)
            a, b = model(s, job, mch)
            pj.append(a.cpu().numpy()); pm.append(b.cpu().numpy())
    pj = np.concatenate(pj) * cfg["t_norm"]
    pm = np.concatenate(pm); pm = np.stack([pm[..., 0] * cfg["t_norm"], pm[..., 1] * cfg["e_norm"]], -1)
    tj, tm = V["tj"][:, :, :R], V["tm"][:, :, :R]
    mean = json.loads((P["res"] / "mean_predictor.json").read_text())
    mj = np.broadcast_to(np.asarray(mean["job"]), tj.shape); mm = np.broadcast_to(np.asarray(mean["mch"]), tm.shape)
    sj = np.broadcast_to(V["sj"][:, :, None], tj.shape); sm = np.broadcast_to(V["sm"][:, :, None], tm.shape)
    valid = V["rec_cand"] >= 0
    out = {"states": int(len(V["instance"])), "samples": int(valid.sum() * R), "groups": {}}
    for gi, g in enumerate(GROUPS):
        if gi == 0:
            t, preds = tj[valid], {"world model": pj[valid], "mean": mj[valid], "shortcut": sj[valid]}
            scale = cfg["t_norm"]
        else:
            t = tm[valid][..., gi - 1]
            preds = {"world model": pm[valid][..., gi - 1], "mean": mm[valid][..., gi - 1],
                     "shortcut": sm[valid][..., gi - 1]}
            scale = cfg["t_norm"] if gi == 1 else cfg["e_norm"]
        out["groups"][g] = {k: {"mae": float(np.mean(np.abs(v - t))), "mae_norm": float(np.mean(np.abs(v - t)) / scale)}
                            for k, v in preds.items()}
        wm = out["groups"][g]["world model"]["mae"]
        out["groups"][g]["world_model_loses"] = bool(wm >= out["groups"][g]["mean"]["mae"]
                                                     or wm >= out["groups"][g]["shortcut"]["mae"])
    (P["res"] / "sanity.json").write_text(json.dumps(out, indent=1))
    log("sanity: " + "; ".join(f"{g}: WM {v['world model']['mae']:.2f}, mean {v['mean']['mae']:.2f}, "
                               f"shortcut {v['shortcut']['mae']:.2f}" for g, v in out["groups"].items()))


# ------------------------------------------------------------------------------------------------ Question B

def trunk_sources(cfg, smoke):
    return {"world-model trunk": paths(smoke)["ckpt"] / "world_model_trunk.pth",
            "random initialisation (seed 111)": None,
            "equal-weight shared base": REPO / cfg["trunks"]["shared_base"]["path"],
            "makespan specialist (context only)": REPO / cfg["trunks"]["makespan_specialist"]["path"]}


def load_trunk(path, cfg, dev):
    trunk = DualAttentionNetwork(configs).to(dev)
    if path is None:
        setup_seed(cfg["train_seed"])
        trunk.load_state_dict(DANIEL(configs).feature_exact.state_dict())
    else:
        sd, meta = unwrap_checkpoint(torch.load(path, map_location=dev, weights_only=False))
        validate_checkpoint_schema(meta, configs.feature_schema, sd)
        trunk.load_state_dict({k[len("feature_exact."):]: v for k, v in sd.items() if k.startswith("feature_exact.")})
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk.eval()


def embeddings(trunk, V, dev):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(V["instance"]), 256):
            sel = np.arange(b0, min(b0 + 256, len(V["instance"])))
            s, job, mch, _ = batch_tensors(V, sel, dev)
            h_op, h_mch, g_op, g_mch = WM.trunk_forward(trunk, s)
            a, _ = WM.readout(h_op, h_mch, g_op, g_mch, s["candidate"], job, mch)
            out.append(a.cpu().numpy())
    return np.concatenate(out)                                                     # [S, A, 32]


def transfer_targets(cfg, V):
    """Under the unseen LST+ECT continuation (rule index 3): makespan / T_NORM, carbon / (E_NORM x M), tardiness / Tref."""
    jc = V["tj"][:, :, 3]; en = V["tm"][:, :, 3, :, 1]
    mk = jc.max(-1); cb = en.sum(-1)
    td = np.maximum(0.0, jc - V["due"][:, None, :]).sum(-1)
    M = B.SIZES[SIZE][1]
    return np.stack([mk / cfg["t_norm"], cb / (cfg["e_norm"] * M), td / cfg["t_ref"]], -1)       # [S, A, 3]


def fit_probe(kind, Xtr, Ytr, Xva, Yva, seed, cfg):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = (torch.nn.Linear(d, 3) if kind == "linear" else
           torch.nn.Sequential(torch.nn.Linear(d, cfg["probe_hidden"]), torch.nn.ReLU(), torch.nn.Linear(cfg["probe_hidden"], 3)))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["probe_lr"])
    Xt, Yt = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(Ytr, dtype=torch.float32)
    Xv, Yv = torch.as_tensor(Xva, dtype=torch.float32), torch.as_tensor(Yva, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best, wait, best_sd = float("inf"), 0, None
    for ep in range(cfg["probe_max_epochs"]):
        perm = torch.randperm(len(Xt), generator=g)
        for b0 in range(0, len(Xt), 256):
            i = perm[b0:b0 + 256]
            loss = torch.nn.functional.mse_loss(net(Xt[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(net(Xv), Yv))
        if v < best - 1e-9:
            best, wait, best_sd = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["probe_patience"]:
                break
    net.load_state_dict(best_sd)
    return net


def probe(smoke):
    cfg = settings(smoke); set_features(cfg)
    P = paths(smoke); log = Logger(P["res"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"; configs.device = dev
    V = load_split(smoke, "val")
    Y = transfer_targets(cfg, V)
    valid = V["rec_cand"] >= 0
    inst = V["instance"]
    ids = np.unique(inst)
    test_ids = ids[1::2]; train_ids = ids[0::2]
    es_ids = train_ids[-max(1, len(train_ids) // 5):]                               # early-stop instances
    fit_ids = np.setdiff1d(train_ids, es_ids)
    split = {"probe_train": train_ids.tolist(), "probe_test": test_ids.tolist(), "early_stop": es_ids.tolist()}

    def rows(ids_):
        m = np.isin(inst, ids_)[:, None] & valid
        return m
    out = {"split": split, "trunks": {}}
    for name, path in trunk_sources(cfg, smoke).items():
        trunk = load_trunk(path, cfg, dev)
        X = embeddings(trunk, V, dev)
        before = {k: v.clone() for k, v in trunk.state_dict().items()}
        mf, me, mt = rows(fit_ids), rows(es_ids), rows(test_ids)
        mu, sd = X[mf].mean(0), X[mf].std(0) + 1e-6                                 # standardise on probe-train
        Z = (X - mu) / sd
        res = {}
        for kind in ("linear", "mlp"):
            maes, accs = [], []
            for seed in range(cfg["probe_seeds"]):
                net = fit_probe(kind, Z[mf], Y[mf], Z[me], Y[me], seed, cfg)
                with torch.no_grad():
                    pred = net(torch.as_tensor(Z, dtype=torch.float32)).numpy()        # [S, A, 3]
                maes.append(np.abs(pred[mt] - Y[mt]).mean(0))
                accs.append(pairwise_accuracy(pred, Y, mt))
            maes, accs = np.asarray(maes), np.asarray(accs)
            res[kind] = {"mae_mean": maes.mean(0).tolist(), "mae_std": maes.std(0).tolist(),
                         "pair_acc_mean": accs.mean(0).tolist(), "pair_acc_std": accs.std(0).tolist()}
        assert all(torch.equal(before[k], v) for k, v in trunk.state_dict().items()), "a frozen trunk changed"
        out["trunks"][name] = {"source": str(path.relative_to(REPO)) if path else "seed 111 fresh DANIEL",
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None, **res}
        log(f"probe {name}: MLP MAE {np.round(res['mlp']['mae_mean'], 4).tolist()}")
    out["n_test_samples"] = int(rows(test_ids).sum()); out["n_fit_samples"] = int(rows(fit_ids).sum())
    (P["res"] / "probe.json").write_text(json.dumps(out, indent=1))


def pairwise_accuracy(pred, Y, mask):
    """Share of same-state candidate pairs (different true values) that the prediction orders correctly."""
    acc = []
    for o in range(3):
        ok = tot = 0
        A = Y.shape[1]
        for i in range(A):
            for j in range(i + 1, A):
                m = mask[:, i] & mask[:, j]
                dt = Y[m, i, o] - Y[m, j, o]; dp = pred[m, i, o] - pred[m, j, o]
                use = np.abs(dt) > 1e-9
                ok += int((np.sign(dt[use]) == np.sign(dp[use])).sum()); tot += int(use.sum())
        acc.append(ok / max(tot, 1))
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["generate", "pretrain", "sanity", "probe"])
    ap.add_argument("--split", choices=["train", "val"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.action == "generate":
        generate(a.split, a.smoke)
    else:
        {"pretrain": pretrain, "sanity": sanity, "probe": probe}[a.action](a.smoke)


if __name__ == "__main__":
    main()
