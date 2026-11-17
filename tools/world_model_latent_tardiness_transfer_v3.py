"""Tardiness-specialist transfer arm for the centred latent encoder (single seed, 10x5, waiting allowed).

One new arm, launched with the earlier tardiness pilot's own code (tools/world_model_m1_tardiness_pilot.py: args(),
run_arm()), so the recipe is identical: configs/benchmark_v3/t.json, seed 111, generator stream, contrastive recipe
constants, --dispatch_rule semi_active, validation every 50 updates (18 instances, T / Tref), 2,000 updates, encoder
fine-tuned with the rest. Start bundle: the M1 start DANIEL bundle (the same weights outside the encoder as the earlier
random and pretrained arms) with feature_exact replaced by the ONLINE encoder of
checkpoints/world_model_latent_pilot_aux_centred_v3. The earlier arms are only read; the protocol is checked against
their saved run records (every config key equal except the run name and the start file; same data fingerprints).

    python tools/world_model_latent_tardiness_transfer_v3.py   # -> results/world_model_latent_tardiness_transfer_v3/
"""
from __future__ import annotations
import json, shutil, sys, threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import world_model_m1_tardiness_pilot as T                          # noqa: E402

EARLIER = T.RES                                                      # results/world_model_m1_v3/tardiness_pilot
EARLIER_CK = T.CK
SRC = REPO / "checkpoints" / "world_model_latent_pilot_aux_centred_v3"
NAME = "world_model_latent_tardiness_transfer_v3"
T.RES = REPO / "results" / NAME
T.CK = REPO / "checkpoints" / NAME
T.ARMS = {"latc": ("centred latent encoder", "0-3")}
ALLOWED_DIFF = {"model_suffix", "init_from"}


def build_start():
    import torch
    T.CK.mkdir(parents=True, exist_ok=True)
    b = torch.load(T.START, map_location="cpu", weights_only=False)
    enc = torch.load(SRC / "online_encoder.pth", map_location="cpu")
    sd = b["state_dict"]
    fe = {k for k in sd if k.startswith("feature_exact.")}
    assert {f"feature_exact.{k}" for k in enc} == fe
    new = dict(b); new["state_dict"] = {k: (enc[k[len("feature_exact."):]].clone() if k in fe else v.clone())
                                        for k, v in sd.items()}
    torch.save(new, T.CK / "start_latc.pth")
    a = new["state_dict"]
    checks = {}
    for arm in ("rnd", "pre"):
        r = torch.load(EARLIER_CK / f"start_{arm}.pth", map_location="cpu", weights_only=False)["state_dict"]
        checks[f"identical_outside_encoder_vs_{arm}"] = all(torch.equal(a[k], r[k]) for k in a if k not in fe)
        checks[f"encoder_differs_vs_{arm}"] = any(not torch.equal(a[k], r[k]) for k in fe)
    checks["encoder_equals_saved_online_encoder"] = all(torch.equal(a["feature_exact." + k], v) for k, v in enc.items())
    assert all(checks.values()), checks
    (T.CK / "SHA256SUMS.json").write_text(json.dumps({"start_latc.pth": T.sha(T.CK / "start_latc.pth"),
                                                      "source_online_encoder": T.sha(SRC / "online_encoder.pth")}, indent=1) + "\n")
    return checks


def protocol_check(s):
    """Command identical to the earlier arms' except run name and start file; then the saved run records agree."""
    a_new = T.args("latc", s)
    out = {}
    for arm in ("pre", "rnd"):
        T.CK, ck_new = EARLIER_CK, T.CK
        a_old = T.args(arm, s)
        T.CK = ck_new
        diff = [(x, y) for x, y in zip(a_old, a_new) if x != y]
        out[f"command_diff_vs_{arm}"] = [list(d) for d in diff]
        assert len(a_old) == len(a_new) and len(diff) == 2, diff
    return out


def record_check():
    new = json.loads((T.RES / "latc_run.json").read_text())
    out = {}
    for arm in ("pre", "rnd"):
        old = json.loads((EARLIER / f"{arm}_run.json").read_text())
        keys = set(old["config"]) | set(new["config"])
        diff = sorted(k for k in keys if old["config"].get(k) != new["config"].get(k))
        out[f"config_keys_differing_vs_{arm}"] = diff
        out[f"data_fingerprints_equal_vs_{arm}"] = old["data_fingerprints"] == new["data_fingerprints"]
        out[f"protocol_matches_{arm}"] = set(diff) <= ALLOWED_DIFF and out[f"data_fingerprints_equal_vs_{arm}"]
    return out


def main():
    T.RES.mkdir(parents=True, exist_ok=True)
    s = json.loads((EARLIER / "settings.json").read_text())         # the earlier pilot's budget, target and recipe
    (T.RES / "settings.json").write_text(json.dumps({**s, "earlier_settings": str((EARLIER / "settings.json").relative_to(REPO)),
                                                     "new_arm": {"encoder": str((SRC / "online_encoder.pth").relative_to(REPO)),
                                                                 "sha256": T.sha(SRC / "online_encoder.pth")}}, indent=1) + "\n")
    chk = {"start": build_start(), "command": protocol_check(s)}
    (T.RES / "start_check.json").write_text(json.dumps(chk, indent=1) + "\n")
    print(json.dumps(chk), flush=True)
    out = {}
    T.run_arm("latc", s, out)
    if out.get("latc") != "ok":
        raise SystemExit(f"arm failed: {out}")
    chk["record"] = record_check()
    (T.RES / "start_check.json").write_text(json.dumps(chk, indent=1) + "\n")
    summarise(s)


def summarise(s):
    curves = {"random encoder": EARLIER / "rnd_valiquality.json",
              "earlier outcome-prediction encoder": EARLIER / "pre_valiquality.json",
              "centred latent encoder": T.RES / "latc_valiquality.json"}
    rows = {}
    for lab, f in curves.items():
        v = json.loads(f.read_text())
        hit = next((e for e in v if e["selection_metric"] <= s["target"] + 1e-12), None)
        at = next(e for e in v if e["update"] == s["budget_updates"])
        rows[lab] = {"start": v[0]["selection_metric"], "reached_update": hit["update"] if hit else None,
                     "reached_hours": hit["train_seconds"] / 3600 if hit else None,
                     "at_budget": at["selection_metric"], "best": min(e["selection_metric"] for e in v),
                     "hours_total": at["train_seconds"] / 3600}
    cov = json.loads((REPO / "results/world_model_latent_pilot_v3/coverage.json").read_text())
    tl = json.loads((REPO / "results/world_model_latent_pilot_aux_centred_v3/train_log.json").read_text())
    summ = {"rows": rows, "target": s["target"], "budget": s["budget_updates"],
            "centred_pretraining_cost_s": {"data_collection": cov["collection_seconds"],
                                           "encoder_pretraining": tl["train_seconds"]}}
    (T.RES / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for (lab, f), col in zip(curves.items(), ("#7f7f7f", "#d62728", "#1f77b4")):
        v = json.loads(f.read_text())
        ax.plot([e["update"] for e in v], [e["selection_metric"] for e in v], color=col, lw=1.8, marker="o", ms=2.5, label=lab)
    ax.axhline(s["target"], color="#555555", ls="--", lw=.9, label=f"common target {s['target']:.4f}")
    ax.set_yscale("log"); ax.set_xlabel("updates"); ax.set_ylabel("validation tardiness (T / Tref, log scale)")
    ax.set_title("Tardiness specialist, waiting allowed: starting encoders (10x5, seed 111)", fontsize=11)
    ax.grid(alpha=.3); ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(T.RES / "curves.png", dpi=140, facecolor="white"); plt.close(fig)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
