"""Figures, the CP-SAT subset table, the chosen-configuration due-date manifest and the PROPOSED problem spec
(results/objective_validation.md).

  results/objective_validation/heatmap_edd_tardy_share.png
  results/objective_validation/scatter_examples_<size>.png
  results/objective_validation/tardiness_distributions.png
  results/objective_validation/cpsat_table.md
  daniel/data/objective_dev/manifest_objective_dev_duedates.json  (d_j for the chosen (k, s), all DEV instances)
  configs/problem_spec_v3_PROPOSED.yaml

    python tools/objective_validation/analyze.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                       # noqa: E402
import diagnostics as D                                  # noqa: E402

DIAG = json.loads((C.OUT / "diagnostics.json").read_text())


def heatmap():
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, size in zip(axs, ("10x5", "20x10")):
        g = {(r["k"], r["s"]): r for r in DIAG[size]["grid"]}
        Z = np.array([[100 * g[(k, s)]["T1_edd_tardy_share"] for k in C.K_GRID] for s in C.S_GRID])
        im = ax.imshow(Z, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=100)
        for a, s in enumerate(C.S_GRID):
            for b, k in enumerate(C.K_GRID):
                ok = D.passes(g[(k, s)])
                ch = DIAG[size]["chosen"] and (k, s) == (DIAG[size]["chosen"]["k"], DIAG[size]["chosen"]["s"])
                ax.text(b, a, f"{Z[a, b]:.0f}{'*' if ok else ''}", ha="center", va="center", fontsize=7,
                        color="w" if Z[a, b] < 60 else "k", fontweight="bold" if ch else "normal")
                if ch:
                    ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="red", lw=2))
        ax.set_xticks(range(len(C.K_GRID))); ax.set_xticklabels([f"{k:.1f}" for k in C.K_GRID], fontsize=7)
        ax.set_yticks(range(len(C.S_GRID))); ax.set_yticklabels(C.S_GRID)
        ax.set_xlabel("tightness k"); ax.set_ylabel("dispersion s")
        ax.set_title(f"{size} DEV-CAL: EDD+ECT tardy share (%), * pass, red = chosen", fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.8, label="% tardy jobs")
    fig.savefig(C.OUT / "heatmap_edd_tardy_share.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def scatter():
    for size in ("10x5", "20x10"):
        d = D.load("cal", size); L = d["labels"]; ch = DIAG[size]["chosen"]
        sel = D.idx(L, "tard", ch["s"])
        dd = ch["k"] * (1 + ch["s"] * (d["v"] - 0.5)) * d["W"]
        T = np.maximum(0, d["Cj"] - dd[:, None, :]).sum(-1)
        fig, axs = plt.subplots(2, 3, figsize=(13, 7))
        kind = np.array(["random" if L[i].startswith("rand") else ("EDD" if L[i].startswith("edd") else "rule") for i in sel])
        for c, i in enumerate((0, 1, 2)):
            for r, (y, lab) in enumerate(((d["c_new"], "carbon (kg CO2e, new)"), (T, "total tardiness"))):
                ax = axs[r, c]
                for kd, col in (("random", "0.6"), ("rule", "tab:blue"), ("EDD", "tab:red")):
                    m = kind == kd
                    ax.scatter(d["mk"][i, sel][m], y[i, sel][m], s=14, color=col, label=kd, alpha=0.8)
                ax.set_xlabel("makespan"); ax.set_ylabel(lab)
                ax.set_title(f"{size} DEV-CAL #{i + 1}", fontsize=9)
                if r == 0 and c == 0:
                    ax.legend(fontsize=7)
        fig.suptitle(f"{size}: makespan vs carbon (top) and vs total tardiness (bottom; k={ch['k']}, s={ch['s']}), 68 schedules per instance", fontsize=10)
        fig.tight_layout(); fig.savefig(C.OUT / f"scatter_examples_{size}.png", dpi=120); plt.close(fig)


def distributions():
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, size in zip(axs, ("10x5", "20x10")):
        d = D.load("cal", size); L = d["labels"]; ch = DIAG[size]["chosen"]
        dd = ch["k"] * (1 + ch["s"] * (d["v"] - 0.5)) * d["W"]
        T = np.maximum(0, d["Cj"] - dd[:, None, :]).sum(-1)
        for lab, cols, col in (("EDD/ECT", [L.index(f"edd:EDD/ECT@s={ch['s']}")], "tab:red"),
                               ("SPT/ECT", [L.index("rule:SPT/ECT")], "tab:blue"),
                               ("random (50 per instance)", [i for i, l in enumerate(L) if l.startswith("rand")], "0.5")):
            ax.hist(T[:, cols].ravel(), bins=40, alpha=0.5, density=True, color=col, label=lab)
        ax.set_xlabel("total tardiness"); ax.set_ylabel("density"); ax.legend(fontsize=8)
        ax.set_title(f"{size} DEV-CAL, chosen k={ch['k']}, s={ch['s']}", fontsize=9)
    fig.tight_layout(); fig.savefig(C.OUT / "tardiness_distributions.png", dpi=130); plt.close(fig)


def cpsat_table():
    f = C.OUT / "cpsat_subset.json"
    if not f.exists():
        return "(CP-SAT subset not run)"
    res = json.loads(f.read_text())
    L = ["| size | instance | min makespan: status / makespan / carbon | min tardiness: status / T / tardy share | min-energy assignment: status / makespan / carbon | TM_makespan (CP-SAT) | TM_carbon (CP-SAT) | best set makespan / best set T |",
         "|---|---|---|---|---|---|---|---|"]
    out = {}
    for size in ("10x5", "20x10"):
        d = D.load("cal", size); Ls = d["labels"]; ch = DIAG[size]["chosen"]
        sel_c = D.idx(Ls, "carbon_new"); sel_t = D.idx(Ls, "tard", ch["s"])
        dd = ch["k"] * (1 + ch["s"] * (d["v"] - 0.5)) * d["W"]
        T = np.maximum(0, d["Cj"] - dd[:, None, :]).sum(-1)
        tm1, tm2, st = [], [], {}
        for i in range(10):
            key = d["keys"][i]
            r = {p: res.get(f"{key}|{p}") for p in ("mk", "tard", "mineE")}
            if any(v is None for v in r.values()):
                continue
            for v in r.values():
                st[v["status"]] = st.get(v["status"], 0) + 1
            a = (r["mineE"]["mk"] - r["mk"]["mk"]) / r["mk"]["mk"]; b = (r["mk"]["c_new"] - r["mineE"]["c_new"]) / r["mineE"]["c_new"]
            tm1.append(a); tm2.append(b)
            L.append(f"| {size} | {Path(key).name} | {r['mk']['status']} / {r['mk']['mk']:.0f} / {r['mk']['c_new']:.1f} | "
                     f"{r['tard']['status']} / {r['tard']['tard']:.1f} / {100*r['tard']['tardy_share']:.0f}% | "
                     f"{r['mineE']['status']} / {r['mineE']['mk']:.0f} / {r['mineE']['c_new']:.1f} | {100*a:.1f}% | {100*b:.1f}% | "
                     f"{d['mk'][i, sel_t].min():.0f} / {T[i, sel_t].min():.1f} |")
        out[size] = dict(statuses=st, TM_makespan_median=float(np.median(tm1)) if tm1 else None,
                         TM_carbon_median=float(np.median(tm2)) if tm2 else None, n=len(tm1))
    L.append("\n" + "; ".join(f"{s}: statuses {v['statuses']}, median TM_makespan {100*v['TM_makespan_median']:.1f}%, "
                              f"median TM_carbon {100*v['TM_carbon_median']:.1f}% (n={v['n']})" for s, v in out.items() if v["n"]))
    txt = "\n".join(L)
    (C.OUT / "cpsat_table.md").write_text(txt + "\n")
    (C.OUT / "cpsat_summary.json").write_text(json.dumps(out, indent=1))
    return txt


def dd_manifest():
    man = C.load_manifest(); out = {}
    for split in ("cal", "check"):
        for size in ("10x5", "20x10"):
            ch = DIAG[size]["chosen"]
            for inst in C.instances(split, size):
                p = C.params_for(inst, man)
                out[inst.key] = {"sha256": man["instances"][inst.key]["sha256"], "k": ch["k"], "s": ch["s"],
                                 "W_j": [round(float(x), 6) for x in inst.work_content()],
                                 "k_j": [round(float(x), 6) for x in ch["k"] * (1 + ch["s"] * (p["v"] - 0.5))],
                                 "due_dates": [round(float(x), 6) for x in C.due_dates(inst, p["v"], ch["k"], ch["s"])]}
    f = C.DEV_ROOT / "manifest_objective_dev_duedates.json"
    f.write_text(json.dumps({"parent_manifest_sha256": C.file_sha256(C.MANIFEST), "formula":
                             "d_j = k_j * W_j, k_j = k*(1 + s*(v_j - 0.5)), W_j = sum of mean eligible processing times",
                             "release_times": "r_j = 0", "chosen": {s: DIAG[s]["chosen"] for s in ("10x5", "20x10")},
                             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "instances": out}))
    return f


def spec(ddf):
    cpass = DIAG["carbon_pass_both"]
    c10, c20 = DIAG["10x5"]["carbon_new"], DIAG["20x10"]["carbon_new"]
    ch = {s: DIAG[s]["chosen"] for s in ("10x5", "20x10")}
    y = f"""# PROPOSED problem specification v3 -- AWAITING APPROVAL. NOT wired into training, environment or evaluation.
# Source: results/objective_validation.md (pre-registered validation on new DEV instances).
status: PROPOSED
created: "{time.strftime('%Y-%m-%d')}"
instances:
  family: SD2 (daniel/data_utils.SD2_instance_generator)
  sizes: {{"10x5": {{n_jobs: 10, n_machines: 5, ops_per_job: 5}}, "20x10": {{n_jobs: 20, n_machines: 10, ops_per_job: 10}}}}
  processing_times: integer U{{1..99}} on each eligible machine; eligible machines per operation U{{1..n_machines}}
  release_times: 0 for every job
units:
  time_unit_assumption: "1 SD2 time unit = 1 minute"     # carbon scales linearly with this choice; rankings do not
objectives:
  makespan: max_j C_j
  total_tardiness:
    definition: "T = sum_j max(0, C_j - d_j)"
    due_dates: "d_j = k_j * W_j  (TWK); W_j = sum over the job's operations of the MEAN processing time over eligible machines"
    k_j: "k * (1 + s * (v_j - 0.5)),  v_j ~ U(0, 1) per job  (i.e. k_j ~ U[k(1 - s/2), k(1 + s/2)])"
    v_j_rng: "numpy default_rng(int(sha256('<instance key>|kdraw')[:16], 16))"
    chosen:
      "10x5": {{k: {ch['10x5']['k']}, s: {ch['10x5']['s']}}}      # pre-registered selection rule, confirmed on DEV-CHECK
      "20x10": {{k: {ch['20x10']['k']}, s: {ch['20x10']['s']}}}
    validation: PASS (both sizes; DEV-CHECK confirmation PASS)   # rule-based schedule set
    caveat: >-
      The pre-registered tightness checks use dispatching-rule schedules. On the CP-SAT subset (10 DEV-CAL instances
      per size, 60 s) the minimum total tardiness at the chosen due dates is 0 on 8/10 instances at 10x5 and on 1/10 at
      20x10 (several more below 50). For a strong scheduler the 10x5 due dates are loose; recalibrating against
      stronger schedules is recommended before freezing (results/objective_validation.md).
  carbon:
    definition: "C = gamma * sum over chosen (o, m) of P_m * p_om / 60   (processing energy only, no idle energy)"
    machine_power_kw: "P_m ~ U(4, 15) per machine; numpy default_rng(int(sha256('<instance key>|power')[:16], 16))"
    gamma_kg_co2e_per_kwh: {C.GAMMA}
    source: "CEA-FJSP, Wang et al., Frontiers in Environmental Science (2022)"
    validation: {'PASS' if cpass else 'FAIL'}   # pre-registered C1 >= 20% at both sizes; measured {100*c10['C1_share']:.1f}% (10x5), {100*c20['C1_share']:.1f}% (20x10)
    note: >-
      {'Passes the pre-registered check.' if cpass else 'FAILED the pre-registered operation-level check C1 (the fastest eligible machine is also the lowest-energy machine for about 84-86% of operations) although the schedule-level trade-off magnitudes pass. Per the pre-registration the carbon part stopped there; no alternative carbon model was tried. This objective needs a decision before the spec can be frozen.'}
dev_instances:
  seeds: {{DEV-CAL-10x5: 20261013, DEV-CAL-20x10: 20261014, DEV-CHECK-10x5: 20261015, DEV-CHECK-20x10: 20261016}}
  location: daniel/data/objective_dev/{{cal,check}}/SD2/<size>+mix/
  manifest: daniel/data/objective_dev/manifest_objective_dev.json
  manifest_sha256: {C.file_sha256(C.MANIFEST)}
  due_date_manifest: {ddf.relative_to(C.REPO).as_posix()}
  due_date_manifest_sha256: {C.file_sha256(ddf)}
  disjointness: results/objective_validation/disjointness.json (0 overlaps; data_final_test never opened)
next_steps_if_approved:
  - "freeze due dates for the training/validation/evaluation pools with the chosen (k, s) into a new hashed manifest (never the reserved final-test sets until approved)"
  - "environment changes listed in results/objective_validation.md (not implemented)"
"""
    f = C.REPO / "configs/problem_spec_v3_PROPOSED.yaml"
    f.write_text(y)
    return f


if __name__ == "__main__":
    heatmap(); scatter(); distributions()
    print(cpsat_table())
    print(spec(dd_manifest()))
