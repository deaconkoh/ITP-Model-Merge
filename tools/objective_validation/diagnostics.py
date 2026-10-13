"""Parts 3-5 of results/objective_validation.md: carbon diagnostics (new vs old), the TWK due-date calibration grid
with the pre-registered selection rule, the DEV-CHECK confirmation and the joint three-objective check.
Reads results/objective_validation/sets_<split>_<size>.npz; writes diagnostics.json and tables.md.

    python tools/objective_validation/diagnostics.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                       # noqa: E402


def load(split, size):
    d = dict(np.load(C.OUT / f"sets_{split}_{size}.npz"))
    d["labels"] = list(d["labels"])
    return d


def idx(labels, kind, s=None):
    if kind == "carbon_new":
        return [i for i, l in enumerate(labels) if (l.startswith("rule:") and not l.endswith("/OLDC")) or l.startswith("rand:")]
    if kind == "carbon_old":
        return [i for i, l in enumerate(labels) if (l.startswith("rule:") and not l.endswith("/MINE")) or l.startswith("rand:")]
    if kind == "tard":            # 15 rule + 3 EDD at s + 50 random = 68
        return [i for i, l in enumerate(labels) if (l.startswith("rule:") and not l.endswith("/OLDC"))
                or l.startswith("rand:") or (l.startswith("edd:") and l.endswith(f"@s={s}"))]
    if kind == "rules18":
        return [i for i, l in enumerate(labels) if (l.startswith("rule:") and not l.endswith("/OLDC"))
                or (l.startswith("edd:") and l.endswith(f"@s={s}"))]
    raise ValueError(kind)


def rho(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return stats.spearmanr(a, b).statistic


def nd2(a, b):
    """Number of distinct non-dominated points (minimisation) in 2-D."""
    pts = np.unique(np.stack([a, b], 1), axis=0)
    keep = [not np.any((pts[:, 0] <= x) & (pts[:, 1] <= y) & ((pts[:, 0] < x) | (pts[:, 1] < y))) for x, y in pts]
    return int(np.sum(keep))


def nd_share(X):
    """Share of schedules (rows of X, minimisation) not dominated by any other schedule."""
    le = (X[:, None, :] <= X[None, :, :]).all(-1); lt = (X[:, None, :] < X[None, :, :]).any(-1)
    dominated = (le & lt).any(0)
    return float(1 - dominated.mean()), int((~dominated).sum())


def med_iqr(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return dict(median=float(np.median(x)), q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)), n=int(len(x)))


# ------------------------------------------------------------------------------------------ carbon (Part 3)
def carbon_diag(d, which):
    insts = C.instances(*d["_split_size"])
    man = C.load_manifest()
    c1 = []
    for inst in insts:
        p = C.params_for(inst, man)
        cost = C.energy_carbon(inst, p["power"]) if which == "new" else p["oldcarbon"]
        n_diff = 0
        for o in range(inst.n_ops):
            ms = np.flatnonzero(inst.pt[o] > 0)
            fast = ms[np.lexsort((cost[o, ms], inst.pt[o, ms]))[0]]
            low = ms[np.lexsort((inst.pt[o, ms], cost[o, ms]))[0]]
            n_diff += int(fast != low)
        c1.append(n_diff / inst.n_ops)
    sel = idx(d["labels"], f"carbon_{which}")
    cc = d[f"c_{which}"][:, sel]; mk = d["mk"][:, sel]; wl = d["workload"][:, sel]
    r_mc, ndn, tm_mk, tm_c, r_cw = [], [], [], [], []
    for i in range(len(mk)):
        r_mc.append(rho(mk[i], cc[i])); ndn.append(nd2(mk[i], cc[i])); r_cw.append(rho(cc[i], wl[i]))
        cmin, mmin = cc[i].min(), mk[i].min()
        atc = np.isclose(cc[i], cmin, rtol=1e-12, atol=1e-9); atm = mk[i] == mmin
        tm_mk.append((mk[i][atc].min() - mmin) / mmin); tm_c.append((cc[i][atm].min() - cmin) / cmin)
    return dict(C1_share=float(np.mean(c1)), C1_per_instance=med_iqr(c1), spearman_mk_carbon=med_iqr(r_mc),
                nd2_mk_carbon=med_iqr(ndn), TM_makespan=med_iqr(tm_mk), TM_carbon=med_iqr(tm_c),
                spearman_carbon_workload=med_iqr(r_cw), n_schedules=len(sel))


# ------------------------------------------------------------------------------------------ tardiness (Part 4)
def tard_stats(d, k, s):
    L = d["labels"]; sel = idx(L, "tard", s); r18 = idx(L, "rules18", s)
    edd = L.index(f"edd:EDD/ECT@s={s}"); spt = L.index("rule:SPT/ECT"); mwkr = L.index("rule:MWKR/ECT")
    W, v, Cj, mk = d["W"], d["v"], d["Cj"], d["mk"]
    dd = k * (1 + s * (v - 0.5)) * W                         # [I, J]
    late = Cj > dd[:, None, :]                               # [I, S, J]
    T = np.maximum(0, Cj - dd[:, None, :]).sum(-1)           # [I, S]
    sumC = Cj.sum(-1)
    t1 = late[:, edd, :].mean(1)
    red = lambda X: np.where(T[:, X] > 0, (T[:, X] - T[:, edd]) / np.where(T[:, X] > 0, T[:, X], 1), 0.0)
    zero = (T[:, r18].min(1) == 0)
    alllate = late[:, r18, :].all(axis=(1, 2))
    cv = [np.std(T[i, sel]) / np.mean(T[i, sel]) if np.mean(T[i, sel]) > 0 else np.nan for i in range(len(T))]
    r_tm = [rho(T[i, sel], mk[i, sel]) for i in range(len(T))]
    r_tc = [rho(T[i, sel], sumC[i, sel]) for i in range(len(T))]
    nd = [nd2(mk[i, sel], T[i, sel]) for i in range(len(T))]
    n_pairs = n_diff = 0
    for i in range(len(T)):
        m_ = mk[i, sel]; t_ = T[i, sel]
        a, b = np.triu_indices(len(sel), 1)
        close = np.abs(m_[a] - m_[b]) <= 0.01 * np.minimum(m_[a], m_[b])
        big = np.maximum(t_[a], t_[b])
        diff = np.where(big > 0, np.abs(t_[a] - t_[b]) > 0.10 * big, False)
        n_pairs += int(close.sum()); n_diff += int((close & diff).sum())
    return dict(k=k, s=s, T1_edd_tardy_share=float(t1.mean()),
                T2_red_edd_vs_spt=float(np.median(red(spt))), T2_red_edd_vs_mwkr=float(np.median(red(mwkr))),
                T2_zero_share=float(zero.mean()), T2_all_late_share=float(alllate.mean()),
                T2_cv_median=float(np.nanmedian(cv)),
                T3_spearman_T_mk=float(np.nanmedian(r_tm)), T3_spearman_T_sumC=float(np.nanmedian(r_tc)),
                T3_close_mk_pairs=n_pairs, T3_close_pairs_T_differs_share=float(n_diff / n_pairs) if n_pairs else float("nan"),
                T3_nd2_mk_T=float(np.median(nd)), mean_total_tardiness=float(T[:, edd].mean()))


def passes(r):
    return (0.20 <= r["T1_edd_tardy_share"] <= 0.50 and r["T2_zero_share"] < 0.10 and r["T2_all_late_share"] < 0.10
            and r["T3_spearman_T_mk"] < 0.9)


# ------------------------------------------------------------------------------------------ joint (Part 5)
def joint(d, k, s):
    L = d["labels"]; sel = idx(L, "tard", s)
    dd = k * (1 + s * (d["v"] - 0.5)) * d["W"]
    T = np.maximum(0, d["Cj"] - dd[:, None, :]).sum(-1)[:, sel]
    obj = {"makespan": d["mk"][:, sel], "carbon": d["c_new"][:, sel], "tardiness": T,
           "sum_Cj": d["Cj"].sum(-1)[:, sel], "workload": d["workload"][:, sel]}
    names = list(obj)
    M = np.zeros((len(names), len(names)))
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            M[a, b] = np.nanmedian([rho(obj[na][i], obj[nb][i]) for i in range(len(T))])
    shares, counts = zip(*[nd_share(np.stack([obj["makespan"][i], obj["carbon"][i], obj["tardiness"][i]], 1))
                           for i in range(len(T))])
    trio = [abs(M[names.index(a), names.index(b)]) for a, b in (("makespan", "carbon"), ("makespan", "tardiness"),
                                                                 ("carbon", "tardiness"))]
    return dict(names=names, spearman_median=M.tolist(), nd3_share=med_iqr(shares), nd3_count=med_iqr(counts),
                distinct=bool(max(trio) < 0.9 and np.median(counts) >= 3))


def fmt(x, p=3):
    return f"{x:.{p}f}" if isinstance(x, float) else str(x)


def main():
    out, L = {}, []
    P_ = L.append
    for size in ("10x5", "20x10"):
        d = load("cal", size); d["_split_size"] = ("cal", size)
        out[size] = {"carbon_new": carbon_diag(d, "new"), "carbon_old": carbon_diag(d, "old")}
        grid = [tard_stats(d, k, s) for s in C.S_GRID for k in C.K_GRID]
        out[size]["grid"] = grid
        ok = [r for r in grid if passes(r)]
        out[size]["passing"] = [(r["k"], r["s"]) for r in ok]
        if ok:
            best = min(ok, key=lambda r: (abs(r["T1_edd_tardy_share"] - 0.35), -r["T2_red_edd_vs_spt"]))
            out[size]["chosen"] = {"k": best["k"], "s": best["s"]}
            chk = load("check", size); chk["_split_size"] = ("check", size)
            conf = tard_stats(chk, best["k"], best["s"])
            out[size]["check_confirmation"] = dict(stats=conf, passes=passes(conf))
            out[size]["check_carbon_new"] = carbon_diag(chk, "new")
            out[size]["joint_cal"] = joint(d, best["k"], best["s"])
            out[size]["joint_check"] = joint(chk, best["k"], best["s"])
        else:
            out[size]["chosen"] = None
            out[size]["closest"] = sorted(grid, key=lambda r: abs(r["T1_edd_tardy_share"] - 0.35))[:5]
        cn, co = out[size]["carbon_new"], out[size]["carbon_old"]
        out[size]["carbon_pass"] = bool(cn["C1_share"] >= 0.20 and cn["TM_makespan"]["median"] >= 0.05
                                        and cn["TM_carbon"]["median"] >= 0.05)
    out["carbon_pass_both"] = all(out[s]["carbon_pass"] for s in ("10x5", "20x10"))
    (C.OUT / "diagnostics.json").write_text(json.dumps(out, indent=1, default=float))

    # ------------------------------------------------------------------ tables
    P_("### Carbon diagnostics on DEV-CAL (Part 3): new energy-based carbon vs old i.i.d. carbon\n")
    P_("| size | carbon | C1 (fastest != lowest-carbon machine) | Spearman(makespan, carbon) median [IQR] | 2-D non-dominated set (median) | TM_makespan median [IQR] | TM_carbon median [IQR] | Spearman(carbon, workload) median |")
    P_("|---|---|---|---|---|---|---|---|")
    for size in ("10x5", "20x10"):
        for w in ("new", "old"):
            c = out[size][f"carbon_{w}"]
            P_(f"| {size} | {'NEW P_m x p' if w == 'new' else 'old i.i.d. 1-100'} | {100*c['C1_share']:.1f}% | "
               f"{c['spearman_mk_carbon']['median']:+.3f} [{c['spearman_mk_carbon']['q1']:+.2f}, {c['spearman_mk_carbon']['q3']:+.2f}] | "
               f"{c['nd2_mk_carbon']['median']:.0f} | {100*c['TM_makespan']['median']:.1f}% [{100*c['TM_makespan']['q1']:.1f}, {100*c['TM_makespan']['q3']:.1f}] | "
               f"{100*c['TM_carbon']['median']:.1f}% [{100*c['TM_carbon']['q1']:.1f}, {100*c['TM_carbon']['q3']:.1f}] | "
               f"{c['spearman_carbon_workload']['median']:+.3f} |")
    P_(f"\n**Carbon verdict (pre-registered rule, NEW carbon): {'PASS' if out['carbon_pass_both'] else 'FAIL'}** "
       f"(10x5 {'pass' if out['10x5']['carbon_pass'] else 'fail'}, 20x10 {'pass' if out['20x10']['carbon_pass'] else 'fail'}).\n")
    for size in ("10x5", "20x10"):
        P_(f"### Tardiness calibration grid, {size} (DEV-CAL; EDD+ECT tardy share; * = passes all pre-registered conditions)\n")
        P_("| s \\ k | " + " | ".join(f"{k:.1f}" for k in C.K_GRID) + " |")
        P_("|---|" + "---|" * len(C.K_GRID))
        g = {(r["k"], r["s"]): r for r in out[size]["grid"]}
        for s in C.S_GRID:
            P_(f"| {s} | " + " | ".join(f"{100*g[(k, s)]['T1_edd_tardy_share']:.0f}%{'*' if passes(g[(k, s)]) else ''}" for k in C.K_GRID) + " |")
        P_(f"\nPassing configurations ({len(out[size]['passing'])}): " + ", ".join(f"(k={k}, s={s})" for k, s in out[size]["passing"]) + "\n")
        P_("| config | T1 tardy share | EDD vs SPT | EDD vs MWKR | zero-T share | all-late share | CV(T) | Spearman(T, mk) | Spearman(T, sum C) | close-mk pairs with T diff > 10% | ND(mk, T) | mean T (EDD) |")
        P_("|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = sorted([r for r in out[size]["grid"] if passes(r)], key=lambda r: abs(r["T1_edd_tardy_share"] - 0.35))[:8]
        if out[size]["chosen"]:
            rows = [r for r in rows]
        for r in rows:
            mark = " (CHOSEN)" if out[size]["chosen"] and (r["k"], r["s"]) == (out[size]["chosen"]["k"], out[size]["chosen"]["s"]) else ""
            P_(f"| k={r['k']}, s={r['s']}{mark} | {100*r['T1_edd_tardy_share']:.1f}% | {100*r['T2_red_edd_vs_spt']:.1f}% | {100*r['T2_red_edd_vs_mwkr']:.1f}% | "
               f"{100*r['T2_zero_share']:.0f}% | {100*r['T2_all_late_share']:.0f}% | {r['T2_cv_median']:.2f} | {r['T3_spearman_T_mk']:+.3f} | "
               f"{r['T3_spearman_T_sumC']:+.3f} | {100*r['T3_close_pairs_T_differs_share']:.1f}% (of {r['T3_close_mk_pairs']}) | {r['T3_nd2_mk_T']:.0f} | {r['mean_total_tardiness']:.1f} |")
        if out[size]["chosen"]:
            c = out[size]["check_confirmation"]["stats"]
            P_(f"\n**Chosen {size}: k = {out[size]['chosen']['k']}, s = {out[size]['chosen']['s']}.** DEV-CHECK confirmation: "
               f"T1 {100*c['T1_edd_tardy_share']:.1f}%, zero-T {100*c['T2_zero_share']:.0f}%, all-late {100*c['T2_all_late_share']:.0f}%, "
               f"Spearman(T, mk) {c['T3_spearman_T_mk']:+.3f}, Spearman(T, sum C) {c['T3_spearman_T_sumC']:+.3f}, EDD vs SPT "
               f"{100*c['T2_red_edd_vs_spt']:.1f}% -> **{'PASS' if out[size]['check_confirmation']['passes'] else 'FAIL'}**\n")
        else:
            P_(f"\n**No configuration passes at {size}.** Closest: " + "; ".join(
                f"(k={r['k']}, s={r['s']}) T1 {100*r['T1_edd_tardy_share']:.0f}%" for r in out[size]["closest"]) + "\n")
    P_("### Joint three-objective check (Part 5; chosen carbon and due dates; median per-instance Spearman over 68 schedules)\n")
    for size in ("10x5", "20x10"):
        for part in ("joint_cal", "joint_check"):
            if part not in out[size]:
                continue
            j = out[size][part]
            P_(f"**{size}, {'DEV-CAL' if part == 'joint_cal' else 'DEV-CHECK'}**\n")
            P_("| | " + " | ".join(j["names"]) + " |"); P_("|---|" + "---|" * len(j["names"]))
            for a, na in enumerate(j["names"]):
                P_(f"| {na} | " + " | ".join(f"{j['spearman_median'][a][b]:+.2f}" for b in range(len(j['names']))) + " |")
            P_(f"\n3-D non-dominated set (makespan, carbon, tardiness): median {j['nd3_count']['median']:.0f} schedules "
               f"[IQR {j['nd3_count']['q1']:.0f}-{j['nd3_count']['q3']:.0f}] = {100*j['nd3_share']['median']:.1f}% of the set; "
               f"**three distinct objectives: {'YES' if j['distinct'] else 'NO'}**\n")
    (C.OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
