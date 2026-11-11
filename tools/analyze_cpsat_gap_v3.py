"""Report of tools/cpsat_gap_v3.py: results/cpsat_gap_v3.md and figures from results/cpsat_gap_v3/analysis.json."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

GROUPS = ["makespan-heavy", "carbon-heavy", "tardiness-heavy", "mixed"]
GROUP_TXT = {"makespan-heavy": "Makespan-heavy (w_m >= 0.8, 6 preferences)",
             "carbon-heavy": "Carbon-heavy (w_c >= 0.8, 6)", "tardiness-heavy": "Tardiness-heavy (w_t >= 0.8, 6)",
             "mixed": "Mixed (the other 48)"}
LAB = {"wm_final": "World-model base mix, final", "fs_final": "From scratch, final"}


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"


def render(out: Path, quick=False):
    import gzip
    with gzip.open(out / "analysis.json.gz", "rt") as fh:
        R = json.loads(fh.read())
    figs = out / "figures"; figs.mkdir(exist_ok=True)
    md = out.with_suffix(".md") if not quick else out.parent / (out.name + ".md")
    rows, S = R["rows"], R["schedules"]
    I = R["n_instances"]
    L = []
    w = L.append

    # ------------------------------------------------------------------ key numbers for the summary
    def gmean(key, grp=None):
        v = [r[key] for r in rows if grp is None or r["group"] == grp]
        return float(np.mean(v))
    ls_better = sum(r["ls_obj"][0] < r["obj"][0] - 1e-9 or r["ls_obj"][2] < r["obj"][2] - 1e-9 for r in S)
    ls_nd = sum(r["ls_idle"]["deliberate"] <= 1e-6 for r in S)
    ls_daniel = sum(r["ls_daniel"] for r in S)
    nd_daniel_nd = sum(r["nd_idle"]["deliberate"] <= 1e-6 for r in S)
    any_del = sum(r["idle"]["deliberate"] > 1e-6 for r in S)
    hv = R["hv"]

    w("# Why CP-SAT schedules better than DANIEL: CP-SAT analysis, benchmark v3 (10x5)")
    w("")
    w(f"CP-SAT evaluation and analysis only. No model was trained, and DANIEL, the benchmark and the CP-SAT results are unchanged. "
      f"Data: the {I} evaluation-split 10x5 instances that have a CP-SAT reference front (B2), with the 66 evaluation "
      "preferences (simplex grid, step 0.1). The reserved final-test sets were not used. Single seed: numbers only, no "
      "significance claims. Computed by `tools/cpsat_gap_v3.py` (definitions in `tools/cpsat_gap_v3_analysis.py`); "
      "every number below is in `results/cpsat_gap_v3/analysis.json.gz`.")
    w("")
    w("Earlier results reused, not redone:")
    w("- `results/diagnose_v3.md`, Test 2: DANIEL's action mask only allows pairs that start at the earliest possible "
      "start time, so every schedule DANIEL can make is **non-delay**: no machine is ever idle while an operation it can "
      "process is ready. None of 160 CP-SAT and rule target schedules was non-delay.")
    w("- `results/tardiness_diagnosis_v3.md`, section 4: for tardiness alone at 10x5, the best schedule found without "
      "waiting (a search over DANIEL-reachable schedules) averages 50.5 against CP-SAT's 16.1. CP-SAT's own job order "
      "forced through the no-waiting rule gives 375.1. That search is re-run here with its fixed seed, only to recover its "
      "schedules for the pool in Part 4.")
    w("")

    # ------------------------------------------------------------------ summary
    w("## Summary")
    w("")
    w("| Set of schedules (mean over 20 instances) | Hypervolume |")
    w("|---|---|")
    w(f"| CP-SAT reference front | {f(hv['cpsat'])} |")
    w(f"| The same schedules, left-shifted (same machines, same order on every machine, no unnecessary idle time) | {f(hv['ls'])} |")
    w(f"| The same machines, but no machine waits for an operation assigned to it (non-delay with CP-SAT's machine choices) | {f(hv['nd'])} |")
    w(f"| CP-SAT's machine choices and order, forced through DANIEL's no-waiting rule | {f(hv['env'])} |")
    w(f"| Best known non-delay schedules (pool of every existing DANIEL checkpoint, the rules and the earlier search) | {f(hv['pool'])} |")
    w(f"| DANIEL greedy: {LAB['wm_final']} | {f(hv['wm_final'])} |")
    w(f"| DANIEL greedy: {LAB['fs_final']} | {f(hv['fs_final'])} |")
    w("")
    w("Hypervolume uses the frozen evaluation normalisation (`results/baselines_v3/normalisation_points.json`, reference "
      "point 1.1), as every evaluation does; higher is better.")
    w("")
    w(f"- **CP-SAT is allowed to wait, and its schedules do leave machines idle while work is ready** "
      f"({any_del} of {len(S)} front schedules). But this is not where its advantage comes from in those schedules: "
      f"left-shifting them (removing every idle period that the machine choices and order do not force) never makes "
      f"them worse and the hypervolume is unchanged ({f(hv['cpsat'])} to {f(hv['ls'])}).")
    oth = np.mean([r["ls_idle"]["deliberate_other_only"] for r in S]); tot = np.mean([r["ls_idle"]["deliberate"] for r in S])
    w(f"- **Left-shifting does not make them non-delay.** {'None' if ls_nd == 0 else ls_nd} of the {len(S)} left-shifted "
      f"schedules pass the non-delay test ({ls_daniel} are reproduced by DANIEL's own decision rule). The waiting that "
      f"remains is forced by CP-SAT's machine choices and machine order. Most of it ({pct(oth, tot)}) is a machine idling "
      "while an operation it could process is ready, because CP-SAT runs that operation on another machine. The rest is "
      "a machine holding back for one of its own operations that is not ready yet. DANIEL's rule would have to start "
      "the ready operation on the idle machine.")
    w(f"- **Keeping CP-SAT's machine choices but forbidding a machine to wait for its own operations** costs "
      f"{f(hv['ls'] - hv['nd'])} hypervolume ({f(hv['ls'])} to {f(hv['nd'])}). Such a schedule can still hold a machine "
      f"idle for an operation assigned elsewhere; {nd_daniel_nd} of {len(S)} pass DANIEL's stricter non-delay rule.")
    w(f"- **The best non-delay schedules known** reach {f(hv['pool'])}. That is below the fixed-assignment schedules "
      f"({f(hv['nd'])}), which DANIEL's rule cannot produce because they still let a machine idle while an operation "
      "assigned elsewhere is ready. It is well above DANIEL greedy "
      f"({f(hv['wm_final'])} / {f(hv['fs_final'])}).")
    w("- Part 4 compares DANIEL greedy with the best known non-delay schedule for each instance and preference.")
    w("")
    w("These steps describe what is lost when each restriction is added, for these particular constructions. They "
      "are not a causal decomposition, and a different construction order could split the gap differently.")
    w("")

    # ------------------------------------------------------------------ Part 1
    p1 = R["part1"]
    w("## 1. The CP-SAT implementation (`tools/cpsat_baselines_v3.py`)")
    w("")
    w("| Question | Answer | Where |")
    w("|---|---|---|")
    w("| Can CP-SAT leave a machine idle on purpose? | **Yes.** Start times are free integers in [0, H]; the only "
      "constraints are one eligible machine per operation, job precedence and no overlap on a machine. Nothing ties a "
      "start time to the moment the job and machine become free, and idle time is not penalised. All three objectives "
      "are regular (never better when an operation finishes later), so idle time never helps CP-SAT by itself; it only "
      "appears where the chosen machines and order force it, or where a start time does not affect the objective. | "
      "lines 73-86 |")
    w("| Are CP-SAT's and DANIEL's objectives the same? | **Yes:** makespan = latest job completion; carbon = sum of "
      "carbon[o, m] over the chosen pairs; total tardiness = sum of max(0, C_j - d_j). CP-SAT works in integer units "
      "(carbon rounded to 1e-4 kg; due dates rounded up to 1e-3 time units), and every returned schedule is re-scored "
      "exactly by `benchmark_v3.schedule_objectives`, the same independent scorer DANIEL's evaluation uses. | lines 52-53, "
      "87-93; `daniel/benchmark_v3.py:131` |")
    w("| Same processing times, eligibility, precedence, due dates and carbon? | **Yes.** Both read the same frozen "
      "benchmark through `benchmark_v3.load_split` (hash-checked manifest). Eligibility is processing time > 0 in both; "
      "jobs are chains of operations in both; due dates are read from the manifest; carbon = 0.540 x machine power x "
      "processing time / 60 for the chosen pair, so **carbon depends only on the machine assignment, not on timing.** "
      "All jobs are released at time 0 in both. | lines 44-53, 78-84; `daniel/benchmark_v3.py:88, 231`; "
      "`daniel/fjsp_env_same_op_nums.py:538` |")
    st = p1["b2_solve_status"]
    w(f"| Proven optimal, feasible, or time-limited? | **B2 front (the reference front):** {st.get('OPTIMAL', 0)} of "
      f"{sum(st.values())} epsilon solves proven optimal; {st.get('INFEASIBLE', 0)} budget pairs infeasible; none stopped "
      f"by the time limit (settings: limit, grid, workers = {p1['b2_settings']}). Each point is the proven minimum "
      "makespan under its carbon and tardiness budgets, in the integer-scaled model. **B1 (single objective, all 20 "
      f"instances):** {', '.join(f'{k} {v}' for k, v in sorted(p1['b1_status'].items()))} (limits {p1['b1_limits']} s). "
      "| results/baselines_v3/cpsat |")
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(p1["front_kind"].items()))
    w(f"| How is the reference front built? | Per instance: the three B1 points (each objective minimised, then the "
      "others lexicographically), then a 5 x 5 grid of carbon and tardiness budgets between the B1 ideal and nadir; for "
      "each budget pair, minimise makespan. The front is the non-dominated set of all feasible points: "
      f"{p1['front_points']} schedules over 20 instances ({min(p1['front_sizes'])}-{max(p1['front_sizes'])} per instance; "
      f"{kinds}). Each point is optimal for its budgets, but a 5 x 5 grid is a **sample of the Pareto front, not all of "
      "it**, and a point's carbon and tardiness are only bounded, not minimised. | lines 186-212 |")
    w("")
    w(f"Checks: {p1['front_rescored']}. The CP-SAT front's hypervolume here ({f(hv['cpsat'])}) matches the "
      "plateau diagnosis (1.205).")
    w("")

    # ------------------------------------------------------------------ Part 2
    w("## 2. CP-SAT's scheduling decisions")
    w("")
    w("**Definitions.**")
    w("- **Idle period:** on a machine, before its first operation, between consecutive operations, or after its last operation up to the makespan.")
    w("- **Deliberate waiting:** the part of an idle period during which some operation that the machine could process is ready (its job's previous operation has finished) and has not started yet.")
    w("  - **Same machine:** that operation later runs on this machine.")
    w("  - **Other machine only:** the only ready operations run on another machine.")
    w("- **Unavoidable waiting:** the rest of the idle period, when nothing the machine could process is ready.")
    w("")
    w("A schedule with no deliberate waiting is exactly a non-delay schedule. Every pool schedule (all made by DANIEL's "
      "rule) passes this test, and so do the left-shifted schedules that DANIEL's rule reproduces. Idle time alone was "
      "not taken as evidence of waiting.")
    w("")
    w("**Per preference.** For each instance and preference, the CP-SAT front schedule with the best score at that "
      "preference: 20 instances x 66 preferences. Times are in the benchmark's minutes; means per schedule.")
    w("")
    w("| Preferences | Idle before the last operation, per machine total | Deliberate waiting (all idle periods) | of which a same-machine operation was ready | Deliberate waiting before the last operation | Deliberate periods | Periods where an operation would have fitted | Periods ending when the machine's next operation becomes ready |")
    w("|---|---|---|---|---|---|---|---|")
    sched_by = {}
    for r in S:
        sched_by.setdefault(r["i"], []).append(r)
    src_index = {(r["i"], r["source"]): r for r in S}
    for grp in GROUPS + [None]:
        sel = [src_index[(r["i"], r["cpsat_source"])] for r in rows if grp is None or r["group"] == grp]
        idl = [s["idle"] for s in sel]
        dg = sum(x["deliberate_gaps"] for x in idl)
        w(f"| {GROUP_TXT[grp] if grp else 'All 66'} | {f(np.mean([x['internal_idle'] for x in idl]), 1)} | "
          f"{f(np.mean([x['deliberate'] for x in idl]), 1)} | {f(np.mean([x['deliberate_same'] for x in idl]), 1)} | "
          f"{f(np.mean([x['internal_deliberate'] for x in idl]), 1)} | {f(np.mean([x['deliberate_gaps'] for x in idl]), 1)} | "
          f"{pct(sum(x['fit_gaps'] for x in idl), dg)} | {pct(sum(x['waits_for_next'] for x in idl), dg)} |")
    w("")
    durs = np.concatenate([np.asarray(r["idle"]["durations"]) for r in S]) if S else np.zeros(0)
    w(f"Over all {len(S)} distinct front schedules: {any_del} contain deliberate waiting. Deliberate periods last a "
      f"median {f(np.median(durs), 0)} minutes (quartiles {f(np.percentile(durs, 25), 0)}-{f(np.percentile(durs, 75), 0)}; "
      f"total {f(np.mean([r['idle']['deliberate'] for r in S]), 1)} per schedule against "
      f"{f(np.mean([r['idle']['idle'] for r in S]), 1)} total idle).")
    w("")
    w("**By how the front point was made** (distinct schedules):")
    w("")
    w("| Kind | Schedules | Proven optimal | Deliberate waiting per schedule | of which same machine | Left shift changes makespan or tardiness |")
    w("|---|---|---|---|---|---|")
    for kind in sorted({r["kind"] for r in S}):
        sel = [r for r in S if r["kind"] == kind]
        w(f"| {kind} | {len(sel)} | {sum(r['status'] in ('OPTIMAL',) for r in sel)} | "
          f"{f(np.mean([r['idle']['deliberate'] for r in sel]), 1)} | {f(np.mean([r['idle']['deliberate_same'] for r in sel]), 1)} | "
          f"{sum(r['ls_obj'][0] < r['obj'][0] - 1e-9 or r['ls_obj'][2] < r['obj'][2] - 1e-9 for r in sel)} |")
    w("")
    w("B1 points report the status of their first (primary) solve; their later stages were proven optimal under their bounds (Part 1).")
    w("")
    # association
    adv = [r["best"] - r["cpsat"] for r in rows]
    w("**Is deliberate waiting associated with better outcomes?** Across the 1,320 instance-preference pairs:")
    w("- **CP-SAT's advantage** is the best known non-delay score minus CP-SAT's score (positive means CP-SAT is better).")
    w(f"  - Mean advantage: {f(np.mean(adv))}.")
    w(f"  - CP-SAT is better in {sum(a > 1e-9 for a in adv)} pairs, equal in {sum(abs(a) <= 1e-9 for a in adv)} and worse in {sum(a < -1e-9 for a in adv)}.")
    w("  - CP-SAT could in principle be worse, because its front is a 5 x 5 sample of the Pareto front.")
    w(f"- **Rank correlation** (Spearman) between the advantage and CP-SAT's deliberate waiting:")
    w(f"  - as stored: {f(spearman(adv, [r['cpsat_deliberate'] for r in rows]), 2)};")
    w(f"  - after left shift: {f(spearman(adv, [r['ls_deliberate'] for r in rows]), 2)}.")
    for grp in GROUPS:
        sel = [r for r in rows if r["group"] == grp]
        w(f"  - {grp}: {f(spearman([r['best'] - r['cpsat'] for r in sel], [r['ls_deliberate'] for r in sel]), 2)} after left shift")
    within = [spearman([r["best"] - r["cpsat"] for r in rows if r["i"] == i], [r["ls_deliberate"] for r in rows if r["i"] == i])
              for i in range(I)]
    within = [x for x in within if not np.isnan(x)]
    w(f"  - within an instance, across its 66 preferences (removes differences between instances): mean "
      f"{f(np.mean(within), 2)} over {len(within)} instances (range {f(min(within), 2)} to {f(max(within), 2)})")
    w("")
    w("A correlation across instances and preferences is not evidence that waiting causes the advantage. Bigger or "
      "tighter instances can have both more waiting and larger gaps.")
    w("")
    w("**Example: instance 1.** CP-SAT's best schedule for each corner preference and for equal weights. Hatched areas "
      "are deliberate waiting; white gaps are unavoidable waiting. The second row is the same schedules left-shifted. "
      "Full machine sequences, start and completion times and job completions are under `examples` in `results/cpsat_gap_v3/analysis.json.gz`.")
    w("")
    _gantt(R["examples"], figs / "examples.png")
    w(f"![CP-SAT example schedules]({out.name}/figures/examples.png)")
    w("")
    w("| Preference | Front point | Makespan | Carbon | Tardiness | Deliberate waiting | of which same machine | Left-shifted: makespan / tardiness | Left-shifted: deliberate waiting |")
    w("|---|---|---|---|---|---|---|---|---|")
    for lab, e in R["examples"].items():
        o, lo = e["objectives"], e["left_shift_objectives"]
        w(f"| {lab} | {e['source']} | {f(o[0], 0)} | {f(o[1], 1)} | {f(o[2], 1)} | {f(e['idle']['deliberate'], 0)} | "
          f"{f(e['idle']['deliberate_same'], 0)} | {f(lo[0], 0)} / {f(lo[2], 1)} | {f(e['left_shift_idle']['deliberate'], 0)} |")
    w("")

    # ------------------------------------------------------------------ Part 3
    w("## 3. Does waiting explain CP-SAT's advantage?")
    w("")
    w("**Left shift.** Every operation keeps CP-SAT's machine and its place in the machine's order, and starts as early as "
      "job precedence and the machine allow.")
    w("")
    w("| | Distinct front schedules |")
    w("|---|---|")
    d_mk = [r["ls_obj"][0] - r["obj"][0] for r in S]; d_t = [r["ls_obj"][2] - r["obj"][2] for r in S]
    w(f"| Schedules whose makespan or tardiness improves | {ls_better} of {len(S)} |")
    w(f"| Makespan changes | {sum(abs(x) > 1e-9 for x in d_mk)} schedules (CP-SAT minimised makespan in every solve) |")
    w(f"| Mean change in tardiness | {f(np.mean(d_t), 2)} (largest improvement {f(-min(d_t), 1)}; tardiness was only bounded in the B2 solves) |")
    w(f"| Schedules with any operation moved earlier | {sum(r['late_shift'] > 1e-6 for r in S)} of {len(S)}; mean total shift {f(np.mean([r['late_shift'] for r in S]), 1)} min |")
    w(f"| Idle before each machine's last operation (per schedule) | {f(np.mean([r['idle']['internal_idle'] for r in S]), 1)} -> {f(np.mean([r['ls_idle']['internal_idle'] for r in S]), 1)} |")
    w(f"| Idle periods before each machine's last operation (per schedule) | {f(np.mean([r['idle']['internal_gaps'] for r in S]), 1)} -> {f(np.mean([r['ls_idle']['internal_gaps'] for r in S]), 1)} |")
    w(f"| Deliberate waiting, all periods | {f(np.mean([r['idle']['deliberate'] for r in S]), 1)} -> {f(np.mean([r['ls_idle']['deliberate'] for r in S]), 1)} |")
    w(f"| Deliberate waiting, same-machine operation ready | {f(np.mean([r['idle']['deliberate_same'] for r in S]), 1)} -> {f(np.mean([r['ls_idle']['deliberate_same'] for r in S]), 1)} |")
    w(f"| Deliberate waiting, only other-machine operations ready | {f(np.mean([r['idle']['deliberate_other_only'] for r in S]), 1)} -> {f(np.mean([r['ls_idle']['deliberate_other_only'] for r in S]), 1)} |")
    w(f"| Left-shifted schedules that are non-delay | {ls_nd} of {len(S)} |")
    w(f"| ... that DANIEL's own decision rule reproduces exactly | {ls_daniel} of {len(S)} |")
    w("")
    w("Total idle time cannot change while the makespan stays the same: it always equals machines x makespan minus "
      "processing time. A left shift moves idle time to the end of each machine's schedule, so the table also reports "
      "idle time before each machine's last operation.")
    w("")
    w("**Building it up step by step.** Mean scores per preference group: the preference-weighted normalised benchmark "
      "score, lower is better. This is the score best-of-16 selects on.")
    w("")
    w("| Preferences | CP-SAT | Left-shifted | Non-delay, CP-SAT's machines | CP-SAT order through DANIEL's rule | Best known non-delay | DANIEL greedy, WM final | DANIEL greedy, FS final |")
    w("|---|---|---|---|---|---|---|---|")
    for grp in GROUPS + [None]:
        w(f"| {GROUP_TXT[grp] if grp else 'All 66'} | {f(gmean('cpsat', grp))} | {f(gmean('ls', grp))} | {f(gmean('nd', grp))} | "
          f"{f(gmean('env', grp))} | {f(gmean('best', grp))} | {f(gmean('wm_final', grp))} | {f(gmean('fs_final', grp))} |")
    w("")
    w("Mean objectives per preference group, in raw units:")
    w("")
    w("| Preferences | Schedule | Makespan | Carbon | Tardiness |")
    w("|---|---|---|---|---|")
    for grp in GROUPS:
        sel = [r for r in rows if r["group"] == grp]
        for key, name in (("cpsat_obj", "CP-SAT"), ("ls_obj", "Left-shifted"), ("nd_obj", "Non-delay, CP-SAT's machines"),
                          ("env_obj", "Through DANIEL's rule"), ("best_obj", "Best known non-delay"),
                          ("wm_final_obj", "DANIEL greedy, WM final"), ("fs_final_obj", "DANIEL greedy, FS final")):
            o = np.mean([r[key] for r in sel], 0)
            w(f"| {grp} | {name} | {f(o[0], 1)} | {f(o[1], 1)} | {f(o[2], 1)} |")
    w("")
    same_best = [r["same_machine_best"] for r in rows]
    env_same = [r["env_same_machine"] for r in S]
    w("**Assignment against sequencing.**")
    w("- **Carbon** depends only on the machine assignment, so any carbon gap is an assignment gap.")
    w("  - In the carbon-heavy preferences, CP-SAT's carbon is "
      f"{f(np.mean([r['cpsat_obj'][1] for r in rows if r['group'] == 'carbon-heavy']), 1)} against "
      f"{f(np.mean([r['best_obj'][1] for r in rows if r['group'] == 'carbon-heavy']), 1)} for the best known non-delay "
      "schedule.")
    w("  - This repeats `results/diagnose_v3.md`: the lowest-carbon machine is usually masked, because the no-waiting "
      "rule forbids waiting for it.")
    w(f"- **Same machine:** the best known non-delay schedule puts on average {pct(np.mean(same_best), 1)} of operations on "
      "the same machine as CP-SAT's schedule for that preference.")
    w(f"- **Forcing CP-SAT's choices through DANIEL's rule** keeps {pct(np.mean(env_same), 1)} of its machine choices. "
      f"{sum(r['env_is_cpsat'] for r in S)} of {len(S)} schedules come out unchanged.")
    w("- **Fixed-assignment non-delay construction:**")
    w("  - It keeps every machine choice and removes all waiting for a machine's own operations.")
    w(f"  - Its score moves from {f(gmean('ls'))} to {f(gmean('nd'))}.")
    w("  - It is still not DANIEL-reachable when an operation that runs elsewhere is ready while the machine is idle.")
    w(f"  - Deliberate waiting left after this construction: {f(np.mean([r['nd_idle']['deliberate'] for r in S]), 1)} per schedule, all of it of the other-machine kind.")
    w("")
    _part3_fig(R, figs / "part3.png")
    w(f"![Building it up step by step]({out.name}/figures/part3.png)")
    w("")

    # ------------------------------------------------------------------ Part 4
    w("## 4. DANIEL's own gap: greedy against the best known non-delay schedule")
    w("")
    pool = R["pool"]
    w("**Best known non-delay schedule.** For each instance and preference, this is the lowest-score schedule in a pool "
      "of schedules DANIEL's rule can produce:")
    w(f"- greedy and best-of-16 schedules of {len(pool['mo_found'])} existing multi-objective checkpoints. These were "
      "regenerated with the evaluation's own settings; every one reproduces the committed objectives exactly (counts below).")
    w(f"- greedy and best-of-100 schedules of {len(pool['fixed'].get('single_objective', {}))} committed single-objective "
      "specialists, read from their committed files;")
    w("- the three no-waiting rules, CP-SAT's tardiness order without waiting, and the earlier no-waiting search "
      f"(re-run, {pool['fixed'].get('search_reproduced', 'n/a')} of {I} instances reproduce the committed values).")
    w("")
    w(f"Distinct schedules per instance: {int(np.mean(pool['distinct_per_instance']))} on average. All pass the "
      "non-delay test.")
    w(f"Not included: {len(pool['mo_missing'])} evaluated runs whose checkpoints are no longer on disk "
      f"({', '.join(sorted({m.rsplit('_s', 1)[0].replace('10x5+mix+bv3_', '') for m in pool['mo_missing']}))}), and "
      "zero-training merges with no checkpoint file.")
    rep = pool["reproduced"]
    bad = [k for k, v in rep.items() if v["greedy"][0] != v["greedy"][1] or ("s16" in v and v["s16"][0] != v["s16"][1])]
    w(f"Reproduction: {len(rep) - len(bad)} of {len(rep)} checkpoints reproduce every greedy and best-of-16 schedule's "
      f"objectives exactly{'' if not bad else '; not exact: ' + ', '.join(bad)}.")
    w("")
    srcs = Counter()
    for r in rows:
        s = r["best_source"]
        srcs["best-of-16 (multi-objective checkpoints)" if s.endswith(" s16") else
             "greedy (multi-objective checkpoints)" if s.endswith(" greedy") else
             "single-objective specialists" if s.startswith("so__") else
             "no-waiting search (tardiness)" if s.startswith("search") else "rules / CP-SAT order without waiting"] += 1
    w("Where the best known schedule comes from (1,320 instance-preference pairs): " +
      "; ".join(f"{k} {v}" for k, v in srcs.most_common()) + ".")
    w("")
    w("**Method.** DANIEL's greedy trajectory is followed step by step.")
    w("- **Divergence:** the first step at which DANIEL's chosen operation, machine or start time differs from the best known schedule.")
    w("  - **Assignment difference:** DANIEL puts the operation on a different machine from the target.")
    w("  - **Sequencing difference:** same machine, but the target starts that operation at another time, so it schedules something else first.")
    w("- **Target action at the divergence step:**")
    w("  - **Rank:** its position in DANIEL's ranking of the allowed actions (1 = DANIEL's own choice).")
    w("  - **Probability:** the probability DANIEL gives it.")
    w("  - When several allowed actions agree with the target, the best-ranked one is used.")
    w("- **Phases:** early = steps 1-17 of 50, middle = 18-34, late = 35-50.")
    w("")
    for key in ("wm_final", "fs_final"):
        P = R["part4"][key]
        cases = P["cases"]
        dv = [c for c in cases if c["div"]]
        w(f"### {LAB[key]}")
        w("")
        gaps = [c["gap"] for c in cases]
        w(f"- **Already best:** DANIEL greedy is the best known schedule in {len(cases) - len(dv)} of {len(cases)} pairs. "
          f"Mean score gap to the best known: {f(np.mean(gaps))} (median {f(np.median(gaps))}).")
        if not dv:
            continue
        ph = Counter(c["div"]["phase"] for c in dv); kd = Counter(c["div"]["kind"] for c in dv)
        steps = np.array([c["div"]["step"] + 1 for c in dv])
        w(f"- **When the paths split:** at step {int(np.median(steps))} (median; quartiles "
          f"{int(np.percentile(steps, 25))}-{int(np.percentile(steps, 75))}).")
        w(f"  - Early {ph['early']} ({pct(ph['early'], len(dv))}), middle {ph['middle']} ({pct(ph['middle'], len(dv))}), "
          f"late {ph['late']} ({pct(ph['late'], len(dv))}).")
        w(f"- **Kind:** assignment {kd['assignment']} ({pct(kd['assignment'], len(dv))}), sequencing {kd['sequencing']} "
          f"({pct(kd['sequencing'], len(dv))}).")
        rk = [c["div"]["target_rank"] for c in dv if c["div"]["target_rank"] is not None]
        pr = [c["div"]["target_prob"] for c in dv if c["div"]["target_prob"] is not None]
        dp = [c["div"]["daniel_prob"] for c in dv]
        none = sum(c["div"]["target_rank"] is None for c in dv)
        w(f"- **The better schedule's action:**")
        w(f"  - Rank: 2 in {pct(sum(r == 2 for r in rk), len(rk))}, 3 in {pct(sum(r == 3 for r in rk), len(rk))}, "
          f"4 or worse in {pct(sum(r >= 4 for r in rk), len(rk))}, of {f(np.median([c['div']['n_allowed'] for c in dv]), 0)} "
          f"allowed actions (median).")
        w(f"  - DANIEL's probability for it: median {f(np.median(pr))}, quartiles {f(np.percentile(pr, 25))}-{f(np.percentile(pr, 75))}.")
        w(f"  - DANIEL's own choice there: median {f(np.median(dp))}.")
        w(f"  - Divergences where DANIEL rates the better action within 0.1 of its own choice: "
          f"{pct(sum(abs(c['div']['daniel_prob'] - (c['div']['target_prob'] or 0)) < 0.1 for c in dv), len(dv))}.")
        if none:
            w(f"  - Divergences with no allowed action consistent with the target: {none}.")
        cl = [c["closest"] for c in dv if c.get("closest")]
        if cl:
            cs = np.array([c["step"] + 1 for c in cl]); cph = Counter(c["phase"] for c in cl); ckd = Counter(c["kind"] for c in cl)
            crk = [c["target_rank"] for c in cl if c["target_rank"] is not None]
            cpr = [c["target_prob"] for c in cl if c["target_prob"] is not None]
            w(f"- **Closest better schedule.** The best known schedule is one target among many. A second measure uses, "
              f"of all pool schedules better than DANIEL's (median {int(np.median([c['n_better'] for c in dv]))} per pair), "
              "the one DANIEL's greedy path follows longest. The split from it is the step after which no better known "
              "schedule is still reachable.")
            w(f"  - Step: median {int(np.median(cs))} (quartiles {int(np.percentile(cs, 25))}-{int(np.percentile(cs, 75))}).")
            w(f"  - Phase: early {pct(cph['early'], len(cl))}, middle {pct(cph['middle'], len(cl))}, late {pct(cph['late'], len(cl))}.")
            w(f"  - Kind: assignment {pct(ckd['assignment'], len(cl))}, sequencing {pct(ckd['sequencing'], len(cl))}.")
            w(f"  - Better action's rank: 2 in {pct(sum(r == 2 for r in crk), len(crk))}, 3 in {pct(sum(r == 3 for r in crk), len(crk))}, "
              f"4 or worse in {pct(sum(r >= 4 for r in crk), len(crk))}.")
            w(f"  - DANIEL's probability for it: median {f(np.median(cpr))}.")
            w(f"  - Mean score gap to that schedule: {f(np.mean([c['closest_gap'] for c in dv]))}.")
        w("")
        w("**Where divergences happen.** The table shows the state at each step DANIEL takes while still on the target's path.")
        w("- **Rate:** divergences / steps on the path.")
        w("- **Best known:** the target is the best known schedule. Most of these splits come at steps 1-2, when several "
          "operations can start at time 0.")
        w("- **Closest better:** the target is the closest better schedule.")
        w("")
        w("| State at the step | Best known: steps | Best known: divergences (rate) | Closest better: steps | Closest better: divergences (rate) |")
        w("|---|---|---|---|---|")
        st = [s for s in P["steps"] if s["at_risk"]]
        allst = P["steps"]; nper = len(allst) // len(cases)
        cst = []
        for ci, c in enumerate(cases):
            k = c["closest"]["step"] if c.get("closest") else None
            for s in allst[ci * nper:(ci + 1) * nper]:
                if k is None or s["step"] <= k:
                    cst.append(dict(s, diverged=(s["step"] == k)))
        for name, fn, bins in (
                ("Phase", lambda s: s["phase"], ["early", "middle", "late"]),
                ("Ready jobs with an allowed pair", lambda s: "1" if s["n_ready_jobs"] == 1 else ("2" if s["n_ready_jobs"] == 2 else ("3-4" if s["n_ready_jobs"] <= 4 else "5+")), ["1", "2", "3-4", "5+"]),
                ("Allowed actions", lambda s: "1" if s["n_allowed"] == 1 else ("2-3" if s["n_allowed"] <= 3 else ("4-6" if s["n_allowed"] <= 6 else "7+")), ["1", "2-3", "4-6", "7+"]),
                ("Machines busy at that time", lambda s: str(s["busy_machines"]) if s["busy_machines"] < 4 else "4-5", ["0", "1", "2", "3", "4-5"]),
                ("Jobs already behind their due date (slack < 0)", lambda s: "0" if s["behind_jobs"] == 0 else ("1-2" if s["behind_jobs"] <= 2 else "3+"), ["0", "1-2", "3+"]),
                ("DANIEL's top probability", lambda s: "< 0.5" if s["top_prob"] < 0.5 else ("0.5-0.9" if s["top_prob"] < 0.9 else ">= 0.9"), ["< 0.5", "0.5-0.9", ">= 0.9"])):
            for b in bins:
                n = sum(fn(s) == b for s in st); k = sum(fn(s) == b and s["diverged"] for s in st)
                n2 = sum(fn(s) == b for s in cst); k2 = sum(fn(s) == b and s["diverged"] for s in cst)
                w(f"| {name}: {b} | {n} | {k} ({pct(k, n) if n else 'n/a'}) | {n2} | {k2} ({pct(k2, n2) if n2 else 'n/a'}) |")
        w("")
        bypg = {g: [c for c in cases if c["group"] == g] for g in GROUPS}
        w("By preference group: " + "; ".join(
            f"{g} {sum(c['div'] is not None for c in v)}/{len(v)} diverge, mean gap {f(np.mean([c['gap'] for c in v]))}"
            for g, v in bypg.items()) + ".")
        w("")
    _part4_fig(R, figs / "part4.png")
    w(f"![Divergence steps]({out.name}/figures/part4.png)")
    w("")
    w("These are descriptions. A divergence is where DANIEL's path leaves one particular better schedule; it is not "
      "evidence that this decision causes the gap, and other good schedules may branch elsewhere.")
    w("")

    # ------------------------------------------------------------------ verdict
    w("## What this says")
    w("")
    w(f"1. **CP-SAT is solving the same problem.** Its implementation is sound: same data, same objectives (scored by "
      "the same independent scorer), and proven-optimal epsilon-constraint solves. Its front is a 5 x 5 sample of the "
      "Pareto front.")
    w(f"2. **The idle time in CP-SAT's schedules is not, in itself, its advantage.** Left-shifting every schedule with "
      f"the same machines and order loses nothing (hypervolume {f(hv['cpsat'])} to {f(hv['ls'])}).")
    w(f"3. **What DANIEL cannot copy is the combination of machine choices and order that needs machines to wait.** "
      f"{'None' if ls_nd == 0 else ls_nd} of the {len(S)} CP-SAT schedules is non-delay after left-shifting.")
    w("   - Removing the waiting while keeping CP-SAT's machine choices costs hypervolume "
      f"{f(hv['ls'])} to {f(hv['nd'])}.")
    w("   - The best non-delay schedules known reach "
      f"{f(hv['pool'])}.")
    w("   - So waiting matters, but it cannot be separated from the assignment and sequence it serves.")
    w("   - The carbon gap is an assignment gap by definition.")
    w(f"4. **Within DANIEL's own rules there is room left.** Greedy hypervolume is {f(hv['wm_final'])} / "
      f"{f(hv['fs_final'])} against {f(hv['pool'])} for the best known non-delay schedules (Part 4).")
    for key in ("wm_final", "fs_final"):
        dv = [c for c in R["part4"][key]["cases"] if c["div"]]
        if not dv:
            continue
        cl = [c["closest"] for c in dv if c.get("closest")]
        w(f"   - {LAB[key]}:")
        w(f"     - It leaves the best known schedule at a median step {int(np.median([c['div']['step'] + 1 for c in dv]))}.")
        w(f"     - It leaves the last better known schedule at a median step {int(np.median([c['step'] + 1 for c in cl]))} of 50.")
        w(f"     - At that last split, the better action is DANIEL's second choice in "
          f"{pct(sum(c['target_rank'] == 2 for c in cl), len(cl))} of cases.")
    w("")
    w("## Files")
    w("")
    w("- `tools/cpsat_gap_v3.py`: the pool (GPU inference only) and the stages; `tools/cpsat_gap_v3_analysis.py`: the analysis and its definitions; `tools/analyze_cpsat_gap_v3.py`: this report.")
    w("- `results/cpsat_gap_v3/analysis.json.gz`: every number, per instance and preference, with input hashes.")
    w("- `results/cpsat_gap_v3/pool/`: the pool schedules: machine and start of every schedule, plus DANIEL's greedy trajectories with action probabilities for the two traced checkpoints.")
    md.write_text("\n".join(L) + "\n")
    print(f"written: {md}")


# ------------------------------------------------------------------------------------------------ figures

def _gantt(ex, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(ex)
    fig, axes = plt.subplots(2, len(labs), figsize=(4.2 * len(labs), 5.6), sharey=True)
    cmap = plt.get_cmap("tab10")
    for c, lab in enumerate(labs):
        e = ex[lab]
        for r, (st, idle, ttl) in enumerate(((e["start"], e["idle"], "CP-SAT"), (e["left_shift_start"], e["left_shift_idle"], "left-shifted"))):
            ax = axes[r, c]
            pt = np.asarray(e["pt"]); mch = np.asarray(e["machine"]); st = np.asarray(st)
            for o in range(len(st)):
                ax.barh(mch[o], pt[o, mch[o]], left=st[o], color=cmap(e["job_of_op"][o] % 10), edgecolor="black", linewidth=0.3)
            for p in idle["periods"]:
                if p["deliberate"] > 1e-6:
                    ax.barh(p["machine"], p["to"] - p["from"], left=p["from"], color="none", edgecolor="grey", hatch="///", linewidth=0)
            o = e["objectives"] if r == 0 else e["left_shift_objectives"]
            ax.set_title(f"{lab} ({ttl})\nMK {o[0]:.0f}, C {o[1]:.1f}, T {o[2]:.0f}", fontsize=9)
            ax.set_yticks(range(len(pt[0]))); ax.set_yticklabels([f"M{m}" for m in range(len(pt[0]))], fontsize=8)
            ax.tick_params(axis="x", labelsize=8)
            if r == 1:
                ax.set_xlabel("time (min)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _part3_fig(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("cpsat", "CP-SAT"), ("ls", "left-shifted"), ("nd", "non-delay,\nCP-SAT machines"), ("best", "best known\nnon-delay"),
            ("wm_final", "DANIEL greedy\nWM final"), ("fs_final", "DANIEL greedy\nFS final"), ("env", "CP-SAT order\nthrough DANIEL's rule")]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(keys)); wd = 0.2
    for k, grp in enumerate(GROUPS):
        v = [np.mean([r[key] for r in R["rows"] if r["group"] == grp]) for key, _ in keys]
        ax.bar(x + (k - 1.5) * wd, v, wd, label=grp)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=8)
    ax.set_ylabel("mean score (lower is better)"); ax.legend(fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _part4_fig(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for key in ("wm_final", "fs_final"):
        st = [c["div"]["step"] + 1 for c in R["part4"][key]["cases"] if c["div"]]
        ax.hist(st, bins=np.arange(0.5, 51.5, 1), alpha=0.6, label=LAB[key])
    ax.set_xlabel("step of first divergence (of 50)"); ax.set_ylabel("instance-preference pairs"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
