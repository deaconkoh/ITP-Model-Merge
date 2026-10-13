### Carbon diagnostics on DEV-CAL (Part 3): new energy-based carbon vs old i.i.d. carbon

| size | carbon | C1 (fastest != lowest-carbon machine) | Spearman(makespan, carbon) median [IQR] | 2-D non-dominated set (median) | TM_makespan median [IQR] | TM_carbon median [IQR] | Spearman(carbon, workload) median |
|---|---|---|---|---|---|---|---|
| 10x5 | NEW P_m x p | 14.4% | +0.659 [+0.59, +0.73] | 4 | 47.9% [31.1, 64.7] | 22.6% [17.4, 29.6] | +0.908 |
| 10x5 | old i.i.d. 1-100 | 53.5% | -0.021 [-0.20, +0.16] | 5 | 82.9% [67.7, 108.8] | 69.4% [57.3, 84.3] | -0.093 |
| 20x10 | NEW P_m x p | 16.5% | +0.628 [+0.58, +0.69] | 5 | 76.8% [51.3, 114.1] | 33.6% [29.3, 39.0] | +0.916 |
| 20x10 | old i.i.d. 1-100 | 69.8% | -0.017 [-0.19, +0.13] | 5 | 159.9% [143.0, 187.1] | 141.6% [129.3, 157.0] | -0.051 |

**Carbon verdict (pre-registered rule, NEW carbon): FAIL** (10x5 fail, 20x10 fail).

### Tardiness calibration grid, 10x5 (DEV-CAL; EDD+ECT tardy share; * = passes all pre-registered conditions)

| s \ k | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 89% | 82% | 73% | 65% | 56% | 46%* | 39%* | 31%* | 23% | 16% | 12% | 9% | 6% | 3% | 2% |
| 0.4 | 92% | 85% | 77% | 66% | 58% | 49%* | 41%* | 32%* | 24% | 18% | 13% | 9% | 6% | 4% | 2% |
| 0.6 | 95% | 89% | 81% | 73% | 63% | 52% | 41%* | 32%* | 23% | 16% | 10% | 6% | 4% | 2% | 1% |
| 0.8 | 96% | 91% | 83% | 76% | 67% | 57% | 45%* | 35%* | 24% | 17% | 11% | 7% | 4% | 3% | 2% |

Passing configurations (10): (k=1.1, s=0.2), (k=1.2, s=0.2), (k=1.3, s=0.2), (k=1.1, s=0.4), (k=1.2, s=0.4), (k=1.3, s=0.4), (k=1.2, s=0.6), (k=1.3, s=0.6), (k=1.2, s=0.8), (k=1.3, s=0.8)

| config | T1 tardy share | EDD vs SPT | EDD vs MWKR | zero-T share | all-late share | CV(T) | Spearman(T, mk) | Spearman(T, sum C) | close-mk pairs with T diff > 10% | ND(mk, T) | mean T (EDD) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k=1.3, s=0.8 (CHOSEN) | 34.8% | 60.9% | 67.3% | 9% | 0% | 0.47 | +0.866 | +0.993 | 62.3% (of 6880) | 3 | 226.7 |
| k=1.3, s=0.6 | 32.2% | 56.6% | 60.4% | 8% | 0% | 0.48 | +0.867 | +0.994 | 62.8% (of 6886) | 3 | 216.6 |
| k=1.3, s=0.4 | 32.1% | 40.7% | 48.3% | 7% | 0% | 0.48 | +0.868 | +0.995 | 63.5% (of 6894) | 2 | 266.3 |
| k=1.3, s=0.2 | 31.5% | 42.5% | 42.1% | 4% | 0% | 0.49 | +0.866 | +0.995 | 63.7% (of 6873) | 2 | 256.4 |
| k=1.2, s=0.2 | 38.9% | 36.0% | 41.6% | 2% | 0% | 0.46 | +0.864 | +0.997 | 61.1% (of 6873) | 3 | 356.5 |
| k=1.2, s=0.4 | 41.0% | 34.6% | 44.3% | 1% | 0% | 0.45 | +0.865 | +0.997 | 60.7% (of 6894) | 3 | 369.9 |
| k=1.2, s=0.6 | 41.3% | 47.2% | 53.9% | 3% | 0% | 0.45 | +0.865 | +0.996 | 60.3% (of 6886) | 3 | 316.6 |
| k=1.2, s=0.8 | 45.1% | 49.4% | 55.6% | 0% | 0% | 0.44 | +0.867 | +0.996 | 59.7% (of 6880) | 3 | 333.7 |

**Chosen 10x5: k = 1.3, s = 0.8.** DEV-CHECK confirmation: T1 32.2%, zero-T 8%, all-late 0%, Spearman(T, mk) +0.876, Spearman(T, sum C) +0.994, EDD vs SPT 70.2% -> **PASS**

### Tardiness calibration grid, 20x10 (DEV-CAL; EDD+ECT tardy share; * = passes all pre-registered conditions)

| s \ k | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 77% | 64% | 50% | 39%* | 28%* | 20% | 12% | 7% | 3% | 2% | 1% | 0% | 0% | 0% | 0% |
| 0.4 | 84% | 70% | 55% | 40%* | 27%* | 17% | 9% | 5% | 2% | 1% | 0% | 0% | 0% | 0% | 0% |
| 0.6 | 89% | 76% | 59% | 41%* | 26%* | 13% | 7% | 2% | 1% | 0% | 0% | 0% | 0% | 0% | 0% |
| 0.8 | 91% | 78% | 62% | 43%* | 26%* | 12% | 6% | 2% | 1% | 1% | 0% | 0% | 0% | 0% | 0% |

Passing configurations (8): (k=0.9, s=0.2), (k=1.0, s=0.2), (k=0.9, s=0.4), (k=1.0, s=0.4), (k=0.9, s=0.6), (k=1.0, s=0.6), (k=0.9, s=0.8), (k=1.0, s=0.8)

| config | T1 tardy share | EDD vs SPT | EDD vs MWKR | zero-T share | all-late share | CV(T) | Spearman(T, mk) | Spearman(T, sum C) | close-mk pairs with T diff > 10% | ND(mk, T) | mean T (EDD) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k=0.9, s=0.2 (CHOSEN) | 39.1% | 50.0% | 34.1% | 0% | 0% | 0.48 | +0.847 | +0.998 | 32.6% (of 8199) | 3 | 1024.2 |
| k=0.9, s=0.4 | 40.4% | 55.1% | 45.5% | 0% | 0% | 0.47 | +0.847 | +0.998 | 32.5% (of 8219) | 3 | 912.2 |
| k=0.9, s=0.6 | 41.1% | 63.9% | 55.5% | 0% | 0% | 0.47 | +0.847 | +0.998 | 32.3% (of 8209) | 3 | 779.7 |
| k=1.0, s=0.2 | 28.0% | 57.8% | 23.4% | 2% | 0% | 0.50 | +0.848 | +0.997 | 35.7% (of 8199) | 2 | 658.6 |
| k=1.0, s=0.4 | 27.1% | 66.0% | 46.0% | 0% | 0% | 0.50 | +0.848 | +0.997 | 35.9% (of 8219) | 3 | 540.1 |
| k=0.9, s=0.8 | 43.0% | 69.1% | 63.4% | 0% | 0% | 0.47 | +0.847 | +0.998 | 32.3% (of 8208) | 3 | 719.4 |
| k=1.0, s=0.8 | 25.9% | 82.5% | 76.2% | 1% | 0% | 0.49 | +0.847 | +0.998 | 35.7% (of 8208) | 4 | 354.8 |
| k=1.0, s=0.6 | 25.8% | 75.8% | 63.4% | 1% | 0% | 0.49 | +0.848 | +0.998 | 35.8% (of 8209) | 3 | 412.4 |

**Chosen 20x10: k = 0.9, s = 0.2.** DEV-CHECK confirmation: T1 38.5%, zero-T 0%, all-late 0%, Spearman(T, mk) +0.846, Spearman(T, sum C) +0.998, EDD vs SPT 52.7% -> **PASS**

### Joint three-objective check (Part 5; chosen carbon and due dates; median per-instance Spearman over 68 schedules)

**10x5, DEV-CAL**

| | makespan | carbon | tardiness | sum_Cj | workload |
|---|---|---|---|---|---|
| makespan | +1.00 | +0.68 | +0.87 | +0.86 | +0.72 |
| carbon | +0.68 | +1.00 | +0.75 | +0.75 | +0.91 |
| tardiness | +0.87 | +0.75 | +1.00 | +0.99 | +0.80 |
| sum_Cj | +0.86 | +0.75 | +0.99 | +1.00 | +0.80 |
| workload | +0.72 | +0.91 | +0.80 | +0.80 | +1.00 |

3-D non-dominated set (makespan, carbon, tardiness): median 8 schedules [IQR 6-9] = 11.8% of the set; **three distinct objectives: YES**

**10x5, DEV-CHECK**

| | makespan | carbon | tardiness | sum_Cj | workload |
|---|---|---|---|---|---|
| makespan | +1.00 | +0.70 | +0.88 | +0.86 | +0.73 |
| carbon | +0.70 | +1.00 | +0.77 | +0.77 | +0.94 |
| tardiness | +0.88 | +0.77 | +1.00 | +0.99 | +0.81 |
| sum_Cj | +0.86 | +0.77 | +0.99 | +1.00 | +0.81 |
| workload | +0.73 | +0.94 | +0.81 | +0.81 | +1.00 |

3-D non-dominated set (makespan, carbon, tardiness): median 8 schedules [IQR 7-9] = 11.8% of the set; **three distinct objectives: YES**

**20x10, DEV-CAL**

| | makespan | carbon | tardiness | sum_Cj | workload |
|---|---|---|---|---|---|
| makespan | +1.00 | +0.66 | +0.85 | +0.84 | +0.70 |
| carbon | +0.66 | +1.00 | +0.75 | +0.75 | +0.92 |
| tardiness | +0.85 | +0.75 | +1.00 | +1.00 | +0.80 |
| sum_Cj | +0.84 | +0.75 | +1.00 | +1.00 | +0.80 |
| workload | +0.70 | +0.92 | +0.80 | +0.80 | +1.00 |

3-D non-dominated set (makespan, carbon, tardiness): median 10 schedules [IQR 8-11] = 14.7% of the set; **three distinct objectives: YES**

**20x10, DEV-CHECK**

| | makespan | carbon | tardiness | sum_Cj | workload |
|---|---|---|---|---|---|
| makespan | +1.00 | +0.66 | +0.85 | +0.84 | +0.70 |
| carbon | +0.66 | +1.00 | +0.74 | +0.74 | +0.92 |
| tardiness | +0.85 | +0.74 | +1.00 | +1.00 | +0.80 |
| sum_Cj | +0.84 | +0.74 | +1.00 | +1.00 | +0.80 |
| workload | +0.70 | +0.92 | +0.80 | +0.80 | +1.00 |

3-D non-dominated set (makespan, carbon, tardiness): median 10 schedules [IQR 8-10] = 14.7% of the set; **three distinct objectives: YES**

