**Short verdict.**

- **Sparse reward: ruled out.** The tardiness reward is already dense, built the same way as DANIEL's makespan reward. Part B was therefore not run. The reward is back-loaded, though: more than 90% of each episode's tardiness return arrives in the second half.
- **10x5: learning explains most of the gap.**
  - A schedule DANIEL's own decision rule can produce reaches a mean of 50.5 or better on these instances, against CP-SAT's 16.1.
  - DANIEL-T greedy reaches 247.7, so at least 85% of the greedy gap is the policy not finding schedules it is allowed to make.
  - Its best of 100 samples (111.2) still leaves at least 64% to learning.
- **20x10: both causes matter, and this test cannot fully separate them.**
  - The best no-waiting schedule found (728.3) is far above CP-SAT (100.0), so the no-waiting rule costs at most 628 of the 1,078 greedy gap (58%).
  - The search only improved DANIEL-T's own schedules by 8% (789.5 to 728.3), so the true best no-waiting value may be much lower. That would move more of the gap to learning.
- **Two signs that learning is weak at 20x10:**
  - DANIEL-T removes only 14% of the reference rule's tardiness (10x5: 57%).
  - It is beaten on tardiness by the carbon specialist (1206.9 against 1058.6, `results/baselines_v3.md`).
- **The no-waiting rule is costly for simple rules, not only for CP-SAT.**
  - The same EDD+ECT rule gives 932 with waiting allowed but 5,723 without, because without waiting a job is forced onto whichever eligible machine is free now, however slow.
  - CP-SAT's own job order, forced through the same mask, gives 2,766.

**Most of the gap:** at 10x5, learning. At 20x10, the no-waiting action mask accounts for at most about half, and the rest is learning.
