# Step 6 -- reward-weight recalibration for the (makespan, carbon, tardiness) triple, 10x5

Reference composition: simplex centroid (333,333,334) built from m_s, c_s and t_k125s060_s specialists,
4 seeds, frozen due dates k125s060. Pool means: makespan 486.5, carbon 2060.7, tardiness 669.2.
0% of instances have zero tardiness at the reference (tardiness is still normalised by its pool mean,
as fixed in advance).

Closed-form anchors for (1/3, 1/3, 1/3): carbon_reward_weight 0.002384, tardiness_reward_weight
0.726920. Verification: 250 updates from the seed-111 centroid, goal mct.

| candidate | w_c | w_t | makespan | carbon | tardiness | objective | operating point m : c : t |
|---|---|---|---|---|---|---|---|
| anchor | 0.002384 | 0.7269 | 464.0 | 2165.0 | 488.8 | 0.8877 | 0.954 : 1.051 : 0.730 |
| w_c x0.5 | 0.001192 | 0.7269 | 427.9 | 2449.9 | 371.8 | 0.8605 | 0.880 : 1.189 : 0.556 |
| w_c x2 | 0.004769 | 0.7269 | 467.1 | 2062.4 | 613.3 | 0.9264 | 0.960 : 1.001 : 0.916 |
| w_c x4 | 0.009538 | 0.7269 | 495.9 | 1988.8 | 839.1 | 1.0314 | 1.019 : 0.965 : 1.254 |
| w_c x8 | 0.019076 | 0.7269 | 526.1 | 1924.2 | 1011.6 | 1.1154 | 1.081 : 0.934 : 1.512 |
| w_t x0.5 | 0.002384 | 0.3635 | 453.6 | 2230.1 | 460.7 | 0.8799 | 0.932 : 1.082 : 0.688 |
| w_t x2 | 0.002384 | 1.4538 | 437.9 | 2412.7 | 352.9 | 0.8534 | 0.900 : 1.171 : 0.527 |
| w_t x4 | 0.002384 | 2.9077 | 431.9 | 2321.1 | 427.7 | 0.8659 | 0.888 : 1.126 : 0.639 |
| w_t x8 | 0.002384 | 5.8154 | 445.8 | 2463.0 | 309.8 | 0.8483 | 0.916 : 1.195 : 0.463 |

The first sweep (x0.5, x2) put the best point on its boundary (w_t x2), so the same axis was extended
to x4 and x8 to bracket it.

Paired over the 100 instances:
* w_t x2 beats the anchor (63% of instances, p = 0.0012); w_t x8 beats the anchor (65%, p = 6.6e-05).
* w_t x2, x4 and x8 are NOT distinguishable from each other (p = 0.57, 0.16, 0.075).

Reading: the closed-form anchor does NOT carry over cleanly. It under-weights tardiness by at least 2x,
and beyond that the objective is flat from 2x to 8x rather than showing a sharp optimum. Every good
candidate buys a large tardiness reduction (to 0.46-0.56 of the reference) with a carbon increase of
17-20%: tardiness is by far the cheapest objective for fine-tuning to improve. The weight for the
fine-tuning arms is NOT fixed here -- they are not being run -- and the choice within the plateau
(smallest, x2 = 1.4538, or numerically best, x8 = 5.8154) is left open.

The anchor derivation carried the calibration for priority to within a factor of 2 (half-anchor won,
p = 0.026); for tardiness it is off by at least a factor of 2 in the other direction.
