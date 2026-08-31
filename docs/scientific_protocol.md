# Canonical scientific protocol (v2)

This protocol preserves the inherited makespan, carbon, and operation-priority
methodology while removing implementation ambiguity.

## Canonical model and observations

New experiments use `canonical_f11_p9_v2`: 11 operation features, 8 machine
features, and 9 operation-machine pair features. The eleventh operation feature
is operation priority and the ninth pair feature is carbon. Observation
availability does not activate reward terms.

Legacy F10/P8, F11/P8, and F11/P9-v1 checkpoints require explicit legacy
schemas and must not be silently evaluated with v2 feature semantics.

## Objectives

Let `M_t` be the inherited maximum operation completion lower-bound potential,
`C` total selected operation-machine carbon, and
`P_t = sum(q_o * C_hat_o,t) / sum(q_o)`.

| Goal | Step reward |
|---|---|
| `m` | `M_(t-1) - M_t` |
| `c` | `-w_c * chosen_carbon` |
| `p` | `w_p * (P_(t-1) - P_t)` |
| `mc` | makespan + carbon terms |
| `mp` | makespan + operation-priority terms |
| `mcp` | all three terms |

Validation checkpoint selection minimizes the matching terminal loss in the
same units used by training: time-based terms are divided by the environment's
recorded processing-time scale and carbon remains unscaled. Raw makespan,
carbon, and priority metrics are also reported. No goal receives a term merely
because its feature is observable.

Priority is canonical only at operation scope. Job-priority experiments are
legacy/experimental and must pass `--priority_scope legacy_job`.

## Data isolation

Every run uses separate training and validation directories. `train.py` does
not load final-test data. Exact hashes and structural FJSP hashes are checked
across training and validation before model construction. Carbon/priority
variants with the same processing-time instance therefore cannot cross splits.

Use `tools/build_clean_splits.py` to construct deterministic grouped splits and
`tools/validate_dataset_splits.py` before freezing a final protocol. Historical
test data used for selection is legacy development data, not a final holdout.
The input pool used to create a new final-test split must itself be newly
generated or otherwise untouched; reshuffling the inherited test set does not
make it an independent final test set.

## Selection and reporting

All hyperparameters, reward weights, checkpoints, merge weights, and fine-tune
durations are selected on validation only. A final-test manifest is frozen
before final evaluation. Report every seed and aggregate at least five
independent training seeds.

Use `tools/summarize_canonical_multiseed.py` for canonical result tables. It
rejects missing, unexpected, or duplicated seeds before computing aggregates.

Create that lock with `tools/freeze_evaluation_manifest.py`. The command refuses
to overwrite a prior manifest and validates all three splits before recording
the selected checkpoint hashes. Pass the resulting manifest through
`--final-evaluation-manifest` to the canonical carbon or operation-priority
evaluator; it rechecks both the checkpoint and test-data hashes.

Task Arithmetic requires specialists whose checkpoint metadata all identify
the same base checkpoint hash. The merge tool rejects unverified Task
Arithmetic unless the explicit legacy-reproduction override is supplied.

## Artifact requirements

New checkpoints are versioned bundles containing weights, resolved config,
data fingerprints, validation metrics, selection metric, parent hashes, and Git
state. State-dict-only files remain loadable as legacy artifacts but emit an
incomplete-provenance warning.
