# ITP DANIEL Scheduling Project

This repository contains the code, datasets, model checkpoints, and result files for an ITP project based on DANIEL for the Flexible Job Shop Scheduling Problem.

New research runs must follow [the canonical scientific protocol](docs/scientific_protocol.md).
Inherited checkpoints and results are governed by [the legacy artifact policy](docs/legacy_artifacts.md).

The project extends DANIEL-style scheduling experiments with carbon and priority objectives, then compares specialist models, Model Soup merges, Task Arithmetic merges, and a combined-objective baseline.

## Key Folders

```text
daniel/                         DANIEL source code, datasets, checkpoints, and evaluation outputs
tools/                          dataset, merge, and evaluation utilities
experiments/flexibility/         experiment CSV summaries used by the final report
test_results/                    final-report spreadsheet evidence
```

## Training Command

Canonical training requires physically separate training and validation data:

```bash
cd daniel
python train.py \
  --config ../configs/canonical/m.json \
  --train_data_path ./data/data_train/SD2/20x10+carbon+priority \
  --validation_data_path ./data/data_validation/SD2/20x10+carbon+priority \
  --test_data_path ./data/data_final_test/SD2/20x10+carbon+priority \
  --n_j 20 --n_m 10 --data_suffix carbon+priority
```

`train.py` records the final-test path for provenance but never reads it. The
commands below are retained for legacy reference.

From the repository root:

```powershell
cd daniel
..\.venv\Scripts\python.exe train.py --data_suffix carbon+priority --n_j 20 --n_m 10 --model_suffix priority --goal p
```

Replace `p` in --goal with `c` to train model towards minimizing carbon, or leave out --goal to train model towards minimizing makespan.

> [!TIP]
> If you encounter `Torch not compiled with CUDA enabled` error, add the argument `--device cpu` to your command and rerun it.

## Soup Merge Command

From the repository root:

```powershell
cd daniel
..\.venv\Scripts\python.exe ..\tools\merge_daniel_checkpoints.py --method soup --feature-schema legacy_f11_p9_v1 --data-source SD2 --checkpoints 20x10+carbon+priority+speed 20x10+carbon+priority+carbon 20x10+carbon+priority+priority --weights 0.5 0.25 0.25 --out-name merged_20x10_m5
```

Modify the weight of models in merge by modifying the respective weight value in --weights argument. Modify the name of the merged model using --out-name argument.

## Task Arithmetic Merge Command

From the repository root:

```powershell
cd daniel
..\.venv\Scripts\python.exe ..\tools\merge_daniel_checkpoints.py --method task_arithmetic --feature-schema legacy_f11_p9_v1 --allow-legacy-task-arithmetic --data-source SD2 --base 20x10+carbon+priority+speed --checkpoints 20x10+carbon+priority+carbon 20x10+carbon+priority+priority --weights 0.6 0.6 --out-name merged_20x10_TA0606
```

Modify the weight of models in merge by modifying the respective weight value in --weights argument. Modify the name of the merged model using --out-name argument.

## Main Evaluation Command

From the repository root:

```powershell
cd daniel
..\.venv\Scripts\python.exe test_trained_model.py --test_data 10x5+carbon+priority --test_model 20x10+carbon+priority+speed
```

Replace `20x10+carbon+priority+speed` with another checkpoint name in `daniel/trained_network/SD2/` to evaluate a different model.

Note: the final-report comparison evaluates the 20x10-trained checkpoints on the `10x5+carbon+priority` test set (100 instances, greedy), as recorded in `daniel/test_results/test_results_summary.csv` and `daniel/test_results/SD2/10x5+carbon+priority/`.

## Important Result Files

```text
test_results/ModelTestResults_relevant_sheets.xlsx
experiments/flexibility/merge_ratio_sweep_raw_summary.csv
experiments/flexibility/merged_p20_c20_s60_finetune_sweep_summary.csv
experiments/flexibility/merged_p20_c20_s60_seed_repeat_summary.csv
experiments/flexibility/quick_flex_eval.csv
experiments/flexibility/machine_disruption_eval.csv
```

`test_results/ModelTestResults_relevant_sheets.xlsx` contains the final report table evidence, including the 20x10-trained specialist, Model Soup, Task Arithmetic, and all-objective comparison values evaluated on `10x5+carbon+priority`.
