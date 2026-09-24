#!/usr/bin/env bash
# AutoGluon canonical flow, transcribed from README.md + tabular quick start (1.6.3).
# Run with: uv run --with autogluon.tabular python - <<'PY'
set -euo pipefail
python - <<'PY'
from autogluon.tabular import TabularDataset, TabularPredictor
train = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
test  = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
predictor = TabularPredictor(label="class").fit(train, presets="medium_quality", time_limit=120)
print(predictor.problem_type)                 # inferred: binary
print(predictor.leaderboard(test))            # model, score_test, score_val, pred_time_*, fit_time, stack_level
print(predictor.feature_importance(test))     # permutation importance on raw columns
predictor.clone_for_deployment("deploy/")     # keep_only_best + save_space
PY
