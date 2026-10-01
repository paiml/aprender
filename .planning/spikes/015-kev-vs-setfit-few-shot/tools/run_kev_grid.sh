#!/usr/bin/env bash
# Kev feature/zero-shot grid. Emotion is capped (test 1000, train 3000): only the shot pool and a stable
# test estimate are needed. Run from vendor/kev.
set -euo pipefail
R=../../runs
for run in 0.8b 4b; do
  for task in stance-abortion emotion; do
    lim_test=""; lim_train=""
    if [ "$task" = emotion ]; then lim_test="--limit 1000"; lim_train="--limit 3000"; fi
    for split in test train; do
      lim=$lim_test; [ "$split" = train ] && lim=$lim_train
      uv run python ../../tools/kev_eval.py --run "jaredpalmer/kev-$run" --task "$task" --split "$split" $lim \
        --out "$R/kev-$run-$task-$split.npz" > "$R/kev-$run-$task-$split.log" 2>&1
      tail -1 "$R/kev-$run-$task-$split.log"
    done
  done
  uv run python ../../tools/kev_eval.py --run "jaredpalmer/kev-$run" --task stance-abortion --split test --names_only \
    --out "$R/kev-$run-stance-abortion-test-names.npz" > "$R/kev-$run-stance-names.log" 2>&1
  tail -1 "$R/kev-$run-stance-names.log"
done
echo GRID-DONE
