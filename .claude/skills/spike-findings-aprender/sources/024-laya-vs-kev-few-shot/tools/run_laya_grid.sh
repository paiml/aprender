#!/usr/bin/env bash
# Laya feature/zero-shot grid on the spike-015 rows (emotion capped test 1000 / train 3000, like Kev).
set -euo pipefail
cd "$(dirname "$0")/.."
run() { uv run --quiet --python 3.12 --with ./vendor/laya --with datasets python tools/laya_eval.py "$@"; }
for sub in en typed-decisions; do
  sf=""; [ "$sub" = typed-decisions ] && sf="typed-decisions"
  for task in stance-abortion emotion; do
    for split in test train; do
      lim=""; [ "$task" = emotion ] && { [ "$split" = test ] && lim="--limit 1000" || lim="--limit 3000"; }
      run --subfolder "$sf" --task "$task" --split "$split" $lim --out "runs/laya-$sub-$task-$split.npz" \
        > "runs/laya-$sub-$task-$split.log" 2>&1
      tail -1 "runs/laya-$sub-$task-$split.log"
    done
  done
  run --subfolder "$sf" --task stance-abortion --split test --names_only --out "runs/laya-$sub-stance-abortion-test-names.npz" \
    > "runs/laya-$sub-stance-names.log" 2>&1
  tail -1 "runs/laya-$sub-stance-names.log"
done
echo GRID-DONE
