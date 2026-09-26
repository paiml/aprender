#!/usr/bin/env bash
# check_cuda_module_key_gate.sh - the PR-time half of the #3759 module-key release gate.
#
# scripts/cuda_module_key_gate.sh reads the per-host receipts of the CUDA lib suites run
# under the module-key guard. Producing a receipt needs a CUDA device, and CI's tree guards
# have none, so the gate itself runs at release time: it is declared in Cargo.toml
# [package.metadata.dogfood] and executed by `scripts/dogfood.sh --phase pre-publish`.
#
# This is what scripts/guard_tree.sh runs on every pull request: the receipt evaluator's
# case table (missing host, skipped mutant, device skips, stale CUDA tree, ...). It needs no
# device and takes well under a second. Without it the evaluator could rot between releases
# and still look wired. Exit 0 iff every row holds.
set -euo pipefail
exec bash "$(dirname "$0")/cuda_module_key_gate.sh" --self-test
