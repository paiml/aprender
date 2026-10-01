#!/opt/homebrew/bin/bash
# Spike 016 regression set. Each suite's rc is captured directly (never through a pipe).
set -u
export CARGO_TARGET_DIR=/Users/guy/Development/machine-learning/aprender/target
L=/private/tmp/claude-501/-Users-guy-Development-machine-learning-aprender/63502490-acb9-4604-a5c8-251a1085a9a9/scratchpad/t016
mkdir -p $L
run() { name=$1; shift; s=$(date +%s); "$@" > "$L/$name.log" 2>&1; rc=$?; echo "$name rc=$rc $(( $(date +%s)-s ))s $(grep -E '^test result:' "$L/$name.log" | awk '{p+=$4; f+=$6; i+=$8} END {print "passed="p" failed="f" ignored="i}')"; }
run forecast        cargo test --release -p aprender-forecast
run core-setfit     cargo test --release -p aprender-core --features setfit --lib setfit
run train-setfit    cargo test --release -p aprender-train --features setfit --lib setfit
run compute-lib     cargo test --release -p aprender-compute --lib
run serve-qwen35    cargo test --release -p aprender-serve --lib qwen35
run mcp-setfit      cargo test --release -p aprender-mcp-setfit
echo T016-DONE
