#!/usr/bin/env bash
# EXT-28 (aprender#4410): one record's arm outputs -> one cell receipt.
#
#   assemble_cell.sh <record dir> <tag> <cell> > cell.json
#
# Every arm that c2_cell.sh measured is recorded as `measured`; an arm whose run
# failed is `refused` with the failure line. Ollama is `refused` under S-14 on the
# inspection in ollama-blob.json (its blob carries a vision tower). Whether a
# `measured` arm is really like-for-like is not decided here: the Rust check
# (`speed_arms::check_cell`) re-judges S-14 and turns the receipt RED if not.
set -euo pipefail
dir=${1:?record dir}
tag=${2:?tag}
cell=${3:?cell}
here=$(cd "$(dirname "$0")" && pwd)
arms='{}'
for arm in apr llama.cpp mistral.rs; do
    if [ -s "$dir/$arm.json" ]; then
        arms=$(jq --arg a "$arm" --slurpfile r "$dir/$arm.json" '.[$a] = {measured: $r[0]}' <<<"$arms")
    else
        why=$(tail -1 "$dir/cell.out" 2>/dev/null || echo "no output")
        arms=$(jq --arg a "$arm" --arg w "run failed: $why" '.[$a] = {refused: {reason: $w}}' <<<"$arms")
    fi
done
reason=$(jq -r '"S-14 not like-for-like: the served blob sha256:\(.model_blob_sha256) carries a vision path (\(.vision_tensors) of \(.tensors) tensors; the server log reads \"\(.server_log_line)\"); ours carries \(.ours_vision_tensors). Refused, not normalised; see ollama-blob.json"' "$here/ollama-blob.json")
arms=$(jq --arg w "$reason" '.ollama = {refused: {reason: $w}}' <<<"$arms")
jq -n --arg tag "$tag" --arg cell "$cell" --argjson arms "$arms" '{tag: $tag, cell: $cell, arms: $arms}'
