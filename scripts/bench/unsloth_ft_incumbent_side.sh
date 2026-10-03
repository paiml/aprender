#!/usr/bin/env bash
# unsloth_ft_incumbent_side.sh — incumbent (Unsloth) side runner for unsloth_finetune_throughput.sh.
#
# Runs unsloth_ft_incumbent.py in the pinned uv project scripts/bench/unsloth-incumbent
# (unsloth + fla, versions recorded in every receipt). All arguments pass through.
# Exit 3 if the project is missing or uv is absent: no run in an unpinned environment.
set -uo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
project=${UNSLOTH_INCUMBENT_PROJECT:-$root/scripts/bench/unsloth-incumbent}

[ -f "$project/pyproject.toml" ] || { echo "$0: REFUSED: no pinned project at $project" >&2; exit 3; }
command -v uv > /dev/null || { echo "$0: REFUSED: uv not found" >&2; exit 3; }
exec uv run --project "$project" python "$root/scripts/bench/unsloth_ft_incumbent.py" "$@"
