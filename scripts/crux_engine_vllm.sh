#!/usr/bin/env bash
# scripts/crux_engine_vllm.sh — the vLLM CRUX engine (#3952).
#
#   crux_engine_vllm.sh probe | gen …     (arguments: scripts/crux_vllm/engine.py)
#
# `uv run --frozen` over the committed scripts/crux_vllm/uv.lock — never an ambient python, never a
# resolution at run time: vllm and ninja are pinned exactly (torch and transformers through the lock), and
# `probe` prints the resolved versions plus the lockfile's sha256 into the receipt.
#
# The environment lives OUTSIDE the tree (vllm + torch are gigabytes): $UV_PROJECT_ENVIRONMENT if the
# caller set it, else ~/.local/share/crux/vllm-venv. The model cache is CRUX's own: $CRUX_HF_HOME, else
# ~/.local/share/crux/hf-home — never the shared ~/.cache/huggingface (#3971).
#
# The venv's bin goes on PATH because vLLM's engine init shells out to `ninja` (measured on lambda,
# 2026-09-23: FileNotFoundError: 'ninja' without it). engine.py checks it by name before any model loads.
#
# No uv on this host is an UNUSABLE engine (exit 3, the reason on stderr), which the CRUX driver records
# as a refused row naming it — never a silent skip.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! command -v uv >/dev/null 2>&1; then
  echo "crux_engine_vllm: uv is not on PATH on $(hostname) — the vLLM engine cannot run here" >&2
  exit 3
fi
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-"$HOME/.local/share/crux/vllm-venv"}"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
# CRUX's OWN model cache (#3971): the shared ~/.cache/huggingface held a blob rewritten through its
# snapshot symlink, and any other consumer's write can reach a shared cache. A declared path that only
# CRUX populates, and engine.py hashes every file against its blob name before loading it.
export HF_HOME="${CRUX_HF_HOME:-"$HOME/.local/share/crux/hf-home"}"
exec uv run --quiet --frozen --project "$HERE/crux_vllm" python "$HERE/crux_vllm/engine.py" "$@"
