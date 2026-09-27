#!/usr/bin/env bash
# scripts/crux_engine_hf.sh — the Hugging Face transformers CRUX engine (#3739, PMAT-3778).
#
#   crux_engine_hf.sh probe | gen … | tok … | tmpl … | greedy …     (arguments: scripts/crux_hf/engine.py)
#
# `uv run --frozen` over the committed scripts/crux_hf/uv.lock — never an ambient python, never a
# resolution at run time: transformers[serving], torch, tokenizers, jinja2, numpy, safetensors,
# huggingface_hub and requests are pinned exactly, and `probe` prints the resolved versions plus the
# lockfile's sha256 into the receipt.
#
# The environment lives OUTSIDE the tree (torch alone is gigabytes): $UV_PROJECT_ENVIRONMENT if the
# caller set it, else ~/.local/share/crux/hf-venv. The model cache is CRUX's own (#3971): $CRUX_HF_HOME, else
# ~/.local/share/crux/hf-home.
#
# No uv on this host is an UNUSABLE engine (exit 3, the reason on stderr), which the CRUX driver records
# as a refused row naming it — never a silent skip.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! command -v uv >/dev/null 2>&1; then
  echo "crux_engine_hf: uv is not on PATH on $(hostname) — the HF engine cannot run here" >&2
  exit 3
fi
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/.local/share/crux/hf-venv}
# CRUX's OWN model cache (#3971), shared with the vllm engine and populated only by CRUX; engine.py hashes
# every source file against its blob name before loading it. Never the shared ~/.cache/huggingface.
export HF_HOME="${CRUX_HF_HOME:-"$HOME/.local/share/crux/hf-home"}"
# NOT `exec uv run`: uv did not forward SIGTERM to python, and a stopped driver left VLLM::EngineCore on the card
# holding 58928 MiB (aprender-dd, gx10, 2026-09-23). Sync the locked environment, then exec its python, so a
# signal reaches engine.py, whose handler (scripts/lib/crux_proc.py) takes every engine child with it.
uv sync --quiet --frozen --project "$HERE/crux_hf" || {
  echo "crux_engine_hf: uv sync --frozen failed for $HERE/crux_hf — the locked environment cannot be built here" >&2
  exit 3
}
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$HERE/crux_hf/engine.py" "$@"
