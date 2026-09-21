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
# caller set it, else ~/.local/share/crux/hf-venv. The model cache is the caller's HF_HOME.
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
exec uv run --quiet --frozen --project "$HERE/crux_hf" python "$HERE/crux_hf/engine.py" "$@"
