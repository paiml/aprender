#!/usr/bin/env bash
# #4221 row runner: one engine x one model, the llama side PINNED through scripts/llama_bin.sh.
# Usage: gpu-q -- bash scripts/crux_llamacpp_parity_4221.sh <apr|llama|llama-nongating> <model.gguf> <out-dir> [apr-bin|llama-server]
set -euo pipefail
engine="$1"; model="$2"; out="$3"; aprbin="${4:-}"
export TMPDIR="${TMPDIR:-/mnt/nvme-raid0/tmp}"
case "$engine" in
  llama)
    . scripts/llama_bin.sh || true
    llama_bin_resolve || { echo "BLOCKER: llama_bin_resolve rc=$LLAMA_PIN_RC reason=$LLAMA_PIN_REASON" >&2; exit 2; }
    [ -n "${LLAMA_SERVER:-}" ] || { echo "BLOCKER: pinned build has no llama-server" >&2; exit 2; }
    flags=$(llama_comparator_server_flags 999 1) || { echo "BLOCKER: comparator flags rc=$?" >&2; exit 2; }
    exec python3 scripts/crux_llamacpp_parity_4221.py --engine llama --bin "$LLAMA_SERVER" \
        --model "$model" --out "$out" --llama-flags "$flags" --llama-build "$LLAMA_BUILD" ;;
  llama-nongating)
    # Cop ruling 2026-09-24: v0.5.0 is a SEPARATE column, "non-gating, upstream latest".
    # The binary is named explicitly (4th arg); the pin's flags are still used.
    [ -x "$aprbin" ] || { echo "llama-nongating needs an explicit llama-server path" >&2; exit 2; }
    . scripts/llama_bin.sh || true
    flags=$(llama_comparator_server_flags 999 1) || { echo "BLOCKER: comparator flags rc=$?" >&2; exit 2; }
    build=$("$aprbin" --version 2>&1 | grep -m1 version || true)
    exec python3 scripts/crux_llamacpp_parity_4221.py --engine llama --non-gating --bin "$aprbin" \
        --model "$model" --out "$out" --llama-flags "$flags" --llama-build "$build" ;;
  apr)
    [ -x "$aprbin" ] || { echo "apr engine needs an explicit apr binary path" >&2; exit 2; }
    exec python3 scripts/crux_llamacpp_parity_4221.py --engine apr --bin "$aprbin" \
        --model "$model" --out "$out" ;;
  *) echo "engine must be apr or llama" >&2; exit 2 ;;
esac
