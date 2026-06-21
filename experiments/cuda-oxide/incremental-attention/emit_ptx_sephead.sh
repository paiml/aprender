#!/usr/bin/env bash
# PMAT-884 — emit the standalone embeddable PTX for the oxide
# `attn_warp_sephead_rawptr` decode-attention kernel (sm_121, GB10 Blackwell).
#
# Same compute as PMAT-883 `attn_warp_rawptr`, but indexes the LIVE serve KV
# cache layout [num_kv_heads, max_len, head_dim] directly (10-param ABI with an
# extra `kv_stride` arg) — no interleave/gather adapter. This wrapper just runs
# the shared emit_ptx.sh pipeline with ENTRY pinned to the sephead entry.
#
# Run on gx10 (GB10, sm_121) ONLY:
#   ssh gx10
#   cd /tmp/incattn_spike   # or rsync this dir
#   ./emit_ptx_sephead.sh [out.ptx]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" || exit 1; pwd)"
OUT="${1:-generated/attn_warp_sephead.sm121.ptx}"
ENTRY="attn_warp_sephead_rawptr" exec "${SCRIPT_DIR}/emit_ptx.sh" "${OUT}"
