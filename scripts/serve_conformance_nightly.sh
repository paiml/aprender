#!/usr/bin/env bash
# serve_conformance_nightly.sh — LAB nightly for the agent conformance suite (#5349, EPIC 0.73 row R-6).
# Starts `apr serve run` on one model on CPU, runs scripts/serve_conformance.sh against it, stops the server.
# LAB (C324): nothing on a PR or release path runs this, and a red opens/updates ONE ticket, then stays silent.
# The CI wiring (a nightly workflow, the GPU certified-matrix leg, the PR job) is a gate-content change and
# is not part of this script; the PR job also needs its own ruling and three green nights.
#
# Usage: bash scripts/serve_conformance_nightly.sh MODEL_FILE [--name LABEL] [--port N] [--out receipt.json]
#   MODEL_FILE  a local GGUF/APR/safetensors file (the nightly uses qwen3-1.7b-q4km; --name defaults to that)
# Pins the binary through scripts/apr_bin.sh ($APR), never a bare `apr`.
# Exit: the suite's (0 pass · 1 fail · 3 not_measured/missing) · 2 usage/ENV.
set -euo pipefail
case "${1:-}" in -h | --help | "") sed -n '/^# Usage:/,/^# Pins/p' "$0" | sed 's/^# \{0,1\}//'; [ -n "${1:-}" ] && exit 0 || exit 2 ;; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_FILE=$1; shift
NAME=qwen3-1.7b-q4km PORT=18473 OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME=${2:?--name wants a label}; shift 2 ;;
    --port) PORT=${2:?--port wants a number}; shift 2 ;;
    --out) OUT=${2:?--out wants a path}; shift 2 ;;
    *) printf 'serve_conformance_nightly: unknown argument %s\n' "$1" >&2; exit 2 ;;
  esac
done
[ -f "$MODEL_FILE" ] && [ -r "$MODEL_FILE" ] || { printf 'serve_conformance_nightly: ENV - model file %s unreadable\n' "$MODEL_FILE" >&2; exit 2; }
# shellcheck disable=SC1091
. "$ROOT/scripts/apr_bin.sh" || exit 2
T=$(mktemp -d) || exit 2
SRV=""
cleanup() { [ -z "$SRV" ] || { kill "$SRV" 2> /dev/null || true; wait "$SRV" 2> /dev/null || true; }; rm -rf -- "${T:?}"; }
trap cleanup EXIT
"$APR" serve run "$MODEL_FILE" --host 127.0.0.1 --port "$PORT" --no-gpu > "$T/serve.log" 2>&1 < /dev/null &
SRV=$!
up=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
  kill -0 "$SRV" 2> /dev/null || break
  sleep 1
done
[ "$up" = 1 ] || { printf 'serve_conformance_nightly: ENV - serve never answered /health (log tail: %s)\n' "$(tail -n 3 "$T/serve.log" | tr '\n' ' ')" >&2; exit 2; }
rc=0
bash "$ROOT/scripts/serve_conformance.sh" --base "http://127.0.0.1:$PORT" --model "$NAME" --backend cpu ${OUT:+--out "$OUT"} </dev/null || rc=$?
exit "$rc"
