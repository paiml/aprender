#!/usr/bin/env bash
# APR-071 V1 baseline — reference TTFT ÷ apr TTFT, with its 95% lower bound,
# on ONE GPU host, from the canonical client.
#
# This script measures nothing itself. It starts the two servers the band
# compares (the pinned llama.cpp and `apr serve run`, same host, same GGUF) and
# hands both URLs to `apr test llm bench --band --comparator-url`, which owns the
# timing, the join and the bootstrap (PERF-009: one entrypoint). The V1 figure
# is `bands[].ratios.ttft` in each `receipt.r<k>.json` it writes.
#
#   make v1-ttft-baseline HOST=lambda MODEL=/path/Qwen3.5-4B-Q4_K_M.gguf
#   bash scripts/v1_ttft_baseline.sh --host lambda --model <gguf> [--out <dir>] \
#        [--replicates N] [--duration S] [--dry-run]
#
# APR defaults to the binary scripts/apr_bin.sh proves was built from HEAD; set
# APR=<path> to measure another (the receipt then names that binary's sha256).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="" MODEL="" OUT="" REPLICATES=5 DURATION=60 DRY=0
APORT=18090 LPORT=18091 CTX=1024

while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --replicates) REPLICATES="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) printf 'FAIL  unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[ -n "$HOST" ] || { printf 'FAIL  --host is required: the receipt names the host\n' >&2; exit 2; }
[ -f "$MODEL" ] || { printf 'FAIL  --model %s: no such GGUF\n' "${MODEL:-<unset>}" >&2; exit 2; }
OUT="${OUT:-$ROOT/target/v1-ttft/$HOST}"

LLAMA_HOST="$HOST"
# shellcheck source=scripts/llama_bin.sh
. "$ROOT/scripts/llama_bin.sh" || { printf 'FAIL  pinned llama.cpp unresolved (rc=%s)\n' "$?" >&2; exit 1; }
[ -n "${LLAMA_SERVER:-}" ] || { printf 'FAIL  no llama-server beside the pinned build\n' >&2; exit 1; }
if [ -z "${APR:-}" ]; then
    # shellcheck source=scripts/apr_bin.sh
    . "$ROOT/scripts/apr_bin.sh" || exit 1
fi
lflags="$(llama_comparator_server_flags all 1)"
sha() { sha256sum "$1" | cut -d' ' -f1; }
APR_COMMIT="$("$APR" --version | sed -n 's/.*(\([0-9a-f]\{7,\}\)).*/\1/p' | head -1)"
[ -n "$APR_COMMIT" ] || { printf 'FAIL  %s --version names no commit; the receipt must name the commit under test\n' "$APR" >&2; exit 1; }
APR_COMMIT="$(git -C "$ROOT" rev-parse --verify --quiet "${APR_COMMIT}^{commit}")" || { printf 'FAIL  commit %s (from %s --version) is not in this clone; fetch it first\n' "$APR_COMMIT" "$APR" >&2; exit 1; }
MODEL_SHA="$(sha "$MODEL")"
LLAMA_SHA="$(sha "$LLAMA_SERVER")"
LLAMA_COMMIT="$(llama_pin_get build_commit)"

printf 'host       : %s\n' "$HOST"
printf 'gguf       : %s sha256=%s\n' "$MODEL" "$MODEL_SHA"
printf 'comparator : %s (pin %s, sha256=%s) %s\n' "$LLAMA_SERVER" "$LLAMA_COMMIT" "$LLAMA_SHA" "$lflags"
printf 'subject    : %s (commit %s, sha256=%s) serve run --gpu-layers all --context-length %s\n' "$APR" "$APR_COMMIT" "$(sha "$APR")" "$CTX"
printf 'band       : c=1, %s replicates x %s s, receipt dir %s\n' "$REPLICATES" "$DURATION" "$OUT"
[ "$DRY" -eq 0 ] || exit 0

mkdir -p "$OUT"
PIDS=""
stop_servers() {
    local p
    for p in $PIDS; do kill "$p" 2>/dev/null || true; done
    for p in $PIDS; do wait "$p" 2>/dev/null || true; done
}
trap stop_servers EXIT

wait_healthy() { # wait_healthy <port> <limit>
    local i=0
    until curl -sf "http://127.0.0.1:$1/health" > /dev/null 2>&1; do
        i=$((i + 1)); [ "$i" -lt "$2" ] || return 1; sleep 1
    done
}

# shellcheck disable=SC2086
"$LLAMA_SERVER" -m "$MODEL" --port "$LPORT" $lflags > "$OUT/llama-server.log" 2>&1 &
PIDS="$PIDS $!"
wait_healthy "$LPORT" 300 || { printf 'FAIL  llama-server did not become healthy\n' >&2; exit 1; }
nbatch=$(sed -n 's/.*n_batch[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$OUT/llama-server.log" | head -1)

"$APR" serve run "$MODEL" --gpu-layers all --port "$APORT" --context-length "$CTX" > "$OUT/apr-serve.log" 2>&1 &
PIDS="$PIDS $!"
wait_healthy "$APORT" 300 || { printf 'FAIL  apr serve did not become healthy\n' >&2; exit 1; }

set +e
"$APR" test llm bench --url "http://127.0.0.1:$APORT" --band --stream --bands 1 \
    --replicates "$REPLICATES" --duration "$DURATION" --receipt "$OUT" \
    --workload W1 --prompts "$ROOT/crates/aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl" \
    --commit "$APR_COMMIT" --host "$HOST" --accelerator cuda --compute-class cuda --quantization Q4_K_M \
    --subject-binary "$APR" \
    --comparator-url "http://127.0.0.1:$LPORT" --comparator-owner llama.cpp \
    --comparator-commit "$LLAMA_COMMIT" --comparator-sha256 "$LLAMA_SHA" \
    --comparator-cmake "$(llama_pin_get "build_flags_$HOST")" --comparator-pin-expiry "$(llama_pin_get pin_expiry)T00:00:00.000Z" \
    --comparator-n-batch "${nbatch:-2048}" --comparator-n-ctx-slot "$CTX" \
    --comparator-fa auto --comparator-kv-type f16 \
    > "$OUT/bench.log" 2>&1
rc=$?
set -e
printf 'bench rc=%s (log %s/bench.log)\n' "$rc" "$OUT"
for r in "$OUT"/receipt.r*.json; do
    [ -f "$r" ] || continue
    printf '%s: ' "$(basename "$r")"
    jq -c '[.bands[] | {c: .concurrency, comparator_status, ttft: .ratios.ttft,
            apr_ttft_p50_ms: .ttft_p50_ms, ref_ttft_p50_ms: .baseline.ttft_p50_ms}]' "$r"
done
exit "$rc"
