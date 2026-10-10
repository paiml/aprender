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
#        [--replicates N] [--duration S] [--apr-port N] [--llama-port N] [--dry-run]
#
# Refused before anything starts (exit 2), --dry-run included:
#   - --host is not the host scripts/llama_bin.sh resolves the pin for
#     (LLAMA_PIN_HOST, else this machine's hostname);
#   - --out is not empty: an earlier run's receipt.r*.json would be read as
#     this run's evidence;
#   - a port already answers: /health would answer from that process, not
#     from the server this run starts.
#
# APR (the bench client) defaults to the binary scripts/apr_bin.sh proves was
# built from HEAD. APR_SERVE (the subject under test) defaults to APR; set it to
# measure another binary, e.g. a CUDA build. The receipt names the SUBJECT's
# commit and sha256, never the client's.
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
        --apr-port) APORT="$2"; shift 2 ;;
        --llama-port) LPORT="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) printf 'FAIL  unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[ -n "$HOST" ] || { printf 'FAIL  --host is required: the receipt names the host\n' >&2; exit 2; }
[ -f "$MODEL" ] || { printf 'FAIL  --model %s: no such GGUF\n' "${MODEL:-<unset>}" >&2; exit 2; }
OUT="${OUT:-$ROOT/target/v1-ttft/$HOST}"

# Every receipt in $OUT is read as this run's evidence, so $OUT starts empty.
if [ -d "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    printf 'FAIL  --out %s is not empty: receipts left by an earlier run would be read as evidence of this one; pass a new --out\n' "$OUT" >&2
    exit 2
fi

# A port that already answers is held by a process this run did not start.
command -v curl > /dev/null || { printf 'FAIL  curl not found: the health checks need it\n' >&2; exit 2; }
port_free() { # port_free <port>: curl exit 7 = nothing listening
    local rc=0
    curl -s --max-time 2 -o /dev/null "http://127.0.0.1:$1/" > /dev/null 2>&1 || rc=$?
    [ "$rc" -eq 7 ]
}
for p in "$APORT" "$LPORT"; do
    case "$p" in ''|*[!0-9]*) printf 'FAIL  port %s is not a number\n' "$p" >&2; exit 2 ;; esac
    port_free "$p" || { printf 'FAIL  port %s already answers: stop that process or pass another --apr-port/--llama-port\n' "$p" >&2; exit 2; }
done
[ "$APORT" != "$LPORT" ] || { printf 'FAIL  --apr-port and --llama-port are both %s\n' "$APORT" >&2; exit 2; }

# The pin and its cmake line are resolved for the host llama_bin.sh names, so
# the receipt's --host must be that host.
pin_rc=0
# shellcheck source=scripts/llama_bin.sh
. "$ROOT/scripts/llama_bin.sh" || pin_rc=$?
pin_host="$(llama_pin_host)" || pin_host=""
[ "$pin_host" = "$HOST" ] || {
    printf 'FAIL  --host %s, but scripts/llama_bin.sh resolved the pin for host %s (LLAMA_PIN_HOST, else hostname %s)\n' \
        "$HOST" "${pin_host:-<unknown>}" "$(hostname 2>/dev/null)" >&2
    exit 2
}
[ "$pin_rc" -eq 0 ] || { printf 'FAIL  pinned llama.cpp unresolved (rc=%s, %s)\n' "$pin_rc" "${LLAMA_PIN_REASON:-unknown}" >&2; exit 1; }
[ -n "${LLAMA_SERVER:-}" ] || { printf 'FAIL  no llama-server beside the pinned build\n' >&2; exit 1; }
if [ -z "${APR:-}" ]; then
    # shellcheck source=scripts/apr_bin.sh
    . "$ROOT/scripts/apr_bin.sh" || exit 1
fi
APR_SERVE="${APR_SERVE:-$APR}"
lflags="$(llama_comparator_server_flags all 1)"
sha() { sha256sum "$1" | cut -d' ' -f1; }
APR_COMMIT="$("$APR_SERVE" --version | sed -n 's/.*(\([0-9a-f]\{7,\}\)).*/\1/p' | head -1)"
[ -n "$APR_COMMIT" ] || { printf 'FAIL  %s --version names no commit; the receipt must name the commit under test\n' "$APR_SERVE" >&2; exit 1; }
APR_COMMIT="$(git -C "$ROOT" rev-parse --verify --quiet "${APR_COMMIT}^{commit}")" || { printf 'FAIL  commit %s (from %s --version) is not in this clone; fetch it first\n' "$APR_COMMIT" "$APR_SERVE" >&2; exit 1; }
MODEL_SHA="$(sha "$MODEL")"
LLAMA_SHA="$(sha "$LLAMA_SERVER")"
LLAMA_COMMIT="$(llama_pin_get build_commit)"

printf 'host       : %s\n' "$HOST"
printf 'gguf       : %s sha256=%s\n' "$MODEL" "$MODEL_SHA"
printf 'comparator : %s (pin %s, sha256=%s) %s\n' "$LLAMA_SERVER" "$LLAMA_COMMIT" "$LLAMA_SHA" "$lflags"
printf 'subject    : %s (commit %s, sha256=%s) serve run --gpu-layers all --context-length %s\n' "$APR_SERVE" "$APR_COMMIT" "$(sha "$APR_SERVE")" "$CTX"
printf 'client     : %s (sha256=%s)\n' "$APR" "$(sha "$APR")"
printf 'band       : c=1, %s replicates x %s s, receipt dir %s\n' "$REPLICATES" "$DURATION" "$OUT"
[ "$DRY" -eq 0 ] || exit 0

mkdir -p "$OUT"
# The receipt names the subject, the client and the comparator, but not the
# GGUF: its model field is the server's alias. This sidecar binds every receipt
# in $OUT to the file both servers loaded and to the interval method.
jq -n --arg host "$HOST" --arg model "$MODEL" --arg model_sha "$MODEL_SHA" \
    --arg llama "$LLAMA_SERVER" --arg llama_commit "$LLAMA_COMMIT" --arg llama_sha "$LLAMA_SHA" \
    --arg lflags "$lflags" --arg subject "$APR_SERVE" --arg subject_commit "$APR_COMMIT" \
    --arg subject_sha "$(sha "$APR_SERVE")" --arg client "$APR" --arg client_sha "$(sha "$APR")" \
    --argjson replicates "$REPLICATES" --argjson duration "$DURATION" --argjson ctx "$CTX" \
    '{host: $host, gguf: {path: $model, sha256: $model_sha},
      comparator: {binary: $llama, pin: $llama_commit, sha256: $llama_sha, flags: $lflags},
      subject: {binary: $subject, commit: $subject_commit, sha256: $subject_sha,
                args: "serve run --gpu-layers all --context-length \($ctx)"},
      client: {binary: $client, sha256: $client_sha},
      band: {concurrency: 1, replicates: $replicates, duration_s: $duration},
      interval: {field: "bands[].ratios.ttft", ratio: "comparator ttft_p50 / subject ttft_p50",
                 method: "paired_percentile_bootstrap", resamples: 10000, seed: 2026,
                 bound: "one-sided 95% lower (5th percentile)"}}' > "$OUT/v1-provenance.json"
PIDS=""
stop_servers() {
    local p
    for p in $PIDS; do kill "$p" 2>/dev/null || true; done
    for p in $PIDS; do wait "$p" 2>/dev/null || true; done
}
trap stop_servers EXIT

wait_healthy() { # wait_healthy <port> <pid> <limit>: the server this run started answers
    local i=0
    until curl -sf --max-time 5 "http://127.0.0.1:$1/health" > /dev/null 2>&1; do
        kill -0 "$2" 2> /dev/null || { printf 'FAIL  pid %s exited before port %s answered\n' "$2" "$1" >&2; return 1; }
        i=$((i + 1)); [ "$i" -lt "$3" ] || return 1; sleep 1
    done
    kill -0 "$2" 2> /dev/null || { printf 'FAIL  pid %s exited, yet port %s answers\n' "$2" "$1" >&2; return 1; }
}

# shellcheck disable=SC2086
"$LLAMA_SERVER" -m "$MODEL" --port "$LPORT" $lflags > "$OUT/llama-server.log" 2>&1 &
lpid=$!
PIDS="$PIDS $lpid"
wait_healthy "$LPORT" "$lpid" 300 || { printf 'FAIL  llama-server did not become healthy\n' >&2; exit 1; }
nbatch=$(sed -n 's/.*n_batch[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$OUT/llama-server.log" | head -1)

"$APR_SERVE" serve run "$MODEL" --gpu-layers all --port "$APORT" --context-length "$CTX" > "$OUT/apr-serve.log" 2>&1 &
apid=$!
PIDS="$PIDS $apid"
wait_healthy "$APORT" "$apid" 300 || { printf 'FAIL  apr serve did not become healthy\n' >&2; exit 1; }

set +e
"$APR" test llm bench --url "http://127.0.0.1:$APORT" --band --stream --bands 1 \
    --replicates "$REPLICATES" --duration "$DURATION" --receipt "$OUT" \
    --workload W1 --prompts "$ROOT/crates/aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl" \
    --commit "$APR_COMMIT" --host "$HOST" --accelerator cuda --compute-class cuda --quantization Q4_K_M \
    --subject-binary "$APR_SERVE" \
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
