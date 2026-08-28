#!/usr/bin/env bash
# PERF-041 - decompose serialization_index into the two factors it multiplies.
#
# batch-admission-v1 F-BATCH-001 records 1.00/2.45/2.29/2.39 at c=1/2/4/8 and
# reads the 2.45 as serialization at c=2. The shape argues otherwise: the index
# is FLAT from c=2, and wall(2) > wall(4) in the recorded numbers themselves
# (2.94s vs 2.75s) -- adding two clients cannot make a serialized server faster.
# A constant is being charged the moment batching engages.
#
#   serialization_index(c) = wall(c) / wall_fast(1)
#                          = [wall_batched(1)/wall_fast(1)] * [wall(c)/wall_batched(1)]
#                          =        path_penalty            *    scaling_index(c)
#
# Only scaling_index is serialization. wall_batched(1) is the one term no
# recorded run contains, because c=1 always takes the CUDA-graph fast path.
# APR_FORCE_BATCHED_PATH=1 (contract F-BATCH-004's own stated mutation, "force
# the batched path unconditionally") is how this script obtains it.
#
# The two c=1 bands are INTERLEAVED across replicates, not run in two blocks.
# gguf/cuda/generate_1.rs already documents a knob whose verdict INVERTS between
# a quiet box and load average 128; two blocks measure box drift as much as the
# change.
set -uo pipefail

. scripts/apr_bin.sh || exit 1
BIN="$APR"

MODEL="${APR_MODELS:-$HOME/models}/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
PORT="${PERF041_PORT:-8441}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"
OUT="${PERF041_OUT:-/tmp/perf041}"
REPS="${PERF041_REPS:-3}"          # §4.4.2: N = 3 full band runs per cell
MIN_SECONDS="${PERF041_MIN_SECONDS:-60}"  # §4.4.2: minimum wall-clock per band
MAX_TOKENS="${PERF041_MAX_TOKENS:-400}"
BANDS="${PERF041_BANDS:-1 2 4 8}"

mkdir -p "$OUT"

if [ ! -f "$MODEL" ]; then
    printf 'perf041: model not found: %s\n' "$MODEL" >&2
    exit 1
fi

SERVER_PID=""
stop_server() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        SERVER_PID=""
    fi
}
trap stop_server EXIT

# $1 = "fast" (production) or "forced" (F-BATCH-004 mutation armed)
start_server() {
    mode="$1"
    log="$OUT/server-${mode}.log"
    if [ "$mode" = "forced" ]; then
        APR_FORCE_BATCHED_PATH=1 "$BIN" serve run "$MODEL" --gpu --port "$PORT" \
            --context-length 4096 > "$log" 2>&1 &
    else
        "$BIN" serve run "$MODEL" --gpu --port "$PORT" \
            --context-length 4096 > "$log" 2>&1 &
    fi
    SERVER_PID=$!
    i=0
    while [ "$i" -lt 300 ]; do
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)
        if [ "$code" = "200" ]; then
            return 0
        fi
        i=$((i + 1))
        sleep 1
    done
    printf 'perf041: server (%s) never became healthy\n' "$mode" >&2
    return 1
}

# $1 = mode, $2 = concurrency, $3 = replicate index
run_band() {
    mode="$1"; c="$2"; rep="$3"
    label="${mode}-c${c}-r${rep}"
    # Status is read directly, NEVER through a pipe: `cmd | tee` reports tee's
    # exit code, which is how three green nightly runs once proved nothing.
    python3 scripts/perf041_client.py \
        --url "$URL" --concurrency "$c" --max-tokens "$MAX_TOKENS" \
        --min-seconds "$MIN_SECONDS" --label "$label" \
        > "$OUT/${label}.json" 2>"$OUT/${label}.err"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'perf041: band %s FAILED (rc=%d): %s\n' "$label" "$rc" \
            "$(cat "$OUT/${label}.err" 2>/dev/null)" >&2
        return 1
    fi
    printf '  %-16s %s\n' "$label" "$(cat "$OUT/${label}.json")"
    return 0
}

printf 'perf041: binary %s\n' "$BIN"
printf 'perf041: %s\n' "$("$BIN" --version 2>&1)"
printf 'perf041: model %s\n' "$MODEL"
printf 'perf041: reps=%s min_seconds=%s max_tokens=%s bands="%s"\n' \
    "$REPS" "$MIN_SECONDS" "$MAX_TOKENS" "$BANDS"
printf 'perf041: load at start: %s\n' "$(uptime)"
# Record the knobs that change what is being measured. CUDA_MAX_BATCH caps both
# the accumulation loop and max_kv_slots, so a band at c > CUDA_MAX_BATCH is
# measuring recycling, not batching, and the two must not be confused.
printf 'perf041: CUDA_MAX_BATCH=%s CUDA_BATCH_WINDOW_MS=%s ITERATION_SCHEDULER=%s BATCHED_GRAPH=%s STAGGERED_PREFILL=%s\n' \
    "${CUDA_MAX_BATCH:-unset(default 4)}" "${CUDA_BATCH_WINDOW_MS:-unset(default 0)}" \
    "${ITERATION_SCHEDULER:-unset(default off -> cuda_batch_scheduler)}" \
    "${BATCHED_GRAPH:-unset(default off -> eager)}" "${STAGGERED_PREFILL:-unset(default off)}"

rep=1
while [ "$rep" -le "$REPS" ]; do
    printf '=== replicate %d/%d ===\n' "$rep" "$REPS"

    # Interleave A (production) and B (forced) at c=1 within the replicate.
    start_server fast || exit 1
    for c in $BANDS; do
        run_band fast "$c" "$rep" || true
    done
    stop_server

    start_server forced || exit 1
    run_band forced 1 "$rep" || true
    stop_server

    rep=$((rep + 1))
done

printf 'perf041: load at end: %s\n' "$(uptime)"
python3 scripts/perf041_report.py "$OUT"
