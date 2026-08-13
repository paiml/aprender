#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# SHIP-TWO-001 Parallel Eval Lane — orchestrator
# ─────────────────────────────────────────────────────────────
# Contract:   contracts/eval-sharding-v1.yaml
# Spec ref:   docs/specifications/aprender-train/ship-two-models-spec.md §12.6
# Discharges: AC-EX-007, FALSIFY-SHARD-001/002
#
# Splits a benchmark.jsonl into N stride-shards, rsyncs model + shard to each
# host, dispatches eval-pass-at-k.sh in parallel over ssh, fetches per-shard
# JSON results, hands off to eval-shard-merge.py for merge + FALSIFY-SHARD
# gates.
#
# Usage:
#   HOSTS="yoga gx10"  MODEL=/path/to/model.apr \
#     bash scripts/ship-two-001/eval-shard.sh humaneval
#
# Environment:
#   HOSTS              space-separated ssh aliases (first entry = localhost by
#                      convention). "yoga" treated as localhost unless
#                      LOCALHOST_ALIAS=<other>. REQUIRED.
#   MODEL              absolute path to .apr on the orchestrator. REQUIRED.
#   TOKENIZER          absolute path to tokenizer.json (auto-derived from
#                      MODEL if sibling .tokenizer.json exists).
#   TEMPERATURE        default 0.0 (required for FALSIFY-SHARD-003 parity).
#   MAX_TOKENS         default 512.
#   NUM_SAMPLES        default 1.
#   RESULTS_DIR        default evidence/ship-two-001/shard-eval
#   REMOTE_WORKDIR     default /tmp/apr-shard-eval.
#   LOCALHOST_ALIAS    default "yoga" — host treated as local (no ssh needed).
#   DRY_RUN            1 = print plan + exit without dispatch.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# The LOCAL shard runs `apr run --batch-jsonl` directly; remote shards go
# through eval-pass-at-k.sh on the far host. The local invocation was bare, so
# a benchmark number attributed to this commit could come from any apr on the
# orchestrator's PATH - and FALSIFY-SHARD-003 compares shards for parity, which
# a version skew between local and remote would quietly break (#2358).
. "$(dirname "$0")/../apr_bin.sh" || exit 1

BENCHMARK="${1:?Usage: eval-shard.sh BENCHMARK (humaneval|mbpp|bigcodebench)}"

: "${HOSTS:?HOSTS env var required (space-separated ssh aliases)}"
: "${MODEL:?MODEL env var required (abs path to .apr)}"

TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
RESULTS_DIR="${RESULTS_DIR:-evidence/ship-two-001/shard-eval}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/tmp/apr-shard-eval}"
LOCALHOST_ALIAS="${LOCALHOST_ALIAS:-yoga}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$MODEL" ]] || { echo "ERROR: MODEL not found: $MODEL" >&2; exit 1; }

# Tokenizer: look for sibling .tokenizer.json if not provided
if [[ -z "${TOKENIZER:-}" ]]; then
    AUTO_TOK="${MODEL%.apr}.tokenizer.json"
    if [[ -f "$AUTO_TOK" ]]; then
        TOKENIZER="$AUTO_TOK"
    fi
fi

# Host list
read -r -a HOST_ARR <<< "$HOSTS"
N="${#HOST_ARR[@]}"
if (( N < 1 )); then
    echo "ERROR: HOSTS must list at least one entry" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SHARD_DIR="${RESULTS_DIR}/run_${TIMESTAMP}"
mkdir -p "$SHARD_DIR"

# ── Download benchmark locally (reuses apr-leaderboard's cache layout) ───────
BENCHMARK_DIR="data/benchmarks"
mkdir -p "$BENCHMARK_DIR"
BENCH_FILE="${BENCHMARK_DIR}/${BENCHMARK}.jsonl"

if [[ ! -f "$BENCH_FILE" ]]; then
    case "$BENCHMARK" in
        humaneval)
            echo "[shard] downloading HumanEval..."
            curl -sfL "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz" \
                | gunzip > "$BENCH_FILE"
            ;;
        mbpp)
            echo "[shard] downloading MBPP..."
            curl -sfL "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl" \
                -o "${BENCH_FILE}.tmp"
            # Filter to standard test split
            jq -c 'select(.task_id >= 11 and .task_id <= 510)' "${BENCH_FILE}.tmp" > "$BENCH_FILE"
            rm -f "${BENCH_FILE}.tmp"
            ;;
        *)
            echo "ERROR: auto-download not implemented for $BENCHMARK (pre-stage manually in $BENCH_FILE)" >&2
            exit 1
            ;;
    esac
fi

TOTAL="$(wc -l < "$BENCH_FILE")"
echo "[shard] benchmark=$BENCHMARK total_tasks=$TOTAL N_hosts=$N temperature=$TEMPERATURE"

# ── Phase A: split benchmark into N stride shards ────────────────────────────
# Round-robin: line i (0-indexed) goes to shard (i mod N)
awk -v N="$N" -v DIR="$SHARD_DIR" '{
    shard = NR - 1
    print > (DIR "/shard_" (shard % N) ".jsonl")
}' "$BENCH_FILE"

for i in $(seq 0 $((N - 1))); do
    SHARD_FILE="${SHARD_DIR}/shard_${i}.jsonl"
    [[ -f "$SHARD_FILE" ]] || { echo "ERROR: missing shard $SHARD_FILE" >&2; exit 1; }
    COUNT=$(wc -l < "$SHARD_FILE")
    echo "[shard] shard_$i host=${HOST_ARR[$i]} tasks=$COUNT"
done

# Completeness sanity check (FALSIFY-SHARD-001 pre-flight)
SUM_CHECK=0
for i in $(seq 0 $((N - 1))); do
    SUM_CHECK=$((SUM_CHECK + $(wc -l < "${SHARD_DIR}/shard_${i}.jsonl")))
done
if [[ "$SUM_CHECK" -ne "$TOTAL" ]]; then
    echo "PRE-FLIGHT FAIL: sum(shard_i)=$SUM_CHECK != TOTAL=$TOTAL (FALSIFY-SHARD-001)" >&2
    exit 2
fi
echo "[shard] pre-flight completeness OK: sum(shard_i)=$SUM_CHECK == TOTAL=$TOTAL"

# ── Phase B: dispatch plan ───────────────────────────────────────────────────
MODEL_BASENAME="$(basename "$MODEL")"
TOK_BASENAME=""
[[ -n "${TOKENIZER:-}" ]] && TOK_BASENAME="$(basename "$TOKENIZER")"

echo ""
echo "=== Dispatch Plan ==="
for i in $(seq 0 $((N - 1))); do
    HOST="${HOST_ARR[$i]}"
    IS_LOCAL=0
    [[ "$HOST" == "$LOCALHOST_ALIAS" || "$HOST" == "localhost" ]] && IS_LOCAL=1
    echo "  shard_$i → $HOST (local=$IS_LOCAL) workdir=$REMOTE_WORKDIR"
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "[DRY_RUN=1] exiting before dispatch"
    exit 0
fi

# ── Phase C: rsync + ssh dispatch (parallel) ─────────────────────────────────
PID_FILE="${SHARD_DIR}/pids"
: > "$PID_FILE"
LOG_DIR="${SHARD_DIR}/logs"
mkdir -p "$LOG_DIR"

dispatch_one() {
    local i="$1"
    local host="${HOST_ARR[$i]}"
    local shard_file="${SHARD_DIR}/shard_${i}.jsonl"
    local is_local=0
    [[ "$host" == "$LOCALHOST_ALIAS" || "$host" == "localhost" ]] && is_local=1
    local log="${LOG_DIR}/shard_${i}.log"
    local remote_shard="${REMOTE_WORKDIR}/shard_${i}.jsonl"
    local remote_model="${REMOTE_WORKDIR}/${MODEL_BASENAME}"
    local remote_tok=""
    [[ -n "$TOK_BASENAME" ]] && remote_tok="${REMOTE_WORKDIR}/${TOK_BASENAME}"
    local remote_out="${REMOTE_WORKDIR}/result_shard_${i}.json"

    {
        echo "[$host] mkdir workdir"
        if (( is_local )); then
            mkdir -p "$REMOTE_WORKDIR"
        else
            ssh "$host" "mkdir -p '$REMOTE_WORKDIR'"
        fi

        echo "[$host] rsync model (content-checksum)"
        if (( is_local )); then
            cp -u "$MODEL" "$remote_model"
            [[ -n "$TOKENIZER" ]] && cp -u "$TOKENIZER" "$remote_tok"
        else
            rsync -c --partial --inplace "$MODEL" "$host:$remote_model"
            [[ -n "$TOKENIZER" ]] && rsync -c --partial --inplace "$TOKENIZER" "$host:$remote_tok"
        fi

        echo "[$host] rsync shard"
        if (( is_local )); then
            cp "$shard_file" "$remote_shard"
        else
            rsync "$shard_file" "$host:$remote_shard"
        fi

        echo "[$host] run apr batch"
        # Per-shard scoring happens in the merge step; here we just capture
        # stdout from `apr run --batch-jsonl` (one JSON per completion).
        local raw_out="${REMOTE_WORKDIR}/completions_shard_${i}.jsonl"
        if (( is_local )); then
            "$APR" run "$remote_model" \
                --batch-jsonl "$remote_shard" \
                --max-tokens "$MAX_TOKENS" \
                --temperature "$TEMPERATURE" \
                --top-k 1 > "$raw_out" 2>&1
        else
            # Remote command: paths are from our orchestrator (not user input),
            # quoted via printf %q to survive one SSH shell pass.
            local q_model q_shard q_out
            q_model="$(printf '%q' "$remote_model")"
            q_shard="$(printf '%q' "$remote_shard")"
            q_out="$(printf '%q' "$raw_out")"
            ssh "$host" "apr run $q_model --batch-jsonl $q_shard --max-tokens $MAX_TOKENS --temperature $TEMPERATURE --top-k 1 > $q_out 2>&1"
        fi

        echo "[$host] fetch completions"
        if (( is_local )); then
            cp "$raw_out" "${SHARD_DIR}/completions_shard_${i}.jsonl"
        else
            rsync "$host:$raw_out" "${SHARD_DIR}/completions_shard_${i}.jsonl"
        fi
    } > "$log" 2>&1
}

echo ""
echo "=== Dispatching $N shards ==="
PIDS=()
for i in $(seq 0 $((N - 1))); do
    dispatch_one "$i" &
    P=$!
    PIDS+=("$P")
    echo "$i $P ${HOST_ARR[$i]}" >> "$PID_FILE"
    echo "  dispatched shard_$i → ${HOST_ARR[$i]} (pid=$P)"
done

echo ""
echo "=== Waiting for shards ==="
FAILURES=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "  shard_$i: FAILED (see ${LOG_DIR}/shard_${i}.log)"
        FAILURES=$((FAILURES + 1))
    else
        echo "  shard_$i: OK"
    fi
done

if (( FAILURES > 0 )); then
    echo "$FAILURES shard(s) failed; inspect $LOG_DIR" >&2
    exit 3
fi

# ── Phase D: merge ───────────────────────────────────────────────────────────
echo ""
echo "=== Merging per-shard completions ==="
MERGED="${SHARD_DIR}/humaneval_merged_${TIMESTAMP}.json"

# Compose per-shard eval-pass-at-k.sh result JSONs by re-running its test+score
# phase on each shard's completions. We reuse the existing harness by feeding
# it the pre-generated completions via APR_BATCH_MODE=off and a pre-seeded
# completions dir — simpler path here: defer full test+score to the merge
# script, which can dispatch python test execution on each completion.
python3 "$(dirname "$0")/eval-shard-merge.py" \
    --benchmark-jsonl "$BENCH_FILE" \
    --shard-completions "${SHARD_DIR}/completions_shard_"*.jsonl \
    --shard-dir "$SHARD_DIR" \
    --benchmark "$BENCHMARK" \
    --output "$MERGED"

echo ""
echo "=== Shard run complete ==="
echo "Merged result: $MERGED"
echo "Evidence dir:  $SHARD_DIR"
