#!/usr/bin/env bash
# edge_fast_verbs_nightly.sh: the edge-device LAB nightly (#5254). One model on
# one 8 GB unified-memory CUDA device, three measurements, one ranked gap list:
#
#   1. CRUX smoke   scripts/crux_inference_dogfood.sh, apr + ollama, every verb
#   2. apr decode   `apr bench --fast --json`, tok/s as apr reports it
#   3. ollama       the SAME file through /api/generate, raw, temperature 0,
#                   tok/s = eval_count / eval_duration as ollama reports it
#
# LAB, never a gate: no PR, merge-queue or release job runs this. A CRUX RED
# cell or a slow decode is a GAP and goes on the list; it is not a failure.
# The run FAILS only when one of the three could not be measured, because a
# list missing a row cannot be told apart from a list with nothing to say.
#
# Usage: bash scripts/edge_fast_verbs_nightly.sh --model <gguf> --out <dir>
#          [--host <id>] [--ollama-url <url>]
#   $APR  the apr binary under test (default target/release/apr)
#
# Writes <out>/gaps.md (the comment body) and <out>/nightly.json.
# Exit: 0 all three measured · 1 a measurement missing · 2 usage/tooling.
#
# Every status is captured as `cmd; rc=$?`, never through a pipe (#2336, #2360).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$ROOT" || exit 2

die() { printf 'edge-nightly: %s\n' "$*" >&2; exit 2; }

MODEL=""
OUT=""
HOST_ID="edge"
OLLAMA_URL="http://localhost:11434"
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --out) OUT="${2:-}"; shift 2 ;;
        --host) HOST_ID="${2:-}"; shift 2 ;;
        --ollama-url) OLLAMA_URL="${2:-}"; shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[ -n "$MODEL" ] || die "--model is required"
[ -f "$MODEL" ] || die "no model file: $MODEL"
[ -n "$OUT" ] || die "--out is required"
for t in jq curl sha256sum ollama; do
    command -v "$t" >/dev/null 2>&1 || die "missing tool: $t"
done
APR="${APR:-$ROOT/target/release/apr}"
[ -x "$APR" ] || die "no apr binary at $APR"
mkdir -p "$OUT" || die "cannot create $OUT"

MODEL_SHA=$(sha256sum "$MODEL") || die "sha256sum failed on $MODEL"
MODEL_SHA=${MODEL_SHA%% *}
MODEL_NAME=$(basename "$MODEL" .gguf)
QUANT=$(printf '%s' "$MODEL_NAME" | command grep -oE 'Q[0-9]_[A-Z0-9_]+$' || true)
APR_VERSION=$("$APR" --version 2>/dev/null) || die "$APR --version failed"
APR_SHA=$(sha256sum "$APR") || die "sha256sum failed on $APR"
APR_SHA=${APR_SHA%% *}
VERSION=$(printf '%s' "$APR_VERSION" | command grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
missing=()

# 1. CRUX smoke.
DOGFOOD_ALLOW_UNPINNED=1 APR="$APR" bash scripts/crux_inference_dogfood.sh "$VERSION" \
    --model "$MODEL" --host "$HOST_ID" --backend gpu --engines apr,ollama \
    --verbs run,chat,serve,code --out "$OUT/crux" > "$OUT/crux.log" 2>&1
crux_rc=$?
CRUX="$OUT/crux/$HOST_ID-gpu.json"
if [ ! -s "$CRUX" ] || ! jq -e '.cells | length > 0' "$CRUX" >/dev/null 2>&1; then
    missing+=("CRUX smoke (dogfood rc $crux_rc, no cells; see crux.log)")
    CRUX=""
fi

# 2. apr decode.
"$APR" bench "$MODEL" --fast --json --warmup 3 --iterations 5 --max-tokens 64 \
    > "$OUT/apr-bench.json" 2> "$OUT/apr-bench.err"
bench_rc=$?
APR_TPS=$(jq -r '.tokens_per_second // empty' "$OUT/apr-bench.json" 2>/dev/null || true)
[ -n "$APR_TPS" ] || missing+=("apr bench (rc $bench_rc, no tokens_per_second)")
# Name the backend apr actually used, not the one asked for.
APR_BACKEND=$(command grep -m1 -oE 'Backend: [^)]*\)?' "$OUT/apr-bench.err" || true)
APR_BACKEND=${APR_BACKEND#Backend: }
GPU_REFUSALS=$(command grep -c 'would not fit' "$OUT/apr-bench.err" || true)

# 3. ollama on the same file. The model is imported under a name this run owns
# and removed on every exit path.
OL_NAME="edge-nightly-${MODEL_SHA:0:12}"
cleanup() { ollama rm "$OL_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
printf 'FROM %s\n' "$MODEL" > "$OUT/Modelfile"
OL_TPS=""
if ollama create "$OL_NAME" -f "$OUT/Modelfile" > "$OUT/ollama-create.log" 2>&1; then
    body=$(jq -cn --arg m "$OL_NAME" '{model: $m, prompt: "Count from 1 to 40 separated by spaces.",
        raw: true, stream: false, options: {num_predict: 64, temperature: 0}}')
    for i in 1 2 3; do
        curl -s -m 300 "$OLLAMA_URL/api/generate" -d "$body" > "$OUT/ollama-$i.json"
    done
    OL_TPS=$(jq -s -r '[.[] | select(.eval_duration > 0) | .eval_count / (.eval_duration / 1e9)]
        | sort | if length == 3 then .[1] * 10 | round / 10 else empty end' \
        "$OUT"/ollama-[123].json 2>/dev/null || true)
fi
[ -n "$OL_TPS" ] || missing+=("ollama raw decode (no 3 measured generations; see ollama-*.json)")

RATIO=""
if [ -n "$APR_TPS" ] && [ -n "$OL_TPS" ]; then
    RATIO=$(jq -n --argjson a "$APR_TPS" --argjson o "$OL_TPS" '[$a, $o] | .[0] / .[1] | . * 100 | round / 100')
fi

# The ranked list: what costs the most cells first, then decode, then refusals.
{
    printf '**Edge nightly** · model `%s` · quant `%s` · sha256 `%s`\n' \
        "$MODEL_NAME" "${QUANT:-unknown}" "$MODEL_SHA"
    printf 'apr `%s` (binary sha256 `%s`) · backend: %s\n\n' \
        "$APR_VERSION" "${APR_SHA:0:12}" "${APR_BACKEND:-not reported}"
    printf '| measure | value |\n|---|---|\n'
    printf '| apr decode (`apr bench --fast`) | %s tok/s |\n' "${APR_TPS:-not_measured}"
    printf '| ollama decode, same file, raw | %s tok/s |\n' "${OL_TPS:-not_measured}"
    printf '| apr / ollama | %s |\n' "${RATIO:-not_measured}"
    printf '| batched prefill refusals in the bench | %s |\n' "${GPU_REFUSALS:-0}"
    if [ -n "$CRUX" ]; then
        jq -r '"| CRUX smoke verdict | \(.summary.verdict): \(.summary.GREEN // 0) GREEN, \(.summary.RED // 0) RED of \(.summary.cells) cells |"' "$CRUX"
    else
        printf '| CRUX smoke verdict | not_measured |\n'
    fi
    printf '\n### Ranked gaps\n\n'
    n=0
    # Per verb, the cells apr did NOT answer correctly, most first. A verdict
    # alone hides them: with no comparator answering, every cell is RED,
    # including the ones apr got right.
    if [ -n "$CRUX" ]; then
        while IFS= read -r line; do
            n=$((n + 1))
            printf '%d. %s\n' "$n" "$line"
        done < <(jq -r -f "$ROOT/scripts/lib/edge_nightly_gaps.jq" "$CRUX")
    fi
    if [ -n "$RATIO" ] && jq -e -n --argjson r "$RATIO" '$r < 1' >/dev/null; then
        n=$((n + 1))
        printf '%d. decode at %s× ollama (%s vs %s tok/s)\n' "$n" "$RATIO" "$APR_TPS" "$OL_TPS"
    fi
    if [ "${GPU_REFUSALS:-0}" -gt 0 ]; then
        n=$((n + 1))
        printf '%d. batched prefill refused %s times in the bench (one token at a time)\n' \
            "$n" "$GPU_REFUSALS"
    fi
    [ "$n" -gt 0 ] || printf 'none\n'
    if [ "${#missing[@]}" -gt 0 ]; then
        printf '\n### Not measured\n\n'
        printf -- '- %s\n' "${missing[@]}"
    fi
} > "$OUT/gaps.md"

jq -n --arg model "$MODEL_NAME" --arg quant "${QUANT:-}" --arg sha "$MODEL_SHA" \
    --arg apr "$APR_VERSION" --arg apr_sha "$APR_SHA" --arg backend "${APR_BACKEND:-}" \
    --arg apr_tps "${APR_TPS:-}" --arg ol_tps "${OL_TPS:-}" --arg ratio "${RATIO:-}" \
    --argjson missing "$(printf '%s\n' "${missing[@]+"${missing[@]}"}" | jq -R . | jq -s 'map(select(. != ""))')" \
    '{model: $model, quant: $quant, model_sha256: $sha, apr: $apr, apr_sha256: $apr_sha,
      backend: $backend, apr_tok_s: $apr_tps, ollama_tok_s: $ol_tps, ratio: $ratio,
      not_measured: $missing}' > "$OUT/nightly.json"

cat "$OUT/gaps.md"
[ "${#missing[@]}" -eq 0 ] || exit 1
exit 0
