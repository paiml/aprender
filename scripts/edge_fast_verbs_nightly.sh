#!/usr/bin/env bash
# edge_fast_verbs_nightly.sh: the edge-device LAB nightly (#5254). One model on
# one 8 GB unified-memory CUDA device, three measurements, one ranked gap list:
#
#   1. CRUX smoke   scripts/crux_inference_dogfood.sh, apr + ollama, every verb
#   2. apr decode   `apr bench --fast --json`, tok/s as apr reports it
#   3. ollama       the in-tree comparator, `apr qa`'s ollama_parity gate. This
#                   script times no engine itself (PERF-009: one harness); when
#                   that gate cannot measure the model, its stated reason is
#                   the gap
#
# LAB, never a gate: no PR, merge-queue or release job runs this. A CRUX RED
# cell or a slow decode is a GAP and goes on the list; it is not a failure.
# The run FAILS only when one of the three could not be measured, because a
# list missing a row cannot be told apart from a list with nothing to say.
#
# Usage: bash scripts/edge_fast_verbs_nightly.sh --model <gguf> --out <dir>
#          [--host <id>]   (ollama: apr qa probes localhost:11434)
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
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --out) OUT="${2:-}"; shift 2 ;;
        --host) HOST_ID="${2:-}"; shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[ -n "$MODEL" ] || die "--model is required"
[ -f "$MODEL" ] || die "no model file: $MODEL"
[ -n "$OUT" ] || die "--out is required"
case "$OUT" in *..*) die "--out must not contain '..': $OUT" ;; esac
for t in jq sha256sum ollama; do
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

# 3. ollama, through `apr qa`'s ollama_parity gate and nothing else. Every
# other qa gate is skipped: this asks one question. A skip or a failure is a
# gap; only a qa JSON with no ollama_parity gate in it is "not measured".
"$APR" qa "$MODEL" --json --skip-golden --skip-throughput \
    --skip-gpu-speedup --skip-contract --skip-format-parity --skip-ptx-parity \
    --skip-gpu-state --skip-metadata --skip-capability \
    > "$OUT/apr-qa.json" 2> "$OUT/apr-qa.err"
qa_rc=$?
OL_STATE=$(jq -r '.gates[] | select(.name == "ollama_parity")
    | if .skipped then "skipped" elif .passed then "passed" else "failed" end' \
    "$OUT/apr-qa.json" 2>/dev/null || true)
OL_MSG=$(jq -r '.gates[] | select(.name == "ollama_parity") | .message // ""' \
    "$OUT/apr-qa.json" 2>/dev/null || true)
[ -n "$OL_STATE" ] || missing+=("ollama parity (qa rc $qa_rc, no ollama_parity gate in its JSON; see apr-qa.err)")

# The ranked list: what costs the most cells first, then decode, then refusals.
{
    printf '**Edge nightly** · model `%s` · quant `%s` · sha256 `%s`\n' \
        "$MODEL_NAME" "${QUANT:-unknown}" "$MODEL_SHA"
    printf 'apr `%s` (binary sha256 `%s`) · backend: %s\n\n' \
        "$APR_VERSION" "${APR_SHA:0:12}" "${APR_BACKEND:-not reported}"
    printf '| measure | value |\n|---|---|\n'
    printf '| apr decode (`apr bench --fast`) | %s tok/s |\n' "${APR_TPS:-not_measured}"
    printf '| ollama parity (`apr qa`) | %s |\n' "${OL_STATE:-not_measured}"
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
    if [ -n "$OL_STATE" ] && [ "$OL_STATE" != passed ]; then
        n=$((n + 1))
        printf '%d. ollama comparison: `apr qa` ollama_parity %s (%s)\n' "$n" "$OL_STATE" "$OL_MSG"
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
    --arg apr_tps "${APR_TPS:-}" --arg ol_state "${OL_STATE:-}" --arg ol_msg "${OL_MSG:-}" \
    --argjson missing "$(printf '%s\n' "${missing[@]+"${missing[@]}"}" | jq -R . | jq -s 'map(select(. != ""))')" \
    '{model: $model, quant: $quant, model_sha256: $sha, apr: $apr, apr_sha256: $apr_sha,
      backend: $backend, apr_tok_s: $apr_tps,
      ollama_parity: {state: $ol_state, message: $ol_msg},
      not_measured: $missing}' > "$OUT/nightly.json"

cat "$OUT/gaps.md"
[ "${#missing[@]}" -eq 0 ] || exit 1
exit 0
