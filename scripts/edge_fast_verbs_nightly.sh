#!/usr/bin/env bash
# edge_fast_verbs_nightly.sh: the edge-device LAB nightly (#5254). One model on
# one 8 GB unified-memory CUDA device, six measurements, one ranked gap list:
#
#   0. template     the GGUF must carry tokenizer.chat_template (apr inspect)
#   1. CRUX smoke   scripts/crux_inference_dogfood.sh, apr + ollama, every verb
#                   on gpu AND cpu; a verb x device x engine that answered
#                   nothing is a missing measurement (edge_nightly_matrix.jq)
#   2. apr decode   `apr bench --fast --json`, tok/s as apr reports it
#   3. ollama       the in-tree comparator, `apr qa`'s ollama_parity gate, run
#                   with --ollama-from-file so ollama decodes THIS file. This
#                   script times no engine itself (PERF-009: one harness); when
#                   that gate cannot measure the model, its stated reason is
#                   the gap. The ratio is printed against parity, 1.0, never
#                   against the gate's 0.2 floor
#   4. apr CPU      `apr bench --no-gpu --json` (#5276): the same harness, CPU
#                   only; a binary without the flag is a stated gap
#   5. apr profile  its throughput, or its stated refusal as a gap
#
# Every number is apr's own (inspect, crux, bench, qa, profile), stamped with the apr
# version and binary sha256 in the header line.
#
# LAB, never a gate: no PR, merge-queue or release job runs this. A CRUX RED
# cell, a slow decode or a ratio under 1.0 is a GAP and goes on the list; it is
# not a failure. The run FAILS only when a measurement is missing, because a
# list missing a row cannot be told apart from a list with nothing to say.
#
# Usage: bash scripts/edge_fast_verbs_nightly.sh --model <gguf> --out <dir>
#          [--host <id>]   (ollama: apr qa probes localhost:11434)
#   $APR  the apr binary under test (default target/release/apr)
#
# Writes <out>/gaps.md (the comment body) and <out>/nightly.json.
# Exit: 0 all measured · 1 a measurement missing · 2 usage/tooling.
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

# 0. The model's own chat template. A GGUF without one leaves every runtime
# that reads the file alone (ollama) completing raw text; that is a missing
# measurement of the chat verbs, never a quiet RED (#5254).
"$APR" inspect "$MODEL" --json > "$OUT/apr-inspect.json" 2> "$OUT/apr-inspect.err"
inspect_rc=$?
if ! jq -e '[.. | objects | .["tokenizer.chat_template"]? // empty | strings
        | select(test("\\S"))] | length > 0' "$OUT/apr-inspect.json" >/dev/null 2>&1; then
    missing+=("chat template (inspect rc $inspect_rc: the GGUF carries no tokenizer.chat_template)")
fi

# 1. CRUX smoke, every verb on each device. On cpu the dogfood asks ollama for
# num_gpu 0 as well, so its ollama cells are the ollama CPU rows.
CRUX_GPU=""
CRUX_CPU=""
for dev in gpu cpu; do
    DOGFOOD_ALLOW_UNPINNED=1 APR="$APR" bash scripts/crux_inference_dogfood.sh "$VERSION" \
        --model "$MODEL" --host "$HOST_ID" --backend "$dev" --engines apr,ollama \
        --verbs run,chat,serve,code --out "$OUT/crux" > "$OUT/crux-$dev.log" 2>&1
    crux_rc=$?
    f="$OUT/crux/$HOST_ID-$dev.json"
    if [ ! -s "$f" ] || ! jq -e '.cells | length > 0' "$f" >/dev/null 2>&1; then
        missing+=("CRUX smoke on $dev (dogfood rc $crux_rc, no cells; see crux-$dev.log)")
        continue
    fi
    if [ "$dev" = gpu ]; then CRUX_GPU="$f"; else CRUX_CPU="$f"; fi
done
CRUX="$CRUX_GPU"
# A verb x device x engine that answered nothing is an empty output: missing.
MATRIX_ARGS=(--slurpfile gpu "${CRUX_GPU:-/dev/null}" --slurpfile cpu "${CRUX_CPU:-/dev/null}")
while IFS= read -r line; do
    [ -n "$line" ] && missing+=("$line")
done < <(jq -nr "${MATRIX_ARGS[@]}" --arg zero 1 -f "$ROOT/scripts/lib/edge_nightly_matrix.jq")

# 2. apr decode.
"$APR" bench "$MODEL" --fast --json --warmup 3 --iterations 5 --max-tokens 64 \
    > "$OUT/apr-bench.json" 2> "$OUT/apr-bench.err"
bench_rc=$?
APR_TPS=$(jq -r '.tokens_per_second // empty | select(. > 0)' "$OUT/apr-bench.json" 2>/dev/null || true)
[ -n "$APR_TPS" ] || missing+=("apr bench (rc $bench_rc, no tokens_per_second)")
APR_TTFT=$(jq -r '.time_to_first_token_ms // empty | select(. > 0) | . * 10 | round / 10' "$OUT/apr-bench.json" 2>/dev/null || true)
[ -n "$APR_TTFT" ] || missing+=("apr TTFT (bench rc $bench_rc, no non-zero time_to_first_token_ms)")
# The kernel's high-water mark for the bench process (VmHWM). Zero or absent
# (a binary that predates the field) is a missing memory row.
APR_RSS=$(jq -r '.peak_rss_mb // empty | select(. > 0) | round' "$OUT/apr-bench.json" 2>/dev/null || true)
[ -n "$APR_RSS" ] || missing+=("apr peak RSS (bench rc $bench_rc, no non-zero peak_rss_mb)")
# Name the backend apr actually used, not the one asked for.
APR_BACKEND=$(command grep -m1 -oE 'Backend: [^)]*\)?' "$OUT/apr-bench.err" || true)
APR_BACKEND=${APR_BACKEND#Backend: }
GPU_REFUSALS=$(command grep -c 'would not fit' "$OUT/apr-bench.err" || true)

# 3. ollama, through `apr qa`'s ollama_parity gate and nothing else. Every
# other qa gate is skipped: this asks one question. A skip or a failure is a
# gap; only a qa JSON with no ollama_parity gate in it is "not measured".
"$APR" qa "$MODEL" --json --ollama-from-file --skip-golden --skip-throughput \
    --skip-gpu-speedup --skip-contract --skip-format-parity --skip-ptx-parity \
    --skip-gpu-state --skip-metadata --skip-capability \
    > "$OUT/apr-qa.json" 2> "$OUT/apr-qa.err"
qa_rc=$?
OL_STATE=$(jq -r '.gates[] | select(.name == "ollama_parity")
    | if .skipped then "skipped" elif .passed then "passed" else "failed" end' \
    "$OUT/apr-qa.json" 2>/dev/null || true)
OL_MSG=$(jq -r '.gates[] | select(.name == "ollama_parity") | .message // ""' \
    "$OUT/apr-qa.json" 2>/dev/null || true)
# apr tok/s over ollama tok/s on this file. Parity is 1.0: the gate's own
# threshold (0.2) is a no-collapse floor, so "passed" says nothing about parity.
OL_RAW=$(jq -r '.gates[] | select(.name == "ollama_parity" and (.skipped | not)) | .value // empty' \
    "$OUT/apr-qa.json" 2>/dev/null || true)
[ -n "$OL_STATE" ] || missing+=("ollama parity (qa rc $qa_rc, no ollama_parity gate in its JSON; see apr-qa.err)")
# A skipped gate measured nothing: ollama down, the import refused, or 0 tok/s.
[ "$OL_STATE" != skipped ] || missing+=("ollama parity skipped: ${OL_MSG:-no reason given}")
OL_RATIO=""
below_parity=0
if [ -n "$OL_RAW" ]; then
    # Compare the unrounded ratio; floor only for display, to 3 places, so a
    # ratio under parity never prints as 1 (0.9996 shows 0.999).
    OL_RATIO=$(jq -rn --argjson r "$OL_RAW" '$r | . * 1000 | floor / 1000' 2>/dev/null || true)
    if jq -en --argjson r "$OL_RAW" '$r < 1.0' >/dev/null 2>&1; then below_parity=1; fi
fi
# ollama's own /api/ps size for the import: a different measure from apr's
# VmHWM (resident model + runner buffers vs one process peak), never compared.
OL_RESIDENT=$(printf '%s' "$OL_MSG" | command grep -oE 'ollama resident [0-9]+ MiB' \
    | command grep -oE '[0-9]+' || true)
[ -z "$OL_RAW" ] || [ -n "$OL_RESIDENT" ] \
    || missing+=("ollama resident MiB (the parity gate measured, but /api/ps reported no size)")

# 4. apr decode on the CPU: the same bench with --no-gpu (#5276), so the CPU
# figure comes from the one harness. A binary without the flag names its
# ticket, and is still a missing row.
"$APR" bench "$MODEL" --no-gpu --json --warmup 3 --iterations 5 --max-tokens 64 \
    > "$OUT/apr-bench-cpu.json" 2> "$OUT/apr-bench-cpu.err"
cpu_rc=$?
APR_CPU_TPS=$(jq -r 'select(.compute_class == "cpu") | .tokens_per_second // empty | select(. > 0)
    | . * 10 | round / 10' "$OUT/apr-bench-cpu.json" 2>/dev/null || true)
APR_CPU_WHY=""
if [ -z "$APR_CPU_TPS" ]; then
    APR_CPU_WHY=$(command grep -m1 -E "unexpected argument '--no-gpu'" "$OUT/apr-bench-cpu.err" || true)
    if [ -n "$APR_CPU_WHY" ]; then
        APR_CPU_WHY="this binary has no bench --no-gpu flag (pending #5276)"
        missing+=("apr decode on CPU: $APR_CPU_WHY")
    else
        missing+=("apr bench --no-gpu (rc $cpu_rc, no CPU tokens_per_second; see apr-bench-cpu.err)")
    fi
fi

# 5. apr profile. A refusal is a gap with its stated reason, not a missing row.
"$APR" profile "$MODEL" --format json > "$OUT/profile.json" 2> "$OUT/profile.err"
profile_rc=$?
PROF_TPS=$(jq -r '.timing.throughput_tok_s // empty | select(. > 0) | . * 10 | round / 10' \
    "$OUT/profile.json" 2>/dev/null || true)
PROF_WHY=""
if [ -z "$PROF_TPS" ]; then
    # Only a non-zero exit is a refusal; rc 0 with no (or zero) throughput is missing.
    if [ "$profile_rc" -ne 0 ]; then
        PROF_WHY=$(command grep -m1 -iE 'error|fail|refus' "$OUT/profile.err" || true)
    fi
    [ -n "$PROF_WHY" ] || missing+=("apr profile (rc $profile_rc, no non-zero throughput and no stated refusal; see profile.err)")
fi

# The ranked list: what costs the most cells first, then decode, then refusals.
{
    printf '**Edge nightly** · model `%s` · quant `%s` · sha256 `%s`\n' \
        "$MODEL_NAME" "${QUANT:-unknown}" "$MODEL_SHA"
    printf 'apr `%s` (binary sha256 `%s`) · backend: %s\n\n' \
        "$APR_VERSION" "${APR_SHA:0:12}" "${APR_BACKEND:-not reported}"
    printf '| measure | value |\n|---|---|\n'
    printf '| apr decode (`apr bench --fast`) | %s tok/s |\n' "${APR_TPS:-not_measured}"
    printf '| apr TTFT (`apr bench --fast`) | %s ms |\n' "${APR_TTFT:-not reported}"
    printf '| apr peak RSS, VmHWM (`apr bench --fast`) | %s MiB |\n' "${APR_RSS:-not reported}"
    printf '| apr vs ollama on this file (`apr qa --ollama-from-file`) | %s |\n' \
        "${OL_RATIO:+ratio ${OL_RATIO} (parity = 1.0); }${OL_STATE:-not_measured}${OL_MSG:+: $OL_MSG}"
    if [ -n "$APR_CPU_TPS" ]; then
        printf '| apr decode, CPU (`apr bench --no-gpu`) | %s tok/s |\n' "$APR_CPU_TPS"
    else
        printf '| apr decode, CPU (`apr bench --no-gpu`) | %s |\n' "${APR_CPU_WHY:-not_measured}"
    fi
    printf '| ollama resident, `/api/ps` size (not comparable with VmHWM) | %s |\n' \
        "$([ -n "$OL_RESIDENT" ] && printf '%s MiB' "$OL_RESIDENT" || printf 'not reported')"
    if [ -n "$PROF_TPS" ]; then
        printf '| apr profile | %s tok/s |\n' "$PROF_TPS"
    else
        printf '| apr profile | %s |\n' "${PROF_WHY:-not_measured}"
    fi
    printf '| batched prefill refusals in the bench | %s |\n' "${GPU_REFUSALS:-0}"
    for dev in gpu cpu; do
        f=$CRUX_GPU
        [ "$dev" = cpu ] && f=$CRUX_CPU
        if [ -n "$f" ]; then
            jq -r --arg dev "$dev" '"| CRUX smoke verdict, \($dev) | \(.summary.verdict): \(.summary.GREEN // 0) GREEN, \(.summary.RED // 0) RED of \(.summary.cells) cells |"' "$f"
        else
            printf '| CRUX smoke verdict, %s | not_measured |\n' "$dev"
        fi
    done
    printf '\n### Verb x device (correct/answered/cells)\n\n'
    jq -nr "${MATRIX_ARGS[@]}" --arg zero 0 -f "$ROOT/scripts/lib/edge_nightly_matrix.jq"
    printf '\n### Ranked gaps\n\n'
    n=0
    # Per verb, the cells apr did NOT answer correctly, most first. A verdict
    # alone hides them: with no comparator answering, every cell is RED,
    # including the ones apr got right.
    for f in "$CRUX_GPU" "$CRUX_CPU"; do
        [ -n "$f" ] || continue
        dev=$(jq -r '.backend // empty' "$f" 2>/dev/null || true)
        while IFS= read -r line; do
            n=$((n + 1))
            printf '%d. %s%s\n' "$n" "${dev:+[$dev] }" "$line"
        done < <(jq -r -f "$ROOT/scripts/lib/edge_nightly_gaps.jq" "$f")
    done
    if [ -n "$OL_STATE" ] && [ "$OL_STATE" != passed ]; then
        n=$((n + 1))
        printf '%d. ollama comparison: `apr qa` ollama_parity %s (%s)\n' "$n" "$OL_STATE" "$OL_MSG"
    fi
    if [ "$below_parity" -eq 1 ]; then
        n=$((n + 1))
        printf '%d. apr decodes this file at %s of ollama'"'"'s rate (parity = 1.0)\n' "$n" "$OL_RATIO"
    fi
    if [ -n "$PROF_WHY" ]; then
        n=$((n + 1))
        printf '%d. apr profile refused this model: %s\n' "$n" "$PROF_WHY"
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
    --arg ttft "${APR_TTFT:-}" --arg rss "${APR_RSS:-}" --arg ratio "${OL_RATIO:-}" \
    --arg cpu_tps "${APR_CPU_TPS:-}" --arg cpu_why "${APR_CPU_WHY:-}" \
    --arg prof_tps "${PROF_TPS:-}" --arg prof_why "${PROF_WHY:-}" \
    --arg ol_res "${OL_RESIDENT:-}" --arg crux_cpu "${CRUX_CPU:-}" \
    --argjson missing "$(printf '%s\n' "${missing[@]+"${missing[@]}"}" | jq -R . | jq -s 'map(select(. != ""))')" \
    '{model: $model, quant: $quant, model_sha256: $sha, apr: $apr, apr_sha256: $apr_sha,
      backend: $backend, apr_tok_s: $apr_tps, apr_ttft_ms: $ttft, apr_peak_rss_mb: $rss,
      ollama_parity: {state: $ol_state, message: $ol_msg, ratio: $ratio, parity: 1.0},
      apr_cpu: {tok_s: $cpu_tps, gap: $cpu_why},
      apr_profile: {tok_s: $prof_tps, refused: $prof_why},
      ollama_resident_mb: $ol_res, crux_cpu: ($crux_cpu != ""),
      not_measured: $missing}' > "$OUT/nightly.json"

cat "$OUT/gaps.md"
[ "${#missing[@]}" -eq 0 ] || exit 1
exit 0
