#!/usr/bin/env bash
# float16_greedy_parity.sh - #3076: apr's F16/BF16 CPU matvec against the pinned
# llama.cpp, greedy, on an unquantized GGUF.
#
# For each prompt, run the pinned `apr run --no-gpu --temperature 0` and the pinned
# `llama-completion --temp 0 -ngl 0`, then compare the whitespace-trimmed generated
# text:
#   PASS   identical
#   KNOWN  differs from llama.cpp, but byte-identical to the --baseline apr's output: the
#          divergence predates the change under test (only with --baseline)
#   FAIL   differs, and is not explained by the baseline; the first differing character
#          offset and the texts are printed
# Exit 0 iff no prompt FAILs. Exit 1 on any FAIL. Exit 2 when the comparison cannot be made
# honestly (see below).
#
# WHY A BASELINE. Two implementations can round a near-tie differently. ggml's CPU BF16 dot
# converts the activations to BF16, while apr keeps them in f32. So a divergence from
# llama.cpp is not by itself evidence against the change. --baseline <apr> (the build before
# the change) separates "this change moved the output" from "apr and llama.cpp already
# disagreed here". Without --baseline every divergence is a FAIL.
#
# WHY A TEMPLATE-FREE MODEL. `apr run` on an instruct GGUF wraps the prompt in the chat
# template twice (#3672), and llama.cpp does not, so a `--prompt` comparison through an
# instruct model compares two different inputs. The model given here must carry no
# `tokenizer.chat_template`, and its file name must contain neither "instruct" nor "-chat"
# (apr's other trigger). The gate REFUSES (rc 2) unless apr's own verbose line reports
# `has_chat_template=false, filename_instruct=false` for it. Make one with
#   python3 gguf-py/gguf/scripts/gguf_new_metadata.py --remove-metadata tokenizer.chat_template IN OUT
#
# WHY TRIMMED TEXT. apr's `text` drops the first token's leading space, and
# llama-completion prints a trailing newline pair after the last token. Neither is a
# generated-token difference. A wrong kernel shows up as different words, not as
# whitespace at the ends.
#
# A HOST-SIDE GATE, NOT A TREE GUARD. It needs the models and the pinned llama.cpp build,
# which CI runners do not carry, so it is not named check_*.sh (scripts/guard_tree.sh runs
# those bare, in a required job). Its receipts are committed next to the change they judge.
# It measures agreement only; throughput belongs to the measured A/B, not to this script.
#
# THE COMPARATOR IS PINNED. llama-completion must report the `build_commit` declared in
# scripts/llama_pin.toml. Any other build is refused (rc 2): an unpinned denominator is
# not a measurement.
#
# RELEASE MODE (no arguments) is the pre-publish gate declared in Cargo.toml
# [package.metadata.dogfood]. For every committed receipt in evidence/pmat-3076-f16-bf16-matvec/
# it runs the model that receipt names, found under ${APR_MODELS_DIR:-~/models}/parity/,
# checked against the receipt's sha256, with the receipt as the baseline: a divergence is
# KNOWN only if apr's text is byte-identical to the text the receipt recorded for that
# prompt. A model missing from the host is a FAIL, never a skip: a release gate that did
# not run is not green.
#
# usage:
#   bash scripts/float16_greedy_parity.sh                  # release mode, see above
#   bash scripts/float16_greedy_parity.sh --model <gguf> [--apr <bin>] [--llama <bin>]
#        [--baseline <apr-before> | --known-from <receipt.json>] [--expect-sha <sha256>]
#        [--n <tokens>] [--out <receipt.json>]
#   (the comparison helpers' case table runs on every PR: scripts/check_float16_greedy_parity.sh)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
N=32
MODEL=""
APR_BIN=""
BASE_BIN=""
KNOWN_FROM=""
EXPECT_SHA=""
LLAMA_BIN=""
OUT=""

PROMPTS=(
    "The capital of France is Paris. The capital of Germany is"
    "def fibonacci(n):"
    "Water boils at 100 degrees Celsius. Ice melts at"
    "Once upon a time, in a small village by the sea,"
    "1, 2, 3, 5, 8, 13,"
    "The three primary colors are"
)

die2() { printf 'float16_greedy_parity: %s\n' "$*" >&2; exit 2; }

# shellcheck source=float16_parity_lib.sh
. "$ROOT/scripts/float16_parity_lib.sh" || die2 "cannot source scripts/float16_parity_lib.sh"

while [ $# -gt 0 ]; do
    case $1 in
        --model) MODEL=$2; shift 2 ;;
        --apr) APR_BIN=$2; shift 2 ;;
        --baseline) BASE_BIN=$2; shift 2 ;;
        --known-from) KNOWN_FROM=$2; shift 2 ;;
        --expect-sha) EXPECT_SHA=$2; shift 2 ;;
        --llama) LLAMA_BIN=$2; shift 2 ;;
        --n) N=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        *) die2 "unknown argument: $1" ;;
    esac
done

command -v jq > /dev/null || die2 "jq is required (apr --json parsing and the receipt)"

release_mode() {
    local dir=${APR_MODELS_DIR:-$HOME/models}/parity worst=0 r model sha rc
    local receipts=("$ROOT"/evidence/pmat-3076-f16-bf16-matvec/greedy-parity-*.json)
    [ -f "${receipts[0]}" ] || die2 "no committed receipts under evidence/pmat-3076-f16-bf16-matvec/"
    for r in "${receipts[@]}"; do
        model=$(jq -r '.model' "$r")
        sha=$(jq -r '.model_sha256' "$r")
        printf '=== %s (receipt %s)\n' "$model" "${r#"$ROOT"/}"
        if [ ! -f "$dir/$model" ]; then
            printf 'FAIL   %s is not on this host under %s. Provision it: download the source GGUF named\n' "$model" "$dir"
            printf '       in evidence/pmat-3076-f16-bf16-matvec/findings.json and strip its chat template with\n'
            printf '       gguf_new_metadata.py --remove-metadata tokenizer.chat_template (sha256 must be %s)\n' "$sha"
            worst=1
            continue
        fi
        rc=0
        bash "$0" --model "$dir/$model" --expect-sha "$sha" --known-from "$r" || rc=$?
        [ "$rc" -gt "$worst" ] && worst=$rc
    done
    printf -- '--- release mode: %s model(s), worst rc=%s ---\n' "${#receipts[@]}" "$worst"
    return "$worst"
}
if [ -z "$MODEL" ] && [ -z "$APR_BIN$BASE_BIN$KNOWN_FROM$LLAMA_BIN$OUT" ]; then
    rc=0
    release_mode || rc=$?
    exit "$rc"
fi
[ -n "$MODEL" ] || die2 "--model <gguf> is required"
[ -z "$BASE_BIN" ] || [ -z "$KNOWN_FROM" ] || die2 "--baseline and --known-from are alternatives; give one"
[ -z "$KNOWN_FROM" ] || [ -f "$KNOWN_FROM" ] || die2 "--known-from receipt not found: $KNOWN_FROM"
[ -f "$MODEL" ] || die2 "model not found: $MODEL"

# --- the apr under test: pinned to this tree unless given explicitly -------------------
if [ -z "$APR_BIN" ]; then
    # shellcheck source=/dev/null
    . "$ROOT/scripts/apr_bin.sh" || die2 "apr_bin.sh could not attribute an apr binary to this tree"
    APR_BIN=$APR
fi
[ -x "$APR_BIN" ] || die2 "apr binary not executable: $APR_BIN"
APR_VERSION=$("$APR_BIN" --version 2>&1 | head -1)
BASE_VERSION=""
if [ -n "$BASE_BIN" ]; then
    [ -x "$BASE_BIN" ] || die2 "baseline apr not executable: $BASE_BIN"
    BASE_VERSION=$("$BASE_BIN" --version 2>&1 | head -1)
fi

# --- the comparator: must be the pinned llama.cpp build ---------------------------------
PIN=$(sed -n 's/^build_commit = "\([0-9a-f]*\)".*/\1/p' "$ROOT/scripts/llama_pin.toml" | head -1)
[ -n "$PIN" ] || die2 "no build_commit in scripts/llama_pin.toml"
[ -n "$LLAMA_BIN" ] || LLAMA_BIN="$HOME/src/llama.cpp-$PIN/build/bin/llama-completion"
[ -x "$LLAMA_BIN" ] || die2 "comparator not found: $LLAMA_BIN (build llama.cpp at $PIN)"
LLAMA_VERSION=$("$LLAMA_BIN" --version 2>&1 | grep -m1 'commit' || true)
case $LLAMA_VERSION in
    *"commit $PIN"*) ;;
    *) die2 "comparator is not the pinned build $PIN: '${LLAMA_VERSION:-no version line}'" ;;
esac

# --- the input must be the same on both sides: no chat template ------------------------
PROBE=$("$APR_BIN" run "$MODEL" --no-gpu -p "${PROMPTS[0]}" -n 1 --temperature 0 -v 2>&1 || true)
case $PROBE in
    *"has_chat_template=false, filename_instruct=false"*) ;;
    *) die2 "apr would template this model (#3672), so the inputs would differ; use a copy without tokenizer.chat_template whose name has no 'instruct'/'-chat'" ;;
esac

MODEL_SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
[ -z "$EXPECT_SHA" ] || [ "$MODEL_SHA" = "$EXPECT_SHA" ] \
    || die2 "model sha256 $MODEL_SHA is not the receipt's $EXPECT_SHA: a different file is not the model the receipt judged"
HOST=$(hostname)
CPU=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)

printf 'apr:        %s (%s)\n' "$APR_VERSION" "$APR_BIN"
if [ -n "$KNOWN_FROM" ]; then BASE_DESC="receipt ${KNOWN_FROM#"$ROOT"/}"; else BASE_DESC=${BASE_VERSION:-"none (strict: every divergence FAILs)"}; fi
printf 'baseline:   %s\n' "$BASE_DESC"
printf 'comparator: %s (%s)\n' "$LLAMA_VERSION" "$LLAMA_BIN"
printf 'model:      %s sha256=%s\n' "$MODEL" "$MODEL_SHA"
printf 'host:       %s, %s; greedy, CPU, n=%s\n' "$HOST" "$CPU" "$N"

WORK=$(mktemp -d)
[ -d "$WORK" ] || die2 "mktemp -d failed"
trap 'rm -rf -- "${WORK:?}"' EXIT
fails=0
known=0
rows=""
for p in "${PROMPTS[@]}"; do
    apr_json=$("$APR_BIN" run "$MODEL" --no-gpu -p "$p" -n "$N" --temperature 0 --json 2>/dev/null) \
        || die2 "apr run failed on prompt: $p"
    apr_text=$(printf '%s' "$apr_json" | jq -r '.text') || die2 "apr --json had no .text"
    "$LLAMA_BIN" -m "$MODEL" -p "$p" -n "$N" --temp 0 -ngl 0 -no-cnv --no-display-prompt \
        --no-warmup -s 0 --simple-io > "$WORK/out" 2> "$WORK/err" < /dev/null \
        || die2 "llama-completion failed on prompt: $p"
    llama_text=$(cat "$WORK/out")
    a=$(trim "$apr_text")
    b=$(trim "$llama_text")
    [ -n "$a" ] || die2 "apr generated no text for: $p"
    d=$(first_diff "$a" "$b")
    base_note=""
    if [ "$d" = -1 ]; then
        verdict=PASS
    elif [ -n "$BASE_BIN" ]; then
        base_json=$("$BASE_BIN" run "$MODEL" --no-gpu -p "$p" -n "$N" --temperature 0 --json 2>/dev/null) \
            || die2 "baseline apr run failed on prompt: $p"
        base_text=$(trim "$(printf '%s' "$base_json" | jq -r '.text')")
        if [ "$base_text" = "$a" ]; then verdict=KNOWN; base_note=" (baseline apr identical)"; else
            verdict=FAIL; base_note=" (baseline apr differs too, at char $(first_diff "$base_text" "$a"))"; fi
    elif [ -n "$KNOWN_FROM" ]; then
        rec=$(jq -r --arg p "$p" '[.prompts[] | select(.prompt == $p and .verdict == "KNOWN")][0].apr_text // empty' "$KNOWN_FROM")
        if [ -n "$rec" ] && [ "$rec" = "$a" ]; then verdict=KNOWN; base_note=" (apr text identical to the receipt's KNOWN row)"
        elif [ -n "$rec" ]; then verdict=FAIL; base_note=" (receipt KNOWN, but apr's text moved at char $(first_diff "$rec" "$a"))"
        else verdict=FAIL; base_note=" (not KNOWN in the receipt)"; fi
    else
        verdict=FAIL
    fi
    case $verdict in
        PASS) printf 'PASS   %s\n' "$(json_str "$p")" ;;
        *) printf '%-6s %s  first difference from llama.cpp at char %s%s\n       apr:       %s\n       llama.cpp: %s\n' \
               "$verdict" "$(json_str "$p")" "$d" "$base_note" "$(json_str "$a")" "$(json_str "$b")" ;;
    esac
    [ "$verdict" = FAIL ] && fails=$((fails + 1))
    [ "$verdict" = KNOWN ] && known=$((known + 1))
    rows="${rows}${rows:+,}{\"prompt\":$(json_str "$p"),\"verdict\":\"$verdict\",\"first_diff\":$d,\"apr_text\":$(json_str "$a"),\"llama_text\":$(json_str "$b")}"
done

if [ -n "$OUT" ]; then
    printf '{"schema":"float16-greedy-parity-v1","apr":%s,"baseline":%s,"comparator":%s,"comparator_pin":"%s","model":%s,"model_sha256":"%s","host":%s,"cpu":%s,"n":%s,"failures":%s,"known":%s,"prompts":[%s]}\n' \
        "$(json_str "$APR_VERSION")" "$(json_str "$BASE_VERSION")" "$(json_str "$LLAMA_VERSION")" "$PIN" "$(json_str "$(basename "$MODEL")")" \
        "$MODEL_SHA" "$(json_str "$HOST")" "$(json_str "$CPU")" "$N" "$fails" "$known" "$rows" > "$OUT"
    jq -e . "$OUT" > /dev/null || die2 "receipt is not valid JSON: $OUT"
    printf 'receipt: %s\n' "$OUT"
fi
printf -- '--- %s/%s identical to llama.cpp, %s KNOWN (predate the change), %s FAIL ---\n' \
    "$(( ${#PROMPTS[@]} - fails - known ))" "${#PROMPTS[@]}" "$known" "$fails"
[ "$fails" -eq 0 ]
