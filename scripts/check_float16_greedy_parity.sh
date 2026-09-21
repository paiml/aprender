#!/usr/bin/env bash
# check_float16_greedy_parity.sh - #3076: apr's F16/BF16 CPU matvec against the pinned
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
# THE COMPARATOR IS PINNED. llama-completion must report the `build_commit` declared in
# scripts/llama_pin.toml. Any other build is refused (rc 2): an unpinned denominator is
# not a measurement.
#
# usage:
#   bash scripts/check_float16_greedy_parity.sh --model <gguf> [--apr <bin>] [--llama <bin>]
#        [--baseline <apr-before>] [--n <tokens>] [--out <receipt.json>]
#   bash scripts/check_float16_greedy_parity.sh --self-test
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
N=32
MODEL=""
APR_BIN=""
BASE_BIN=""
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

die2() { printf 'check_float16_greedy_parity: %s\n' "$*" >&2; exit 2; }

# first_diff A B -> prints the first differing character offset, or -1 when equal. The
# strings go in through the environment, not `awk -v`, which would interpret backslash
# escapes in generated code.
first_diff() {
    A=$1 B=$2 awk 'BEGIN {
        a = ENVIRON["A"]; b = ENVIRON["B"]
        if (a == b) { print -1; exit }
        n = length(a) < length(b) ? length(a) : length(b)
        for (i = 1; i <= n; i++) if (substr(a, i, 1) != substr(b, i, 1)) { print i - 1; exit }
        print n
    }'
}

trim() {
    local s=$1
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# json_str S -> S as a JSON string literal (backslash, quote, control characters escaped).
json_str() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\t'/\\t}
    s=${s//$'\r'/\\r}
    printf '"%s"' "$s"
}

self_test() {
    local fails=0 got
    # (a, b, expected first_diff)
    check() {
        got=$(first_diff "$1" "$2")
        if [ "$got" = "$3" ]; then
            printf 'ok    first_diff(%q, %q) = %s\n' "$1" "$2" "$got"
        else
            printf 'FAIL  first_diff(%q, %q) = %s, want %s\n' "$1" "$2" "$got" "$3"
            fails=$((fails + 1))
        fi
    }
    check "Berlin. The capital" "Berlin. The capital" -1
    check "Berlin. The capital" "Berlin. A capital" 8
    check "Berlin" "Berlin." 6
    check "" "" -1
    check "x" "" 0
    check 'print("a\\nb")' 'print("a\\nc")' 11
    local t
    t=$(trim $'  Berlin. The capital\n\n')
    if [ "$t" = "Berlin. The capital" ]; then printf 'ok    trim strips both ends\n'; else
        printf 'FAIL  trim gave %q\n' "$t"; fails=$((fails + 1)); fi
    t=$(trim $'a  b')
    if [ "$t" = $'a  b' ]; then printf 'ok    trim keeps inner whitespace\n'; else
        printf 'FAIL  trim changed inner whitespace: %q\n' "$t"; fails=$((fails + 1)); fi
    t=$(json_str $'a"b\\c\nd')
    if [ "$t" = '"a\"b\\c\nd"' ]; then printf 'ok    json_str escapes\n'; else
        printf 'FAIL  json_str gave %s\n' "$t"; fails=$((fails + 1)); fi
    printf -- '--- self-test: %s failure(s) ---\n' "$fails"
    [ "$fails" -eq 0 ]
}

while [ $# -gt 0 ]; do
    case $1 in
        --model) MODEL=$2; shift 2 ;;
        --apr) APR_BIN=$2; shift 2 ;;
        --baseline) BASE_BIN=$2; shift 2 ;;
        --llama) LLAMA_BIN=$2; shift 2 ;;
        --n) N=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --self-test) self_test; exit $? ;;
        *) die2 "unknown argument: $1" ;;
    esac
done

command -v jq > /dev/null || die2 "jq is required (apr --json parsing and the receipt)"
[ -n "$MODEL" ] || die2 "--model <gguf> is required"
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
HOST=$(hostname)
CPU=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)

printf 'apr:        %s (%s)\n' "$APR_VERSION" "$APR_BIN"
printf 'baseline:   %s\n' "${BASE_VERSION:-none (strict: every divergence FAILs)}"
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
    apr_tps=$(printf '%s' "$apr_json" | jq -r '.tok_per_sec')
    "$LLAMA_BIN" -m "$MODEL" -p "$p" -n "$N" --temp 0 -ngl 0 -no-cnv --no-display-prompt \
        --no-warmup -s 0 --simple-io > "$WORK/out" 2> "$WORK/err" < /dev/null \
        || die2 "llama-completion failed on prompt: $p"
    llama_text=$(cat "$WORK/out")
    llama_tps=$(sed -n 's/.* eval time = .*, *\([0-9.]*\) tokens per second.*/\1/p' "$WORK/err" | grep -v '^$' | tail -1)
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
    else
        verdict=FAIL
    fi
    case $verdict in
        PASS) printf 'PASS   %-58s  apr run %s tok/s (whole run), llama.cpp eval %s tok/s\n' \
                  "$(json_str "$p")" "$apr_tps" "${llama_tps:-?}" ;;
        *) printf '%-6s %s  first difference from llama.cpp at char %s%s\n       apr:       %s\n       llama.cpp: %s\n' \
               "$verdict" "$(json_str "$p")" "$d" "$base_note" "$(json_str "$a")" "$(json_str "$b")" ;;
    esac
    [ "$verdict" = FAIL ] && fails=$((fails + 1))
    [ "$verdict" = KNOWN ] && known=$((known + 1))
    rows="${rows}${rows:+,}{\"prompt\":$(json_str "$p"),\"verdict\":\"$verdict\",\"first_diff\":$d,\"apr_text\":$(json_str "$a"),\"llama_text\":$(json_str "$b"),\"apr_run_tok_per_sec_whole_run\":${apr_tps:-null},\"llama_eval_tok_per_sec\":${llama_tps:-null}}"
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
